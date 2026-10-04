"""Additive pyramids: each object (or point) stored at one level only.

An ordinary pyramid stores a coarse level as a *replacement* for the finer
one: the 1/2 of streamlines kept at level 1 are stored again at level 0,
and a viewer zooming in throws level 1 away to draw level 0.  An additive
pyramid stores them once.  Every level but the coarsest is marked
``refinement: "add"`` and holds only what the next coarser level does not;
a level's complete content is its own data plus the complete content of the
next coarser level (zarr-vectors-py's ``level_chain``).  Storage drops from
about 1.14-1.33x of level 0 to about 1x, and a viewer that has drawn a
coarse level fetches only the difference when it zooms in.

That union is only the *same picture* as the replacement level when the
levels differ by which objects (or points) they keep, not by how those are
drawn.  It is exact for

* object sparsity with the geometry unchanged -- a streamline or polyline
  pyramid built with ``coarsen_factor`` 1 keeps every kept streamline
  byte-identical;
* nested point subsets (``build_point_subset_pyramid`` in
  :mod:`zarr_vectors_tools.multiresolution.strategies.points`).

It is not for meshes or for any level that spatially coarsens geometry
(binning into metanodes, RDP or decimation, skeleton simplification, mesh
clustering or decimation): an object kept at a coarse level is then drawn
at the coarse resolution even in the finest view.  Those builds still run,
with an :class:`AdditiveRefinementWarning`.

The pyramid is built as an ordinary one and then converted, finest level
first, by zarr-vectors-py's ``make_levels_additive`` -- level 0 is
rewritten in place.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

REFINEMENT_CHOICES = ("replace", "add")

#: Coarseners that keep a kept object's geometry exactly when the coarsen
#: factor is 1 (checked byte for byte in the tests, for both coarsen modes
#: and with chunk scaling).
_IDENTITY_AT_FACTOR_ONE = frozenset({"polyline"})


class AdditiveRefinementWarning(UserWarning):
    """An additive pyramid was asked for where its union is not the
    picture the replacement pyramid draws, or where a level adds nothing."""


def check_refinement(refinement: str) -> str:
    if refinement not in REFINEMENT_CHOICES:
        raise ValueError(
            f"refinement must be one of {list(REFINEMENT_CHOICES)}, "
            f"got {refinement!r}"
        )
    return refinement


def supports_additive() -> bool:
    """Whether the installed zarr-vectors-py can write additive levels."""
    try:
        from zarr_vectors.building import level_chain  # noqa: F401
        from zarr_vectors.multiresolution.additive import (  # noqa: F401
            make_levels_additive,
        )
    except ImportError:
        return False
    return True


def refuse_additive(store_path: Any, action: str) -> None:
    """Raise if ``store_path`` already has additive levels: ``action``
    rebuilds levels from one another, which an additive pyramid -- each
    level holding only what the next one lacks -- cannot survive."""
    try:
        from zarr_vectors.building import refuse_additive as _refuse
    except ImportError:  # no additive levels can exist without it
        return
    _refuse(str(store_path) if not hasattr(store_path, "attrs") else store_path, action)


def additive_concerns(
    *,
    geometry_types: Sequence[str],
    coarsener: str,
    factors: Sequence[tuple[float, float]],
    coarsen_mode: str = "rdp",
    rdp_tolerances: Sequence[float] | None = None,
) -> list[str]:
    """Why an additive version of this pyramid would not draw what the
    replacement pyramid draws -- empty when it would.  Known before a level
    is built."""
    out: list[str] = []
    if "mesh" in geometry_types:
        out.append(
            "the store is a mesh: every mesh coarsener re-tessellates, so an "
            "object kept at a coarse level is drawn coarse in the finest view, "
            "and coarse and fine surfaces of neighbouring objects mix"
        )
    coarsened = [
        i + 1 for i, (coarsen, _sparsity) in enumerate(factors) if float(coarsen) > 1.0
    ]
    if coarsened:
        out.append(
            f"level(s) {coarsened} coarsen geometry (coarsen factor > 1): the "
            f"objects kept there are stored only at that resolution, so the "
            f"complete finest view shows them coarse"
        )
    elif coarsener not in _IDENTITY_AT_FACTOR_ONE and "mesh" not in geometry_types:
        out.append(
            f"the {coarsener!r} coarsener replaces vertices even at coarsen "
            f"factor 1 (binning into metanodes, or simplification), so the "
            f"objects kept at coarse levels are stored only in coarsened form"
        )
    if rdp_tolerances is not None and coarsener == "polyline" and coarsen_mode == "rdp":
        out.append(
            "explicit RDP tolerances simplify every coarse level, so the "
            "streamlines kept there are stored only simplified"
        )
    return out


def warn_concerns(concerns: Sequence[str]) -> None:
    for concern in concerns:
        warnings.warn(
            f"additive pyramid: {concern}. Proceeding as asked.",
            AdditiveRefinementWarning, stacklevel=3,
        )


def _level_sizes(root: Any) -> dict[int, tuple[Any, int]]:
    """Per level: stored object ids (or None) and vertex count."""
    from zarr_vectors.building import (
        get_resolution_level,
        list_resolution_levels,
        read_level_metadata,
        stored_object_ids,
    )

    out = {}
    for lv in list_resolution_levels(root):
        lg = get_resolution_level(root, lv)
        out[lv] = (
            stored_object_ids(lg),
            int(read_level_metadata(root, lv).vertex_count or 0),
        )
    return out


def make_pyramid_additive(
    store_path: Any, *, concerns: Sequence[str] = (),
) -> dict[str, Any]:
    """Convert a finished replacement pyramid to an additive one, in place.

    **Level 0 is rewritten**: afterwards it holds only the objects or
    points no coarser level has, and level 0 alone is no longer the whole
    dataset.  zarr-vectors-py readers return a level's complete content by
    default; ``own_level_only=True`` reads what one level stores.

    Warns (:class:`AdditiveRefinementWarning`) for each of ``concerns``
    and for a level that keeps every object or point of the level below,
    whose additive level would be empty.  Recounts each level's
    ``object_attributes/vertex_count`` column, where there is one, to the
    vertices that level now stores.

    Returns:
        ``{"levels": [per-level report], "concerns": [...]}``.
    """
    if not supports_additive():
        raise RuntimeError(
            "the installed zarr-vectors-py cannot write additive levels "
            "(zarr_vectors.multiresolution.additive.make_levels_additive)"
        )
    from zarr_vectors.building import list_resolution_levels, open_store
    from zarr_vectors.multiresolution.additive import make_levels_additive

    root = open_store(str(store_path), mode="r+")
    levels = list_resolution_levels(root)
    if len(levels) < 2:
        raise ValueError("an additive pyramid needs at least two levels")

    notes = list(concerns)
    sizes = _level_sizes(root)
    for lv in levels[:-1]:
        ids, count = sizes[lv]
        next_ids, next_count = sizes[lv + 1]
        same = (
            ids is not None and next_ids is not None and len(ids) == len(next_ids)
        ) if ids is not None else count == next_count
        if same:
            notes.append(
                f"level {lv + 1} keeps every "
                f"{'object' if ids is not None else 'point'} of level {lv}, so "
                f"additive level {lv} will be empty"
            )
    warn_concerns(notes)

    reports = make_levels_additive(root)
    _recount_object_vertex_counts(root, [r["level"] for r in reports])
    return {"levels": reports, "concerns": notes}


def _recount_object_vertex_counts(root: Any, levels: Sequence[int]) -> None:
    """``object_attributes/vertex_count`` counts a level's own vertices;
    the conversion changed them."""
    from zarr_vectors.building import get_resolution_level

    from zarr_vectors_tools.convert.ingest._object_columns import (
        VERTEX_COUNT_ATTR,
        stamp_level_object_columns,
    )

    for lv in levels:
        lg = get_resolution_level(root, lv)
        if not lg.array_exists(f"object_attributes/{VERTEX_COUNT_ATTR}"):
            continue
        stamp_level_object_columns(
            lg, vertex_count=True, segment_id=False, recount=True,
        )


def level_zero_notice(store_path: Any) -> str:
    """The banner a CLI prints after converting: what changed about level 0."""
    return (
        f"NOTE: {store_path} now has an ADDITIVE pyramid. Level 0 was "
        f"rewritten in place and holds only what no coarser level has: level "
        f"0 alone is no longer the whole dataset. The store lists "
        f"'additive_levels' in required_capabilities; a reader that "
        f"implements it (zarr-vectors-py from this version) returns each "
        f"level's complete content, and one that checks the list but lacks "
        f"it refuses the store. Older readers do not check the list and "
        f"would show level 0 incomplete."
    )


@contextmanager
def warnings_as_banner(stream: Any = None) -> Iterator[list[str]]:
    """Collect :class:`AdditiveRefinementWarning`\\ s raised inside the block
    and print them as one banner on ``stream`` (stderr) at the end; every
    other warning is shown as usual.  For command-line front ends, where a
    warning line is easy to miss."""
    import sys

    caught: list[str] = []
    original = warnings.showwarning

    def show(message, category, filename, lineno, file=None, line=None):  # noqa: ANN001
        if issubclass(category, AdditiveRefinementWarning):
            caught.append(str(message))
            return
        original(message, category, filename, lineno, file, line)

    with warnings.catch_warnings():
        warnings.simplefilter("always", AdditiveRefinementWarning)
        warnings.showwarning = show
        yield caught
    if caught:
        out = stream or sys.stderr
        rule = "=" * 72
        print(rule, file=out)
        print("WARNING: additive pyramid (refinement 'add')", file=out)
        for text in caught:
            print(f"  - {text}", file=out)
        print(rule, file=out, flush=True)
