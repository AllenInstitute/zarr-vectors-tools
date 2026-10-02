"""The things that are not geometry, and are lost if nobody carries them.

Moving objects between stores is the visible half of a merge.  The half
that gets forgotten is everything indexed *alongside* those objects — the
group taxonomy, the object-attribute columns, the format headers, and the
pyramid built from the level being changed.  Each is lost differently and
each is lost quietly:

* **Groups** name object ids.  Ids are renumbered by a merge, so a group
  copied verbatim points at whatever now occupies those slots.
* **Object attributes** are dense columns indexed by id.  Append a
  thousand objects and every column is a thousand rows short, which reads
  back as the array's fill value rather than as an error.
* **Headers** are keyed by format name, so a second TRK's header
  overwrites the first's and the store then claims a provenance that is
  true of only half its contents.
* **Pyramids** are derived data.  After a merge, every coarse level
  describes the store as it was, and a viewer that opens level 2 sees the
  merge silently missing.

Nothing here is clever.  It exists because all four are easy to skip and
none of them fail loudly when skipped.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from zarr_vectors.exceptions import StoreError

__all__ = [
    "append_object_attributes",
    "carry_headers",
    "expand_grid",
    "handle_pyramid",
    "merge_groups",
    "read_provenance",
    "record_provenance",
    "stale_levels",
]


# =====================================================================
# Growing a grid
# =====================================================================


def expand_grid(
    dataset: Any,
    level: Any,
    needed: Sequence[int],
    *,
    lower: Sequence[int] | None = None,
    extent: tuple[Sequence[float], Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Grow every per-chunk array so chunk coordinates below ``needed`` fit.

    ``needed`` is one past the highest chunk coordinate to hold, per axis
    (``floor(hi / cell) + 1``); ``lower``, when given, is the lowest.  Both
    are absolute chunk coordinates — ``floor(p / cell)``, the chunk key a
    write uses.

    Safe *upwards only*, and the asymmetry is the whole point.  Each array
    stores cell ``key - origin``, where ``origin`` is the chunk coordinate
    of its first cell, so adding cells at the top leaves every existing
    cell where it was and the resize touches no data.  Extending
    *downwards* moves the first cell and therefore renumbers every cell
    already written: the same bytes, all under the wrong keys.  That is a
    rewrite, not an expansion, so it is refused rather than attempted.

    Every per-chunk family is grown together — vertices, fragments, link
    families, and each attribute — because they are addressed by the same
    key and a family left at the old shape simply fails on first write to
    a new cell.  :func:`per_chunk_array_paths` is what enumerates them;
    guessing the names misses the link families, which nest two levels
    deeper.

    The declared bounds grow to cover the new cells (to ``extent`` when
    given), because arrays created later — a new link segment, a new
    attribute — are sized from them.
    """
    from zarr_vectors.building import (
        per_chunk_array_paths,
        read_root_metadata,
        update_root_metadata,
    )

    group = level.store
    try:
        vertices = group.zarr_group["vertices"]
        current = tuple(int(s) for s in vertices.shape)
    except Exception as exc:  # noqa: BLE001
        raise StoreError("cannot read the target's grid shape to expand it") from exc
    origin = tuple(
        int(o) for o in (vertices.attrs.get("chunk_grid_origin") or (0,) * len(current))
    )

    want_hi = tuple(int(v) for v in needed)
    # The highest cell wanted lying below the first cell means the data on
    # that axis is entirely below the grid, which is as downward as it gets.
    want_lo = tuple(
        int(v) for v in (lower if lower is not None else (h - 1 for h in want_hi))
    )
    below = [
        (axis, lo, first)
        for axis, (lo, first) in enumerate(zip(want_lo, origin))
        if lo < first
    ]
    if below:
        axis, lo, first = below[0]
        cell = float(level.scale[axis])
        raise StoreError(
            f"cannot grow the target's grid downward: axis {axis} needs the "
            f"cell starting at {lo * cell:g} (chunk coordinate {lo}), but the "
            f"grid's first cell starts at {first * cell:g} (chunk coordinate "
            f"{first}). Every cell already written is stored relative to the "
            "first cell, so moving it would renumber every chunk in the store. "
            "Move the source into the target's frame "
            "(--transform FILE, or a Source built with transform=), rebuild "
            "the target over bounds that cover both, or drop the objects that "
            "do not fit (--on-out-of-bounds skip / on_out_of_bounds='skip')."
        )

    target = tuple(
        max(size, hi - first) for size, hi, first in zip(current, want_hi, origin)
    )
    if target == current:
        return {"expanded": False, "grid": list(current)}

    grown: list[str] = []
    for path in per_chunk_array_paths(group):
        try:
            array = group.zarr_group[path]
        except Exception:  # noqa: BLE001
            continue
        shape = tuple(int(s) for s in array.shape)
        if len(shape) != len(target):
            continue
        # Each array is grown to reach the same top cell, from its own first
        # cell: arrays created at different times need not share an origin.
        own = tuple(
            int(o) for o in (array.attrs.get("chunk_grid_origin") or (0,) * len(shape))
        )
        merged = tuple(
            max(size, first + size_v - own_first)
            for size, first, size_v, own_first in zip(shape, origin, target, own)
        )
        if merged != shape:
            array.resize(merged)
            grown.append(path)

    cell = np.asarray([float(v) for v in level.scale], dtype=np.float64)
    try:
        lo, hi = read_root_metadata(dataset.store).bounds
    except Exception:  # noqa: BLE001
        lo, hi = dataset.bounds
    if extent is not None:
        top = np.asarray(extent[1], dtype=np.float64)
    else:
        # The middle of the new top cell: in it, whatever the rounding.
        top = (np.asarray(origin) + np.asarray(target) - 0.5) * cell
    new_hi = np.maximum(np.asarray(hi, dtype=np.float64), top)
    try:
        update_root_metadata(
            dataset.store,
            bounds=[[float(v) for v in lo], [float(v) for v in new_hi]],
        )
    except Exception:  # noqa: BLE001 - the arrays are what a write checks
        pass
    # Core sizes an array it creates later (a new link segment, a new
    # attribute) from the grid it derives from these bounds, and caches that
    # grid on the level handle.  ``level.store`` hands out a new handle on
    # every access, so the writes after this one see the grown grid; a
    # caller must not keep a level group from before the expansion.

    return {
        "expanded": True,
        "from": list(current),
        "to": list(target),
        "arrays": grown,
    }


# =====================================================================
# Object attributes
# =====================================================================


def _existing_column_spec(
    level_group: Any, name: str,
) -> tuple[np.dtype, tuple[int, ...]]:
    """The dtype and per-row shape of an object-attribute column on disk.

    Read from the array's metadata rather than by loading the column, so
    this stays O(1) on a store with millions of objects.
    """
    try:
        meta = level_group.read_array_meta(f"object_attributes/{name}") or {}
    except Exception:  # noqa: BLE001 - an unreadable column gets the default
        return np.dtype(np.float32), ()
    dtype = np.dtype(meta.get("dtype", "float32"))
    shape = tuple(int(s) for s in (meta.get("shape") or ()))
    return dtype, shape[1:]


def _fill_for_dtype(dtype: np.dtype, fill: float) -> Any:
    """A sentinel ``dtype`` can actually hold.

    Mirrors core's own rule for absent object-attribute rows: NaN for
    floating point, dtype-min for signed integers, dtype-max for unsigned.
    ``fill`` is honoured when it survives the cast, so a caller asking for a
    specific value still gets it.
    """
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        return fill
    if dtype.kind == "i":
        return np.iinfo(dtype).min
    if dtype.kind == "u":
        return np.iinfo(dtype).max
    if dtype.kind == "b":
        return False
    return fill


def append_object_attributes(
    level_group: Any,
    columns: Mapping[str, npt.NDArray[Any]],
    *,
    existing_names: Sequence[str],
    base_count: int,
    added: int,
    fill: float = float("nan"),
) -> dict[str, int]:
    """Extend every object-attribute column to cover the new objects.

    Three cases, and the last two are the ones that go wrong:

    * A column both sides have — append the incoming values.
    * A column only the *target* has — append ``fill``, so the column
      stays as long as the object index.  Skipping it leaves the array
      short, and a short dense column does not raise: it reads back as
      the fill value for the missing tail, which is indistinguishable
      from data that was genuinely absent.
    * A column only the *source* has — create it, backfilled with
      ``fill`` for every pre-existing object, then append.

    Returns:
        ``{name: rows_written}``.
    """
    from zarr_vectors.building import (
        create_object_attributes_array,
        write_object_attributes,
    )

    written: dict[str, int] = {}
    names = sorted(set(existing_names) | set(columns))
    for name in names:
        incoming = columns.get(name)
        if incoming is None:
            # A column only the TARGET has.  The filler must match the
            # column already on disk in BOTH dtype and width:
            #
            # * a float32 NaN appended to a uint32 column is cast to 0 --
            #   a value indistinguishable from a real label, and
            # * a 1-D filler appended to a multi-channel column (what
            #   ``--compute-endpoints`` writes: start and end are (O, 3))
            #   raises "append shape mismatch" outright, so merging into
            #   any store with endpoints failed.
            dtype, row_shape = _existing_column_spec(level_group, name)
            values = np.full(
                (added, *row_shape), _fill_for_dtype(dtype, fill), dtype=dtype,
            )
        else:
            values = np.asarray(incoming)
            if len(values) != added:
                raise StoreError(
                    f"object attribute {name!r} has {len(values)} values for "
                    f"{added} new objects"
                )

        if name in existing_names:
            # ``at`` pins the appended rows to the object ids they describe.
            # Without it an already-short column (the very failure this
            # function exists to prevent) is extended from wherever it
            # happened to end, silently rebinding every value after it.
            write_object_attributes(
                level_group, name, values, mode="append", at=base_count,
            )
        else:
            channels = int(values.shape[1]) if values.ndim > 1 else 1
            # NaN is only a sentinel for floats.  ``np.full(n, nan,
            # dtype=int64)`` yields INT64_MIN, and for uint32 it yields 0 --
            # a value that reads back as real data.  Use the same per-dtype
            # sentinel core writes for absent rows (see
            # ``write_object_attributes(fill_value=...)``).
            hole = _fill_for_dtype(values.dtype, fill)
            backfill = (
                np.full(base_count, hole, dtype=values.dtype)
                if channels == 1
                else np.full((base_count, channels), hole, dtype=values.dtype)
            )
            try:
                create_object_attributes_array(
                    level_group, name,
                    dtype=str(values.dtype), num_channels=channels,
                )
            except Exception:  # noqa: BLE001 - already exists is fine
                pass
            whole = (
                np.concatenate([backfill, values], axis=0)
                if base_count else values
            )
            write_object_attributes(level_group, name, whole)
        written[name] = int(len(values))
    return written


# =====================================================================
# Groups
# =====================================================================


def merge_groups(
    level_group: Any,
    catalog: Any,
    incoming: Mapping[str, npt.NDArray[np.int64]],
    *,
    remap: Mapping[int, int],
    prefix: str = "",
) -> dict[str, int]:
    """Fold ``incoming`` groups into the level's existing taxonomy.

    Members arrive in the source's numbering and are translated through
    ``remap``; a member with no mapping was not carried and is dropped
    rather than pointed at whichever object now holds that id.

    A name already present is *extended*, not replaced — merging two
    tractograms that both have a ``"cst"`` bundle should give one bundle
    with both sets of streamlines.  Pass ``prefix`` when that is the wrong
    reading and the two should stay distinct.

    Returns:
        ``{group_name: member_count}`` for the taxonomy as written.
    """
    from zarr_vectors.building import create_groupings_array, read_all_groupings, write_groupings

    try:
        existing_rows = list(read_all_groupings(level_group))
    except Exception:  # noqa: BLE001 - no groupings array yet
        existing_rows = []
    names = list(catalog.names())
    while len(names) < len(existing_rows):
        names.append(f"group_{len(names)}")

    rows: dict[int, list[int] | range] = {}
    for gid, members in enumerate(existing_rows):
        rows[gid] = members if isinstance(members, range) else [int(m) for m in members]

    index_of = {name: i for i, name in enumerate(names)}
    for raw_name, members in incoming.items():
        name = f"{prefix}{raw_name}" if prefix else str(raw_name)
        translated = [
            int(remap[int(m)]) for m in np.asarray(members).ravel()
            if int(m) in remap
        ]
        if not translated:
            continue
        if name in index_of:
            gid = index_of[name]
            current = rows.get(gid, [])
            current = list(current) if isinstance(current, range) else list(current)
            rows[gid] = current + translated
        else:
            gid = len(names)
            names.append(name)
            index_of[name] = gid
            rows[gid] = translated

    if not rows:
        return {}
    # write_groupings requires contiguous ids from 0; a gap would silently
    # renumber every group above it.
    for gid in range(len(names)):
        rows.setdefault(gid, [])
    create_groupings_array(level_group)
    write_groupings(level_group, rows)
    catalog.name_rows(names)
    return {names[gid]: len(rows[gid]) for gid in sorted(rows)}


# =====================================================================
# Headers
# =====================================================================


def carry_headers(
    dataset: Any,
    headers: Mapping[str, Mapping[str, Any]],
    *,
    label: str,
) -> dict[str, str]:
    """Copy format headers across, without letting one overwrite another.

    The registry is keyed by format name, so two TRK sources both want
    the key ``"trk"``.  The second is filed under ``"trk@<label>"``
    instead of replacing the first: a header that describes half the
    store is useful, and a header that silently describes the wrong half
    is not.  ``label`` is the source's short name
    (:attr:`~zarr_vectors_tools.compose.sources.Source.name`), so the key
    reads ``trk@subject_b`` rather than carrying a whole URL.

    Returns:
        ``{format_name: key_written_under}``.
    """
    registry = dataset.headers
    try:
        taken = set(registry.available_formats)
    except Exception:  # noqa: BLE001 - no headers group yet
        taken = set()

    placed: dict[str, str] = {}
    for name, payload in headers.items():
        key = str(name)
        if key in taken:
            key = f"{name}@{label}"
        suffix = 2
        while key in taken:
            key = f"{name}@{label}-{suffix}"
            suffix += 1
        try:
            registry.add(key, dict(payload))
        except Exception:  # noqa: BLE001 - a header is never worth failing on
            continue
        taken.add(key)
        placed[str(name)] = key
    return placed


# =====================================================================
# Provenance
# =====================================================================


PROVENANCE_NAMESPACE = "zarr_vectors_tools.compose"


def provenance(dataset: Any) -> Any:
    """This package's metadata namespace on ``dataset``.

    ``Dataset.metadata`` hands out *namespaces*, not a dict — it has no
    ``get`` and no ``__setitem__``, so treating it as a mapping raises
    ``AttributeError`` and, behind a defensive ``except``, writes nothing
    at all while reporting success.
    """
    return dataset.metadata.namespace(PROVENANCE_NAMESPACE)


def record_provenance(dataset: Any, entry: Mapping[str, Any]) -> bool:
    """Append one merge/split record to the store's own metadata.

    Without this a merged store is unattributable: the object ids past
    some boundary came from somewhere, and nothing on disk says where or
    what offset was applied.  :func:`~zarr_vectors_tools.compose.split`
    reads these back, which is what makes a merge reversible — so whether
    the write succeeded is returned rather than swallowed.
    """
    try:
        namespace = provenance(dataset)
        history = list(namespace.get("history", []))
        history.append(dict(entry))
        namespace["history"] = history
        return True
    except Exception:  # noqa: BLE001 - a read-only or exotic backend
        return False


def read_provenance(dataset: Any) -> list[dict[str, Any]]:
    """Every merge/split this store records, oldest first."""
    try:
        return [dict(e) for e in provenance(dataset).get("history", [])]
    except Exception:  # noqa: BLE001
        return []


# =====================================================================
# Pyramids
# =====================================================================


def stale_levels(dataset: Any) -> list[int]:
    """Levels that describe the store as it was before this write."""
    return [int(i) for i in dataset.levels if int(i) != 0]


def handle_pyramid(
    dataset: Any,
    policy: str,
    *,
    factors: Sequence[tuple[float, float]] | None = None,
    **build_options: Any,
) -> dict[str, Any]:
    """Bring the pyramid back in line with a level 0 that just changed.

    ``policy`` is one of:

    ``"rebuild"``
        Drop the coarse levels and build them again.  Correct, and the
        default, because it is the only option that leaves the store
        self-consistent.  Expensive in proportion to level 0.

    ``"drop"``
        Remove the coarse levels and leave the store single-resolution.
        Cheap and honest; a reader gets no pyramid rather than a wrong one.

    ``"keep"``
        Leave them, and stamp ``compose_stale`` on each so a reader can at
        least find out.  For when the merge is one of several and the
        rebuild is deferred to the end.
    """
    from zarr_vectors.building import (
        remove_resolution_level,
        update_level_metadata,
        write_multiscale_metadata,
    )

    levels = stale_levels(dataset)
    if policy == "keep":
        for index in levels:
            try:
                # The level group, not the root plus an index: the root
                # form is a TypeError, and a swallowed TypeError here
                # means the "stale" marker never lands.
                update_level_metadata(dataset.level(index).store, compose_stale=True)
            except Exception:  # noqa: BLE001
                pass
        return {"pyramid": "keep", "stale_levels": levels}

    if policy not in ("rebuild", "drop"):
        raise ValueError(
            f"pyramid={policy!r} is not one of 'rebuild', 'drop', 'keep'"
        )

    if policy == "rebuild" and factors is None and levels:
        # Re-run each level with the parameters it records -- method, bin and
        # chunk scale, rdp tolerance and per-level sparsity -- rather than
        # dividing numbers back out of the metadata, which lost the chunk
        # scale and tolerance and read per-level sparsity as cumulative.
        from zarr_vectors_tools.multiresolution.refresh import (
            rebuild_pyramid_from_level,
        )

        forwarded = (
            "sparsity_strategy", "sparsity_seed", "compressor", "executor",
            "cross_level_storage", "cross_level_depth",
        )
        specs = rebuild_pyramid_from_level(
            dataset.store, 0,
            **{k: build_options[k] for k in forwarded if k in build_options},
        )
        return {"pyramid": "rebuild", "rebuilt_levels": levels, "build": specs}

    for index in sorted(levels, reverse=True):
        try:
            remove_resolution_level(dataset.store, index)
        except Exception:  # noqa: BLE001 - already gone is fine
            continue
    if levels:
        from zarr_vectors_tools.multiresolution.coarsen import (
            _clear_cross_level_families,
            _stamp_root_cross_level,
        )

        # remove_resolution_level leaves the level listed in `multiscales`,
        # which is where a viewer reads the level list from.
        write_multiscale_metadata(dataset.store)
        # Level 0's +N families pointed into the removed levels, and the
        # root still claimed them.
        _clear_cross_level_families(dataset.store, from_level=0)
        _stamp_root_cross_level(dataset.store)

    if policy == "drop" or not factors:
        # Reporting "drop" for a rebuild that had nothing to rebuild
        # (a single-level store) said a policy the caller never chose.
        return {
            "pyramid": "drop" if policy == "drop" else "none",
            "removed_levels": levels,
        }

    from zarr_vectors_tools.multiresolution.coarsen import build_pyramid

    summary = build_pyramid(dataset.url, factors=list(factors), **build_options)
    return {
        "pyramid": "rebuild",
        "removed_levels": levels,
        "factors": [list(f) for f in factors],
        "build": summary,
    }
