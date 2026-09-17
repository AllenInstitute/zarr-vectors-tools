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
    "infer_pyramid_factors",
    "merge_groups",
    "read_provenance",
    "record_provenance",
    "stale_levels",
]


# =====================================================================
# Growing a grid
# =====================================================================


def expand_grid(
    dataset: Any, level: Any, needed: Sequence[int],
) -> dict[str, Any]:
    """Grow every per-chunk array so ``needed`` cells fit.

    Safe *upwards only*, and the asymmetry is the whole point.  A chunk
    key is ``floor(p / cell)`` — an absolute cell index with the grid's
    corner pinned at the coordinate origin — so adding cells at the top
    leaves every existing key meaning exactly what it meant before, and
    the resize touches no data.  Extending *downwards*, into negative
    coordinates, would shift the origin and therefore renumber every
    chunk in the store: the same bytes, all under the wrong keys.  That
    is a rewrite, not an expansion, so it is refused rather than
    attempted.

    Every per-chunk family is grown together — vertices, fragments, link
    families, and each attribute — because they are addressed by the same
    key and a family left at the old shape simply fails on first write to
    a new cell.  :func:`per_chunk_array_paths` is what enumerates them;
    guessing the names misses the link families, which nest two levels
    deeper.
    """
    from zarr_vectors.building import per_chunk_array_paths, update_root_metadata

    group = level.store
    want = tuple(int(v) for v in needed)
    if any(v < 0 for v in want):
        raise StoreError(
            "cannot expand a grid to hold negative chunk coordinates: chunk "
            "keys are absolute cell indices from the coordinate origin, so "
            "moving the origin renumbers every chunk already written. "
            "Translate the incoming geometry into positive coordinates, or "
            "rebuild the target over bounds that cover both."
        )

    try:
        current = tuple(int(s) for s in group.zarr_group["vertices"].shape)
    except Exception as exc:  # noqa: BLE001
        raise StoreError("cannot read the target's grid shape to expand it") from exc

    target = tuple(max(a, b) for a, b in zip(current, want))
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
        merged = tuple(max(a, b) for a, b in zip(shape, target))
        if merged != shape:
            array.resize(merged)
            grown.append(path)

    cell = np.asarray([float(v) for v in level.scale], dtype=np.float64)
    lo, hi = dataset.bounds
    new_hi = np.maximum(np.asarray(hi, dtype=np.float64), np.asarray(target) * cell)
    try:
        update_root_metadata(
            dataset.store,
            bounds=[[float(v) for v in lo], [float(v) for v in new_hi]],
        )
    except Exception:  # noqa: BLE001 - the arrays are what a write checks
        pass

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
    is not.

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


def infer_pyramid_factors(dataset: Any) -> list[tuple[float, float]] | None:
    """Recover the ``(coarsen, sparsity)`` factors a pyramid was built with.

    Levels record what they *are* — ``bin_ratio`` against level 0 and a
    cumulative ``object_sparsity`` — not the per-level ratios
    :func:`build_pyramid` takes.  Dividing consecutive levels recovers
    them, which is what lets a rebuild reproduce the pyramid the store
    had rather than asking the caller to remember it.

    Returns ``None`` when a level is missing either number, because a
    guessed pyramid is worse than an honest refusal.
    """
    from zarr_vectors.building import read_level_metadata

    levels = sorted(int(i) for i in dataset.levels)
    if len(levels) < 2:
        return None

    ratios: list[float] = []
    sparsities: list[float] = []
    for index in levels:
        try:
            meta = read_level_metadata(dataset.store, index)
        except Exception:  # noqa: BLE001
            return None
        bin_ratio = getattr(meta, "bin_ratio", None)
        sparsity = getattr(meta, "object_sparsity", None)
        if sparsity is None:
            return None
        ratios.append(float(np.mean(bin_ratio)) if bin_ratio else 1.0)
        sparsities.append(float(sparsity))

    factors: list[tuple[float, float]] = []
    for i in range(1, len(levels)):
        coarsen = ratios[i] / ratios[i - 1] if ratios[i - 1] else 1.0
        keep = sparsities[i] / sparsities[i - 1] if sparsities[i - 1] else 1.0
        # build_pyramid takes sparsity as "one in N", the reciprocal of the
        # retained fraction the level records.
        factors.append((round(coarsen, 6), round(1.0 / keep, 6) if keep else 1.0))
    return factors


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
    from zarr_vectors.building import remove_resolution_level, update_level_metadata

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

    if policy == "rebuild" and factors is None:
        factors = infer_pyramid_factors(dataset)
        if factors is None and levels:
            raise StoreError(
                "pyramid='rebuild' cannot reconstruct the factors this store's "
                "pyramid was built with (its levels do not record enough to "
                "divide out per-level ratios). Pass pyramid_factors=[(coarsen, "
                "sparsity), ...] explicitly, or choose pyramid='drop' to leave "
                "the store single-resolution."
            )

    for index in sorted(levels, reverse=True):
        try:
            remove_resolution_level(dataset.store, index)
        except Exception:  # noqa: BLE001 - already gone is fine
            continue

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
