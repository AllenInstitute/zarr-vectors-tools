"""Putting one dataset into another.

``convert`` builds a store from a file.  ``attach`` joins columns onto
vertices a store already has.  Neither adds *objects* to a store that
exists, which is what a second tractogram, a second imaging run or a
bundle atlas actually needs.

The obvious approach does not work.  ``Dataset.add_polylines`` reads like
an append and is a replace: called twice on one store, the second call
leaves the first call's objects gone, its object-attribute columns
overwritten, and no error anywhere.  (Measured, not assumed: two writes
of two and one polylines leave a store with one object.)  The append
primitive is instead :meth:`EditSession.add_object`, which allocates an
id past the existing ones, splits the vertices across the chunks they
land in, and appends a fragment to each — leaving everything already
written byte-intact.

So this module is that primitive plus the four things around it that are
lost if nobody carries them — ids, groups, attributes, headers — and the
pyramid, which is derived from the level being changed and is wrong the
moment it changes.  See :mod:`zarr_vectors_tools.compose._carry`.

**The grid is the target's.**  A merge never re-bins the store it writes
into: incoming geometry is assigned to the target's existing cells, which
is what keeps the write proportional to what is added rather than to what
is already there.  Geometry that lands outside those cells is checked for
up front, against the shape the arrays were *allocated* with rather than
the bounds they declare — the two can disagree, and it is the allocation
a write is checked against.  ``on_out_of_bounds="expand"`` then grows the
grid upwards, which costs nothing because a chunk key is an absolute cell
index and adding cells at the top leaves every existing key meaning what
it meant.  Downwards is refused: that moves the origin, and moving the
origin renumbers every chunk already written.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from zarr_vectors.exceptions import IngestError, StoreError

from zarr_vectors_tools.compose._carry import (
    append_object_attributes,
    carry_headers,
    expand_grid,
    handle_pyramid,
    merge_groups,
    record_provenance,
)
from zarr_vectors_tools.compose.sources import (
    DEFAULT_MAX_OBJECTS,
    DEFAULT_MAX_VERTICES,
    Source,
    open_source,
    require_geometry_match,
)

__all__ = ["merge_stores", "plan_merge"]

# Kinds whose pyramid strategy needs the per-fragment segment_id stamp.
_POLYLINE_KINDS = frozenset({"streamline", "polyline"})


def plan_merge(
    target: Any,
    sources: Sequence[Any],
    **source_options: Any,
) -> dict[str, Any]:
    """Everything a merge would do, without doing any of it.

    Answers the question that actually blocks a merge — does the incoming
    geometry fit the target's grid — before a single vertex is written.

    Cheap on a *store* source, which is read through its metadata.  A
    **file** source is opened the way the merge would open it, which for a
    format with no header-level object count means parsing it; budget for
    that rather than assuming this is free on any input.
    """
    import zarr_vectors as zv

    opened = [open_source(s, **source_options) for s in sources]
    try:
        dataset = zv.open(str(target), mode="r")
        kind = dataset.kinds[0] if dataset.kinds else None
        lo, hi = dataset.bounds
        existing = int(len(dataset.level(0).objects))
        grid_box = _grid_box(*allocated_grid(dataset.level(0)))
        exists = True
    except Exception:  # noqa: BLE001 - a target that is not there yet
        kind, lo, hi, existing, exists, grid_box = None, None, None, 0, False, None

    entries = []
    offset = existing
    for source in opened:
        info = source.info
        fits = None
        if exists and info.bounds is not None and grid_box is not None:
            fits = _fits(info.bounds, grid_box)
        entries.append({
            "label": info.label,
            "kind": info.kind,
            "objects": info.object_count,
            "vertices": info.vertex_count,
            "bounds": info.bounds,
            "id_offset": offset,
            "fits_target_bounds": fits,
            "object_attributes": list(info.object_attributes),
            "vertex_attributes": list(info.vertex_attributes),
            "groups": list(info.group_names),
        })
        offset += int(info.object_count or 0)

    # Sources own scratch state -- a staged ingest's temporary directory,
    # an open memmap -- and planning is the one entry point that used to
    # walk away without releasing it, so a planning loop over many files
    # leaked one temp directory per call.
    for source in opened:
        try:
            source.close()
        except Exception:  # noqa: BLE001 - a source with nothing to release
            pass

    return {
        "target": str(target),
        "target_exists": exists,
        "target_kind": kind,
        "target_objects": existing,
        "target_bounds": (list(lo), list(hi)) if exists and lo is not None else None,
        "sources": entries,
        "total_objects_after": offset,
    }


def merge_stores(
    target: Any,
    sources: Sequence[Any] | Any,
    *,
    create: bool = False,
    cell_size: Sequence[float] | None = None,
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    pyramid: Literal["rebuild", "drop", "keep"] = "rebuild",
    pyramid_factors: Sequence[tuple[float, float]] | None = None,
    pyramid_options: Mapping[str, Any] | None = None,
    groups: bool = True,
    group_prefix: str | Mapping[str, str] | None = None,
    source_attribute: str | None = None,
    on_out_of_bounds: Literal["raise", "skip", "expand"] = "raise",
    max_objects: int = DEFAULT_MAX_OBJECTS,
    max_vertices: int = DEFAULT_MAX_VERTICES,
    progress: bool = False,
    source_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge one or more sources into ``target``.

    Args:
        target: Store to merge into — a path, URL or open ``Dataset``.
        sources: What to merge.  Each may be a store path, an open
            ``Dataset``, a path to a file in any supported format, or an
            already-constructed
            :class:`~zarr_vectors_tools.compose.sources.Source` (which is
            how a coordinate transform is supplied).
        create: Create ``target`` when it does not exist, sizing its grid
            from the union of the sources' bounds.  Off by default: a
            typo in a store path should not silently produce a new store.
        cell_size: Grid cell size for a created target.  Defaults to the
            first source store's, or a 64-cell-per-axis split of the
            union bounds when no source is a store.
        bounds: Pin a created target's extent instead of using the union.
        pyramid: What to do with coarse levels, which every merge
            invalidates.  See
            :func:`~zarr_vectors_tools.compose._carry.handle_pyramid`.
        pyramid_factors: Per-level ``(coarsen, sparsity)`` for a rebuild.
            Inferred from the existing levels when omitted.
        groups: Carry each source's named object groups across.
        group_prefix: Namespace incoming group names — a single string
            applied to all, or ``{source_label: prefix}``.  Without it, a
            name the target already has is *extended* rather than
            duplicated.
        source_attribute: Name of a per-object attribute to stamp with
            each source's index, so the merged objects stay separable.
        on_out_of_bounds: What to do when a source does not fit the
            target's grid.  ``"raise"`` (default) refuses;  ``"skip"``
            drops the offending objects and counts them;  ``"expand"``
            grows the grid upwards to cover them, which is safe because
            chunk keys are absolute and adding cells at the top renumbers
            nothing.  Geometry needing *negative* cells is always refused
            — moving the origin would renumber every chunk in the store.
        max_objects: Objects per write batch.
        max_vertices: Vertices per write batch.
        progress: Print per-batch progress.
        source_options: Extra keyword arguments passed to
            :func:`~zarr_vectors_tools.compose.sources.open_source` for
            any source given as a path.

    Returns:
        A summary dict: per-source object counts, the id offsets applied,
        the group taxonomy as written, and what happened to the pyramid.
    """
    if isinstance(sources, (str, bytes)) or not isinstance(sources, Sequence):
        sources = [sources]
    opened = [open_source(s, **dict(source_options or {})) for s in sources]
    if not opened:
        raise IngestError("merge_stores needs at least one source")

    try:
        dataset, created = _open_or_create_target(
            target, opened,
            create=create, cell_size=cell_size, bounds=bounds,
        )
    except Exception:
        for source in opened:
            source.close()
        raise

    summary: dict[str, Any] = {
        "target": dataset.url,
        "created": created,
        "sources": [],
        "objects_added": 0,
        "vertices_added": 0,
        "skipped_out_of_bounds": 0,
    }

    try:
        kind = require_geometry_match(opened, dataset.kinds[0] if dataset.kinds else None)
        from zarr_vectors_tools.compose.readers import check_supported

        check_supported(kind)

        level = dataset.level(0)
        for index, source in enumerate(opened):
            summary["sources"].append(
                _merge_one(
                    dataset, level, source, index,
                    groups=groups,
                    group_prefix=_prefix_for(group_prefix, source),
                    source_attribute=source_attribute,
                    on_out_of_bounds=on_out_of_bounds,
                    max_objects=max_objects,
                    max_vertices=max_vertices,
                    progress=progress,
                    totals=summary,
                )
            )

        # Before the pyramid: the coarseners read level 0's metadata, and a
        # level that claims zero vertices is one they will decline to
        # coarsen.
        _rebuild_manifests(dataset)
        summary["vertex_count"] = _refresh_level_metadata(
            dataset, level,
            int(level.vertex_count or 0) + int(summary["vertices_added"]),
        )
        summary.update(
            handle_pyramid(
                dataset, pyramid,
                factors=pyramid_factors, **dict(pyramid_options or {}),
            )
        )
        summary["provenance_recorded"] = record_provenance(dataset, {
            "operation": "merge",
            "sources": [entry["label"] for entry in summary["sources"]],
            "id_offsets": [entry["id_offset"] for entry in summary["sources"]],
            "objects_added": summary["objects_added"],
        })
    finally:
        for source in opened:
            source.close()

    return summary


# =====================================================================
# One source
# =====================================================================


def _merge_one(
    dataset: Any,
    level: Any,
    source: Source,
    index: int,
    *,
    groups: bool,
    group_prefix: str,
    source_attribute: str | None,
    on_out_of_bounds: str,
    max_objects: int,
    max_vertices: int,
    progress: bool,
    totals: dict[str, Any],
) -> dict[str, Any]:
    info = source.info
    base_count = int(len(level.objects))
    target_bounds = dataset.bounds
    target_object_attrs = tuple(level.attribute_names("object"))
    target_vertex_attrs, skipped_attrs = _vertex_attribute_plan(
        level, info.vertex_attributes, base_count,
    )
    cell_shape = tuple(float(v) for v in level.scale)

    expansion: dict[str, Any] = {}
    if on_out_of_bounds == "expand" and info.bounds is not None:
        expansion = expand_grid(
            dataset, level, _cells_for(info.bounds[1], level.scale),
        )

    grid_box = _grid_box(*allocated_grid(level))
    if (
        info.bounds is not None
        and grid_box is not None
        and not _fits(info.bounds, grid_box)
        and on_out_of_bounds == "raise"
    ):
        raise StoreError(
            f"{info.label} does not fit the target's grid.\n"
            f"  source extent {_fmt(info.bounds)}\n"
            f"  grid covers   {_fmt(grid_box)}\n"
            f"  target bounds {_fmt(target_bounds)}\n"
            f"  overhang      {_overhang(info.bounds, grid_box)}\n"
            "A merge writes onto the target's existing cells and cannot grow "
            "them. Either transform the source into the target's frame (pass "
            "a Source with transform=), rebuild the target over bounds that "
            "cover both, or pass on_out_of_bounds='skip' to drop what does "
            "not fit."
        )

    remap: dict[int, int] = {}
    object_columns: dict[str, list[npt.NDArray[Any]]] = {}
    added = 0
    vertices = 0
    skipped = 0
    empty = 0

    grid_shape, cell_size = allocated_grid(level)

    for batch_no, batch in enumerate(
        source.iter_batches(max_objects=max_objects, max_vertices=max_vertices)
    ):
        keep, offenders = _grid_mask(batch.parts, grid_shape, cell_size)
        if not keep.all():
            if on_out_of_bounds == "expand":
                # Expansion grows the grid upward only -- the cell origin is
                # fixed at coordinate 0 by ``floor(p / cell)`` -- so anything
                # still outside cannot be accommodated.  Dropping it here
                # would be "skip" behaviour under a flag that promised to
                # make room.
                negative = [o for o in offenders if any(c < 0 for c in o)]
                why = (
                    "negative cell coordinates, which no expansion can reach: "
                    "the grid's origin is fixed at coordinate 0"
                    if negative else
                    "cells beyond the grid even after expansion"
                )
                raise StoreError(
                    f"{info.label}: {int((~keep).sum())} object(s) in batch "
                    f"{batch_no} land in {why}.\n"
                    f"  first offender lands in cell "
                    f"{(negative or offenders)[0]}\n"
                    f"  grid          {list(grid_shape)} cells of "
                    f"{list(cell_size)}\n"
                    f"  target bounds {_fmt(target_bounds)}\n"
                    "Transform the source into the target's frame (pass a "
                    "Source with transform=), rebuild the target over bounds "
                    "that cover both, or pass on_out_of_bounds='skip' to drop "
                    "what does not fit."
                )
            if on_out_of_bounds == "raise":
                bad = int((~keep).sum())
                raise StoreError(
                    f"{info.label}: {bad} object(s) in batch {batch_no} land "
                    f"outside the target's allocated grid.\n"
                    f"  grid          {list(grid_shape)} cells of {list(cell_size)}\n"
                    f"  covering      {_fmt(_grid_box(grid_shape, cell_size))}\n"
                    f"  first offender lands in cell {offenders[0] if offenders else '?'}\n"
                    f"  target bounds {_fmt(target_bounds)}\n"
                    "Transform the source into the target's frame (pass a "
                    "Source with transform=), rebuild the target over bounds "
                    "that cover both, or pass on_out_of_bounds='skip'."
                )
            skipped += int((~keep).sum())

        _ensure_attribute_arrays(level, batch.vertex_attributes, target_vertex_attrs)
        with dataset.editing() as plan:
            session = plan.session
            _preload_attributes(
                session,
                [p for p, k in zip(batch.parts, keep) if k],
                cell_shape, target_vertex_attrs,
            )
            cursor = 0
            # A kept object with no vertices writes no geometry, so it must
            # not claim an attribute row either: ``added`` counted only the
            # objects written while the columns below were sliced by ``keep``
            # alone, and append_object_attributes then raised "N values for
            # N-1 new objects" -- after every batch had already committed.
            written = keep.copy()
            for i, part in enumerate(batch.parts):
                length = len(part)
                start, cursor = cursor, cursor + length
                if not keep[i] or length == 0:
                    written[i] = False
                    continue
                ref = session.add_object(
                    level=0,
                    vertices=part,
                    attrs=_vertex_attrs_for(
                        batch.vertex_attributes, start, cursor,
                        length, target_vertex_attrs,
                    ),
                )
                remap[int(batch.object_ids[i])] = int(ref.object_id)
                added += 1
                vertices += length

        if info.kind in _POLYLINE_KINDS:
            _write_segment_ids(
                level, [remap[int(o)] for o in batch.object_ids if int(o) in remap],
            )

        empty += int(np.count_nonzero(keep & ~written))
        for name, column in batch.object_attributes.items():
            values = np.asarray(column)[written]
            if len(values):
                object_columns.setdefault(name, []).append(values)

        if progress:
            print(
                f"  {info.label}: batch {batch_no} -> {added:,} objects, "
                f"{vertices:,} vertices",
                flush=True,
            )

    columns: dict[str, npt.NDArray[Any]] = {
        name: np.concatenate(chunks, axis=0) for name, chunks in object_columns.items()
    }
    if source_attribute and added:
        columns[source_attribute] = np.full(added, float(index), dtype=np.float32)

    attributes_written: dict[str, int] = {}
    if added:
        attributes_written = append_object_attributes(
            level.store, columns,
            existing_names=target_object_attrs,
            base_count=base_count,
            added=added,
        )

    groups_written: dict[str, int] = {}
    if groups and added:
        incoming = source.groups()
        if incoming:
            groups_written = merge_groups(
                level.store, level.groups, incoming,
                remap=remap, prefix=group_prefix,
            )

    headers_written = carry_headers(
        dataset, info.headers, label=_slug(info.label),
    )

    totals["objects_added"] += added
    totals["vertices_added"] += vertices
    totals["skipped_out_of_bounds"] += skipped

    return {
        "label": info.label,
        "kind": info.kind,
        "id_offset": base_count,
        "objects_added": added,
        "vertices_added": vertices,
        "skipped_out_of_bounds": skipped,
        "grid_expansion": expansion,
        "object_attributes": attributes_written,
        "vertex_attributes": list(target_vertex_attrs),
        "skipped_vertex_attributes": skipped_attrs,
        "groups": groups_written,
        "headers": headers_written,
    }


def _vertex_attribute_plan(
    level: Any, incoming: Sequence[str], base_count: int,
) -> tuple[tuple[str, ...], list[str]]:
    """Which per-vertex attributes this merge can carry, and which it cannot.

    A per-vertex attribute is stored per *chunk*, fragment-aligned with
    that chunk's vertices.  Introducing a new one into a store that
    already holds objects would therefore only populate the chunks this
    merge happens to touch, leaving a column that exists in some cells and
    not others — which reads back as zeros in the untouched ones and looks
    like data.  So a new name is only accepted when the target is still
    empty; otherwise it is reported as skipped and the caller can decide.
    """
    existing = tuple(level.attribute_names("vertex"))
    if base_count == 0:
        return tuple(dict.fromkeys((*existing, *incoming))), []
    carried = tuple(existing)
    skipped = [name for name in incoming if name not in existing]
    return carried, skipped


SEGMENT_ID_ATTR = "segment_id"


def _write_segment_ids(level: Any, new_ids: Sequence[int]) -> int:
    """Stamp ``fragment_attributes/segment_id`` for freshly appended fragments.

    Not core's convention but this package's: the polyline coarsener
    reconstructs which object a fragment belongs to from a per-fragment
    ``segment_id``, and refuses outright without one — *"requires
    fragment_attributes/segment_id on the source level"*.  The ingesters
    write it; ``EditSession.add_object`` does not, because core has no
    such requirement.  So a merge that skipped this would succeed, look
    correct, and then fail the moment anyone rebuilt the pyramid.

    Written per chunk, extending rather than replacing: existing
    fragments keep their ids, and the new ones are filled in at the
    indices the manifests report.
    """
    from zarr_vectors.building import (
        create_fragment_attribute_array,
        read_chunk_fragment_attributes,
        read_object_manifests,
        write_chunk_fragment_attributes,
    )

    if not new_ids:
        return 0
    group = level.store
    manifests = read_object_manifests(group, ids=[int(i) for i in new_ids])

    per_chunk: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for oid, manifest in manifests.items():
        for cc, fi in manifest or ():
            per_chunk.setdefault(tuple(int(c) for c in cc), []).append((int(fi), int(oid)))
    if not per_chunk:
        return 0

    create_fragment_attribute_array(
        group, SEGMENT_ID_ATTR, dtype="uint64", exist_ok=True,
    )
    written = 0
    for cc, entries in per_chunk.items():
        existing = read_chunk_fragment_attributes(
            group, SEGMENT_ID_ATTR, cc, dtype=np.uint64, default=None,
        )
        current = (
            np.asarray(existing, dtype=np.uint64).ravel()
            if existing is not None else np.zeros(0, dtype=np.uint64)
        )
        size = max(int(max(fi for fi, _ in entries)) + 1, len(current))
        column = np.zeros(size, dtype=np.uint64)
        column[: len(current)] = current
        for fi, oid in entries:
            column[fi] = oid
        write_chunk_fragment_attributes(
            group, SEGMENT_ID_ATTR, cc, column, dtype=np.uint64,
        )
        written += len(entries)
    return written


def _ensure_attribute_arrays(
    level: Any, columns: Mapping[str, npt.NDArray[Any]], names: Sequence[str],
) -> None:
    """Allocate the per-chunk array for any vertex attribute that lacks one.

    The edit session will happily buffer values for an attribute with no
    array behind it and then fail at flush with "no chunk array at that
    path" — after the vertices for that batch have already been staged.
    Creating the family first turns that into a no-op.
    """
    from zarr_vectors.building import create_attribute_array

    existing = set(level.attribute_names("vertex"))
    for name in names:
        if name in existing:
            continue
        values = columns.get(name)
        dtype = str(np.asarray(values).dtype) if values is not None else "float32"
        ncols = (
            int(np.asarray(values).shape[1])
            if values is not None and np.asarray(values).ndim > 1
            else 1
        )
        create_attribute_array(
            level.store, name, dtype=dtype, ncols=ncols, exist_ok=True,
        )


def _preload_attributes(
    session: Any,
    parts: Sequence[npt.NDArray[Any]],
    cell_shape: Sequence[float],
    names: Sequence[str],
) -> None:
    """Load each touched chunk's attribute lists before anything is appended.

    ``ChunkChangeBuilder.append_fragment`` extends only the attributes
    already decoded into the builder, so an ``add_object(attrs=...)`` on a
    builder that has loaded none silently discards every value *and*
    leaves that chunk's attribute blob one fragment short of its vertex
    blob.  Nothing raises: the blob is still a valid array, just shorter,
    and the next read pairs each attribute fragment with the wrong
    vertices.  Touching the attributes first is what makes the append
    extend them.
    """
    if not names or not parts:
        return
    from zarr_vectors.building import assign_chunks

    touched: set[tuple[int, ...]] = set()
    for part in parts:
        arr = np.asarray(part, dtype=np.float64)
        if not arr.size:
            continue
        for cc in assign_chunks(arr, tuple(cell_shape)):
            touched.add(tuple(int(c) for c in cc))
    for cc in touched:
        session._ensure_attrs_loaded(session._builder(0, cc), names)


def _vertex_attrs_for(
    columns: Mapping[str, npt.NDArray[Any]],
    start: int,
    stop: int,
    length: int,
    required: Sequence[str],
) -> dict[str, npt.NDArray[Any]] | None:
    """One object's slice of every vertex attribute the target carries.

    Names the target has and the source lacks are filled with NaN rather
    than omitted.  Omitting them would leave that attribute's fragment
    list one short for every chunk the new object touches, and since a
    fragment is addressed by *index* the mismatch does not raise — it
    silently pairs each subsequent object's attribute values with the
    wrong object's vertices.
    """
    out: dict[str, npt.NDArray[Any]] = {}
    for name in required:
        column = columns.get(name)
        if column is None:
            out[name] = np.full(length, np.nan, dtype=np.float32)
        else:
            out[name] = np.asarray(column)[start:stop]
    for name, column in columns.items():
        if name not in out:
            out[name] = np.asarray(column)[start:stop]
    return out or None


# =====================================================================
# Target resolution
# =====================================================================


def _open_or_create_target(
    target: Any,
    sources: Sequence[Source],
    *,
    create: bool,
    cell_size: Sequence[float] | None,
    bounds: tuple[Sequence[float], Sequence[float]] | None,
) -> tuple[Any, bool]:
    import zarr_vectors as zv
    from zarr_vectors.api.schema import Layout, Schema

    if hasattr(target, "level") and hasattr(target, "bounds"):
        return target, False
    exists = Path(str(target)).exists()
    try:
        return zv.open(str(target), mode="r+"), False
    except Exception as exc:  # noqa: BLE001 - absent, or present but unusable
        if exists:
            # A store that IS there but will not open (permissions, a
            # half-written metadata block, a format this build cannot read)
            # was reported as "does not exist", which sends the caller off to
            # create=True and a second, equally failing attempt.
            raise StoreError(
                f"{target} exists but could not be opened: {exc}"
            ) from exc

    if not create:
        raise StoreError(
            f"{target} does not exist. Pass create=True to make it, sized from "
            "the sources' combined extent."
        )

    union = bounds or _union_bounds(sources)
    if union is None:
        raise StoreError(
            "cannot create the target: no source declares bounds, so there is "
            "nothing to size a grid from. Pass bounds=."
        )
    size = cell_size or _default_cell_size(sources, union)
    kind = require_geometry_match(sources, None)
    schema = Schema(
        ndim=len(union[0]),
        bounds=(tuple(union[0]), tuple(union[1])),
        kind=kind,
        layout=Layout(cell_size=tuple(float(v) for v in size)),
    )
    dataset = zv.create(str(target), schema=schema)

    # A created store's grid is sized from the *extent* of its bounds,
    # while a chunk key is ``floor(p / cell)`` — an index from the
    # coordinate origin.  The two agree only when the lower corner is
    # zero.  For bounds starting at, say, y=29 with 21-unit cells, the
    # allocation is five cells and the top of the data indexes cell five,
    # which is one past the end: the very first write fails with a
    # chunk-coordinate error that names neither the axis nor the cause.
    # Growing to cover the absolute indices closes the gap at creation.
    expand_grid(
        dataset, dataset.level(0),
        _cells_for(union[1], tuple(float(v) for v in size)),
    )
    return dataset, True


def _union_bounds(
    sources: Sequence[Source],
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    boxes = [s.info.bounds for s in sources if s.info.bounds is not None]
    if not boxes:
        return None
    lo = np.min([np.asarray(b[0], dtype=np.float64) for b in boxes], axis=0)
    hi = np.max([np.asarray(b[1], dtype=np.float64) for b in boxes], axis=0)
    return tuple(float(v) for v in lo), tuple(float(v) for v in hi)


def _default_cell_size(
    sources: Sequence[Source], union: tuple[Sequence[float], Sequence[float]],
) -> tuple[float, ...]:
    """A created target's cell size.

    A source that is itself a store already has a grid someone chose, and
    inheriting it means the merge writes onto aligned cells.  Failing
    that, 64 per axis is a middle that keeps cell blobs to a workable size
    without producing a metadata explosion.
    """
    for source in sources:
        dataset = getattr(source, "dataset", None)
        if dataset is not None:
            try:
                return tuple(float(v) for v in dataset.level(0).scale)
            except Exception:  # noqa: BLE001
                continue
    lo = np.asarray(union[0], dtype=np.float64)
    hi = np.asarray(union[1], dtype=np.float64)
    return tuple(float(max(e, 1e-9) / 64.0) for e in (hi - lo))


# =====================================================================
# Bounds helpers
# =====================================================================


def _cells_for(upper: Sequence[float], cell: Sequence[float]) -> tuple[int, ...]:
    """How many cells per axis are needed to hold coordinates up to ``upper``.

    ``floor(u / c) + 1``, not ``ceil(u / c)``: a coordinate that lands
    exactly on a cell boundary belongs to the cell *above* it, so ceil is
    one short whenever the extent is an exact multiple of the cell size —
    the off-by-one that puts a vertex in a chunk the array does not have.
    """
    return tuple(
        int(np.floor(float(u) / float(c))) + 1 for u, c in zip(upper, cell)
    )


def _grid_box(
    shape: Sequence[int], cell: Sequence[float],
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    """The physical region an allocated grid covers.

    ``floor(p / cell)`` puts the grid's lower corner at the coordinate
    origin, not at the store's declared minimum — so a store whose bounds
    start above zero still has cells numbered from zero, and the region it
    can hold runs ``0 .. shape * cell``.
    """
    if not shape:
        return None
    return (
        tuple(0.0 for _ in shape),
        tuple(float(s) * float(c) for s, c in zip(shape, cell)),
    )


def _fits(
    inner: tuple[Sequence[float], Sequence[float]],
    outer: tuple[Sequence[float], Sequence[float]],
) -> bool:
    lo_i = np.asarray(inner[0], dtype=np.float64)
    hi_i = np.asarray(inner[1], dtype=np.float64)
    lo_o = np.asarray(outer[0], dtype=np.float64)
    hi_o = np.asarray(outer[1], dtype=np.float64)
    return bool((lo_i >= lo_o).all() and (hi_i <= hi_o).all())


def allocated_grid(level: Any) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """The cell grid a level's arrays were actually allocated with.

    Not the same thing as the declared bounds divided by the cell size,
    and the difference is not academic: a store created with bounds
    ``0..10`` and 3-unit cells declares a ``4x4x4`` grid and
    ``level_grid_layout`` agrees, while the ``vertices`` array on disk can
    be ``4x3x4`` because the writer sized it from the extent of the data
    it happened to be given.  A write to the missing cell then fails deep
    inside the storage layer with a chunk-coordinate error that says
    nothing about which object caused it.

    The allocation is what a write is actually checked against, so it is
    what this module checks against too.
    """
    cell = tuple(float(v) for v in level.scale)
    try:
        # The array's own shape, not ``read_array_meta`` -- that returns
        # the array's *attributes* (nonempty_chunks, dtype, encoding) and
        # has no shape key at all, so reading it here would silently fall
        # through to the declared grid and defeat the whole check.
        shape = tuple(int(s) for s in level.store.zarr_group["vertices"].shape)
    except Exception:  # noqa: BLE001 - nothing allocated yet
        shape = ()
    if not shape:
        grid = level.grid
        shape = tuple(int(s) for s in getattr(grid, "shape", ()) or ())
    return shape, cell


def _grid_mask(
    parts: Sequence[npt.NDArray[Any]],
    shape: Sequence[int],
    cell: Sequence[float],
) -> tuple[npt.NDArray[np.bool_], list[tuple[int, ...]]]:
    """Per-object: do all its vertices land in an allocated cell?

    Whole objects, not vertices.  Dropping the stray vertices of a
    polyline would join the two sides of the gap into one edge, inventing
    a connection that does not exist — so an object that does not fit is
    refused or skipped entire.

    The chunk arithmetic is ``floor(p / cell)``, which is what
    ``assign_chunks`` does and therefore what the writer will do.
    Re-deriving it any other way risks disagreeing by one cell at a
    boundary, which is precisely where these failures happen.
    """
    keep = np.ones(len(parts), dtype=bool)
    offenders: list[tuple[int, ...]] = []
    if not shape:
        return keep, offenders
    extent = np.asarray(shape, dtype=np.int64)
    size = np.asarray(cell, dtype=np.float64)
    for i, part in enumerate(parts):
        arr = np.asarray(part, dtype=np.float64)
        if arr.size == 0:
            continue
        coords = np.floor(arr / size).astype(np.int64)
        bad = (coords < 0).any(axis=1) | (coords >= extent).any(axis=1)
        if bad.any():
            keep[i] = False
            offenders.append(tuple(int(c) for c in coords[bad][0]))
    return keep, offenders


def _overhang(
    inner: tuple[Sequence[float], Sequence[float]],
    outer: tuple[Sequence[float], Sequence[float]],
) -> str:
    lo_i = np.asarray(inner[0], dtype=np.float64)
    hi_i = np.asarray(inner[1], dtype=np.float64)
    lo_o = np.asarray(outer[0], dtype=np.float64)
    hi_o = np.asarray(outer[1], dtype=np.float64)
    below = np.maximum(lo_o - lo_i, 0.0)
    above = np.maximum(hi_i - hi_o, 0.0)
    return f"below min by {np.round(below, 3).tolist()}, above max by {np.round(above, 3).tolist()}"


def _fmt(box: tuple[Sequence[float], Sequence[float]]) -> str:
    lo = np.round(np.asarray(box[0], dtype=np.float64), 3).tolist()
    hi = np.round(np.asarray(box[1], dtype=np.float64), 3).tolist()
    return f"{lo} -> {hi}"


def _prefix_for(spec: str | Mapping[str, str] | None, source: Source) -> str:
    if spec is None:
        return ""
    if isinstance(spec, str):
        return spec
    return str(spec.get(source.info.label, ""))


def _slug(label: str) -> str:
    keep = [c if (c.isalnum() or c in "-_.") else "-" for c in str(label)]
    return "".join(keep).strip("-")[:48] or "source"


def _refresh_level_metadata(dataset: Any, level: Any, fallback: int) -> int:
    """Re-stamp level 0's ``vertex_count`` and array inventory after a write.

    The edit path updates neither.  A level that says it holds zero
    vertices is not merely cosmetic: ``Level.vertex_count`` is read from
    metadata precisely so callers need not scan, so every consumer that
    trusts it — a viewer sizing a progress bar, a coarsener deciding
    whether there is anything to do — is told the level is empty.

    Counted from the fragment index rather than accumulated, so a count
    that was already wrong before this merge is corrected rather than
    added to.
    """
    from zarr_vectors.building import refresh_arrays_present
    from zarr_vectors.building import update_level_metadata

    total = fallback
    try:
        total = int(_count_level_vertices(level.store))
    except Exception:  # noqa: BLE001 - fall back to what we added
        pass
    # update_level_metadata takes the LEVEL group, not the root plus an
    # index -- the root form raises TypeError, which an except-and-carry-on
    # turns into a level that silently keeps its stale count.
    try:
        update_level_metadata(level.store, vertex_count=int(total))
    except Exception:  # noqa: BLE001
        pass
    try:
        refresh_arrays_present(level.store)
    except Exception:  # noqa: BLE001
        pass
    return total


def _count_level_vertices(level_group: Any) -> int:
    """Total stored vertices at a level, from the fragment indices.

    ``chunk_local_to_global_offsets`` derives the count by dividing each
    ``vertices`` blob's byte length by ``ndim * itemsize`` with **ndim
    hardcoded to 3**, so on a 2-D store it reported two thirds of the truth
    and stamped that on the level.  The fragment index carries the row count
    directly, is dimension-independent, and its blobs are a rounding error
    next to the vertex payloads this used to read in full.
    """
    from zarr_vectors.building import list_chunk_keys, read_vertex_fragment_index

    total = 0
    for chunk in list_chunk_keys(level_group):
        try:
            index = read_vertex_fragment_index(level_group, chunk)
        except Exception:  # noqa: BLE001 - a chunk with no index holds nothing
            continue
        rows = 0
        for f in range(index.num_fragments):
            if index.is_range(f):
                start, count = index.range(f)
                rows = max(rows, int(start) + int(count))
            else:
                idx = np.asarray(index.indices(f), dtype=np.int64)
                if idx.size:
                    rows = max(rows, int(idx.max()) + 1)
        total += rows
    return total


def _rebuild_manifests(dataset: Any) -> None:
    """Re-derive every per-chunk presence manifest from what is on disk.

    The same step the parallel ingesters need: ``nonempty_chunks`` is kept
    by read-modify-write, so a write that touches a cell the manifest does
    not list leaves ``list_chunk_keys`` under-reporting, and a later read
    driven off it silently skips those cells.
    """
    from zarr_vectors.building import rebuild_presence

    for index in dataset.levels:
        # ``Group.rebuild_nonempty_manifests`` does not exist -- the call
        # raised AttributeError on the first level and the except swallowed
        # it, so this never rebuilt anything.  ``rebuild_presence`` is the
        # supported spelling, and the one the parallel ingesters use.
        rebuild_presence(dataset.level(int(index)).store)
