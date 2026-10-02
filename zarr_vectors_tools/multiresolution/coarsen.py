"""Multi-resolution pyramid construction orchestrator.

Two entry points (use one):

* ``build_pyramid(store, factors=[(cf_1, sf_1), ...])`` builds every
  coarser level in sequence, optionally emitting cross-level link
  arrays (``cross_level_storage="implicit"`` or ``"explicit"``).
* ``coarsen_level(store, source, target, coarsen_factor=..., sparsity_factor=...)``
  writes a single coarser level for callers that want manual control.

Points, graphs and lines use the per-object pyramid: each surviving
object's vertices are aggregated into bin centroids (metavertices), each
object keeps its own row for a bin it shares with others, and per-object
OIDs are preserved across levels.  Other geometries are routed to their own
coarseners (see :func:`select_coarsener_key`).
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from zarr_vectors.building import (
    LevelMetadata,
    build_vertex_chunk_mapping,
    create_links_array,
    create_links_family,
    create_object_index_array,
    create_resolution_level,
    create_vertices_array,
    finalize_links,
    get_level_chunk_shape,
    get_resolution_level,
    link_endpoint_scales,
    link_family_policy,
    links_group_path,
    links_path,
    list_chunk_keys,
    list_link_deltas,
    list_link_offsets,
    list_resolution_levels,
    open_store,
    read_all_object_manifests,
    read_chunk_vertices,
    read_level_metadata,
    read_link_arrays,
    read_links,
    read_object_attributes,
    read_root_metadata,
    read_vertex_fragment_index,
    rebuild_presence,
    refresh_arrays_present,
    update_root_metadata,
    write_chunk_links,
    write_chunk_vertices,
    write_links,
    write_object_index,
)
from zarr_vectors.constants import (
    CAP_MULTISCALE_LINKS,
    CAP_PRESERVED_OBJECT_IDS,
    COARSEN_PER_OBJECT,
    DEFAULT_CROSS_LEVEL_DEPTH,
    DEFAULT_CROSS_LEVEL_STORAGE,
    GEOM_LINE,
    GEOM_MESH,
    GEOM_POINT_CLOUD,
    LINKS_IMPLICIT_BRANCHES,
    LINKS_IMPLICIT_SEQUENTIAL,
    VALID_XLEVEL_STORAGE,
    VERTEX_FRAGMENTS,
    VERTICES,
    XLEVEL_EXPLICIT,
    XLEVEL_IMPLICIT,
    XLEVEL_NONE,
)
from zarr_vectors.exceptions import ArrayError, CoarseningError, StoreError
from zarr_vectors.typing import ChunkCoords

from zarr_vectors_tools.algorithms._graph_edges import component_roots
from zarr_vectors_tools.algorithms._links import chunk_key_str
from zarr_vectors_tools.multiresolution.coarsen_implicit import (
    coarse_chunks_of,
    positions_in_run,
    segment_object_by_coarse_chunk,
)
from zarr_vectors_tools.multiresolution.groupings import (
    group_labels_for,
    propagate_groupings,
    surviving_oids_from,
)
from zarr_vectors_tools.multiresolution.object_index import carry_object_columns
from zarr_vectors_tools.multiresolution.object_selection import apply_sparsity
from zarr_vectors_tools.multiresolution.strategies.graphs import contract_edges

# ===================================================================
# Coarsener registry (pluggable per-geometry downsampling strategies)
# ===================================================================

# A coarsener takes ``(store_path, source_level, target_level)`` plus the
# uniform keyword set ``coarsen_level`` forwards, and writes the target
# level, returning a summary dict.  Built-ins are registered at module load
# (bottom of file); callers/experiments may override or add keys via
# :func:`register_coarsener` without editing :func:`coarsen_level`.
Coarsener = Callable[..., dict[str, Any]]
_COARSENERS: dict[str, Coarsener] = {}

#: Cells per deferred-write flush.  ``Group.batched_writes`` buffers each
#: queued cell's encoded bytes until the block exits, so an unbounded block
#: over a wide level would hold a second copy of that level in memory.  This
#: caps the buffer while still amortising the per-cell round-trip away.
_WRITE_BATCH_CHUNKS = 4096


def coarsener_keys() -> list[str]:
    """The registered coarsener keys, sorted: what ``method=`` accepts."""
    return sorted(_COARSENERS)


def register_coarsener(key: str, fn: Coarsener) -> None:
    """Register (or override) the coarsener used for a dispatch ``key``."""
    _COARSENERS[key] = fn


def get_coarsener(key: str) -> Coarsener:
    """Look up a registered coarsener; raise if none is registered."""
    try:
        return _COARSENERS[key]
    except KeyError:
        raise CoarseningError(
            f"no coarsener registered for {key!r} "
            f"(registered: {sorted(_COARSENERS)})"
        ) from None


def select_coarsener_key(root_meta: Any) -> str:
    """Pick the coarsener key for a store from its root metadata.

    Skeleton stores (``implicit_sequential_with_branches``) use the
    skeleton-aware decimator; mesh stores use chunk-local vertex clustering;
    streamline stores (``implicit_sequential`` with geometry_type
    ``"streamline"``) use the RDP polyline coarsener; everything else uses the
    per-object pyramid.
    """
    if root_meta.links_convention == LINKS_IMPLICIT_BRANCHES:
        return "skeleton"
    # Meshes must not reach the per-object pyramid: it concatenates every
    # fragment each object names (quadratic when objects share chunk-wide
    # fragments) and rebuilds links at a hardcoded ``link_width=2``, which a
    # triangle cannot survive.
    if GEOM_MESH in (root_meta.geometry_types or []):
        return "mesh"
    if (
        root_meta.links_convention == LINKS_IMPLICIT_SEQUENTIAL
        and "streamline" in (root_meta.geometry_types or [])
    ):
        return "polyline"
    return "per_object"


# ===================================================================
# Explicit RDP tolerance, and the level record that keeps it
# ===================================================================

#: Level-group attribute key under which this package records coarsening
#: parameters that core's ``LevelMetadata`` has no field for.
#:
#: A sibling of core's ``zarr_vectors_level`` block rather than a key inside
#: it.  ``create_resolution_level`` rewrites that block wholesale from
#: ``LevelMetadata.to_dict()``, so an unknown key inside it is dropped the
#: next time a writer restamps the level -- which the polyline coarsener
#: itself does once Phase A knows the vertex count -- and
#: ``update_level_metadata`` refuses fields the dataclass does not declare.
#: A separate top-level key is left alone by both, and core's validation
#: does not inspect it.
TOOLS_LEVEL_ATTRS_KEY: str = "zarr_vectors_tools"

#: Sub-key of :data:`TOOLS_LEVEL_ATTRS_KEY` holding the coarsening record.
_COARSENING_RECORD: str = "coarsening"


def read_coarsening_record(root: Any, level: int) -> dict[str, Any]:
    """The tools-owned coarsening record of one level, or ``{}`` if it has none.

    Polyline levels built in ``rdp`` mode carry:

    * ``rdp_tolerance`` -- the Douglas-Peucker tolerance the level was
      simplified at, in store coordinate units.  ``None`` when nothing was
      simplified (a ``coarsen_factor <= 1`` with no explicit tolerance).
    * ``rdp_tolerance_source`` -- ``"explicit"`` when the caller supplied
      the tolerance, ``"derived"`` when it came from the level's bin.

    Levels written by other strategies, and stores written before the record
    existed, return ``{}``.
    """
    level_group = get_resolution_level(root, level)
    block = level_group.attrs.get(TOOLS_LEVEL_ATTRS_KEY) or {}
    record = block.get(_COARSENING_RECORD) if isinstance(block, dict) else None
    return dict(record) if isinstance(record, dict) else {}


def _write_coarsening_record(root: Any, level: int, **fields: Any) -> None:
    """Merge ``fields`` into one level's tools-owned coarsening record.

    Takes the root and re-opens the level rather than accepting a level
    handle: a level group's attributes are written back whole from the
    handle's in-memory copy, and a coarsener's own handle predates its
    second ``create_resolution_level`` stamp, so writing through it would
    put that handle's stale ``vertex_count`` back on disk.
    """
    level_group = get_resolution_level(root, level)
    block = dict(level_group.attrs.get(TOOLS_LEVEL_ATTRS_KEY) or {})
    record = dict(block.get(_COARSENING_RECORD) or {})
    record.update(fields)
    block[_COARSENING_RECORD] = record
    level_group.attrs.update({TOOLS_LEVEL_ATTRS_KEY: block})


def check_rdp_tolerance(value: Any, *, name: str = "rdp_tolerance") -> float:
    """``value`` as a float, or raise naming ``name`` if it is no tolerance.

    Zero and negative values are refused rather than read as "simplify
    nothing".  The worker does treat a non-positive epsilon as a no-op, but
    a caller who wants no simplification has ``coarsen_factor=1`` for that,
    and a level recording a tolerance of 0 would read as though one applied.
    """
    try:
        tolerance = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{name}={value!r} is not a number; give a distance in store "
            f"coordinate units"
        ) from None
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(
            f"{name}={value!r} must be a finite distance > 0, in store "
            f"coordinate units"
        )
    return tolerance


def validate_rdp_tolerances(
    rdp_tolerances: Sequence[float] | None,
    *,
    n_levels: int,
    coarsen_mode: str,
    name: str = "rdp_tolerances",
) -> list[float] | None:
    """Check a per-level explicit RDP tolerance list, without touching a store.

    What can be checked from the arguments alone: one entry per coarser
    level, every entry a positive distance, and a coarsen mode that has a
    tolerance at all.  Whether the store's geometry goes through the RDP
    coarsener is checked by :func:`build_pyramid`, which has the store.
    Ingesters that build the pyramid inline call this before their own
    (expensive) level-0 write, so a bad list fails before any work is done.

    Returns:
        ``None`` for ``None``, else the tolerances as floats.

    Raises:
        ValueError: Naming ``name`` and what is wrong with it.
    """
    if rdp_tolerances is None:
        return None
    refusal = _rdp_mode_refusal(coarsen_mode)
    if refusal is not None:
        raise ValueError(f"{name} does not apply: {refusal}")
    if isinstance(rdp_tolerances, (str, bytes, int, float)):
        raise ValueError(
            f"{name} must be a sequence with one tolerance per coarser level, "
            f"got {rdp_tolerances!r}"
        )
    values = list(rdp_tolerances)
    if len(values) != n_levels:
        raise ValueError(
            f"{name} has {len(values)} entries for {n_levels} coarser "
            f"level(s); give exactly one tolerance per level"
        )
    return [
        check_rdp_tolerance(value, name=f"{name}[{i}]")
        for i, value in enumerate(values)
    ]


def _rdp_mode_refusal(coarsen_mode: str) -> str | None:
    """Why an explicit RDP tolerance cannot apply in ``coarsen_mode``, or ``None``."""
    if coarsen_mode != "decimate":
        return None
    return (
        "coarsen_mode='decimate' thins each level by a stride (its coarsen "
        "factor) and has no distance tolerance to set"
    )


def _rdp_tolerance_refusal(root_meta: Any, key: str) -> str | None:
    """Why an explicit RDP tolerance cannot apply to a store, or ``None``.

    Only the polyline coarsener runs Douglas-Peucker.  The skeleton coarsener
    decimates by stride, the per-fragment one reads ``rdp`` as a stride as
    well, and the point, mesh and graph coarseners bin -- so a tolerance
    handed to any of them would be silently meaningless.
    """
    if key == "polyline":
        return None
    return (
        f"only the RDP polyline coarsener has a distance tolerance, and this "
        f"store is coarsened by {key!r} (geometry_types="
        f"{list(root_meta.geometry_types or [])}, links_convention="
        f"{root_meta.links_convention!r})"
    )


# ===================================================================
# Single-level coarsening
# ===================================================================

def coarsen_level(
    store_path: str | Path,
    source_level: int,
    target_level: int,
    *,
    coarsen_factor: float = 1.0,
    sparsity_factor: float = 1.0,
    chunk_scale_factor: int | tuple[int, ...] = 1,
    sparsity_strategy: str = "random",
    sparsity_seed: int | None = None,
    cross_level_storage: str = XLEVEL_NONE,
    coarsen_mode: str = "rdp",
    compressor: Any = None,
    executor: Any = None,
    method: str | None = None,
    rdp_tolerance: float | None = None,
    sparsity_attribute: str | None = None,
) -> dict[str, Any]:
    """Coarsen a single level and write it to the store.

    Per-object vertex aggregation with stable OIDs across levels.  Points
    and paths (lines, polylines) merge every object's vertices that share a
    bin into one centroid, which each of those objects stores as its own
    row.  Graphs merge vertices only within one object and one connected
    component, and their coarse edges are exactly the images of the source
    level's edges.  See :func:`_per_object_coarsen`.

    Args:
        store_path: Path to the zarr vectors store.
        source_level: Level to read from.
        target_level: Level to write to (must not exist).
        coarsen_factor: Per-object vertex aggregation factor (>= 1),
            expressed as a **ratio against the source level's bin**, not the
            root's: the target bin is ``source_level.bin_shape *
            coarsen_factor``, seeded from the root's effective bin at level 0.
            Factors therefore compound down a pyramid.  For the per-object
            coarsener ``1.0`` still bins at the source level's bin, which at
            level 1 is the root bin (the chunk shape unless ``bin_shape`` was
            set), so it is not the identity there.
        sparsity_factor: Object-dropping factor (≥ 1).  Survivors keep
            their OIDs; dropped objects leave empty manifest slots.
            ``1.0`` is the identity (no drop).
        chunk_scale_factor: Per-axis multiplier applied to the source
            level's ``chunk_shape`` to derive the target level's
            ``chunk_shape``.  ``1`` (default) keeps the chunk grid
            unchanged.  Scalar values apply uniformly to every axis;
            tuples set per-axis multipliers.  Each multiplier must be a
            positive integer (nested chunk grids).  When the resulting
            target chunk_shape differs from the root chunk_shape it is
            stamped on the target level's ``LevelMetadata.chunk_shape``;
            otherwise the target inherits from root.
        sparsity_strategy: Object selection strategy.
        sparsity_seed: Random seed.
        sparsity_attribute: Name of a scalar ``object_attributes`` column on
            level 0 to rank objects by, keeping the highest values.  Required
            with ``sparsity_strategy="attribute"`` and refused otherwise.
            Level 0 is read whatever the source level: object ids are kept
            across levels, so its column describes every level's objects.
        cross_level_storage: When called via ``build_pyramid`` this is
            threaded through to enable inline ``±1`` cross-level link
            emission.  Standalone callers should leave it at the
            ``"none"`` default.
        coarsen_mode: Only consulted by the polyline (streamline) coarsener:
            ``"rdp"`` (default) does Douglas-Peucker simplification;
            ``"decimate"`` does uniform stride decimation, in which case
            ``coarsen_factor`` is interpreted as the stride. Ignored by the
            other coarseners.
        compressor: Codec for the TARGET level's per-chunk arrays.  ``None``
            (default) stores raw.  Each level's codec is independent — a
            chunk array's pipeline is fixed when it is created — so pass the
            same value the level-0 ingest used to keep a store uniform.
            See :func:`zarr_vectors.encoding.compression.resolve_compressor`.
        executor: Optional ``map``-like ``(func, items, shared) ->
            list[result]`` callable (e.g. ``dask_executor``) used by the
            chunk-local skeleton and polyline coarseners to parallelize
            per-target-chunk work.  ``None`` (default) runs serially
            in-process.  Ignored by the per-object coarsener.
        rdp_tolerance: Explicit Douglas-Peucker tolerance for this level, in
            store coordinate units: every SOURCE-level vertex lies within
            this distance of the target level's simplified line.  ``None``
            (default) derives it as half the smallest edge of the target
            bin (``source_level.bin_shape * coarsen_factor``), or simplifies
            nothing when ``coarsen_factor <= 1``.  Only the polyline
            coarsener in ``rdp`` mode has a tolerance, so a value is refused
            with ``coarsen_mode="decimate"`` and for any store routed to
            another coarsener.  Forwarded to the coarsener only when set, so
            coarseners registered without the keyword keep working.  The
            tolerance used is recorded on the level; see
            :func:`read_coarsening_record`.

    Returns:
        Summary dict.  Always includes ``method``,
        ``preserves_object_ids``, ``vertex_count``.

    Skeleton stores (``links_convention =
    "implicit_sequential_with_branches"``) are routed to the
    skeleton-aware decimator
    (:func:`zarr_vectors_tools.multiresolution.strategies.skeletons.coarsen_skeleton_level`);
    for those stores ``coarsen_factor`` is interpreted as the decimation
    ``stride`` (keep every k-th vertex) rather than a vertex aggregation
    factor.
    """
    root_meta = read_root_metadata(open_store(str(store_path), mode="r"))
    # Dispatch via the coarsener registry (see ``register_coarsener``) so new
    # or improved per-geometry coarseners can be plugged in without editing
    # this function.  Default keys: ``"skeleton"`` (implicit-branch stores)
    # and ``"per_object"`` (everything else).
    # ``method`` overrides the automatic geometry routing. The default (None)
    # keeps ``select_coarsener_key``'s behaviour; "per_fragment" opts into the
    # reuse-preserving strategy, which no root metadata can imply because it is
    # a policy choice (structure over compression), not a property of the data.
    key = method or select_coarsener_key(root_meta)
    coarsener = get_coarsener(key)
    extra: dict[str, Any] = {}
    if rdp_tolerance is not None:
        # Checked before the coarsener runs so a refused tolerance leaves no
        # half-written target level behind.
        refusal = _rdp_mode_refusal(coarsen_mode) or _rdp_tolerance_refusal(
            root_meta, key,
        )
        if refusal is not None:
            raise ValueError(f"rdp_tolerance does not apply: {refusal}")
        extra["rdp_tolerance"] = check_rdp_tolerance(rdp_tolerance)
    if sparsity_strategy == "attribute" or sparsity_attribute is not None:
        # Forwarded only when used, like rdp_tolerance, so coarseners
        # registered without the keyword keep working.
        extra["attribute_values"] = _sparsity_attribute_values(
            store_path, sparsity_strategy, sparsity_attribute,
        )
    return coarsener(
        store_path,
        source_level,
        target_level,
        coarsen_factor=coarsen_factor,
        sparsity_factor=sparsity_factor,
        chunk_scale_factor=chunk_scale_factor,
        sparsity_strategy=sparsity_strategy,
        sparsity_seed=sparsity_seed,
        cross_level_storage=cross_level_storage,
        coarsen_mode=coarsen_mode,
        compressor=compressor,
        executor=executor,
        **extra,
    )


def _carry_vertex_attributes(
    src_group: Any,
    level_group: Any,
    flat_refs: list[tuple[ChunkCoords, int]],
    inverse: npt.NDArray[np.int64],
    n_metavertices: int,
    per_chunk_mvs: dict[ChunkCoords, list[npt.NDArray[np.int64]]],
    *,
    compressor: Any = None,
) -> list[str]:
    """Give every metavertex a value for each of the source's vertex attributes.

    Without this a coarse point level had positions only, so a reader
    colouring by an attribute saw nothing (or zeros) as soon as it zoomed out.
    A float column takes the mean of the vertices merged into the bin -- the
    same aggregation the positions get; any other column (integer codes,
    labels, booleans) takes the first vertex's value, since a mean of
    category codes is not a category.  One attribute at a time, so a wide
    store costs one column of memory, not all of them.
    """
    from zarr_vectors.building import (
        create_attribute_array,
        read_chunk_attributes,
        refresh_arrays_present,
        write_chunk_attributes,
    )
    from zarr_vectors.constants import VERTEX_ATTRIBUTES

    if VERTEX_ATTRIBUTES not in src_group or not flat_refs:
        return []
    names = list(src_group[VERTEX_ATTRIBUTES].children())
    chunks = sorted({tuple(int(c) for c in cc) for cc, _ in flat_refs})
    chunk_keys = [".".join(str(c) for c in cc) for cc in chunks]
    first = None
    counts = None
    carried: list[str] = []
    for name in names:
        try:
            meta = dict(src_group.read_array_meta(f"{VERTEX_ATTRIBUTES}/{name}"))
        except Exception:  # noqa: BLE001 - unreadable column: leave it out
            continue
        dtype = np.dtype(meta.get("dtype", "float32"))
        row_shape = tuple(int(d) for d in (meta.get("row_shape") or ()))
        ncols = int(np.prod(row_shape)) if row_shape else None
        # One prefetch for the column's cells and the fragment indexes that
        # split them: read cell by cell, the round-trips were most of a
        # pyramid's build time.
        plan = [
            (f"{VERTEX_ATTRIBUTES}/{name}", chunk_keys),
            (VERTEX_FRAGMENTS, chunk_keys),
        ]
        cells: dict[ChunkCoords, list[np.ndarray]] = {}
        parts: list[np.ndarray] = []
        try:
            with src_group.cached_nodes(), src_group.batched_reads(plan):
                for cc in chunks:
                    cells[cc] = read_chunk_attributes(
                        src_group, name, cc, dtype=dtype, ncols=ncols,
                    )
            for cc, fragment_idx in flat_refs:
                parts.append(np.asarray(cells[tuple(int(c) for c in cc)][fragment_idx]))
        except Exception:  # noqa: BLE001 - a column missing some chunks
            continue
        column = np.concatenate(parts, axis=0)
        if len(column) != len(inverse):
            continue
        if dtype.kind == "f":
            if counts is None:
                counts = np.bincount(inverse, minlength=n_metavertices)
            flat = column.reshape(len(column), -1).astype(np.float64)
            agg = np.stack([
                np.bincount(inverse, weights=flat[:, c], minlength=n_metavertices) / counts
                for c in range(flat.shape[1])
            ], axis=1).astype(dtype).reshape((n_metavertices, *column.shape[1:]))
        else:
            if first is None:
                _, first = np.unique(inverse, return_index=True)
            agg = column[first]
        # A dictionary-encoded column keeps its labels, order and missing code,
        # or a missing value (-1) would decode as the last category.
        extra = {
            k: meta[k] for k in ("encoding", "categories", "ordered", "_FillValue")
            if k in meta
        }
        create_attribute_array(
            level_group, name, dtype=str(dtype),
            channel_names=meta.get("channel_names") or (
                [f"ch{i}" for i in range(ncols)] if ncols else None
            ),
            extra_meta=extra or None,
            exist_ok=True,
        )
        items = sorted(per_chunk_mvs.items())
        for start in range(0, len(items), _WRITE_BATCH_CHUNKS):
            with level_group.batched_writes(compressor=compressor):
                for cc, mv_lists in items[start:start + _WRITE_BATCH_CHUNKS]:
                    write_chunk_attributes(
                        level_group, name, cc, [agg[mvs] for mvs in mv_lists],
                        dtype=dtype,
                    )
        carried.append(name)
    if carried:
        refresh_arrays_present(level_group)
    return carried


def _sparsity_attribute_values(
    store_path: str | Path, strategy: str, name: str | None,
) -> npt.NDArray[np.float64]:
    """The per-object values ``sparsity_strategy="attribute"`` ranks by."""
    if strategy != "attribute":
        raise ValueError(
            f"sparsity_attribute={name!r} only applies with "
            f"sparsity_strategy='attribute', not {strategy!r}"
        )
    if not name:
        raise ValueError(
            "sparsity_strategy='attribute' needs sparsity_attribute: the "
            "object attribute to rank objects by"
        )
    level0 = get_resolution_level(open_store(str(store_path), mode="r"), 0)
    try:
        values = np.asarray(read_object_attributes(level0, name), dtype=np.float64)
    except Exception as exc:  # noqa: BLE001 - absent, or not numeric
        raise ValueError(
            f"sparsity_strategy='attribute' ranks by object_attributes/{name}, "
            f"which level 0 does not have as a numeric column ({exc})"
        ) from exc
    if values.ndim != 1:
        raise ValueError(
            f"object_attributes/{name} has {values.shape[1:]} values per "
            f"object; ranking needs one"
        )
    return values


def _per_object_signals(
    src_manifests: list,
    src_fragment_positions: dict,
    n_objects: int,
    ndim: int,
    *,
    needed: bool,
) -> tuple[npt.NDArray[np.float64] | None, npt.NDArray[np.float64] | None]:
    """Per-object ``(size, representative point)`` for the sparsity strategies.

    ``size`` is the object's vertex count — the generalisation of "length" for
    a geometry that has no path, and the same quantity the skeleton and
    polyline coarseners fall back to.  The representative point is the
    object's first stored vertex, which is what ``spatial_coverage`` and
    ``point_thinning`` bin on.

    Returns ``(None, None)`` when ``needed`` is False so the unconditional
    path stays free of an O(fragments) walk.
    """
    if not needed:
        return None, None
    lengths = np.zeros(n_objects, dtype=np.float64)
    points = np.zeros((n_objects, ndim), dtype=np.float64)
    for oid in range(n_objects):
        first: npt.NDArray | None = None
        total = 0
        for cc, fragment_idx in src_manifests[oid]:
            fragment = src_fragment_positions.get((cc, fragment_idx))
            if fragment is None or len(fragment) == 0:
                continue
            if first is None:
                first = fragment[0]
            total += len(fragment)
        lengths[oid] = float(total)
        if first is not None:
            points[oid] = first
    return lengths, points


def _per_object_coarsen(
    *,
    store_path: str | Path,
    source_level: int,
    target_level: int,
    coarsen_factor: float,
    sparsity_factor: float,
    chunk_scale_factor: int | tuple[int, ...] = 1,
    sparsity_strategy: str,
    sparsity_seed: int | None,
    cross_level_storage: str = XLEVEL_NONE,
    compressor: Any = None,
    attribute_values: npt.NDArray[np.float64] | None = None,
) -> dict[str, Any]:
    """Per-object pyramid: aggregate within-bin source vertices into
    metavertices (bin centroids), preserving each surviving object's OID.

    * **Points** (``implicit_sequential``, point cloud): every bin an object
      visits becomes one row of that object, at the centroid of all the
      vertices -- of any object -- in the bin.  No ``links`` family.
    * **Lines and polylines** (``implicit_sequential``): as points, but each
      object keeps its walk order, split into one fragment per coarse chunk
      visited; consecutive fragments are joined by a directed ``links/0``
      record.  A line whose two ends fall in one bin is dropped from the
      level and counted (``objects_collapsed``; the level's coarsening
      record keeps ``collapsed_objects``).
    * **Graphs** (explicit links): vertices merge only within one object and
      one connected component, so no level joins two components of the
      level below.  The coarse edges are the source edges' images under the
      vertex -> metavertex map, without self-loops or duplicates; edge
      attributes are averaged (floats) or take the first merged edge's value.

    Every coarse vertex is stored in the chunk containing it.  With
    ``cross_level_storage`` set, each source vertex of a surviving object is
    linked to the coarse row of the same object it was merged into.
    """
    root = open_store(str(store_path), mode="r+")
    root_meta = read_root_metadata(root)
    ndim = root_meta.sid_ndim

    # Source level's chunk_shape — may itself be a per-level override.
    try:
        src_level_meta = read_level_metadata(root, source_level)
    except Exception:
        src_level_meta = None
    source_chunk_shape = get_level_chunk_shape(root_meta, src_level_meta)

    # The bin ``coarsen_factor`` multiplies is the SOURCE LEVEL's, not the
    # root's, so factors compose per level: ``[2, 2, 2]`` bins at 2x, 4x, 8x
    # the root bin rather than 2x three times over.  Level 0 carries no
    # bin_shape of its own, so the root's effective bin seeds the chain.
    _src_bin = getattr(src_level_meta, "bin_shape", None) if src_level_meta else None
    base_bin = (
        tuple(float(b) for b in _src_bin) if _src_bin
        else root_meta.effective_bin_shape
    )

    # Target level's chunk_shape = source × chunk_scale_factor (per-axis).
    if isinstance(chunk_scale_factor, (tuple, list)):
        if len(chunk_scale_factor) != ndim:
            raise CoarseningError(
                f"chunk_scale_factor rank {len(chunk_scale_factor)} "
                f"!= sid_ndim {ndim}"
            )
        chunk_scale = tuple(int(r) for r in chunk_scale_factor)
    else:
        chunk_scale = tuple(int(chunk_scale_factor) for _ in range(ndim))
    if any(r < 1 for r in chunk_scale):
        raise CoarseningError(
            f"chunk_scale_factor must be positive integers per axis, "
            f"got {chunk_scale}"
        )
    target_chunk_shape = tuple(
        float(s) * int(r) for s, r in zip(source_chunk_shape, chunk_scale)
    )
    # The on-disk per-level chunk_shape field is omitted when the
    # target equals root (the implicit default).  Compare via float.
    target_chunk_shape_override: tuple[float, ...] | None
    if all(
        abs(t - r) < 1e-9
        for t, r in zip(target_chunk_shape, root_meta.chunk_shape)
    ):
        target_chunk_shape_override = None
    else:
        target_chunk_shape_override = target_chunk_shape

    src_group = get_resolution_level(root, source_level)

    # --- Step 0: read source manifests + vertex positions ----------------
    # Read source vertex positions, indexed by (chunk_coords, fragment_idx).
    src_chunk_keys = list(list_chunk_keys(src_group, VERTICES))
    src_fragment_positions: dict[tuple[ChunkCoords, int], npt.NDArray] = {}
    # Chunk-local start row of every fragment, accumulated as its chunk is
    # read.  ``read_chunk_vertices`` returns fragments in fragment-index
    # order, so the running sum here is the same answer the post-hoc scan
    # used to produce -- at O(fragments) rather than O(chunks x fragments).
    src_chunk_fragment_starts: dict[ChunkCoords, dict[int, int]] = {}
    # Rows per source chunk, for the (chunk, row) -> source vertex tables the
    # edge and cross-level mapping below look endpoints up in.
    src_chunk_rows: dict[ChunkCoords, int] = {}
    # One asyncio.gather for the whole source level rather than one
    # round-trip per chunk.  read_chunk_vertices' own _maybe_batched_reads
    # is a no-op inside an outer plan, so each chunk is served from this
    # prefetch instead of opening a prefetch of its own.
    _src_key_strs = [chunk_key_str(cc) for cc in src_chunk_keys]
    with src_group.batched_reads([
        (VERTICES, _src_key_strs),
        (VERTEX_FRAGMENTS, _src_key_strs),
    ]):
        for cc in src_chunk_keys:
            try:
                fragments = read_chunk_vertices(
                    src_group, cc, dtype=np.float32, ndim=ndim,
                )
            except ArrayError:
                continue
            starts_map: dict[int, int] = {}
            cum = 0
            for fragment_idx, fragment in enumerate(fragments):
                src_fragment_positions[(cc, fragment_idx)] = fragment
                starts_map[fragment_idx] = cum
                cum += len(fragment)
            src_chunk_fragment_starts[cc] = starts_map
            src_chunk_rows[cc] = cum

    src_has_objects = "object_index" in src_group
    if src_has_objects:
        src_manifests = read_all_object_manifests(src_group)
    else:
        # No object_index — treat the level as one implicit object whose
        # manifest enumerates every fragment in chunk-major order.
        implicit: list[tuple[ChunkCoords, int]] = []
        for cc in src_chunk_keys:
            fragment_idx = 0
            while (cc, fragment_idx) in src_fragment_positions:
                implicit.append((cc, fragment_idx))
                fragment_idx += 1
        src_manifests = [implicit] if implicit else []
    n_src_objects = len(src_manifests)
    if n_src_objects == 0:
        return {
            "vertex_count": 0,
            "object_count": 0,
            "objects_kept": 0,
            "method": COARSEN_PER_OBJECT,
            "preserves_object_ids": True,
        }

    # --- Step 1: drop a fraction of source objects ----------------------
    keep_oids: list[int]
    if sparsity_factor > 1.0 and n_src_objects > 1:
        keep_frac = 1.0 / sparsity_factor
        # Objects already emptied by an earlier pyramid level's sparsity
        # drop must not be re-"kept" here — see `apply_sparsity`'s
        # `alive_mask` docstring.
        alive_mask = np.array(
            [len(src_manifests[oid]) > 0 for oid in range(n_src_objects)],
            dtype=bool,
        )
        # Every strategy the CLI offers needs a per-object signal, and this
        # coarsener used to pass none: ``--sparsity-strategy length`` — which
        # the docs recommend for large pyramids — raised "'length' strategy
        # requires 'lengths' array" on any store that routes here, i.e. every
        # point cloud, mesh and graph.  All three signals come from fragments
        # Step 0 already read, so supplying them costs one pass over the
        # manifests and only when sparsity is actually active.
        lengths, representative_points = _per_object_signals(
            src_manifests, src_fragment_positions, n_src_objects, ndim,
            needed=sparsity_strategy in ("length", "spatial_coverage", "point_thinning"),
        )
        group_labels = (
            group_labels_for(src_group, n_src_objects)
            if sparsity_strategy == "group" else None
        )
        kept = apply_sparsity(
            n_src_objects, keep_frac, sparsity_strategy,
            seed=sparsity_seed,
            lengths=lengths,
            representative_points=representative_points,
            group_labels=group_labels,
            attribute_values=attribute_values,
            bin_shape=base_bin,
            alive_mask=alive_mask,
            # Cumulative per level: fraction of the surviving pool, not of
            # the original count.  See apply_sparsity's `relative_to`.
            relative_to="alive",
        )
        keep_oids = sorted(int(o) for o in kept)
    else:
        # Objects already gone at the source level (dropped by sparsity, or
        # collapsed) stay gone, here too: kept, they read as present in the
        # object attributes' mask with nothing in their manifest.
        keep_oids = [
            oid for oid in range(n_src_objects) if len(src_manifests[oid]) > 0
        ]

    # Target bin shape: the SOURCE level's bin_shape x coarsen_factor, so the
    # factor is a per-level ratio and successive levels compound.
    target_bin_shape = tuple(float(b) * float(coarsen_factor) for b in base_bin)
    bin_shape_arr = np.asarray(target_bin_shape, dtype=np.float64)

    # ``implicit_sequential`` stores (points, lines, polylines) keep each
    # object's walk in one multi-vertex fragment per coarse chunk it visits,
    # so consecutive rows are its edges.  Explicit-links stores (graphs) keep
    # one fragment per object per chunk and carry every edge as a ``links/0``
    # record, which the coarse level maps rather than invents.
    use_implicit_sequential = (
        root_meta.links_convention == LINKS_IMPLICIT_SEQUENTIAL
    )
    point_cloud = GEOM_POINT_CLOUD in (root_meta.geometry_types or ())
    # A line (a two-vertex segment) whose ends fall in one bin would be stored
    # as a single vertex: no longer a segment, and nothing a renderer can
    # draw.  It is dropped from this level -- and so from every coarser one,
    # which sees an empty manifest -- and counted, rather than kept as a
    # one-vertex line.  A polyline that shrinks to one vertex keeps it: it is
    # still the object's only trace at this scale.
    drop_collapsed = (
        use_implicit_sequential and GEOM_LINE in (root_meta.geometry_types or ())
    )

    # --- Step 2: each surviving object's source vertices ----------------
    per_object_positions: dict[int, np.ndarray] = {}
    # The source fragments of each object in the order its vertices enter
    # ``all_pos``, so per-vertex attributes and source rows line up with it.
    per_object_refs: dict[int, list[tuple[ChunkCoords, int]]] = {}
    collapsed_oids: list[int] = []
    for oid in keep_oids:
        parts: list[np.ndarray] = []
        refs: list[tuple[ChunkCoords, int]] = []
        for cc, fragment_idx in src_manifests[oid]:
            fragment = src_fragment_positions.get((cc, fragment_idx))
            if fragment is None or len(fragment) == 0:
                continue
            parts.append(np.asarray(fragment, dtype=np.float32))
            refs.append((cc, fragment_idx))
        obj_positions = (
            np.concatenate(parts, axis=0) if parts
            else np.zeros((0, ndim), dtype=np.float32)
        )
        if drop_collapsed and len(obj_positions):
            obj_bins = np.floor(obj_positions / bin_shape_arr)
            if (obj_bins == obj_bins[0]).all():
                collapsed_oids.append(oid)
                continue
        per_object_positions[oid] = obj_positions
        per_object_refs[oid] = refs
    if collapsed_oids:
        _collapsed = set(collapsed_oids)
        keep_oids = [oid for oid in keep_oids if oid not in _collapsed]
    nonempty_oids = [oid for oid in keep_oids if len(per_object_positions[oid])]
    flat_refs = [ref for oid in nonempty_oids for ref in per_object_refs[oid]]

    if not nonempty_oids:
        # Surviving objects had no vertices.  Write an empty level.
        _write_empty_preserve_level(
            root, source_level, target_level,
            base_bin=base_bin,
            root_bin=root_meta.effective_bin_shape,
            coarsen_factor=coarsen_factor,
            sparsity_factor=sparsity_factor,
            inherited_num_objects=n_src_objects,
        )
        if drop_collapsed:
            _write_coarsening_record(
                root, target_level, collapsed_objects=len(collapsed_oids),
            )
        return {
            "vertex_count": 0,
            "object_count": 0,
            "objects_kept": len(keep_oids),
            "objects_collapsed": len(collapsed_oids),
            "method": COARSEN_PER_OBJECT,
            "preserves_object_ids": True,
            "shared_fragments": False,
        }

    all_pos = np.concatenate(
        [per_object_positions[oid] for oid in nonempty_oids], axis=0,
    )
    n_flat = int(all_pos.shape[0])
    obj_sizes = np.array(
        [len(per_object_positions[oid]) for oid in nonempty_oids], dtype=np.int64,
    )
    obj_starts = np.concatenate(([0], np.cumsum(obj_sizes)[:-1])).astype(np.int64)
    oid_of_v = np.repeat(np.asarray(nonempty_oids, dtype=np.int64), obj_sizes)

    # Source (chunk, row) of every flat vertex -- what a ``links/0`` record and
    # a cross-level record address -- and the reverse table, -1 for the rows
    # of dropped objects.
    src_chunk_list = sorted(src_chunk_rows)
    src_chunk_index = {cc: i for i, cc in enumerate(src_chunk_list)}
    src_rows_per_chunk = np.array(
        [src_chunk_rows[cc] for cc in src_chunk_list], dtype=np.int64,
    )
    src_chunk_base = np.concatenate(
        ([0], np.cumsum(src_rows_per_chunk)),
    ).astype(np.int64)
    flat_src_chunk = np.empty(n_flat, dtype=np.int64)
    flat_src_row = np.empty(n_flat, dtype=np.int64)
    cursor = 0
    for cc, fragment_idx in flat_refs:
        n = len(src_fragment_positions[(cc, fragment_idx)])
        flat_src_chunk[cursor:cursor + n] = src_chunk_index[cc]
        flat_src_row[cursor:cursor + n] = (
            src_chunk_fragment_starts[cc][fragment_idx] + np.arange(n)
        )
        cursor += n
    flat_of_src = np.full(int(src_chunk_base[-1]), -1, dtype=np.int64)
    flat_of_src[src_chunk_base[flat_src_chunk] + flat_src_row] = np.arange(
        n_flat, dtype=np.int64,
    )

    def _source_vertices(
        chunks: npt.NDArray[np.int64], rows: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.int64]:
        """Flat vertex of each ``(chunk, row)`` endpoint, -1 where none."""
        return _lookup_rows(
            chunks, rows, src_chunk_index, src_chunk_base,
            src_rows_per_chunk, flat_of_src,
        )

    # A graph's edges, as flat-vertex pairs.  Records whose endpoint belongs
    # to a dropped object are left out.  A path's edges need no records: each
    # object's vertices are its walk, in order.
    edge_a = edge_b = np.empty(0, dtype=np.int64)
    edge_rows = np.empty(0, dtype=np.int64)
    edges_directed = use_implicit_sequential
    if not use_implicit_sequential:
        e_chunks, e_vi = read_link_arrays(src_group, delta=0)
        if e_vi.size:
            if e_vi.shape[1] != 2:
                raise CoarseningError(
                    f"level {source_level} links/0 holds {e_vi.shape[1]}-vertex "
                    f"records; the per-object coarsener maps edges (2-vertex "
                    f"records) only"
                )
            a = _source_vertices(e_chunks[:, 0], e_vi[:, 0])
            b = _source_vertices(e_chunks[:, 1], e_vi[:, 1])
            edge_rows = np.flatnonzero((a >= 0) & (b >= 0))
            edge_a, edge_b = a[edge_rows], b[edge_rows]
        policy = link_family_policy(src_group, 0)
        edges_directed = bool(policy[2]) if policy is not None else False

    # --- Step 3: metavertices ------------------------------------------
    bin_coords = np.floor(all_pos / bin_shape_arr).astype(np.int64)
    if use_implicit_sequential:
        key_cols = bin_coords
    else:
        # A graph's vertices merge only within one object AND one connected
        # component.  Binning alone would merge two components that pass
        # within a bin of each other, joining what level 0 keeps apart; with
        # the component in the key, every coarse component is the image of
        # exactly one source component.
        components = component_roots(n_flat, edge_a, edge_b)
        key_cols = np.column_stack([oid_of_v, components, bin_coords])
    key_cols = np.ascontiguousarray(key_cols)
    bin_keys = key_cols.view(
        np.dtype((np.void, key_cols.dtype.itemsize * key_cols.shape[1]))
    ).ravel()
    _, first_of_mv, inverse = np.unique(
        bin_keys, return_index=True, return_inverse=True,
    )
    inverse = inverse.astype(np.int64, copy=False).reshape(-1)
    n_metavertices = int(first_of_mv.shape[0])

    # Centroid per metavertex.  ``np.bincount`` rather than ``np.add.at``:
    # the latter is the unbuffered ufunc.at path and runs an order of
    # magnitude slower for the same scatter-add, which matters once a level
    # carries millions of vertices.
    bin_counts = np.bincount(inverse, minlength=n_metavertices)
    meta_positions = np.empty((n_metavertices, ndim), dtype=np.float32)
    for d in range(ndim):
        meta_positions[:, d] = np.bincount(
            inverse, weights=all_pos[:, d], minlength=n_metavertices,
        ) / bin_counts

    # --- Step 4-5: coarse layout -----------------------------------------
    # Both paths place every coarse vertex in the chunk that contains it, and
    # end with each flat source vertex's coarse ``(chunk, row)``: the row of
    # its OWN object that it was merged into.
    new_manifests: dict[int, list[tuple[ChunkCoords, int]]] = {
        oid: [] for oid in keep_oids
    }
    per_chunk_groups: dict[ChunkCoords, list[np.ndarray]] = {}
    per_chunk_mvs: dict[ChunkCoords, list[np.ndarray]] = {}
    link_records: list[tuple[tuple[ChunkCoords, int], tuple[ChunkCoords, int]]] = []
    edge_attr_groups: npt.NDArray[np.int64] | None = None
    if use_implicit_sequential:
        # Pass 1: per-(oid, coarse chunk) segmentation.  Each surviving
        # object's walk is split wherever the coarse chunk of its vertices
        # changes; within each run, consecutive same-bin vertices collapse to
        # a single metavertex.  The chunk is the metavertex CENTROID's: a bin
        # need not divide the chunk (80-unit bins in 100-unit chunks), and
        # placing by the source point stored a merged vertex in a chunk that
        # does not contain it.
        per_object_runs: dict[int, list[tuple[ChunkCoords, list[int]]]] = {}
        per_object_aux: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for oid, start, n_obj in zip(nonempty_oids, obj_starts, obj_sizes):
            mv_seq = inverse[start:start + n_obj]
            coarse_cc_seq = coarse_chunks_of(
                meta_positions[mv_seq], target_chunk_shape,
            )
            if point_cloud:
                # A point cloud has no edges, so the order of an object's
                # points is arbitrary, and collapsing only *consecutive*
                # repeats kept one row per return to a bin -- the same
                # metavertex written again every time the walk came back to
                # it (35k rows for 2.4k bins measured).  Walking each
                # object's points in (coarse chunk, bin) order instead makes
                # every bin a single visit.
                order = np.lexsort((mv_seq, *coarse_cc_seq.T[::-1]))
                runs = segment_object_by_coarse_chunk(
                    mv_seq[order], coarse_cc_seq[order],
                )
                sorted_run, sorted_pos = positions_in_run(
                    int(n_obj), mv_seq[order], coarse_cc_seq[order],
                )
                # Back to the source order the flat vertices are in.
                run_idx = np.empty_like(sorted_run)
                run_idx[order] = sorted_run
                pos_in_run = np.empty_like(sorted_pos)
                pos_in_run[order] = sorted_pos
            else:
                # Paths keep their order.
                runs = segment_object_by_coarse_chunk(mv_seq, coarse_cc_seq)
                run_idx, pos_in_run = positions_in_run(
                    int(n_obj), mv_seq, coarse_cc_seq,
                )
            per_object_runs[oid] = runs
            per_object_aux[oid] = (run_idx, pos_in_run)

        # Per-chunk assembly: deterministic order — by oid (keep_oids
        # order), then by run index within the object.
        per_chunk_assembly: dict[
            ChunkCoords, list[tuple[int, int, list[int]]]
        ] = {}
        for oid in nonempty_oids:
            for r_idx, (coarse_cc, mv_list) in enumerate(per_object_runs[oid]):
                per_chunk_assembly.setdefault(coarse_cc, []).append(
                    (oid, r_idx, mv_list),
                )

        # Fragment index + chunk-local start per run.  Both quantities are
        # determined once the per-chunk run order is fixed.
        run_to_fragment: dict[
            tuple[int, int], tuple[ChunkCoords, int, int]
        ] = {}
        for coarse_cc, entries in per_chunk_assembly.items():
            cum = 0
            for fragment_idx, (oid, r_idx, mv_list) in enumerate(entries):
                run_to_fragment[(oid, r_idx)] = (coarse_cc, fragment_idx, cum)
                cum += len(mv_list)
            per_chunk_mvs[coarse_cc] = [
                np.asarray(mv_list, dtype=np.int64) for (_, _, mv_list) in entries
            ]
            per_chunk_groups[coarse_cc] = [
                meta_positions[mvs] for mvs in per_chunk_mvs[coarse_cc]
            ]

        # Per-object manifest at the coarse level: one entry per run.
        for oid in nonempty_oids:
            new_manifests[oid] = [
                run_to_fragment[(oid, r_idx)][:2]
                for r_idx in range(len(per_object_runs[oid]))
            ]

        # Each flat source vertex's coarse (chunk, row), and the run it is
        # in, from the (run, position-in-run) the segmentation gave it.
        coarse_chunk_list = sorted(per_chunk_assembly)
        coarse_chunk_index = {cc: i for i, cc in enumerate(coarse_chunk_list)}
        dst_chunk = np.empty(n_flat, dtype=np.int64)
        dst_row = np.empty(n_flat, dtype=np.int64)
        dst_run = np.empty(n_flat, dtype=np.int64)
        run_base = 0
        for oid, start, n_obj in zip(nonempty_oids, obj_starts, obj_sizes):
            run_idx, pos_in_run = per_object_aux[oid]
            placed = [
                run_to_fragment[(oid, r_idx)]
                for r_idx in range(len(per_object_runs[oid]))
            ]
            run_chunk = np.array(
                [coarse_chunk_index[cc] for cc, _, _ in placed], dtype=np.int64,
            )
            run_start = np.array([s for _, _, s in placed], dtype=np.int64)
            dst_chunk[start:start + n_obj] = run_chunk[run_idx]
            dst_row[start:start + n_obj] = run_start[run_idx] + pos_in_run
            dst_run[start:start + n_obj] = run_base + run_idx
            run_base += len(placed)
        coarse_chunk_arr = np.asarray(coarse_chunk_list, dtype=np.int64).reshape(
            len(coarse_chunk_list), ndim,
        )
        dst_chunk_coords = coarse_chunk_arr[dst_chunk]

        if not point_cloud:
            # Pass 2: join each object's consecutive runs.  Inside a run an
            # edge is the implicit one between consecutive rows; where the
            # walk moves to its next run (always in another chunk) the step
            # needs a directed (predecessor -> successor) record.  Taken from
            # the walk itself rather than remapped from the source's records:
            # a centroid's chunk can split a source fragment, which no source
            # record marks, and a line store's level-0 records are not
            # reliable (core writes a fragment index where the vertex row
            # belongs).
            step = np.flatnonzero(
                (oid_of_v[1:] == oid_of_v[:-1]) & (dst_run[1:] != dst_run[:-1])
            )
            link_records = [
                ((coarse_chunk_list[ca], int(ra)), (coarse_chunk_list[cb], int(rb)))
                for ca, ra, cb, rb in zip(
                    dst_chunk[step].tolist(), dst_row[step].tolist(),
                    dst_chunk[step + 1].tolist(), dst_row[step + 1].tolist(),
                )
            ]
    else:
        # One fragment per object per coarse chunk, as level 0 is written:
        # metavertices sorted by (chunk, owner, id) and numbered in that order.
        mv_owner = oid_of_v[first_of_mv]
        mv_cc = coarse_chunks_of(meta_positions, target_chunk_shape)
        mv_order = np.lexsort((
            np.arange(n_metavertices), mv_owner, *mv_cc.T[::-1],
        ))
        cc_sorted = mv_cc[mv_order]
        owner_sorted = mv_owner[mv_order]
        cut = np.concatenate(([True], (
            np.any(cc_sorted[1:] != cc_sorted[:-1], axis=1)
            | (owner_sorted[1:] != owner_sorted[:-1])
        )))
        group_starts = np.flatnonzero(cut)
        group_ends = np.append(group_starts[1:], n_metavertices)
        mv_row = np.empty(n_metavertices, dtype=np.int64)
        current: ChunkCoords | None = None
        row = 0
        for start, end in zip(group_starts.tolist(), group_ends.tolist()):
            cc = tuple(int(c) for c in cc_sorted[start])
            if cc != current:
                current, row = cc, 0
                per_chunk_groups[cc] = []
                per_chunk_mvs[cc] = []
            mvs = mv_order[start:end]
            mv_row[mvs] = row + np.arange(end - start, dtype=np.int64)
            row += end - start
            new_manifests[int(owner_sorted[start])].append(
                (cc, len(per_chunk_mvs[cc])),
            )
            per_chunk_mvs[cc].append(mvs)
            per_chunk_groups[cc].append(meta_positions[mvs])
        dst_chunk_coords = mv_cc[inverse]
        dst_row = mv_row[inverse]

        # Edges: exactly the images of the source's edges under the vertex ->
        # metavertex map, without self-loops (an edge inside one metavertex)
        # and without duplicates (parallel edges merged into one).
        pairs, mapped, edge_attr_groups = contract_edges(
            np.column_stack([edge_a, edge_b]), inverse,
            directed=edges_directed,
        )
        if mapped.size:
            # The source record behind each surviving source edge, for its
            # attributes.
            edge_rows = edge_rows[mapped]
            link_records = [
                (
                    (tuple(int(c) for c in mv_cc[p]), int(mv_row[p])),
                    (tuple(int(c) for c in mv_cc[q]), int(mv_row[q])),
                )
                for p, q in pairs.tolist()
            ]

    # --- Step 6: write per-chunk fragments --------------------------
    arrays_present = [VERTICES, "object_index"] if src_has_objects else [VERTICES]
    if not use_implicit_sequential:
        arrays_present.insert(1, "links")
    # The rows actually written, which is what a reader loads: on the
    # implicit-sequential path a bin two objects share is stored once for each.
    stored_vertex_count = sum(
        len(group) for groups in per_chunk_groups.values() for group in groups
    )
    level_meta_initial = LevelMetadata(
        level=target_level,
        vertex_count=int(stored_vertex_count),
        arrays_present=arrays_present,
        bin_shape=target_bin_shape,
        # Fold-change relative to LEVEL 0, not to the source level: this is
        # what becomes the NGFF ``scale`` transform. With per-level coarsen
        # factors the two differ — [2, 2] is ratio 2 then 4 — so it has to be
        # derived from the bin shapes rather than echoing coarsen_factor.
        bin_ratio=tuple(
            max(1, int(round(float(t) / float(r))))
            for t, r in zip(target_bin_shape, root_meta.effective_bin_shape)
        ),
        chunk_shape=target_chunk_shape_override,
        object_sparsity=(1.0 / sparsity_factor),
        coarsening_method=COARSEN_PER_OBJECT,
        parent_level=source_level,
        preserves_object_ids=src_has_objects,
        inherited_num_objects=n_src_objects if src_has_objects else 0,
        # Every coarse fragment belongs to one object on both paths.
        shared_fragments=False,
    )
    level_group = create_resolution_level(root, target_level, level_meta_initial)
    # A chunk array's codec pipeline is fixed at creation; the writes below
    # then encode to match.  Close the block before them so batched_writes'
    # deferred metas are flushed first.
    _codec_ctx = (
        level_group.batched_writes(compressor=compressor)
        if compressor else nullcontext()
    )
    with _codec_ctx:
        create_vertices_array(level_group, dtype="float32")
        if src_has_objects:
            create_object_index_array(level_group)

    # Batched: each per-chunk write is two cells (vertices + vertex_fragments)
    # and each cell was a separate sync round-trip plus a read-modify-write of
    # the array-wide ``nonempty_chunks`` attribute -- which grows with the
    # level, so the serial form cost O(chunks^2) bytes of metadata rewriting on
    # top of the round-trips.  The flush writes each array's cells in one
    # concurrent ``set_coordinate_selection`` and stamps the manifest once.
    # Codecs are fixed when an array is created (the block above), so opening
    # this one with compressor=None does not alter the arrays' encoding.
    #
    # Flushed in slices rather than as one block: a batch holds every queued
    # cell's encoded bytes in memory until its flush, so a level wide enough to
    # matter would otherwise buffer a second copy of itself.  A few thousand
    # cells per flush keeps the round-trip saving and bounds the buffer.
    _write_items = sorted(per_chunk_groups.items())
    for _i in range(0, len(_write_items), _WRITE_BATCH_CHUNKS):
        with level_group.batched_writes(compressor=compressor):
            for cc, groups in _write_items[_i:_i + _WRITE_BATCH_CHUNKS]:
                write_chunk_vertices(level_group, cc, groups, dtype=np.float32)

    # --- Step 6b: per-vertex attributes, one value per metavertex --------
    _carry_vertex_attributes(
        src_group, level_group, flat_refs, inverse, n_metavertices,
        per_chunk_mvs, compressor=compressor,
    )

    # --- Step 9: emit object_index (gap-fill for dropped OIDs) ----------
    survivors = (
        list(keep_oids) if collapsed_oids
        else surviving_oids_from(keep_oids, sparsity_factor)
    )
    if src_has_objects:
        write_object_index(
            level_group, new_manifests, sid_ndim=ndim,
            total_objects=n_src_objects,
        )
        # Carry the group taxonomy forward. Object ids are preserved, so a
        # group's membership is meaningful here unchanged; without this the
        # coarse level has an object index but no way to say what any object
        # IS, and a reader has to reach back to level 0 for the taxonomy.
        propagate_groupings(src_group, level_group, surviving_oids=survivors)

    # --- Step 9b: links/0 -----------------------------------------------
    # ``directed`` is a family-wide, un-flippable policy per (level, delta).
    # Path records carry endpoint order as data (predecessor -> successor);
    # graph records keep the source family's policy.  A point cloud has no
    # edges and gets no family at all: an empty one reads as a links array
    # on a geometry that has none.
    if not point_cloud:
        create_links_family(
            level_group, delta=0, link_width=2, sid_ndim=ndim,
            directed=edges_directed,
        )
        if link_records:
            partition = write_links(
                level_group, link_records, sid_ndim=ndim, delta=0,
                directed=edges_directed,
            )
            if edge_attr_groups is not None:
                _carry_link_attributes(
                    src_group, level_group, edge_rows, edge_attr_groups,
                    n_links=len(link_records), partition=partition,
                )
        else:
            # ``write_links`` stamps the family counts as a side effect, so a
            # family with nothing to write would otherwise carry policy but no
            # ``num_links`` — a shape every family is supposed to be free of.
            # Finalize explicitly to stamp the zero.
            finalize_links(level_group, delta=0)

    # --- Step 10: per-object attributes with present_mask ---------------
    if src_has_objects:
        carry_object_columns(src_group, level_group, keep_oids, n_src_objects)

    # --- Step 12: stamp root capability tokens --------------------------
    if src_has_objects:
        _stamp_root_capability(root, CAP_PRESERVED_OBJECT_IDS)

    # --- Step 13: emit inline ±1 cross-level link arrays ----------------
    if cross_level_storage != XLEVEL_NONE and n_metavertices > 0:
        src_chunk_arr = np.asarray(src_chunk_list, dtype=np.int64).reshape(
            len(src_chunk_list), ndim,
        )
        _emit_inline_cross_level_links(
            root,
            src_group=src_group,
            level_group=level_group,
            source_level=source_level,
            ndim=ndim,
            storage=cross_level_storage,
            fine_chunks=src_chunk_arr[flat_src_chunk],
            fine_rows=flat_src_row,
            coarse_chunks=dst_chunk_coords,
            coarse_rows=dst_row,
        )

    # This coarsener writes serially (no cross-process manifest race), but
    # re-derive the per-array ``nonempty_chunks`` manifests from disk anyway for
    # uniformity with the parallel coarseners and idempotence.
    rebuild_presence(level_group)
    refresh_arrays_present(level_group)
    if drop_collapsed:
        # Last: the record is merged into attrs re-read from disk, after every
        # write through this coarsener's own level handle.
        _write_coarsening_record(
            root, target_level, collapsed_objects=len(collapsed_oids),
        )

    return {
        "vertex_count": int(stored_vertex_count),
        "object_count": len(keep_oids),
        "objects_kept": len(keep_oids),
        "objects_collapsed": len(collapsed_oids),
        "source_objects": n_src_objects,
        "method": COARSEN_PER_OBJECT,
        "preserves_object_ids": True,
        "shared_fragments": False,
    }


def _chunk_ids(
    chunks: npt.NDArray[np.int64], chunk_index: dict[ChunkCoords, int],
) -> npt.NDArray[np.int64]:
    """``chunk_index`` entry of each row of ``chunks`` (``(K, D)``), -1 if absent.

    One dict probe per distinct chunk rather than per row.
    """
    if chunks.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    uniq, inv = np.unique(chunks, axis=0, return_inverse=True)
    return np.array(
        [chunk_index.get(tuple(u), -1) for u in uniq.tolist()], dtype=np.int64,
    )[inv.reshape(-1)]


def _lookup_rows(
    chunks: npt.NDArray[np.int64],
    rows: npt.NDArray[np.int64],
    chunk_index: dict[ChunkCoords, int],
    chunk_base: npt.NDArray[np.int64],
    rows_per_chunk: npt.NDArray[np.int64],
    table: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """``table`` entry of each ``(chunk, row)``, or -1 for an unknown one.

    ``chunk_index`` numbers the chunks, ``chunk_base`` is each chunk's first
    entry in ``table`` and ``rows_per_chunk`` its row count, so a chunk-local
    row is one gather once the chunk is known.
    """
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    idx = _chunk_ids(
        np.asarray(chunks, dtype=np.int64).reshape(rows.shape[0], -1),
        chunk_index,
    )
    ok = idx >= 0
    ok[ok] = (rows[ok] >= 0) & (rows[ok] < rows_per_chunk[idx[ok]])
    out = np.full(rows.shape[0], -1, dtype=np.int64)
    out[ok] = table[chunk_base[idx[ok]] + rows[ok]]
    return out


def _carry_link_attributes(
    src_group: Any,
    level_group: Any,
    source_records: npt.NDArray[np.int64],
    groups: npt.NDArray[np.int64],
    *,
    n_links: int,
    partition: Any,
) -> list[str]:
    """Give every coarse edge a value for each of the source's edge attributes.

    ``source_records[k]`` is the source record (in ``read_link_arrays``
    order) behind the ``k``-th surviving source edge and ``groups[k]`` the
    coarse edge it merged into.  As for vertex attributes, a float column
    takes the mean over the merged edges and any other column the value of
    the first.  A column whose row count does not match the source's records
    is left out rather than misaligned.
    """
    from zarr_vectors.building import (
        LINK_ATTRIBUTES,
        read_link_attributes,
        write_link_attributes,
    )

    if LINK_ATTRIBUTES not in src_group:
        return []
    n_source = int(source_records.max()) + 1 if source_records.size else 0
    carried: list[str] = []
    for name in list(src_group[LINK_ATTRIBUTES]):
        try:
            column = np.asarray(read_link_attributes(src_group, name, delta=0))
        except (ArrayError, StoreError, KeyError):
            continue
        if column.shape[0] < n_source:
            continue
        values = column[source_records]
        if values.dtype.kind == "f":
            counts = np.bincount(groups, minlength=n_links)
            flat = values.reshape(len(values), -1).astype(np.float64)
            agg = np.stack([
                np.bincount(groups, weights=flat[:, c], minlength=n_links) / counts
                for c in range(flat.shape[1])
            ], axis=1).astype(values.dtype).reshape((n_links, *values.shape[1:]))
        else:
            _, first = np.unique(groups, return_index=True)
            agg = values[first]
        write_link_attributes(
            level_group, name, agg, num_links=n_links, delta=0,
            partition=partition,
        )
        carried.append(name)
    return carried


def _emit_inline_cross_level_links(
    root,
    *,
    src_group,
    level_group,
    source_level: int,
    ndim: int,
    storage: str,
    fine_chunks: npt.NDArray[np.int64],
    fine_rows: npt.NDArray[np.int64],
    coarse_chunks: npt.NDArray[np.int64],
    coarse_rows: npt.NDArray[np.int64],
) -> None:
    """Emit the ``±1`` links family for one coarsen step.

    Record ``k`` links fine vertex ``(fine_chunks[k], fine_rows[k])`` to the
    coarse row ``(coarse_chunks[k], coarse_rows[k])`` it was merged into.
    The coarsener hands over the row of the vertex's OWN object: where two
    objects share a bin each stores its own copy of the merged vertex, and
    pointing at whichever copy came first linked most vertices to another
    object.  Vertices of dropped objects have no record.  The pairs are
    turned into the flat ``parent`` array :func:`_write_cross_level_edges`
    and the ``±N`` composition take.
    """
    fine_assn, n_fine = _reconstruct_chunk_assignments(src_group, ndim)
    coarse_assn, n_coarse = _reconstruct_chunk_assignments(level_group, ndim)
    fine_idx = _assigned_rows(fine_assn, fine_chunks, fine_rows)
    coarse_idx = _assigned_rows(coarse_assn, coarse_chunks, coarse_rows)
    ok = (fine_idx >= 0) & (coarse_idx >= 0)
    parent = np.full(n_fine, -1, dtype=np.int64)
    parent[fine_idx[ok]] = coarse_idx[ok]
    _write_cross_level_edges(
        root,
        fine_level=source_level,
        delta=1,
        fine_chunk_assignments=fine_assn,
        coarse_chunk_assignments=coarse_assn,
        n_fine=n_fine,
        n_coarse=n_coarse,
        parent=parent,
        sid_ndim=ndim,
        storage=storage,
    )


def _assigned_rows(
    assignments: dict[ChunkCoords, npt.NDArray[np.int64]],
    chunks: npt.NDArray[np.int64],
    rows: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """Flat index of each ``(chunk, row)`` under ``assignments``, -1 if none.

    ``assignments`` is :func:`_reconstruct_chunk_assignments`' output: every
    chunk's rows are one ``arange``, so a row's flat index is its chunk's
    first index plus the row.
    """
    keys = list(assignments)
    starts = np.array([int(assignments[cc][0]) for cc in keys], dtype=np.int64)
    counts = np.array([len(assignments[cc]) for cc in keys], dtype=np.int64)
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    idx = _chunk_ids(
        np.asarray(chunks, dtype=np.int64).reshape(rows.shape[0], -1),
        {cc: i for i, cc in enumerate(keys)},
    )
    ok = idx >= 0
    ok[ok] = (rows[ok] >= 0) & (rows[ok] < counts[idx[ok]])
    out = np.full(rows.shape[0], -1, dtype=np.int64)
    out[ok] = starts[idx[ok]] + rows[ok]
    return out


def _write_empty_preserve_level(
    root,
    source_level: int,
    target_level: int,
    *,
    base_bin: tuple[float, ...],
    root_bin: tuple[float, ...],
    coarsen_factor: float,
    sparsity_factor: float,
    inherited_num_objects: int,
) -> None:
    """Write an empty ID-preserving level when no surviving object has vertices.

    ``base_bin`` is the SOURCE level's bin (what ``coarsen_factor`` multiplies);
    ``root_bin`` is level 0's, needed for the level-0-relative ``bin_ratio``.
    """
    ndim = len(base_bin)
    target_bin_shape = tuple(float(b) * float(coarsen_factor) for b in base_bin)
    level_meta = LevelMetadata(
        level=target_level,
        vertex_count=0,
        arrays_present=[VERTICES, "object_index"],
        bin_shape=target_bin_shape,
        # Fold-change relative to LEVEL 0, not to the source level: this is
        # what becomes the NGFF ``scale`` transform. With per-level coarsen
        # factors the two differ — [2, 2] is ratio 2 then 4 — so it has to be
        # derived from the bin shapes rather than echoing coarsen_factor.
        bin_ratio=tuple(
            max(1, int(round(float(t) / float(r))))
            for t, r in zip(target_bin_shape, root_bin)
        ),
        object_sparsity=(1.0 / sparsity_factor),
        coarsening_method=COARSEN_PER_OBJECT,
        parent_level=source_level,
        preserves_object_ids=True,
        inherited_num_objects=inherited_num_objects,
        shared_fragments=False,
    )
    level_group = create_resolution_level(root, target_level, level_meta)
    create_vertices_array(level_group, dtype="float32")
    create_object_index_array(level_group)
    # Empty object_index with the inherited size — all manifests are [].
    write_object_index(
        level_group, {}, sid_ndim=ndim,
        total_objects=inherited_num_objects,
    )
    _stamp_root_capability(root, CAP_PRESERVED_OBJECT_IDS)


def _stamp_root_capability(root_group, cap: str) -> None:
    """Add ``cap`` to root metadata's ``format_capabilities`` (idempotent).

    A thin alias now: core owns the read-modify-write of its own attrs
    block, which is what this hand-rolled while there was no primitive.
    """
    update_root_metadata(root_group, add_capabilities=[cap])


def _stamp_root_cross_level(root_group) -> tuple[int, str]:
    """Record on root metadata the cross-level links the store holds.

    Not the ones asked for: only the per-object coarsener writes any, so a
    streamline, skeleton or mesh pyramid built with the default
    ``cross_level_storage="explicit"`` has none, and a root claiming
    ``explicit`` sends a reader looking for arrays that do not exist.  The
    depth is the largest ``|delta|`` present, the storage ``explicit`` when
    a ``-N`` family exists, ``implicit`` when only ``+N`` ones do, and
    ``none`` (depth 0, no multiscale-links capability) when there are none.

    Returns:
        ``(depth, storage)`` as stamped.
    """
    deltas: set[int] = set()
    for level in list_resolution_levels(root_group):
        level_group = get_resolution_level(root_group, level)
        deltas.update(
            int(d) for d in list_link_deltas(level_group)
            if int(d) != 0 and list_link_offsets(level_group, int(d))
        )
    if not deltas:
        caps = [
            cap for cap in read_root_metadata(root_group).format_capabilities or []
            if cap != CAP_MULTISCALE_LINKS
        ]
        update_root_metadata(
            root_group, cross_level_depth=0, cross_level_storage=XLEVEL_NONE,
            format_capabilities=caps,
        )
        return 0, XLEVEL_NONE
    depth = max(abs(d) for d in deltas)
    storage = XLEVEL_EXPLICIT if any(d < 0 for d in deltas) else XLEVEL_IMPLICIT
    update_root_metadata(
        root_group, cross_level_depth=depth, cross_level_storage=storage,
    )
    return depth, storage


def _clear_cross_level_families(root_group, *, from_level: int) -> None:
    """Remove the cross-level families a rebuild from ``from_level`` makes stale.

    Rebuilding the levels above ``from_level`` invalidates every family that
    reaches one of them: every family at a level above it, and every ``+N``
    family below or at it that reaches past it (level 0's ``+2`` when the
    rebuild starts at level 1).  The families are rewritten only where new
    records land, so without this an old pyramid's records (or all of them,
    when the new one is built with ``cross_level_storage="none"``) survived
    and pointed at vertices that had moved.
    """
    for level in list_resolution_levels(root_group):
        level_group = get_resolution_level(root_group, level)
        for delta in list_link_deltas(level_group):
            if delta != 0 and (level > from_level or level + delta > from_level):
                level_group.delete_subtree(links_group_path(delta))


def _reconstruct_chunk_assignments(
    level_group, ndim: int,
) -> tuple[dict[ChunkCoords, npt.NDArray[np.int64]], int]:
    """Rebuild ``{chunk_coords: vertex_indices}`` for a level's vertex chunks.

    The "vertex index" assigned to each vertex is the position it would
    occupy in a flat enumeration that walks chunks in
    ``list_chunk_keys`` order and concatenates each chunk's vertex
    groups in order.  This matches the convention used by
    ``build_vertex_chunk_mapping`` for in-memory edge partitioning.

    Only row *counts* are needed, and a chunk's fragment index already
    carries them — the rows ``read_chunk_vertices`` would return are exactly
    the rows its fragments reference — so the vertex buffers themselves are
    never fetched or decoded.  ``ndim`` is kept for signature compatibility.

    Returns the assignments dict and the total vertex count.
    """
    del ndim
    chunk_keys = list_chunk_keys(level_group, VERTICES)
    assignments: dict[ChunkCoords, npt.NDArray[np.int64]] = {}
    cursor = 0
    key_strs = [chunk_key_str(cc) for cc in chunk_keys]
    # One prefetch of the (small) fragment-index cells for the whole level.
    with level_group.batched_reads([(VERTEX_FRAGMENTS, key_strs)]):
        for cc in chunk_keys:
            try:
                n = _fragment_row_count(read_vertex_fragment_index(level_group, cc))
            except (ArrayError, StoreError):
                continue
            if n == 0:
                continue
            assignments[cc] = np.arange(cursor, cursor + n, dtype=np.int64)
            cursor += n
    return assignments, cursor


def _fragment_row_count(fi: Any) -> int:
    """Total rows a chunk's fragments reference (``sum`` of fragment sizes).

    The same number ``sum(len(f) for f in read_chunk_vertices(...))`` gives.
    The common layout — range fragments tiling ``[0, N)`` — is decided by
    two whole-array checks; anything else is summed fragment by fragment.
    """
    n_frag = int(fi.num_fragments)
    if n_frag == 0:
        return 0
    if fi.num_explicit_fragments == 0:
        extent = int(fi.vertex_extent)
        if fi.tiles(extent):
            return extent
    total = 0
    for f in range(n_frag):
        if fi.is_range(f):
            total += int(fi.range(f)[1])
        else:
            total += int(fi.indices_view(f).shape[0])
    return total


def _decode_parent_from_plus_one(
    fine_lg,
    *,
    fine_assn: dict[ChunkCoords, npt.NDArray[np.int64]],
    coarse_assn: dict[ChunkCoords, npt.NDArray[np.int64]],
    n_fine: int,
) -> npt.NDArray[np.int64] | None:
    """Decode a fine→coarse ``parent`` array from already-written ``+1`` arrays.

    Reads every record under ``links/<+1>/`` at the fine level and
    converts each ``(chunk, local_idx)`` pair to global flat indices via
    the supplied chunk-assignment dicts.  Returns ``None`` when the
    family holds nothing.
    """
    parent = np.full(n_fine, -1, dtype=np.int64)

    # One family read covers both the chunk-aligned records (endpoints in
    # the same chunk on both grids — all-zero offsets) and the ones that
    # span chunks.  A cross-level record's endpoints are distinguished by
    # level, so endpoint 0 is always the fine source and endpoint 1 the
    # coarse target, in input order.
    #
    # Every cell of the family is prefetched in one gather first; read_links
    # otherwise issues one synchronous read per cell.
    family = links_group_path(1)
    plan = [
        (f"{family}/{seg}", fine_lg.list_chunks(f"{family}/{seg}"))
        for seg in list_link_offsets(fine_lg, 1)
    ]
    try:
        with fine_lg.batched_reads([p for p in plan if p[1]]):
            records = read_links(fine_lg, delta=1)
    except (ArrayError, KeyError):
        records = []
    if not records:
        return None

    # Chunk-local → global is ``start + local`` since every assignment is an
    # arange; later records overwrite earlier ones, as the per-record loop did.
    fine_start = {cc: int(rows[0]) for cc, rows in fine_assn.items()}
    coarse_start = {cc: int(rows[0]) for cc, rows in coarse_assn.items()}
    n = len(records)
    src = np.fromiter(
        (fine_start[cc] + vi for (cc, vi), _ in records), dtype=np.int64, count=n,
    )
    trg = np.fromiter(
        (coarse_start[cc] + vi for _, (cc, vi) in records), dtype=np.int64, count=n,
    )
    parent[src] = trg
    return parent


def _finalize_cross_level_for_store(
    store_path: str | Path,
    *,
    cross_level_depth: int,
    cross_level_storage: str,
) -> tuple[int, str]:
    """Emit ``±N`` (N ≥ 2) link arrays, then record what the store holds.

    Adjacent ``±1`` arrays are emitted inline during coarsening (see
    :func:`_emit_inline_cross_level_links`).  This finalize pass walks
    every adjacent (fine, coarse) level pair, decodes the on-disk
    ``+1`` parent map back into a flat fine→coarse array, then composes
    step-by-step to produce ``+N``/``-N`` link arrays for N ≥ 2 up to
    ``cross_level_depth``.

    ``cross_level_depth=-1`` means "walk all available level pairs".

    Returns:
        The ``(depth, storage)`` stamped on root metadata, which describe
        the arrays written; see :func:`_stamp_root_cross_level`.
    """
    root = open_store(str(store_path), mode="r+")
    if cross_level_storage != XLEVEL_NONE and cross_level_depth != 0:
        _compose_cross_level_links(
            root,
            cross_level_depth=cross_level_depth,
            cross_level_storage=cross_level_storage,
        )
    return _stamp_root_cross_level(root)


def _compose_cross_level_links(
    root,
    *,
    cross_level_depth: int,
    cross_level_storage: str,
) -> None:
    """Compose the inline ``+1`` parent maps into ``±N`` arrays, N ≥ 2."""
    meta = read_root_metadata(root)
    ndim = meta.sid_ndim
    levels = sorted(list_resolution_levels(root))
    if len(levels) < 2:
        return

    # Decide whether there is anything to compose BEFORE reading anything.
    #
    # ``±1`` is emitted inline during coarsening, so this pass exists only
    # for ``±N`` with N >= 2.  At the default depth of 1 there is nothing to
    # do -- but the work below ran first and cost a full re-read of every
    # level plus one int64 per vertex per level, which on a whole-brain
    # tractogram is the largest allocation in the whole build and the most
    # likely thing to have been killed as "the pyramid OOM".
    max_delta = (
        max(levels) - min(levels)
        if cross_level_depth == -1
        else int(cross_level_depth)
    )
    if max_delta < 2:
        return

    # No capability stamp here.  The token marks the presence of delta != 0
    # arrays, and this function does not know yet whether any will be
    # written: ``max_delta < 2`` returns below without emitting, and a
    # coarsener that never emits inline ±1 links (the polyline strategy does
    # not) leaves the store with none at all.  _write_cross_level_edges is
    # the single choke point that actually writes them and stamps there,
    # once it knows.  Stamping optimistically here claimed cross-LEVEL links
    # on stores that had none.

    # Build per-level chunk_assignments + total counts once.  This is the
    # expensive part the early return above protects.
    per_level: dict[int, tuple[dict[ChunkCoords, npt.NDArray[np.int64]], int]] = {}
    for lvl in levels:
        lg = get_resolution_level(root, lvl)
        per_level[lvl] = _reconstruct_chunk_assignments(lg, ndim)

    # Cache each adjacent (fine_level, fine_level+1) parent array.
    adjacent_parent: dict[int, npt.NDArray[np.int64]] = {}
    for fine_level in levels[:-1]:
        coarse_level = fine_level + 1
        if coarse_level not in per_level:
            continue
        fine_assn, n_fine = per_level[fine_level]
        coarse_assn, _ = per_level[coarse_level]
        if n_fine == 0:
            continue
        fine_lg = get_resolution_level(root, fine_level)
        parent = _decode_parent_from_plus_one(
            fine_lg,
            fine_assn=fine_assn,
            coarse_assn=coarse_assn,
            n_fine=n_fine,
        )
        if parent is not None:
            adjacent_parent[fine_level] = parent

    # Compose deeper-delta parents and emit.
    for fine_level in levels[:-1]:
        if fine_level not in adjacent_parent:
            continue
        fine_assn, n_fine = per_level[fine_level]
        parent = adjacent_parent[fine_level].copy()
        for step in range(2, max_delta + 1):
            coarse_level = fine_level + step
            if coarse_level not in per_level:
                break
            inter_level = coarse_level - 1
            if inter_level not in adjacent_parent:
                break
            inter_parent = adjacent_parent[inter_level]
            coarse_assn, n_coarse = per_level[coarse_level]
            if n_coarse == 0:
                break

            composed = np.full(n_fine, -1, dtype=np.int64)
            valid = parent >= 0
            composed[valid] = inter_parent[parent[valid]]
            parent = composed
            if not np.any(parent >= 0):
                break

            _write_cross_level_edges(
                root,
                fine_level=fine_level,
                delta=step,
                fine_chunk_assignments=fine_assn,
                coarse_chunk_assignments=coarse_assn,
                n_fine=n_fine,
                n_coarse=n_coarse,
                parent=parent,
                sid_ndim=ndim,
                storage=cross_level_storage,
            )


def _write_cross_level_edges(
    root_group,
    *,
    fine_level: int,
    delta: int,
    fine_chunk_assignments: dict[ChunkCoords, npt.NDArray[np.int64]],
    coarse_chunk_assignments: dict[ChunkCoords, npt.NDArray[np.int64]],
    n_fine: int,
    n_coarse: int,
    parent: npt.NDArray[np.int64],
    sid_ndim: int,
    storage: str,
) -> None:
    """Materialize ``delta``-step cross-level edges between two adjacent levels.

    ``parent[i]`` is the metanode index in the coarser level that fine
    vertex ``i`` belongs to.  The cross-level edges are trivially
    ``(i, parent[i])`` for each fine vertex.

    Writes the ``+delta`` arrays under the fine level.  When
    ``storage='explicit'`` also writes the matching ``-delta`` arrays
    under the coarse level by swapping endpoint roles.
    """
    if storage == XLEVEL_NONE or delta == 0:
        return
    coarse_level = fine_level + delta

    # Drop orphaned fine vertices (parent < 0) before building edges.
    valid_mask = parent >= 0
    if not np.any(valid_mask):
        return
    fine_global = np.flatnonzero(valid_mask).astype(np.int64)
    parent_valid = parent[valid_mask].astype(np.int64)

    # Per-edge chunk coords and chunk-local rows for both endpoints.
    fine_cc, fine_local = _endpoint_chunk_rows(
        fine_chunk_assignments, n_fine, fine_global,
    )
    coarse_cc, coarse_local = _endpoint_chunk_rows(
        coarse_chunk_assignments, n_coarse, parent_valid,
    )

    _write_cross_level_family(
        root_group, fine_level, delta=delta, sid_ndim=sid_ndim,
        src_cc=fine_cc, src_vi=fine_local, trg_cc=coarse_cc, trg_vi=coarse_local,
        # delta != 0 by construction here, which is what the token marks.
        stamp_capability=True,
    )

    if storage == XLEVEL_EXPLICIT:
        # Mirror at the coarse level under -delta: swap endpoint roles.
        # Chunk alignment is re-evaluated from the coarse side, and records
        # keep the fine-vertex order the forward family was written in.
        _write_cross_level_family(
            root_group, coarse_level, delta=-delta, sid_ndim=sid_ndim,
            src_cc=coarse_cc, src_vi=coarse_local,
            trg_cc=fine_cc, trg_vi=fine_local,
        )


def _endpoint_chunk_rows(
    chunk_assignments: dict[ChunkCoords, npt.NDArray[np.int64]],
    n_vertices: int,
    global_idx: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """``(M, sid_ndim)`` chunk coords and ``(M,)`` chunk-local rows of vertices.

    The array form of the per-edge lookup ``partition_cross_level_edges``
    made through ``build_vertex_chunk_mapping``'s tables.
    """
    chunk_list = sorted(chunk_assignments.keys())
    vchunks, vlocal, chunk_list = build_vertex_chunk_mapping(
        chunk_assignments, n_vertices, chunk_list,
    )
    chunk_arr = np.asarray(chunk_list, dtype=np.int64).reshape(len(chunk_list), -1)
    return chunk_arr[vchunks[global_idx]], vlocal[global_idx]


def _group_rows_by_key(
    key: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.intp], list[tuple[int, int]]]:
    """Stable grouping of rows by an ``(M, K)`` integer key.

    Returns the permutation that sorts ``key`` (stable, so rows keep their
    input order within a group) and the ``[start, end)`` span of each group
    in that permutation.
    """
    order = np.lexsort(key.T[::-1])
    key_sorted = key[order]
    starts = np.flatnonzero(np.concatenate((
        [True], np.any(key_sorted[1:] != key_sorted[:-1], axis=1),
    )))
    ends = np.append(starts[1:], key.shape[0])
    return order, list(zip(starts.tolist(), ends.tolist()))


#: Threads writing one cross-level family's offsets arrays.
_XLEVEL_ARRAY_WORKERS = 8


def _write_cross_level_arrays(
    root_group,
    level: int,
    cells_by_offsets: dict[Any, list[tuple[ChunkCoords, npt.NDArray[np.int64]]]],
    *,
    delta: int,
    sid_ndim: int,
) -> None:
    """Allocate each offsets array of one family and write its cells.

    A family has one array per distinct offset -- dozens to hundreds -- and
    each costs a chain of synchronous store round-trips: the allocation
    (existence probes, the family policy read, the array's ``zarr.json``),
    the first node lookup of the write block, and the block's flush.  The
    arrays are independent objects under a family group that already exists,
    and none of these steps writes the group (``create_links_array`` stamps
    it only when absent), so the arrays are handled concurrently.  Each
    worker thread uses its own level handle, since a ``batched_writes``
    block is state on the handle; within an array the cells still go out as
    one flush per ``_WRITE_BATCH_CHUNKS``.
    """
    if not cells_by_offsets:
        return
    local = threading.local()

    def _one(offsets: Any) -> None:
        lg = getattr(local, "lg", None)
        if lg is None:
            lg = local.lg = get_resolution_level(root_group, level)
        cells = cells_by_offsets[offsets]
        for i in range(0, len(cells), _WRITE_BATCH_CHUNKS):
            with lg.batched_writes():
                if i == 0:
                    # Allocated inside the block so the handle it returns
                    # seeds the block's node cache; the block's default
                    # codec selection resolves to the same empty compressor
                    # list an unbatched allocation uses.
                    create_links_array(
                        lg, link_width=2, delta=delta, sid_ndim=sid_ndim,
                        offsets=offsets,
                    )
                for cc, cell_rows in cells[i:i + _WRITE_BATCH_CHUNKS]:
                    write_chunk_links(
                        lg, cc, [cell_rows], delta=delta, offsets=offsets,
                    )

    items = list(cells_by_offsets)
    if len(items) == 1:
        _one(items[0])
        return
    with ThreadPoolExecutor(
        max_workers=min(_XLEVEL_ARRAY_WORKERS, len(items)),
    ) as pool:
        for future in [pool.submit(_one, off) for off in items]:
            # Re-raise the first failure, after every worker settles.
            future.result()


def _write_cross_level_family(
    root_group,
    level: int,
    *,
    delta: int,
    sid_ndim: int,
    src_cc: npt.NDArray[np.int64],
    src_vi: npt.NDArray[np.int64],
    trg_cc: npt.NDArray[np.int64],
    trg_vi: npt.NDArray[np.int64],
    stamp_capability: bool = False,
) -> None:
    """Write the ``links/<delta>/`` cross-level family owned by ``level``:
    record ``k`` is ``((src_cc[k], src_vi[k]), (trg_cc[k], trg_vi[k]))``.

    Writes what core's ``write_links`` would (a scoped replace, then the
    family counts) without its per-record Python objects or its per-cell
    synchronous writes:

    * **Placement** is the arithmetic every reader decodes with
      (``cell_endpoint_chunks``): a record files under its source chunk, in
      the array named by ``trg - floor(src * scale_src / scale_trg)``, the
      target chunk re-anchored onto the source's grid.  Every record is
      placed this way, including one whose two chunk coords happen to be
      equal: on grids of different sizes equal coords are different places,
      and filing such a record under all-zero offsets made it decode to
      ``floor(src * scale_src / scale_trg)`` (``2 * c`` in the ``-1`` family
      of a chunk-scale-2 pyramid).  Rows keep input order within a cell.
    * **The replace** deletes exactly the offsets arrays the records target.
    * **Cells** are written in batched blocks, one array per worker thread:
      a flush per array instead of a write plus a ``nonempty_chunks``
      read-modify-write per cell.
    * **Counts** are known exactly when the family held nothing beforehand,
      so the finalize pass that decodes every cell to count rows only runs
      when earlier arrays may survive under the family.
    """
    n_records = int(src_vi.shape[0])
    if n_records == 0:
        return
    if src_cc.shape[1] != sid_ndim or trg_cc.shape[1] != sid_ndim:
        raise ArrayError(
            f"chunk coords arity mismatch in links/{delta}: sid_ndim="
            f"{sid_ndim}, got {src_cc.shape[1]}/{trg_cc.shape[1]}"
        )
    lg = get_resolution_level(root_group, level)
    create_links_family(lg, delta=delta, link_width=2, sid_ndim=sid_ndim)
    if stamp_capability:
        _stamp_root_capability(root_group, CAP_MULTISCALE_LINKS)
    family = links_group_path(delta)
    pre_existing = bool(list_link_offsets(lg, delta))

    rows = np.stack([
        np.asarray(src_vi, dtype=np.int64), np.asarray(trg_vi, dtype=np.int64),
    ], axis=1)
    src_cc = np.asarray(src_cc, dtype=np.int64)
    scale_src, scale_trg = link_endpoint_scales(lg, delta, sid_ndim)
    offs = np.asarray(trg_cc, dtype=np.int64) - (
        (src_cc * np.asarray(scale_src, dtype=np.int64))
        // np.asarray(scale_trg, dtype=np.int64)
    )

    # offsets -> [(source chunk, rows), ...]; one entry per cell.
    cells_by_offsets: dict[Any, list[tuple[ChunkCoords, npt.NDArray[np.int64]]]] = {}
    key = np.concatenate([src_cc, offs], axis=1)
    order, spans = _group_rows_by_key(key)
    key_sorted = key[order]
    block = rows[order]
    for start, end in spans:
        head = key_sorted[start].tolist()
        cells_by_offsets.setdefault((tuple(head[sid_ndim:]),), []).append(
            (tuple(head[:sid_ndim]), block[start:end]),
        )

    if pre_existing:
        # The replace: only arrays these records target are dropped.
        for off in cells_by_offsets:
            path = links_path(delta, off)
            if lg.array_exists(path):
                lg.delete_subtree(path)

    _write_cross_level_arrays(
        root_group, level, cells_by_offsets, delta=delta, sid_ndim=sid_ndim,
    )

    if pre_existing:
        # Arrays from before this call may survive under the family, so the
        # totals have to be recounted from the store.
        finalize_links(lg, delta=delta)
    else:
        # Canonical family, one row per record: logical == physical.
        physical = sum(
            int(cell_rows.shape[0])
            for cells in cells_by_offsets.values() for _cc, cell_rows in cells
        )
        lg.write_array_meta(family, {
            "num_links": physical, "num_physical_records": physical,
        })




# ===================================================================
# Full pyramid builder
# ===================================================================

def check_pyramid_request(
    store_path: str | Path,
    *,
    factors: list[tuple[float, float]],
    chunk_scale_factors: list[int | tuple[int, ...]] | None = None,
    sparsity_strategy: str = "random",
    sparsity_attribute: str | None = None,
    cross_level_depth: int = DEFAULT_CROSS_LEVEL_DEPTH,
    cross_level_storage: str = DEFAULT_CROSS_LEVEL_STORAGE,
    coarsen_mode: str = "rdp",
    method: str | None = None,
    rdp_tolerances: Sequence[float] | None = None,
    start_level: int = 0,
) -> list[float] | None:
    """Refuse a :func:`build_pyramid` request before anything is written.

    :func:`build_pyramid` runs this first.  A caller that removes an old
    pyramid to make way for the new one runs it before removing anything, so
    a request that would be refused leaves the old pyramid where it was.

    Returns the per-level RDP tolerances, checked, or ``None``.
    """
    if not 0 <= int(start_level) <= len(factors):
        raise ValueError(
            f"start_level must be between 0 and {len(factors)} (the number of "
            f"levels factors describes), got {start_level}"
        )
    if cross_level_storage not in VALID_XLEVEL_STORAGE:
        raise ValueError(
            f"cross_level_storage={cross_level_storage!r} not in "
            f"{sorted(VALID_XLEVEL_STORAGE)}"
        )
    if cross_level_depth < -1:
        raise ValueError(
            f"cross_level_depth must be ≥ -1 (got {cross_level_depth})"
        )
    for i, fac in enumerate(factors):
        # Both are "times coarser than the level below"; a fraction would
        # refine a level, or be read as 1 without a word.
        if (
            not isinstance(fac, (tuple, list)) or len(fac) != 2
            or not all(math.isfinite(float(v)) and float(v) >= 1.0 for v in fac)
        ):
            raise ValueError(
                f"factors[{i}] must be a (coarsen_factor, sparsity_factor) "
                f"pair of numbers >= 1; got {fac!r}"
            )
    if chunk_scale_factors is not None and len(chunk_scale_factors) != len(factors):
        raise ValueError(
            f"chunk_scale_factors length {len(chunk_scale_factors)} != "
            f"factors length {len(factors)}",
        )
    root_meta = read_root_metadata(open_store(str(store_path), mode="r"))
    for i, scale in enumerate(chunk_scale_factors or ()):
        axes = list(scale) if isinstance(scale, (tuple, list)) else [scale]
        if isinstance(scale, (tuple, list)) and len(axes) != root_meta.sid_ndim:
            raise ValueError(
                f"chunk_scale_factors[{i}] has rank {len(axes)}, but the store "
                f"has {root_meta.sid_ndim} spatial axes; got {scale!r}"
            )
        if not all(float(r) == int(r) >= 1 for r in axes):
            raise ValueError(
                f"chunk_scale_factors[{i}] must be positive integers per axis, "
                f"got {scale!r}"
            )
    key = method or select_coarsener_key(root_meta)
    if key not in _COARSENERS:
        raise ValueError(
            f"method={key!r} is not a registered coarsener; "
            f"choose from {coarsener_keys()}"
        )
    if sparsity_strategy == "attribute" or sparsity_attribute is not None:
        _sparsity_attribute_values(store_path, sparsity_strategy, sparsity_attribute)
    tolerances = validate_rdp_tolerances(
        rdp_tolerances, n_levels=len(factors), coarsen_mode=coarsen_mode,
    )
    if tolerances is not None:
        # The geometry does not change between levels, so one check here
        # refuses a store the tolerance cannot apply to before level 1 is
        # written, rather than after.
        refusal = _rdp_tolerance_refusal(root_meta, key)
        if refusal is not None:
            raise ValueError(f"rdp_tolerances does not apply: {refusal}")
    return tolerances


def build_pyramid(
    store_path: str | Path,
    *,
    factors: list[tuple[float, float]],
    chunk_scale_factors: list[int | tuple[int, ...]] | None = None,
    sparsity_strategy: str = "random",
    sparsity_seed: int | None = None,
    cross_level_depth: int = DEFAULT_CROSS_LEVEL_DEPTH,
    cross_level_storage: str = DEFAULT_CROSS_LEVEL_STORAGE,
    coarsen_mode: str = "rdp",
    compressor: Any = None,
    executor: Any = None,
    method: str | None = None,
    rdp_tolerances: Sequence[float] | None = None,
    start_level: int = 0,
    on_level_done: Callable[[int, dict[str, Any]], None] | None = None,
    sparsity_attribute: str | None = None,
) -> dict[str, Any]:
    """Build a multi-resolution pyramid for an existing store.

    Pass ``factors=[(coarsen_2, sparsity_3), ...]`` where ``factors[i]``
    is applied to produce level ``i+1`` from level ``i``.  Either factor
    at ``1.0`` opts out of that axis.  Each level is written by the
    coarsener :func:`select_coarsener_key` picks for the store's geometry
    (or ``method``); object ids are preserved across levels.

    Args:
        store_path: Path to the store with level 0.
        factors: List of ``(coarsen_factor, sparsity_factor)`` tuples,
            one per coarser level.  Both are **per-level ratios against the
            level below**, so they compound: ``[(2, 1), (2, 1), (2, 1)]`` bins
            at 2x, 4x and 8x the root bin.  (Sparsity was already cumulative
            via ``relative_to="alive"``; coarsening now matches it.)
        chunk_scale_factors: Optional per-level multipliers applied to
            the source level's ``chunk_shape`` to derive each target
            level's ``chunk_shape``.  Aligned with ``factors`` (same
            length).  Each entry is either a scalar int (uniform per
            axis) or a per-axis tuple.  ``None`` (default) means
            all-ones: every level inherits root ``chunk_shape``.
        sparsity_strategy: Object selection strategy.
        sparsity_seed: Random seed.
        cross_level_depth: Maximum absolute level delta for materialized
            cross-pyramid-level link arrays.  ``0`` = none, ``N`` = up
            to ``±N`` per pair (or ``+N`` only when
            ``cross_level_storage='implicit'``), ``-1`` = walk all
            available level pairs.  Default ``1``.
        cross_level_storage: ``"none"`` / ``"implicit"`` / ``"explicit"``.
            ``"explicit"`` materializes both ``+N`` (at the finer level)
            and ``-N`` (at the coarser level); ``"implicit"`` writes
            only ``+N``.  Default ``"explicit"``.  Each record links a
            vertex to the coarse vertex of the same object it was merged
            into.  Only the per-object coarsener (points, graphs, lines)
            writes them; other geometries ignore both settings.
        coarsen_mode: Only consulted for streamline/polyline stores:
            ``"rdp"`` (default) does Douglas-Peucker simplification;
            ``"decimate"`` does uniform stride decimation, in which case
            each level's ``coarsen_factor`` is interpreted as the stride.
        executor: Optional ``map``-like ``(func, items, shared) ->
            list[result]`` callable forwarded to every level's
            :func:`coarsen_level` call, parallelizing the chunk-local
            skeleton/polyline coarseners.  ``None`` (default) runs serially.
        rdp_tolerances: Explicit Douglas-Peucker tolerance per coarser level,
            aligned with ``factors`` (same length), in store coordinate
            units -- millimetres for a tractogram in RAS mm.  Level ``i+1``
            is simplified so that no vertex of level ``i`` lies further than
            ``rdp_tolerances[i]`` from it; deviation from level 0 is
            therefore bounded by the sum of the tolerances up to that level.
            ``None`` (default) keeps the derived tolerance, half the
            smallest edge of each level's bin, which compounds with the
            factors.  With explicit tolerances the coarsen factors no longer
            decide how much is simplified; each level's ``bin_shape`` (its
            NGFF scale) grows by its coarsen factor or its chunk scale,
            whichever is larger.  Refused
            with ``coarsen_mode="decimate"`` and for stores not coarsened by
            the polyline coarsener -- points, meshes, skeletons, graphs --
            before any level is written.
        start_level: How many of the levels ``factors`` describes already
            exist and are complete; building starts at level
            ``start_level + 1``.  For resuming a pyramid that stopped
            partway -- the caller removes any half-written level first.
        on_level_done: Called as ``on_level_done(level, summary)`` after
            each level is written, so a caller can record progress.
        sparsity_attribute: The object attribute ``sparsity_strategy=
            "attribute"`` ranks by; see :func:`coarsen_level`.

    Returns:
        Summary dict.  ``method`` is the coarsener that built the levels
        (``level_specs[i]["method"]`` per level).  ``cross_level_depth`` and
        ``cross_level_storage`` describe the cross-level links actually
        written, as stamped on root metadata: only the per-object coarsener
        (points, graphs, lines) writes any, so other geometries report
        ``0`` and ``"none"``.
    """
    tolerances = check_pyramid_request(
        store_path,
        factors=factors,
        chunk_scale_factors=chunk_scale_factors,
        sparsity_strategy=sparsity_strategy,
        sparsity_attribute=sparsity_attribute,
        cross_level_depth=cross_level_depth,
        cross_level_storage=cross_level_storage,
        coarsen_mode=coarsen_mode,
        method=method,
        rdp_tolerances=rdp_tolerances,
        start_level=start_level,
    )

    # Families reaching a level about to be rebuilt describe vertices that
    # are about to change.
    _clear_cross_level_families(
        open_store(str(store_path), mode="r+"), from_level=int(start_level),
    )

    summaries: list[dict[str, Any]] = []
    for i, fac in enumerate(factors):
        if i < int(start_level):
            continue
        if isinstance(fac, (tuple, list)) and len(fac) == 2:
            cf, sf = float(fac[0]), float(fac[1])
        else:
            raise ValueError(
                f"factors[{i}] must be a (coarsen_factor, sparsity_factor) "
                f"tuple; got {fac!r}"
            )
        chunk_scale = (
            chunk_scale_factors[i] if chunk_scale_factors is not None else 1
        )
        summaries.append(coarsen_level(
            store_path,
            source_level=i,
            target_level=i + 1,
            coarsen_factor=cf,
            sparsity_factor=sf,
            chunk_scale_factor=chunk_scale,
            sparsity_strategy=sparsity_strategy,
            # Advance the seed per level so "random" sparsity picks a
            # different survivor subset at each level; a constant seed would
            # otherwise apply the same selection to a nested candidate pool.
            sparsity_seed=(
                None if sparsity_seed is None else int(sparsity_seed) + i
            ),
            cross_level_storage=cross_level_storage,
            coarsen_mode=coarsen_mode,
            # Every level's codec is fixed at its own array creation, so the
            # compressor has to be forwarded per level — level 0's codec does
            # not propagate to the coarser ones.
            compressor=compressor,
            executor=executor,
            # Explicit strategy override, or None to keep the automatic
            # geometry routing for every level.
            method=method,
            rdp_tolerance=None if tolerances is None else tolerances[i],
            sparsity_attribute=sparsity_attribute,
        ))
        if on_level_done is not None:
            on_level_done(i + 1, summaries[-1])

    # Compose deeper-delta cross-level links from the inline-emitted +1
    # arrays, then stamp on root metadata the cross-level links the store
    # actually holds -- none, whatever was asked, for a geometry whose
    # coarsener writes none.
    written_depth, written_storage = _finalize_cross_level_for_store(
        store_path,
        cross_level_depth=cross_level_depth,
        cross_level_storage=cross_level_storage,
    )

    methods = sorted({str(s["method"]) for s in summaries if s.get("method")})
    return {
        "levels_created": len(summaries),
        "level_specs": summaries,
        "method": ", ".join(methods) or COARSEN_PER_OBJECT,
        "cross_level_depth": written_depth,
        "cross_level_storage": written_storage,
    }


# ===================================================================
# Built-in coarsener registrations
# ===================================================================

def _skeleton_coarsener(
    store_path: str | Path,
    source_level: int,
    target_level: int,
    *,
    coarsen_factor: float,
    sparsity_factor: float,
    chunk_scale_factor: int | tuple[int, ...],
    sparsity_strategy: str,
    sparsity_seed: int | None,
    cross_level_storage: str,
    coarsen_mode: str = "rdp",
    compressor: Any = None,
    executor: Any = None,
    attribute_values: Any = None,
) -> dict[str, Any]:
    """Skeleton stores: route to the skeleton-aware decimator.  ``coarsen_factor``
    is the decimation stride (1 being the identity, as elsewhere), and the
    random sparsity strategy degrades to deterministic ``"length"``.
    ``coarsen_mode`` is accepted for signature parity with other coarseners
    but has no effect here — this coarsener always decimates."""
    from zarr_vectors_tools.multiresolution.strategies.skeletons import (
        coarsen_skeleton_level,
    )
    # ``chunk_scale_factor`` is honoured as given.  It used to be silently
    # replaced by 2 whenever the caller asked for 1, so a pyramid built with
    # an explicit "keep the chunk grid" got a doubled grid at every level and
    # the store's own metadata was the only place that said so.  The default
    # of 2 now lives in ``coarsen_skeleton_level``'s signature, where a
    # caller can see it.
    csf = chunk_scale_factor
    extra = {} if attribute_values is None else {"attribute_values": attribute_values}
    return coarsen_skeleton_level(
        store_path, source_level, target_level,
        stride=max(1, int(round(coarsen_factor))),
        sparsity_factor=sparsity_factor,
        chunk_scale_factor=csf,
        sparsity_strategy=(
            sparsity_strategy if sparsity_strategy != "random" else "length"
        ),
        sparsity_seed=sparsity_seed,
        compressor=compressor,
        executor=executor,
        **extra,
    )


def _per_object_coarsener(
    store_path: str | Path,
    source_level: int,
    target_level: int,
    *,
    coarsen_factor: float,
    sparsity_factor: float,
    chunk_scale_factor: int | tuple[int, ...],
    sparsity_strategy: str,
    sparsity_seed: int | None,
    cross_level_storage: str,
    coarsen_mode: str = "rdp",
    compressor: Any = None,
    executor: Any = None,
    attribute_values: Any = None,
) -> dict[str, Any]:
    """Default geometries: per-object metavertex aggregation.
    ``coarsen_mode`` and ``executor`` are accepted for signature parity with
    other coarseners but have no effect here (this coarsener is not chunk-local)."""
    return _per_object_coarsen(
        store_path=store_path,
        source_level=source_level,
        target_level=target_level,
        coarsen_factor=coarsen_factor,
        sparsity_factor=sparsity_factor,
        chunk_scale_factor=chunk_scale_factor,
        sparsity_strategy=sparsity_strategy,
        sparsity_seed=sparsity_seed,
        cross_level_storage=cross_level_storage,
        compressor=compressor,
        attribute_values=attribute_values,
    )


def _polyline_coarsener(
    store_path: str | Path,
    source_level: int,
    target_level: int,
    *,
    coarsen_factor: float,
    sparsity_factor: float,
    chunk_scale_factor: int | tuple[int, ...],
    sparsity_strategy: str,
    sparsity_seed: int | None,
    cross_level_storage: str,
    coarsen_mode: str = "rdp",
    compressor: Any = None,
    executor: Any = None,
    rdp_tolerance: float | None = None,
    attribute_values: Any = None,
) -> dict[str, Any]:
    """Geometry-preserving polyline/streamline coarsener (RDP simplification
    or uniform stride decimation, selected by ``coarsen_mode``).

    Chunk-local and executor-parallel — see
    :func:`zarr_vectors_tools.multiresolution.strategies.polylines.coarsen_polyline_level`
    for the implementation (peak memory O(one target chunk), not O(the whole
    source level)).  ``rdp_tolerance`` is that function's
    ``simplify_epsilon``; ``None`` leaves it to derive one from the bin."""
    from zarr_vectors_tools.multiresolution.strategies.polylines import (
        coarsen_polyline_level,
    )
    return coarsen_polyline_level(
        store_path,
        source_level,
        target_level,
        coarsen_factor=coarsen_factor,
        sparsity_factor=sparsity_factor,
        chunk_scale_factor=chunk_scale_factor,
        sparsity_strategy=sparsity_strategy,
        sparsity_seed=sparsity_seed,
        coarsen_mode=coarsen_mode,
        simplify_epsilon=rdp_tolerance,
        compressor=compressor,
        executor=executor,
        attribute_values=attribute_values,
    )


register_coarsener("skeleton", _skeleton_coarsener)
register_coarsener("per_object", _per_object_coarsener)
register_coarsener("polyline", _polyline_coarsener)

from zarr_vectors_tools.multiresolution.strategies.fragments import (  # noqa: E402
    _per_fragment_coarsener,
)

register_coarsener("per_fragment", _per_fragment_coarsener)

from zarr_vectors_tools.multiresolution.strategies.meshes import (  # noqa: E402
    _mesh_coarsener,
)

register_coarsener("mesh", _mesh_coarsener)

from zarr_vectors_tools.multiresolution.strategies.mesh_decimate_level import (  # noqa: E402
    _mesh_decimate_coarsener,
)

# Quadric edge collapse. Registered under its own key rather than replacing
# "mesh", so a store can be rebuilt either way and the two compared on the
# same data; select_coarsener_key still routes mesh stores to the incumbent
# unless build_pyramid is given method="mesh_decimate".
register_coarsener("mesh_decimate", _mesh_decimate_coarsener)
