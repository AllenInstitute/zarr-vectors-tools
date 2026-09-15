"""Multi-resolution pyramid construction orchestrator.

Two entry points (use one):

* ``build_pyramid(store, factors=[(cf_1, sf_1), ...])`` builds every
  coarser level in sequence, optionally emitting cross-level link
  arrays (``cross_level_storage="implicit"`` or ``"explicit"``).
* ``coarsen_level(store, source, target, coarsen_factor=..., sparsity_factor=...)``
  writes a single coarser level for callers that want manual control.

Both use the per-object pyramid: each surviving object's vertices are
aggregated into bin centroids (metavertices) that may be shared
between objects, and per-object OIDs are preserved across levels.
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
    assign_chunks,
    build_vertex_chunk_mapping,
    create_links_array,
    create_links_family,
    create_object_attributes_array,
    create_object_index_array,
    create_resolution_level,
    create_vertices_array,
    finalize_links,
    get_level_chunk_shape,
    get_resolution_level,
    intra_offsets,
    is_intra,
    link_endpoint_scales,
    links_group_path,
    links_path,
    list_chunk_keys,
    list_link_offsets,
    list_resolution_levels,
    open_store,
    read_all_object_manifests,
    read_chunk_vertices,
    read_level_metadata,
    read_links,
    read_object_attributes,
    read_root_metadata,
    read_vertex_fragment_index,
    rebuild_presence,
    update_root_metadata,
    write_chunk_links,
    write_chunk_vertices,
    write_links,
    write_object_attributes,
    write_object_index,
)
from zarr_vectors.constants import (
    CAP_MULTISCALE_LINKS,
    CAP_PRESERVED_OBJECT_IDS,
    CAP_SHARED_FRAGMENTS,
    COARSEN_PER_OBJECT,
    DEFAULT_CROSS_LEVEL_DEPTH,
    DEFAULT_CROSS_LEVEL_STORAGE,
    GEOM_MESH,
    LINKS_IMPLICIT_BRANCHES,
    LINKS_IMPLICIT_SEQUENTIAL,
    OBJECT_ATTRIBUTES,
    VALID_XLEVEL_STORAGE,
    VERTEX_FRAGMENTS,
    VERTICES,
    XLEVEL_EXPLICIT,
    XLEVEL_NONE,
)
from zarr_vectors.exceptions import ArrayError, CoarseningError, StoreError
from zarr_vectors.typing import ChunkCoords

from zarr_vectors_tools.algorithms._links import chunk_key_str, read_cross_links
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
from zarr_vectors_tools.multiresolution.object_selection import apply_sparsity

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
) -> dict[str, Any]:
    """Coarsen a single level and write it to the store.

    Per-object vertex aggregation with stable OIDs across levels.  A
    metavertex's source vertices may come from multiple source objects;
    the resulting metavertex appears in each of those objects' manifests
    at the coarser level.

    Args:
        store_path: Path to the zarr vectors store.
        source_level: Level to read from.
        target_level: Level to write to (must not exist).
        coarsen_factor: Per-object vertex aggregation factor (>= 1),
            expressed as a **ratio against the source level's bin**, not the
            root's: the target bin is ``source_level.bin_shape *
            coarsen_factor``, seeded from the root's effective bin at level 0.
            Factors therefore compound down a pyramid.  ``1.0`` is the identity
            (no aggregation).
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
    factor, and ``chunk_scale_factor`` defaults to 2.
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
) -> dict[str, Any]:
    """Per-object pyramid: aggregate within-bin source vertices into
    shared metavertices, preserving each surviving object's OID and
    its trajectory through the new metavertices.

    See the 12-step implementation sketch in the plan file
    ``Provenance-preserving pyramid: shared metavertices + ID-stable
    objects`` (`schema/zarr_vectors.linkml.yaml` schema captures the
    persistent metadata side).
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
    chunk_shape = target_chunk_shape  # used for assign_chunks below

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
            bin_shape=base_bin,
            alive_mask=alive_mask,
            # Cumulative per level: fraction of the surviving pool, not of
            # the original count.  See apply_sparsity's `relative_to`.
            relative_to="alive",
        )
        keep_oids = sorted(int(o) for o in kept)
    else:
        keep_oids = list(range(n_src_objects))

    # --- Step 2-3: build (source vertex → bin → metavertex) map ---------
    # Per-object ordered source-vertex positions (with their global index
    # in the flat source-vertex array).
    per_object_positions: dict[int, np.ndarray] = {}
    flat_positions: list[np.ndarray] = []
    flat_oid_of_v: list[int] = []
    next_global = 0
    for oid in keep_oids:
        manifest = src_manifests[oid]
        parts: list[np.ndarray] = []
        for cc, fragment_idx in manifest:
            fragment = src_fragment_positions.get((cc, fragment_idx))
            if fragment is None or len(fragment) == 0:
                continue
            parts.append(np.asarray(fragment, dtype=np.float32))
        if not parts:
            per_object_positions[oid] = np.zeros((0, ndim), dtype=np.float32)
            continue
        obj_positions = np.concatenate(parts, axis=0)
        per_object_positions[oid] = obj_positions
        flat_positions.append(obj_positions)
        flat_oid_of_v.extend([oid] * obj_positions.shape[0])
        next_global += obj_positions.shape[0]

    if not flat_positions:
        # Surviving objects had no vertices.  Write an empty level.
        _write_empty_preserve_level(
            root, source_level, target_level,
            base_bin=base_bin,
            root_bin=root_meta.effective_bin_shape,
            coarsen_factor=coarsen_factor,
            sparsity_factor=sparsity_factor,
            inherited_num_objects=n_src_objects,
        )
        return {
            "vertex_count": 0,
            "object_count": 0,
            "objects_kept": len(keep_oids),
            "method": COARSEN_PER_OBJECT,
            "preserves_object_ids": True,
            "shared_fragments": True,
        }

    all_pos = np.concatenate(flat_positions, axis=0)

    # Target bin shape: the SOURCE level's bin_shape x coarsen_factor, so the
    # factor is a per-level ratio and successive levels compound.
    target_bin_shape = tuple(float(b) * float(coarsen_factor) for b in base_bin)

    # Compute per-vertex bin coords: (N, ndim) int64.
    bin_shape_arr = np.asarray(target_bin_shape, dtype=np.float64)
    bin_coords = np.floor(all_pos / bin_shape_arr).astype(np.int64)
    # Combine each bin coord tuple into a single sort-key for np.unique.
    bin_keys = np.ascontiguousarray(bin_coords).view(
        np.dtype((np.void, bin_coords.dtype.itemsize * bin_coords.shape[1]))
    ).ravel()
    _, inverse = np.unique(bin_keys, return_inverse=True)
    inverse = inverse.astype(np.int64, copy=False)
    n_metavertices = int(inverse.max()) + 1 if inverse.size > 0 else 0

    # --- Step 3 (continued): centroid per bin --------------------------
    # ``np.bincount`` rather than ``np.add.at``: the latter is the unbuffered
    # ufunc.at path and runs an order of magnitude slower for the same
    # scatter-add, which matters once a level carries millions of vertices.
    bin_counts = np.bincount(inverse, minlength=n_metavertices)
    meta_positions = np.empty((n_metavertices, ndim), dtype=np.float32)
    for d in range(ndim):
        meta_positions[:, d] = np.bincount(
            inverse, weights=all_pos[:, d], minlength=n_metavertices,
        ) / bin_counts

    # --- Step 4: chunk-assign metavertices ------------------------------
    chunk_assignments = assign_chunks(meta_positions, chunk_shape)

    # --- Step 5-9b: branch on links_convention --------------------------
    # The ``implicit_sequential`` (streamline / polyline) path keeps each
    # object's path in a single multi-vertex fragment per coarsened chunk,
    # so consecutive metavertices belong to the same fragment and their
    # implicit edges encode the connectivity.  Other conventions stay on
    # the legacy "one fragment per metavertex" layout where Step 9b
    # bridges consecutive manifest entries with explicit links/0 records.
    use_implicit_sequential = (
        root_meta.links_convention == LINKS_IMPLICIT_SEQUENTIAL
    )

    if use_implicit_sequential:
        # Pass 1: per-(oid, coarsened chunk) segmentation.  Each surviving
        # object's source vertex sequence is split at coarsened-chunk
        # boundaries; within each per-chunk run, consecutive same-bin
        # vertices collapse to a single metavertex.  Source cross-chunk
        # edges whose endpoints both fall in the same coarsened chunk are
        # absorbed into the merged run for that chunk.
        per_object_runs: dict[int, list[tuple[ChunkCoords, list[int]]]] = {}
        per_object_aux: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        cursor = 0
        for oid in keep_oids:
            n_obj = per_object_positions[oid].shape[0]
            if n_obj == 0:
                per_object_runs[oid] = []
                per_object_aux[oid] = (
                    np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
                )
                continue
            mv_seq = inverse[cursor:cursor + n_obj].astype(np.int64, copy=False)
            cursor += n_obj
            coarse_cc_seq = coarse_chunks_of(
                per_object_positions[oid], target_chunk_shape,
            )
            runs = segment_object_by_coarse_chunk(mv_seq, coarse_cc_seq)
            run_idx, pos_in_run = positions_in_run(
                n_obj, mv_seq, coarse_cc_seq,
            )
            per_object_runs[oid] = runs
            per_object_aux[oid] = (run_idx, pos_in_run)

        # Per-chunk assembly: deterministic order — by oid (keep_oids
        # order), then by run index within the object.
        per_chunk_assembly: dict[
            ChunkCoords, list[tuple[int, int, list[int]]]
        ] = {}
        for oid in keep_oids:
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

        # Per-object manifest at the coarse level: one entry per run.
        new_manifests = {}
        for oid in keep_oids:
            manifest: list[tuple[ChunkCoords, int]] = []
            for r_idx, _ in enumerate(per_object_runs[oid]):
                cc_out, frag_idx_out, _ = run_to_fragment[(oid, r_idx)]
                manifest.append((cc_out, frag_idx_out))
            new_manifests[oid] = manifest

        # Vertex groups for write_chunk_vertices (range fragments).
        per_chunk_groups = {}
        for coarse_cc, entries in per_chunk_assembly.items():
            groups: list[np.ndarray] = []
            for (_oid, _r_idx, mv_list) in entries:
                if mv_list:
                    groups.append(
                        meta_positions[np.asarray(mv_list, dtype=np.int64)],
                    )
                else:
                    groups.append(np.zeros((0, ndim), dtype=np.float32))
            per_chunk_groups[coarse_cc] = groups

        # Source-vertex → coarse endpoint map for Pass 2 (cross-chunk
        # link remapping) and for cross-level link emission.  Built by
        # walking the source manifest in fragment order and pairing each
        # source vertex with its (run_idx, pos_in_run) so we know which
        # coarse fragment owns it.
        # src_chunk_fragment_starts is built up in Step 0 as each chunk is
        # read -- see there.
        src_endpoint_map: dict[
            tuple[ChunkCoords, int], tuple[ChunkCoords, int]
        ] = {}
        for oid in keep_oids:
            n_obj_total = per_object_positions[oid].shape[0]
            if n_obj_total == 0:
                continue
            run_idx_arr, pos_in_run_arr = per_object_aux[oid]
            obj_runs = per_object_runs[oid]
            src_vidx_within_obj = 0
            for (m_cc, m_fid) in src_manifests[oid]:
                fragment_arr = src_fragment_positions.get((m_cc, m_fid))
                if fragment_arr is None or len(fragment_arr) == 0:
                    continue
                n_frag = len(fragment_arr)
                f_start = src_chunk_fragment_starts[m_cc][m_fid]
                for i in range(n_frag):
                    src_local_vi = f_start + i
                    r_idx = int(run_idx_arr[src_vidx_within_obj])
                    pos = int(pos_in_run_arr[src_vidx_within_obj])
                    coarse_cc = obj_runs[r_idx][0]
                    _, _, chunk_start = run_to_fragment[(oid, r_idx)]
                    coarse_local_vi = chunk_start + pos
                    # First write wins: multiple source vertices in the
                    # same source-chunk row are impossible, but multiple
                    # source vertices may map to the same coarse row.
                    # The (src_cc, src_local_vi) key is unique by
                    # construction so simple assignment is fine.
                    src_endpoint_map[(m_cc, src_local_vi)] = (
                        coarse_cc, coarse_local_vi,
                    )
                    src_vidx_within_obj += 1

        # mv_to_chunk_first_row: for each metavertex, the (chunk, first
        # chunk-local row) it lives in.  Used by cross-level link
        # emission since a single metavertex can now occupy multiple
        # rows (one per per-object fragment that visits it).
        mv_first_row_chunk: dict[int, ChunkCoords] = {}
        mv_first_row_local: dict[int, int] = {}
        for coarse_cc, entries in per_chunk_assembly.items():
            cum = 0
            for (_oid, _r_idx, mv_list) in entries:
                for p, mv in enumerate(mv_list):
                    mv_int = int(mv)
                    if mv_int not in mv_first_row_chunk:
                        mv_first_row_chunk[mv_int] = coarse_cc
                        mv_first_row_local[mv_int] = cum + p
                cum += len(mv_list)
    # else: legacy path computes its own per_chunk_groups / new_manifests
    # below.

    # --- Step 5 (legacy): per-chunk fragment layout (one fragment per metavertex)
    if not use_implicit_sequential:
        metavertex_to_ref: dict[int, tuple[ChunkCoords, int]] = {}
        per_chunk_groups: dict[ChunkCoords, list[np.ndarray]] = {}
        for cc, indices in sorted(chunk_assignments.items()):
            for fragment_idx, mv_idx in enumerate(indices.tolist()):
                metavertex_to_ref[int(mv_idx)] = (cc, fragment_idx)
                per_chunk_groups.setdefault(cc, []).append(
                    meta_positions[mv_idx:mv_idx + 1]
                )

    # --- Step 6: write per-chunk fragments --------------------------
    arrays_present = [VERTICES, "object_index"] if src_has_objects else [VERTICES]
    # ``shared_fragments`` is False on the implicit_sequential path:
    # fragments are per-(object, coarsened-chunk), not shared between
    # objects.  Legacy path keeps the historical True so the existing
    # CAP_SHARED_FRAGMENTS contract is preserved for non-streamline
    # geometries.
    shared_fragments_flag = not use_implicit_sequential
    level_meta_initial = LevelMetadata(
        level=target_level,
        vertex_count=int(n_metavertices),
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
        shared_fragments=shared_fragments_flag,
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

    # --- Step 7 (legacy): emit per-object manifests ---------------------
    if not use_implicit_sequential:
        cursor = 0
        new_manifests = {}
        for oid in keep_oids:
            n = per_object_positions[oid].shape[0]
            if n == 0:
                new_manifests[oid] = []
                continue
            mv_seq = inverse[cursor:cursor + n].tolist()
            cursor += n
            # Deduplicate consecutive duplicates while preserving order.
            manifest = []
            prev = -1
            for mv_idx in mv_seq:
                if mv_idx == prev:
                    continue
                prev = mv_idx
                manifest.append(metavertex_to_ref[int(mv_idx)])
            new_manifests[oid] = manifest

    # --- Step 9: emit object_index (gap-fill for dropped OIDs) ----------
    if src_has_objects:
        write_object_index(
            level_group, new_manifests, sid_ndim=ndim,
            total_objects=n_src_objects,
        )
        # Carry the group taxonomy forward. Object ids are preserved, so a
        # group's membership is meaningful here unchanged; without this the
        # coarse level has an object index but no way to say what any object
        # IS, and a reader has to reach back to level 0 for the taxonomy.
        propagate_groupings(
            src_group, level_group,
            surviving_oids=surviving_oids_from(keep_oids, sparsity_factor),
        )

    # --- Step 9b: boundary-spanning links at delta 0 --------------------
    # ``directed`` is a family-wide, un-flippable policy per (level, delta):
    # every offsets array under links/0 decodes against it.  The two
    # branches below disagree on it deliberately — implicit-sequential
    # records carry endpoint order as data (predecessor→successor), the
    # legacy bridges do not — and they are mutually exclusive, so exactly
    # one policy is ever stamped for a given level.  Anything that later
    # writes links/0 at this level must match whichever ran.
    if use_implicit_sequential:
        # Pass 2: remap the source level's boundary-spanning records to the
        # new coarse-chunk-local indices.  Drop records whose endpoints
        # both fell into the same coarsened chunk — those were absorbed by
        # Pass 1's merged fragments.  Cross-only: a source intra record is
        # by definition already inside one chunk and has nothing to bridge.
        src_cross_records = read_cross_links(src_group, delta=0)
        new_cross_links = []
        for record in src_cross_records:
            if len(record) != 2:
                continue
            (cc_a, vi_a), (cc_b, vi_b) = record  # type: ignore[misc]
            new_a = src_endpoint_map.get((cc_a, int(vi_a)))
            new_b = src_endpoint_map.get((cc_b, int(vi_b)))
            if new_a is None or new_b is None:
                continue
            new_cc_a, new_vi_a = new_a
            new_cc_b, new_vi_b = new_b
            if new_cc_a == new_cc_b:
                continue
            new_cross_links.append(
                ((new_cc_a, new_vi_a), (new_cc_b, new_vi_b)),
            )
        create_links_family(
            level_group, delta=0, link_width=2, sid_ndim=ndim, directed=True,
        )
        if new_cross_links:
            write_links(
                level_group, new_cross_links, sid_ndim=ndim, delta=0,
                directed=True,
            )
        else:
        # ``write_links`` stamps the family counts as a side effect, so a
        # family with nothing to write would otherwise carry policy but no
        # ``num_links`` — a shape every family is supposed to be free of.
        # Finalize explicitly to stamp the zero.
            finalize_links(level_group, delta=0)
    else:
        # Legacy Step 9b: one fragment per metavertex, so consecutive
        # manifest entries are bridged with an explicit record.  Entries
        # that share a chunk are bridged too, and that is intentional.
        # Note such a record has all-zero offsets, so it now lands in the
        # intra array (links/0/0.0.0) rather than a separate cross family —
        # which also makes it visible to read_chunk_links, where before it
        # was only reachable through the cross reader.
        cross_links = []
        for oid, manifest in new_manifests.items():
            if len(manifest) < 2:
                continue
            for i in range(len(manifest) - 1):
                cc_a, frag_a = manifest[i]
                cc_b, frag_b = manifest[i + 1]
                # vi_a == frag_a, vi_b == frag_b (one metavertex per fragment).
                cross_links.append(((cc_a, frag_a), (cc_b, frag_b)))
        create_links_family(
            level_group, delta=0, link_width=2, sid_ndim=ndim,
        )
        if cross_links:
            write_links(
                level_group, cross_links, sid_ndim=ndim, delta=0,
            )
        else:
        # ``write_links`` stamps the family counts as a side effect, so a
        # family with nothing to write would otherwise carry policy but no
        # ``num_links`` — a shape every family is supposed to be free of.
        # Finalize explicitly to stamp the zero.
            finalize_links(level_group, delta=0)

    # --- Step 10: per-object attributes with present_mask ---------------
    src_obj_attr_group_name = f"{OBJECT_ATTRIBUTES}"
    if src_obj_attr_group_name in src_group:
        src_obj_attr_group = src_group[src_obj_attr_group_name]
        # Object attributes are flat arrays; enumerate via children() —
        # iterating the group yields only sub-group names (none here).
        attr_names = list(src_obj_attr_group.children())
    else:
        attr_names = []
    for attr_name in attr_names:
        try:
            src_data = read_object_attributes(src_group, attr_name)
        except ArrayError:
            continue
        # Dense (O, C) or (O,) padded to the inherited OID space, with
        # rows for survivors copied over.  Layout matches the source's
        # OID space (which already equals n_src_objects).
        out_data = np.zeros_like(src_data)
        for oid in keep_oids:
            if oid < len(src_data):
                out_data[oid] = src_data[oid]
        mask = np.zeros(n_src_objects, dtype=np.uint8)
        for oid in keep_oids:
            mask[oid] = 1
        create_object_attributes_array(level_group, attr_name)
        write_object_attributes(level_group, attr_name, out_data, present_mask=mask)

    # --- Step 12: stamp root capability tokens --------------------------
    if src_has_objects:
        _stamp_root_capability(root, CAP_PRESERVED_OBJECT_IDS)
    if not use_implicit_sequential:
        # On the implicit_sequential path fragments are per-(object,
        # coarsened-chunk) — not shared between objects — so we don't
        # claim the shared-fragments capability.
        _stamp_root_capability(root, CAP_SHARED_FRAGMENTS)

    # --- Step 13: emit inline ±1 cross-level link arrays ----------------
    if cross_level_storage != XLEVEL_NONE and n_metavertices > 0:
        if use_implicit_sequential:
            # A metavertex may occupy multiple rows in its chunk (one per
            # per-object fragment that visits it).  Pass the precomputed
            # "first row per metavertex" map so cross-level edges point
            # to a canonical row.
            _emit_inline_cross_level_links(
                root,
                src_group=src_group,
                level_group=level_group,
                source_level=source_level,
                ndim=ndim,
                bin_shape_arr=bin_shape_arr,
                bin_keys=bin_keys,
                coarse_chunk_assignments_mv=None,
                storage=cross_level_storage,
                mv_first_row_chunk=mv_first_row_chunk,
                mv_first_row_local=mv_first_row_local,
            )
        else:
            _emit_inline_cross_level_links(
                root,
                src_group=src_group,
                level_group=level_group,
                source_level=source_level,
                ndim=ndim,
                bin_shape_arr=bin_shape_arr,
                bin_keys=bin_keys,
                coarse_chunk_assignments_mv=chunk_assignments,
                storage=cross_level_storage,
            )

    # This coarsener writes serially (no cross-process manifest race), but
    # re-derive the per-array ``nonempty_chunks`` manifests from disk anyway for
    # uniformity with the parallel coarseners and idempotence.
    rebuild_presence(level_group)

    return {
        "vertex_count": int(n_metavertices),
        "object_count": len(keep_oids),
        "objects_kept": len(keep_oids),
        "source_objects": n_src_objects,
        "method": COARSEN_PER_OBJECT,
        "preserves_object_ids": True,
        "shared_fragments": shared_fragments_flag,
    }


def _emit_inline_cross_level_links(
    root,
    *,
    src_group,
    level_group,
    source_level: int,
    ndim: int,
    bin_shape_arr: npt.NDArray[np.float64],
    bin_keys: npt.NDArray,
    coarse_chunk_assignments_mv: dict[ChunkCoords, npt.NDArray[np.int64]] | None,
    storage: str,
    mv_first_row_chunk: dict[int, ChunkCoords] | None = None,
    mv_first_row_local: dict[int, int] | None = None,
) -> None:
    """Emit the ``±1`` links family for one coarsen step.

    Re-walks the source level once in chunk-major order, re-bins each
    vertex against ``bin_shape_arr``, and looks up the matching
    metavertex via the ``bin_key`` ↔ ``mv_idx`` map implicit in
    ``np.unique(bin_keys, return_inverse=inverse)``.  Translates
    metavertex IDs to chunk-major-flat coarse indices via the
    just-written coarse level's fragment index, then dispatches to
    :func:`_write_cross_level_edges`.

    Two modes for the metavertex → coarse-row lookup:

    * **Legacy** (one fragment per metavertex): pass
      ``coarse_chunk_assignments_mv``.  Position k in the per-chunk array
      is the chunk-local row of metavertex ``coarse_chunk_assignments_mv[cc][k]``.
    * **Per-(object, chunk) fragments**: pass ``mv_first_row_chunk`` +
      ``mv_first_row_local``.  Each metavertex maps to its canonical
      first row in its chunk (multiple per-object fragments may include
      the same metavertex, but cross-level links use the first).
    """
    # bin_key → mv_idx.  ``np.unique`` output is sorted, so a metavertex id is
    # ``searchsorted`` on it — which lets the per-vertex lookup below run as
    # one vectorised call per chunk instead of a dict probe (and a fresh
    # ``bytes`` object) for every source vertex in the level.
    unique_keys = np.unique(bin_keys)
    # n_mv >= 1: callers gate this function on ``n_metavertices > 0`` and
    # unique_keys is the same set of bins.
    n_mv = int(unique_keys.shape[0])

    # mv_idx → chunk-major-flat coarse index.  The coarse level was written a
    # moment ago, so its row counts come from the fragment index alone rather
    # than from decoding every vertex again.
    coarse_chunk_assignments, n_coarse = _reconstruct_chunk_assignments(
        level_group, ndim,
    )
    # mv_idx → coarse row, as an array so the per-vertex translation below is a
    # gather rather than a dict probe.  -1 marks a metavertex with no coarse
    # row, which reads back as the same "leave parent at -1" outcome.
    mv_to_coarse_arr = np.full(max(n_mv, 1), -1, dtype=np.int64)
    if mv_first_row_chunk is not None and mv_first_row_local is not None:
        for mv_idx, cc in mv_first_row_chunk.items():
            local_row = mv_first_row_local[mv_idx]
            chunk_rows = coarse_chunk_assignments.get(cc)
            if chunk_rows is None or local_row >= len(chunk_rows):
                continue
            if 0 <= mv_idx < n_mv:
                mv_to_coarse_arr[mv_idx] = chunk_rows[local_row]
    elif coarse_chunk_assignments_mv is not None:
        for cc, mv_indices_for_chunk in sorted(coarse_chunk_assignments_mv.items()):
            mv_idx = np.asarray(mv_indices_for_chunk, dtype=np.int64)
            rows = coarse_chunk_assignments[cc][:mv_idx.shape[0]]
            if rows.shape[0] != mv_idx.shape[0]:
                raise IndexError(
                    f"coarse chunk {cc} holds {rows.shape[0]} rows but "
                    f"{mv_idx.shape[0]} metavertices were assigned to it"
                )
            in_range = (mv_idx >= 0) & (mv_idx < n_mv)
            mv_to_coarse_arr[mv_idx[in_range]] = rows[in_range]
    else:
        raise ValueError(
            "Either coarse_chunk_assignments_mv or "
            "(mv_first_row_chunk, mv_first_row_local) must be supplied",
        )

    # Build fine→coarse parent[] by walking the source in chunk-major order.
    # One walk yields both the source's chunk assignments (the row count of
    # each chunk as ``read_chunk_vertices`` sees it) and the parents, where
    # this used to be two full reads of the level.
    key_dtype = np.dtype((
        np.void, int(bin_shape_arr.shape[0]) * np.dtype(np.int64).itemsize,
    ))
    fine_chunk_assignments: dict[ChunkCoords, npt.NDArray[np.int64]] = {}
    parent_parts: list[npt.NDArray[np.int64]] = []
    n_fine = 0
    src_chunk_keys = list(list_chunk_keys(src_group, VERTICES))
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
            fragments = [f for f in fragments if int(f.shape[0]) > 0]
            if not fragments:
                continue
            positions = (
                fragments[0] if len(fragments) == 1
                else np.concatenate(fragments, axis=0)
            )
            n_local = int(positions.shape[0])
            local_bins = np.floor(
                np.asarray(positions, dtype=np.float32) / bin_shape_arr,
            ).astype(np.int64)
            local_keys = np.ascontiguousarray(local_bins).view(key_dtype).ravel()
            # searchsorted gives the insertion point, which is the mv id only
            # where the key is actually present — hence the equality check,
            # standing in for the dict's ``.get(...) is None``.
            idx = np.searchsorted(unique_keys, local_keys)
            np.clip(idx, 0, n_mv - 1, out=idx)
            hit = unique_keys[idx] == local_keys
            parent_parts.append(np.where(hit, mv_to_coarse_arr[idx], -1))
            fine_chunk_assignments[cc] = np.arange(
                n_fine, n_fine + n_local, dtype=np.int64,
            )
            n_fine += n_local
    parent = (
        np.concatenate(parent_parts).astype(np.int64, copy=False)
        if parent_parts else np.empty(0, dtype=np.int64)
    )

    _write_cross_level_edges(
        root,
        fine_level=source_level,
        delta=1,
        fine_chunk_assignments=fine_chunk_assignments,
        coarse_chunk_assignments=coarse_chunk_assignments,
        n_fine=n_fine,
        n_coarse=n_coarse,
        parent=parent,
        sid_ndim=ndim,
        storage=storage,
    )


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
        shared_fragments=True,
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


def _stamp_root_cross_level(
    root_group, *, depth: int, storage: str,
) -> None:
    """Persist cross_level_depth/cross_level_storage on root metadata."""
    update_root_metadata(
        root_group, cross_level_depth=int(depth), cross_level_storage=storage,
    )


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
) -> None:
    """Persist root cross-level metadata and emit ``±N`` (N ≥ 2) link arrays.

    Adjacent ``±1`` arrays are emitted inline during coarsening (see
    :func:`_emit_inline_cross_level_links`).  This finalize pass walks
    every adjacent (fine, coarse) level pair, decodes the on-disk
    ``+1`` parent map back into a flat fine→coarse array, then composes
    step-by-step to produce ``+N``/``-N`` link arrays for N ≥ 2 up to
    ``cross_level_depth``.

    ``cross_level_depth=-1`` means "walk all available level pairs".
    """
    root = open_store(str(store_path), mode="r+")
    if cross_level_storage == XLEVEL_NONE or cross_level_depth == 0:
        _stamp_root_cross_level(
            root, depth=cross_level_depth, storage=cross_level_storage,
        )
        return
    # Stamped only once we know arrays can exist: a strategy that emits no
    # inline ±1 links (polyline, skeleton, mesh, per-fragment) would
    # otherwise leave the store advertising a cross-level depth it has no
    # arrays for, which a reader then goes looking for.
    _stamp_root_cross_level(
        root, depth=cross_level_depth, storage=cross_level_storage,
    )

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

    Produces the store the three-writer sequence did -- the chunk-aligned
    rows via ``write_chunk_links``, the rest via ``write_links`` (a scoped
    replace), then ``finalize_links`` -- without its per-record Python
    objects or its per-cell synchronous writes:

    * **Placement** is the arithmetic ``write_links`` applies to a
      cross-level record (``links_has_perm`` is False at ``delta != 0``, so
      the placement is the identity): a record is *aligned* when both chunk
      coords are equal, and goes to the all-zero-offsets array; otherwise
      its offsets are ``trg - floor(src * scale_src / scale_trg)``.  Rows
      keep input order within a cell.
    * **The replace** deletes exactly the offsets arrays the spanning records
      target.  When that set includes the all-zero array -- possible whenever
      the two grids differ, since re-anchoring maps a coarser chunk onto a
      finer one -- the aligned rows written into it first did not survive.
      That outcome is reproduced here by not writing them.
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
    aligned = np.all(src_cc == trg_cc, axis=1)

    # offsets -> [(source chunk, rows), ...]; one entry per cell.
    cross_cells: dict[Any, list[tuple[ChunkCoords, npt.NDArray[np.int64]]]] = {}
    cross_idx = np.flatnonzero(~aligned)
    if cross_idx.size:
        scale_src, scale_trg = link_endpoint_scales(lg, delta, sid_ndim)
        c_src = src_cc[cross_idx]
        offs = trg_cc[cross_idx] - (
            (c_src * np.asarray(scale_src, dtype=np.int64))
            // np.asarray(scale_trg, dtype=np.int64)
        )
        key = np.concatenate([c_src, offs], axis=1)
        order, spans = _group_rows_by_key(key)
        key_sorted = key[order]
        block = rows[cross_idx[order]]
        for start, end in spans:
            head = key_sorted[start].tolist()
            cross_cells.setdefault((tuple(head[sid_ndim:]),), []).append(
                (tuple(head[:sid_ndim]), block[start:end]),
            )
    intra = intra_offsets(sid_ndim, 2)
    intra_targeted = any(is_intra(off) for off in cross_cells)

    cells_by_offsets = dict(cross_cells)
    aligned_idx = np.flatnonzero(aligned)
    if aligned_idx.size and not intra_targeted:
        order, spans = _group_rows_by_key(src_cc[aligned_idx])
        a_sorted = aligned_idx[order]
        cells_by_offsets[intra] = [
            (tuple(src_cc[a_sorted[start]].tolist()), rows[a_sorted[start:end]])
            for start, end in spans
        ]

    if pre_existing:
        # The replace: only arrays the spanning records target are dropped.
        for off in cross_cells:
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
) -> dict[str, Any]:
    """Build a multi-resolution pyramid for an existing store.

    Pass ``factors=[(coarsen_2, sparsity_3), ...]`` where ``factors[i]``
    is applied to produce level ``i+1`` from level ``i``.  Either factor
    at ``1.0`` opts out of that axis.  Uses the per-object pyramid:
    each surviving object's vertices are aggregated into bin centroids
    (metavertices); metavertices may be shared between objects and OIDs
    are preserved across levels.

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
            only ``+N``.  Default ``"explicit"``.
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
            factors.  With explicit tolerances the coarsen factors still set
            each level's ``bin_shape`` (its NGFF scale) but no longer decide
            how much is simplified.  Refused
            with ``coarsen_mode="decimate"`` and for stores not coarsened by
            the polyline coarsener -- points, meshes, skeletons, graphs --
            before any level is written.
        start_level: How many of the levels ``factors`` describes already
            exist and are complete; building starts at level
            ``start_level + 1``.  For resuming a pyramid that stopped
            partway -- the caller removes any half-written level first.
        on_level_done: Called as ``on_level_done(level, summary)`` after
            each level is written, so a caller can record progress.

    Returns:
        Summary dict.
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
    if chunk_scale_factors is not None and len(chunk_scale_factors) != len(factors):
        raise ValueError(
            f"chunk_scale_factors length {len(chunk_scale_factors)} != "
            f"factors length {len(factors)}",
        )
    tolerances = validate_rdp_tolerances(
        rdp_tolerances, n_levels=len(factors), coarsen_mode=coarsen_mode,
    )
    if tolerances is not None:
        # The geometry does not change between levels, so one check here
        # refuses a store the tolerance cannot apply to before level 1 is
        # written, rather than after.
        root_meta = read_root_metadata(open_store(str(store_path), mode="r"))
        refusal = _rdp_tolerance_refusal(
            root_meta, method or select_coarsener_key(root_meta),
        )
        if refusal is not None:
            raise ValueError(f"rdp_tolerances does not apply: {refusal}")

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
        ))
        if on_level_done is not None:
            on_level_done(i + 1, summaries[-1])

    # Compose deeper-delta cross-level links from the inline-emitted +1
    # arrays.  Also stamps root cross-level metadata + the multiscale
    # links capability.
    _finalize_cross_level_for_store(
        store_path,
        cross_level_depth=cross_level_depth,
        cross_level_storage=cross_level_storage,
    )

    return {
        "levels_created": len(summaries),
        "level_specs": summaries,
        "method": COARSEN_PER_OBJECT,
        "cross_level_depth": cross_level_depth,
        "cross_level_storage": cross_level_storage,
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
