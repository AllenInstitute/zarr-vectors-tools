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

from pathlib import Path
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import (
    CAP_PRESERVED_OBJECT_IDS,
    CAP_SHARED_FRAGMENTS,
    COARSEN_PER_OBJECT,
    DEFAULT_CROSS_LEVEL_DEPTH,
    DEFAULT_CROSS_LEVEL_STORAGE,
    LINKS,
    LINKS_IMPLICIT_BRANCHES,
    LINKS_IMPLICIT_SEQUENTIAL,
    OBJECT_ATTRIBUTES,
    VERTICES,
    XLEVEL_EXPLICIT,
    XLEVEL_NONE,
    VALID_XLEVEL_STORAGE,
)
from zarr_vectors.core.arrays import (
    create_cross_chunk_links_array,
    create_links_array,
    create_object_attributes_array,
    create_object_index_array,
    create_vertices_array,
    list_chunk_keys,
    read_all_object_manifests,
    read_chunk_links,
    read_chunk_vertices,
    read_cross_chunk_links,
    read_object_attributes,
    write_chunk_links,
    write_chunk_vertices,
    write_cross_chunk_links,
    write_object_attributes,
    write_object_index,
)
from zarr_vectors.core.metadata import (
    LevelMetadata,
    get_level_chunk_shape,
)
from zarr_vectors.core.store import (
    create_resolution_level,
    get_resolution_level,
    list_resolution_levels,
    open_store,
    read_level_metadata,
    read_root_metadata,
)
from zarr_vectors_tools.multiresolution._ccl_compat import stamp_ccl_capabilities
from zarr_vectors.exceptions import ArrayError, CoarseningError
from zarr_vectors_tools.multiresolution.coarsen_implicit import (
    coarse_chunks_of,
    positions_in_run,
    segment_object_by_coarse_chunk,
)
from zarr_vectors_tools.multiresolution.object_selection import apply_sparsity
from zarr_vectors.spatial.boundary import (
    build_vertex_chunk_mapping,
    cross_chunk_links_for_segments,
    partition_cross_level_edges,
)
from zarr_vectors.spatial.chunking import assign_chunks
from zarr_vectors.typing import ChunkCoords


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
    skeleton-aware decimator; streamline stores (``implicit_sequential`` with
    geometry_type ``"streamline"``) use the RDP polyline coarsener; everything
    else uses the per-object pyramid.
    """
    if root_meta.links_convention == LINKS_IMPLICIT_BRANCHES:
        return "skeleton"
    if (
        root_meta.links_convention == LINKS_IMPLICIT_SEQUENTIAL
        and "streamline" in (root_meta.geometry_types or [])
    ):
        return "polyline"
    return "per_object"


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
    executor: Any = None,
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
        coarsen_factor: Per-object vertex aggregation factor (≥ 1).
            ``1.0`` is the identity (no aggregation).
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
        executor: Optional ``map``-like ``(func, items, shared) ->
            list[result]`` callable (e.g. ``dask_executor``) used by the
            chunk-local skeleton and polyline coarseners to parallelize
            per-target-chunk work.  ``None`` (default) runs serially
            in-process.  Ignored by the per-object coarsener.

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
    coarsener = get_coarsener(select_coarsener_key(root_meta))
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
        executor=executor,
    )


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
    base_bin = root_meta.effective_bin_shape

    # Source level's chunk_shape — may itself be a per-level override.
    try:
        src_level_meta = read_level_metadata(root, source_level)
    except Exception:
        src_level_meta = None
    source_chunk_shape = get_level_chunk_shape(root_meta, src_level_meta)

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
    src_fragment_positions: dict[tuple[ChunkCoords, int], npt.NDArray] = {}
    for cc in list_chunk_keys(src_group, VERTICES):
        try:
            fragments = read_chunk_vertices(src_group, cc, dtype=np.float32, ndim=ndim)
        except ArrayError:
            continue
        for fragment_idx, fragment in enumerate(fragments):
            src_fragment_positions[(cc, fragment_idx)] = fragment

    src_has_objects = "object_index" in src_group
    if src_has_objects:
        src_manifests = read_all_object_manifests(src_group)
    else:
        # No object_index — treat the level as one implicit object whose
        # manifest enumerates every fragment in chunk-major order.
        implicit: list[tuple[ChunkCoords, int]] = []
        for cc in list_chunk_keys(src_group, VERTICES):
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
        kept = apply_sparsity(
            n_src_objects, keep_frac, sparsity_strategy,
            seed=sparsity_seed,
            representative_points=None,
            bin_shape=base_bin,
            alive_mask=alive_mask,
        )
        keep_oids = sorted(int(o) for o in kept)
    else:
        keep_oids = list(range(n_src_objects))
    keep_set = set(keep_oids)

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

    # Target bin shape: source bin_shape × coarsen_factor.
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
    meta_positions = np.zeros((n_metavertices, ndim), dtype=np.float32)
    bin_counts = np.zeros(n_metavertices, dtype=np.int64)
    np.add.at(meta_positions, inverse, all_pos)
    np.add.at(bin_counts, inverse, 1)
    meta_positions /= bin_counts[:, None]

    # --- Step 4: chunk-assign metavertices ------------------------------
    chunk_assignments = assign_chunks(meta_positions, chunk_shape)

    # --- Step 5-9b: branch on links_convention --------------------------
    # The ``implicit_sequential`` (streamline / polyline) path keeps each
    # object's path in a single multi-vertex fragment per coarsened chunk,
    # so consecutive metavertices belong to the same fragment and their
    # implicit edges encode the connectivity.  Other conventions stay on
    # the legacy "one fragment per metavertex" layout where Step 9b
    # records same-chunk bridges as ``cross_chunk_links/0`` rows.
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
        src_chunk_fragment_starts: dict[ChunkCoords, dict[int, int]] = {}
        src_chunks_seen = {c for (c, _) in src_fragment_positions.keys()}
        for cc in src_chunks_seen:
            fids = sorted(
                fid for (c, fid) in src_fragment_positions.keys() if c == cc
            )
            starts_map: dict[int, int] = {}
            cum = 0
            for fid in fids:
                starts_map[fid] = cum
                cum += len(src_fragment_positions[(cc, fid)])
            src_chunk_fragment_starts[cc] = starts_map

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
        bin_ratio=tuple(max(1, int(round(coarsen_factor))) for _ in range(ndim)),
        chunk_shape=target_chunk_shape_override,
        object_sparsity=(1.0 / sparsity_factor),
        coarsening_method=COARSEN_PER_OBJECT,
        parent_level=source_level,
        preserves_object_ids=src_has_objects,
        inherited_num_objects=n_src_objects if src_has_objects else 0,
        shared_fragments=shared_fragments_flag,
    )
    level_group = create_resolution_level(root, target_level, level_meta_initial)
    create_vertices_array(level_group, dtype="float32")
    if src_has_objects:
        create_object_index_array(level_group)

    for cc, groups in sorted(per_chunk_groups.items()):
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

    # --- Step 9b: cross_chunk_links/0 ----------------------------------
    if use_implicit_sequential:
        # Pass 2: remap source-level ``cross_chunk_links/0`` records to
        # the new coarse-chunk-local indices.  Drop records whose
        # endpoints both fell into the same coarsened chunk — those were
        # absorbed by Pass 1's merged fragments.
        src_cross_records = read_cross_chunk_links(src_group, delta=0)
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
        create_cross_chunk_links_array(level_group, delta=0, sid_ndim=ndim)
        if new_cross_links:
            write_cross_chunk_links(
                level_group, new_cross_links, sid_ndim=ndim, delta=0,
                directed=True,
            )
    else:
        # Legacy Step 9b: one fragment per metavertex, so consecutive
        # same-chunk manifest entries are bridged via cross_chunk_links/0
        # (with delta=0 same-chunk records being intentional).
        cross_links = []
        for oid, manifest in new_manifests.items():
            if len(manifest) < 2:
                continue
            for i in range(len(manifest) - 1):
                cc_a, frag_a = manifest[i]
                cc_b, frag_b = manifest[i + 1]
                # vi_a == frag_a, vi_b == frag_b (one metavertex per fragment).
                cross_links.append(((cc_a, frag_a), (cc_b, frag_b)))
        create_cross_chunk_links_array(level_group, delta=0, sid_ndim=ndim)
        if cross_links:
            write_cross_chunk_links(
                level_group, cross_links, sid_ndim=ndim, delta=0,
            )

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
    level_group.rebuild_nonempty_manifests()

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
    """Emit ``±1`` link/cross_chunk_link arrays for one coarsen step.

    Re-walks the source level in chunk-major order, re-bins each
    vertex against ``bin_shape_arr``, and looks up the matching
    metavertex via the ``bin_key`` ↔ ``mv_idx`` map implicit in
    ``np.unique(bin_keys, return_inverse=inverse)``.  Translates
    metavertex IDs to chunk-major-flat coarse indices via the
    just-written coarse-level chunks, then dispatches to
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
    # bin_key_bytes → mv_idx (bin-key-ordered, matches np.unique output).
    unique_keys = np.unique(bin_keys)
    bin_key_to_mv: dict[bytes, int] = {
        bytes(k): i for i, k in enumerate(unique_keys)
    }

    # mv_idx → chunk-major-flat coarse index.
    coarse_chunk_assignments, n_coarse = _reconstruct_chunk_assignments(
        level_group, ndim,
    )
    mv_to_coarse_global: dict[int, int] = {}
    if mv_first_row_chunk is not None and mv_first_row_local is not None:
        for mv_idx, cc in mv_first_row_chunk.items():
            local_row = mv_first_row_local[mv_idx]
            chunk_rows = coarse_chunk_assignments.get(cc)
            if chunk_rows is None or local_row >= len(chunk_rows):
                continue
            mv_to_coarse_global[int(mv_idx)] = int(chunk_rows[local_row])
    elif coarse_chunk_assignments_mv is not None:
        for cc, mv_indices_for_chunk in sorted(coarse_chunk_assignments_mv.items()):
            for local_vg, mv_idx in enumerate(mv_indices_for_chunk.tolist()):
                mv_to_coarse_global[int(mv_idx)] = int(
                    coarse_chunk_assignments[cc][local_vg]
                )
    else:
        raise ValueError(
            "Either coarse_chunk_assignments_mv or "
            "(mv_first_row_chunk, mv_first_row_local) must be supplied",
        )

    # Build fine→coarse parent[] by re-walking source in chunk-major order.
    fine_chunk_assignments, n_fine = _reconstruct_chunk_assignments(
        src_group, ndim,
    )
    parent = np.full(n_fine, -1, dtype=np.int64)
    cursor = 0
    key_dtype = np.dtype((
        np.void, int(bin_shape_arr.shape[0]) * np.dtype(np.int64).itemsize,
    ))
    for cc in list_chunk_keys(src_group, VERTICES):
        try:
            fragments = read_chunk_vertices(
                src_group, cc, dtype=np.float32, ndim=ndim,
            )
        except ArrayError:
            continue
        for fragment in fragments:
            n_local = int(fragment.shape[0])
            if n_local == 0:
                continue
            local_bins = np.floor(
                np.asarray(fragment, dtype=np.float32) / bin_shape_arr,
            ).astype(np.int64)
            local_keys = np.ascontiguousarray(local_bins).view(key_dtype).ravel()
            for j in range(n_local):
                mv = bin_key_to_mv.get(bytes(local_keys[j]))
                if mv is not None:
                    parent[cursor + j] = mv_to_coarse_global[int(mv)]
            cursor += n_local

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
    coarsen_factor: float,
    sparsity_factor: float,
    inherited_num_objects: int,
) -> None:
    """Write an empty ID-preserving level when no surviving object has vertices."""
    ndim = len(base_bin)
    target_bin_shape = tuple(float(b) * float(coarsen_factor) for b in base_bin)
    level_meta = LevelMetadata(
        level=target_level,
        vertex_count=0,
        arrays_present=[VERTICES, "object_index"],
        bin_shape=target_bin_shape,
        bin_ratio=tuple(max(1, int(round(coarsen_factor))) for _ in range(ndim)),
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
    """Add ``cap`` to root metadata's ``format_capabilities`` (idempotent)."""
    attrs = root_group.attrs.to_dict()
    zv = attrs.get("zarr_vectors", {})
    caps = list(zv.get("format_capabilities", []))
    if cap not in caps:
        caps.append(cap)
        zv["format_capabilities"] = caps
        root_group.attrs.update({"zarr_vectors": zv})


def _stamp_root_cross_level(
    root_group, *, depth: int, storage: str,
) -> None:
    """Persist cross_level_depth/cross_level_storage on root metadata."""
    attrs = root_group.attrs.to_dict()
    zv = attrs.get("zarr_vectors", {})
    zv["cross_level_depth"] = int(depth)
    zv["cross_level_storage"] = storage
    root_group.attrs.update({"zarr_vectors": zv})


def _reconstruct_chunk_assignments(
    level_group, ndim: int,
) -> tuple[dict[ChunkCoords, npt.NDArray[np.int64]], int]:
    """Rebuild ``{chunk_coords: vertex_indices}`` from on-disk vertex chunks.

    The "vertex index" assigned to each vertex is the position it would
    occupy in a flat enumeration that walks chunks in
    ``list_chunk_keys`` order and concatenates each chunk's vertex
    groups in order.  This matches the convention used by
    ``build_vertex_chunk_mapping`` for in-memory edge partitioning.

    Returns the assignments dict and the total vertex count.
    """
    chunk_keys = list_chunk_keys(level_group, VERTICES)
    assignments: dict[ChunkCoords, npt.NDArray[np.int64]] = {}
    cursor = 0
    for cc in chunk_keys:
        try:
            fragments = read_chunk_vertices(level_group, cc, dtype=np.float32, ndim=ndim)
        except ArrayError:
            continue
        n = sum(int(fragment.shape[0]) for fragment in fragments)
        if n == 0:
            continue
        assignments[cc] = np.arange(cursor, cursor + n, dtype=np.int64)
        cursor += n
    return assignments, cursor


def _decode_parent_from_plus_one(
    fine_lg,
    *,
    fine_assn: dict[ChunkCoords, npt.NDArray[np.int64]],
    coarse_assn: dict[ChunkCoords, npt.NDArray[np.int64]],
    n_fine: int,
) -> npt.NDArray[np.int64] | None:
    """Decode a fine→coarse ``parent`` array from already-written ``+1`` arrays.

    Reads ``links/<+1>/<chunk_key>`` (intra-chunk edges) and
    ``cross_chunk_links/<+1>/`` (cross-chunk edges) at the fine level
    and converts each ``(chunk, local_idx)`` pair to global flat indices
    via the supplied chunk-assignment dicts.  Returns ``None`` when
    neither array exists.
    """
    parent = np.full(n_fine, -1, dtype=np.int64)
    found_any = False

    # Aligned (intra-chunk) edges: read each chunk in links/+1/.
    try:
        chunk_keys = list_chunk_keys(fine_lg, f"{LINKS}/+1")
    except (ArrayError, KeyError):
        chunk_keys = []
    for cc in chunk_keys:
        try:
            link_groups = read_chunk_links(fine_lg, cc, delta=1)
        except ArrayError:
            continue
        for rows in link_groups:
            if rows is None or len(rows) == 0:
                continue
            local_src = rows[:, 0].astype(np.int64)
            local_tgt = rows[:, 1].astype(np.int64)
            fine_global = fine_assn[cc][local_src]
            coarse_global = coarse_assn[cc][local_tgt]
            parent[fine_global] = coarse_global
            found_any = True

    # Cross-chunk edges.
    try:
        records = read_cross_chunk_links(fine_lg, delta=1)
    except (ArrayError, KeyError):
        records = []
    for (cc_s, vi_s), (cc_t, vi_t) in records:
        parent[int(fine_assn[cc_s][vi_s])] = int(coarse_assn[cc_t][vi_t])
        found_any = True

    return parent if found_any else None


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
    _stamp_root_cross_level(
        root, depth=cross_level_depth, storage=cross_level_storage,
    )
    if cross_level_storage == XLEVEL_NONE or cross_level_depth == 0:
        return

    meta = read_root_metadata(root)
    ndim = meta.sid_ndim
    levels = sorted(list_resolution_levels(root))
    if len(levels) < 2:
        return

    stamp_ccl_capabilities(root)

    # Build per-level chunk_assignments + total counts once.
    per_level: dict[int, tuple[dict[ChunkCoords, npt.NDArray[np.int64]], int]] = {}
    for lvl in levels:
        lg = get_resolution_level(root, lvl)
        per_level[lvl] = _reconstruct_chunk_assignments(lg, ndim)

    max_delta = (
        max(levels) - min(levels)
        if cross_level_depth == -1
        else int(cross_level_depth)
    )
    if max_delta < 2:
        return  # +1/-1 was already emitted inline

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

    # Build chunk-mapping tables for both levels.
    fine_chunk_list = sorted(fine_chunk_assignments.keys())
    fine_vchunks, fine_vlocal, fine_chunk_list = build_vertex_chunk_mapping(
        fine_chunk_assignments, n_fine, fine_chunk_list,
    )
    coarse_chunk_list = sorted(coarse_chunk_assignments.keys())
    coarse_vchunks, coarse_vlocal, coarse_chunk_list = build_vertex_chunk_mapping(
        coarse_chunk_assignments, n_coarse, coarse_chunk_list,
    )

    # Trivial fine→parent edge list.
    edges = np.stack([fine_global, parent_valid], axis=1)
    aligned, cross = partition_cross_level_edges(
        edges,
        fine_vchunks, fine_vlocal, fine_chunk_list,
        coarse_vchunks, coarse_vlocal, coarse_chunk_list,
    )

    fine_lg = get_resolution_level(root_group, fine_level)
    if aligned:
        create_links_array(fine_lg, link_width=2, delta=delta)
        for cc, rows in aligned.items():
            write_chunk_links(fine_lg, cc, [rows], delta=delta)
    if cross:
        create_cross_chunk_links_array(fine_lg, delta=delta)
        write_cross_chunk_links(fine_lg, cross, sid_ndim=sid_ndim, delta=delta)

    if storage == XLEVEL_EXPLICIT:
        # Mirror at the coarse level under -delta: swap endpoint roles.
        coarse_lg = get_resolution_level(root_group, coarse_level)
        # Re-partition from the coarse side so chunk-alignment is
        # evaluated against the coarse chunk grid (intra/cross split
        # may differ from the fine-side view when grids don't align).
        rev_edges = np.stack([parent_valid, fine_global], axis=1)
        rev_aligned, rev_cross = partition_cross_level_edges(
            rev_edges,
            coarse_vchunks, coarse_vlocal, coarse_chunk_list,
            fine_vchunks, fine_vlocal, fine_chunk_list,
        )
        if rev_aligned:
            create_links_array(coarse_lg, link_width=2, delta=-delta)
            for cc, rows in rev_aligned.items():
                write_chunk_links(coarse_lg, cc, [rows], delta=-delta)
        if rev_cross:
            create_cross_chunk_links_array(coarse_lg, delta=-delta)
            write_cross_chunk_links(
                coarse_lg, rev_cross, sid_ndim=sid_ndim, delta=-delta,
            )


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
    executor: Any = None,
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
            one per coarser level.
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

    Returns:
        Summary dict.
    """
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

    summaries: list[dict[str, Any]] = []
    for i, fac in enumerate(factors):
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
            sparsity_seed=sparsity_seed,
            cross_level_storage=cross_level_storage,
            coarsen_mode=coarsen_mode,
            executor=executor,
        ))

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
    executor: Any = None,
) -> dict[str, Any]:
    """Skeleton stores: route to the skeleton-aware decimator.  ``coarsen_factor``
    is the decimation stride, ``chunk_scale_factor`` defaults to 2, and the
    random sparsity strategy degrades to deterministic ``"length"``.
    ``coarsen_mode`` is accepted for signature parity with other coarseners
    but has no effect here — this coarsener always decimates."""
    from zarr_vectors_tools.multiresolution.strategies.skeletons import (
        coarsen_skeleton_level,
    )
    csf = chunk_scale_factor if chunk_scale_factor != 1 else 2
    return coarsen_skeleton_level(
        store_path, source_level, target_level,
        stride=max(1, int(round(coarsen_factor))),
        sparsity_factor=sparsity_factor,
        chunk_scale_factor=csf,
        sparsity_strategy=(
            sparsity_strategy if sparsity_strategy != "random" else "length"
        ),
        sparsity_seed=sparsity_seed,
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
    executor: Any = None,
) -> dict[str, Any]:
    """Geometry-preserving polyline/streamline coarsener (RDP simplification
    or uniform stride decimation, selected by ``coarsen_mode``).

    Chunk-local and executor-parallel — see
    :func:`zarr_vectors_tools.multiresolution.strategies.polylines.coarsen_polyline_level`
    for the implementation (peak memory O(one target chunk), not O(the whole
    source level))."""
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
        executor=executor,
    )


register_coarsener("skeleton", _skeleton_coarsener)
register_coarsener("per_object", _per_object_coarsener)
register_coarsener("polyline", _polyline_coarsener)
