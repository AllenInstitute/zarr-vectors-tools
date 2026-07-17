"""Memory-bounded parallel TRX -> zarr-vectors ingest (level 0, faithful).

Converts a large TRX tractogram into a zarr-vectors streamline store without
loading the whole file into RAM, mirroring the phase structure of
:mod:`zarr_vectors_tools.ingest.trk_parallel` but adapted to TRX.

TRX specifics
-------------
- TRX stores ``positions (NB_VERTICES, 3)`` and ``offsets (NB_STREAMLINES,)``
  as separate memmappable arrays, so there is no header record-scan; the
  "offset index" is just the offsets array + derived lengths.
- Positions are in RASMM world space (typically NEGATIVE coordinates). We
  translate positions by ``-lower_bound`` so the stored grid starts at the
  origin and all chunk coordinates are non-negative — REQUIRED for main's
  native sharding (its grid anchors chunk 0 at the origin and cannot
  represent negative chunk coords). The subtracted offset is recorded in the
  store CRS (``coordinate_offset``) so true RASMM is recoverable
  (``rasmm = stored + coordinate_offset``); the VOXEL_TO_RASMM affine +
  DIMENSIONS are stored too. NOTE: neuroglancer renders stored vertex
  coordinates directly (it does not apply the NGFF translation to vertices),
  so the view is in this offset space — shape-identical to RASMM, shifted by
  ``lower_bound``.
- Faithful capture at level 0: ``positions`` -> ``vertices``,
  ``dpv/<name>`` -> ``vertex_attributes/<name>``, ``dps/<name>`` ->
  ``object_attributes/<name>``, ``groups/<name>`` -> ``groups`` +
  ``group_attributes["name"]`` (tract names), ``dpg`` -> ``group_attributes``.
  Plus cross-chunk links + object index for full-streamline reconstruction.

Pipeline (mirrors trk_parallel)
-------------------------------
Phase 0  Open TRX (memmap); header; bounds from positions; grid; offset.
Phase 1  Load offsets, derive lengths; partition streamlines into N parts.
Phase A  Parallel: bin streamlines (offset-applied) -> spatial chunks; write
         .npz + .verts.npy (+ per-dpv sidecars).
Phase B  Parallel per SHARD (native sharding on): batched level-0 write.
Coord    Object index + cross-chunk links; dps/groups/group_attributes; CRS.
Phase 6  Optional multiscale pyramid (:func:`build_pyramid`), same
         coarsen/sparsity machinery as trk_parallel — level-0-only
         ``groups``/``group_attributes`` are not propagated to coarser
         levels (the frontend reads them once, from level 0, regardless
         of which geometry level is displayed).

Requires ``trx-python`` (``pip install trx-python``).
"""

from __future__ import annotations

import json
import math
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import (
    CAP_MULTISCALE_LINKS,
    FRAGMENT_ATTRIBUTES,
    VERTEX_ATTRIBUTES,
    VERTICES,
)
from zarr_vectors.core.arrays import (
    OBJECT_INDEX,
    OBJECT_INDEX_LAYOUT_V1,
    _write_object_index_manifests,
    create_attribute_array,
    create_cross_chunk_links_array,
    create_fragment_attribute_array,
    create_groupings_array,
    create_groupings_attributes_array,
    create_object_attributes_array,
    create_object_index_array,
    create_vertices_array,
    open_write_session,
    write_chunk_attributes,
    write_chunk_fragment_attributes_batch,
    write_chunk_vertices_batch,
    write_cross_chunk_links_bulk,
    write_groupings,
    write_groupings_attributes,
    write_object_attributes,
)
from zarr_vectors.core.metadata import LevelMetadata
from zarr_vectors.core.store import (
    create_resolution_level,
    create_store,
    get_resolution_level,
    open_store,
)
from zarr_vectors.encoding.fragments import encode_object_manifest_blocks
from zarr_vectors.spatial.boundary import split_polyline_at_boundaries

from zarr_vectors_tools.ingest.trk_parallel import (
    _compute_chunk_shape,
    _reduce_level0_oid_shard,
    _sort_seg_rows,
)
from zarr_vectors_tools.multiresolution.coarsen import (
    _stamp_root_capability,
    build_pyramid,
)


# ---------------------------------------------------------------------------
# Phase 0/1 helpers
# ---------------------------------------------------------------------------

def _load_trx(input_path: str | Path):
    try:
        from trx.trx_file_memmap import load as trx_load
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "trx-python is required for TRX ingest. "
            "Install with: pip install trx-python"
        ) from e
    return trx_load(str(input_path))


def _trx_arrays(trx) -> dict[str, Any]:
    header = trx.header
    return {
        "positions": trx.streamlines._data,
        "offsets": np.asarray(trx.streamlines._offsets, dtype=np.int64),
        "lengths": np.asarray(trx.streamlines._lengths, dtype=np.int64),
        "n_vertices": int(header["NB_VERTICES"]),
        "n_streamlines": int(header["NB_STREAMLINES"]),
        "vox_to_rasmm": np.asarray(header["VOXEL_TO_RASMM"], dtype=np.float64),
        "dimensions": [int(x) for x in header["DIMENSIONS"]],
        "dpv_names": list(getattr(trx, "data_per_vertex", {}) or {}),
        "dps_names": list(getattr(trx, "data_per_streamline", {}) or {}),
        "group_names": list(getattr(trx, "groups", {}) or {}),
    }


def _compute_bounds_from_positions(
    positions: npt.NDArray,
) -> tuple[list[float], list[float]]:
    """Streaming min/max over the (memmapped) positions — never materialises a
    float copy of the whole array."""
    n = len(positions)
    if n == 0:
        return ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    block = 8_000_000
    lo = np.full(3, np.inf, dtype=np.float64)
    hi = np.full(3, -np.inf, dtype=np.float64)
    for s in range(0, n, block):
        c = np.asarray(positions[s:s + block], dtype=np.float64)
        lo = np.minimum(lo, c.min(axis=0))
        hi = np.maximum(hi, c.max(axis=0))
    return (lo.tolist(), hi.tolist())


def _partition_streamlines(
    lengths: npt.NDArray, n_parts: int,
) -> list[tuple[int, int]]:
    """Contiguous streamline index ranges balanced by cumulative vertex count."""
    n = len(lengths)
    if n == 0:
        return []
    n_parts = max(1, min(n_parts, n))
    cum = np.concatenate([[0], np.cumsum(lengths)])
    total = int(cum[-1])
    ranges: list[tuple[int, int]] = []
    prev = 0
    for i in range(n_parts):
        if i == n_parts - 1:
            end = n
        else:
            target = int(round(total * (i + 1) / n_parts))
            end = int(np.searchsorted(cum, target, side="left"))
            end = max(prev + 1, min(end, n))
        if end <= prev:
            continue
        ranges.append((prev, end))
        prev = end
        if prev >= n:
            break
    return ranges


# ---------------------------------------------------------------------------
# Phase A: bin one streamline range into spatial chunks (offset applied)
# ---------------------------------------------------------------------------

def _phase_a_worker(part_index: int, shared: dict[str, Any]) -> str:
    trx_path = shared["trx_path"]
    chunk_shape = shared["chunk_shape"]
    intermediate_dir = shared["intermediate_dir"]
    dtype = shared["dtype"]
    dpv_names = shared["dpv_names"]
    ranges = shared["ranges"]
    offset = np.asarray(shared["offset"], dtype=np.float32)  # subtract from positions
    np_dtype = np.dtype(dtype)

    start_sid, end_sid = ranges[part_index]

    trx = _load_trx(trx_path)
    positions = trx.streamlines._data
    offsets = np.asarray(trx.streamlines._offsets, dtype=np.int64)
    n_streamlines_total = len(offsets)
    n_vertices_total = len(positions)
    dpv_arrays = {name: trx.data_per_vertex[name] for name in dpv_names}

    npz_path = str(Path(intermediate_dir) / f"part_{part_index:06d}.npz")
    verts_npy_path = npz_path.replace(".npz", ".verts.npy")

    v_start = int(offsets[start_sid])
    v_end = (
        int(offsets[end_sid]) if end_sid < n_streamlines_total else n_vertices_total
    )
    total_pts = v_end - v_start

    verts_mm = np.lib.format.open_memmap(
        verts_npy_path, mode="w+", dtype=np.float32, shape=(total_pts, 3)
    )
    dpv_mm: dict[str, np.ndarray] = {}
    dpv_ncols: dict[str, int] = {}
    for name in dpv_names:
        arr = dpv_arrays[name]
        ncols = 1 if arr.ndim == 1 else int(arr.shape[1])
        dpv_ncols[name] = ncols
        dpv_mm[name] = np.lib.format.open_memmap(
            npz_path.replace(".npz", f".dpv_{name}.npy"),
            mode="w+", dtype=np.float32, shape=(total_pts, ncols),
        )

    seg_poly_ids: list[int] = []
    seg_chunk_x: list[int] = []
    seg_chunk_y: list[int] = []
    seg_chunk_z: list[int] = []
    seg_vertex_counts: list[int] = []

    write_cursor = 0
    for sid in range(start_sid, end_sid):
        vs = int(offsets[sid])
        ve = (
            int(offsets[sid + 1])
            if sid + 1 < n_streamlines_total
            else n_vertices_total
        )
        verts = np.asarray(positions[vs:ve], dtype=np_dtype) - offset
        dpv_rows = {
            name: np.asarray(dpv_arrays[name][vs:ve], dtype=np.float32).reshape(
                ve - vs, dpv_ncols[name]
            )
            for name in dpv_names
        }

        segments = split_polyline_at_boundaries(verts, chunk_shape)
        seg_v0 = 0
        for cc, seg_verts in segments:
            n = len(seg_verts)
            verts_mm[write_cursor:write_cursor + n] = seg_verts
            for name in dpv_names:
                dpv_mm[name][write_cursor:write_cursor + n] = dpv_rows[name][
                    seg_v0:seg_v0 + n
                ]
            write_cursor += n
            seg_v0 += n
            seg_poly_ids.append(sid)
            seg_chunk_x.append(int(cc[0]))
            seg_chunk_y.append(int(cc[1]))
            seg_chunk_z.append(int(cc[2]))
            seg_vertex_counts.append(n)

    del verts_mm
    for name in dpv_names:
        del dpv_mm[name]

    np.savez_compressed(
        npz_path,
        start_sid=np.int64(start_sid),
        end_sid=np.int64(end_sid),
        seg_poly_ids=np.array(seg_poly_ids, dtype=np.int64),
        seg_chunk_x=np.array(seg_chunk_x, dtype=np.int32),
        seg_chunk_y=np.array(seg_chunk_y, dtype=np.int32),
        seg_chunk_z=np.array(seg_chunk_z, dtype=np.int32),
        seg_vertex_counts=np.array(seg_vertex_counts, dtype=np.int32),
    )
    return npz_path


def _load_part_chunk_index(npz_path: str, dpv_names: list[str]) -> dict[str, Any]:
    data = np.load(npz_path)
    px = data["seg_chunk_x"]
    py = data["seg_chunk_y"]
    pz = data["seg_chunk_z"]
    seg_poly_ids = data["seg_poly_ids"]
    seg_vtx_counts = data["seg_vertex_counts"]
    vtx_starts = np.concatenate([[0], np.cumsum(seg_vtx_counts)])[:-1]

    change_pts = np.concatenate([[0], np.where(np.diff(seg_poly_ids))[0] + 1])
    group_id = (
        np.searchsorted(change_pts, np.arange(len(seg_poly_ids)), side="right") - 1
    )
    within_poly_idx_arr = np.arange(len(seg_poly_ids)) - change_pts[group_id]
    del data

    n = len(px)
    chunk_to_match_idx: dict[tuple[int, int, int], npt.NDArray] = {}
    if n > 0:
        order = np.lexsort((pz, py, px))
        px_s, py_s, pz_s = px[order], py[order], pz[order]
        changed = np.empty(n, dtype=bool)
        changed[0] = True
        changed[1:] = (
            (px_s[1:] != px_s[:-1])
            | (py_s[1:] != py_s[:-1])
            | (pz_s[1:] != pz_s[:-1])
        )
        gs = np.flatnonzero(changed)
        ge = np.append(gs[1:], n)
        for s, e in zip(gs, ge):
            chunk_to_match_idx[(int(px_s[s]), int(py_s[s]), int(pz_s[s]))] = order[s:e]

    return {
        "chunk_to_match_idx": chunk_to_match_idx,
        "seg_poly_ids": seg_poly_ids,
        "within_poly_idx_arr": within_poly_idx_arr,
        "vtx_starts": vtx_starts,
        "seg_vtx_counts": seg_vtx_counts,
        "verts_npy_path": npz_path.replace(".npz", ".verts.npy"),
        "dpv_npy_paths": {
            name: npz_path.replace(".npz", f".dpv_{name}.npy") for name in dpv_names
        },
    }


# ---------------------------------------------------------------------------
# Phase B: assemble one chunk (helper) + write one shard's chunks (worker)
# ---------------------------------------------------------------------------

def _assemble_one_chunk(chunk_coords, per_part, out_dtype, dpv_names):
    """Assemble one chunk's fragments in (part, poly, within) order.

    Returns ``(vert_groups, seg_ids, dpv_groups, seg_rows)`` or ``None``.
    ``dpv_groups`` = ``{name: [per-fragment arrays]}`` aligned with vert_groups.
    """
    records: list[tuple[int, int, int, npt.NDArray, dict[str, npt.NDArray]]] = []
    verts_cache: dict[str, np.ndarray] = {}
    dpv_cache: dict[str, np.ndarray] = {}
    for pp in per_part:
        vpath = pp["verts_npy_path"]
        if vpath not in verts_cache:
            verts_cache[vpath] = np.load(vpath, mmap_mode="r")
        vmm = verts_cache[vpath]
        for name in dpv_names:
            dp = pp["dpv_npy_paths"][name]
            if dp not in dpv_cache:
                dpv_cache[dp] = np.load(dp, mmap_mode="r")
        part_idx = int(pp["part_idx"])
        poly_ids = pp["poly_ids"]
        within = pp["within"]
        vtx_starts = pp["vtx_starts"]
        vtx_counts = pp["vtx_counts"]
        for i in range(len(poly_ids)):
            vst = int(vtx_starts[i])
            vc = int(vtx_counts[i])
            seg_verts = np.array(vmm[vst:vst + vc], dtype=out_dtype)
            seg_dpv = {
                name: np.array(dpv_cache[pp["dpv_npy_paths"][name]][vst:vst + vc])
                for name in dpv_names
            }
            records.append((part_idx, int(poly_ids[i]), int(within[i]), seg_verts, seg_dpv))

    if not records:
        return None
    records.sort(key=lambda r: (r[0], r[1], r[2]))
    vert_groups = [r[3] for r in records]
    seg_ids = np.array([r[1] for r in records], dtype=np.uint64)
    dpv_groups = {name: [r[4][name] for r in records] for name in dpv_names}

    cx, cy, cz = chunk_coords
    seg_rows = np.empty((len(records), 9), dtype=np.int64)
    cum = 0
    for fi, (pidx, pid, wi, sv, _d) in enumerate(records):
        seg_rows[fi] = (pidx, pid, wi, cx, cy, cz, fi, cum, cum + len(sv) - 1)
        cum += len(sv)
    return vert_groups, seg_ids, dpv_groups, seg_rows


def _phase_b_worker(batch: dict[str, Any], shared: dict[str, Any]) -> npt.NDArray:
    """Write ONE shard's worth of chunks (native sharding on).

    A single worker owns a whole shard so two processes never RMW the same
    shard file. Vertices + segment_id use the batched (one-RMW-per-shard)
    writers; dpv attributes are written per chunk — safe because this one
    worker owns every chunk in the shard, so its sequential per-cell writes
    don't race another process.
    """
    store_path = shared["store_path"]
    out_dtype = np.dtype(shared["dtype"])
    dpv_names = shared["dpv_names"]

    with open(batch["spill_path"], "rb") as fh:
        batch_data = pickle.load(fh)  # {chunk_coords: per_part_list}

    root = open_store(str(store_path), mode="r+")
    level_group = get_resolution_level(root, 0)

    coords_to_groups: dict[tuple[int, int, int], list[npt.NDArray]] = {}
    coords_to_seg_ids: dict[tuple[int, int, int], npt.NDArray] = {}
    coords_to_dpv: dict[tuple[int, int, int], dict[str, list[npt.NDArray]]] = {}
    seg_row_blocks: list[npt.NDArray] = []

    for cc in [tuple(int(x) for x in c) for c in batch["chunks"]]:
        per_part = batch_data.get(cc)
        if not per_part:
            continue
        assembled = _assemble_one_chunk(cc, per_part, out_dtype, dpv_names)
        if assembled is None:
            continue
        vert_groups, seg_ids, dpv_groups, seg_rows = assembled
        coords_to_groups[cc] = vert_groups
        coords_to_seg_ids[cc] = seg_ids
        coords_to_dpv[cc] = dpv_groups
        seg_row_blocks.append(seg_rows)

    if coords_to_groups:
        write_chunk_vertices_batch(level_group, coords_to_groups, dtype=out_dtype)
        write_chunk_fragment_attributes_batch(
            level_group, "segment_id", coords_to_seg_ids, dtype=np.uint64,
        )
        for name in dpv_names:
            for cc, dpv_groups in coords_to_dpv.items():
                groups = dpv_groups[name]
                write_chunk_attributes(
                    level_group, name, cc, groups, dtype=groups[0].dtype,
                )

    if seg_row_blocks:
        return np.concatenate(seg_row_blocks, axis=0)
    return np.empty((0, 9), dtype=np.int64)


# ---------------------------------------------------------------------------
# Top-level ingest
# ---------------------------------------------------------------------------

def ingest_trx_parallel(
    input_path: str | Path,
    output_path: str | Path,
    *,
    num_chunks: int | tuple[int, int, int] | None = None,
    n_parts: int | None = None,
    workers: int | None = None,
    executor: Any = None,
    dtype: str = "float32",
    shard_shape: int | tuple[int, int, int] | None = 2,
    max_streamlines: int | None = None,
    build_multiscale: bool = False,
    pyramid_factors: list[tuple[float, float]] | None = None,
    chunk_scale_factors: list[int] | None = None,
    sparsity_strategy: str = "random",
    pyramid_coarsen_mode: str = "decimate",
    keep_intermediate: bool = False,
    random_sample_attr: str | None = "random_sample",
    random_sample_seed: int | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Ingest a TRX file into a zarr-vectors streamline store (level 0).

    Args:
        input_path: Path to the input ``.trx`` file.
        output_path: Output zarr-vectors store directory.
        num_chunks: Target total spatial chunk count (or a ``(nx, ny, nz)``
            override).
        n_parts: Number of Phase A binning parts.
        workers: Advisory worker count, used to size ``n_parts`` when it is
            unset; does not itself control parallelism (``executor`` does).
        executor: Optional ``map``-like ``(func, items, shared) ->
            list[result]`` callable for parallel Phase A/B and pyramid
            coarsening. ``None`` runs everything serially.
        dtype: Vertex dtype.
        shard_shape: Native-sharding shard shape (chunks per shard edge).
        max_streamlines: For quick test runs, ingest only the first N
            streamlines (in on-disk order).
        build_multiscale: Build a coarser multiscale pyramid after level 0,
            via :func:`zarr_vectors_tools.multiresolution.coarsen.build_pyramid`
            — same knobs as :func:`zarr_vectors_tools.ingest.trk_parallel.ingest_trk_parallel`.
        pyramid_factors: List of ``(coarsen_factor, sparsity_factor)``
            tuples, one per coarser level. Required when
            ``build_multiscale=True``.
        chunk_scale_factors: Per-level chunk-shape multiplier, aligned with
            ``pyramid_factors``. ``None`` means all-ones (every level
            inherits the level-0 ``chunk_shape``).
        sparsity_strategy: ``"length"`` (keep longest) or ``"random"`` for
            per-level object dropping.
        pyramid_coarsen_mode: ``"rdp"`` (Douglas-Peucker simplification) or
            ``"decimate"`` (uniform stride decimation).
        keep_intermediate: Keep the Phase A/B scratch directory after
            ingest (useful for debugging); normally cleaned up.
        random_sample_attr: Name of a per-streamline ``object_attributes``
            column to write with one uniform-random float32 in ``[0, 1)``
            per streamline — lets the neuroglancer segment-properties UI
            filter to a random subset (e.g. ``random_sample < 0.1`` for a
            ~10% sample). Written before the pyramid phase so it
            propagates to coarser levels like any other object attribute.
            ``None`` skips it.
        random_sample_seed: Optional seed for reproducible sampling.
        progress: Print phase progress to stdout.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    def _log(msg: str) -> None:
        if progress:
            print(msg, flush=True)

    # --- Phase 0 -------------------------------------------------------
    _log("Phase 0: opening TRX (memmap) + reading header...")
    trx = _load_trx(input_path)
    meta = _trx_arrays(trx)
    positions = meta["positions"]
    offsets = meta["offsets"]
    lengths = meta["lengths"]
    n_streamlines = meta["n_streamlines"]
    dpv_names = meta["dpv_names"]
    dps_names = meta["dps_names"]
    group_names = meta["group_names"]
    _log(
        f"  {n_streamlines} streamlines, {meta['n_vertices']} vertices; "
        f"dpv={dpv_names} dps={dps_names} groups={len(group_names)}"
    )

    rasmm_lo, rasmm_hi = _compute_bounds_from_positions(positions)
    # Translate so the grid starts at the origin (non-negative chunk coords,
    # required for native sharding). Store bounds/positions in this offset
    # space; record the offset for RASMM recovery.
    offset = list(rasmm_lo)
    bounds = ([0.0, 0.0, 0.0], [rasmm_hi[i] - rasmm_lo[i] for i in range(3)])
    chunk_shape = _compute_chunk_shape(bounds, num_chunks)
    grid_shape = tuple(
        max(1, math.ceil(bounds[1][i] / chunk_shape[i])) for i in range(3)
    )
    _log(
        f"  RASMM bounds {rasmm_lo} -> {rasmm_hi}; offset {offset}; "
        f"chunk_shape {chunk_shape}; grid {grid_shape}"
    )

    # --- Phase 1 -------------------------------------------------------
    if max_streamlines is not None and max_streamlines < n_streamlines:
        _log(f"  limiting to first {max_streamlines} of {n_streamlines} streamlines")
        n_streamlines = int(max_streamlines)
        offsets = offsets[:n_streamlines]
        lengths = lengths[:n_streamlines]

    _n_workers = workers if workers and workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    _n_parts = n_parts if n_parts and n_parts > 0 else max(_n_workers * 4, 16)
    ranges = _partition_streamlines(lengths, _n_parts)
    _log(f"  partitioned into {len(ranges)} parts")

    _tmpdir_obj = tempfile.TemporaryDirectory(prefix="trx_parallel_")
    _intermediate_dir = _tmpdir_obj.name

    try:
        # --- Create store ---------------------------------------------
        _log("Creating zarr-vectors store...")
        crs_dict = {
            "input_space": "RASmm",
            "output_space": "RASmm",
            "units": "mm",
            "affine": meta["vox_to_rasmm"].flatten().tolist(),
            "dimensions": meta["dimensions"],
            # stored coord + coordinate_offset == true RASMM world coord.
            "coordinate_offset": [float(x) for x in offset],
        }
        root = create_store(
            str(output_path),
            bounds=bounds,
            chunk_shape=chunk_shape,
            axes=[
                {"name": "x", "type": "space", "unit": "mm"},
                {"name": "y", "type": "space", "unit": "mm"},
                {"name": "z", "type": "space", "unit": "mm"},
            ],
            geometry_types=["streamline"],
            links_convention="implicit_sequential",
            cross_chunk_strategy="explicit_links",
            crs=crs_dict,
        )

        arrays_present = ["vertices", "object_index", "fragment_attributes"]
        if dpv_names:
            arrays_present.append("vertex_attributes")
        level_meta = LevelMetadata(
            level=0, vertex_count=0, arrays_present=arrays_present,
        )
        level_group = create_resolution_level(root, 0, level_meta)

        # bounds are non-negative -> native sharding enabled.
        level_shard_shape = shard_shape
        with open_write_session(
            level_group, shard_shape=level_shard_shape,
            bounds=bounds, chunk_shape=chunk_shape,
        ):
            create_vertices_array(level_group, dtype=dtype, compress=False)
            create_object_index_array(level_group)
            create_cross_chunk_links_array(level_group, delta=0, sid_ndim=3)
            create_fragment_attribute_array(level_group, "segment_id", dtype="uint64")
            for name in dpv_names:
                create_attribute_array(level_group, name, dtype="float32")

        # --- Phase A --------------------------------------------------
        _log(f"Phase A: binning streamlines ({len(ranges)} parts)...")
        shared_a = {
            "trx_path": str(input_path),
            "chunk_shape": chunk_shape,
            "intermediate_dir": _intermediate_dir,
            "dtype": dtype,
            "dpv_names": dpv_names,
            "ranges": ranges,
            "offset": offset,
        }
        part_indices = list(range(len(ranges)))
        if executor is not None:
            part_npz_paths = executor(_phase_a_worker, part_indices, shared=shared_a)
        else:
            part_npz_paths = [_phase_a_worker(i, shared_a) for i in part_indices]
        _log(f"  wrote {len(part_npz_paths)} intermediate files")

        part_loaded = [_load_part_chunk_index(p, dpv_names) for p in part_npz_paths]
        all_chunk_coords: set[tuple[int, int, int]] = set()
        for pidx in part_loaded:
            all_chunk_coords.update(pidx["chunk_to_match_idx"].keys())
        _log(f"  {len(all_chunk_coords)} occupied spatial chunks")

        # --- Phase B: per-shard batched write -------------------------
        _log(f"Phase B: writing level-0 ({len(all_chunk_coords)} chunks)...")
        shard_shape_3 = (
            (level_shard_shape,) * 3
            if isinstance(level_shard_shape, int)
            else tuple(level_shard_shape)
        )
        shard_groups: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
        for cc in sorted(all_chunk_coords):
            addr = tuple(cc[i] // shard_shape_3[i] for i in range(3))
            shard_groups.setdefault(addr, []).append(cc)
        chunk_lists = [shard_groups[k] for k in sorted(shard_groups)]

        phaseb_spill_dir = tempfile.mkdtemp(prefix="trx_phaseb_")
        chunk_batches: list[dict[str, Any]] = []
        for bi, chunks in enumerate(chunk_lists):
            batch_data: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
            for cc in chunks:
                per_part: list[dict[str, Any]] = []
                for part_idx, pidx in enumerate(part_loaded):
                    mi = pidx["chunk_to_match_idx"].get(cc)
                    if mi is None or len(mi) == 0:
                        continue
                    per_part.append({
                        "part_idx": part_idx,
                        "verts_npy_path": pidx["verts_npy_path"],
                        "dpv_npy_paths": pidx["dpv_npy_paths"],
                        "poly_ids": pidx["seg_poly_ids"][mi],
                        "within": pidx["within_poly_idx_arr"][mi],
                        "vtx_starts": pidx["vtx_starts"][mi],
                        "vtx_counts": pidx["seg_vtx_counts"][mi],
                    })
                if per_part:
                    batch_data[cc] = per_part
            spill_path = os.path.join(phaseb_spill_dir, f"pb_{bi}.pkl")
            with open(spill_path, "wb") as fh:
                pickle.dump(batch_data, fh, protocol=pickle.HIGHEST_PROTOCOL)
            chunk_batches.append({"chunks": chunks, "spill_path": spill_path})
        del part_loaded

        shared_b = {"store_path": str(output_path), "dtype": dtype, "dpv_names": dpv_names}
        try:
            if executor is not None:
                phase_b_results = executor(_phase_b_worker, chunk_batches, shared=shared_b)
            else:
                phase_b_results = [_phase_b_worker(b, shared_b) for b in chunk_batches]
        finally:
            shutil.rmtree(phaseb_spill_dir, ignore_errors=True)

        # Repair the racy nonempty_chunks manifest once, from the coordinator.
        level_group.set_nonempty_chunks(VERTICES, all_chunk_coords)
        level_group.set_nonempty_chunks(
            f"{FRAGMENT_ATTRIBUTES}/segment_id", all_chunk_coords
        )
        for name in dpv_names:
            level_group.set_nonempty_chunks(
                f"{VERTEX_ATTRIBUTES}/{name}", all_chunk_coords
            )

        # --- Coordinator: object index + cross-chunk links ------------
        _log("Coordinator: object index + cross-chunk links...")
        seg = _sort_seg_rows(phase_b_results)
        del phase_b_results

        n_oid_shards = (
            max(1, min(4096, -(-n_streamlines // 20_000))) if n_streamlines else 1
        )
        oid_bounds = np.linspace(0, n_streamlines, n_oid_shards + 1).astype(np.int64)
        row_bounds = (
            np.searchsorted(seg[:, 1], oid_bounds)
            if len(seg)
            else np.zeros(n_oid_shards + 1, dtype=np.int64)
        )
        oid_reduce_dir = tempfile.mkdtemp(prefix="trx_oidreduce_")
        link_spill_dir = tempfile.mkdtemp(prefix="trx_crosslink_")
        cross_chunk_link_count = 0
        try:
            reduce_payloads: list[dict[str, Any]] = []
            for i in range(n_oid_shards):
                lo_r, hi_r = int(row_bounds[i]), int(row_bounds[i + 1])
                if hi_r <= lo_r:
                    continue
                seg_path = os.path.join(oid_reduce_dir, f"seg_{i}.npy")
                np.save(seg_path, seg[lo_r:hi_r])
                reduce_payloads.append({
                    "seg_path": seg_path,
                    "link_spill_path": os.path.join(link_spill_dir, f"links_{i}.npy"),
                })
            del seg

            shared_reduce = {"sid_ndim": 3}
            if executor is not None:
                reduce_results = executor(
                    _reduce_level0_oid_shard, reduce_payloads, shared=shared_reduce,
                )
            else:
                reduce_results = [
                    _reduce_level0_oid_shard(p, shared_reduce) for p in reduce_payloads
                ]

            empty_blob = encode_object_manifest_blocks([], sid_ndim=3)
            manifest_blobs: list[bytes] = [empty_blob] * n_streamlines
            link_paths: list[str] = []
            for rc in reduce_results:
                oids = rc["oid"]
                if len(oids):
                    blobs = pickle.loads(rc["blobs"])
                    for i, oid in enumerate(oids.tolist()):
                        if 0 <= int(oid) < n_streamlines:
                            manifest_blobs[int(oid)] = blobs[i]
                if rc.get("link_path"):
                    link_paths.append(rc["link_path"])
                    cross_chunk_link_count += int(rc.get("n_links", 0))
            del reduce_results
        finally:
            shutil.rmtree(oid_reduce_dir, ignore_errors=True)

        _log(f"  writing object index ({n_streamlines} streamlines)...")
        _write_object_index_manifests(level_group, manifest_blobs)
        level_group.write_array_meta(OBJECT_INDEX, {
            "zv_array": "object_index",
            "num_objects": n_streamlines,
            "sid_ndim": 3,
            "layout": OBJECT_INDEX_LAYOUT_V1,
        })
        del manifest_blobs

        try:
            if cross_chunk_link_count:
                _log(f"  writing {cross_chunk_link_count} cross-chunk links...")
                all_link_rows = np.concatenate(
                    [np.load(p, allow_pickle=False) for p in link_paths], axis=0,
                )
                write_cross_chunk_links_bulk(level_group, all_link_rows, sid_ndim=3, delta=0)
                del all_link_rows
                _stamp_root_capability(root, CAP_MULTISCALE_LINKS)
        finally:
            shutil.rmtree(link_spill_dir, ignore_errors=True)

        # --- dps -> object_attributes ---------------------------------
        for name in dps_names:
            data = np.asarray(trx.data_per_streamline[name])
            if data.ndim == 2 and data.shape[1] == 1:
                data = data[:, 0]
            data = data[:n_streamlines]
            create_object_attributes_array(level_group, name)
            write_object_attributes(level_group, name, data)

        # --- random-sample object attribute ----------------------------
        if random_sample_attr:
            rng = np.random.default_rng(random_sample_seed)
            sample_data = rng.random(n_streamlines, dtype=np.float32)
            create_object_attributes_array(level_group, random_sample_attr)
            write_object_attributes(level_group, random_sample_attr, sample_data)

        # --- groups + group_attributes --------------------------------
        group_count = _write_groups(level_group, trx, group_names, n_streamlines, _log)

        total_vertices = sum(
            int(np.load(p)["seg_vertex_counts"].sum()) for p in part_npz_paths
        )
        _restamp_level0_vertex_count(output_path, total_vertices)

        # --- Phase 6: multiscale pyramid -------------------------------
        pyramid_summary: dict[str, Any] = {}
        if build_multiscale:
            _log("Phase 6: building multiscale pyramid...")
            _pyramid_factors = pyramid_factors or [(8.0, 1.0), (8.0, 1.0)]
            _chunk_scale_factors = chunk_scale_factors or [1] * len(_pyramid_factors)
            pyramid_summary = build_pyramid(
                str(output_path),
                factors=_pyramid_factors,
                chunk_scale_factors=_chunk_scale_factors,
                sparsity_strategy=sparsity_strategy,
                coarsen_mode=pyramid_coarsen_mode,
                executor=executor,
            )

        _log("Done.")
        return {
            "streamline_count": n_streamlines,
            "vertex_count": total_vertices,
            "chunk_count": len(all_chunk_coords),
            "cross_chunk_link_count": cross_chunk_link_count,
            "group_count": group_count,
            "dpv_names": dpv_names,
            "dps_names": dps_names,
            "n_parts": len(ranges),
            "chunk_shape": chunk_shape,
            "rasmm_bounds": (rasmm_lo, rasmm_hi),
            "coordinate_offset": offset,
            "random_sample_attr": random_sample_attr,
            "pyramid": pyramid_summary,
        }
    finally:
        if not keep_intermediate:
            _tmpdir_obj.cleanup()


def _write_groups(level_group, trx, group_names, n_streamlines, _log) -> int:
    if not group_names:
        return 0
    _log(f"  writing {len(group_names)} groups + attributes...")
    groups: dict[int, list[int]] = {}
    for gid, gname in enumerate(group_names):
        idx = np.asarray(trx.groups[gname], dtype=np.int64)
        idx = idx[idx < n_streamlines]  # honor max_streamlines truncation
        groups[gid] = idx.tolist()
    create_groupings_array(level_group)
    write_groupings(level_group, groups)

    # Tract names as a string group_attribute (round-trips as numpy <U..).
    try:
        str_arr = np.asarray(group_names, dtype=str)
        create_groupings_attributes_array(level_group, "name", str(str_arr.dtype), 1)
        write_groupings_attributes(level_group, "name", str_arr)
    except Exception:
        from zarr_vectors.constants import GROUPS
        level_group.write_array_meta(
            GROUPS,
            {"zv_array": "groups", "group_names_json": json.dumps(list(group_names))},
        )

    # data_per_group -> numeric group_attributes columns.
    dpg = getattr(trx, "data_per_group", None) or {}
    attr_names: list[str] = []
    for gname in group_names:
        for aname in (dpg.get(gname, {}) or {}):
            if aname not in attr_names:
                attr_names.append(aname)
    for aname in attr_names:
        rows = []
        for gname in group_names:
            v = (dpg.get(gname, {}) or {}).get(aname)
            rows.append(
                np.asarray(v, dtype=np.float32).ravel()
                if v is not None else np.array([np.nan], np.float32)
            )
        width = max(len(r) for r in rows)
        mat = np.full((len(group_names), width), np.nan, dtype=np.float32)
        for gi, r in enumerate(rows):
            mat[gi, :len(r)] = r
        col = mat[:, 0] if width == 1 else mat
        create_groupings_attributes_array(
            level_group, aname, "float32", 1 if width == 1 else width,
        )
        write_groupings_attributes(level_group, aname, col)
    return len(group_names)


def _restamp_level0_vertex_count(output_path, total_vertices) -> None:
    try:
        import zarr as _zarr
        root_z = _zarr.open(str(output_path), mode="r+")
        lvl0 = root_z["0"]
        attrs = dict(lvl0.attrs)
        if "zarr_vectors_level" in attrs:
            attrs["zarr_vectors_level"]["vertex_count"] = int(total_vertices)
            attrs["zarr_vectors_level"]["arrays_present"] = [
                k for k in [
                    "vertices", "vertex_fragments", "object_index",
                    "cross_chunk_links", "object_attributes",
                    "fragment_attributes", "vertex_attributes", "groups",
                    "group_attributes",
                ] if k in lvl0
            ]
        lvl0.attrs.update(attrs)
    except Exception:
        pass
