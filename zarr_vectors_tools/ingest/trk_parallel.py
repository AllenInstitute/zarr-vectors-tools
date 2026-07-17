"""Memory-bounded parallel TRK → zarr-vectors ingest.

Processes large TRK (TrackVis) files without loading them whole into RAM.
Coordinate convention: stored as-is in "voxmm" space (voxel_index × voxel_size).
The vox_to_ras affine is stored in the store's CRS metadata for downstream use.

Pipeline
--------
Phase 0  Parse 1000-byte header; derive spatial grid from ``num_chunks``.
Phase 1  Offset-index scan (~17 s for 5M streamlines); partition into N parts.
Phase A  Parallel over N parts: bin streamlines → spatial chunks, write .npz.
Phase B  Parallel over S chunks: assemble from all N parts → write level-0.
Coord    Single-process: reconstruct manifests + cross-chunk links.
Phase 5  Store CRS/affine metadata + TRKHeader for round-trip.
Phase 6  Build multiscale pyramid via coarsen.build_pyramid.
"""

from __future__ import annotations

import math
import os
import pickle
import shutil
import struct
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import CAP_MULTISCALE_LINKS, FRAGMENT_ATTRIBUTES, VERTICES
from zarr_vectors.core.arrays import (
    OBJECT_INDEX,
    OBJECT_INDEX_LAYOUT_V1,
    _write_object_index_manifests,
    create_cross_chunk_links_array,
    create_fragment_attribute_array,
    create_object_index_array,
    create_vertices_array,
    open_write_session,
    write_chunk_fragment_attributes_batch,
    write_chunk_vertices_batch,
    write_cross_chunk_links_bulk,
)
from zarr_vectors.encoding.fragments import encode_object_manifest_blocks
from zarr_vectors.core.metadata import LevelMetadata
from zarr_vectors.core.store import create_resolution_level, create_store, open_store
from zarr_vectors.spatial.boundary import (
    cross_chunk_links_for_segments,
    split_polyline_at_boundaries,
)
from zarr_vectors.typing import ChunkShape

from zarr_vectors_tools.multiresolution.coarsen import (
    _stamp_root_capability,
    build_pyramid,
)


# ---------------------------------------------------------------------------
# TRK header parsing (no nibabel required)
# ---------------------------------------------------------------------------

def parse_trk_header(path: str | Path) -> dict[str, Any]:
    """Parse the 1000-byte TRK header using struct.

    Returns a dict with keys: dim, voxel_size, origin, n_scalars, n_properties,
    vox_to_ras (4×4 float32 array), voxel_order (str), n_count, version, hdr_size.
    """
    path = Path(path)
    with open(path, "rb") as f:
        raw = f.read(1000)

    if len(raw) < 1000:
        raise ValueError(f"File too short to be a valid TRK: {path}")

    id_string = raw[0:6]
    if id_string[:5] != b"TRACK":
        raise ValueError(f"Not a TRK file (bad magic): {path}")

    dim = struct.unpack_from("<3h", raw, 6)
    voxel_size = struct.unpack_from("<3f", raw, 12)
    origin = struct.unpack_from("<3f", raw, 24)
    n_scalars = struct.unpack_from("<h", raw, 36)[0]
    n_properties = struct.unpack_from("<h", raw, 236)[0]
    vox_to_ras_flat = struct.unpack_from("<16f", raw, 440)
    vox_to_ras = np.array(vox_to_ras_flat, dtype=np.float32).reshape(4, 4)
    voxel_order_raw = raw[948:952]
    voxel_order = voxel_order_raw.rstrip(b"\x00").decode("ascii", errors="replace")
    n_count = struct.unpack_from("<i", raw, 988)[0]
    version = struct.unpack_from("<i", raw, 992)[0]
    hdr_size = struct.unpack_from("<i", raw, 996)[0]

    return {
        "dim": tuple(int(d) for d in dim),
        "voxel_size": tuple(float(v) for v in voxel_size),
        "origin": tuple(float(o) for o in origin),
        "n_scalars": int(n_scalars),
        "n_properties": int(n_properties),
        "vox_to_ras": vox_to_ras,
        "voxel_order": voxel_order,
        "n_count": int(n_count),
        "version": int(version),
        "hdr_size": int(hdr_size),
    }


def _compute_bounds_from_header(header: dict[str, Any]) -> tuple[list[float], list[float]]:
    """Compute voxmm bounding box from header dim and voxel_size."""
    dim = header["dim"]
    vs = header["voxel_size"]
    lo = [0.0, 0.0, 0.0]
    hi = [float(dim[i]) * float(vs[i]) for i in range(3)]
    return lo, hi


def _compute_chunk_shape(
    bounds: tuple[list[float], list[float]],
    num_chunks: int | tuple[int, int, int] | None = None,
) -> tuple[float, float, float]:
    """Derive near-isotropic chunk shape from bounds and target chunk count.

    Args:
        bounds: (min_corner, max_corner) in voxmm.
        num_chunks: Integer total target chunk count (near-isotropic default),
            or explicit (nx, ny, nz) 3-tuple, or None (uses default of 125).

    Returns:
        Chunk shape (cx, cy, cz) in mm.
    """
    lo, hi = bounds
    extent = [hi[i] - lo[i] for i in range(3)]

    if isinstance(num_chunks, tuple):
        nx, ny, nz = num_chunks
    else:
        T = int(num_chunks) if num_chunks is not None else 125
        vol = extent[0] * extent[1] * extent[2]
        if vol <= 0:
            return (10.0, 10.0, 10.0)
        s = (T / vol) ** (1.0 / 3.0)
        nx = max(1, round(extent[0] * s))
        ny = max(1, round(extent[1] * s))
        nz = max(1, round(extent[2] * s))

    cx = round(extent[0] / nx)
    cy = round(extent[1] / ny)
    cz = round(extent[2] / nz)
    return (cx, cy, cz)


# ---------------------------------------------------------------------------
# Offset index: scan the file to build streamline byte offsets
# ---------------------------------------------------------------------------

def build_offset_index(path: str | Path, header: dict[str, Any]) -> dict[str, npt.NDArray]:
    """Scan the TRK file and record each streamline's byte offset.

    Returns dict with:
        byte_offset  int64 (O,)  byte position of each streamline's n_points int32
        n_points     int32 (O,)  point count per streamline
        nbytes       int64 (O,)  byte span of each streamline record (4 + n_pts*pt_stride + props)
    """
    n_scalars = header["n_scalars"]
    n_properties = header["n_properties"]
    pt_stride = (3 + n_scalars) * 4
    prop_bytes = n_properties * 4

    byte_offsets = []
    n_points_list = []

    path = Path(path)
    with open(path, "rb") as f:
        f.seek(1000)
        while True:
            pos = f.tell()
            b = f.read(4)
            if len(b) < 4:
                break
            n = struct.unpack("<i", b)[0]
            if n <= 0:
                break
            n_points_list.append(n)
            byte_offsets.append(pos)
            f.seek(n * pt_stride + prop_bytes, 1)

    byte_offsets_arr = np.array(byte_offsets, dtype=np.int64)
    n_points_arr = np.array(n_points_list, dtype=np.int32)
    pt_stride_arr = np.int64(pt_stride)
    prop_bytes_arr = np.int64(prop_bytes)
    nbytes_arr = (4 + n_points_arr.astype(np.int64) * pt_stride_arr + prop_bytes_arr)

    return {
        "byte_offset": byte_offsets_arr,
        "n_points": n_points_arr,
        "nbytes": nbytes_arr,
    }


def partition_offset_index(
    offset_index: dict[str, npt.NDArray],
    n_parts: int,
) -> list[dict[str, Any]]:
    """Split offset index into N byte-balanced, non-overlapping parts.

    Each part dict has: byte_offset, n_points (arrays for that part),
    poly_id_base (global index of first streamline in part).
    """
    cum_bytes = np.concatenate([[0], np.cumsum(offset_index["nbytes"])])
    total_bytes = int(cum_bytes[-1])
    n_streamlines = len(offset_index["byte_offset"])

    # Compute end indices for each part independently, then build non-overlapping
    # [start, end) ranges. Each boundary snaps to the streamline that first crosses
    # the target byte mark, so no streamline appears in more than one part.
    end_indices: list[int] = []
    for i in range(n_parts):
        target_end = total_bytes * (i + 1) // n_parts
        end_idx = int(np.searchsorted(cum_bytes, target_end, side="right"))
        end_indices.append(min(end_idx, n_streamlines))

    start_indices = [0] + end_indices[:-1]

    parts = []
    for start_idx, end_idx in zip(start_indices, end_indices):
        if start_idx >= end_idx:
            continue
        parts.append({
            "poly_id_base": int(start_idx),
            "byte_offset": offset_index["byte_offset"][start_idx:end_idx],
            "n_points": offset_index["n_points"][start_idx:end_idx],
        })
    return parts


# ---------------------------------------------------------------------------
# Phase A worker: read one part, bin into spatial chunks, write .npz
# ---------------------------------------------------------------------------

def _phase_a_worker(
    part_index: int,
    shared: dict[str, Any],
) -> str:
    """Process one byte-range part of the TRK file.

    Reads streamlines, bins each into spatial chunks via split_polyline_at_boundaries,
    and writes a .npz with per-streamline segment descriptors + raw vertices.

    Returns the path to the written .npz file.
    """
    trk_path = shared["trk_path"]
    header = shared["header"]
    chunk_shape = shared["chunk_shape"]
    intermediate_dir = shared["intermediate_dir"]
    dtype = shared["dtype"]
    compute_length = shared["compute_length"]
    compute_endpoints = shared["compute_endpoints"]
    part_specs = shared["part_specs"]

    part_spec = part_specs[part_index]
    poly_id_base = part_spec["poly_id_base"]
    byte_offsets = part_spec["byte_offset"]
    n_points_arr = part_spec["n_points"]

    n_scalars = header["n_scalars"]
    n_properties = header["n_properties"]
    pt_stride = (3 + n_scalars) * 4
    prop_bytes = n_properties * 4
    np_dtype = np.dtype(dtype)

    npz_path = str(Path(intermediate_dir) / f"part_{part_index:06d}.npz")
    verts_npy_path = npz_path.replace(".npz", ".verts.npy")

    # Pre-allocate the vertex output file as a memory-mapped array.
    # split_polyline_at_boundaries never drops vertices (it only re-assigns them
    # to chunks), so the total output vertex count == total input vertex count.
    total_pts = int(n_points_arr.sum())
    # open_memmap writes the .npy header so np.load(path, mmap_mode='r') works.
    verts_mm = np.lib.format.open_memmap(
        verts_npy_path, mode="w+", dtype=np.float32, shape=(total_pts, 3)
    )

    seg_poly_ids: list[int] = []
    seg_chunk_x: list[int] = []
    seg_chunk_y: list[int] = []
    seg_chunk_z: list[int] = []
    seg_vertex_counts: list[int] = []

    lengths: list[float] = []
    starts: list[npt.NDArray] = []
    ends: list[npt.NDArray] = []

    write_cursor = 0
    with open(trk_path, "rb") as f:
        for local_idx, (byte_off, n_pts) in enumerate(zip(byte_offsets, n_points_arr)):
            poly_id = poly_id_base + local_idx
            f.seek(int(byte_off) + 4)  # skip the n_points int32 we already know
            raw = f.read(int(n_pts) * pt_stride)
            verts_flat = np.frombuffer(raw, dtype="<f4")
            if n_scalars == 0:
                verts = verts_flat.reshape(n_pts, 3).astype(np_dtype)
            else:
                verts = verts_flat.reshape(n_pts, 3 + n_scalars)[:, :3].astype(np_dtype)

            segments = split_polyline_at_boundaries(verts, chunk_shape)

            for cc, seg_verts in segments:
                n = len(seg_verts)
                verts_mm[write_cursor:write_cursor + n] = seg_verts
                write_cursor += n
                seg_poly_ids.append(poly_id)
                seg_chunk_x.append(cc[0])
                seg_chunk_y.append(cc[1])
                seg_chunk_z.append(cc[2])
                seg_vertex_counts.append(n)

            if compute_length:
                if len(verts) >= 2:
                    diffs = np.diff(verts, axis=0)
                    length = float(np.sum(np.linalg.norm(diffs, axis=1)))
                else:
                    length = 0.0
                lengths.append(length)

            if compute_endpoints:
                starts.append(verts[0].astype(np.float32))
                ends.append(verts[-1].astype(np.float32))

    # Flush and release the memmap — pages can now be evicted by the OS.
    del verts_mm

    save_dict: dict[str, Any] = {
        "poly_id_base": np.int64(poly_id_base),
        "n_streamlines": np.int64(len(byte_offsets)),
        "seg_poly_ids": np.array(seg_poly_ids, dtype=np.int64),
        "seg_chunk_x": np.array(seg_chunk_x, dtype=np.int32),
        "seg_chunk_y": np.array(seg_chunk_y, dtype=np.int32),
        "seg_chunk_z": np.array(seg_chunk_z, dtype=np.int32),
        "seg_vertex_counts": np.array(seg_vertex_counts, dtype=np.int32),
    }
    if compute_length:
        save_dict["lengths"] = np.array(lengths, dtype=np.float32)
    if compute_endpoints:
        save_dict["starts"] = np.stack(starts, axis=0).astype(np.float32) if starts else np.zeros((0, 3), dtype=np.float32)
        save_dict["ends"] = np.stack(ends, axis=0).astype(np.float32) if ends else np.zeros((0, 3), dtype=np.float32)

    np.savez_compressed(npz_path, **save_dict)
    return npz_path


# ---------------------------------------------------------------------------
# Phase B worker: assemble one spatial chunk from all N parts
# ---------------------------------------------------------------------------

def _load_part_chunk_index(npz_path: str) -> dict[str, Any]:
    """Load one part's segment-index arrays ONCE and group them by spatial
    chunk coordinate, for reuse across every chunk a Phase-B task handles.

    ``.npz`` is a zip archive: each array access on a freshly-``np.load``ed
    file decompresses it again from disk. Loading + grouping once per
    (part, task) — instead of once per (part, chunk) — matters now that a
    single task owns every chunk in one shard (see the shard-batching
    comment on :func:`_phase_b_worker`): a shard can hold dozens of
    occupied chunks, and reloading every part file from scratch for each
    one was observed to build up several GiB of "unmanaged" worker memory
    (repeated zip-decompression garbage the allocator didn't return to the
    OS fast enough) under real multi-worker runs, triggering dask worker
    restarts.
    """
    data = np.load(npz_path)
    px = data["seg_chunk_x"]
    py = data["seg_chunk_y"]
    pz = data["seg_chunk_z"]
    seg_poly_ids = data["seg_poly_ids"]
    seg_vtx_counts = data["seg_vertex_counts"]
    vtx_starts = np.concatenate([[0], np.cumsum(seg_vtx_counts)])[:-1]

    # Vectorised within_poly_idx: position of each segment within its polyline.
    # seg_poly_ids is monotonically non-decreasing within each part (Phase A
    # writes segments in poly_id order), so group boundaries are diff > 0.
    change_pts = np.concatenate([[0], np.where(np.diff(seg_poly_ids))[0] + 1])
    group_id = np.searchsorted(change_pts, np.arange(len(seg_poly_ids)), side="right") - 1
    within_poly_idx_arr = np.arange(len(seg_poly_ids)) - change_pts[group_id]
    del data  # release the npz handle; derived arrays above are independent copies

    # Group every segment's index by its chunk coordinate, once, via a
    # single lexsort — replaces an O(all_segments) boolean mask per chunk.
    n = len(px)
    chunk_to_match_idx: dict[tuple[int, int, int], npt.NDArray] = {}
    if n > 0:
        order = np.lexsort((pz, py, px))
        px_s, py_s, pz_s = px[order], py[order], pz[order]
        changed = np.empty(n, dtype=bool)
        changed[0] = True
        changed[1:] = (
            (px_s[1:] != px_s[:-1]) | (py_s[1:] != py_s[:-1]) | (pz_s[1:] != pz_s[:-1])
        )
        group_starts = np.flatnonzero(changed)
        group_ends = np.append(group_starts[1:], n)
        for s, e in zip(group_starts, group_ends):
            key = (int(px_s[s]), int(py_s[s]), int(pz_s[s]))
            chunk_to_match_idx[key] = order[s:e]

    return {
        "chunk_to_match_idx": chunk_to_match_idx,
        "seg_poly_ids": seg_poly_ids,
        "within_poly_idx_arr": within_poly_idx_arr,
        "vtx_starts": vtx_starts,
        "seg_vtx_counts": seg_vtx_counts,
        "verts_npy_path": npz_path.replace(".npz", ".verts.npy"),
    }


def _phase_b_assemble_one_chunk(
    chunk_coords: tuple[int, int, int],
    per_part: list[dict[str, Any]],
    out_dtype: np.dtype,
) -> tuple[
    list[npt.NDArray],
    npt.NDArray,
    npt.NDArray,
] | None:
    """Assemble (but do not write) one spatial chunk's level-0 data from a
    PRE-SLICED per-part payload (built by the coordinator for this chunk;
    see the per-shard spill in :func:`ingest_trk_parallel`).

    ``per_part`` is a list of dicts, one per contributing part, each with
    the segment rows already restricted to THIS chunk:
    ``{part_idx, verts_npy_path, poly_ids, within, vtx_starts, vtx_counts}``.
    Segments are taken in ascending (part_index, poly_id, seg_within_poly)
    order — the determinism pin. Writing is deferred to the caller
    (:func:`_phase_b_worker`), which batches every chunk in the task into
    one write per array.

    Returns ``None`` if this chunk has no records, else
    ``(vert_groups, seg_ids, seg_rows)`` where ``seg_rows`` is a compact
    ``(n_segments, 9)`` int64 array with columns
    ``[part_idx, poly_id, within, cx, cy, cz, frag_idx, first_row, last_row]``.
    A dense array (rather than a dict-of-tuples) keeps the coordinator's
    fragment map O(segments) *dense* instead of paying Python dict+tuple
    overhead (~14x), which was the dominant coordinator-RSS term at scale.
    """
    # Records for this chunk across all parts, in (part, poly_id, seg_within_poly) order
    # We collect: (part_idx, poly_id, seg_idx_within_poly, vertex_array)
    records: list[tuple[int, int, int, npt.NDArray]] = []

    for pp in per_part:
        part_idx = int(pp["part_idx"])
        m_poly_ids = pp["poly_ids"]
        m_within = pp["within"]
        m_vtx_starts = pp["vtx_starts"]
        m_vtx_counts = pp["vtx_counts"]

        vertices = np.load(pp["verts_npy_path"], mmap_mode="r")

        for i in range(len(m_poly_ids)):
            vs = int(m_vtx_starts[i])
            vc = int(m_vtx_counts[i])
            # np.array() forces a copy: releases the mmap reference so pages can
            # be evicted once we close `vertices` below.  Create it directly in
            # the output dtype so the write path needs no further copy (a no-op
            # cast from the float32 mmap when out_dtype is float32).
            seg_verts = np.array(vertices[vs:vs + vc], dtype=out_dtype)
            records.append((part_idx, int(m_poly_ids[i]), int(m_within[i]), seg_verts))

        del vertices  # close mmap; OS can evict pages immediately

    if not records:
        return None

    # Sort: (part_idx, poly_id, seg_within_poly) — the determinism pin
    records.sort(key=lambda r: (r[0], r[1], r[2]))

    # Reference the arrays already held in `records` (they are already
    # `out_dtype`) instead of copying — avoids a full duplicate of the whole
    # chunk's vertices and a second set of per-segment array objects.
    vert_groups = [r[3] for r in records]

    # Per-fragment segment id (global poly_id = streamline index in file).
    # One uint64 per fragment, same order as vert_groups.  Neuroglancer uses this
    # to map a picked spatial fragment back to the full streamline for pass-2 fetch.
    seg_ids = np.array([r[1] for r in records], dtype=np.uint64)

    # Build compact fragment-map rows for this chunk:
    #   [part_idx, poly_id, within, cx, cy, cz, frag_idx, first_row, last_row]
    # one int64 row per fragment, in the same (part, poly, within) order.
    cx, cy, cz = chunk_coords
    seg_rows = np.empty((len(records), 9), dtype=np.int64)
    cum_row = 0
    for frag_idx, (part_idx, poly_id, seg_within, seg_verts) in enumerate(records):
        first_row = cum_row
        last_row = cum_row + len(seg_verts) - 1
        seg_rows[frag_idx] = (part_idx, poly_id, seg_within, cx, cy, cz,
                              frag_idx, first_row, last_row)
        cum_row += len(seg_verts)

    return vert_groups, seg_ids, seg_rows


def _phase_b_worker(
    batch: dict[str, Any],
    shared: dict[str, Any],
) -> npt.NDArray:
    """Write ONE shard's worth of chunks from a coordinator-spilled,
    PRE-SLICED per-chunk payload.

    ``batch`` = ``{"chunks": [chunk_coords, ...], "spill_path": <path>}``.
    ``chunks`` is every occupied chunk mapping to ONE shard file — a single
    worker owns the whole shard so two processes never read-modify-write
    the same native-sharded shard file (the "missing blocks" write-race
    fix). The spill file holds, per chunk, the segment rows for that chunk
    already sliced out of each part's index.

    The per-chunk data is read from the spill file on LOCAL DISK — it is
    NOT broadcast to every worker. Scattering the whole (all-parts,
    all-chunks) index via dask (the previous approach) broadcast an
    O(total-segments) object to every worker and exhausted OS socket
    buffers at 5M streamlines (``OSError: No buffer space available``);
    the per-shard spill sends each worker only its shard's slice, once,
    over disk. Vertex floats never enter the spill — they stay on the
    mmap'd ``.verts.npy`` and are read lazily by chunk.

    Returns one compact ``(n_segments, 9)`` int64 array covering every
    fragment written by this shard, columns
    ``[part_idx, poly_id, within, cx, cy, cz, frag_idx, first_row, last_row]``.
    The coordinator concatenates these dense arrays (see
    :func:`_build_manifests_and_cross_links`) instead of merging O(segments)
    Python dicts — the dominant coordinator-RSS term at scale.
    """
    import pickle
    store_path = shared["store_path"]
    dtype = shared["dtype"]
    out_dtype = np.dtype(dtype)

    chunk_coords_batch = [tuple(int(x) for x in c) for c in batch["chunks"]]
    spill_path = batch.get("spill_path")
    batch_data: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    if spill_path:
        with open(spill_path, "rb") as fh:
            batch_data = pickle.load(fh)

    root = open_store(str(store_path), mode="r+")
    # Level group must already exist (created in coordinator pre-phase-B)
    from zarr_vectors.core.store import get_resolution_level
    level_group = get_resolution_level(root, 0)

    # Assemble every chunk in this batch first, then write each array with
    # ONE batched call covering the whole shard — see
    # write_chunk_vertices_batch's docstring for why that's not just an
    # optimization: sequential per-chunk writes to a native-sharded array
    # each pay a full shard read-modify-write, and with a whole shard's
    # worth of chunks now landing in one task (the "missing blocks" write-
    # race fix), that repeated cost was observed to build up several GiB
    # of worker memory and trigger dask worker restarts on a real run.
    seg_row_blocks: list[npt.NDArray] = []
    coords_to_groups: dict[tuple[int, int, int], list[npt.NDArray]] = {}
    coords_to_seg_ids: dict[tuple[int, int, int], npt.NDArray] = {}
    for chunk_coords in chunk_coords_batch:
        per_part = batch_data.get(chunk_coords)
        if not per_part:
            continue
        assembled = _phase_b_assemble_one_chunk(chunk_coords, per_part, out_dtype)
        if assembled is None:
            continue
        vert_groups, seg_ids, seg_rows = assembled
        coords_to_groups[chunk_coords] = vert_groups
        coords_to_seg_ids[chunk_coords] = seg_ids
        seg_row_blocks.append(seg_rows)

    if coords_to_groups:
        write_chunk_vertices_batch(level_group, coords_to_groups, dtype=out_dtype)
        write_chunk_fragment_attributes_batch(
            level_group, "segment_id", coords_to_seg_ids, dtype=np.uint64,
        )
    if seg_row_blocks:
        return np.concatenate(seg_row_blocks, axis=0)
    return np.empty((0, 9), dtype=np.int64)


# ---------------------------------------------------------------------------
# Coordinator: shard-reduce object index + cross-chunk links by streamline id
# ---------------------------------------------------------------------------

def _sort_seg_rows(phase_b_results: list[npt.NDArray]) -> npt.NDArray:
    """Concatenate Phase B's per-shard ``(n, 9)`` int64 arrays and order by
    ``(poly, part, within)`` — reproduces the old part-order / npz
    file-order determinism pin (``within`` is unique per ``(part, poly)``).

    Columns: ``[part, poly, within, cx, cy, cz, frag_idx, first_row, last_row]``.
    Sorting by ``poly`` first (as the PRIMARY key) also makes every
    streamline's rows a contiguous run, which is what lets
    :func:`_reduce_level0_oid_shard` be dispatched on arbitrary contiguous
    row ranges without ever splitting one streamline's segments across two
    shards.
    """
    blocks = [a for a in phase_b_results if a is not None and len(a)]
    if not blocks:
        return np.empty((0, 9), dtype=np.int64)
    seg = np.concatenate(blocks, axis=0)
    del blocks
    order = np.lexsort((seg[:, 2], seg[:, 0], seg[:, 1]))
    return seg[order]


def _reduce_level0_oid_shard(
    payload: dict[str, Any], shared: dict[str, Any],
) -> dict[str, Any]:
    """Reduce one contiguous streamline-id (OID) range's segment rows into
    encoded object-index manifest blobs and this shard's local cross-chunk
    links.

    ``payload["seg_path"]`` is a coordinator-spilled slice of the globally
    sorted (by poly, part, within) segment-row array from
    :func:`_sort_seg_rows` — sliced on a streamline-id BOUNDARY, so every
    streamline's rows land entirely within one shard. Because a
    cross-chunk link only ever connects two CONSECUTIVE segments of the
    SAME streamline, this means links can't straddle two shards either —
    no cross-shard merge step is needed for either output.

    Both outputs are returned in compact form: manifest blobs as already-
    encoded bytes (O(streamlines) in this shard, not O(fragments) Python
    tuples), and cross-chunk links as one dense int64 array spilled to
    disk (not Python tuples). This is the fix for the coordinator holding
    O(total-fragments)/O(total-links) Python objects at once, which
    dominated RSS at 5M+ streamlines (~200+ bytes of pure Python-object
    overhead per record, measured).
    """
    sid_ndim = int(shared["sid_ndim"])
    seg = np.load(payload["seg_path"], allow_pickle=False)
    if len(seg) == 0:
        return {
            "oid": np.empty(0, dtype=np.int64),
            "blobs": pickle.dumps([], protocol=pickle.HIGHEST_PROTOCOL),
            "link_path": None,
            "n_links": 0,
        }

    poly_col = seg[:, 1]
    chunks = seg[:, 3:6]
    frags = seg[:, 6]

    boundaries = np.flatnonzero(np.diff(poly_col)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(seg)]))

    oids = np.empty(len(starts), dtype=np.int64)
    blobs: list[bytes] = []
    for i, (s, e) in enumerate(zip(starts.tolist(), ends.tolist())):
        oids[i] = int(poly_col[s])
        blocks = [
            ((int(chunks[k, 0]), int(chunks[k, 1]), int(chunks[k, 2])), int(frags[k]))
            for k in range(s, e)
        ]
        blobs.append(encode_object_manifest_blocks(blocks, sid_ndim=sid_ndim))

    # Vectorized cross-link derivation: a link exists at row i when row i
    # and row i+1 share a poly_id but land in different chunks.
    same_poly = poly_col[1:] == poly_col[:-1]
    diff_chunk = np.any(chunks[1:] != chunks[:-1], axis=1)
    idxs = np.flatnonzero(same_poly & diff_chunk)

    link_path = None
    n_links = int(len(idxs))
    if n_links:
        first_rows = seg[:, 7]
        last_rows = seg[:, 8]
        link_rows = np.empty((n_links, 8), dtype=np.int64)
        link_rows[:, 0:3] = chunks[idxs]
        link_rows[:, 3] = last_rows[idxs]
        link_rows[:, 4:7] = chunks[idxs + 1]
        link_rows[:, 7] = first_rows[idxs + 1]
        link_path = payload["link_spill_path"]
        np.save(link_path, link_rows)

    return {
        "oid": oids,
        "blobs": pickle.dumps(blobs, protocol=pickle.HIGHEST_PROTOCOL),
        "link_path": link_path,
        "n_links": n_links,
    }


# ---------------------------------------------------------------------------
# Top-level ingest function
# ---------------------------------------------------------------------------

def ingest_trk_parallel(
    input_path: str | Path,
    output_path: str | Path,
    *,
    num_chunks: int | tuple[int, int, int] | None = None,
    n_parts: int | None = None,
    workers: int | None = None,
    executor: Any = None,
    dtype: str = "float32",
    max_streamlines: int | None = None,
    compute_length: bool = False,
    compute_endpoints: bool = False,
    preserve_header: bool = True,
    shard_shape: int | tuple[int, int, int] | None = 2,
    build_multiscale: bool = True,
    pyramid_factors: list[tuple[float, float]] | None = None,
    chunk_scale_factors: list[int] | None = None,
    sparsity_strategy: str = "length",
    pyramid_coarsen_mode: str = "rdp",
    intermediate_dir: str | Path | None = None,
    keep_intermediate: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Ingest a large TRK file into a zarr-vectors streamline store.

    Memory-bounded: processes the file in parallel byte-range parts, binning
    streamlines into spatial chunks without loading the whole file into RAM.

    Args:
        input_path: Path to the .trk file.
        output_path: Path for the new zarr-vectors store.
        num_chunks: Target total chunk count (integer, near-isotropic default),
            or explicit (nx, ny, nz) 3-tuple, or None (uses 125).
        n_parts: Number of file parts for Phase A parallelism. Controls how
            finely the input file is sliced — does NOT affect how many
            processes run simultaneously (that is ``workers``). Defaults to
            4× workers (fine-grained enough for good load balancing).
        workers: Number of Dask worker processes. None = cpu_count-1.
        executor: Injected executor (func, items, shared) callable. If None,
            either uses dask_executor (when workers>1) or runs serially.
        dtype: Numpy dtype for vertex positions.
        max_streamlines: If set, only ingest the first N streamlines from the
            file (in on-disk order). Useful for quick test runs on a subset
            of a large tractogram; leave None to ingest all streamlines.
        compute_length: Write per-streamline path length to object_attributes.
        compute_endpoints: Write start/end points to object_attributes.
        preserve_header: Store TRKHeader (affine, dims) in the zarr store.
        shard_shape: Outer-chunk shape (in spatial-chunk units) for the
            level's native-sharded per-chunk arrays (``vertices``,
            ``vertex_fragments``, ``fragment_attributes/segment_id``),
            passed to :func:`zarr_vectors.core.arrays.open_write_session`.
            ``vertices`` cells are written uncompressed (``compress=False``)
            so a reader can byte-range-read one fragment's rows within a
            cell; the other per-chunk arrays keep the default codec. Pass
            ``None`` to fall back to the legacy per-chunk-array-per-object
            layout (no sharding, vertices compressed).
        build_multiscale: Build coarser pyramid levels after level 0.
        pyramid_factors: List of (coarsen_factor, sparsity_factor) per level.
            Default: [(8.0, 1.0), (8.0, 1.0)] for two coarser levels. In
            "decimate" mode, coarsen_factor is the decimation stride rather
            than an RDP epsilon multiplier.
        chunk_scale_factors: Per-axis chunk multiplier per level. Default [1, 1].
        sparsity_strategy: "length" or "random" for object dropping.
        pyramid_coarsen_mode: "rdp" (default, Douglas-Peucker simplification)
            or "decimate" (uniform per-object stride decimation, keeping
            every stride-th vertex plus endpoints).
        intermediate_dir: Directory for .npz scratch files. Default: temp dir.
        keep_intermediate: If True, don't delete the intermediate dir on exit.
        progress: Print progress messages to stdout.

    Returns:
        Summary dict with vertex_count, streamline_count, chunk_count, etc.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    def _log(msg: str) -> None:
        if progress:
            print(msg)

    # --- Phase 0: header + grid -------------------------------------------
    _log("Phase 0: parsing TRK header...")
    header = parse_trk_header(input_path)
    bounds = _compute_bounds_from_header(header)
    chunk_shape = _compute_chunk_shape(bounds, num_chunks)
    lo, hi = bounds
    extent = [hi[i] - lo[i] for i in range(3)]
    grid_shape = (
        max(1, math.ceil(extent[0] / chunk_shape[0])),
        max(1, math.ceil(extent[1] / chunk_shape[1])),
        max(1, math.ceil(extent[2] / chunk_shape[2])),
    )
    _log(f"  bounds: {lo} → {hi} mm")
    _log(f"  chunk_shape: {chunk_shape}")
    _log(f"  grid: {grid_shape} = {grid_shape[0]*grid_shape[1]*grid_shape[2]} chunks")

    # --- Phase 1: offset index + partition --------------------------------
    _log("Phase 1: building offset index (scanning file)...")
    offset_index = build_offset_index(input_path, header)
    n_streamlines = len(offset_index["byte_offset"])
    n_count_hdr = header["n_count"]
    _log(f"  found {n_streamlines} streamlines "
         f"(header says {n_count_hdr if n_count_hdr > 0 else 'unknown'})")

    if max_streamlines is not None and max_streamlines < n_streamlines:
        _log(f"  max_streamlines set: limiting to first {max_streamlines} "
             f"of {n_streamlines} streamlines")
        offset_index = {
            "byte_offset": offset_index["byte_offset"][:max_streamlines],
            "n_points": offset_index["n_points"][:max_streamlines],
            "nbytes": offset_index["nbytes"][:max_streamlines],
        }
        n_streamlines = len(offset_index["byte_offset"])

    _n_workers = workers if workers and workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    # n_parts controls file split granularity independently of worker count.
    # More parts → better load balancing; excess parts are queued by the executor.
    _n_parts = n_parts if n_parts and n_parts > 0 else max(_n_workers * 4, 16)
    _n_parts = min(_n_parts, n_streamlines)

    part_specs = partition_offset_index(offset_index, _n_parts)
    _log(f"  partitioned into {len(part_specs)} parts")

    # --- Setup intermediate directory -------------------------------------
    _own_tempdir = intermediate_dir is None
    if _own_tempdir:
        _tmpdir_obj = tempfile.TemporaryDirectory(prefix="trk_parallel_")
        _intermediate_dir = _tmpdir_obj.name
    else:
        _intermediate_dir = str(intermediate_dir)
        Path(_intermediate_dir).mkdir(parents=True, exist_ok=True)
        _tmpdir_obj = None

    try:
        # --- Create zarr-vectors store ------------------------------------
        _log("Creating zarr-vectors store...")
        vox_to_ras = header["vox_to_ras"]
        crs_dict = {
            "input_space": "voxmm",
            "output_space": "RASmm",
            "units": "mm",
            "affine": vox_to_ras.flatten().tolist(),
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

        # Pre-create level 0 arrays (Phase B workers need the level group)
        _log("Creating level-0 arrays...")
        level_meta = LevelMetadata(
            level=0,
            vertex_count=0,  # updated after Phase B
            # "fragment_attributes" listed up front (not just in the
            # best-effort re-stamp near the end of this function): segment_id
            # is created right below and neuroglancer's reader gates whether
            # it even attempts to fetch per-fragment segment ids on this
            # list being accurate — a level left showing the initial
            # placeholder here (e.g. if a later pipeline stage crashes
            # before the re-stamp runs, as happened when Phase 6 OOM'd on a
            # real run) silently falls back to a meaningless per-chunk
            # fragment index for picking/selection.
            arrays_present=["vertices", "object_index", "fragment_attributes"],
        )
        level_group = create_resolution_level(root, 0, level_meta)
        # main's native-sharding grid math anchors chunk coord 0 at the
        # global origin and cannot represent a negative chunk coordinate;
        # bounds here are always non-negative (voxmm space, lo=[0,0,0] by
        # convention — see _compute_bounds_from_header) but guard anyway
        # rather than let a future convention change write out-of-grid
        # coordinates main's Group.write_bytes rejects.
        level_shard_shape = shard_shape
        if level_shard_shape is not None and any(b < 0 for b in bounds[0]):
            level_shard_shape = None
        with open_write_session(
            level_group,
            shard_shape=level_shard_shape,
            bounds=bounds,
            chunk_shape=chunk_shape,
        ):
            # compress=False: sharded vertices cells are written raw so a
            # reader can byte-range-read one fragment's rows within a cell
            # instead of fetching and decompressing the whole chunk.
            create_vertices_array(level_group, dtype=dtype, compress=False)
            create_object_index_array(level_group)
            create_cross_chunk_links_array(level_group, delta=0, sid_ndim=3)
            create_fragment_attribute_array(level_group, "segment_id", dtype="uint64")

        # --- Phase A: per-part spatial binning ---------------------------
        _log(f"Phase A: binning streamlines ({len(part_specs)} parts)...")

        shared_a = {
            "trk_path": str(input_path),
            "header": header,
            "chunk_shape": chunk_shape,
            "intermediate_dir": _intermediate_dir,
            "dtype": dtype,
            "compute_length": compute_length,
            "compute_endpoints": compute_endpoints,
            "part_specs": part_specs,
        }

        def _run_phase_a_serial(items: list[int], shared: dict) -> list[str]:
            return [_phase_a_worker(i, shared) for i in items]

        part_indices = list(range(len(part_specs)))

        if executor is not None:
            part_npz_paths = executor(_phase_a_worker, part_indices, shared=shared_a)
        else:
            part_npz_paths = _run_phase_a_serial(part_indices, shared_a)

        _log(f"  wrote {len(part_npz_paths)} intermediate files")

        # Build each part's chunk-grouping index ONCE here (one lexsort per
        # part), then SPILL each shard's slice to a temp file the workers
        # read from local disk. The grouping is NOT scattered whole to every
        # worker: at 5M streamlines the all-parts index is hundreds of MB,
        # and `client.scatter(broadcast=True)` shipping it to every worker
        # at once exhausted OS socket send buffers (`OSError: No buffer
        # space available`). Per-shard spill sends each worker only its
        # shard's slice, once, over disk. Vertex floats never enter the
        # spill — they stay on the mmap'd `.verts.npy`. (The grouping keys
        # also give the occupied-chunk set for free, avoiding a separate
        # O(all_segments) scan.)
        part_indices = [_load_part_chunk_index(p) for p in part_npz_paths]
        all_chunk_coords: set[tuple[int, int, int]] = set()
        for pidx in part_indices:
            all_chunk_coords.update(pidx["chunk_to_match_idx"].keys())
        _log(f"  {len(all_chunk_coords)} occupied spatial chunks")

        # --- Phase B: per-shard level-0 write -----------------------------
        _log(f"Phase B: writing level-0 ({len(all_chunk_coords)} chunks)...")

        shared_b = {
            "store_path": str(output_path),
            "dtype": dtype,
        }

        # Batch chunks by shard address: the vertices/fragment_attributes
        # arrays are native-sharded (when level_shard_shape is set), and a
        # partial write to a shard reads-modifies-writes the WHOLE shard
        # file with no cross-process coordination — two workers racing on
        # the same shard silently drop whichever one loses. Dispatching one
        # task per shard (instead of per chunk) makes that impossible: each
        # shard is written by exactly one worker. Unsharded stores have no
        # such collision (one file per chunk), so keep the finer per-chunk
        # dispatch there.
        if level_shard_shape is not None:
            shard_shape_3 = (
                (level_shard_shape,) * 3
                if isinstance(level_shard_shape, int)
                else tuple(level_shard_shape)
            )
            shard_groups: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
            for cc in sorted(all_chunk_coords):
                shard_addr = (
                    cc[0] // shard_shape_3[0],
                    cc[1] // shard_shape_3[1],
                    cc[2] // shard_shape_3[2],
                )
                shard_groups.setdefault(shard_addr, []).append(cc)
            chunk_lists = [shard_groups[k] for k in sorted(shard_groups)]
        else:
            chunk_lists = [[cc] for cc in sorted(all_chunk_coords)]

        # Spill each batch's pre-sliced per-part segment rows to a temp file;
        # pass the worker only chunk coords + the path (tiny graph payload).
        import pickle as _pickle
        phaseb_spill_dir = tempfile.mkdtemp(prefix="trk_phaseb_")
        chunk_batches: list[dict[str, Any]] = []
        for bi, chunks in enumerate(chunk_lists):
            batch_data: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
            for cc in chunks:
                per_part: list[dict[str, Any]] = []
                for part_idx, pidx in enumerate(part_indices):
                    mi = pidx["chunk_to_match_idx"].get(cc)
                    if mi is None or len(mi) == 0:
                        continue
                    per_part.append({
                        "part_idx": part_idx,
                        "verts_npy_path": pidx["verts_npy_path"],
                        "poly_ids": pidx["seg_poly_ids"][mi],
                        "within": pidx["within_poly_idx_arr"][mi],
                        "vtx_starts": pidx["vtx_starts"][mi],
                        "vtx_counts": pidx["seg_vtx_counts"][mi],
                    })
                if per_part:
                    batch_data[cc] = per_part
            spill_path = None
            if batch_data:
                spill_path = os.path.join(phaseb_spill_dir, f"pb_{bi}.pkl")
                with open(spill_path, "wb") as fh:
                    _pickle.dump(batch_data, fh, protocol=_pickle.HIGHEST_PROTOCOL)
            chunk_batches.append({"chunks": chunks, "spill_path": spill_path})
        # Free the coordinator's full index now that slices are spilled.
        del part_indices

        try:
            if executor is not None:
                phase_b_results = executor(_phase_b_worker, chunk_batches, shared=shared_b)
            else:
                phase_b_results = [_phase_b_worker(batch, shared_b) for batch in chunk_batches]
        finally:
            shutil.rmtree(phaseb_spill_dir, ignore_errors=True)

        if level_shard_shape is not None:
            # Each Phase-B worker's write_bytes call also incrementally
            # updates the array's own `nonempty_chunks` manifest attribute
            # (a single shared zarr.json per array) — safe from one
            # process, but racy across the concurrent workers dispatched
            # above: two workers finishing their own zarr.json
            # read-modify-write around the same time can each silently
            # drop the other's additions, even though the chunk data
            # itself (independent per-shard writes) is unaffected. Repair
            # it once here from the coordinator's own already-known-
            # correct occupied-chunk set.
            level_group.set_nonempty_chunks(VERTICES, all_chunk_coords)
            level_group.set_nonempty_chunks(
                f"{FRAGMENT_ATTRIBUTES}/segment_id", all_chunk_coords
            )

        # --- Coordinator: shard-reduce object index + cross-chunk links ---
        # Dispatched to workers by contiguous streamline-id (OID) range —
        # each shard's manifest blobs come back already encoded (bytes,
        # O(streamlines)) and its local cross-chunk links come back as one
        # dense int64 array on disk (not Python tuples): the coordinator
        # never materializes an O(total-fragments) dict-of-tuples or an
        # O(total-links) list-of-tuples, which is what made RSS scale
        # ~linearly with link count at 5M+ (measured ~200+ bytes of pure
        # Python-object overhead per record).
        _log("Coordinator: reducing object index + cross-chunk links "
             "(sharded by streamline id)...")
        seg = _sort_seg_rows(phase_b_results)
        del phase_b_results

        n_oid_shards = max(1, min(4096, -(-n_streamlines // 20_000))) if n_streamlines else 1
        oid_bounds = np.linspace(0, n_streamlines, n_oid_shards + 1).astype(np.int64)
        row_bounds = (
            np.searchsorted(seg[:, 1], oid_bounds) if len(seg)
            else np.zeros(n_oid_shards + 1, dtype=np.int64)
        )

        oid_reduce_dir = tempfile.mkdtemp(prefix="trk_oidreduce_")
        link_spill_dir = tempfile.mkdtemp(prefix="trk_crosslink_")
        try:
            reduce_payloads: list[dict[str, Any]] = []
            for i in range(n_oid_shards):
                lo, hi = int(row_bounds[i]), int(row_bounds[i + 1])
                if hi <= lo:
                    continue
                seg_path = os.path.join(oid_reduce_dir, f"seg_{i}.npy")
                np.save(seg_path, seg[lo:hi])
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
            cross_chunk_link_count = 0
            for rc in reduce_results:
                oids = rc["oid"]
                if len(oids):
                    blobs = pickle.loads(rc["blobs"])
                    for i, oid in enumerate(oids.tolist()):
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
                link_row_blocks = [np.load(p, allow_pickle=False) for p in link_paths]
                all_link_rows = np.concatenate(link_row_blocks, axis=0)
                del link_row_blocks
                # write_cross_chunk_links_bulk: numpy-native writer restricted
                # to exactly this policy (link_width=2, directed, canonical,
                # packed_sharded, replace) — the general write_cross_chunk_links
                # rebuilds one Python tuple/dict per record internally
                # (_normalise_cross_records -> partition_cross_records_by_tuple
                # -> _write_cross_chunk_links_packed's merged dict), which both
                # grows CPython's GC-scan cost with N and, at ~200-400 bytes of
                # object overhead per record, pushes large runs into swap. The
                # bulk path stays in dense int64 arrays throughout (measured
                # 27.8x faster / ~3x lower peak RSS at 5M links) — the row
                # layout the shard spill files already use, [chunk_a(3), vi_a,
                # chunk_b(3), vi_b], is exactly its input format.
                #
                # layout="packed_sharded": stores all cells in ONE native-
                # sharded 1-D array instead of thousands of tiny per-cell
                # arrays — level 0 has by far the most cells, so packing it
                # collapses both this write and the coarsener's
                # read_cross_chunk_links(src=level 0) into a handful of shard
                # files. Readers dispatch on the stored layout meta.
                write_cross_chunk_links_bulk(
                    level_group, all_link_rows, sid_ndim=3, delta=0,
                )
                del all_link_rows
                _stamp_root_capability(root, CAP_MULTISCALE_LINKS)
        finally:
            shutil.rmtree(link_spill_dir, ignore_errors=True)

        # Object attributes
        obj_attrs_to_write: dict[str, npt.NDArray] = {}
        if compute_length:
            length_parts = []
            for npz_path in part_npz_paths:
                d = np.load(npz_path)
                if "lengths" in d:
                    length_parts.append(d["lengths"])
            if length_parts:
                obj_attrs_to_write["length"] = np.concatenate(length_parts, axis=0)

        if compute_endpoints:
            start_parts, end_parts = [], []
            for npz_path in part_npz_paths:
                d = np.load(npz_path)
                if "starts" in d:
                    start_parts.append(d["starts"])
                if "ends" in d:
                    end_parts.append(d["ends"])
            if start_parts:
                obj_attrs_to_write["start"] = np.concatenate(start_parts, axis=0)
            if end_parts:
                obj_attrs_to_write["end"] = np.concatenate(end_parts, axis=0)

        if obj_attrs_to_write:
            from zarr_vectors.core.arrays import write_object_attributes
            from zarr_vectors_tools.multiresolution.coarsen import create_object_attributes_array
            for attr_name, data in obj_attrs_to_write.items():
                create_object_attributes_array(level_group, attr_name)
                write_object_attributes(level_group, attr_name, data)

        # --- Phase 5: header + affine metadata ---------------------------
        if preserve_header:
            try:
                from zarr_vectors_tools.headers.registry import HeaderRegistry
                from zarr_vectors_tools.headers.formats import TRKHeader
                trk_header = TRKHeader(
                    voxel_size=header["voxel_size"],
                    dimensions=header["dim"],
                    vox_to_ras=vox_to_ras.flatten().tolist(),
                    voxel_order=header["voxel_order"],
                    n_scalars=header["n_scalars"],
                    scalar_names=[],
                    n_properties=header["n_properties"],
                    property_names=[],
                    n_count=n_streamlines,
                )
                reg = HeaderRegistry(str(output_path))
                reg.add("trk", trk_header)
            except Exception:
                pass  # best-effort

        # --- Phase 6: multiscale pyramid ---------------------------------
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

        # Count total vertices written (seg_vertex_counts is always in the npz)
        total_vertices = int(sum(
            int(np.load(p)["seg_vertex_counts"].sum()) for p in part_npz_paths
        ))

        # Stamp level-0 metadata now that we know the true vertex count and
        # which arrays are present (create_resolution_level runs before Phase B
        # and leaves vertex_count=0 as a placeholder).
        try:
            import zarr as _zarr
            _root_z = _zarr.open(str(output_path), mode="r+")
            _lvl0_z = _root_z["0"]
            _lvl0_attrs = dict(_lvl0_z.attrs)
            if "zarr_vectors_level" in _lvl0_attrs:
                _lvl0_attrs["zarr_vectors_level"]["vertex_count"] = total_vertices
                _lvl0_attrs["zarr_vectors_level"]["arrays_present"] = [
                    k for k in ["vertices", "vertex_fragments", "object_index",
                                 "cross_chunk_links", "object_attributes",
                                 "fragment_attributes"]
                    if k in _lvl0_z
                ]
            _lvl0_z.attrs.update(_lvl0_attrs)
        except Exception:
            pass  # best-effort metadata stamp

        _log("Done.")

        return {
            "streamline_count": n_streamlines,
            "vertex_count": total_vertices,
            "chunk_count": len(all_chunk_coords),
            "cross_chunk_link_count": cross_chunk_link_count,
            "n_parts": len(part_specs),
            "chunk_shape": chunk_shape,
            "bounds": bounds,
            "pyramid": pyramid_summary,
        }

    finally:
        if _tmpdir_obj is not None and not keep_intermediate:
            _tmpdir_obj.cleanup()
