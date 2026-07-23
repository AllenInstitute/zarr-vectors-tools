"""Memory-bounded parallel TRK → zarr-vectors ingest.

Processes large TRK (TrackVis) files without loading them whole into RAM.
Coordinate convention: by default the raw TrackVis voxmm coordinates are stored
as-is and the ``vox_to_ras`` affine is recorded in CRS metadata (a viewer that
understands the affine renders it registered).  Pass ``register_to_rasmm=True``
to instead bake that affine into the vertices at ingest — matching
``nibabel.streamlines.load(...).streamlines`` — and store an identity CRS
affine.  That is the layout viewers which can only scale/translate (not rotate),
e.g. neuroglancer, need: NGFF transforms can't undo the affine's axis flips, so
the geometry itself must carry them.

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

import os
import struct
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.core.arrays import (
    create_fragment_attribute_array,
    create_links_family,
    create_object_index_array,
    create_vertices_array,
    level_grid_layout,
    write_chunk_fragment_attributes,
    write_chunk_vertices,
    write_links,
    write_object_index,
)
from zarr_vectors.core.metadata import LevelMetadata
from zarr_vectors.core.store import create_resolution_level, create_store, open_store
from zarr_vectors.exceptions import IngestError
from zarr_vectors.spatial.boundary import split_polyline_at_boundaries
from zarr_vectors.typing import ChunkShape

from zarr_vectors_tools._manifests import rebuild_nonempty_manifests
from zarr_vectors_tools.multiresolution.coarsen import build_pyramid


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


def _trackvis_to_rasmm_affine(input_path: str | Path) -> np.ndarray:
    """4×4 affine mapping the TRK's on-disk trackvis-voxmm coords to RASmm.

    Uses nibabel's canonical transform, which correctly handles ``voxel_order``
    and the voxel-corner→center convention (a naive ``vox_to_ras @
    (voxmm / voxel_size)`` gets those subtly wrong).  Baking this into the
    geometry makes the stored streamlines match what
    ``nibabel.streamlines.load(...).streamlines`` returns — i.e. registered to
    the source image's world space — so neuroglancer renders them aligned.
    """
    try:
        import nibabel as nib
        from nibabel.streamlines.trk import get_affine_trackvis_to_rasmm
    except ImportError as e:  # pragma: no cover - env without nibabel
        raise IngestError(
            "nibabel is required to register TRK streamlines to RASmm world "
            "space (so neuroglancer aligns them with the source image). "
            "Install with: pip install nibabel"
        ) from e
    hdr = nib.streamlines.load(str(input_path), lazy_load=True).header
    return np.asarray(get_affine_trackvis_to_rasmm(hdr), dtype=np.float64)


def _transform_bounds(
    bounds: tuple[list[float], list[float]], affine: np.ndarray,
) -> tuple[list[float], list[float]]:
    """Axis-aligned bbox of a min/max corner pair after applying ``affine``.

    The affine maps the voxmm box to a rotated parallelepiped; its
    axis-aligned bounding box is the min/max over the 8 transformed corners.
    """
    lo, hi = bounds
    a = np.asarray(affine, dtype=np.float64)
    corners = np.array(
        [[x, y, z] for x in (lo[0], hi[0])
         for y in (lo[1], hi[1]) for z in (lo[2], hi[2])],
        dtype=np.float64,
    )
    out = corners @ a[:3, :3].T + a[:3, 3]
    return out.min(axis=0).tolist(), out.max(axis=0).tolist()


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
    grid_shape = shared["grid_shape"]
    grid_origin = shared["grid_origin"]
    affine_tv2ras = shared["affine_tv2ras"]
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

            if affine_tv2ras is not None:
                # Bake trackvis-voxmm → RASmm so the stored geometry is
                # registered to the source image's world space (see module
                # docstring).  A rigid/orthogonal transform, so streamline
                # lengths and endpoints computed below are unaffected.
                verts = (
                    verts @ affine_tv2ras[:3, :3].T + affine_tv2ras[:3, 3]
                ).astype(np_dtype)

            segments = split_polyline_at_boundaries(verts, chunk_shape)

            for cc, seg_verts in segments:
                n = len(seg_verts)
                verts_mm[write_cursor:write_cursor + n] = seg_verts
                write_cursor += n
                seg_poly_ids.append(poly_id)
                # Clamp the chunk assignment to the array grid derived from the
                # TRK header bbox.  Vertices can sit fractionally outside that
                # bbox (a point at z = -epsilon floors to chunk z = -1), which
                # the single-array layout's grid bounds-check rejects.  Clamping
                # the *assignment* (not the stored position) folds such boundary
                # points into the edge chunk while keeping their exact
                # coordinates, so no StoreError and no geometry change.
                seg_chunk_x.append(
                    min(max(int(cc[0]), grid_origin[0]), grid_origin[0] + grid_shape[0] - 1)
                )
                seg_chunk_y.append(
                    min(max(int(cc[1]), grid_origin[1]), grid_origin[1] + grid_shape[1] - 1)
                )
                seg_chunk_z.append(
                    min(max(int(cc[2]), grid_origin[2]), grid_origin[2] + grid_shape[2] - 1)
                )
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


def _close_memmap(arr: Any) -> None:
    """Release a ``np.load(..., mmap_mode=...)`` mapping's file handle.

    Dropping the last Python reference is not enough on Windows: the handle
    survives until GC runs, and until it does the backing file cannot be
    deleted — which is what makes the intermediate-dir cleanup blow up at
    the very end of an otherwise successful ingest.  ``np.memmap`` exposes
    the mapping as ``._mmap``; a plain ndarray (no mmap) has none.
    """
    mm = getattr(arr, "_mmap", None)
    if mm is not None:
        mm.close()


# ---------------------------------------------------------------------------
# Phase B worker: assemble one spatial chunk from all N parts
# ---------------------------------------------------------------------------

def _phase_b_worker(
    chunk_coords: tuple[int, int, int],
    shared: dict[str, Any],
) -> dict[str, Any]:
    """Write one spatial chunk's level-0 data from all part .npz files.

    Loads this chunk's segments from every part file in ascending
    (part_index, poly_id, seg_idx_within_poly) order — the determinism pin.

    Returns a dict mapping (part_index, poly_id, seg_idx_within_poly) →
    (chunk_coords, fragment_idx, first_local_row, last_local_row).
    """
    part_npz_paths = shared["part_npz_paths"]
    store_path = shared["store_path"]
    dtype = shared["dtype"]
    out_dtype = np.dtype(dtype)

    # Records for this chunk across all parts, in (part, poly_id, seg_within_poly) order
    # We collect: (part_idx, poly_id, seg_idx_within_poly, vertex_array)
    records: list[tuple[int, int, int, npt.NDArray]] = []

    for part_idx, npz_path in enumerate(part_npz_paths):
        # ``with``: NpzFile holds an open zipfile handle until closed, and
        # ``del`` only drops a reference — on Windows a surviving handle makes
        # the intermediate dir undeletable and the whole ingest fails in
        # cleanup, long after the data is safely written.
        with np.load(npz_path) as data:
            px = data["seg_chunk_x"]
            py = data["seg_chunk_y"]
            pz = data["seg_chunk_z"]
            mask = (
                (px == chunk_coords[0])
                & (py == chunk_coords[1])
                & (pz == chunk_coords[2])
            )
            if not np.any(mask):
                continue

            seg_poly_ids = data["seg_poly_ids"]
            seg_vtx_counts = data["seg_vertex_counts"]
            vtx_starts = np.concatenate([[0], np.cumsum(seg_vtx_counts)])[:-1]

            # Vectorised within_poly_idx: position of each segment within its
            # polyline.  seg_poly_ids is monotonically non-decreasing within
            # each part (Phase A writes segments in poly_id order), so group
            # boundaries are diff > 0.
            change_pts = np.concatenate(
                [[0], np.where(np.diff(seg_poly_ids))[0] + 1]
            )
            group_id = np.searchsorted(
                change_pts, np.arange(len(seg_poly_ids)), side="right"
            ) - 1
            within_poly_idx_arr = np.arange(len(seg_poly_ids)) - change_pts[group_id]

            # Restrict all arrays to matching segments — avoids an
            # O(all_segments) Python loop.
            match_idx = np.where(mask)[0]
            m_poly_ids    = seg_poly_ids[match_idx]
            m_within      = within_poly_idx_arr[match_idx]
            m_vtx_starts  = vtx_starts[match_idx]
            m_vtx_counts  = seg_vtx_counts[match_idx]

        verts_npy_path = npz_path.replace(".npz", ".verts.npy")
        vertices = np.load(verts_npy_path, mmap_mode="r")
        try:
            for i in range(len(match_idx)):
                vs = int(m_vtx_starts[i])
                vc = int(m_vtx_counts[i])
                # np.array() forces a copy: releases the mmap reference so pages
                # can be evicted once the mapping closes below.  Create it
                # directly in the output dtype so the write path needs no
                # further copy (a no-op cast when out_dtype is float32).
                seg_verts = np.array(vertices[vs:vs + vc], dtype=out_dtype)
                records.append(
                    (part_idx, int(m_poly_ids[i]), int(m_within[i]), seg_verts)
                )
        finally:
            # Close the mapping explicitly: `del` drops a reference but leaves
            # the file handle open until GC, which on Windows is long enough to
            # block cleanup of the intermediate dir.
            _close_memmap(vertices)

    if not records:
        return {}

    # Sort: (part_idx, poly_id, seg_within_poly) — the determinism pin
    records.sort(key=lambda r: (r[0], r[1], r[2]))

    # Write to zarr-vectors store
    root = open_store(str(store_path), mode="r+")
    # Level group must already exist (created in coordinator pre-phase-B)
    from zarr_vectors.core.store import get_resolution_level
    level_group = get_resolution_level(root, 0)

    # Reference the arrays already held in `records` (they are already
    # `out_dtype`) instead of copying — avoids a full duplicate of the whole
    # chunk's vertices and a second set of per-segment array objects.
    vert_groups = [r[3] for r in records]
    # record_presence=False: Phase B runs one task per spatial chunk, across
    # worker PROCESSES.  ``nonempty_chunks`` is an array-wide attribute living
    # in the array's zarr.json, so stamping it per cell is a read-modify-write
    # that concurrent workers collide on (a Windows atomic-rename hard-fail).
    # The coordinator re-derives every manifest once, after Phase B, via
    # rebuild_nonempty_manifests.
    write_chunk_vertices(
        level_group, chunk_coords, vert_groups, dtype=out_dtype,
        record_presence=False,
    )

    # Write per-fragment segment id (global poly_id = streamline index in file).
    # One uint64 per fragment, same order as vert_groups.  Neuroglancer uses this
    # to map a picked spatial fragment back to the full streamline for pass-2 fetch.
    seg_ids = np.array([r[1] for r in records], dtype=np.uint64)
    write_chunk_fragment_attributes(
        level_group, "segment_id", chunk_coords, seg_ids, dtype=np.uint64,
        record_presence=False,
    )

    # Build fragment result map: key = (part_idx, poly_id, seg_within_poly)
    result: dict[tuple[int, int, int], tuple[tuple[int, int, int], int, int, int]] = {}
    cum_row = 0
    for frag_idx, (part_idx, poly_id, seg_within, seg_verts) in enumerate(records):
        first_row = cum_row
        last_row = cum_row + len(seg_verts) - 1
        result[(part_idx, poly_id, seg_within)] = (chunk_coords, frag_idx, first_row, last_row)
        cum_row += len(seg_verts)

    return result


# ---------------------------------------------------------------------------
# Coordinator: reconstruct manifests and cross-chunk links
# ---------------------------------------------------------------------------

def _build_manifests_and_cross_links(
    part_npz_paths: list[str],
    phase_b_results: list[dict[str, Any]],
    n_total_streamlines: int,
) -> tuple[dict[int, list], list]:
    """Build object_manifests and cross_chunk_links from Phase B fragment maps.

    Returns:
        object_manifests  dict[poly_id → list[(chunk_coords, fragment_idx)]]
        cross_chunk_links  list[CrossChunkLink]
    """
    # Merge all Phase B result dicts into one global map
    global_map: dict[tuple[int, int, int], tuple[tuple[int, int, int], int, int, int]] = {}
    for d in phase_b_results:
        global_map.update(d)

    # For each poly_id, rebuild the ordered segment list from part npz files
    # We need: for each poly_id, the ordered segments (part, poly_id, within_poly_idx)
    # We reconstruct this by scanning the npz files in part order.

    # Build poly_id → sorted list of (part_idx, within_poly_idx, chunk_coords)
    poly_segments: dict[int, list[tuple[int, int, tuple[int, int, int]]]] = {}

    for part_idx, npz_path in enumerate(part_npz_paths):
        with np.load(npz_path) as data:
            seg_poly_ids = data["seg_poly_ids"]
            seg_chunk_x = data["seg_chunk_x"]
            seg_chunk_y = data["seg_chunk_y"]
            seg_chunk_z = data["seg_chunk_z"]

        prev_poly = -1
        within_poly_idx = 0
        for i in range(len(seg_poly_ids)):
            poly_id = int(seg_poly_ids[i])
            if poly_id != prev_poly:
                within_poly_idx = 0
                prev_poly = poly_id
            cc = (int(seg_chunk_x[i]), int(seg_chunk_y[i]), int(seg_chunk_z[i]))
            poly_segments.setdefault(poly_id, []).append((part_idx, within_poly_idx, cc))
            within_poly_idx += 1

    object_manifests: dict[int, list] = {}
    cross_chunk_links: list = []

    for poly_id in range(n_total_streamlines):
        segs = poly_segments.get(poly_id)
        if not segs:
            object_manifests[poly_id] = []
            continue

        manifest: list[tuple[tuple[int, int, int], int]] = []
        manifest_with_indices: list[tuple[tuple[int, int, int], int, int, int]] = []

        for part_idx, within_poly_idx, cc in segs:
            key = (part_idx, poly_id, within_poly_idx)
            info = global_map.get(key)
            if info is None:
                continue
            out_cc, frag_idx, first_row, last_row = info
            manifest.append((out_cc, frag_idx))
            manifest_with_indices.append((out_cc, frag_idx, first_row, last_row))

        object_manifests[poly_id] = manifest

        # Cross-chunk links: consecutive segments in different chunks
        if len(manifest_with_indices) > 1:
            for i in range(len(manifest_with_indices) - 1):
                cc_a, _, _, last_a = manifest_with_indices[i]
                cc_b, _, first_b, _ = manifest_with_indices[i + 1]
                if cc_a != cc_b:
                    cross_chunk_links.append(((cc_a, last_a), (cc_b, first_b)))

    return object_manifests, cross_chunk_links


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
    compressor: Any = None,
    max_streamlines: int | None = None,
    compute_length: bool = False,
    compute_endpoints: bool = False,
    preserve_header: bool = True,
    register_to_rasmm: bool = False,
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
        compressor: Codec for the level-0 per-chunk arrays.  ``None``
            (default) stores raw — vertices then cost exactly
            ``n_vertices * ndim * itemsize`` bytes, the same payload a .trk
            holds.  ``"zstd"`` / ``"blosc"`` roughly halve that (measured
            ~2.4x on HCP tract coordinates) at the cost of a slower,
            sync codec-encoding write path.  See
            :func:`zarr_vectors.encoding.compression.resolve_compressor`
            for accepted values (a full codec list also works).  The codec
            pipeline is fixed when each array is created, so every later
            per-cell write — including from parallel workers — encodes to
            match automatically.
        max_streamlines: If set, only ingest the first N streamlines from the
            file (in on-disk order). Useful for quick test runs on a subset
            of a large tractogram; leave None to ingest all streamlines.
        compute_length: Write per-streamline path length to object_attributes.
        compute_endpoints: Write start/end points to object_attributes.
        preserve_header: Store TRKHeader (affine, dims) in the zarr store.
        register_to_rasmm: When True, bake the trackvis-voxmm→RASmm affine into
            the stored vertex positions (and chunk grid) and record an identity
            CRS affine, so the store is in registered world space with no
            transform left for the viewer to apply — the layout neuroglancer
            needs.  When False (default) the raw voxmm coordinates are kept and
            the real ``vox_to_ras`` affine is recorded in CRS metadata instead.
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
    # Register to RASmm world space by baking the trackvis-voxmm→RASmm affine
    # into the geometry (applied per streamline in Phase A).  Derive the store
    # bounds/chunk grid in that same RASmm space so the chunk layout matches the
    # stored coordinates.  Without this the store sits in voxmm — shifted to the
    # chunk-grid origin and axis-flipped relative to the source image.
    affine_tv2ras = _trackvis_to_rasmm_affine(input_path) if register_to_rasmm else None
    if affine_tv2ras is not None:
        bounds = _transform_bounds(bounds, affine_tv2ras)
    chunk_shape = _compute_chunk_shape(bounds, num_chunks)
    lo, hi = bounds
    # Use the SAME (origin, grid_shape) the single-array layout derives for the
    # on-disk arrays (floor-based, +1 to hold a point exactly on the max
    # boundary) so the chunk-assignment clamp in Phase A matches the array grid
    # exactly.  A ceil-based grid is one cell short on axes whose extent is an
    # exact multiple of the chunk size (e.g. 189/9 = 21 cells can't hold the
    # boundary chunk 21, which needs a grid of 22).
    grid_origin, grid_shape = level_grid_layout(bounds, chunk_shape)
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
        # ignore_cleanup_errors: by the time cleanup runs the store is written
        # and the summary returned, so a straggling intermediate file must
        # never fail the ingest.  Every read site closes its handle explicitly
        # (see _close_memmap and the `with np.load(...)` blocks); this is the
        # backstop, and it costs temp space rather than correctness.
        _tmpdir_obj = tempfile.TemporaryDirectory(
            prefix="trk_parallel_", ignore_cleanup_errors=True,
        )
        _intermediate_dir = _tmpdir_obj.name
    else:
        _intermediate_dir = str(intermediate_dir)
        Path(_intermediate_dir).mkdir(parents=True, exist_ok=True)
        _tmpdir_obj = None

    try:
        # --- Create zarr-vectors store ------------------------------------
        _log("Creating zarr-vectors store...")
        vox_to_ras = header["vox_to_ras"]
        if affine_tv2ras is not None:
            # Geometry is already baked to RASmm, so the store-space transform
            # is the identity — a crs-aware viewer applying it must be a no-op
            # (otherwise it would double-transform).  The original voxmm→RASmm
            # affine is preserved in the Phase-5 TRKHeader metadata for
            # provenance / round-trip export.
            crs_dict = {
                "input_space": "RASmm",
                "output_space": "RASmm",
                "units": "mm",
                "affine": np.eye(4, dtype=np.float64).flatten().tolist(),
            }
        else:
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
            # create_store warm-creates vertices/vertex_fragments, and a chunk
            # array's codecs are fixed at creation — so the compressor has to
            # be set HERE or the store's largest arrays stay raw forever.
            compressor=compressor,
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
        # A chunk array's codec pipeline is fixed when the array is CREATED
        # (it lands in the array's zarr.json), and every later write — the
        # Phase B workers included, in their own processes — encodes to match
        # it via zarr's array API.  So the compressor only has to be active
        # here, around creation; there is nothing to thread into the workers.
        # ``nullcontext`` when no compressor is asked for, so the default path
        # is byte-for-byte what it was before this option existed.
        _codec_ctx = (
            level_group.batched_writes(compressor=compressor)
            if compressor else nullcontext()
        )
        with _codec_ctx:
            create_vertices_array(level_group, dtype=dtype)
            create_object_index_array(level_group)
            # Stamp the family policy only; streamline connectivity is
            # implicit-sequential within a fragment, so there are no intra
            # records and the only arrays that should appear under links/0 are
            # the ones the boundary-crossing records actually land in.
            create_links_family(
                level_group, delta=0, link_width=2, sid_ndim=3, directed=True,
            )
            create_fragment_attribute_array(
                level_group, "segment_id", dtype="uint64",
            )

        # --- Phase A: per-part spatial binning ---------------------------
        _log(f"Phase A: binning streamlines ({len(part_specs)} parts)...")

        shared_a = {
            "trk_path": str(input_path),
            "header": header,
            "chunk_shape": chunk_shape,
            "grid_shape": grid_shape,
            "grid_origin": grid_origin,
            "affine_tv2ras": affine_tv2ras,
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

        # Enumerate all occupied chunk coords across all parts
        all_chunk_coords: set[tuple[int, int, int]] = set()
        for npz_path in part_npz_paths:
            with np.load(npz_path) as data:
                xs = data["seg_chunk_x"].tolist()
                ys = data["seg_chunk_y"].tolist()
                zs = data["seg_chunk_z"].tolist()
            for x, y, z in zip(xs, ys, zs):
                all_chunk_coords.add((int(x), int(y), int(z)))
        _log(f"  {len(all_chunk_coords)} occupied spatial chunks")

        # --- Phase B: per-chunk level-0 write ----------------------------
        _log(f"Phase B: writing level-0 ({len(all_chunk_coords)} chunks)...")

        shared_b = {
            "part_npz_paths": part_npz_paths,
            "store_path": str(output_path),
            "dtype": dtype,
        }

        chunk_list = sorted(all_chunk_coords)

        if executor is not None:
            phase_b_results = executor(_phase_b_worker, chunk_list, shared=shared_b)
        else:
            phase_b_results = [_phase_b_worker(cc, shared_b) for cc in chunk_list]

        # Phase B workers each read-modify-write the shared per-chunk arrays'
        # ``nonempty_chunks`` manifest from separate processes, so those RMWs
        # race and can under-report even though every cell file is on disk.
        # Re-derive the manifests single-process from the on-disk cells before
        # anything downstream enumerates chunks (coarsening source scan,
        # algorithms readers).
        rebuild_nonempty_manifests(level_group)

        # --- Coordinator: manifests + cross-chunk links -------------------
        _log("Coordinator: building manifests and cross-chunk links...")
        object_manifests, cross_chunk_links = _build_manifests_and_cross_links(
            part_npz_paths, phase_b_results, n_streamlines,
        )

        _log(f"  writing object index ({n_streamlines} streamlines)...")
        write_object_index(level_group, object_manifests, sid_ndim=3)

        if cross_chunk_links:
            _log(f"  writing {len(cross_chunk_links)} cross-chunk links...")
            # No capability stamp: these are cross-CHUNK links at delta 0.
            # Since the links merge that is just a non-zero offsets segment
            # in the ordinary links/0 family, and multiscale_links marks the
            # presence of delta != 0 (cross-LEVEL) arrays only.
            write_links(
                level_group, cross_chunk_links, sid_ndim=3, delta=0,
                mode="replace", directed=True,
            )

        # Object attributes
        obj_attrs_to_write: dict[str, npt.NDArray] = {}
        if compute_length:
            length_parts = []
            for npz_path in part_npz_paths:
                with np.load(npz_path) as d:
                    if "lengths" in d:
                        length_parts.append(d["lengths"])
            if length_parts:
                obj_attrs_to_write["length"] = np.concatenate(length_parts, axis=0)

        if compute_endpoints:
            start_parts, end_parts = [], []
            for npz_path in part_npz_paths:
                with np.load(npz_path) as d:
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
                # Coarser levels create their own arrays, so they need the
                # same compressor to keep the store uniform — level 0's codec
                # does not propagate.
                compressor=compressor,
                executor=executor,
            )

        # Count total vertices written (seg_vertex_counts is always in the npz)
        total_vertices = 0
        for p in part_npz_paths:
            with np.load(p) as d:
                total_vertices += int(d["seg_vertex_counts"].sum())

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
                                 "links", "object_attributes",
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
            # Matches the key core's write_polylines returns, so the CLI
            # prints the same field whichever ingest path produced it.
            # "cross-chunk link" still names the concept (a record whose
            # endpoints span chunks); the merge changed where it is stored,
            # not what it counts.
            "cross_chunk_link_count": len(cross_chunk_links),
            "n_parts": len(part_specs),
            "chunk_shape": chunk_shape,
            "bounds": bounds,
            "pyramid": pyramid_summary,
        }

    finally:
        if _tmpdir_obj is not None and not keep_intermediate:
            _tmpdir_obj.cleanup()
