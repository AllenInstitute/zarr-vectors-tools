"""Post-hoc per-vertex mesh attributes (normals, mean curvature).

Streaming intra-chunk accumulation. Cross-chunk faces (records of
arity >= 3 spanning more than one chunk) contribute their endpoints
to the boundary-vertex set but their per-face contributions are not
accumulated — boundary-vertex normals/curvatures will be slightly off
near chunk boundaries. The returned dict reports
``incomplete_boundary_vertices`` so callers can quantify the effect.

``write_back=True`` persists the result via
:class:`zarr_vectors.lazy.ZVWriter.add_node_attribute_sync` under
``attributes/<name>/`` (``vertex_normal`` for normals,
``mean_curvature`` for curvature).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from zarr_vectors.constants import (
    VERTEX_FRAGMENTS,
    VERTICES,
)
from zarr_vectors.core.arrays import (
    list_chunk_keys,
    read_chunk_links,
    read_chunk_vertices,
)
from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors.lazy import open_zv
from zarr_vectors.spatial.boundary import chunk_local_to_global_offsets

from zarr_vectors_tools.algorithms._links import (
    chunk_key_str,
    link_prefetch_plan,
    read_cross_links,
)


def _read_triangles(level_group, chunk_key):
    """Yield (positions, triangle_groups) for a chunk; missing data returns None."""
    try:
        vgroups = read_chunk_vertices(level_group, chunk_key)
    except Exception:
        return None, None
    if not vgroups:
        return None, None
    positions = np.concatenate(vgroups, axis=0).astype(np.float64)
    try:
        link_groups = read_chunk_links(level_group, chunk_key)
    except Exception:
        return positions, []
    triangles: list[np.ndarray] = []
    for lg in link_groups:
        if not len(lg):
            continue
        arr = np.asarray(lg, dtype=np.int64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            # Non-triangle face data; skip rather than miscompute normals.
            continue
        triangles.append(arr)
    return positions, triangles


def _count_boundary_vertex_set(cross_links) -> set[tuple]:
    """Vertices appearing in any cross-chunk record are 'boundary'.

    Each record may carry 2 endpoints (cross-chunk edge) or 3+ (cross-chunk
    face). We accumulate every endpoint regardless of link_width.
    """
    boundary: set[tuple] = set()
    for record in cross_links:
        for chunk, vi in record:
            boundary.add((chunk, int(vi)))
    return boundary


def compute_vertex_normals(
    store_path: str | Path,
    *,
    level: int = 0,
    weighting: str = "area",
    write_back: bool = False,
) -> dict[str, Any]:
    """Compute per-vertex normals over a chunked mesh store.

    Args:
        store_path: Path to the mesh store.
        level: Resolution level.
        weighting: ``"area"`` (default) or ``"uniform"``. Area-weighted
            sums each incident face's un-normalised normal; uniform sums
            unit face normals.
        write_back: When True, persist the result under
            ``attributes/vertex_normal/`` via
            :meth:`ZVWriter.add_node_attribute_sync`.

    Returns:
        Dict with:
          - ``normals``: ``(N, 3) float32`` — per-vertex unit normal in
            the store's global vertex ordering.
          - ``incomplete_boundary_vertices`` (int): count of vertices
            that appear in cross-chunk edges; their normals are based
            on intra-chunk faces only.
    """
    if weighting not in ("area", "uniform"):
        raise ValueError(f"unknown weighting={weighting!r}")

    root = open_store(str(store_path))
    level_group = get_resolution_level(root, level)
    offsets, chunk_keys, n_vertices = chunk_local_to_global_offsets(level_group)

    normals = np.zeros((n_vertices, 3), dtype=np.float64)

    chunk_key_strs = [chunk_key_str(cc) for cc in chunk_keys]
    with level_group.batched_reads([
        (VERTICES, chunk_key_strs),
        (VERTEX_FRAGMENTS, chunk_key_strs),
        *link_prefetch_plan(level_group, chunk_keys),
    ]):
        for chunk_key in chunk_keys:
            positions, faces_list = _read_triangles(level_group, chunk_key)
            if positions is None:
                continue
            base = offsets[chunk_key]
            for faces in faces_list:
                v0 = positions[faces[:, 0]]
                v1 = positions[faces[:, 1]]
                v2 = positions[faces[:, 2]]
                face_n = np.cross(v1 - v0, v2 - v0)
                if weighting == "uniform":
                    lens = np.linalg.norm(face_n, axis=1, keepdims=True)
                    safe = np.where(lens == 0, 1.0, lens)
                    face_n = face_n / safe

                for col in (0, 1, 2):
                    np.add.at(normals, faces[:, col] + base, face_n)

        # Cross-only: the per-chunk loop above already consumed every
        # intra record, so the whole-family read_links would double-count.
        try:
            cross_links = read_cross_links(level_group, delta=0)
        except Exception:
            cross_links = []
    boundary = _count_boundary_vertex_set(cross_links)

    norm_lens = np.linalg.norm(normals, axis=1, keepdims=True)
    safe = np.where(norm_lens == 0, 1.0, norm_lens)
    normals_unit = (normals / safe).astype(np.float32)

    if write_back:
        zv = open_zv(str(store_path))
        with zv[level].writer() as w:
            w.add_node_attribute_sync(
                "vertex_normal", normals_unit, dtype=np.float32,
            )

    return {
        "normals": normals_unit,
        "incomplete_boundary_vertices": len(boundary),
    }


def compute_mean_curvature(
    store_path: str | Path,
    *,
    level: int = 0,
    write_back: bool = False,
) -> dict[str, Any]:
    """Cotangent Laplace–Beltrami mean curvature per vertex.

    Implements Meyer et al. 2003. Per-vertex output is ``‖H(i)‖ / 2``
    where ``H(i) = (1/(2A_i)) Σ_j (cot α_ij + cot β_ij)(x_i − x_j)``.

    Cross-chunk faces are not contributed; boundary-vertex curvatures
    are intra-only and the count is reported.

    Args:
        store_path: Path to the mesh store.
        level: Resolution level.
        write_back: When True, persist the result under
            ``attributes/mean_curvature/`` via
            :meth:`ZVWriter.add_node_attribute_sync`.

    Returns:
        Dict with:
          - ``mean_curvature``: ``(N,) float32``.
          - ``incomplete_boundary_vertices`` (int).
    """
    root = open_store(str(store_path))
    level_group = get_resolution_level(root, level)
    offsets, chunk_keys, n_vertices = chunk_local_to_global_offsets(level_group)

    H = np.zeros((n_vertices, 3), dtype=np.float64)
    area_voronoi = np.zeros(n_vertices, dtype=np.float64)

    def _cot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Cotangent of the angle between rows of a and b."""
        dot = np.einsum("ij,ij->i", a, b)
        cr = np.linalg.norm(np.cross(a, b), axis=1)
        safe = np.where(cr == 0, 1.0, cr)
        return dot / safe

    chunk_key_strs = [chunk_key_str(cc) for cc in chunk_keys]
    with level_group.batched_reads([
        (VERTICES, chunk_key_strs),
        (VERTEX_FRAGMENTS, chunk_key_strs),
        *link_prefetch_plan(level_group, chunk_keys),
    ]):
        for chunk_key in chunk_keys:
            positions, faces_list = _read_triangles(level_group, chunk_key)
            if positions is None:
                continue
            base = offsets[chunk_key]
            for faces in faces_list:
                v0 = positions[faces[:, 0]]
                v1 = positions[faces[:, 1]]
                v2 = positions[faces[:, 2]]
                global_0 = faces[:, 0] + base
                global_1 = faces[:, 1] + base
                global_2 = faces[:, 2] + base

                # Edge vectors as seen from each vertex.
                cot_at_0 = _cot(v1 - v0, v2 - v0)  # angle at v0
                cot_at_1 = _cot(v0 - v1, v2 - v1)  # angle at v1
                cot_at_2 = _cot(v0 - v2, v1 - v2)  # angle at v2

                # Edge contributions (opposite-angle weighting).
                # Edge (v1, v2) uses cot_at_0; etc.
                edge12_diff = (v1 - v2) * cot_at_0[:, None]
                edge20_diff = (v2 - v0) * cot_at_1[:, None]
                edge01_diff = (v0 - v1) * cot_at_2[:, None]

                np.add.at(H, global_1, +edge12_diff)
                np.add.at(H, global_2, -edge12_diff)
                np.add.at(H, global_2, +edge20_diff)
                np.add.at(H, global_0, -edge20_diff)
                np.add.at(H, global_0, +edge01_diff)
                np.add.at(H, global_1, -edge01_diff)

                # Voronoi (or barycentric for obtuse) area per vertex.
                face_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
                # Use barycentric (face_area / 3 to each vertex) — a simpler
                # and still standard approximation. The full Voronoi case
                # introduces obtuse-triangle special handling that's not
                # essential for v0.
                contribution = face_area / 3.0
                np.add.at(area_voronoi, global_0, contribution)
                np.add.at(area_voronoi, global_1, contribution)
                np.add.at(area_voronoi, global_2, contribution)

        # Cross-only: the per-chunk loop above already consumed every
        # intra record, so the whole-family read_links would double-count.
        try:
            cross_links = read_cross_links(level_group, delta=0)
        except Exception:
            cross_links = []

    safe_area = np.where(area_voronoi == 0, 1.0, area_voronoi)
    H_per_vertex = H / (2.0 * safe_area[:, None])
    mean_curv = (np.linalg.norm(H_per_vertex, axis=1) * 0.5).astype(np.float32)

    boundary = _count_boundary_vertex_set(cross_links)

    if write_back:
        zv = open_zv(str(store_path))
        with zv[level].writer() as w:
            w.add_node_attribute_sync(
                "mean_curvature", mean_curv, dtype=np.float32,
            )

    return {
        "mean_curvature": mean_curv,
        "incomplete_boundary_vertices": len(boundary),
    }
