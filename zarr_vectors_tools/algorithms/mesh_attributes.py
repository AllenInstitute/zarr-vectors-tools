"""Post-hoc per-vertex mesh attributes (normals, mean curvature).

Streaming accumulation.  Each chunk's own faces are accumulated as the chunk
is read.  Faces whose corners lie in more than one chunk are read once, up
front; their corner positions are collected as the chunks stream past and
they are accumulated at the end (see
:class:`~zarr_vectors_tools.algorithms._mesh_faces.SpanningFaces`), so a
vertex on a chunk boundary gets every face around it and the result does not
depend on the chunk shape.  Faces of four or more corners are split into a
fan of triangles first.

``write_back=True`` persists the result via
:func:`zarr_vectors_tools._attributes.write_vertex_attribute` under
``attributes/<name>/`` (``vertex_normal`` for normals,
``mean_curvature`` for curvature).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from zarr_vectors.building import (
    chunk_local_to_global_offsets,
    get_resolution_level,
    open_store,
)
from zarr_vectors.constants import (
    VERTEX_FRAGMENTS,
    VERTICES,
)

from zarr_vectors_tools._attributes import write_vertex_attribute
from zarr_vectors_tools.algorithms._links import (
    chunk_key_str,
    link_prefetch_plan,
)
from zarr_vectors_tools.algorithms._mesh_faces import (
    SpanningFaces,
    chunk_faces,
    chunk_positions,
    face_width,
    fan,
)

# ``(global ids (T, 3), a, b, c)`` for a batch of triangles.
_Accumulate = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None]


def _for_each_triangle(level_group, layout, width: int, accumulate: _Accumulate) -> None:
    """Feed every triangle of the level to ``accumulate``.

    ``layout`` is ``chunk_local_to_global_offsets(level_group)``; global
    vertex ids are its numbering, the order core's ``read_mesh`` returns
    vertices in.
    """
    offsets, chunk_keys, _n_vertices = layout

    chunk_key_strs = [chunk_key_str(cc) for cc in chunk_keys]
    with level_group.batched_reads([
        (VERTICES, chunk_key_strs),
        (VERTEX_FRAGMENTS, chunk_key_strs),
        *link_prefetch_plan(level_group, chunk_keys),
    ]):
        spanning = SpanningFaces(level_group)
        for chunk_key in chunk_keys:
            positions = chunk_positions(level_group, chunk_key)
            if positions is None:
                continue
            spanning.collect(chunk_key, positions)
            base = offsets[chunk_key]
            for faces in chunk_faces(level_group, chunk_key, width):
                triangles, _ = fan(faces)
                accumulate(
                    triangles + base,
                    positions[triangles[:, 0]],
                    positions[triangles[:, 1]],
                    positions[triangles[:, 2]],
                )

    if len(spanning):
        spanning.require_complete()
        a, b, c, slots, _source = spanning.triangles()
        accumulate(spanning.global_rows(offsets).reshape(-1)[slots], a, b, c)


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
            sums each incident triangle's un-normalised normal; uniform sums
            unit triangle normals.
        write_back: When True, persist the result under
            ``attributes/vertex_normal/`` via
            :func:`~zarr_vectors_tools._attributes.write_vertex_attribute`.

    Returns:
        Dict with:
          - ``normals``: ``(N, 3) float32`` — per-vertex unit normal in
            the store's global vertex ordering.
          - ``incomplete_boundary_vertices`` (int): always 0, since every
            face is used; kept for callers that read it.

    Raises:
        NotImplementedError: If the level holds no faces of three or more
            corners.
        ValueError: If ``weighting`` is unknown, or a face spanning chunks
            names a vertex the level does not hold.
    """
    if weighting not in ("area", "uniform"):
        raise ValueError(f"unknown weighting={weighting!r}")

    root = open_store(str(store_path))
    level_group = get_resolution_level(root, level)
    width = face_width(level_group, what="compute_vertex_normals")
    layout = chunk_local_to_global_offsets(level_group)

    normals = np.zeros((layout[2], 3), dtype=np.float64)

    def accumulate(ids, v0, v1, v2) -> None:
        face_n = np.cross(v1 - v0, v2 - v0)
        if weighting == "uniform":
            lens = np.linalg.norm(face_n, axis=1, keepdims=True)
            safe = np.where(lens == 0, 1.0, lens)
            face_n = face_n / safe
        for col in (0, 1, 2):
            np.add.at(normals, ids[:, col], face_n)

    _for_each_triangle(level_group, layout, width, accumulate)

    norm_lens = np.linalg.norm(normals, axis=1, keepdims=True)
    safe = np.where(norm_lens == 0, 1.0, norm_lens)
    normals_unit = (normals / safe).astype(np.float32)

    if write_back:
        # A second, writable handle: the read path above opens mode="r".
        write_vertex_attribute(
            get_resolution_level(open_store(str(store_path), mode="r+"), level),
            "vertex_normal", normals_unit, dtype=np.float32,
        )

    return {
        "normals": normals_unit,
        "incomplete_boundary_vertices": 0,
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

    Args:
        store_path: Path to the mesh store.
        level: Resolution level.
        write_back: When True, persist the result under
            ``attributes/mean_curvature/`` via
            :func:`~zarr_vectors_tools._attributes.write_vertex_attribute`.

    Returns:
        Dict with:
          - ``mean_curvature``: ``(N,) float32``.
          - ``incomplete_boundary_vertices`` (int): always 0, since every
            face is used; kept for callers that read it.

    Raises:
        NotImplementedError: If the level holds no faces of three or more
            corners.
        ValueError: If a face spanning chunks names a vertex the level does
            not hold.
    """
    root = open_store(str(store_path))
    level_group = get_resolution_level(root, level)
    width = face_width(level_group, what="compute_mean_curvature")
    layout = chunk_local_to_global_offsets(level_group)
    n_vertices = layout[2]

    H = np.zeros((n_vertices, 3), dtype=np.float64)
    area_voronoi = np.zeros(n_vertices, dtype=np.float64)

    def _cot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Cotangent of the angle between rows of a and b."""
        dot = np.einsum("ij,ij->i", a, b)
        cr = np.linalg.norm(np.cross(a, b), axis=1)
        safe = np.where(cr == 0, 1.0, cr)
        return dot / safe

    def accumulate(ids, v0, v1, v2) -> None:
        global_0, global_1, global_2 = ids[:, 0], ids[:, 1], ids[:, 2]

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

    _for_each_triangle(level_group, layout, width, accumulate)

    safe_area = np.where(area_voronoi == 0, 1.0, area_voronoi)
    H_per_vertex = H / (2.0 * safe_area[:, None])
    mean_curv = (np.linalg.norm(H_per_vertex, axis=1) * 0.5).astype(np.float32)

    if write_back:
        # A second, writable handle: the read path above opens mode="r".
        write_vertex_attribute(
            get_resolution_level(open_store(str(store_path), mode="r+"), level),
            "mean_curvature", mean_curv, dtype=np.float32,
        )

    return {
        "mean_curvature": mean_curv,
        "incomplete_boundary_vertices": 0,
    }
