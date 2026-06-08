"""Mesh coarsening strategies.

Two approaches:

1. **Vertex clustering**: merge vertices within spatial bins into
   metanodes, remap face indices, remove degenerate faces (collapsed
   to edges or points), and merge duplicate faces.

2. **Quadric error decimation** (simplified): iterative edge collapse
   using quadric error metrics for vertex placement.  Falls back to
   vertex clustering when the optional ``pyfqmr`` is not available.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors_tools.multiresolution.metanodes import generate_metanodes


# ===================================================================
# Vertex clustering
# ===================================================================

def coarsen_mesh_cluster(
    vertices: npt.NDArray[np.floating],
    faces: npt.NDArray[np.integer],
    bin_size: float | tuple[float, ...],
    *,
    vertex_attributes: dict[str, npt.NDArray] | None = None,
    agg_mode: str = "mean",
) -> dict[str, Any]:
    """Coarsen a mesh by clustering vertices within spatial bins.

    Vertices in the same bin merge into a metanode (centroid).  Face
    indices are remapped; faces that collapse to degenerate triangles
    (two or more vertices in the same bin) are removed.

    Args:
        vertices: ``(V, D)`` vertex positions.
        faces: ``(F, L)`` face index array.
        bin_size: Spatial bin edge length.
        vertex_attributes: Per-vertex attributes to aggregate.
        agg_mode: Attribute aggregation mode.

    Returns:
        Dict with:
        - ``vertices``: ``(K, D)`` metanode positions
        - ``faces``: ``(E, L)`` remapped faces (non-degenerate only)
        - ``vertex_attributes``: aggregated attributes
        - ``children``: list of K arrays of original vertex indices
        - ``vertex_count``, ``face_count``
        - ``vertex_reduction``: V / K
        - ``face_reduction``: F / E
        - ``degenerate_faces_removed``: count
    """
    n_verts, ndim = vertices.shape
    n_faces, link_width = faces.shape

    if n_verts == 0:
        return _empty_mesh_coarsen(ndim, link_width)

    # Generate metanodes from vertices
    meta_result = generate_metanodes(
        vertices, bin_size,
        attributes=vertex_attributes,
        agg_mode=agg_mode,
    )

    meta_verts = meta_result["metanode_positions"]
    children = meta_result["children"]
    meta_attrs = meta_result["metanode_attributes"]
    n_meta = len(meta_verts)

    # Build vertex → metanode mapping
    vert_to_meta = np.empty(n_verts, dtype=np.int64)
    for m_idx in range(n_meta):
        for c in children[m_idx]:
            vert_to_meta[c] = m_idx

    # Remap face indices
    remapped_faces = vert_to_meta[faces]  # (F, L)

    # Remove degenerate faces: faces where any two vertices map to same metanode
    keep_mask = np.ones(n_faces, dtype=bool)
    for i in range(link_width):
        for j in range(i + 1, link_width):
            keep_mask &= remapped_faces[:, i] != remapped_faces[:, j]

    valid_faces = remapped_faces[keep_mask]

    # Remove duplicate faces (sort vertex indices per face, then unique)
    if len(valid_faces) > 0:
        sorted_faces = np.sort(valid_faces, axis=1)
        _, unique_idx = np.unique(sorted_faces, axis=0, return_index=True)
        valid_faces = valid_faces[np.sort(unique_idx)]

    degenerate_count = n_faces - int(keep_mask.sum())
    duplicate_count = int(keep_mask.sum()) - len(valid_faces)

    return {
        "vertices": meta_verts,
        "faces": valid_faces,
        "vertex_attributes": meta_attrs,
        "children": children,
        "vertex_count": n_meta,
        "face_count": len(valid_faces),
        "vertex_reduction": n_verts / max(n_meta, 1),
        "face_reduction": n_faces / max(len(valid_faces), 1),
        "degenerate_faces_removed": degenerate_count + duplicate_count,
    }


# ===================================================================
# Quadric error decimation (simplified)
# ===================================================================

def coarsen_mesh_quadric(
    vertices: npt.NDArray[np.floating],
    faces: npt.NDArray[np.integer],
    target_face_count: int | None = None,
    target_ratio: float | None = None,
    *,
    vertex_attributes: dict[str, npt.NDArray] | None = None,
) -> dict[str, Any]:
    """Coarsen a mesh using quadric error edge collapse.

    Uses ``pyfqmr`` if available; otherwise falls back to vertex
    clustering with an estimated bin size.

    Args:
        vertices: ``(V, 3)`` vertex positions (3D only).
        faces: ``(F, 3)`` triangle face indices.
        target_face_count: Desired face count.  Mutually exclusive
            with ``target_ratio``.
        target_ratio: Fraction of faces to keep (0–1).
        vertex_attributes: Per-vertex attributes (interpolated if
            pyfqmr is used, aggregated otherwise).

    Returns:
        Dict with vertex/face arrays, counts, and method used.
    """
    n_verts = len(vertices)
    n_faces = len(faces)

    if target_ratio is not None:
        target_face_count = max(1, int(n_faces * target_ratio))
    elif target_face_count is None:
        target_face_count = max(1, n_faces // 4)

    # Try pyfqmr
    try:
        return _quadric_pyfqmr(vertices, faces, target_face_count, vertex_attributes)
    except ImportError:
        pass

    # Fallback: estimate bin size from target reduction
    reduction = n_faces / max(target_face_count, 1)
    # Rough heuristic: bin_size ≈ extent * reduction^(1/3)
    extent = np.max(vertices, axis=0) - np.min(vertices, axis=0)
    mean_extent = float(np.mean(extent))
    estimated_bin = mean_extent * (reduction ** (1.0 / 3.0)) / (n_verts ** (1.0 / 3.0)) * 2

    result = coarsen_mesh_cluster(
        vertices, faces, estimated_bin,
        vertex_attributes=vertex_attributes,
    )
    result["method"] = "vertex_clustering_fallback"
    return result


def _quadric_pyfqmr(
    vertices: npt.NDArray,
    faces: npt.NDArray,
    target_faces: int,
    vertex_attributes: dict[str, npt.NDArray] | None,
) -> dict[str, Any]:
    """Quadric decimation via pyfqmr."""
    import pyfqmr

    mesh = pyfqmr.Simplify()
    mesh.setMesh(
        np.ascontiguousarray(vertices, dtype=np.float64),
        np.ascontiguousarray(faces, dtype=np.int32),
    )
    mesh.simplify_mesh(target_count=target_faces, aggressiveness=7)

    new_verts = np.array(mesh.getMesh()[0], dtype=np.float32)
    new_faces = np.array(mesh.getMesh()[1], dtype=np.int64)

    result: dict[str, Any] = {
        "vertices": new_verts,
        "faces": new_faces,
        "vertex_count": len(new_verts),
        "face_count": len(new_faces),
        "vertex_reduction": len(vertices) / max(len(new_verts), 1),
        "face_reduction": len(faces) / max(len(new_faces), 1),
        "method": "quadric_pyfqmr",
    }

    # Attributes: nearest-vertex interpolation
    if vertex_attributes:
        from scipy.spatial import cKDTree
        tree = cKDTree(vertices)
        _, nearest = tree.query(new_verts)
        result["vertex_attributes"] = {
            name: data[nearest] for name, data in vertex_attributes.items()
        }
    else:
        result["vertex_attributes"] = {}

    return result


def _empty_mesh_coarsen(ndim: int, link_width: int) -> dict[str, Any]:
    return {
        "vertices": np.zeros((0, ndim), dtype=np.float32),
        "faces": np.zeros((0, link_width), dtype=np.int64),
        "vertex_attributes": {},
        "children": [],
        "vertex_count": 0,
        "face_count": 0,
        "vertex_reduction": 0,
        "face_reduction": 0,
        "degenerate_faces_removed": 0,
    }
