"""Tests for compute_vertex_normals and compute_mean_curvature."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from zarr_vectors.core.arrays import list_chunk_keys, read_chunk_attributes
from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors.types.meshes import write_mesh
from zarr_vectors_tools.algorithms import (
    compute_mean_curvature,
    compute_vertex_normals,
)


def _read_full_attribute(
    store_path: Path, attr_name: str, dtype: np.dtype, ncols: int
) -> np.ndarray:
    """Concatenate a per-vertex attribute across all chunks.

    The bulk-attribute write path (``ZVWriter.add_node_attribute_sync``)
    doesn't persist ``ncols``/``dtype`` metadata that the lazy attribute
    reader needs, so we use the low-level reader with explicit dtype
    and ncols.
    """
    root = open_store(str(store_path))
    level_group = get_resolution_level(root, 0)
    chunks = list_chunk_keys(level_group)
    parts: list[np.ndarray] = []
    for ck in chunks:
        groups = read_chunk_attributes(
            level_group, attr_name, ck, dtype=dtype, ncols=ncols,
        )
        for g in groups:
            if len(g):
                parts.append(g)
    return np.concatenate(parts, axis=0) if parts else np.zeros((0,), dtype=dtype)


def _unit_cube() -> tuple[np.ndarray, np.ndarray]:
    v = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float32)
    f = np.array([
        [0, 2, 1], [0, 3, 2],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [3, 7, 6], [3, 6, 2],
        [0, 4, 7], [0, 7, 3],
        [1, 2, 6], [1, 6, 5],
    ], dtype=np.int64)
    return v, f


def _uv_sphere(radius: float, lat: int = 12, lon: int = 18):
    verts = []
    for i in range(lat + 1):
        theta = np.pi * i / lat
        for j in range(lon):
            phi = 2 * np.pi * j / lon
            verts.append([
                radius * np.sin(theta) * np.cos(phi),
                radius * np.sin(theta) * np.sin(phi),
                radius * np.cos(theta),
            ])
    verts = np.array(verts, dtype=np.float32)

    faces = []
    for i in range(lat):
        for j in range(lon):
            a = i * lon + j
            b = a + lon
            c = a + 1 if j < lon - 1 else i * lon
            d = b + 1 if j < lon - 1 else (i + 1) * lon
            if i > 0:
                faces.append([a, b, c])
            if i < lat - 1:
                faces.append([b, d, c])
    return verts, np.array(faces, dtype=np.int64)


# ---------------------------------------------------------------------
# Normals
# ---------------------------------------------------------------------

class TestVertexNormals:

    def test_cube_corner_normals(self, tmp_path: Path) -> None:
        """Each corner normal should point INTO the corner's octant.

        The unit-cube triangulation in :func:`_unit_cube` splits each
        quad face along a single diagonal, which makes the two corners
        on that diagonal touch both triangles while the other two touch
        only one. The resulting area-weighted per-vertex normal is unit
        length and on the correct side of every adjacent face, but only
        the two main-diagonal corners (here vertices 0 at (0,0,0) and 6
        at (1,1,1)) land exactly along the ``(±1, ±1, ±1)/√3`` corner
        ray — the other six are skewed by the diagonal choice. So check
        the weaker, well-defined property: unit length and a positive
        dot product with the corner direction.
        """
        v, f = _unit_cube()
        store = tmp_path / "cube.zv"
        write_mesh(str(store), v, f, chunk_shape=(10.0, 10.0, 10.0))
        result = compute_vertex_normals(store)
        normals = result["normals"]
        # Map each vertex to its corner-direction unit vector (storage
        # vertex order equals input order under chunk_shape=(10,10,10)).
        for i, n in enumerate(normals):
            corner_dir = np.where(v[i] >= 0.5, 1.0, -1.0)
            corner_dir = corner_dir / np.linalg.norm(corner_dir)
            assert abs(np.linalg.norm(n) - 1.0) < 1e-3
            assert float(np.dot(n, corner_dir)) > 0.9

    def test_unknown_weighting_rejected(self, tmp_path: Path) -> None:
        v, f = _unit_cube()
        store = tmp_path / "cube.zv"
        write_mesh(str(store), v, f, chunk_shape=(10.0, 10.0, 10.0))
        try:
            compute_vertex_normals(store, weighting="lol")
        except ValueError:
            return
        raise AssertionError("expected ValueError for unknown weighting")


# ---------------------------------------------------------------------
# Mean curvature
# ---------------------------------------------------------------------

class TestMeanCurvature:

    def test_sphere_is_inverse_radius(self, tmp_path: Path) -> None:
        radius = 5.0
        v, f = _uv_sphere(radius, lat=16, lon=24)
        store = tmp_path / "sphere.zv"
        write_mesh(str(store), v, f, chunk_shape=(100.0, 100.0, 100.0))
        result = compute_mean_curvature(store)
        curv = result["mean_curvature"]
        # Mean curvature of a sphere is 1/radius. The discrete operator
        # plus barycentric area gives an approximation; allow a generous
        # tolerance away from the poles.
        median = float(np.median(curv))
        assert 0.5 / radius < median < 2.0 / radius


# ---------------------------------------------------------------------
# write_back round-trip via ZVWriter.add_node_attribute_sync
# ---------------------------------------------------------------------

class TestWriteBack:

    def test_normals_round_trip(self, tmp_path: Path) -> None:
        v, f = _unit_cube()
        store = tmp_path / "cube.zv"
        write_mesh(str(store), v, f, chunk_shape=(10.0, 10.0, 10.0))
        result = compute_vertex_normals(store, write_back=True)
        persisted = _read_full_attribute(
            store, "vertex_normal", dtype=np.float32, ncols=3,
        )
        assert persisted.shape == result["normals"].shape
        np.testing.assert_allclose(persisted, result["normals"], atol=1e-6)

    def test_curvature_round_trip(self, tmp_path: Path) -> None:
        radius = 5.0
        v, f = _uv_sphere(radius, lat=8, lon=12)
        store = tmp_path / "sphere.zv"
        write_mesh(str(store), v, f, chunk_shape=(100.0, 100.0, 100.0))
        result = compute_mean_curvature(store, write_back=True)
        persisted = _read_full_attribute(
            store, "mean_curvature", dtype=np.float32, ncols=1,
        )
        assert persisted.shape == result["mean_curvature"].shape
        np.testing.assert_allclose(persisted, result["mean_curvature"], atol=1e-6)
