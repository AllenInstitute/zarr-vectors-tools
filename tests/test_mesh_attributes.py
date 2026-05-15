"""Tests for compute_vertex_normals and compute_mean_curvature."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from zarr_vectors.lazy import open_zv
from zarr_vectors.types.meshes import write_mesh
from zarr_vectors_tools.algorithms import (
    compute_mean_curvature,
    compute_vertex_normals,
)


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
        v, f = _unit_cube()
        store = tmp_path / "cube.zv"
        write_mesh(str(store), v, f, chunk_shape=(10.0, 10.0, 10.0))
        result = compute_vertex_normals(store)
        normals = result["normals"]
        # Each cube corner's normal should point along (±1, ±1, ±1)/√3.
        # The exact ordering depends on the store's chunk traversal, so
        # check that every normal has unit length and matches one of the
        # 8 corner directions.
        expected = np.array(
            [[(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]],
            dtype=np.float64,
        ).reshape(-1, 3) / np.sqrt(3.0)
        for n in normals:
            assert abs(np.linalg.norm(n) - 1.0) < 1e-3
            # n matches one of the corner directions (within tolerance).
            assert np.any(np.linalg.norm(expected - n, axis=1) < 1e-3)

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
        zv = open_zv(str(store))
        persisted = zv[0]["vertex_normal"].compute()
        assert persisted.shape == result["normals"].shape
        # Persisted values must equal the returned array (modulo dtype).
        np.testing.assert_allclose(
            persisted.astype(np.float32), result["normals"], atol=1e-6,
        )

    def test_curvature_round_trip(self, tmp_path: Path) -> None:
        radius = 5.0
        v, f = _uv_sphere(radius, lat=8, lon=12)
        store = tmp_path / "sphere.zv"
        write_mesh(str(store), v, f, chunk_shape=(100.0, 100.0, 100.0))
        result = compute_mean_curvature(store, write_back=True)
        zv = open_zv(str(store))
        persisted = zv[0]["mean_curvature"].compute()
        assert persisted.shape == result["mean_curvature"].shape
        np.testing.assert_allclose(
            persisted.astype(np.float32), result["mean_curvature"], atol=1e-6,
        )
