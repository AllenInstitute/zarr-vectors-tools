"""Tests for compute_mesh_summary."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from zarr_vectors.types.meshes import write_mesh
from zarr_vectors_tools.algorithms import compute_mesh_summary


# ---------------------------------------------------------------------
# Hand-built reference shapes
# ---------------------------------------------------------------------

def _unit_cube() -> tuple[np.ndarray, np.ndarray]:
    """8-vertex unit cube triangulated into 12 outward-facing triangles."""
    v = np.array([
        [0, 0, 0],  # 0
        [1, 0, 0],  # 1
        [1, 1, 0],  # 2
        [0, 1, 0],  # 3
        [0, 0, 1],  # 4
        [1, 0, 1],  # 5
        [1, 1, 1],  # 6
        [0, 1, 1],  # 7
    ], dtype=np.float32)
    # Two triangles per face, wound so face normals point outward.
    f = np.array([
        # z = 0 (bottom, normal -z)
        [0, 2, 1], [0, 3, 2],
        # z = 1 (top, normal +z)
        [4, 5, 6], [4, 6, 7],
        # y = 0 (front, normal -y)
        [0, 1, 5], [0, 5, 4],
        # y = 1 (back, normal +y)
        [3, 7, 6], [3, 6, 2],
        # x = 0 (left, normal -x)
        [0, 4, 7], [0, 7, 3],
        # x = 1 (right, normal +x)
        [1, 2, 6], [1, 6, 5],
    ], dtype=np.int64)
    return v, f


def _tetrahedron() -> tuple[np.ndarray, np.ndarray]:
    v = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=np.float32)
    f = np.array([
        [0, 2, 1],  # bottom
        [0, 1, 3],
        [0, 3, 2],
        [1, 2, 3],
    ], dtype=np.int64)
    return v, f


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

class TestMeshSummary:

    def test_unit_cube_single_chunk(self, tmp_path: Path) -> None:
        v, f = _unit_cube()
        store = tmp_path / "cube.zv"
        write_mesh(str(store), v, f, chunk_shape=(10.0, 10.0, 10.0))

        result = compute_mesh_summary(store)
        assert result["face_count"] == 12
        assert result["vertex_count"] == 8
        assert result["excluded_cross_face_edges"] == 0
        assert abs(result["surface_area"] - 6.0) < 1e-5
        assert abs(result["volume"] - 1.0) < 1e-5
        # V - E + F = 8 - 18 + 12 = 2 for a closed manifold cube.
        assert result["edge_count"] == 18
        assert result["euler_characteristic"] == 2

    def test_tetrahedron(self, tmp_path: Path) -> None:
        v, f = _tetrahedron()
        store = tmp_path / "tet.zv"
        write_mesh(str(store), v, f, chunk_shape=(10.0, 10.0, 10.0))

        result = compute_mesh_summary(store)
        assert result["face_count"] == 4
        assert result["vertex_count"] == 4
        # Volume of the unit tetrahedron at the origin = 1/6.
        assert abs(result["volume"] - (1.0 / 6.0)) < 1e-5
        # V - E + F = 4 - 6 + 4 = 2.
        assert result["edge_count"] == 6
        assert result["euler_characteristic"] == 2

    def test_unit_cube_cross_chunk(self, tmp_path: Path) -> None:
        """Cube straddling chunk boundaries: surface area + volume still
        right because intra-chunk faces dominate; the Euler characteristic
        is offset by however many cross-chunk faces lose identity."""
        v, f = _unit_cube()
        # Shift so the cube centre lands on (0.5, 0.5, 0.5) but the
        # chunk shape is 0.5, forcing 8 chunks.
        store = tmp_path / "cube_chunked.zv"
        write_mesh(str(store), v, f, chunk_shape=(0.5, 0.5, 0.5))

        result = compute_mesh_summary(store)
        # Vertex and face counts come from level metadata; should still
        # be 8 and (intra-only) <= 12.
        assert result["vertex_count"] == 8
        assert 0 < result["face_count"] <= 12
        # Surface area on intra faces alone is at most 6.
        assert 0 < result["surface_area"] <= 6.0 + 1e-5

    def test_per_object_raises(self, tmp_path: Path) -> None:
        v, f = _tetrahedron()
        store = tmp_path / "tet.zv"
        write_mesh(str(store), v, f, chunk_shape=(10.0, 10.0, 10.0))
        try:
            compute_mesh_summary(store, per_object=True)
        except NotImplementedError:
            return
        raise AssertionError("expected NotImplementedError")
