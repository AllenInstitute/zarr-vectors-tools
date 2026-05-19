"""Tests for closest_point and cast_ray."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from zarr_vectors.types.meshes import write_mesh
from zarr_vectors_tools.algorithms import cast_ray, closest_point


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


def _make_cube_store(tmp_path: Path, chunk_shape) -> Path:
    v, f = _unit_cube()
    store = tmp_path / "cube.zv"
    write_mesh(str(store), v, f, chunk_shape=chunk_shape)
    return store


def _make_two_cube_store(tmp_path: Path, chunk_shape, x_offset: float = 2.0) -> Path:
    """Two unit cubes side-by-side along x: cube A at [0,1]^3, cube B shifted
    by ``x_offset`` along x. Each cube fits in a single chunk when
    ``chunk_shape >= (2, 1, 1)``; with ``chunk_shape=(2.0, 2.0, 2.0)`` the
    store has two chunks, both fully intra-chunk (no cross-chunk faces)."""
    v, f = _unit_cube()
    v2 = v.copy()
    v2[:, 0] += x_offset
    verts = np.concatenate([v, v2], axis=0).astype(np.float32)
    faces2 = f + 8  # second cube's vertex indices live at 8..15
    faces = np.concatenate([f, faces2], axis=0)
    store = tmp_path / "two_cubes.zv"
    write_mesh(str(store), verts, faces, chunk_shape=chunk_shape)
    return store


# ---------------------------------------------------------------------
# closest_point
# ---------------------------------------------------------------------

class TestClosestPoint:

    def test_outside_along_x(self, tmp_path: Path) -> None:
        store = _make_cube_store(tmp_path, (10.0, 10.0, 10.0))
        result = closest_point(store, np.array([5.0, 0.5, 0.5]))
        assert result["found"]
        # The closest face is the +x side; closest point lies on it at x=1.
        assert abs(result["position"][0] - 1.0) < 1e-5
        assert abs(result["distance"] - 4.0) < 1e-5

    def test_inside_cube(self, tmp_path: Path) -> None:
        store = _make_cube_store(tmp_path, (10.0, 10.0, 10.0))
        # Point inside the cube — closest face is whichever wall is nearest.
        result = closest_point(store, np.array([0.5, 0.5, 0.4]))
        assert result["found"]
        # Nearest wall is z=0, distance 0.4.
        assert abs(result["distance"] - 0.4) < 1e-5

    def test_chunked_matches_single_chunk(self, tmp_path: Path) -> None:
        """Multi-chunk traversal: two cubes side-by-side along x, each in
        its own chunk under ``chunk_shape=(2,2,2)``. Both cubes are fully
        intra-chunk (no cross-chunk faces) so the closest-point search
        produces the same answer as the single-chunk control."""
        single = closest_point(
            _make_two_cube_store(tmp_path / "single", (10.0, 10.0, 10.0)),
            np.array([5.0, 0.5, 0.5]),
        )
        chunked = closest_point(
            _make_two_cube_store(tmp_path / "chunked", (2.0, 2.0, 2.0)),
            np.array([5.0, 0.5, 0.5]),
        )
        # Both should find the same +x face of the right cube (at x=3).
        assert single["found"] and chunked["found"]
        np.testing.assert_allclose(
            chunked["position"], single["position"], atol=1e-4,
        )
        assert abs(chunked["distance"] - single["distance"]) < 1e-4

    def test_max_distance_cuts_search(self, tmp_path: Path) -> None:
        store = _make_cube_store(tmp_path, (10.0, 10.0, 10.0))
        # The cube is 4 units away; max_distance=1 should fail.
        result = closest_point(
            store, np.array([5.0, 0.5, 0.5]), max_distance=1.0,
        )
        assert not result["found"]


# ---------------------------------------------------------------------
# cast_ray
# ---------------------------------------------------------------------

class TestCastRay:

    def test_hit_on_plus_x_face(self, tmp_path: Path) -> None:
        store = _make_cube_store(tmp_path, (10.0, 10.0, 10.0))
        result = cast_ray(
            store,
            origin=np.array([5.0, 0.5, 0.5]),
            direction=np.array([-1.0, 0.0, 0.0]),
        )
        assert result["hit"]
        # The +x face sits at x=1; ray starts at x=5 → t should be 4.
        assert abs(result["t"] - 4.0) < 1e-4
        assert abs(result["position"][0] - 1.0) < 1e-4

    def test_miss(self, tmp_path: Path) -> None:
        store = _make_cube_store(tmp_path, (10.0, 10.0, 10.0))
        # Ray pointing away from the cube.
        result = cast_ray(
            store,
            origin=np.array([5.0, 0.5, 0.5]),
            direction=np.array([1.0, 0.0, 0.0]),
        )
        assert not result["hit"]

    def test_zero_direction_raises(self, tmp_path: Path) -> None:
        store = _make_cube_store(tmp_path, (10.0, 10.0, 10.0))
        try:
            cast_ray(store, np.array([0, 0, 0]), np.array([0, 0, 0]))
        except ValueError:
            return
        raise AssertionError("expected ValueError for zero-direction ray")

    def test_max_distance_clip(self, tmp_path: Path) -> None:
        store = _make_cube_store(tmp_path, (10.0, 10.0, 10.0))
        # Cube is at t=4; allowing only t<=2 should miss.
        result = cast_ray(
            store,
            origin=np.array([5.0, 0.5, 0.5]),
            direction=np.array([-1.0, 0.0, 0.0]),
            max_distance=2.0,
        )
        assert not result["hit"]
