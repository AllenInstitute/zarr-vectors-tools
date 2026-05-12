"""Tests for bfs_distances and shortest_path."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from zarr_vectors.types.graphs import write_graph
from zarr_vectors_tools.algorithms import bfs_distances, shortest_path


def _path_graph(tmp_path: Path, n: int = 5) -> Path:
    positions = np.array(
        [[i * 5.0, 0.0, 0.0] for i in range(n)], dtype=np.float32,
    )
    edges = np.array(
        [[i, i + 1] for i in range(n - 1)], dtype=np.int64,
    )
    store = tmp_path / "path.zv"
    write_graph(
        str(store), positions, edges, chunk_shape=(100.0, 100.0, 100.0),
    )
    return store


def _grid_graph(tmp_path: Path) -> tuple[Path, int]:
    """4x4 spatial grid graph spanning multiple chunks."""
    coords = np.array(
        [(x * 30.0, y * 30.0, 0.0) for y in range(4) for x in range(4)],
        dtype=np.float32,
    )
    edges_list = []
    for y in range(4):
        for x in range(4):
            idx = y * 4 + x
            if x < 3:
                edges_list.append((idx, idx + 1))
            if y < 3:
                edges_list.append((idx, idx + 4))
    edges = np.array(edges_list, dtype=np.int64)
    store = tmp_path / "grid.zv"
    write_graph(
        str(store), coords, edges, chunk_shape=(50.0, 50.0, 50.0),
    )
    return store, 4


# ---------------------------------------------------------------------
# bfs_distances
# ---------------------------------------------------------------------

class TestBFSDistances:

    def test_path_distances(self, tmp_path: Path) -> None:
        store = _path_graph(tmp_path, n=5)
        result = bfs_distances(store, source=0)
        np.testing.assert_array_equal(result["distances"], [0, 1, 2, 3, 4])
        # predecessors form the path.
        np.testing.assert_array_equal(
            result["predecessors"], [-1, 0, 1, 2, 3],
        )

    def test_max_distance_cutoff(self, tmp_path: Path) -> None:
        store = _path_graph(tmp_path, n=5)
        result = bfs_distances(store, source=0, max_distance=2)
        # Nodes at distance 0, 1, 2 are reached. Nodes 3, 4 may or may not
        # be reached depending on traversal order, but their distance is
        # capped at 2 — anything beyond ``max_distance`` stays unreached.
        d = result["distances"]
        assert d[0] == 0 and d[1] == 1 and d[2] == 2
        assert d[3] == -1 or d[3] == 3
        # Actually the BFS cap stops expansion at distance==max_distance,
        # so 3 and 4 are unreached.
        assert d[3] == -1 and d[4] == -1

    def test_grid_distances(self, tmp_path: Path) -> None:
        store, n_per_side = _grid_graph(tmp_path)
        # Source is bottom-left node (index 0); top-right is index 15.
        result = bfs_distances(store, source=0)
        # In a 4x4 grid, Manhattan distance from (0,0) to (3,3) is 6.
        assert result["distances"][15] == 6

    def test_source_out_of_range(self, tmp_path: Path) -> None:
        store = _path_graph(tmp_path, n=3)
        try:
            bfs_distances(store, source=99)
        except IndexError:
            return
        raise AssertionError("expected IndexError")


# ---------------------------------------------------------------------
# shortest_path
# ---------------------------------------------------------------------

class TestShortestPath:

    def test_unit_weights_match_bfs(self, tmp_path: Path) -> None:
        store = _path_graph(tmp_path, n=5)
        result = shortest_path(store, source=0, target=4)
        assert result["path"] == [0, 1, 2, 3, 4]
        assert result["cost"] == 4.0

    def test_unreachable(self, tmp_path: Path) -> None:
        positions = np.array(
            [[0, 0, 0], [1, 0, 0], [100, 100, 100]], dtype=np.float32,
        )
        edges = np.array([[0, 1]], dtype=np.int64)  # node 2 disconnected
        store = tmp_path / "split.zv"
        write_graph(
            str(store), positions, edges, chunk_shape=(200.0, 200.0, 200.0),
        )
        result = shortest_path(store, source=0, target=2)
        assert result["path"] == []
        assert result["cost"] == float("inf")

    def test_astar_with_zero_heuristic_matches_dijkstra(
        self, tmp_path: Path,
    ) -> None:
        store, _ = _grid_graph(tmp_path)
        dijkstra = shortest_path(store, source=0, target=15)
        astar = shortest_path(
            store, source=0, target=15, heuristic=lambda _i: 0.0,
        )
        assert dijkstra["cost"] == astar["cost"]
        assert dijkstra["path"][0] == 0 and dijkstra["path"][-1] == 15
        assert astar["path"][0] == 0 and astar["path"][-1] == 15
