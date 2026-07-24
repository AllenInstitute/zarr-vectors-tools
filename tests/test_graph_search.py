"""Tests for bfs_distances and shortest_path."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from zarr_vectors.core.arrays import (
    read_links,
    write_link_attributes,
)
from zarr_vectors.core.store import get_resolution_level, open_store
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


# ---------------------------------------------------------------------
# Cross-chunk link weights (new in the multiscale-links layout)
# ---------------------------------------------------------------------

def _two_chunk_path_graph(tmp_path: Path) -> tuple[Path, int]:
    """Three-node path graph with one cross-chunk edge.

    Layout (chunk_shape=(50, 50, 50)):
        node 0 at (10, 10, 10) — chunk A
        node 1 at (80, 10, 10) — chunk B (cross-chunk edge 0-1)
        node 2 at (85, 10, 10) — chunk B (intra-chunk edge 1-2)

    Returns the store path and the number of cross-chunk links written.
    """
    positions = np.array(
        [[10, 10, 10], [80, 10, 10], [85, 10, 10]], dtype=np.float32,
    )
    edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
    store = tmp_path / "cross.zv"
    summary = write_graph(
        str(store), positions, edges, chunk_shape=(50.0, 50.0, 50.0),
    )
    return store, int(summary.get("cross_edge_count", 0))


class TestCrossChunkWeights:

    def test_unit_weight_when_no_attribute(self, tmp_path: Path) -> None:
        """Without weights stored, cross-chunk edges contribute unit cost."""
        store, n_cross = _two_chunk_path_graph(tmp_path)
        assert n_cross >= 1  # fixture must produce at least one cross-chunk edge
        result = shortest_path(store, source=0, target=2, weight="weight")
        # Two unit edges along the only path → cost 2.0.
        assert result["path"][0] == 0 and result["path"][-1] == 2
        assert result["cost"] == 2.0

    def test_cross_weight_steers_cost(self, tmp_path: Path) -> None:
        """A non-unit cross-chunk weight changes the path cost."""
        store, n_cross = _two_chunk_path_graph(tmp_path)
        assert n_cross == 1

        # Attach a weight of 5.0 to the cross-chunk edge and 1.0 to the
        # intra one.  Attributes are per-FAMILY now, and the only thing
        # aligning row i to record i is that read_links and
        # read_link_attributes share an enumeration order — so build the
        # rows from read_links itself rather than assuming a position.
        root = open_store(str(store), mode="r+")
        level = get_resolution_level(root, 0)
        records = read_links(level, delta=0)
        assert len(records) == 2  # one cross (0-1), one intra (1-2)
        is_cross = [len({tuple(cc) for cc, _vi in r}) > 1 for r in records]
        assert sum(is_cross) == 1
        write_link_attributes(
            level, "weight",
            np.array([5.0 if x else 1.0 for x in is_cross], dtype=np.float32),
            num_links=len(records),
            delta=0,
        )

        # Without weight kwarg: still unit, cost 2.0.
        unit = shortest_path(store, source=0, target=2)
        assert unit["cost"] == 2.0

        # With weight kwarg: cross-chunk edge 0-1 costs 5, intra 1-2 costs 1.
        weighted = shortest_path(store, source=0, target=2, weight="weight")
        assert abs(weighted["cost"] - 6.0) < 1e-6
