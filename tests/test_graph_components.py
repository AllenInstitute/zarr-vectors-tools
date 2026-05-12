"""Tests for compute_connected_components."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from zarr_vectors.types.graphs import write_graph
from zarr_vectors_tools.algorithms import compute_connected_components


class TestConnectedComponents:

    def test_single_component_path(self, tmp_path: Path) -> None:
        positions = np.array(
            [[i * 5.0, 0.0, 0.0] for i in range(5)], dtype=np.float32,
        )
        edges = np.array(
            [[i, i + 1] for i in range(4)], dtype=np.int64,
        )
        store = tmp_path / "path.zv"
        write_graph(
            str(store), positions, edges, chunk_shape=(100.0, 100.0, 100.0),
        )
        result = compute_connected_components(store)
        assert result["n_components"] == 1
        assert result["largest_component_size"] == 5
        assert (result["labels"] == 0).all()

    def test_two_disjoint_triangles(self, tmp_path: Path) -> None:
        positions = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],          # triangle A
            [10, 10, 10], [11, 10, 10], [10, 11, 10], # triangle B
        ], dtype=np.float32)
        edges = np.array([
            [0, 1], [1, 2], [2, 0],
            [3, 4], [4, 5], [5, 3],
        ], dtype=np.int64)
        store = tmp_path / "two_tri.zv"
        write_graph(
            str(store), positions, edges, chunk_shape=(100.0, 100.0, 100.0),
        )
        result = compute_connected_components(store)
        assert result["n_components"] == 2
        assert result["largest_component_size"] == 3
        # First triangle and second triangle each share a label.
        labels = result["labels"]
        assert labels[0] == labels[1] == labels[2]
        assert labels[3] == labels[4] == labels[5]
        assert labels[0] != labels[3]

    def test_cross_chunk_edge_joins_components(self, tmp_path: Path) -> None:
        """Two intra-chunk clusters joined by one cross-chunk edge."""
        positions = np.array([
            [10, 10, 10], [20, 20, 20],     # chunk A
            [110, 110, 110], [120, 120, 120],  # chunk B
        ], dtype=np.float32)
        # Two intra edges + one cross edge.
        edges = np.array([
            [0, 1],   # intra to chunk A
            [2, 3],   # intra to chunk B
            [1, 2],   # cross-chunk
        ], dtype=np.int64)
        store = tmp_path / "joined.zv"
        write_graph(
            str(store), positions, edges, chunk_shape=(100.0, 100.0, 100.0),
        )
        result = compute_connected_components(store)
        assert result["n_components"] == 1
        assert result["largest_component_size"] == 4

    def test_write_back_not_implemented(self, tmp_path: Path) -> None:
        positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        edges = np.array([[0, 1]], dtype=np.int64)
        store = tmp_path / "g.zv"
        write_graph(
            str(store), positions, edges, chunk_shape=(100.0, 100.0, 100.0),
        )
        try:
            compute_connected_components(store, write_back=True)
        except NotImplementedError:
            return
        raise AssertionError("expected NotImplementedError")
