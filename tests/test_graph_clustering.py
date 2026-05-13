"""Tests for compute_k_core, compute_label_propagation, compute_louvain."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from zarr_vectors.types.graphs import write_graph
from zarr_vectors_tools.algorithms import (
    compute_k_core,
    compute_label_propagation,
    compute_louvain,
)


def _write(store: Path, positions: np.ndarray, edges: np.ndarray) -> None:
    """Standard fixture: 100-µm cubic chunks, no bins."""
    write_graph(
        str(store), positions, edges,
        chunk_shape=(100.0, 100.0, 100.0),
    )


def _triangle(offset: tuple[float, float, float]) -> np.ndarray:
    return np.array([
        [offset[0] + 0, offset[1] + 0, offset[2]],
        [offset[0] + 1, offset[1] + 0, offset[2]],
        [offset[0] + 0, offset[1] + 1, offset[2]],
    ], dtype=np.float32)


def _clique_edges(start: int, k: int) -> list[list[int]]:
    return [[start + i, start + j] for i in range(k) for j in range(i + 1, k)]


# =====================================================================
# k-core
# =====================================================================

class TestKCore:

    def test_path_graph(self, tmp_path: Path) -> None:
        positions = np.array(
            [[i * 5.0, 0.0, 0.0] for i in range(5)], dtype=np.float32,
        )
        edges = np.array([[i, i + 1] for i in range(4)], dtype=np.int64)
        store = tmp_path / "path.zv"
        _write(store, positions, edges)
        r = compute_k_core(store)
        # Every vertex in a path has coreness 1.
        assert r["max_core"] == 1
        assert (r["coreness"] == 1).all()
        assert r["core_sizes"].tolist() == [0, 5]

    def test_triangle(self, tmp_path: Path) -> None:
        positions = _triangle((0, 0, 0))
        edges = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)
        store = tmp_path / "tri.zv"
        _write(store, positions, edges)
        r = compute_k_core(store)
        assert r["max_core"] == 2
        assert (r["coreness"] == 2).all()

    def test_triangle_with_tail(self, tmp_path: Path) -> None:
        # Triangle on 0,1,2 plus tail 2-3-4.
        positions = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [5, 0, 0], [10, 0, 0],
        ], dtype=np.float32)
        edges = np.array([
            [0, 1], [1, 2], [2, 0],
            [2, 3], [3, 4],
        ], dtype=np.int64)
        store = tmp_path / "tri_tail.zv"
        _write(store, positions, edges)
        r = compute_k_core(store)
        # Triangle vertices have coreness 2; tail vertices coreness 1.
        assert r["coreness"][0] == 2
        assert r["coreness"][1] == 2
        assert r["coreness"][2] == 2
        assert r["coreness"][3] == 1
        assert r["coreness"][4] == 1
        assert r["max_core"] == 2

    def test_two_disjoint_k4(self, tmp_path: Path) -> None:
        positions = np.concatenate([
            np.array([[i, 0, 0] for i in range(4)], dtype=np.float32),
            np.array([[50 + i, 0, 0] for i in range(4)], dtype=np.float32),
        ])
        edges_list = _clique_edges(0, 4) + _clique_edges(4, 4)
        edges = np.asarray(edges_list, dtype=np.int64)
        store = tmp_path / "two_k4.zv"
        _write(store, positions, edges)
        r = compute_k_core(store)
        # Every K4 vertex has coreness 3.
        assert r["max_core"] == 3
        assert (r["coreness"] == 3).all()


# =====================================================================
# Label propagation
# =====================================================================

class TestLabelPropagation:

    def test_two_disjoint_triangles(self, tmp_path: Path) -> None:
        positions = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [10, 10, 10], [11, 10, 10], [10, 11, 10],
        ], dtype=np.float32)
        edges = np.array([
            [0, 1], [1, 2], [2, 0],
            [3, 4], [4, 5], [5, 3],
        ], dtype=np.int64)
        store = tmp_path / "two_tri.zv"
        _write(store, positions, edges)
        r = compute_label_propagation(store, max_iter=10, seed=1)
        assert r["n_communities"] == 2
        assert r["converged"]
        # Each triangle shares a label.
        labels = r["labels"]
        assert labels[0] == labels[1] == labels[2]
        assert labels[3] == labels[4] == labels[5]
        assert labels[0] != labels[3]

    def test_no_edges_singleton_communities(self, tmp_path: Path) -> None:
        positions = np.array(
            [[i * 5.0, 0, 0] for i in range(4)], dtype=np.float32,
        )
        edges = np.zeros((0, 2), dtype=np.int64)
        store = tmp_path / "isolated.zv"
        _write(store, positions, edges)
        r = compute_label_propagation(store, max_iter=5, seed=0)
        # Each vertex stays in its own community.
        assert r["n_communities"] == 4

    def test_deterministic_with_seed(self, tmp_path: Path) -> None:
        # K4 — LPA can pick any single label; check determinism.
        positions = np.array(
            [[i, 0, 0] for i in range(4)], dtype=np.float32,
        )
        edges = np.asarray(_clique_edges(0, 4), dtype=np.int64)
        store = tmp_path / "k4.zv"
        _write(store, positions, edges)
        r1 = compute_label_propagation(store, max_iter=10, seed=42)
        r2 = compute_label_propagation(store, max_iter=10, seed=42)
        assert (r1["labels"] == r2["labels"]).all()


# =====================================================================
# Louvain
# =====================================================================

class TestLouvain:

    def test_two_k5_bridge(self, tmp_path: Path) -> None:
        """Two K5 cliques joined by a single edge → 2 communities."""
        positions = np.concatenate([
            np.array([[i, 0, 0] for i in range(5)], dtype=np.float32),
            np.array([[50 + i, 0, 0] for i in range(5)], dtype=np.float32),
        ])
        edges_list = _clique_edges(0, 5) + _clique_edges(5, 5)
        edges_list.append([4, 5])  # bridge
        edges = np.asarray(edges_list, dtype=np.int64)
        store = tmp_path / "bridge.zv"
        _write(store, positions, edges)
        r = compute_louvain(store, seed=0)
        assert r["n_communities"] == 2
        # Each clique should share a label.
        labels = r["labels"]
        assert len(set(labels[:5].tolist())) == 1
        assert len(set(labels[5:].tolist())) == 1
        assert labels[0] != labels[5]
        assert r["modularity"] > 0.4

    def test_isolated_nodes(self, tmp_path: Path) -> None:
        positions = np.array(
            [[i * 5.0, 0, 0] for i in range(4)], dtype=np.float32,
        )
        edges = np.zeros((0, 2), dtype=np.int64)
        store = tmp_path / "iso.zv"
        _write(store, positions, edges)
        r = compute_louvain(store)
        assert r["n_communities"] == 4
        assert r["modularity"] == 0.0

    def test_three_clique_triangle(self, tmp_path: Path) -> None:
        """Three K4 cliques arranged in a triangle (single bridge each)."""
        # 3 K4 cliques: vertices 0-3, 4-7, 8-11.
        positions = np.concatenate([
            np.array([[i, 0, 0] for i in range(4)], dtype=np.float32),
            np.array([[50 + i, 0, 0] for i in range(4)], dtype=np.float32),
            np.array([[100 + i, 0, 0] for i in range(4)], dtype=np.float32),
        ])
        edges_list = (
            _clique_edges(0, 4) + _clique_edges(4, 4) + _clique_edges(8, 4)
        )
        edges_list += [[3, 4], [7, 8], [11, 0]]  # triangle of bridges
        edges = np.asarray(edges_list, dtype=np.int64)
        store = tmp_path / "triangle_cliques.zv"
        _write(store, positions, edges)
        r = compute_louvain(store, seed=0)
        assert r["n_communities"] == 3
        assert r["modularity"] > 0.3

    def test_weighted_intra_chunk(self, tmp_path: Path) -> None:
        """Intra-chunk edge weights are read back correctly (regression
        guard for the recent edge-attribute write fix)."""
        positions = np.concatenate([
            np.array([[i, 0, 0] for i in range(5)], dtype=np.float32),
            np.array([[50 + i, 0, 0] for i in range(5)], dtype=np.float32),
        ])
        edges_list = _clique_edges(0, 5) + _clique_edges(5, 5)
        edges_list.append([4, 5])  # bridge
        edges = np.asarray(edges_list, dtype=np.int64)
        # Make the bridge weight tiny, intra weights large — the
        # community split should still come out as the two cliques.
        weights = np.concatenate([
            np.full(len(_clique_edges(0, 5)), 10.0, dtype=np.float32),
            np.full(len(_clique_edges(5, 5)), 10.0, dtype=np.float32),
            np.array([0.01], dtype=np.float32),
        ])
        store = tmp_path / "weighted.zv"
        write_graph(
            str(store), positions, edges,
            chunk_shape=(100.0, 100.0, 100.0),
            edge_attributes={"w": weights},
        )
        r = compute_louvain(store, weight="w", seed=0)
        assert r["n_communities"] == 2
        labels = r["labels"]
        assert len(set(labels[:5].tolist())) == 1
        assert len(set(labels[5:].tolist())) == 1
