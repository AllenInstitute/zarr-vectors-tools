"""Tests for the private algorithm helper modules."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors.types.points import write_points
from zarr_vectors_tools.algorithms._chunk_neighbours import neighbouring_chunk_keys
from zarr_vectors_tools.algorithms._cross_chunk_index import build_cross_chunk_index
from zarr_vectors_tools.algorithms._indexing import (
    build_chunk_to_global_offset,
    resolve_endpoint,
    resolve_endpoints,
)


# ---------------------------------------------------------------------
# _chunk_neighbours
# ---------------------------------------------------------------------

class TestNeighbouringChunkKeys:

    def test_3d_halo_1(self) -> None:
        neighbours = neighbouring_chunk_keys((5, 5, 5), halo=1)
        # 3^3 cube minus the centre = 26 neighbours
        assert len(neighbours) == 26
        assert (5, 5, 5) not in neighbours
        assert (4, 5, 5) in neighbours
        assert (6, 6, 6) in neighbours

    def test_include_self(self) -> None:
        neighbours = neighbouring_chunk_keys((0, 0, 0), halo=1, include_self=True)
        assert len(neighbours) == 27
        assert (0, 0, 0) in neighbours

    def test_halo_2(self) -> None:
        neighbours = neighbouring_chunk_keys((0, 0, 0), halo=2)
        # 5^3 - 1
        assert len(neighbours) == 124

    def test_halo_zero(self) -> None:
        assert neighbouring_chunk_keys((1, 2, 3), halo=0) == []
        assert neighbouring_chunk_keys((1, 2, 3), halo=0, include_self=True) == [(1, 2, 3)]

    def test_occupied_filter(self) -> None:
        occupied = {(4, 5, 5), (6, 5, 5)}
        neighbours = neighbouring_chunk_keys(
            (5, 5, 5), halo=1, occupied_keys=occupied
        )
        assert set(neighbours) == occupied

    def test_2d(self) -> None:
        # 8 neighbours of a 2D centre
        neighbours = neighbouring_chunk_keys((0, 0), halo=1)
        assert len(neighbours) == 8
        for n in neighbours:
            assert len(n) == 2

    def test_negative_halo_rejected(self) -> None:
        try:
            neighbouring_chunk_keys((0, 0, 0), halo=-1)
        except ValueError:
            return
        raise AssertionError("expected ValueError for negative halo")


# ---------------------------------------------------------------------
# _indexing — needs a real store, easiest with a points store
# ---------------------------------------------------------------------

def _multi_chunk_store(tmp_path: Path) -> Path:
    """Build a points store with vertices in three separate chunks."""
    rng = np.random.default_rng(0)
    # 30 points clumped at x=[0,10), x=[100,110), x=[200,210)
    positions = np.concatenate([
        rng.uniform([0, 0, 0], [10, 10, 10], size=(10, 3)),
        rng.uniform([100, 0, 0], [110, 10, 10], size=(10, 3)),
        rng.uniform([200, 0, 0], [210, 10, 10], size=(10, 3)),
    ]).astype(np.float32)
    store = tmp_path / "p.zv"
    write_points(str(store), positions, chunk_shape=(50.0, 50.0, 50.0))
    return store


class TestChunkToGlobalOffset:

    def test_offsets_match_chunk_vertex_counts(self, tmp_path: Path) -> None:
        store = _multi_chunk_store(tmp_path)
        root = open_store(str(store))
        level = get_resolution_level(root, 0)
        offsets, keys, total = build_chunk_to_global_offset(level)
        # 30 points across however-many chunks
        assert total == 30
        assert len(offsets) == len(keys)
        # Offsets are strictly non-decreasing in the iteration order.
        prev = -1
        for key in keys:
            assert offsets[key] >= prev
            prev = offsets[key]

    def test_resolve_endpoint(self, tmp_path: Path) -> None:
        store = _multi_chunk_store(tmp_path)
        root = open_store(str(store))
        level = get_resolution_level(root, 0)
        offsets, keys, total = build_chunk_to_global_offset(level)
        # Resolving local-0 of every chunk gives the chunk's start offset.
        for k in keys:
            assert resolve_endpoint((k, 0), offsets) == offsets[k]

    def test_resolve_endpoints_vectorised(self, tmp_path: Path) -> None:
        store = _multi_chunk_store(tmp_path)
        root = open_store(str(store))
        level = get_resolution_level(root, 0)
        offsets, keys, _ = build_chunk_to_global_offset(level)
        pairs = [(k, 0) for k in keys]
        result = resolve_endpoints(pairs, offsets)
        expected = np.array([offsets[k] for k in keys], dtype=np.int64)
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------
# _cross_chunk_index — exercises the path through a real store
# ---------------------------------------------------------------------

class TestBuildCrossChunkIndex:

    def test_no_cross_chunk_links(self, tmp_path: Path) -> None:
        """A point cloud has no edges → empty cross-chunk array."""
        store = _multi_chunk_store(tmp_path)
        root = open_store(str(store))
        level = get_resolution_level(root, 0)
        links, per_chunk = build_cross_chunk_index(level)
        # No links is fine — the index should be empty, not raise.
        assert links == []
        assert per_chunk == {}

    def test_with_cross_chunk_edges(self, tmp_path: Path) -> None:
        """A graph with one cross-chunk edge appears in both chunks' lists."""
        from zarr_vectors.types.graphs import write_graph

        # Two nodes in different chunks (chunk_shape=(50,50,50))
        positions = np.array([
            [10, 10, 10],
            [80, 80, 80],
        ], dtype=np.float32)
        edges = np.array([[0, 1]], dtype=np.int64)
        store = tmp_path / "g.zv"
        write_graph(
            str(store), positions, edges,
            chunk_shape=(50.0, 50.0, 50.0),
        )

        root = open_store(str(store))
        level = get_resolution_level(root, 0)
        links, per_chunk = build_cross_chunk_index(level)

        # Exactly one cross-chunk link, touching exactly two chunks,
        # each of which lists the same row index.
        assert len(links) == 1
        assert len(per_chunk) == 2
        for indices in per_chunk.values():
            assert indices == [0]
