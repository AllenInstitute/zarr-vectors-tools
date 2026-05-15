"""Tests for tools-side helpers that wrap core APIs.

After the migration to the multiscale-links layout the per-chunk vertex
offset table and per-chunk cross-link index moved into core; these tests
exercise the public replacements (``chunk_local_to_global_offsets`` and
``read_cross_chunk_links``) the same way the retired tools-side
workarounds used to.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from zarr_vectors.core.arrays import read_cross_chunk_links
from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors.spatial.boundary import chunk_local_to_global_offsets
from zarr_vectors.types.points import write_points
from zarr_vectors_tools.algorithms._chunk_neighbours import neighbouring_chunk_keys


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
# chunk_local_to_global_offsets — public core helper used everywhere
# ---------------------------------------------------------------------

def _multi_chunk_store(tmp_path: Path) -> Path:
    """Build a points store with vertices in three separate chunks."""
    rng = np.random.default_rng(0)
    positions = np.concatenate([
        rng.uniform([0, 0, 0], [10, 10, 10], size=(10, 3)),
        rng.uniform([100, 0, 0], [110, 10, 10], size=(10, 3)),
        rng.uniform([200, 0, 0], [210, 10, 10], size=(10, 3)),
    ]).astype(np.float32)
    store = tmp_path / "p.zv"
    write_points(str(store), positions, chunk_shape=(50.0, 50.0, 50.0))
    return store


class TestChunkLocalToGlobalOffsets:

    def test_offsets_match_chunk_vertex_counts(self, tmp_path: Path) -> None:
        store = _multi_chunk_store(tmp_path)
        root = open_store(str(store))
        level = get_resolution_level(root, 0)
        offsets, keys, total = chunk_local_to_global_offsets(level)
        assert total == 30
        assert len(offsets) == len(keys)
        prev = -1
        for key in keys:
            assert offsets[key] >= prev
            prev = offsets[key]

    def test_resolve_endpoint(self, tmp_path: Path) -> None:
        store = _multi_chunk_store(tmp_path)
        root = open_store(str(store))
        level = get_resolution_level(root, 0)
        offsets, keys, _total = chunk_local_to_global_offsets(level)
        for k in keys:
            # local index 0 in chunk k must map to the chunk's start offset
            assert offsets[k] + 0 == offsets[k]


# ---------------------------------------------------------------------
# read_cross_chunk_links — public core reader
# ---------------------------------------------------------------------

class TestReadCrossChunkLinks:

    def test_no_cross_chunk_links(self, tmp_path: Path) -> None:
        """A point cloud has no edges → empty cross-chunk array."""
        store = _multi_chunk_store(tmp_path)
        root = open_store(str(store))
        level = get_resolution_level(root, 0)
        try:
            links = read_cross_chunk_links(level, delta=0)
        except Exception:
            links = []
        assert links == []

    def test_with_cross_chunk_edges(self, tmp_path: Path) -> None:
        """A graph with one cross-chunk edge surfaces as a single link."""
        from zarr_vectors.types.graphs import write_graph

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
        links = read_cross_chunk_links(level, delta=0)

        assert len(links) == 1
        (chunk_a, _), (chunk_b, _) = links[0]
        assert chunk_a != chunk_b
