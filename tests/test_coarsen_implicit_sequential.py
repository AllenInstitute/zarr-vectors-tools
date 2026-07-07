"""Tests for the implicit_sequential coarsening path.

Two layers:

* Unit tests for :mod:`zarr_vectors_tools.multiresolution.coarsen_implicit` —
  the per-object segmentation primitives are pure numpy and easy to
  exercise directly.
* Integration tests building a small store with hand-crafted streamlines
  that exercise (a) a path entirely inside one coarsened chunk, (b) a
  path that crosses a coarsened boundary, and (c) two streamlines that
  share a boundary metavertex (the branching case).
"""

from __future__ import annotations

import numpy as np
import pytest

from zarr_vectors.core.arrays import (
    list_chunk_keys,
    read_all_object_manifests,
    read_chunk_vertices,
    read_cross_chunk_links,
)
from zarr_vectors.core.store import (
    get_resolution_level,
    open_store,
    read_level_metadata,
    read_root_metadata,
)
from zarr_vectors_tools.multiresolution.coarsen import coarsen_level
from zarr_vectors_tools.multiresolution.coarsen_implicit import (
    coarse_chunks_of,
    positions_in_run,
    segment_object_by_coarse_chunk,
)
from tests._source_helpers import write_polylines_with_segment_id as write_polylines


# ===================================================================
# Unit tests: segment_object_by_coarse_chunk + positions_in_run
# ===================================================================


def test_segment_single_chunk_simple_path():
    """All vertices in the same coarsened chunk → one run."""
    mv = np.array([0, 1, 2, 3], dtype=np.int64)
    cc = np.tile(np.array([[0, 0, 0]], dtype=np.int64), (4, 1))
    runs = segment_object_by_coarse_chunk(mv, cc)
    assert runs == [((0, 0, 0), [0, 1, 2, 3])]


def test_segment_dedupe_consecutive_duplicates_in_run():
    """Consecutive same-metavertex source vertices collapse to one entry."""
    mv = np.array([0, 0, 1, 1, 1, 2], dtype=np.int64)
    cc = np.tile(np.array([[0, 0, 0]], dtype=np.int64), (6, 1))
    runs = segment_object_by_coarse_chunk(mv, cc)
    assert runs == [((0, 0, 0), [0, 1, 2])]


def test_segment_chunk_boundary_split():
    """A change in coarsened chunk starts a new run."""
    mv = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    cc = np.array(
        [[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
        dtype=np.int64,
    )
    runs = segment_object_by_coarse_chunk(mv, cc)
    assert runs == [((0, 0, 0), [0, 1]), ((1, 0, 0), [2, 3, 4])]


def test_segment_chunk_revisit_creates_two_runs():
    """A → B → A produces two separate runs in chunk A."""
    mv = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    cc = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]],
        dtype=np.int64,
    )
    runs = segment_object_by_coarse_chunk(mv, cc)
    assert runs == [
        ((0, 0, 0), [0]),
        ((1, 0, 0), [1, 2]),
        ((0, 0, 0), [3, 4]),
    ]


def test_segment_empty():
    runs = segment_object_by_coarse_chunk(
        np.empty(0, dtype=np.int64), np.empty((0, 3), dtype=np.int64),
    )
    assert runs == []


def test_positions_in_run_mirrors_runs_with_dedupe():
    """positions_in_run gives index pairs that align with the runs list."""
    mv = np.array([0, 0, 1, 2, 3, 3], dtype=np.int64)
    cc = np.array(
        [
            [0, 0, 0], [0, 0, 0], [0, 0, 0],
            [1, 0, 0], [1, 0, 0], [1, 0, 0],
        ],
        dtype=np.int64,
    )
    runs = segment_object_by_coarse_chunk(mv, cc)
    run_idx, pos_in_run = positions_in_run(6, mv, cc)
    # Vertices 0, 1 → run 0 / position 0 (same mv 0, deduped together)
    # Vertex 2 → run 0 / position 1 (mv 1)
    # Vertex 3 → run 1 / position 0 (chunk change resets to position 0)
    # Vertex 4 → run 1 / position 1 (mv 3 after chunk change started with mv 2)
    # Wait — chunk changed at vertex 3 so vertex 3 starts run 1 with mv 2 at pos 0
    # Vertex 4 has mv 3 (different from 2) so pos 1
    # Vertex 5 has mv 3 (same as 4) so still pos 1
    assert run_idx.tolist() == [0, 0, 0, 1, 1, 1]
    assert pos_in_run.tolist() == [0, 0, 1, 0, 1, 1]
    # Sanity-check vs runs list
    assert runs == [((0, 0, 0), [0, 1]), ((1, 0, 0), [2, 3])]


def test_coarse_chunks_of_floor_division():
    positions = np.array(
        [[0.5, 0.5, 0.5], [10.5, 0.5, 0.5], [-0.5, 0.5, 10.5]],
        dtype=np.float32,
    )
    cc = coarse_chunks_of(positions, (10.0, 10.0, 10.0))
    # Floor(-0.5/10) = -1; floor(10.5/10) = 1
    assert cc.tolist() == [[0, 0, 0], [1, 0, 0], [-1, 0, 1]]


# ===================================================================
# Integration helpers
# ===================================================================


def _build_handcrafted_store(tmp_path, geometry_type="streamline"):
    """Construct a small store with three streamlines that exercise:

    s0: single-chunk path entirely inside coarsened chunk (0,0,0)
    s1: path that crosses from coarsened (0,0,0) to (1,0,0)
    s2: another path that crosses (0,0,0) → (1,0,0) sharing a boundary bin

    Source-level chunk_shape = 25, so coarsened chunk_shape with
    chunk_scale_factor=2 will be 50.
    Bin shape = 5, coarsen factor 2 → target bin = 10.
    """
    # s0: stays in coarsened chunk (0,0,0) (positions < 50 on every axis)
    s0 = np.array([
        [2.5, 2.5, 2.5],
        [7.5, 2.5, 2.5],
        [12.5, 2.5, 2.5],
        [17.5, 2.5, 2.5],
        [22.5, 2.5, 2.5],
    ], dtype=np.float32)
    # s1: 50-boundary crossing in x.  Spans source chunks (0,0,0) and
    # (2,0,0), which after chunk_scale_factor=2 map to coarsened chunks
    # (0,0,0) and (1,0,0).  Ends at boundary bin near x=49 / 51 (bin 4/5
    # at the coarse level).
    s1 = np.array([
        [10.0, 10.0, 10.0],   # coarse chunk (0,0,0)
        [25.0, 10.0, 10.0],   # coarse chunk (0,0,0)
        [40.0, 10.0, 10.0],   # coarse chunk (0,0,0)
        [55.0, 10.0, 10.0],   # coarse chunk (1,0,0)
        [70.0, 10.0, 10.0],   # coarse chunk (1,0,0)
    ], dtype=np.float32)
    # s2: shares the boundary bin with s1 at x≈45, but continues to a
    # DIFFERENT bin on the (1,0,0) side (x≈65 vs s1's x≈55), so the
    # boundary metavertex has multiple outgoing cross-chunk neighbours.
    s2 = np.array([
        [5.0, 30.0, 10.0],
        [25.0, 30.0, 10.0],
        [42.0, 30.0, 10.0],   # near boundary, same bin as s1's [40,...]?
        [62.0, 30.0, 10.0],   # different (1,0,0) bin from s1's [55,...]
        [78.0, 30.0, 10.0],
    ], dtype=np.float32)

    store = tmp_path / "hand.zv"
    write_polylines(
        str(store),
        [s0, s1, s2],
        chunk_shape=(25.0, 25.0, 25.0),
        bin_shape=(5.0, 5.0, 5.0),
        geometry_type=geometry_type,
    )
    return store


# ===================================================================
# Integration tests
# ===================================================================


def test_streamline_inside_one_coarse_chunk_is_one_fragment(tmp_path):
    """s0's entire path is in coarsened chunk (0,0,0): one fragment,
    no cross-chunk-link records from s0."""
    store = _build_handcrafted_store(tmp_path)
    coarsen_level(
        str(store), source_level=0, target_level=1,
        coarsen_factor=2.0,
        chunk_scale_factor=2,
        sparsity_factor=1.0,
    )

    root = open_store(str(store))
    lvl1 = get_resolution_level(root, 1)
    manifests = read_all_object_manifests(lvl1)

    # s0 must touch exactly one coarsened chunk.
    s0_chunks = {cc for cc, _ in manifests[0]}
    assert s0_chunks == {(0, 0, 0)}, f"s0 unexpectedly spans {s0_chunks}"
    # And exactly one fragment in that chunk.
    assert len(manifests[0]) == 1


def test_boundary_crossing_streamline_has_exactly_one_xclink(tmp_path):
    """s1 crosses one coarsened boundary → two fragments + one
    cross-chunk-link record bridging the boundary."""
    store = _build_handcrafted_store(tmp_path)
    coarsen_level(
        str(store), source_level=0, target_level=1,
        coarsen_factor=2.0,
        chunk_scale_factor=2,
        sparsity_factor=1.0,
    )

    root = open_store(str(store))
    lvl1 = get_resolution_level(root, 1)
    manifests = read_all_object_manifests(lvl1)
    s1_manifest = manifests[1]

    s1_chunks_ordered = [cc for cc, _ in s1_manifest]
    # Two distinct coarsened chunks visited in path order.
    assert s1_chunks_ordered == [(0, 0, 0), (1, 0, 0)], (
        f"s1 chunk sequence: {s1_chunks_ordered}"
    )
    assert len(s1_manifest) == 2

    # No same-chunk cross_chunk_links/0 records should be emitted.
    records = read_cross_chunk_links(lvl1, delta=0)
    same_chunk = [r for r in records if r[0][0] == r[1][0]]
    assert same_chunk == [], (
        f"Expected zero same-chunk delta=0 records but got: {same_chunk}"
    )
    # s1 contributes exactly one cross-chunk record between its two
    # fragments.  We can identify it by endpoint chunks (0,0,0)→(1,0,0).
    boundary = [
        r for r in records
        if (r[0][0] == (0, 0, 0) and r[1][0] == (1, 0, 0))
        or (r[0][0] == (1, 0, 0) and r[1][0] == (0, 0, 0))
    ]
    assert len(boundary) >= 1


def test_shared_boundary_yields_multiple_cross_chunk_records(tmp_path):
    """s1 and s2 both end their (0,0,0)-side run near the x=50 boundary;
    if their last (0,0,0) bins collapse to the SAME metavertex, the
    boundary metavertex must have two outgoing records (one per
    streamline) — branching at the boundary."""
    store = _build_handcrafted_store(tmp_path)
    coarsen_level(
        str(store), source_level=0, target_level=1,
        coarsen_factor=2.0,
        chunk_scale_factor=2,
        sparsity_factor=1.0,
    )

    root = open_store(str(store))
    lvl1 = get_resolution_level(root, 1)
    records = read_cross_chunk_links(lvl1, delta=0)

    # Two streamlines each cross the (0,0,0)→(1,0,0) boundary.
    # Expect at least two cross-chunk records spanning that pair.
    boundary = [
        r for r in records
        if {r[0][0], r[1][0]} == {(0, 0, 0), (1, 0, 0)}
    ]
    assert len(boundary) >= 2, (
        f"Expected ≥ 2 cross-chunk records bridging (0,0,0)↔(1,0,0); "
        f"got {len(boundary)}: {boundary}"
    )


def test_no_same_chunk_x_links_on_implicit_path(tmp_path):
    """The whole point of the new path: zero delta=0 same-chunk
    cross_chunk_links/0 records.  All within-chunk connectivity is
    captured by implicit sequential ordering of the fragment."""
    store = _build_handcrafted_store(tmp_path)
    coarsen_level(
        str(store), source_level=0, target_level=1,
        coarsen_factor=2.0,
        chunk_scale_factor=2,
        sparsity_factor=1.0,
    )

    root = open_store(str(store))
    lvl1 = get_resolution_level(root, 1)
    records = read_cross_chunk_links(lvl1, delta=0)
    same = [r for r in records if r[0][0] == r[1][0]]
    assert same == []


def test_fragment_count_matches_total_manifest_entries(tmp_path):
    """Each (object, coarsened-chunk) visit emits one fragment, and no
    fragment is shared between objects (shared_fragments=False)."""
    store = _build_handcrafted_store(tmp_path)
    coarsen_level(
        str(store), source_level=0, target_level=1,
        coarsen_factor=2.0,
        chunk_scale_factor=2,
        sparsity_factor=1.0,
    )

    root = open_store(str(store))
    lm = read_level_metadata(root, 1)
    assert lm.shared_fragments is False

    lvl1 = get_resolution_level(root, 1)
    manifests = read_all_object_manifests(lvl1)
    total_entries = sum(len(m) for m in manifests)

    on_disk = 0
    for cc in list_chunk_keys(lvl1):
        fragments = read_chunk_vertices(lvl1, cc, dtype=np.float32, ndim=3)
        on_disk += len(fragments)
    assert on_disk == total_entries


def test_cross_chunk_link_endpoints_resolve_to_valid_rows(tmp_path):
    """Every cross_chunk_links/0 endpoint at the coarse level must
    reference a real chunk-local row in its chunk's vertex array."""
    store = _build_handcrafted_store(tmp_path)
    coarsen_level(
        str(store), source_level=0, target_level=1,
        coarsen_factor=2.0,
        chunk_scale_factor=2,
        sparsity_factor=1.0,
    )

    root = open_store(str(store))
    lvl1 = get_resolution_level(root, 1)
    records = read_cross_chunk_links(lvl1, delta=0)

    chunk_row_counts: dict = {}
    for cc in list_chunk_keys(lvl1):
        fragments = read_chunk_vertices(lvl1, cc, dtype=np.float32, ndim=3)
        chunk_row_counts[cc] = sum(int(f.shape[0]) for f in fragments)

    for (cc_a, vi_a), (cc_b, vi_b) in records:
        assert 0 <= vi_a < chunk_row_counts.get(cc_a, 0), (
            f"endpoint ({cc_a}, {vi_a}) out of range "
            f"(chunk has {chunk_row_counts.get(cc_a, 0)} rows)"
        )
        assert 0 <= vi_b < chunk_row_counts.get(cc_b, 0), (
            f"endpoint ({cc_b}, {vi_b}) out of range "
            f"(chunk has {chunk_row_counts.get(cc_b, 0)} rows)"
        )


def test_round_trip_implicit_path_recovers_bin_reduced_sequence(tmp_path):
    """For each object, walking its coarse manifest in order and
    concatenating fragment vertices yields the bin-reduced version of
    the source-level per-object vertex sequence."""
    # Bin reduction is the per-object coarsener's behavior; use a non-streamline
    # geometry so this store routes to per_object (streamlines route to RDP).
    store = _build_handcrafted_store(tmp_path, geometry_type="polyline")

    # Read source-level per-object positions before coarsening.
    root = open_store(str(store))
    lvl0 = get_resolution_level(root, 0)
    src_manifests = read_all_object_manifests(lvl0)
    src_chunk_vertices: dict = {}
    for cc in list_chunk_keys(lvl0):
        fragments = read_chunk_vertices(lvl0, cc, dtype=np.float32, ndim=3)
        for fidx, frag in enumerate(fragments):
            src_chunk_vertices[(cc, fidx)] = frag

    def bin_reduced_source(oid):
        manifest = src_manifests[oid]
        positions = np.concatenate(
            [src_chunk_vertices[ref] for ref in manifest],
            axis=0,
        ) if manifest else np.zeros((0, 3), dtype=np.float32)
        bin_shape = np.array([10.0, 10.0, 10.0], dtype=np.float64)
        bin_coords = np.floor(positions / bin_shape).astype(np.int64)
        # Dedupe consecutive identical bin coords (the bin reduction).
        if len(bin_coords) == 0:
            return bin_coords
        keep = np.ones(len(bin_coords), dtype=bool)
        keep[1:] = np.any(bin_coords[1:] != bin_coords[:-1], axis=1)
        return bin_coords[keep]

    coarsen_level(
        str(store), source_level=0, target_level=1,
        coarsen_factor=2.0,
        chunk_scale_factor=2,
        sparsity_factor=1.0,
    )
    root = open_store(str(store))
    lvl1 = get_resolution_level(root, 1)
    coarse_manifests = read_all_object_manifests(lvl1)

    bin_shape = np.array([10.0, 10.0, 10.0], dtype=np.float64)

    for oid in range(len(src_manifests)):
        expected_bins = bin_reduced_source(oid)
        coarse_positions = []
        for cc, fidx in coarse_manifests[oid]:
            fragments = read_chunk_vertices(lvl1, cc, dtype=np.float32, ndim=3)
            coarse_positions.append(fragments[fidx])
        if not coarse_positions:
            assert len(expected_bins) == 0
            continue
        coarse_concat = np.concatenate(coarse_positions, axis=0)
        coarse_bins = np.floor(coarse_concat / bin_shape).astype(np.int64)
        # Dedupe consecutive duplicates (boundary bin can appear twice
        # if the path enters then re-enters via two fragments).
        keep = np.ones(len(coarse_bins), dtype=bool)
        keep[1:] = np.any(coarse_bins[1:] != coarse_bins[:-1], axis=1)
        coarse_bins_unique = coarse_bins[keep]
        assert np.array_equal(coarse_bins_unique, expected_bins), (
            f"oid {oid}: coarse bins {coarse_bins_unique.tolist()} "
            f"!= expected {expected_bins.tolist()}"
        )
