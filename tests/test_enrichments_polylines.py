"""Tests for the shared polyline-enrichment helpers."""

from __future__ import annotations

import numpy as np

from zarr_vectors_tools.ingest._polyline_enrichments import (
    arc_length_normalized,
    compute_endpoints,
    compute_lengths,
    compute_orientation,
    compute_tangents,
    compute_tortuosity,
    compute_vertex_counts,
    filter_by_length,
    index_normalized,
)


def _make_polyline(points):
    return np.array(points, dtype=np.float32)


class TestComputeLengths:

    def test_known_length(self) -> None:
        # Right-angle L: 3 + 4 = 5? No — segments are 3 and 4 → total = 7.
        p = _make_polyline([[0, 0, 0], [3, 0, 0], [3, 4, 0]])
        lengths = compute_lengths([p])
        assert lengths.shape == (1,)
        np.testing.assert_allclose(lengths[0], 7.0, atol=1e-5)

    def test_single_point_zero_length(self) -> None:
        p = _make_polyline([[1, 2, 3]])
        lengths = compute_lengths([p])
        assert lengths[0] == 0.0

    def test_empty_list(self) -> None:
        lengths = compute_lengths([])
        assert lengths.shape == (0,)

    def test_multiple(self) -> None:
        a = _make_polyline([[0, 0, 0], [1, 0, 0]])  # length 1
        b = _make_polyline([[0, 0, 0], [0, 2, 0], [0, 2, 2]])  # length 4
        lengths = compute_lengths([a, b])
        np.testing.assert_allclose(lengths, [1.0, 4.0], atol=1e-5)


class TestComputeEndpoints:

    def test_first_and_last(self) -> None:
        p = _make_polyline([[0, 0, 0], [1, 1, 1], [5, 5, 5]])
        start, end = compute_endpoints([p])
        np.testing.assert_allclose(start[0], [0, 0, 0])
        np.testing.assert_allclose(end[0], [5, 5, 5])

    def test_two_polylines(self) -> None:
        a = _make_polyline([[1, 2, 3], [10, 20, 30]])
        b = _make_polyline([[4, 5, 6], [40, 50, 60]])
        start, end = compute_endpoints([a, b])
        assert start.shape == (2, 3) and end.shape == (2, 3)
        np.testing.assert_allclose(start[1], [4, 5, 6])
        np.testing.assert_allclose(end[1], [40, 50, 60])


class TestFilterByLength:

    def test_within_range_kept(self) -> None:
        polys = [
            _make_polyline([[0, 0, 0], [1, 0, 0]]),     # 1
            _make_polyline([[0, 0, 0], [5, 0, 0]]),     # 5
            _make_polyline([[0, 0, 0], [100, 0, 0]]),   # 100
        ]
        kept, idx, dropped = filter_by_length(polys, (2.0, 10.0))
        assert len(kept) == 1
        np.testing.assert_array_equal(idx, [1])
        assert dropped == 2

    def test_nothing_dropped(self) -> None:
        polys = [_make_polyline([[0, 0, 0], [1, 0, 0]])]
        kept, idx, dropped = filter_by_length(polys, (0.0, 10.0))
        assert len(kept) == 1 and dropped == 0

    def test_uses_precomputed_lengths(self) -> None:
        polys = [_make_polyline([[0, 0, 0], [1, 0, 0]])]
        lengths = np.array([42.0], dtype=np.float32)  # lie about the length
        kept, _, dropped = filter_by_length(polys, (0.0, 1.0), lengths=lengths)
        # The lie wins because filter_by_length trusts the precomputed value.
        assert len(kept) == 0 and dropped == 1


class TestSyntheticObjectAttributes:

    def test_orientation_unit_vector(self) -> None:
        p = _make_polyline([[0, 0, 0], [1, 1, 1], [0, 3, 0]])  # end - start = (0,3,0)
        o = compute_orientation([p])
        assert o.shape == (1, 3)
        np.testing.assert_allclose(o[0], [0, 1, 0], atol=1e-5)

    def test_orientation_degenerate_is_zero(self) -> None:
        p = _make_polyline([[2, 2, 2], [5, 5, 5], [2, 2, 2]])  # start == end
        o = compute_orientation([p])
        np.testing.assert_allclose(o[0], [0, 0, 0])

    def test_tortuosity_straight_line_is_one(self) -> None:
        p = _make_polyline([[0, 0, 0], [5, 0, 0], [10, 0, 0]])  # length 10, dist 10
        t = compute_tortuosity([p])
        np.testing.assert_allclose(t[0], 1.0, atol=1e-5)

    def test_tortuosity_l_shape_gt_one(self) -> None:
        p = _make_polyline([[0, 0, 0], [3, 0, 0], [3, 4, 0]])  # length 7, dist 5
        t = compute_tortuosity([p])
        np.testing.assert_allclose(t[0], 7.0 / 5.0, atol=1e-5)

    def test_tortuosity_closed_loop_is_one(self) -> None:
        p = _make_polyline([[0, 0, 0], [1, 0, 0], [0, 0, 0]])  # start == end
        t = compute_tortuosity([p])
        np.testing.assert_allclose(t[0], 1.0)

    def test_vertex_counts(self) -> None:
        a = _make_polyline([[0, 0, 0], [1, 0, 0]])
        b = _make_polyline([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        vc = compute_vertex_counts([a, b])
        assert vc.dtype == np.uint32
        np.testing.assert_array_equal(vc, [2, 3])


class TestSyntheticVertexAttributes:

    def test_arc_length_endpoints_and_monotonic(self) -> None:
        p = _make_polyline([[0, 0, 0], [3, 0, 0], [3, 4, 0]])  # segs 3, 4 → cum 0,3,7
        al = arc_length_normalized(p)
        assert al.shape == (3,)
        np.testing.assert_allclose(al, [0.0, 3.0 / 7.0, 1.0], atol=1e-5)
        assert np.all(np.diff(al) >= 0)

    def test_arc_length_zero_length_is_zeros(self) -> None:
        p = _make_polyline([[1, 1, 1], [1, 1, 1]])  # coincident
        np.testing.assert_allclose(arc_length_normalized(p), [0.0, 0.0])

    def test_index_normalized(self) -> None:
        np.testing.assert_allclose(index_normalized(5), [0, 0.25, 0.5, 0.75, 1.0], atol=1e-6)
        np.testing.assert_allclose(index_normalized(1), [0.0])
        assert index_normalized(0).shape == (0,)

    def test_tangent_unit_norm_and_direction(self) -> None:
        p = _make_polyline([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
        tan = compute_tangents(p)
        assert tan.shape == (4, 3)
        norms = np.linalg.norm(tan, axis=1)
        np.testing.assert_allclose(norms, np.ones(4), atol=1e-5)
        # A straight x-line has +x tangent everywhere.
        np.testing.assert_allclose(tan, np.tile([1, 0, 0], (4, 1)), atol=1e-5)

    def test_tangent_single_vertex_is_zero(self) -> None:
        p = _make_polyline([[7, 7, 7]])
        np.testing.assert_allclose(compute_tangents(p), np.zeros((1, 3)))
