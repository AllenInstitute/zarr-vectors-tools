"""The brute-force filter oracle, on streamlines whose answers are known.

Every case here is drawn so that the right answer can be checked by eye:
lines through, past and exactly tangent to each region type, segments that
cross a region with no vertex inside it, one-vertex and empty streamlines,
absent column values.  Both engines must give these answers; the oracle is
held to them here, and the indexed engine is held to the oracle in
test_filtering_engines.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from zarr_vectors_tools.filtering import (
    All,
    Box,
    Column,
    Disk,
    IndexedEngine,
    InGroup,
    Mask,
    Not,
    Passes,
    Slab,
    Sphere,
    Tractogram,
    evaluate,
    histogram,
    summarize,
)


def _line(a, b, n: int = 2) -> np.ndarray:
    return np.linspace(np.asarray(a, float), np.asarray(b, float), n)


def _accepted(tree, tractogram) -> list[int]:
    got = np.flatnonzero(evaluate(tree, tractogram)).tolist()
    # The indexed engine must agree on every known answer too, at a cell
    # size small enough that the segments here span several cells.
    assert np.flatnonzero(IndexedEngine(tractogram, cell_size=1.5).evaluate(tree)).tolist() == got
    return got


# -- sphere ----------------------------------------------------------------------


@pytest.fixture
def around_origin() -> Tractogram:
    return Tractogram.from_polylines([
        _line([-10, 0, 0], [10, 0, 0], 21),       # 0: through the centre
        _line([-10, 5, 0], [10, 5, 0]),           # 1: tangent at exactly r = 5
        _line([-10, 5.000001, 0], [10, 5.000001, 0]),  # 2: misses by 1e-6
        _line([-20, 1, 1], [20, 1, 1]),           # 3: one segment, no vertex inside
        [[1.0, 1.0, 1.0]],                        # 4: a single vertex inside
        [[9.0, 9.0, 9.0]],                        # 5: a single vertex outside
        np.zeros((0, 3)),                         # 6: no geometry at all
        _line([3, 0, 0], [30, 0, 0]),             # 7: starts inside, leaves
        _line([20, 0, 0], [6, 0, 0]),             # 8: stops 1 mm short
    ])


def test_sphere_path(around_origin: Tractogram) -> None:
    assert _accepted(Passes(Sphere((0, 0, 0), 5.0)), around_origin) == [0, 1, 3, 4, 7]


def test_sphere_endpoints(around_origin: Tractogram) -> None:
    sphere = Sphere((0, 0, 0), 5.0)
    # Either end inside: the single vertex (both ends) and the one leaving.
    assert _accepted(Passes(sphere, "endpoints"), around_origin) == [4, 7]
    assert _accepted(Passes(sphere, "both_endpoints"), around_origin) == [4]


def test_a_zero_radius_sphere_is_a_point(around_origin: Tractogram) -> None:
    assert _accepted(Passes(Sphere((1, 1, 1), 0.0)), around_origin) == [3, 4]


# -- box and slab ----------------------------------------------------------------


def test_box_meets_segments_with_no_vertex_inside() -> None:
    tg = Tractogram.from_polylines([
        _line([0, 0, 0], [10, 10, 0]),            # 0: the diagonal clips the corner (4..6)
        _line([0, 6.5, 0], [10, 6.5, 0]),         # 1: passes above the box
        _line([0, 6, 0], [10, 6, 0]),             # 2: runs along the top face
        _line([5, 5, -10], [5, 5, 10]),           # 3: straight through, along z
        _line([4.1, 5.95, 0], [3.9, 6.05, 0]),    # 4: crosses the top-left corner region
        _line([3.0, 5.5, 0], [3.9, 6.6, 0]),      # 5: ends left of the box
    ])
    box = Box((4, 4, -1), (6, 6, 1))
    assert _accepted(Passes(box), tg) == [0, 2, 3, 4]


def test_box_endpoint_modes() -> None:
    tg = Tractogram.from_polylines([
        _line([5, 5, 0], [20, 5, 0]),
        _line([5, 5, 0], [5.5, 5.5, 0.5]),
        _line([-5, 5, 0], [20, 5, 0]),
    ])
    box = Box((4, 4, -1), (6, 6, 1))
    assert _accepted(Passes(box, "endpoints"), tg) == [0, 1]
    assert _accepted(Passes(box, "both_endpoints"), tg) == [1]


def test_slab_is_a_box_unbounded_on_the_other_axes() -> None:
    tg = Tractogram.from_polylines([
        _line([0, -5, 0], [0, 5, 0]),             # 0: crosses y in 0..1, no vertex in it
        _line([100, 0.5, -100], [101, 0.5, -99]),  # 1: inside, far away on x and z
        _line([0, 1.0001, 0], [0, 3, 0]),         # 2: just above
        _line([0, 3, 0], [0, 1, 0]),              # 3: ends on the face
    ])
    assert _accepted(Passes(Slab(1, 0.0, 1.0)), tg) == [0, 1, 3]


# -- disk ------------------------------------------------------------------------


def test_disk_needs_a_crossing_inside_the_radius() -> None:
    tg = Tractogram.from_polylines([
        _line([1, -1, 0], [1, 1, 0]),             # 0: crosses y=0 at (1,0,0), along the normal
        _line([4, -1, 0], [4, 1, 0]),             # 1: crosses at 4 mm, outside r = 3
        _line([-2, 0, 0], [2, 0, 0]),             # 2: lies in the plane: never crosses
        _line([0, 2, 0], [0, 0, 0]),              # 3: ends on the disk
        _line([0, 1, 0], [0, 3, 0]),              # 4: never reaches the plane
        _line([0, 0, 0], [0, 0, 0]),              # 5: a zero-length segment on the plane
    ])
    disk = Disk((0, 0, 0), (0, 1, 0), 3.0)
    assert _accepted(Passes(disk), tg) == [0, 3]


def test_disk_angle_window() -> None:
    # Segments crossing the disk's centre at 10, 50 and 85 degrees to its normal.
    def at(degrees: float) -> np.ndarray:
        t = np.radians(degrees)
        direction = np.array([np.sin(t), np.cos(t), 0.0])
        return np.stack([-direction, direction])

    tg = Tractogram.from_polylines([at(10), at(50), at(85)])
    disk = Disk((0, 0, 0), (0, -2, 0), 3.0)  # the normal's sign and length do not matter
    assert _accepted(Passes(disk), tg) == [0, 1, 2]
    assert _accepted(Passes(Disk(disk.center, disk.normal, 3.0, (0, 30))), tg) == [0]
    assert _accepted(Passes(Disk(disk.center, disk.normal, 3.0, (45, 90))), tg) == [1, 2]
    assert _accepted(Passes(Disk(disk.center, disk.normal, 3.0, (60, 80))), tg) == []


# -- mask ------------------------------------------------------------------------


def test_mask_samples_along_each_segment() -> None:
    voxels = np.zeros((10, 10, 10), dtype=bool)
    voxels[5, 5, 5] = True                         # world 10..12 mm on each axis, 2 mm voxels
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    tg = Tractogram.from_polylines([
        _line([0, 10, 10], [20, 10, 10]),          # 0: through it, no vertex inside
        _line([10.5, 10.5, 10.5], [10.6, 10.6, 10.6]),  # 1: wholly inside
        _line([0, 0, 0], [5, 5, 5]),               # 2: nowhere near
        _line([0, 12.2, 10], [20, 12.2, 10]),      # 3: just outside (y rounds to voxel 6)
    ])
    assert _accepted(Passes(Mask(voxels, affine)), tg) == [0, 1]
    assert _accepted(Passes(Mask(voxels, affine), "both_endpoints"), tg) == [1]


def test_an_empty_mask_accepts_nothing() -> None:
    tg = Tractogram.from_polylines([_line([0, 0, 0], [5, 5, 5])])
    assert _accepted(Passes(Mask(np.zeros((3, 3, 3)), np.eye(4))), tg) == []


# -- columns and groups ----------------------------------------------------------


@pytest.fixture
def with_columns() -> Tractogram:
    lines = [_line([i, 0, 0], [i, 0, 1 + i]) for i in range(6)]   # lengths 1..6
    return Tractogram.from_polylines(
        lines,
        columns={
            "fa": np.array([0.1, 0.2, np.nan, 0.4, 0.5, 0.6], np.float32),
            "label": np.array([3, 3, 7, 7, 9, 9], np.int32),
            "rgb": np.arange(18, dtype=np.float64).reshape(6, 3),
        },
        present={"label": np.array([1, 1, 1, 1, 1, 0], bool)},
        groups={0: [0, 1], 2: [4]},
    )


@pytest.mark.parametrize("leaf, expected", [
    (Column("length", ">", 3), [3, 4, 5]),
    (Column("length", "between", (2, 4)), [1, 2, 3]),
    (Column("n_vertices", "==", 2), [0, 1, 2, 3, 4, 5]),
    (Column("fa", "<", 0.3), [0, 1]),
    (Column("fa", "!=", 0.4), [0, 1, 4, 5]),        # the absent row passes no comparison
    (Column("fa", "absent"), [2]),
    (Column("label", "in", [7, 9]), [2, 3, 4]),     # row 5's label is absent
    (Column("label", "absent"), [5]),
    (Column("rgb", ">=", 9, channel=0), [3, 4, 5]),
    (InGroup((0, 2)), [0, 1, 4]),
    (InGroup((1,)), []),                            # declared, empty
    (InGroup((40,)), []),                           # not declared at all
])
def test_column_and_group_leaves(with_columns: Tractogram, leaf, expected) -> None:
    assert _accepted(leaf, with_columns) == expected


def test_a_multichannel_column_needs_a_channel(with_columns: Tractogram) -> None:
    with pytest.raises(ValueError, match="pick one"):
        evaluate(Column("rgb", ">", 0), with_columns)
    with pytest.raises(KeyError, match="no column"):
        evaluate(Column("missing", ">", 0), with_columns)


def test_composition(with_columns: Tractogram) -> None:
    long = Column("length", ">", 2)
    labelled_7 = Column("label", "==", 7)
    assert _accepted(long & labelled_7, with_columns) == [2, 3]
    assert _accepted(long - labelled_7, with_columns) == [4, 5]
    assert _accepted(~long | InGroup((2,)), with_columns) == [0, 1, 4]
    assert _accepted(Not(All()), with_columns) == []
    assert _accepted(All(), with_columns) == [0, 1, 2, 3, 4, 5]


def test_summary_and_histogram(with_columns: Tractogram) -> None:
    accepted = evaluate(Column("length", ">=", 2), with_columns)   # rows 1..5
    summary = summarize(accepted, with_columns, ["length", "fa", "rgb"])
    assert summary["count"] == 5
    assert summary["length"]["mean"] == pytest.approx(4.0)
    assert summary["length"]["std"] == pytest.approx(np.std([2, 3, 4, 5, 6]))
    # fa is absent on row 2, so 4 of the 5 rows have a value.
    assert summary["fa"]["n"] == 4
    assert summary["fa"]["mean"] == pytest.approx(np.mean([0.2, 0.4, 0.5, 0.6]))
    assert summary["rgb[2]"]["max"] == 17
    nothing = summarize(np.zeros(6, bool), with_columns, ["length"])
    assert nothing["count"] == 0 and nothing["length"]["n"] == 0
    assert np.isnan(nothing["length"]["mean"])
    assert histogram(accepted, with_columns, "length", [0, 3, 10]).tolist() == [1, 4]
