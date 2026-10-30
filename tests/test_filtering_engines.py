"""The indexed engine against the oracle, on random regions and random trees.

The indexed engine answers from a grid of cells and a cache of leaves; the
oracle tests every segment.  They must agree exactly, for every region type
and mode, at grid cells smaller than a segment (so segments span and skip
cells), larger than the data, and with regions placed on cell faces.  Then:
a repeated leaf comes from the cache, an operator flip recomputes nothing,
and the published byte mask follows the groups' draw order.
"""

from __future__ import annotations

import numpy as np
import pytest

from zarr_vectors_tools.filtering import (
    All,
    And,
    Box,
    Column,
    Disk,
    IndexedEngine,
    InGroup,
    Mask,
    Not,
    Or,
    Passes,
    Slab,
    Sphere,
    Tractogram,
    evaluate,
    whole_brain_like,
)


@pytest.fixture(scope="module")
def tractogram() -> Tractogram:
    tg = whole_brain_like(
        1500, seed=3, semi_axes=(25.0, 30.0, 20.0), mean_length=30.0,
        length_range=(5.0, 80.0), n_groups=5,
    )
    # One-vertex and empty streamlines ride along with the rest.
    extra = Tractogram.from_polylines([[[0.5, 0.5, 0.5]], np.zeros((0, 3)), [[3.0, -2.0, 1.0]]])
    return Tractogram(
        np.concatenate([tg.vertices, extra.vertices.astype(np.float32)]),
        np.concatenate([tg.offsets, tg.offsets[-1] + extra.offsets[1:]]),
        columns={
            "fa_mean": np.concatenate(
                [tg.columns["fa_mean"], [0.3, np.nan, 0.6]]).astype(np.float32),
            "label": np.concatenate([tg.columns["label"], [0, 1, 2]]).astype(np.int32),
        },
        groups=tg.groups,
    )


def _region(rng: np.random.Generator, tg: Tractogram, engine: IndexedEngine):
    anchor = tg.vertices[rng.integers(len(tg.vertices))].astype(np.float64)
    if rng.random() < 0.3:
        # Snap to the nearest cell corner, so the region sits on cell faces.
        size = engine.cell_size
        anchor = engine.origin + np.round((anchor - engine.origin) / size) * size
    kind = rng.integers(5)
    if kind == 0:
        return Sphere(anchor + rng.normal(scale=0.5, size=3), rng.uniform(0.2, 6.0))
    if kind == 1:
        half = rng.uniform(0.1, 5.0, size=3)
        return Box(anchor - half, anchor + half)
    if kind == 2:
        axis = int(rng.integers(3))
        width = rng.uniform(0.0, 2.0)
        return Slab(axis, anchor[axis] - width, anchor[axis] + width)
    if kind == 3:
        lo = rng.uniform(0, 60)
        return Disk(anchor, rng.normal(size=3), rng.uniform(0.5, 8.0), (lo, rng.uniform(lo, 90)))
    # A mask: a random blob of voxels, sometimes rotated, around the anchor.
    shape = tuple(rng.integers(2, 7, size=3))
    voxels = rng.random(shape) < 0.4
    size = rng.uniform(0.5, 2.5)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3))) if rng.random() < 0.5 else (np.eye(3), None)
    affine = np.eye(4)
    affine[:3, :3] = q * size
    affine[:3, 3] = anchor - affine[:3, :3] @ (np.asarray(shape) / 2.0)
    return Mask(voxels, affine)


def _leaf(rng: np.random.Generator, tg: Tractogram, engine: IndexedEngine):
    pick = rng.random()
    if pick < 0.65:
        region = _region(rng, tg, engine)
        mode = "path" if isinstance(region, Disk) else str(rng.choice(
            ["path", "path", "endpoints", "both_endpoints"]))
        return Passes(region, mode)
    if pick < 0.85:
        return rng.choice([
            Column("length", ">", float(rng.uniform(5, 60))),
            Column("fa_mean", "between", tuple(sorted(rng.uniform(0.1, 0.8, 2)))),
            Column("fa_mean", "absent"),
            Column("label", "in", rng.integers(0, 5, 2).tolist()),
        ])
    return InGroup(tuple(rng.integers(0, 6, 2).tolist()))


def _tree(rng: np.random.Generator, tg: Tractogram, engine: IndexedEngine, depth: int = 0):
    if depth >= 2 or rng.random() < 0.4:
        return _leaf(rng, tg, engine)
    op = rng.integers(4)
    if op == 0:
        return Not(_tree(rng, tg, engine, depth + 1))
    children = [_tree(rng, tg, engine, depth + 1) for _ in range(int(rng.integers(2, 4)))]
    return And(*children) if op < 3 else Or(*children)


@pytest.mark.parametrize("cell_size, long_cells", [
    (None, 64), (0.7, 64), (4.0, 64), (1000.0, 64),
    (0.7, 1),   # every segment spanning two cells goes on the long list
])
def test_indexed_engine_agrees_with_the_oracle(
    tractogram: Tractogram, cell_size, long_cells,
) -> None:
    engine = IndexedEngine(tractogram, cell_size=cell_size, long_cells=long_cells)
    rng = np.random.default_rng(int(cell_size or 0) + long_cells)
    for _ in range(60):
        leaf = _leaf(rng, tractogram, engine)
        np.testing.assert_array_equal(
            engine.evaluate(leaf), evaluate(leaf, tractogram), err_msg=repr(leaf))
    for _ in range(25):
        tree = _tree(rng, tractogram, engine)
        np.testing.assert_array_equal(
            engine.evaluate(tree), evaluate(tree, tractogram), err_msg=repr(tree))


def test_the_test_regions_hit_something(tractogram: Tractogram) -> None:
    # Guard against a property test that passes because every answer is empty.
    engine = IndexedEngine(tractogram)
    rng = np.random.default_rng(0)
    counts = [
        int(evaluate(Passes(_region(rng, tractogram, engine)), tractogram).sum())
        for _ in range(40)
    ]
    assert sum(c > 0 for c in counts) >= 30


def test_segments_in_cells_inside_the_region_are_accepted_untested(
    tractogram: Tractogram,
) -> None:
    engine = IndexedEngine(tractogram, cell_size=1.0)
    for region in (Sphere((0, 0, 0), 6.0), Box((-4, -4, -4), (4, 4, 4)), Slab(2, -2.0, 2.0)):
        got = engine.evaluate(Passes(region))
        np.testing.assert_array_equal(got, evaluate(Passes(region), tractogram))
        assert engine.last_query["inside_cells"] > 0
        assert engine.last_query["accepted_untested"] > 0
        assert engine.last_query["tested"] < engine.last_query["candidates"]


def test_regions_outside_the_data(tractogram: Tractogram) -> None:
    engine = IndexedEngine(tractogram)
    for region in (Sphere((500, 0, 0), 3), Box((-900, -900, -900), (-800, -800, -800)),
                   Slab(0, 400, 401), Mask(np.zeros((2, 2, 2)), np.eye(4))):
        assert not engine.evaluate(Passes(region)).any()
    everything = Box((-1e6, -1e6, -1e6), (1e6, 1e6, 1e6))
    has = np.diff(tractogram.offsets) > 0
    np.testing.assert_array_equal(engine.evaluate(Passes(everything)), has)


def test_an_empty_tractogram() -> None:
    engine = IndexedEngine(Tractogram(np.zeros((0, 3)), np.zeros(1, np.int64)))
    assert engine.evaluate(Passes(Sphere((0, 0, 0), 1))).shape == (0,)
    assert engine.evaluate(All()).shape == (0,)


def test_leaves_are_cached_and_operator_flips_recompute_nothing(
    tractogram: Tractogram, monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = IndexedEngine(tractogram)
    computed: list = []
    real = engine._compute
    monkeypatch.setattr(engine, "_compute", lambda node: computed.append(node) or real(node))
    a = Passes(Sphere((0, 0, 0), 4))
    b = Passes(Box((-5, -5, -5), (0, 5, 5)))
    c = Column("length", ">", 20)
    first = engine.evaluate(And(a, b, c))
    assert len(computed) == 3
    flipped = engine.evaluate(Or(a, Not(b), c))
    assert len(computed) == 3
    np.testing.assert_array_equal(flipped, evaluate(Or(a, Not(b), c), tractogram))
    np.testing.assert_array_equal(first, evaluate(And(a, b, c), tractogram))
    # The same sphere built again is the same leaf; a moved one is not.
    engine.evaluate(Passes(Sphere((0.0, 0.0, 0.0), 4.0)))
    assert len(computed) == 3
    engine.evaluate(Passes(Sphere((0.0, 0.0, 0.5), 4.0)))
    assert len(computed) == 4
    engine.clear()
    engine.evaluate(a)
    assert len(computed) == 5


def test_cached_answers_cannot_be_changed_through_a_result(tractogram: Tractogram) -> None:
    engine = IndexedEngine(tractogram)
    leaf = Column("length", ">", 20)
    answer = engine.evaluate(leaf)
    with pytest.raises(ValueError):
        answer[0] = not answer[0]
    combined = engine.evaluate(And(leaf, All()))
    combined[:] = False
    assert engine.evaluate(leaf).any()


def test_publish_follows_draw_order() -> None:
    first = np.array([1, 1, 0, 0, 0], bool)
    second = np.array([0, 1, 1, 0, 0], bool)
    third = np.array([0, 0, 1, 1, 0], bool)
    assert IndexedEngine.publish([first, second, third]).tolist() == [1, 1, 2, 3, 0]
    with pytest.raises(ValueError, match="255"):
        IndexedEngine.publish([first] * 256)
