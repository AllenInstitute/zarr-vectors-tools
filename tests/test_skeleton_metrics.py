"""Per-object skeleton metrics, and the vectorised tree engine behind them.

Two skeleton layouts are exercised because they store connectivity
differently: the chunked layout the EM ingesters and the skeleton coarsener
write (path fragments, branch links, undirected cross-chunk coincidence
links), and the ``write_graph`` layout (one fragment per object per chunk,
every stored record re-parenting its child).  Each gets hand-computable
trees, and a randomised comparison against a plain-Python reference taken
from the source trees rather than from the store.
"""

from __future__ import annotations

import math
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from zarr_vectors.building import (
    get_resolution_level,
    list_chunk_keys,
    open_store,
    read_all_object_manifests,
    read_object_attributes,
)
from zarr_vectors.types import skeletons as sk
from zarr_vectors.types.graphs import write_graph
from zarr_vectors.types.points import write_points

from zarr_vectors_tools.algorithms.skeleton_metrics import (
    SKELETON_METRICS,
    compute_skeleton_metrics,
    compute_tree_metrics_vectorized,
)
from zarr_vectors_tools.convert.ingest._tree_enrichments import compute_tree_metrics
from zarr_vectors_tools.multiresolution.object_index import build_object_index

# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------


def _reference(positions, edges, *, root=None) -> dict:
    """Metrics of one object straight from its vertices and undirected edges.

    Strahler order uses ``root`` when given, otherwise the tip with the
    smallest ``(x, y, z)``, per component -- the rule the store pass documents.
    """
    positions = np.asarray(positions, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    n = len(positions)
    degree = np.bincount(edges.reshape(-1), minlength=n)
    neighbours = defaultdict(list)
    for a, b in edges.tolist():
        neighbours[a].append(b)
        neighbours[b].append(a)

    seen = np.zeros(n, dtype=bool)
    max_strahler = 0
    components = 0
    for start in range(n):
        if seen[start]:
            continue
        components += 1
        members, stack = [], [start]
        seen[start] = True
        while stack:
            node = stack.pop()
            members.append(node)
            for other in neighbours[node]:
                if not seen[other]:
                    seen[other] = True
                    stack.append(other)
        if root is not None and root in members:
            chosen = root
        else:
            tips = [m for m in members if degree[m] <= 1] or members
            chosen = min(tips, key=lambda m: (tuple(positions[m]), m))
        parents = np.full(n, -1, dtype=np.int64)
        order, visited = [chosen], {chosen}
        for node in order:
            for other in neighbours[node]:
                if other not in visited:
                    visited.add(other)
                    parents[other] = node
                    order.append(other)
        _, strahler, _ = compute_tree_metrics(parents, root_idx=chosen)
        max_strahler = max(max_strahler, int(strahler[members].max()))

    lengths = np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1)
    return {
        "cable_length": float(lengths.sum()),
        "node_count": n,
        "leaf_count": int(np.count_nonzero(degree <= 1)),
        "branch_count": int(np.count_nonzero(degree >= 3)),
        "component_count": components,
        "max_strahler": max_strahler,
        "extent": np.ptp(positions, axis=0),
    }


def _assert_row(frame, object_id, expected) -> None:
    row = frame.loc[object_id]
    assert row["cable_length"] == pytest.approx(expected["cable_length"], rel=1e-6, abs=1e-6)
    for name in ("node_count", "leaf_count", "branch_count", "component_count", "max_strahler"):
        assert int(row[name]) == expected[name], (object_id, name)
    extent = [row["extent_x"], row["extent_y"], row["extent_z"]]
    assert extent == pytest.approx(list(expected["extent"]), abs=1e-3)


# ---------------------------------------------------------------------------
# Vectorised per-node tree metrics
# ---------------------------------------------------------------------------


def _relabelled(parents: np.ndarray, root: int, rng) -> tuple[np.ndarray, int]:
    """The same tree with its node ids shuffled, so order carries no structure."""
    n = len(parents)
    new_id = rng.permutation(n)
    out = np.full(n, -1, dtype=np.int64)
    has = parents >= 0
    out[new_id[has]] = new_id[parents[has]]
    return out, int(new_id[root])


def _tree(kind: str, n: int, rng) -> np.ndarray:
    idx = np.arange(n)
    if kind == "recursive":
        parents = np.concatenate([[-1], (rng.random(n - 1) * idx[1:]).astype(np.int64)])
    elif kind == "chain":
        parents = idx - 1
    elif kind == "star":
        parents = np.zeros(n, dtype=np.int64)
        parents[0] = -1
    elif kind == "binary":
        parents = (idx - 1) // 2
        parents[0] = -1
    else:  # neuron-like: long runs with occasional branches
        parents = idx - 1
        jumps = np.flatnonzero(rng.random(n) < 0.03)
        jumps = jumps[jumps > 0]
        parents[jumps] = (rng.random(len(jumps)) * jumps).astype(np.int64)
    return parents.astype(np.int64)


@pytest.mark.parametrize("kind", ["recursive", "chain", "star", "binary", "neuron"])
def test_vectorised_tree_metrics_match_reference(kind):
    rng = np.random.default_rng(7)
    # 20_000 nodes crosses the size where list ranking switches strategy.
    for n in (1, 2, 3, 17, 250, 20_000):
        parents, root = _relabelled(_tree(kind, n, rng), 0, rng)
        expected = compute_tree_metrics(parents, root_idx=root)
        got = compute_tree_metrics_vectorized(parents, root_idx=root)
        for e, g, name in zip(expected, got, ("depth", "strahler", "node_kind")):
            assert g.dtype == e.dtype, name
            np.testing.assert_array_equal(g, e, err_msg=f"{kind} n={n} {name}")


def test_vectorised_tree_metrics_zero_outside_the_rooted_tree():
    parents = np.array([-1, 0, 0, -1, 3, 3, 4])
    for e, g in zip(compute_tree_metrics(parents, 0), compute_tree_metrics_vectorized(parents, 0)):
        np.testing.assert_array_equal(g, e)


def test_vectorised_tree_metrics_refuses_a_loop():
    with pytest.raises(ValueError, match="cycle"):
        compute_tree_metrics_vectorized(np.array([1, 2, 0, -1]), root_idx=3)


# ---------------------------------------------------------------------------
# write_graph layout
# ---------------------------------------------------------------------------

# Full binary tree of depth two, rooted at a soma of degree two, crossing
# chunks.  Rooted at the soma its Strahler order is 3; rooted at any tip it
# would be 2, so this pins the stored root being used.
_BINARY_POSITIONS = np.array([
    [5, 5, 5],      # 0 soma
    [15, 5, 5],     # 1
    [25, 0, 5],     # 2
    [25, 10, 5],    # 3
    [5, 15, 5],     # 4
    [0, 25, 5],     # 5
    [10, 25, 5],    # 6
], dtype=np.float32)
_BINARY_EDGES = np.array([[1, 0], [2, 1], [3, 1], [4, 0], [5, 4], [6, 4]])


@pytest.fixture
def graph_store(tmp_path: Path) -> Path:
    """Three objects: the binary tree, a tree re-entering a chunk, one vertex."""
    reentrant = np.array([[1, 1, 1], [12, 1, 1], [14, 1, 1], [5, 5, 1]], dtype=np.float32)
    positions = np.concatenate([
        _BINARY_POSITIONS + 40, reentrant, np.array([[70, 70, 70]], dtype=np.float32),
    ])
    edges = np.concatenate([_BINARY_EDGES, np.array([[8, 7], [9, 8], [10, 9]])])
    object_ids = np.array([0] * 7 + [1] * 4 + [2])
    store = tmp_path / "graph.zv"
    write_graph(
        str(store), positions, edges, chunk_shape=(10.0, 10.0, 10.0),
        kind="skeleton", object_ids=object_ids,
    )
    return store


def test_graph_layout_hand_computed(graph_store):
    frame = compute_skeleton_metrics(graph_store, write=False)
    assert list(frame.index) == [0, 1, 2]
    assert "segment_id" not in frame.columns

    _assert_row(frame, 0, {
        "cable_length": 20 + 4 * math.sqrt(125),
        "node_count": 7, "leaf_count": 4, "branch_count": 2,
        "component_count": 1, "max_strahler": 3, "extent": [25, 25, 0],
    })
    # 0 -> 1 -> 2 -> 3 leaves chunk (0,0,0), crosses to (1,0,0) and comes
    # back: the implied edge between the two rows of chunk (0,0,0) must not
    # be counted.
    _assert_row(frame, 1, {
        "cable_length": 11 + 2 + math.hypot(9, 4),
        "node_count": 4, "leaf_count": 2, "branch_count": 0,
        "component_count": 1, "max_strahler": 1, "extent": [13, 4, 0],
    })
    _assert_row(frame, 2, {
        "cable_length": 0.0, "node_count": 1, "leaf_count": 1, "branch_count": 0,
        "component_count": 1, "max_strahler": 1, "extent": [0, 0, 0],
    })


def test_graph_layout_strahler_matches_swc_enrichment(graph_store):
    parents = np.full(7, -1)
    parents[_BINARY_EDGES[:, 0]] = _BINARY_EDGES[:, 1]
    _, strahler, _ = compute_tree_metrics(parents, root_idx=0)
    frame = compute_skeleton_metrics(graph_store, write=False, metrics=["max_strahler"])
    assert frame.loc[0, "max_strahler"] == strahler.max() == 3


def test_graph_layout_random_forests_match_reference(tmp_path: Path):
    rng = np.random.default_rng(3)
    positions, edges, object_ids, per_object = [], [], [], []
    base = 0
    for oid in range(6):
        n = int(rng.integers(1, 60))
        parents = _tree("neuron" if oid % 2 else "recursive", n, rng)
        steps = rng.normal(0, 4, size=(n, 3))
        pos = np.zeros((n, 3))
        for i in range(1, n):
            pos[i] = pos[parents[i]] + steps[i]
        pos = (pos + rng.uniform(20, 80, 3)).astype(np.float32)
        tree_edges = np.stack([np.arange(1, n), parents[1:]], axis=1)
        positions.append(pos)
        edges.append(tree_edges + base)
        object_ids.append(np.full(n, oid))
        per_object.append(_reference(pos, tree_edges, root=0))
        base += n
    store = tmp_path / "random_graph.zv"
    write_graph(
        str(store), np.concatenate(positions), np.concatenate(edges),
        chunk_shape=(12.0, 12.0, 12.0), kind="skeleton",
        object_ids=np.concatenate(object_ids),
    )
    frame = compute_skeleton_metrics(store, write=False)
    for oid, expected in enumerate(per_object):
        _assert_row(frame, oid, expected)


# ---------------------------------------------------------------------------
# Chunked (EM) layout
# ---------------------------------------------------------------------------


def _write_chunked_store(path: Path) -> None:
    """The binary tree split over two chunks by a cross-chunk link, a
    two-component forest, and a single vertex, written like an EM ingest."""
    root, level = sk.init_skeleton_store(
        str(path), chunk_shape=(20.0, 20.0, 20.0),
        bounds=([0.0, 0.0, 0.0], [40.0, 40.0, 40.0]), ndim=3, attribute_dtypes={},
    )
    records = []
    # Segment 100: soma (15,5,5) with a branch in each chunk.
    left = np.array([[15, 5, 5], [15, 15, 5], [11, 19, 5], [19, 19, 5]], np.float32)
    right = np.array([[25, 5, 5], [35, 1, 5], [35, 9, 5]], np.float32)
    # Segment 200: a two-vertex piece and a lone vertex, never joined.
    pair = np.array([[2, 2, 2], [4, 2, 2]], np.float32)
    lone = np.array([[8, 8, 8]], np.float32)
    recs, anchors_left = sk.write_skeleton_chunk(level, (0, 0, 0), [
        {"segment_id": 100, "positions": left, "edges": np.array([[1, 0], [2, 1], [3, 1]]),
         "attributes": {}, "anchors": {"soma": 0}},
        {"segment_id": 200, "positions": pair, "edges": np.array([[1, 0]]), "attributes": {}},
        {"segment_id": 200, "positions": lone, "edges": np.zeros((0, 2)), "attributes": {}},
    ])
    records += recs
    recs, anchors_right = sk.write_skeleton_chunk(level, (1, 0, 0), [
        {"segment_id": 100, "positions": right, "edges": np.array([[1, 0], [2, 0]]),
         "attributes": {}, "anchors": {"child": 0}},
    ])
    records += recs
    # Segment 300: one vertex in a chunk of its own.
    recs, _ = sk.write_skeleton_chunk(level, (1, 1, 1), [
        {"segment_id": 300, "positions": np.array([[30, 30, 30]], np.float32),
         "edges": np.zeros((0, 2)), "attributes": {}},
    ])
    records += recs
    sk.write_skeleton_cross_chunk_links(
        level, [(anchors_left["soma"], anchors_right["child"])], ndim=3,
    )
    build_object_index(level, records, ndim=3)
    sk.finalize_skeleton_store(root)


@pytest.fixture
def chunked_store(tmp_path: Path) -> Path:
    path = tmp_path / "chunked.zv"
    _write_chunked_store(path)
    return path


def test_chunked_layout_hand_computed(chunked_store):
    frame = compute_skeleton_metrics(chunked_store, write=False)
    assert list(frame["segment_id"]) == [100, 200, 300]

    # No stored root here, so Strahler order is taken from the tip with the
    # smallest (x, y, z), (11, 19, 5): 2, where the soma would give 3.
    _assert_row(frame, 0, {
        "cable_length": 20 + 2 * math.sqrt(116) + 2 * math.sqrt(32),
        "node_count": 7, "leaf_count": 4, "branch_count": 2,
        "component_count": 1, "max_strahler": 2, "extent": [24, 18, 0],
    })
    _assert_row(frame, 1, {
        "cable_length": 2.0, "node_count": 3, "leaf_count": 3, "branch_count": 0,
        "component_count": 2, "max_strahler": 1, "extent": [6, 6, 6],
    })
    _assert_row(frame, 2, {
        "cable_length": 0.0, "node_count": 1, "leaf_count": 1, "branch_count": 0,
        "component_count": 1, "max_strahler": 1, "extent": [0, 0, 0],
    })


def test_object_ids_subset_matches_full_pass(chunked_store):
    full = compute_skeleton_metrics(chunked_store, write=False)
    subset = compute_skeleton_metrics(chunked_store, write=False, object_ids=[2, 0])
    assert list(subset.index) == [0, 2]
    for oid in (0, 2):
        assert subset.loc[oid].to_dict() == pytest.approx(full.loc[oid].to_dict())


def test_executor_path_matches_serial(chunked_store):
    calls = []

    def executor(func, items, shared=None):
        items = list(items)
        calls.append(len(items))
        return [func(item, shared=shared) for item in items]

    serial = compute_skeleton_metrics(chunked_store, write=False)
    mapped = compute_skeleton_metrics(chunked_store, write=False, executor=executor)
    assert calls == [3]
    assert mapped.equals(serial)


def _coincident_ingest(store: Path, *, strides=(2,), sparsity=(1.0,)) -> int:
    """Aligned precomputed ingest where igneous duplicated a boundary vertex."""
    from zarr_vectors_tools.convert.ingest.precomputed_skeletons import (
        InMemoryFragsReader,
        SkeletonInfo,
        enumerate_frag_keys,
        run_ingest,
    )
    info = SkeletonInfo(
        base_url="mem://x", resolution_nm=(32.0, 32.0, 40.0),
        chunk_size_nm=(16384.0, 16384.0, 20480.0))
    keys = enumerate_frag_keys(info, (17398, 10448, 3088), (2, 1, 1))
    shared = [17910 * 32 + 16, 340000.0, 130000.0]
    y, z = shared[1], shared[2]
    seg = 720575940000000999
    a = np.array([[560000, y, z], [565000, y, z], [570000, y, z], shared], np.float32)
    b = np.array([shared, [576000, y, z], [580000, y, z], [578000, y + 3000, z]], np.float32)
    reader = InMemoryFragsReader(info, {
        keys[0]: {
            seg: {"vertices": a, "edges": np.array([[1, 0], [2, 1], [3, 2]])},
            5: {"vertices": a[:2] + 1000, "edges": np.array([[1, 0]])},
        },
        keys[1]: {seg: {"vertices": b, "edges": np.array([[1, 0], [2, 1], [3, 1]])}},
    })
    bounds = ([17398 * 32.0, 10448 * 32.0, 3088 * 40.0],
              [18422 * 32.0, 10960 * 32.0, 3600 * 40.0])
    summary = run_ingest(
        reader, str(store), keys, bounds_nm=bounds, strides=list(strides),
        chunk_scale_factors=[2] * len(strides), sparsity_factors=list(sparsity),
        align=True, progress=False,
    )
    return int(summary["level0_cross_chunk_edges"])


def test_duplicated_boundary_vertex_counts_once(tmp_path: Path):
    store = tmp_path / "coincident.zv"
    assert _coincident_ingest(store) == 1
    expected = {
        "cable_length": 5000 + 5000 + 3136 + 2864 + 4000 + math.hypot(2000, 3000),
        "node_count": 7, "leaf_count": 3, "branch_count": 1,
        "component_count": 1, "max_strahler": 2, "extent": [20000, 3000, 0],
    }
    level0 = compute_skeleton_metrics(store, level=0, write=False)
    assert list(level0["segment_id"]) == [5, 720575940000000999]
    _assert_row(level0, 1, expected)
    _assert_row(level0, 0, {
        "cable_length": 5000.0, "node_count": 2, "leaf_count": 2, "branch_count": 0,
        "component_count": 1, "max_strahler": 1, "extent": [5000, 0, 0],
    })
    # One level up the duplicate is merged and stride 2 drops one interior
    # vertex of a straight run: same cable, same topology, one vertex fewer.
    level1 = compute_skeleton_metrics(store, level=1, write=False)
    _assert_row(level1, 1, dict(expected, node_count=6))


def _random_frags_reader(rng, n_segments: int):
    """Random branching segments in one .frags, spread over several zarr chunks."""
    from zarr_vectors_tools.convert.ingest.precomputed_skeletons import (
        InMemoryFragsReader,
        SkeletonInfo,
    )
    info = SkeletonInfo(
        base_url="mem://x", resolution_nm=(32.0, 32.0, 40.0),
        chunk_size_nm=(16384.0, 16384.0, 20480.0))
    origin = np.array([573120.0, 334336.0, 123520.0])
    pieces, truth = {}, {}
    for s in range(n_segments):
        n = int(rng.integers(2, 50))
        parents = _tree("neuron" if s % 2 else "recursive", n, rng)
        pos = np.zeros((n, 3))
        steps = rng.normal(0, 2500, size=(n, 3))
        for i in range(1, n):
            pos[i] = pos[parents[i]] + steps[i]
        # Clipped into the store bounds: the writer refuses a vertex outside
        # its chunk grid, and the reference is taken from these same values.
        pos = np.clip(
            pos + origin + rng.uniform(15000, 50000, 3), origin + 1000, origin + 69000,
        ).astype(np.float32)
        edges = np.stack([np.arange(1, n), parents[1:]], axis=1)
        seg = 720575940000000000 + 17 * s
        pieces[seg] = {"vertices": pos, "edges": edges}
        truth[seg] = _reference(pos, edges)
    key = "17910-18422_10448-10960_3088-3600.frags"
    bounds = (origin.tolist(), (origin + 70000).tolist())
    return InMemoryFragsReader(info, {key: pieces}), key, bounds, truth


# Several seconds of ingest and coarsening.  The fast tier still crosses chunk
# boundaries: test_chunked_layout_hand_computed (an explicit cross-chunk edge)
# and test_duplicated_boundary_vertex_counts_once (a coincident one).
@pytest.mark.slow
def test_phase_split_ingest_matches_reference_at_every_level(tmp_path: Path):
    from zarr_vectors_tools.convert.ingest.precomputed_skeletons import run_ingest

    rng = np.random.default_rng(11)
    reader, key, bounds, truth = _random_frags_reader(rng, 8)
    store = tmp_path / "phase_split.zv"
    # align=False phase-splits every segment over the origin-0 grid, so most
    # of them are held together only by cross-chunk links.  Stride 1 keeps
    # every vertex, and a 64x chunk scale folds the whole cutout into one
    # level-1 chunk, so level 1 -- pieces merged by the coarsener, no
    # cross-chunk links left -- must measure the same.  (At a smaller scale
    # the coarsener only re-links coincident boundary vertices, and a
    # phase-split edge crossing a level-1 chunk face is not one.)
    summary = run_ingest(
        reader, str(store), [key], bounds_nm=bounds, strides=[1],
        chunk_scale_factors=[64], sparsity_factors=[1.0], align=False, progress=False,
    )
    assert summary["level0_cross_chunk_edges"] > 10
    level1 = get_resolution_level(open_store(str(store)), 1)
    assert len(list_chunk_keys(level1)) == 1
    for level in (0, 1):
        frame = compute_skeleton_metrics(store, level=level, write=False)
        for oid, seg in zip(frame.index, frame["segment_id"]):
            _assert_row(frame, oid, truth[int(seg)])


# ---------------------------------------------------------------------------
# Writing, and the attribute sparsity strategy
# ---------------------------------------------------------------------------


def test_metrics_are_written_as_object_attributes(chunked_store):
    frame = compute_skeleton_metrics(chunked_store)
    level = get_resolution_level(open_store(str(chunked_store)), 0)
    for name in SKELETON_METRICS:
        stored = np.asarray(read_object_attributes(level, name))
        if name == "extent":
            assert stored.shape == (3, 3)
            np.testing.assert_allclose(
                stored, frame[["extent_x", "extent_y", "extent_z"]].to_numpy(),
            )
        else:
            assert stored.shape == (3,)
            np.testing.assert_allclose(stored, frame[name].to_numpy())
    assert np.asarray(read_object_attributes(level, "node_count")).dtype == np.int64

    # Re-running replaces the arrays rather than refusing.
    again = compute_skeleton_metrics(chunked_store, metrics=["cable_length"])
    assert list(again.columns) == ["segment_id", "cable_length"]


def test_write_false_leaves_the_store_untouched(chunked_store):
    compute_skeleton_metrics(chunked_store, write=False)
    level = get_resolution_level(open_store(str(chunked_store)), 0)
    assert not level.array_exists("object_attributes/cable_length")


def _selection_store(path: Path) -> dict[int, float]:
    """Objects whose cable length and fragment count rank them differently.

    Straight single-fragment lines 5 to 8 long, and zigzags 4.8 long that
    hop back and forth over a chunk face, one fragment per vertex: the
    default ``"length"`` strategy (which ranks by fragment count) would keep
    the zigzags, cable length keeps the lines.
    """
    root, level = sk.init_skeleton_store(
        str(path), chunk_shape=(10.0, 10.0, 10.0),
        bounds=([0.0, 0.0, 0.0], [80.0, 80.0, 80.0]), ndim=3, attribute_dtypes={},
    )
    by_chunk: dict[tuple, list] = defaultdict(list)
    links, cable = [], {}
    for s in range(4):
        seg = 10 + s
        x = 1.0 + 2 * s
        line = np.array([[x, 1, 1], [x, 1 + 5 + s, 1]], np.float32)
        by_chunk[(0, 0, 0)].append(
            {"segment_id": seg, "positions": line, "edges": np.array([[1, 0]]), "attributes": {}},
        )
        cable[seg] = 5.0 + s
    for s in range(4):
        seg = 20 + s
        z = 21.0 + 10 * s
        xs = [9.5, 10.5, 9.6, 10.6, 9.7, 10.7]
        cable[seg] = 1.0 + 0.9 + 1.0 + 0.9 + 1.0
        for i, x in enumerate(xs):
            cc = (int(x // 10), 0, int(z // 10))
            by_chunk[cc].append({
                "segment_id": seg, "positions": np.array([[x, 5, z]], np.float32),
                "edges": np.zeros((0, 2)), "attributes": {}, "anchors": {(seg, i): 0},
            })
            if i:
                links.append(((seg, i - 1), (seg, i)))
    records, where = [], {}
    for cc, pieces in sorted(by_chunk.items()):
        recs, anchors = sk.write_skeleton_chunk(level, cc, pieces)
        records += recs
        where.update(anchors)
    sk.write_skeleton_cross_chunk_links(level, [(where[a], where[b]) for a, b in links], ndim=3)
    build_object_index(level, records, ndim=3)
    sk.finalize_skeleton_store(root)
    return cable


def test_cable_length_drives_the_attribute_sparsity_strategy(tmp_path: Path, monkeypatch):
    from zarr_vectors_tools.multiresolution import object_selection
    from zarr_vectors_tools.multiresolution.strategies.skeletons import coarsen_skeleton_level

    store = tmp_path / "select.zv"
    cable = _selection_store(store)
    frame = compute_skeleton_metrics(store, metrics=["cable_length"])
    segment_of = dict(zip(frame.index, frame["segment_id"]))
    for oid, seg in segment_of.items():
        assert frame.loc[oid, "cable_length"] == pytest.approx(cable[int(seg)])

    level0 = get_resolution_level(open_store(str(store)), 0)
    stored = np.asarray(read_object_attributes(level0, "cable_length"), dtype=np.float64)

    # No coarsener passes attribute_values to apply_sparsity yet, so
    # sparsity_strategy="attribute" cannot be driven by name.  This stands in
    # for that parameter with the stored column, to show it lines up with the
    # coarsener's object ids and selects on what it says.
    real = object_selection.apply_sparsity
    seen = {}

    def with_stored_metric(n_objects, sparsity, strategy="random", **kwargs):
        seen["strategy"] = strategy
        kwargs["attribute_values"] = stored
        return real(n_objects, sparsity, strategy, **kwargs)

    monkeypatch.setattr(object_selection, "apply_sparsity", with_stored_metric)
    coarsen_skeleton_level(
        str(store), 0, 1, stride=1, sparsity_factor=2.0, chunk_scale_factor=2,
        sparsity_strategy="attribute",
    )
    assert seen["strategy"] == "attribute"

    level1 = get_resolution_level(open_store(str(store)), 1)
    kept = {oid for oid, m in enumerate(read_all_object_manifests(level1)) if m}
    longest = set(np.argsort(-stored, kind="stable")[: len(stored) // 2].tolist())
    assert kept == longest
    assert {int(segment_of[o]) for o in kept} == {10, 11, 12, 13}

    # Objects the drop emptied read 0, and NaN for extent.
    coarse = compute_skeleton_metrics(store, level=1, write=False)
    dropped = sorted(set(range(len(stored))) - kept)
    assert (coarse.loc[dropped, ["cable_length", "node_count", "max_strahler"]] == 0).all().all()
    assert coarse.loc[dropped, ["extent_x", "extent_y", "extent_z"]].isna().all().all()
    for oid in kept:
        assert coarse.loc[oid, "cable_length"] == pytest.approx(stored[oid])


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_refuses_a_store_that_is_not_a_skeleton(tmp_path: Path):
    points = tmp_path / "points.zv"
    write_points(str(points), np.random.default_rng(0).random((20, 3)), chunk_shape=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="needs a skeleton store"):
        compute_skeleton_metrics(points, write=False)

    graph = tmp_path / "graph.zv"
    write_graph(
        str(graph), np.array([[0, 0, 0], [1, 1, 1], [2, 0, 0]], np.float32),
        np.array([[0, 1], [1, 2], [2, 0]]), chunk_shape=(5.0, 5.0, 5.0),
    )
    with pytest.raises(ValueError, match="links_convention"):
        compute_skeleton_metrics(graph, write=False)


def test_refuses_unknown_metrics_and_partial_writes(chunked_store):
    with pytest.raises(ValueError, match="unknown skeleton metric.*'volume'"):
        compute_skeleton_metrics(chunked_store, metrics=["cable_length", "volume"], write=False)
    with pytest.raises(ValueError, match="write=False"):
        compute_skeleton_metrics(chunked_store, object_ids=[0])
    with pytest.raises(ValueError, match=r"object id\(s\) \[7\]"):
        compute_skeleton_metrics(chunked_store, object_ids=[7], write=False)


# ---------------------------------------------------------------------------
# Parallel
# ---------------------------------------------------------------------------


def _process_pool(func, items, shared=None):
    from concurrent.futures import ProcessPoolExecutor

    items = list(items)
    with ProcessPoolExecutor(max_workers=2) as pool:
        return list(pool.map(partial(func, shared=shared), items))


@pytest.mark.slow
def test_process_pool_executor_matches_serial(chunked_store):
    serial = compute_skeleton_metrics(chunked_store, write=False)
    parallel = compute_skeleton_metrics(chunked_store, write=False, executor=_process_pool)
    assert parallel.equals(serial)
