"""Pin the batched vectorized forest ops (``_forest``) to the per-tree scalar
reference (``decimate_skeleton`` / ``_collapse_to_kept``).

Because the vectorized ``collapse`` uses ``np.flatnonzero(keep)`` ascending
order — exactly like ``_collapse_to_kept`` — a single tree must reproduce the
scalar output *exactly* (keep set, positions, edges, attribute aggregation).  A
multi-tree forest must equal the per-tree results stacked with the right offsets.
"""
from __future__ import annotations

import numpy as np

from zarr_vectors_tools.multiresolution import _forest
from zarr_vectors_tools.multiresolution.strategies.skeletons import (
    _build_rooted_tree,
    _collapse_to_kept,
    decimate_skeleton,
)


def _random_tree(rng, n):
    """Random rooted tree: node i's parent is some j < i (so 0 is the root)."""
    parent = np.full(n, -1, dtype=np.int64)
    for i in range(1, n):
        parent[i] = rng.integers(0, i)
    # [child, parent] edges (skeleton convention)
    edges = np.stack([np.arange(1, n), parent[1:]], axis=1) if n > 1 else np.zeros((0, 2), np.int64)
    return parent, edges


def _decimate_keep_reference(parent, edges, stride, forced):
    """Reproduce decimate_skeleton's keep mask via the scalar tree walk."""
    n = len(parent)
    _, children, roots = _build_rooted_tree(n, edges)
    keep = np.zeros(n, dtype=bool)
    for r in roots:
        keep[r] = True
    for v in range(n):
        nc = len(children.get(v, ()))
        if nc == 0 or nc >= 2:
            keep[v] = True
    if forced is not None:
        for v in np.flatnonzero(forced):
            keep[int(v)] = True
    if stride > 1:
        anchors = np.flatnonzero(keep).tolist()
        for a in anchors:
            for first_child in children.get(a, ()):
                chain = [a]
                cur = first_child
                while True:
                    chain.append(cur)
                    if keep[cur]:
                        break
                    kids = children.get(cur, ())
                    if len(kids) != 1:
                        break
                    cur = kids[0]
                for j in range(stride, len(chain) - 1, stride):
                    keep[chain[j]] = True
    return keep


def test_decimate_keep_matches_reference():
    rng = np.random.default_rng(0)
    for _ in range(500):
        n = int(rng.integers(1, 60))
        parent, edges = _random_tree(rng, n)
        stride = int(rng.integers(1, 6))
        forced = rng.random(n) < 0.1
        ref = _decimate_keep_reference(parent, edges, stride, forced)
        got = _forest.decimate_keep(parent, stride=stride, forced=forced)
        assert np.array_equal(ref, got), (n, stride)


def test_collapse_matches_reference_single_tree():
    rng = np.random.default_rng(1)
    for _ in range(500):
        n = int(rng.integers(1, 60))
        parent, edges = _random_tree(rng, n)
        keep = rng.random(n) < 0.5
        keep[0] = True  # root kept (always true in the decimator)
        attrs = {"radius": rng.uniform(1, 9, size=n).astype(np.float32)}
        for agg in ("max", "mean", "first"):
            ref = _collapse_to_kept(np.arange(n)[:, None].astype(np.float64),
                                    parent, keep, attrs, agg)
            got = _forest.collapse(parent, keep, attrs, agg)
            assert np.array_equal(ref["kept_source_indices"], got["kept"])
            assert np.array_equal(ref["edges"], got["edges"]), (n, agg)
            assert np.allclose(ref["attributes"]["radius"],
                               got["attributes"]["radius"]), (n, agg)


def test_full_decimate_matches_decimate_skeleton():
    """End-to-end: keep + collapse on one tree == decimate_skeleton output."""
    rng = np.random.default_rng(2)
    for _ in range(400):
        n = int(rng.integers(1, 60))
        parent, edges = _random_tree(rng, n)
        pos = rng.uniform(0, 100, size=(n, 3)).astype(np.float32)
        stride = int(rng.integers(2, 6))
        forced_idx = np.flatnonzero(rng.random(n) < 0.1)
        attrs = {"radius": rng.uniform(1, 9, size=n).astype(np.float32)}

        ref = decimate_skeleton(pos, edges, stride=stride,
                                forced_keep=forced_idx, attributes=attrs,
                                attr_agg="max")

        forced_mask = np.zeros(n, dtype=bool)
        forced_mask[forced_idx] = True
        keep = _forest.decimate_keep(parent, stride=stride, forced=forced_mask)
        col = _forest.collapse(parent, keep, attrs, "max")
        assert np.array_equal(ref["kept_source_indices"], col["kept"])
        assert np.array_equal(ref["positions"], pos[col["kept"]])
        assert np.array_equal(ref["edges"], col["edges"]), (n, stride)
        assert np.allclose(ref["attributes"]["radius"],
                           col["attributes"]["radius"])


def test_batched_forest_equals_per_tree():
    """A multi-tree forest decimated in one batch == per-tree results with
    the right index offsets."""
    rng = np.random.default_rng(3)
    sizes = [int(rng.integers(1, 25)) for _ in range(8)]
    parents, off = [], 0
    forced_parts = []
    for s in sizes:
        p, _ = _random_tree(rng, s)
        p = np.where(p >= 0, p + off, -1)
        parents.append(p)
        forced_parts.append(rng.random(s) < 0.1)
        off += s
    parent = np.concatenate(parents)
    forced = np.concatenate(forced_parts)
    stride = 3

    keep_batch = _forest.decimate_keep(parent, stride=stride, forced=forced)

    # Per-tree keep, offset back together.
    keep_ref = np.zeros(len(parent), dtype=bool)
    off = 0
    for s, fp in zip(sizes, forced_parts):
        sub_parent = np.where(parent[off:off + s] >= 0,
                              parent[off:off + s] - off, -1)
        keep_ref[off:off + s] = _forest.decimate_keep(
            sub_parent, stride=stride, forced=fp)
        off += s
    assert np.array_equal(keep_batch, keep_ref)
