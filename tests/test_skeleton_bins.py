"""Tests for the connected-bin metavertex skeleton coarsener (pure kernel).

The incumbent stride coarsener shipped a pyramid in which every neuron was
severed into ~54 pieces and no vertex had a bounded error, and neither fault was
caught because nothing asserted them. So the invariants here are stated as
properties of the algorithm, not as golden numbers:

* **B1** Components are preserved exactly — the count is identical before and
  after, for any bin size, on any forest. This is the fault that made the
  incumbent's coarse levels unusable.
* **B2** Two branches that merely pass through one bin are never welded.
  Clustering by proximity alone would fuse them and invent a cycle.
* **B3** Error is bounded by the bin diagonal. Index-stride decimation has no
  bound at any ratio; this is the guarantee that replaces it.
* **B4** The result is a forest with inherited ``[child, parent]`` orientation:
  ``V - E == components``, no self-loops, no re-rooting.
* **B5** ``owner_new`` is total — every source vertex resolves to a metavertex,
  which is what lets a cross-chunk link endpoint survive coarsening.
* **B6** Attributes aggregate per kind: a continuous attribute averages, a
  categorical one takes a value that genuinely occurred, and dtypes survive.
* **B7** Heavy-path relabelling lengthens runs but cannot change how many
  fragments there are.
"""

from __future__ import annotations

import numpy as np
import pytest

from zarr_vectors_tools.multiresolution.strategies.skeleton_bins import (
    contract_skeleton_bins,
    default_attr_agg,
    heavy_path_relabel,
)


def _components(n: int, edges: np.ndarray) -> int:
    """Connected-component count by union-find, independent of the kernel."""
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in np.asarray(edges).reshape(-1, 2):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    return len({find(i) for i in range(n)})


def _random_forest(rng, n_max: int = 150):
    """A random rooted forest as ``[child, parent]`` rows, with real geometry."""
    n = int(rng.integers(5, n_max))
    pos = np.cumsum(rng.normal(0, 3.0, size=(n, 3)), axis=0).astype(np.float32)
    rows = [
        [i, int(rng.integers(0, i))] for i in range(1, n) if rng.random() < 0.9
    ]
    return pos, np.array(rows, dtype=np.int64).reshape(-1, 2)


# --------------------------------------------------------------------------
# B1 / B3 / B4 / B5 — the structural guarantees, over many random forests
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bin_shape", [2.0, 5.0, 20.0])
def test_b1_b3_b4_b5_invariants_hold_on_random_forests(bin_shape):
    rng = np.random.default_rng(0)
    for _ in range(120):
        pos, edges = _random_forest(rng)
        out = contract_skeleton_bins(pos, edges, bin_shape=bin_shape)
        k = len(out["positions"])

        # B1: component count is invariant.
        assert _components(len(pos), edges) == _components(k, out["edges"])

        # B4: still a forest, no self-loops.
        assert k - len(out["edges"]) == _components(k, out["edges"])
        assert not np.any(out["edges"][:, 0] == out["edges"][:, 1])

        # B5: owner_new is total and in range.
        assert out["owner_new"].shape == (len(pos),)
        assert out["owner_new"].min() >= 0 and out["owner_new"].max() < k

        # B3: no vertex is further from its metavertex than the bin diagonal.
        dev = np.linalg.norm(pos - out["positions"][out["owner_new"]], axis=1)
        assert dev.max() <= bin_shape * np.sqrt(3) + 1e-6


def test_b2_branches_sharing_a_bin_are_not_welded():
    """Proximity alone must not merge two components (it would add a cycle)."""
    pos = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [100, 0, 0], [101, 0, 0], [1.2, 0.2, 0]],
        dtype=np.float32,
    )
    # Two components; vertex 5 sits inside vertex 1's bin but is not joined to it.
    edges = np.array([[1, 0], [2, 1], [4, 3], [5, 4]], dtype=np.int64)
    out = contract_skeleton_bins(pos, edges, bin_shape=10.0)
    assert _components(len(pos), edges) == _components(
        len(out["positions"]), out["edges"]
    )


def test_b4_orientation_is_inherited_not_recomputed():
    """A contracted chain keeps [child, parent] pointing back towards the root."""
    pos = (np.arange(10, dtype=np.float32)[:, None] * np.ones((1, 3), np.float32))
    edges = np.array([[i, i - 1] for i in range(1, 10)], dtype=np.int64)
    out = contract_skeleton_bins(pos, edges, bin_shape=4.0)
    assert len(out["positions"]) == 3
    # Child index always exceeds parent index for a contracted ascending chain.
    assert np.all(out["edges"][:, 0] > out["edges"][:, 1])


def test_bin_zero_or_negative_is_the_identity():
    pos, edges = _random_forest(np.random.default_rng(7))
    out = contract_skeleton_bins(pos, edges, bin_shape=0.0)
    assert len(out["positions"]) == len(pos)
    assert np.array_equal(out["owner_new"], np.arange(len(pos)))


def test_empty_input_returns_the_full_result_shape():
    out = contract_skeleton_bins(
        np.zeros((0, 3), np.float32), np.zeros((0, 2), np.int64), bin_shape=4.0
    )
    for key in ("positions", "edges", "attributes", "kept_source_indices", "owner_new"):
        assert key in out
    assert len(out["positions"]) == 0


# --------------------------------------------------------------------------
# B6 — attributes
# --------------------------------------------------------------------------


def test_b6_continuous_averages_categorical_takes_a_real_member():
    pos = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    edges = np.array([[1, 0], [2, 1]], dtype=np.int64)
    out = contract_skeleton_bins(
        pos,
        edges,
        bin_shape=100.0,  # one metavertex
        attributes={
            "radius": np.array([10, 20, 30], np.float32),
            "compartment": np.array([2, 2, 3], np.uint8),
        },
    )
    assert out["attributes"]["radius"][0] == pytest.approx(20.0)
    # Never averaged into a code that does not exist.
    assert out["attributes"]["compartment"][0] in (2, 3)
    assert out["attributes"]["radius"].dtype == np.float32
    assert out["attributes"]["compartment"].dtype == np.uint8


def test_b6_defaults_are_chosen_per_name_then_dtype():
    modes = default_attr_agg(
        {"radius": np.float32, "compartment": np.uint8, "score": np.float64}
    )
    assert modes == {"radius": "mean", "compartment": "nearest", "score": "mean"}


def test_b6_max_does_not_clamp_negative_values():
    """The incumbent seeds its max accumulator with zeros, clamping negatives."""
    pos = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    edges = np.array([[1, 0], [2, 1]], dtype=np.int64)
    out = contract_skeleton_bins(
        pos,
        edges,
        bin_shape=100.0,
        attributes={"v": np.array([-5, -3, -9], np.float32)},
        attr_agg="max",
    )
    assert out["attributes"]["v"][0] == pytest.approx(-3.0)


def test_b6_unknown_aggregation_is_rejected():
    pos = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    edges = np.array([[1, 0]], dtype=np.int64)
    with pytest.raises(ValueError, match="unknown attribute aggregation"):
        contract_skeleton_bins(
            pos,
            edges,
            bin_shape=100.0,
            attributes={"v": np.array([1, 2], np.float32)},
            attr_agg="median",
        )


# --------------------------------------------------------------------------
# B7 — heavy-path relabelling
# --------------------------------------------------------------------------


def test_b7_heavy_path_lengthens_runs_without_changing_fragment_count():
    from zarr_vectors.building import decompose_tree_to_paths

    def run_lengths(pos, edges):
        _, _, frag_ranges, _, _ = decompose_tree_to_paths(
            {"positions": pos, "edges": edges, "attributes": {}}
        )
        return sorted((count for _, count in frag_ranges), reverse=True)

    # Branch at v1: the SHORT arm has the lower index, so the default
    # lowest-index rule follows it and cuts the long arm into its own fragment.
    pos = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [1, 9, 0], [1, 19, 0], [1, 29, 0]],
        dtype=np.float32,
    )
    edges = np.array([[1, 0], [2, 1], [3, 2], [4, 1], [5, 4], [6, 5]], dtype=np.int64)

    before = run_lengths(pos, edges)
    pos2, edges2, _, new_of_old = heavy_path_relabel(pos, edges)
    after = run_lengths(pos2, edges2)

    assert after[0] > before[0], "the heavy arm should become one longer run"
    assert len(after) == len(before), "ordering cannot change fragment count"
    assert sorted(new_of_old.tolist()) == list(range(len(pos)))


def test_b7_relabel_preserves_topology_and_attributes():
    rng = np.random.default_rng(3)
    pos, edges = _random_forest(rng)
    attrs = {"radius": rng.random(len(pos)).astype(np.float32)}
    pos2, edges2, attrs2, new_of_old = heavy_path_relabel(pos, edges, attrs)

    assert _components(len(pos), edges) == _components(len(pos2), edges2)
    # Relabelling is a permutation: geometry and attributes travel together.
    assert np.allclose(pos2[new_of_old], pos)
    assert np.allclose(attrs2["radius"][new_of_old], attrs["radius"])
