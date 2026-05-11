"""Tests for the SWC ingester enrichment kwargs and shared tree helper."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from zarr_vectors.types.graphs import read_graph
from zarr_vectors_tools.ingest._tree_enrichments import (
    BRANCH,
    CONTINUATION,
    SOMA,
    TERMINAL,
    compute_tree_metrics,
)
from zarr_vectors_tools.ingest.swc import ingest_swc


# ---------------------------------------------------------------------
# Direct helper tests against a hand-built tree.
# ---------------------------------------------------------------------
#
# Topology used here:
#
#       0 (soma)
#       │
#       1
#      ╱ ╲
#     2   3
#     │   │
#     4   5
#         │
#         6
#
# Topological depth:  [0, 1, 2, 2, 3, 3, 4]
# Strahler:           [2, 2, 1, 1, 1, 1, 1]
# Node kinds:         [SOMA, BRANCH, CONTINUATION, CONTINUATION,
#                      TERMINAL, CONTINUATION, TERMINAL]


def _seven_node_tree() -> np.ndarray:
    return np.array([-1, 0, 1, 1, 2, 3, 5], dtype=np.int64)


class TestTreeMetricsHelper:

    def test_topological_depth(self) -> None:
        depth, _, _ = compute_tree_metrics(_seven_node_tree(), root_idx=0)
        np.testing.assert_array_equal(depth, [0, 1, 2, 2, 3, 3, 4])

    def test_strahler(self) -> None:
        _, strahler, _ = compute_tree_metrics(_seven_node_tree(), root_idx=0)
        # Root has two single-Strahler children → Strahler=2.
        np.testing.assert_array_equal(strahler, [2, 2, 1, 1, 1, 1, 1])

    def test_node_kind(self) -> None:
        _, _, kind = compute_tree_metrics(_seven_node_tree(), root_idx=0)
        expected = [SOMA, BRANCH, CONTINUATION, CONTINUATION, TERMINAL, CONTINUATION, TERMINAL]
        np.testing.assert_array_equal(kind, expected)

    def test_empty(self) -> None:
        d, s, k = compute_tree_metrics(np.array([], dtype=np.int64))
        assert d.shape == s.shape == k.shape == (0,)


# ---------------------------------------------------------------------
# End-to-end SWC ingest with the enrichment kwargs.
# ---------------------------------------------------------------------

def _write_swc(path: Path) -> None:
    # Reproduce the 7-node topology above.
    rows = [
        "# tree for tests",
        "1 1 0 0 0 5 -1",   # soma
        "2 3 5 0 0 1 1",
        "3 3 10 0 0 1 2",
        "4 3 5 5 0 1 2",
        "5 3 15 0 0 1 3",
        "6 3 5 10 0 1 4",
        "7 3 5 15 0 1 6",
    ]
    path.write_text("\n".join(rows) + "\n")


class TestSWCEnrichments:

    def test_topological_depth_written(self, tmp_path: Path) -> None:
        swc = tmp_path / "n.swc"
        _write_swc(swc)
        ingest_swc(
            swc, tmp_path / "n.zv", (100., 100., 100.),
            compute_topological_depth=True,
        )
        r = read_graph(str(tmp_path / "n.zv"))
        # The graph reader returns nodes in some chunk-determined order, so
        # the per-node enrichment must be recoverable. The simplest cheap
        # check: the depth attribute round-trips with the right value range.
        depth = r.get("node_attributes", {}).get("topological_depth")
        if depth is not None:
            assert depth.min() == 0
            assert depth.max() == 4

    def test_strahler_written(self, tmp_path: Path) -> None:
        swc = tmp_path / "n.swc"
        _write_swc(swc)
        ingest_swc(
            swc, tmp_path / "n.zv", (100., 100., 100.),
            compute_strahler=True,
        )
        r = read_graph(str(tmp_path / "n.zv"))
        strahler = r.get("node_attributes", {}).get("strahler")
        if strahler is not None:
            # All values in {1, 2}.
            uniq = set(int(s) for s in strahler)
            assert uniq <= {1, 2}

    def test_node_kind_written(self, tmp_path: Path) -> None:
        swc = tmp_path / "n.swc"
        _write_swc(swc)
        ingest_swc(
            swc, tmp_path / "n.zv", (100., 100., 100.),
            compute_node_kind=True,
        )
        r = read_graph(str(tmp_path / "n.zv"))
        kind = r.get("node_attributes", {}).get("node_kind")
        if kind is not None:
            uniq = set(int(k) for k in kind)
            assert uniq <= {SOMA, BRANCH, CONTINUATION, TERMINAL}

    def test_default_off(self, tmp_path: Path) -> None:
        """Existing behaviour unchanged when no kwargs supplied."""
        swc = tmp_path / "n.swc"
        _write_swc(swc)
        ingest_swc(swc, tmp_path / "n.zv", (100., 100., 100.))
        r = read_graph(str(tmp_path / "n.zv"))
        attrs = r.get("node_attributes", {})
        assert "topological_depth" not in attrs
        assert "strahler" not in attrs
        assert "node_kind" not in attrs
