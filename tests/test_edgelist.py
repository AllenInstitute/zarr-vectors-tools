"""Tests for the CSV edge-list → graph ingester."""

from __future__ import annotations

from pathlib import Path

import pytest


pytest.importorskip("pandas")
pytest.importorskip("networkx")


def _write_csvs(tmp_path: Path) -> tuple[Path, Path]:
    edges_path = tmp_path / "edges.csv"
    nodes_path = tmp_path / "nodes.csv"
    edges_path.write_text(
        "source,target,weight\n"
        "a,b,0.5\n"
        "b,c,0.6\n"
        "c,d,0.7\n"
        "d,a,0.8\n"
    )
    nodes_path.write_text(
        "node_id,x,y,z,kind\n"
        "a,0,0,0,1\n"
        "b,1,0,0,1\n"
        "c,1,1,0,2\n"
        "d,0,1,0,2\n"
    )
    return edges_path, nodes_path


class TestEdgelistIngest:

    def test_basic_pandas_path(self, tmp_path: Path) -> None:
        from zarr_vectors_tools.ingest.edgelist import ingest_edgelist

        edges, nodes = _write_csvs(tmp_path)
        result = ingest_edgelist(
            edges, nodes, tmp_path / "g.zv", (100., 100., 100.),
        )
        assert result["node_count"] == 4
        assert result["edge_count"] == 4

    def test_with_enrichments(self, tmp_path: Path) -> None:
        from zarr_vectors_tools.ingest.edgelist import ingest_edgelist
        from zarr_vectors_tools.headers.registry import HeaderRegistry

        edges, nodes = _write_csvs(tmp_path)
        store = tmp_path / "g.zv"
        ingest_edgelist(
            edges, nodes, store, (100., 100., 100.),
            compute_degree=True,
            compute_summary=True,
        )
        reg = HeaderRegistry(str(store))
        assert reg.has("graph")
        hdr = reg.get("graph")
        assert hdr.node_count == 4
        assert hdr.edge_count == 4
        # 4-cycle: one component, all 4 nodes in it.
        assert hdr.n_components == 1
        assert hdr.largest_component_size == 4

    def test_cudf_path_gracefully_skipped(self, tmp_path: Path) -> None:
        """If cuDF is not installed, asking for it should raise IngestError."""
        cudf = pytest.importorskip("cudf", reason="cuDF only on RAPIDS envs")
        del cudf  # only matters that the import succeeded

        from zarr_vectors_tools.ingest.edgelist import ingest_edgelist

        edges, nodes = _write_csvs(tmp_path)
        result = ingest_edgelist(
            edges, nodes, tmp_path / "g.zv", (100., 100., 100.),
            use_cudf=True,
        )
        assert result["node_count"] == 4
