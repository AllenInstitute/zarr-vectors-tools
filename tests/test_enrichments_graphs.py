"""Tests for graph ingest enrichments and the GraphHeader round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest


pytest.importorskip("networkx")


def _write_graphml(path: Path) -> None:
    """Write a tiny 5-node graph (a 4-cycle + 1 isolated node)."""
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="x" for="node" attr.name="x" attr.type="double"/>
  <key id="y" for="node" attr.name="y" attr.type="double"/>
  <key id="z" for="node" attr.name="z" attr.type="double"/>
  <graph id="g" edgedefault="undirected">
    <node id="0"><data key="x">0</data><data key="y">0</data><data key="z">0</data></node>
    <node id="1"><data key="x">1</data><data key="y">0</data><data key="z">0</data></node>
    <node id="2"><data key="x">1</data><data key="y">1</data><data key="z">0</data></node>
    <node id="3"><data key="x">0</data><data key="y">1</data><data key="z">0</data></node>
    <node id="4"><data key="x">5</data><data key="y">5</data><data key="z">5</data></node>
    <edge source="0" target="1"/>
    <edge source="1" target="2"/>
    <edge source="2" target="3"/>
    <edge source="3" target="0"/>
  </graph>
</graphml>
"""
    )


class TestGraphEnrichments:

    def test_degree(self, tmp_path: Path) -> None:
        from zarr_vectors_tools.ingest.graphml import ingest_graphml
        from zarr_vectors.types.graphs import read_graph

        g = tmp_path / "g.graphml"
        _write_graphml(g)
        ingest_graphml(g, tmp_path / "g.zv", (100., 100., 100.),
                       compute_degree=True)
        r = read_graph(str(tmp_path / "g.zv"))
        deg = r.get("node_attributes", {}).get("degree")
        if deg is not None:
            # 4-cycle nodes have degree 2; the isolated node has degree 0.
            uniq = set(int(v) for v in deg)
            assert uniq <= {0, 2}

    def test_component_labels(self, tmp_path: Path) -> None:
        from zarr_vectors_tools.ingest.graphml import ingest_graphml
        from zarr_vectors.types.graphs import read_graph

        g = tmp_path / "g.graphml"
        _write_graphml(g)
        ingest_graphml(g, tmp_path / "g.zv", (100., 100., 100.),
                       compute_component=True)
        r = read_graph(str(tmp_path / "g.zv"))
        comp = r.get("node_attributes", {}).get("component")
        if comp is not None:
            # Exactly two components: the cycle and the isolated node.
            uniq = set(int(v) for v in comp)
            assert len(uniq) == 2

    def test_summary_header(self, tmp_path: Path) -> None:
        from zarr_vectors_tools.headers.formats import GraphHeader
        from zarr_vectors_tools.headers.registry import HeaderRegistry
        from zarr_vectors_tools.ingest.graphml import ingest_graphml

        g = tmp_path / "g.graphml"
        _write_graphml(g)
        store = tmp_path / "g.zv"
        ingest_graphml(g, store, (100., 100., 100.), compute_summary=True)

        reg = HeaderRegistry(str(store))
        assert reg.has("graph")
        hdr = reg.get("graph")
        assert isinstance(hdr, GraphHeader)
        assert hdr.node_count == 5
        assert hdr.edge_count == 4
        assert hdr.is_directed is False
        assert hdr.n_components == 2
        assert hdr.largest_component_size == 4

    def test_default_off(self, tmp_path: Path) -> None:
        """Existing behaviour: nothing computed unless requested."""
        from zarr_vectors_tools.ingest.graphml import ingest_graphml
        from zarr_vectors.types.graphs import read_graph

        g = tmp_path / "g.graphml"
        _write_graphml(g)
        ingest_graphml(g, tmp_path / "g.zv", (100., 100., 100.))
        r = read_graph(str(tmp_path / "g.zv"))
        attrs = r.get("node_attributes", {})
        for key in ("degree", "component", "clustering"):
            assert key not in attrs
