"""SWC ingest/export tests, including header round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest
from zarr_vectors.building import get_resolution_level, open_store, read_object_attributes

from zarr_vectors_tools.convert.export.swc import export_swc
from zarr_vectors_tools.convert.ingest.swc import ingest_swc


class TestSWC:

    def test_ingest(self, tmp_path: Path) -> None:
        swc = tmp_path / "n.swc"
        swc.write_text(
            "# test\n1 1 0 0 0 5 -1\n2 3 10 0 0 3 1\n3 3 20 0 0 2 2\n"
            "4 3 15 10 0 2 2\n5 2 -10 0 0 2.5 1\n"
        )
        s = ingest_swc(swc, tmp_path / "n.zv", (100.,100.,100.))
        assert s["node_count"] == 5 and s["kind"] == "skeleton"

    def test_ingest_writes_real_per_object_cable_length(self, tmp_path: Path) -> None:
        """object_attributes/length must be the real summed edge length, not
        a stand-in -- open-items.md item 11: a different skeleton ingest
        path (coarsen_skeleton_level, fixed separately) once silently ranked
        "length" sparsity by fragment count because nothing recorded this.
        Two trees, in root order: tree A is a straight 0->10->20 (x-axis)
        line, length 20; tree B is a 3-4-5 right triangle path, length 7
        (one leg of 3, one of 4 -- not the hypotenuse).
        """
        swc = tmp_path / "forest.swc"
        swc.write_text(
            "1 1 0 0 0 5 -1\n2 3 10 0 0 3 1\n3 3 20 0 0 2 2\n"
            "4 1 100 0 0 5 -1\n5 3 103 0 0 3 4\n6 3 103 4 0 2 5\n"
        )
        store = tmp_path / "forest.zv"
        ingest_swc(swc, store, (1000.0, 1000.0, 1000.0))
        level0 = get_resolution_level(open_store(str(store), mode="r"), 0)
        length = read_object_attributes(level0, "length")
        assert length.tolist() == pytest.approx([20.0, 7.0])

    def test_ingest_cable_length_is_zero_for_an_isolated_node(self, tmp_path: Path) -> None:
        swc = tmp_path / "point.swc"
        swc.write_text("1 1 0 0 0 5 -1\n")
        store = tmp_path / "point.zv"
        ingest_swc(swc, store, (100.0, 100.0, 100.0))
        level0 = get_resolution_level(open_store(str(store), mode="r"), 0)
        assert read_object_attributes(level0, "length").tolist() == [0.0]

    def test_export(self, tmp_path: Path) -> None:
        swc = tmp_path / "n.swc"
        swc.write_text("# test\n1 1 0 0 0 5 -1\n2 3 10 0 0 3 1\n3 3 20 0 0 2 2\n")
        ingest_swc(swc, tmp_path / "n.zv", (100.,100.,100.))
        out = tmp_path / "out.swc"
        export_swc(tmp_path / "n.zv", out)
        lines = [line for line in out.read_text().strip().split("\n") if not line.startswith("#")]
        assert len(lines) == 3


class TestHeaderRoundTrip:
    """Ingest SWC → header preserved → export → header used."""

    def test_swc_header_roundtrip(self, tmp_path: Path) -> None:
        from zarr_vectors_tools.headers.registry import HeaderRegistry

        swc_in = tmp_path / "neuron.swc"
        lines = ["# ORIGINAL_SOURCE: test", "# CREATURE: mouse"]
        for i in range(1, 30):
            p = max(1, i - 1)
            lines.append(f"{i} 3 {i*5:.1f} {i*3:.1f} {i*2:.1f} 2.0 "
                         f"{p if i > 1 else -1}")
        swc_in.write_text("\n".join(lines))

        store = str(tmp_path / "neuron.zv")
        ingest_swc(swc_in, store, (200., 200., 200.))

        # Header preserved on-disk
        reg = HeaderRegistry(store)
        assert reg.has("swc")
        hdr = reg.get("swc")
        assert "# ORIGINAL_SOURCE: test" in hdr.comment_lines

        # Export and verify
        swc_out = tmp_path / "neuron_out.swc"
        export_swc(store, swc_out)
        assert swc_out.exists()

        # Registry sees the header after re-open
        reg2 = HeaderRegistry(store)
        assert "swc" in reg2.available_formats
