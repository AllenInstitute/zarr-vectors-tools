"""SWC ingest/export tests, including header round-trip."""

from __future__ import annotations

from pathlib import Path

from zarr_vectors_tools.ingest.swc import ingest_swc
from zarr_vectors_tools.export.swc import export_swc


class TestSWC:

    def test_ingest(self, tmp_path: Path) -> None:
        swc = tmp_path / "n.swc"
        swc.write_text("# test\n1 1 0 0 0 5 -1\n2 3 10 0 0 3 1\n3 3 20 0 0 2 2\n4 3 15 10 0 2 2\n5 2 -10 0 0 2.5 1\n")
        s = ingest_swc(swc, tmp_path / "n.zv", (100.,100.,100.))
        assert s["node_count"] == 5 and s["is_tree"]

    def test_export(self, tmp_path: Path) -> None:
        swc = tmp_path / "n.swc"
        swc.write_text("# test\n1 1 0 0 0 5 -1\n2 3 10 0 0 3 1\n3 3 20 0 0 2 2\n")
        ingest_swc(swc, tmp_path / "n.zv", (100.,100.,100.))
        out = tmp_path / "out.swc"
        export_swc(tmp_path / "n.zv", out)
        lines = [l for l in out.read_text().strip().split("\n") if not l.startswith("#")]
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
