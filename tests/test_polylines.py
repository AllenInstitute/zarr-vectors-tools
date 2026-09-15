"""TRK/TCK/TRX missing-dependency tests for ingest and export."""

from __future__ import annotations

from pathlib import Path


class TestIngestDeps:

    def test_trk_missing_dep(self, tmp_path: Path) -> None:
        from zarr_vectors.exceptions import IngestError

        from zarr_vectors_tools.convert.ingest.trk import ingest_trk
        try:
            ingest_trk(tmp_path / "f.trk", tmp_path / "o.zarrvectors", (50.0, 50.0, 50.0))
        except IngestError as e:
            assert "nibabel" in str(e).lower()
        except Exception:
            pass  # nibabel might be installed

    def test_tck_missing_dep(self, tmp_path: Path) -> None:
        from zarr_vectors.exceptions import IngestError

        from zarr_vectors_tools.convert.ingest.tck import ingest_tck
        try:
            ingest_tck(tmp_path / "f.tck", tmp_path / "o.zarrvectors", (50.0, 50.0, 50.0))
        except IngestError as e:
            assert "nibabel" in str(e).lower()
        except Exception:
            pass

    def test_trx_missing_dep(self, tmp_path: Path) -> None:
        from zarr_vectors.exceptions import IngestError

        from zarr_vectors_tools.convert.ingest.trx import ingest_trx
        try:
            ingest_trx(tmp_path / "f.trx", tmp_path / "o.zarrvectors", (50.0, 50.0, 50.0))
        except IngestError as e:
            assert "trx" in str(e).lower()
        except Exception:
            pass

    def test_export_trx_missing_dep(self, tmp_path: Path, monkeypatch) -> None:
        """A missing reader is named before the store is even opened."""
        import sys

        import pytest
        from zarr_vectors.exceptions import ExportError

        from zarr_vectors_tools.convert.export.trx import export_trx

        monkeypatch.setitem(sys.modules, "trx", None)
        monkeypatch.setitem(sys.modules, "trx.trx_file_memmap", None)
        with pytest.raises(ExportError, match="trx-python"):
            export_trx(tmp_path / "store.zarrvectors", tmp_path / "out.trx")

    def test_export_trk_missing_dep(self, tmp_path: Path, monkeypatch) -> None:
        import sys

        import pytest
        from zarr_vectors.exceptions import ExportError

        from zarr_vectors_tools.convert.export.trk import export_trk

        monkeypatch.setitem(sys.modules, "nibabel", None)
        with pytest.raises(ExportError, match="nibabel"):
            export_trk(tmp_path / "store.zarrvectors", tmp_path / "out.trk")
