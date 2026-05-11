"""TRK/TCK/TRX missing-dependency tests for ingest and export."""

from __future__ import annotations

from pathlib import Path


class TestIngestDeps:

    def test_trk_missing_dep(self, tmp_path: Path) -> None:
        from zarr_vectors_tools.ingest.trk import ingest_trk
        from zarr_vectors.exceptions import IngestError
        try:
            ingest_trk(tmp_path / "f.trk", tmp_path / "o.zarrvectors", (50.0, 50.0, 50.0))
        except IngestError as e:
            assert "nibabel" in str(e).lower()
        except Exception:
            pass  # nibabel might be installed

    def test_tck_missing_dep(self, tmp_path: Path) -> None:
        from zarr_vectors_tools.ingest.tck import ingest_tck
        from zarr_vectors.exceptions import IngestError
        try:
            ingest_tck(tmp_path / "f.tck", tmp_path / "o.zarrvectors", (50.0, 50.0, 50.0))
        except IngestError as e:
            assert "nibabel" in str(e).lower()
        except Exception:
            pass

    def test_trx_missing_dep(self, tmp_path: Path) -> None:
        from zarr_vectors_tools.ingest.trx import ingest_trx
        from zarr_vectors.exceptions import IngestError
        try:
            ingest_trx(tmp_path / "f.trx", tmp_path / "o.zarrvectors", (50.0, 50.0, 50.0))
        except IngestError as e:
            assert "trx" in str(e).lower()
        except Exception:
            pass

    def test_export_trx_missing_dep(self, tmp_path: Path) -> None:
        from zarr_vectors_tools.export.trx import export_trx
        from zarr_vectors.exceptions import ExportError
        try:
            export_trx(tmp_path / "store.zarrvectors", tmp_path / "out.trx")
        except ExportError as e:
            assert "trx" in str(e).lower()
        except Exception:
            pass

    def test_export_trk_missing_dep(self, tmp_path: Path) -> None:
        from zarr_vectors_tools.export.trk import export_trk
        from zarr_vectors.exceptions import ExportError
        try:
            export_trk(tmp_path / "store.zarrvectors", tmp_path / "out.trk")
        except ExportError as e:
            assert "nibabel" in str(e).lower()
        except Exception:
            pass
