"""Tests for the CSV → lines ingester."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from zarr_vectors.types.lines import read_lines
from zarr_vectors_tools.ingest.lines import ingest_lines_csv


def _write_lines_csv(path: Path, rows: list[list[float]]) -> None:
    header = "x0,y0,z0,x1,y1,z1"
    lines = [header] + [",".join(str(v) for v in r) for r in rows]
    path.write_text("\n".join(lines) + "\n")


class TestLinesCSV:

    def test_basic_ingest(self, tmp_path: Path) -> None:
        csv = tmp_path / "l.csv"
        _write_lines_csv(csv, [
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 2, 0],
        ])
        store = tmp_path / "l.zv"
        result = ingest_lines_csv(csv, store, (100., 100., 100.))
        assert result["line_count"] == 2

    def test_compute_length(self, tmp_path: Path) -> None:
        csv = tmp_path / "l.csv"
        _write_lines_csv(csv, [
            [0, 0, 0, 3, 4, 0],       # length 5
            [0, 0, 0, 1, 0, 0],       # length 1
        ])
        store = tmp_path / "l.zv"
        ingest_lines_csv(csv, store, (100., 100., 100.), compute_length=True)
        r = read_lines(str(store))
        # line_attributes round-trip through the store under "line_attributes"
        attrs = r.get("line_attributes") or r.get("object_attributes", {})
        if "length" in attrs:
            lengths = np.sort(np.asarray(attrs["length"], dtype=np.float32))
            np.testing.assert_allclose(lengths, [1.0, 5.0], atol=1e-5)

    def test_drop_zero_length(self, tmp_path: Path) -> None:
        csv = tmp_path / "l.csv"
        _write_lines_csv(csv, [
            [0, 0, 0, 0, 0, 0],       # zero-length
            [0, 0, 0, 1, 0, 0],
        ])
        store = tmp_path / "l.zv"
        result = ingest_lines_csv(
            csv, store, (100., 100., 100.), drop_zero_length=True,
        )
        assert result["dropped_zero_length"] == 1
        assert result["line_count"] == 1

    def test_drop_na(self, tmp_path: Path) -> None:
        csv = tmp_path / "l.csv"
        csv.write_text(
            "x0,y0,z0,x1,y1,z1\n"
            "0,0,0,1,0,0\n"
            "nan,0,0,1,0,0\n"
        )
        result = ingest_lines_csv(
            csv, tmp_path / "l.zv", (100., 100., 100.), drop_na=True,
        )
        assert result["dropped_na"] == 1
        assert result["line_count"] == 1

    def test_defaults_quiet(self, tmp_path: Path) -> None:
        csv = tmp_path / "l.csv"
        _write_lines_csv(csv, [[0, 0, 0, 1, 0, 0]])
        result = ingest_lines_csv(csv, tmp_path / "l.zv", (100., 100., 100.))
        assert "dropped_na" not in result
        assert "dropped_zero_length" not in result
