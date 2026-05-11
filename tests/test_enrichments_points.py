"""Tests for the optional CSV/point ingest enrichments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zarr_vectors.types.points import read_points
from zarr_vectors_tools.ingest._point_enrichments import (
    compute_per_object_vertex_count,
)
from zarr_vectors_tools.ingest.csv_points import ingest_csv


def _write_csv(path: Path, header: str, rows: list[list[float]]) -> None:
    lines = [header]
    for r in rows:
        lines.append(",".join(str(v) for v in r))
    path.write_text("\n".join(lines) + "\n")


class TestAutoDetectColumns:

    def test_xyz_intensity(self, tmp_path: Path) -> None:
        csv = tmp_path / "scan.csv"
        _write_csv(csv, "x,y,z,intensity",
                   [[1, 2, 3, 0.5], [4, 5, 6, 0.8]])
        result = ingest_csv(
            csv, tmp_path / "out.zv", (100., 100., 100.),
            auto_detect_columns=True,
        )
        assert result["vertex_count"] == 2

    def test_pos_axis_aliases(self, tmp_path: Path) -> None:
        csv = tmp_path / "scan.csv"
        _write_csv(csv, "pos_x,pos_y,pos_z,label",
                   [[1, 2, 3, 0], [4, 5, 6, 1]])
        result = ingest_csv(
            csv, tmp_path / "out.zv", (100., 100., 100.),
            auto_detect_columns=True,
        )
        assert result["vertex_count"] == 2


class TestDropNA:

    def test_drops_nan_rows(self, tmp_path: Path) -> None:
        csv = tmp_path / "nan.csv"
        csv.write_text("x,y,z\n1,2,3\nnan,5,6\n7,8,9\n")
        result = ingest_csv(
            csv, tmp_path / "out.zv", (100., 100., 100.),
            drop_na=True,
        )
        assert result["dropped_na"] == 1
        assert result["vertex_count"] == 2


class TestDropDuplicates:

    def test_drops_duplicate_positions(self, tmp_path: Path) -> None:
        csv = tmp_path / "dup.csv"
        _write_csv(csv, "x,y,z",
                   [[1, 2, 3], [1, 2, 3], [4, 5, 6]])
        result = ingest_csv(
            csv, tmp_path / "out.zv", (100., 100., 100.),
            drop_duplicates=True,
        )
        assert result["dropped_duplicates"] == 1
        assert result["vertex_count"] == 2


class TestNormalise:

    def test_centroid_and_scale_recorded(self, tmp_path: Path) -> None:
        csv = tmp_path / "norm.csv"
        _write_csv(csv, "x,y,z",
                   [[0, 0, 0], [10, 10, 10], [20, 20, 20]])
        store = tmp_path / "out.zv"
        ingest_csv(csv, store, (100., 100., 100.), normalise=True)

        r = read_points(str(store))
        # After normalisation positions should lie in [-1, 1].
        positions = r["positions"]
        assert positions.min() >= -1.0001
        assert positions.max() <= 1.0001

        # The offset and scale should be stored in the CSV header.
        from zarr_vectors_tools.headers.registry import HeaderRegistry
        reg = HeaderRegistry(str(store))
        assert reg.has("csv")
        hdr = reg.get("csv")
        np.testing.assert_allclose(hdr.normalise_offset, [10.0, 10.0, 10.0])
        assert hdr.normalise_scale == 10.0


class TestKnnDistance:
    """kNN tests are conditional on scipy being available."""

    def test_knn_distance_attribute(self, tmp_path: Path) -> None:
        pytest.importorskip("scipy")
        csv = tmp_path / "knn.csv"
        rng = np.random.default_rng(0)
        positions = rng.uniform(0, 100, size=(40, 3))
        _write_csv(csv, "x,y,z", positions.tolist())
        store = tmp_path / "out.zv"
        ingest_csv(csv, store, (200., 200., 200.), knn_distance_k=3)
        r = read_points(str(store), attribute_names=["knn_distance"])
        knn = r["attributes"]["knn_distance"]
        # Every distance is >= 0 and finite.
        assert (knn >= 0).all() and np.isfinite(knn).all()


class TestPerObjectCount:

    def test_helper_counts(self) -> None:
        ids = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
        unique, counts = compute_per_object_vertex_count(ids)
        np.testing.assert_array_equal(unique, [0, 1, 2])
        np.testing.assert_array_equal(counts, [3, 2, 1])

    def test_requires_object_ids(self, tmp_path: Path) -> None:
        csv = tmp_path / "p.csv"
        _write_csv(csv, "x,y,z", [[1, 2, 3]])
        from zarr_vectors.exceptions import IngestError
        with pytest.raises(IngestError):
            ingest_csv(
                csv, tmp_path / "out.zv", (100., 100., 100.),
                per_object_vertex_count=True,
            )


class TestDefaultsUnchanged:
    """Backward-compatibility check: existing call shape still works."""

    def test_minimal_call(self, tmp_path: Path) -> None:
        csv = tmp_path / "min.csv"
        _write_csv(csv, "x,y,z", [[1, 2, 3], [4, 5, 6]])
        result = ingest_csv(csv, tmp_path / "out.zv", (100., 100., 100.))
        assert result["vertex_count"] == 2
        # No enrichment counters appear in default summary.
        assert "dropped_na" not in result
        assert "dropped_duplicates" not in result
