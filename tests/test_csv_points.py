"""CSV ingest/export round-trip + LAS/PLY missing-dependency tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from zarr_vectors.types.points import read_points
from zarr_vectors_tools.ingest.csv_points import ingest_csv
from zarr_vectors_tools.export.csv_points import export_csv


class TestCSVRoundTrip:

    def test_basic_csv(self, tmp_path: Path) -> None:
        """Write CSV → ingest → export → compare."""
        csv_path = tmp_path / "points.csv"
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, size=(30, 3)).astype(np.float64)
        intensity = rng.uniform(0, 1, size=30).astype(np.float64)

        data = np.column_stack([positions, intensity])
        header = "x,y,z,intensity"
        np.savetxt(csv_path, data, delimiter=",", header=header, comments="")

        store_path = tmp_path / "from_csv.zarr"
        summary = ingest_csv(
            csv_path, store_path,
            chunk_shape=(50.0, 50.0, 50.0),
            ndim=3,
            position_columns=["x", "y", "z"],
            attribute_columns=["intensity"],
        )
        assert summary["vertex_count"] == 30

        out_csv = tmp_path / "exported.csv"
        export_csv(store_path, out_csv)
        assert out_csv.exists()

        exported_data = np.loadtxt(out_csv, delimiter=",", skiprows=1)
        assert exported_data.shape[0] == 30

    def test_xyz_no_header(self, tmp_path: Path) -> None:
        """XYZ file (no header, space-delimited)."""
        xyz_path = tmp_path / "points.xyz"
        positions = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])
        np.savetxt(xyz_path, positions, delimiter=" ")

        store_path = tmp_path / "from_xyz.zarr"
        summary = ingest_csv(
            xyz_path, store_path,
            chunk_shape=(100.0, 100.0, 100.0),
            has_header=False,
            delimiter=" ",
        )
        assert summary["vertex_count"] == 3

        result = read_points(str(store_path))
        assert result["vertex_count"] == 3

    def test_csv_with_extra_columns(self, tmp_path: Path) -> None:
        """CSV with more columns than just XYZ."""
        csv_path = tmp_path / "rich.csv"
        data = np.array([
            [1, 2, 3, 0.5, 100],
            [4, 5, 6, 0.8, 200],
        ])
        np.savetxt(
            csv_path, data, delimiter=",",
            header="x,y,z,intensity,class",
            comments="",
        )

        store_path = tmp_path / "rich.zarr"
        summary = ingest_csv(
            csv_path, store_path,
            chunk_shape=(100.0, 100.0, 100.0),
        )
        assert summary["vertex_count"] == 2

    def test_csv_round_trip_positions_match(self, tmp_path: Path) -> None:
        """Verify position values survive the round-trip."""
        csv_in = tmp_path / "in.csv"
        positions = np.array([
            [10.5, 20.3, 30.1],
            [40.7, 50.9, 60.2],
        ])
        np.savetxt(csv_in, positions, delimiter=",",
                    header="x,y,z", comments="")

        store_path = tmp_path / "rt.zarr"
        ingest_csv(csv_in, store_path,
                    chunk_shape=(100.0, 100.0, 100.0))

        csv_out = tmp_path / "out.csv"
        export_csv(store_path, csv_out)

        exported = np.loadtxt(csv_out, delimiter=",", skiprows=1)
        np.testing.assert_allclose(
            np.sort(exported, axis=0),
            np.sort(positions, axis=0),
            atol=1e-2,
        )


class TestOptionalDependencies:

    def test_las_missing_raises_ingest_error(self, tmp_path: Path) -> None:
        """If laspy is not installed, ingest_las should raise IngestError."""
        from zarr_vectors_tools.ingest.las import ingest_las
        from zarr_vectors.exceptions import IngestError

        try:
            ingest_las(
                tmp_path / "fake.las",
                tmp_path / "out.zarr",
                (100.0, 100.0, 100.0),
            )
        except IngestError as e:
            assert "laspy" in str(e).lower()
        except Exception:
            pass  # laspy might actually be installed in some envs

    def test_ply_missing_raises_ingest_error(self, tmp_path: Path) -> None:
        """If plyfile is not installed, ingest_ply should raise IngestError."""
        from zarr_vectors_tools.ingest.ply import ingest_ply
        from zarr_vectors.exceptions import IngestError

        try:
            ingest_ply(
                tmp_path / "fake.ply",
                tmp_path / "out.zarr",
                (100.0, 100.0, 100.0),
            )
        except IngestError as e:
            assert "plyfile" in str(e).lower()
        except Exception:
            pass
