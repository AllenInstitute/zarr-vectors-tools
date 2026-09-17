"""Exporters must emit what the store holds.

Each of these produced a file that opened cleanly and was wrong, or no
file at all with the reason swallowed:

* ``export_trx`` called a method ``TrxFile`` does not have, so every call
  raised and the exception was re-wrapped as "Failed to write TRX".
* ``export_csv`` / ``export_ply`` read the reader's attributes under a key
  it does not use, so a requested column was dropped without a word.
* ``export_swc`` guessed parents from index order, which inverts any edge
  whose child is stored in an earlier chunk than its parent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from zarr_vectors.exceptions import ExportError
from zarr_vectors.types.graphs import write_graph
from zarr_vectors.types.points import write_points
from zarr_vectors.types.polylines import write_polylines

from zarr_vectors_tools.convert.export.csv_points import export_csv
from zarr_vectors_tools.convert.export.swc import export_swc

CHUNK = (10.0, 10.0, 10.0)


# =====================================================================
# Point-cloud exporters
# =====================================================================


def _point_store(path: Path) -> tuple[Path, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    positions = (rng.random((30, 3)) * 10.0).astype(np.float32)
    intensity = np.arange(30, dtype=np.float32)
    write_points(
        str(path), positions, chunk_shape=(5.0, 5.0, 5.0),
        vertex_attributes={"intensity": intensity},
    )
    return path, positions, intensity


class TestCsvExport:

    def test_requested_attribute_reaches_the_file(self, tmp_path: Path) -> None:
        store, _positions, intensity = _point_store(tmp_path / "p.zv")
        out = tmp_path / "p.csv"
        export_csv(str(store), str(out), attribute_names=["intensity"])

        lines = out.read_text().splitlines()
        assert lines[0] == "dim0,dim1,dim2,intensity"
        rows = np.array(
            [[float(v) for v in line.split(",")] for line in lines[1:]],
        )
        assert rows.shape == (30, 4)
        np.testing.assert_allclose(
            np.sort(rows[:, 3]), np.sort(intensity), atol=1e-6,
        )

    def test_missing_attribute_is_reported(self, tmp_path: Path) -> None:
        """Silently writing positions-only is the worst outcome here."""
        store, _p, _i = _point_store(tmp_path / "p.zv")
        with pytest.raises(ExportError, match="not present"):
            export_csv(
                str(store), str(tmp_path / "x.csv"), attribute_names=["nope"],
            )


class TestPlyExport:

    def test_requested_attribute_reaches_the_file(self, tmp_path: Path) -> None:
        plyfile = pytest.importorskip("plyfile")
        from zarr_vectors_tools.convert.export.ply import export_ply

        store, _positions, intensity = _point_store(tmp_path / "p.zv")
        out = tmp_path / "p.ply"
        export_ply(str(store), str(out), attribute_names=["intensity"])

        data = plyfile.PlyData.read(str(out))
        assert [p.name for p in data["vertex"].properties] == [
            "x", "y", "z", "intensity",
        ]
        np.testing.assert_allclose(
            np.sort(np.asarray(data["vertex"]["intensity"])),
            np.sort(intensity),
            atol=1e-6,
        )


# =====================================================================
# TRX
# =====================================================================


class TestTrxExport:

    def test_round_trips_exactly(self, tmp_path: Path) -> None:
        """Positions must survive bit-exact.

        ``TrxFile`` pre-allocates its buffer as float16, so writing into
        it quantises every coordinate; the buffer has to be replaced, not
        filled.
        """
        trx_memmap = pytest.importorskip("trx.trx_file_memmap")
        from zarr_vectors_tools.convert.export.trx import export_trx

        polylines = [
            np.array(
                [[0.1234567, 0.7654321, 1.5], [1.3, 1.7, 1.9], [2.11, 2.22, 2.33]],
                dtype=np.float32,
            ),
            np.array([[5.55, 5.25, 5.125], [6.0625, 6.5, 6.75]], dtype=np.float32),
        ]
        store = tmp_path / "s.zv"
        write_polylines(
            str(store), polylines, chunk_shape=CHUNK, geometry_type="streamline",
        )

        out = tmp_path / "out.trx"
        summary = export_trx(str(store), str(out))
        assert summary["streamline_count"] == 2
        assert summary["vertex_count"] == 5

        loaded = trx_memmap.load(str(out))
        try:
            got = [np.asarray(s).copy() for s in loaded.streamlines]
            dtype = loaded.streamlines._data.dtype
        finally:
            loaded.close()

        assert dtype == np.float32
        assert [g.shape for g in got] == [(3, 3), (2, 3)]
        for written, read_back in zip(polylines, got):
            np.testing.assert_array_equal(written, read_back)


# =====================================================================
# SWC
# =====================================================================


class TestSwcExport:

    def _reentrant_skeleton(self, path: Path) -> tuple[Path, np.ndarray, list]:
        """A neurite that leaves chunk 0, then comes back into it.

        Level order is chunk-major, so the last node is stored *before*
        its own parent.  "Smaller index is the parent" therefore inverts
        that edge and splits the tree into two roots.
        """
        positions = np.array([
            [1.0, 1.0, 1.0],    # 0  chunk (0,0,0)  root
            [12.0, 1.0, 1.0],   # 1  chunk (1,0,0)  child of 0
            [14.0, 1.0, 1.0],   # 2  chunk (1,0,0)  child of 1
            [5.0, 5.0, 1.0],    # 3  chunk (0,0,0)  child of 2  <- re-entry
        ], dtype=np.float32)
        edges = np.array([[1, 0], [2, 1], [3, 2]], dtype=np.int64)
        radius = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        compartment = np.array([1.0, 3.0, 3.0, 2.0], dtype=np.float32)
        write_graph(
            str(path), positions, edges, chunk_shape=CHUNK, kind="skeleton",
            vertex_attributes={"radius": radius, "compartment": compartment},
        )
        expected = [
            (tuple(positions[c]), tuple(positions[p])) for c, p in edges
        ]
        return path, positions, expected

    def _read_swc(self, path: Path) -> list[dict]:
        rows = []
        for line in path.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            rows.append({
                "id": int(parts[0]),
                "type": int(parts[1]),
                "xyz": (float(parts[2]), float(parts[3]), float(parts[4])),
                "radius": float(parts[5]),
                "parent": int(parts[6]),
            })
        return rows

    def test_edges_keep_their_stored_orientation(self, tmp_path: Path) -> None:
        store, _positions, expected = self._reentrant_skeleton(tmp_path / "s.zv")
        out = tmp_path / "out.swc"
        summary = export_swc(str(store), str(out))

        assert summary["node_count"] == 4
        assert summary["root_count"] == 1, "an inverted edge splits the tree"

        rows = self._read_swc(out)
        by_id = {row["id"]: row for row in rows}
        written = {
            (row["xyz"], by_id[row["parent"]]["xyz"])
            for row in rows if row["parent"] != -1
        }
        assert written == set(expected)

    def test_radius_and_compartment_come_from_the_store(
        self, tmp_path: Path,
    ) -> None:
        store, positions, _expected = self._reentrant_skeleton(tmp_path / "s.zv")
        out = tmp_path / "out.swc"
        summary = export_swc(str(store), str(out))
        assert set(summary["attributes_carried"]) == {"radius", "compartment"}

        expected_radius = dict(zip(
            [tuple(p) for p in positions], [1.0, 2.0, 3.0, 4.0],
        ))
        expected_type = dict(zip(
            [tuple(p) for p in positions], [1, 3, 3, 2],
        ))
        for row in self._read_swc(out):
            assert row["radius"] == pytest.approx(expected_radius[row["xyz"]])
            assert row["type"] == expected_type[row["xyz"]]

    def test_store_without_those_attributes_still_exports(
        self, tmp_path: Path,
    ) -> None:
        store = tmp_path / "bare.zv"
        positions = np.array(
            [[1.0, 1, 1], [2, 1, 1], [3, 1, 1]], dtype=np.float32,
        )
        write_graph(
            str(store), positions, np.array([[1, 0], [2, 1]], dtype=np.int64),
            chunk_shape=CHUNK, kind="skeleton",
        )
        out = tmp_path / "bare.swc"
        summary = export_swc(str(store), str(out))
        assert summary["attributes_carried"] == []
        rows = self._read_swc(out)
        assert [row["radius"] for row in rows] == [1.0, 1.0, 1.0]
        assert sum(1 for row in rows if row["parent"] == -1) == 1
