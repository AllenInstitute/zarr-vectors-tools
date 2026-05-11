"""OBJ/STL ingest and OBJ export tests."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from zarr_vectors.exceptions import IngestError
from zarr_vectors_tools.ingest.obj import ingest_obj
from zarr_vectors_tools.ingest.stl import ingest_stl
from zarr_vectors_tools.export.obj import export_obj


class TestOBJIngest:

    def test_triangle_obj(self, tmp_path: Path) -> None:
        obj = tmp_path / "t.obj"
        obj.write_text("v 0 0 0\nv 10 0 0\nv 5 10 0\nv 5 5 10\nf 1 2 3\nf 1 2 4\nf 2 3 4\nf 1 3 4\n")
        s = ingest_obj(obj, tmp_path / "m.zv", (100.,100.,100.))
        assert s["vertex_count"] == 4 and s["face_count"] == 4

    def test_quad_obj(self, tmp_path: Path) -> None:
        obj = tmp_path / "q.obj"
        obj.write_text("v 0 0 0\nv 10 0 0\nv 10 10 0\nv 0 10 0\nf 1 2 3 4\n")
        s = ingest_obj(obj, tmp_path / "m.zv", (100.,100.,100.))
        assert s["face_count"] == 1

    def test_polygon_fan(self, tmp_path: Path) -> None:
        obj = tmp_path / "p.obj"
        obj.write_text("v 0 0 0\nv 10 0 0\nv 10 10 0\nv 5 15 0\nv 0 10 0\nf 1 2 3 4 5\n")
        s = ingest_obj(obj, tmp_path / "m.zv", (100.,100.,100.))
        assert s["face_count"] == 3

    def test_not_found(self, tmp_path: Path) -> None:
        try:
            ingest_obj(tmp_path / "x.obj", tmp_path / "x.zv", (100.,100.,100.))
            assert False
        except IngestError:
            pass


class TestSTLIngest:

    def test_ascii(self, tmp_path: Path) -> None:
        stl = tmp_path / "a.stl"
        stl.write_text(
            "solid t\n"
            "  facet normal 0 0 1\n    outer loop\n"
            "      vertex 0 0 0\n      vertex 10 0 0\n      vertex 5 10 0\n"
            "    endloop\n  endfacet\n"
            "  facet normal 0 0 -1\n    outer loop\n"
            "      vertex 0 0 0\n      vertex 10 0 0\n      vertex 5 5 10\n"
            "    endloop\n  endfacet\n"
            "endsolid t\n"
        )
        s = ingest_stl(stl, tmp_path / "m.zv", (100.,100.,100.))
        assert s["face_count"] == 2

    def test_binary(self, tmp_path: Path) -> None:
        stl = tmp_path / "b.stl"
        with open(stl, "wb") as f:
            f.write(b"\x00" * 80)
            f.write(struct.pack("<I", 1))
            f.write(struct.pack("<3f", 0, 0, 1))
            for v in [(0,0,0),(10,0,0),(5,10,0)]:
                f.write(struct.pack("<3f", *v))
            f.write(struct.pack("<H", 0))
        s = ingest_stl(stl, tmp_path / "m.zv", (100.,100.,100.))
        assert s["face_count"] == 1

    def test_not_found(self, tmp_path: Path) -> None:
        try:
            ingest_stl(tmp_path / "x.stl", tmp_path / "x.zv", (100.,100.,100.))
            assert False
        except IngestError:
            pass


class TestOBJExport:

    def test_export(self, tmp_path: Path) -> None:
        obj = tmp_path / "t.obj"
        obj.write_text("v 0 0 0\nv 10 0 0\nv 5 10 0\nv 5 5 10\nf 1 2 3\nf 1 2 4\nf 2 3 4\nf 1 3 4\n")
        ingest_obj(obj, tmp_path / "m.zv", (100.,100.,100.))
        out = tmp_path / "out.obj"
        export_obj(tmp_path / "m.zv", out)
        lines = out.read_text().strip().split("\n")
        assert len([l for l in lines if l.startswith("v ")]) == 4
        assert len([l for l in lines if l.startswith("f ")]) == 4

    def test_round_trip(self, tmp_path: Path) -> None:
        obj = tmp_path / "rt.obj"
        obj.write_text("v 1.5 2.5 3.5\nv 4.5 5.5 6.5\nv 7.5 8.5 9.5\nf 1 2 3\n")
        ingest_obj(obj, tmp_path / "m.zv", (100.,100.,100.))
        out = tmp_path / "out.obj"
        export_obj(tmp_path / "m.zv", out)
        rv = []
        for line in out.read_text().split("\n"):
            if line.startswith("v "):
                parts = line.split()
                rv.append([float(parts[1]), float(parts[2]), float(parts[3])])
        expected = [[1.5,2.5,3.5],[4.5,5.5,6.5],[7.5,8.5,9.5]]
        np.testing.assert_allclose(np.sort(rv, axis=0), np.sort(expected, axis=0), atol=1e-3)
