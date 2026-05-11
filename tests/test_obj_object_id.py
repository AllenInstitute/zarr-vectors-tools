"""Tests for OBJ ``auto_object_id`` parsing."""

from __future__ import annotations

from pathlib import Path

from zarr_vectors_tools.headers.registry import HeaderRegistry
from zarr_vectors_tools.ingest.obj import ingest_obj


class TestOBJAutoObjectID:

    def test_o_and_g_directives_assign_ids(self, tmp_path: Path) -> None:
        obj = tmp_path / "m.obj"
        # Two groups of 4 vertices each, each forming a triangle.
        obj.write_text(
            "o cell_a\n"
            "v 0 0 0\n"
            "v 1 0 0\n"
            "v 1 1 0\n"
            "v 0 1 0\n"
            "f 1 2 3\n"
            "o cell_b\n"
            "v 10 10 0\n"
            "v 11 10 0\n"
            "v 11 11 0\n"
            "v 10 11 0\n"
            "f 5 6 7\n"
        )
        store = tmp_path / "m.zv"
        ingest_obj(obj, store, (100., 100., 100.), auto_object_id=True)

        # Object names should land in OBJHeader.
        reg = HeaderRegistry(str(store))
        assert reg.has("obj")
        hdr = reg.get("obj")
        assert hdr.object_names == ["cell_a", "cell_b"]

    def test_default_off(self, tmp_path: Path) -> None:
        """Without the flag, OBJHeader is not written."""
        obj = tmp_path / "m.obj"
        obj.write_text(
            "o cell_a\n"
            "v 0 0 0\nv 1 0 0\nv 1 1 0\n"
            "f 1 2 3\n"
        )
        store = tmp_path / "m.zv"
        ingest_obj(obj, store, (100., 100., 100.))
        reg = HeaderRegistry(str(store))
        # No OBJ header because auto_object_id defaulted to False.
        assert not reg.has("obj")
