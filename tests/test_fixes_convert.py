"""Regression tests for the convert fixes: one class per fixed behaviour.

Each test reproduces what used to go wrong -- a CSV export headed
``dim0,dim1,dim2``, a headerless file losing its first point, a LAS GPS time
cut to float32, a TRX ``uint32`` coming back as float, a categorical column
the viewer showed as numbers, an interrupted parallel ingest that would not
stop -- and checks the fixed result.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import textwrap
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from zarr_vectors.exceptions import ExportError, IngestError
from zarr_vectors.types.points import read_points, write_points

from zarr_vectors_tools.cli import main
from zarr_vectors_tools.cli._args import FORMAT_REGISTRY, parse_delimiter
from zarr_vectors_tools.convert.export.csv_points import export_csv
from zarr_vectors_tools.convert.ingest.csv_points import ingest_csv


def _cli(capsys, *argv) -> tuple[int, str]:
    """Run ``zvtools`` in-process; ``(exit code, stdout + stderr)``."""
    try:
        rc = main([str(a) for a in argv])
        text = ""
    except SystemExit as exc:
        rc, text = (1, exc.code) if isinstance(exc.code, str) else (exc.code, "")
    out = capsys.readouterr()
    return rc, out.out + out.err + text


def _attr_meta(store: Path, name: str, level: int = 0) -> dict:
    path = store / str(level) / "vertex_attributes" / name / "zarr.json"
    return json.loads(path.read_text())["attributes"]


def _write_points_text(path: Path, points: np.ndarray, *, sep: str, header: str | None):
    lines = [] if header is None else [header]
    lines += [sep.join(f"{v:.4f}" for v in row) for row in points]
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def cloud() -> np.ndarray:
    return np.random.default_rng(0).uniform(0, 100, (500, 3))


# ---------------------------------------------------------------------------
# 1. CSV export names the position columns after the source
# ---------------------------------------------------------------------------

class TestPositionColumnNames:

    def test_a_csv_header_comes_back_on_export(self, tmp_path, cloud):
        source = tmp_path / "in.csv"
        _write_points_text(source, cloud, sep=",", header="X,Y,Z")
        ingest_csv(source, tmp_path / "s.zv", (50.0,) * 3)
        export_csv(tmp_path / "s.zv", tmp_path / "out.csv")
        assert (tmp_path / "out.csv").read_text().splitlines()[0] == "X,Y,Z"

    def test_a_store_without_names_exports_x_y_z(self, tmp_path, cloud):
        write_points(str(tmp_path / "s.zv"), cloud.astype(np.float32), chunk_shape=(50.0,) * 3)
        export_csv(tmp_path / "s.zv", tmp_path / "out.csv")
        assert (tmp_path / "out.csv").read_text().splitlines()[0] == "x,y,z"

    def test_a_table_exports_its_coordinate_column_names(self, tmp_path, cloud, capsys):
        source = tmp_path / "cells.csv"
        pd.DataFrame({"x_ccf": cloud[:, 0], "y_ccf": cloud[:, 1], "z_ccf": cloud[:, 2],
                      "volume": np.arange(len(cloud))}).to_csv(source, index=False)
        rc, out = _cli(capsys, "convert", source, tmp_path / "t.zv", "--format", "table",
                       "--position-columns", "x_ccf,y_ccf,z_ccf", "--chunk-shape", "50,50,50")
        assert rc == 0, out
        rc, out = _cli(capsys, "convert", tmp_path / "t.zv", tmp_path / "out.csv",
                       "--attribute", "volume")
        assert rc == 0, out
        header = (tmp_path / "out.csv").read_text().splitlines()[0]
        assert header == "x_ccf,y_ccf,z_ccf,volume"


# ---------------------------------------------------------------------------
# 2. Header detection, .xyz, and --delimiter for csv input
# ---------------------------------------------------------------------------

class TestHeaderAndDelimiter:

    def test_a_headerless_csv_keeps_its_first_point(self, tmp_path, cloud):
        source = tmp_path / "in.csv"
        _write_points_text(source, cloud, sep=",", header=None)
        summary = ingest_csv(source, tmp_path / "s.zv", (50.0,) * 3)
        assert summary["vertex_count"] == len(cloud)
        got = read_points(str(tmp_path / "s.zv"))["positions"]
        assert np.any(np.all(np.isclose(got, cloud[0], atol=1e-3), axis=1))

    def test_a_space_delimited_xyz_reads_without_flags(self, tmp_path, cloud, capsys):
        source = tmp_path / "in.xyz"
        _write_points_text(source, cloud, sep="  ", header=None)
        rc, out = _cli(capsys, "convert", source, tmp_path / "s.zv", "--chunk-shape", "50,50,50")
        assert rc == 0, out
        assert "vertex_count: 500" in out

    @pytest.mark.parametrize("flag", [" ", "whitespace", "tab"])
    def test_delimiter_is_accepted_for_csv_input(self, tmp_path, cloud, capsys, flag):
        source = tmp_path / "in.txt"
        sep = "\t" if flag == "tab" else " "
        _write_points_text(source, cloud, sep=sep, header=sep.join("xyz"))
        rc, out = _cli(capsys, "convert", source, tmp_path / "s.zv", "--format", "csv",
                       "--delimiter", flag, "--chunk-shape", "50,50,50")
        assert rc == 0, out
        assert "vertex_count: 500" in out

    def test_delimiter_is_still_refused_where_it_means_nothing(self, tmp_path, capsys):
        source = tmp_path / "in.obj"
        source.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
        rc, out = _cli(capsys, "convert", source, tmp_path / "s.zv", "--delimiter", ",",
                       "--chunk-shape", "5,5,5")
        assert rc != 0 and "--delimiter applies to csv/table input" in out

    def test_delimiter_spellings(self):
        assert parse_delimiter(" ") == "whitespace"
        assert parse_delimiter("Whitespace") == "whitespace"
        assert parse_delimiter("tab") == "\t"
        assert parse_delimiter("\\t") == "\t"
        assert parse_delimiter(";") == ";"

    def test_xyz_export_is_space_delimited_without_a_header(self, tmp_path, cloud):
        write_points(str(tmp_path / "s.zv"), cloud.astype(np.float32), chunk_shape=(50.0,) * 3)
        export_csv(tmp_path / "s.zv", tmp_path / "out.xyz")
        first = (tmp_path / "out.xyz").read_text().splitlines()[0]
        assert len(first.split(" ")) == 3
        float(first.split(" ")[0])  # a number, not a header


# ---------------------------------------------------------------------------
# 3. LAS GPS time keeps float64
# ---------------------------------------------------------------------------

def _fake_laspy(n: int) -> types.ModuleType:
    """Enough of ``laspy`` for the ingest: ``read`` returns a point record."""
    rng = np.random.default_rng(1)
    record = types.SimpleNamespace(
        x=rng.uniform(0, 100, n), y=rng.uniform(0, 100, n), z=rng.uniform(0, 100, n),
        intensity=rng.integers(0, 65535, n).astype(np.uint16),
        classification=rng.integers(0, 20, n).astype(np.uint8),
        red=rng.integers(0, 65535, n).astype(np.uint16),
        green=rng.integers(0, 65535, n).astype(np.uint16),
        blue=rng.integers(0, 65535, n).astype(np.uint16),
        gps_time=400000.0 + np.arange(n) * 0.001,
    )
    module = types.ModuleType("laspy")
    module.read = lambda path: record
    return module


class TestLasGpsTime:

    def test_every_distinct_time_survives(self, tmp_path, monkeypatch):
        from zarr_vectors_tools.convert.ingest.las import ingest_las

        monkeypatch.setitem(sys.modules, "laspy", _fake_laspy(5000))
        source = tmp_path / "p.las"
        source.write_bytes(b"")
        ingest_las(source, tmp_path / "s.zv", (50.0,) * 3)
        gps = read_points(str(tmp_path / "s.zv"), attribute_names=["gps_time"])[
            "vertex_attributes"]["gps_time"]
        assert gps.dtype == np.float64
        assert len(np.unique(gps)) == 5000  # float32 left 161

    def test_float32_positions_coarser_than_the_file_are_reported(self, tmp_path, monkeypatch):
        from zarr_vectors_tools.convert.ingest.las import ingest_las

        module = _fake_laspy(100)
        record = module.read(None)
        record.y = record.y + 4_000_000.0  # a UTM northing
        record.header = types.SimpleNamespace(scales=(0.01, 0.01, 0.01))
        monkeypatch.setitem(sys.modules, "laspy", module)
        (tmp_path / "p.las").write_bytes(b"")
        summary = ingest_las(tmp_path / "p.las", tmp_path / "s.zv", (100.0,) * 3)
        assert "float64" in summary["warnings"][0]
        summary = ingest_las(tmp_path / "p.las", tmp_path / "d.zv", (100.0,) * 3,
                             dtype="float64")
        assert "warnings" not in summary

    def test_ply_export_writes_it_as_double(self, tmp_path, monkeypatch):
        from zarr_vectors_tools.convert.export.ply import export_ply
        from zarr_vectors_tools.convert.ingest.las import ingest_las

        monkeypatch.setitem(sys.modules, "laspy", _fake_laspy(200))
        monkeypatch.setitem(sys.modules, "plyfile", types.ModuleType("plyfile"))
        (tmp_path / "p.las").write_bytes(b"")
        ingest_las(tmp_path / "p.las", tmp_path / "s.zv", (50.0,) * 3)
        export_ply(tmp_path / "s.zv", tmp_path / "o.ply",
                   attribute_names=["gps_time"], binary=False)
        text = (tmp_path / "o.ply").read_text().splitlines()
        assert "property double gps_time" in text
        body = np.loadtxt(text[text.index("end_header") + 1:])
        assert len(np.unique(body[:, 3])) == 200


# ---------------------------------------------------------------------------
# 4 and 5. TRX dtypes and widths; TRX/TCK units
# ---------------------------------------------------------------------------

def _write_trx(path: Path) -> np.ndarray:
    trx_memmap = pytest.importorskip("trx.trx_file_memmap")
    from nibabel.streamlines.array_sequence import ArraySequence

    rng = np.random.default_rng(3)
    streamlines = [rng.uniform(0, 40, (n, 3)).astype(np.float32) for n in (3, 5, 4, 6)]
    positions = np.concatenate(streamlines)
    lengths = np.array([len(s) for s in streamlines], np.uint32)
    offsets = np.concatenate([[0], np.cumsum(lengths[:-1])]).astype(np.uint32)
    trx = trx_memmap.TrxFile(nb_vertices=len(positions), nb_streamlines=len(streamlines))
    trx.header["VOXEL_TO_RASMM"] = np.eye(4).tolist()
    trx.header["DIMENSIONS"] = [50, 50, 50]
    trx.streamlines._data = positions
    trx.streamlines._offsets = offsets
    trx.streamlines._lengths = lengths
    label = ArraySequence()
    label._data = rng.integers(0, 70000, (len(positions), 1)).astype(np.uint32)
    label._offsets, label._lengths = offsets.astype(np.int64), lengths.astype(np.int64)
    trx.data_per_vertex = {"label": label}
    trx.data_per_streamline = {
        "vertex_count": lengths.reshape(-1, 1).copy(),
        "big_id": np.array([[16777217], [16777219], [3], [4]], np.int64),
        "weight": np.array([0.5, 1.5, 2.5, 3.5], np.float32),
    }
    trx_memmap.save(trx, str(path))
    trx.close()
    return lengths


def _axes(store: Path) -> list[dict]:
    return json.loads((store / "zarr.json").read_text())["attributes"]["multiscales"][0]["axes"]


class TestTrx:

    def test_dps_are_scalars_in_their_own_dtype(self, tmp_path):
        from zarr_vectors.building import get_resolution_level, open_store, read_object_attributes

        from zarr_vectors_tools.convert.ingest.trx import ingest_trx

        lengths = _write_trx(tmp_path / "s.trx")
        ingest_trx(tmp_path / "s.trx", tmp_path / "s.zv", (20.0,) * 3)
        level = get_resolution_level(open_store(str(tmp_path / "s.zv")), 0)
        counts = np.asarray(read_object_attributes(level, "vertex_count"))
        assert counts.dtype == np.uint32 and counts.shape == (4,)
        np.testing.assert_array_equal(counts, lengths)
        big = np.asarray(read_object_attributes(level, "big_id"))
        assert big.dtype == np.int64 and big[0] == 16777217  # float32 gives ...216
        assert _attr_meta(tmp_path / "s.zv", "label")["dtype"] == "uint32"

    def test_dtypes_round_trip_to_trx(self, tmp_path):
        trx_memmap = pytest.importorskip("trx.trx_file_memmap")
        from zarr_vectors_tools.convert.export.trx import export_trx
        from zarr_vectors_tools.convert.ingest.trx import ingest_trx

        _write_trx(tmp_path / "s.trx")
        ingest_trx(tmp_path / "s.trx", tmp_path / "s.zv", (20.0,) * 3)
        export_trx(tmp_path / "s.zv", tmp_path / "out.trx")
        back = trx_memmap.load(str(tmp_path / "out.trx"))
        assert back.data_per_streamline["vertex_count"].dtype == np.uint32
        assert back.data_per_streamline["big_id"].ravel()[0] == 16777217
        assert back.data_per_vertex["label"]._data.dtype == np.uint32
        back.close()

    def test_trx_axes_are_millimetres(self, tmp_path):
        from zarr_vectors_tools.convert.ingest.trx import ingest_trx

        _write_trx(tmp_path / "s.trx")
        ingest_trx(tmp_path / "s.trx", tmp_path / "s.zv", (20.0,) * 3)
        assert {axis.get("unit") for axis in _axes(tmp_path / "s.zv")} == {"millimeter"}
        ome = json.loads((tmp_path / "s.zv" / "zarr.json").read_text())["attributes"]["ome"]
        world = ome["attributes"]["scene"]["coordinateSystems"][0]["axes"]
        assert {axis.get("unit") for axis in world} == {"millimeter"}

    def test_tck_axes_are_millimetres(self, tmp_path):
        nib = pytest.importorskip("nibabel")
        from zarr_vectors_tools.convert.ingest.tck import ingest_tck

        lines = [np.random.default_rng(i).uniform(0, 30, (5, 3)).astype(np.float32)
                 for i in range(4)]
        nib.streamlines.save(nib.streamlines.Tractogram(lines, affine_to_rasmm=np.eye(4)),
                             str(tmp_path / "s.tck"))
        ingest_tck(tmp_path / "s.tck", tmp_path / "s.zv", (20.0,) * 3)
        assert {axis.get("unit") for axis in _axes(tmp_path / "s.zv")} == {"millimeter"}


# ---------------------------------------------------------------------------
# 6. Categorical columns are dictionary-encoded
# ---------------------------------------------------------------------------

def _write_h5ad(path: Path, n: int = 400):
    ad = pytest.importorskip("anndata")
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({
        "cell_type": pd.Categorical(rng.choice(["T", "B", "NK"], n), categories=["T", "B", "NK"]),
        "grade": pd.Categorical(rng.choice(["low", "high"], n), categories=["low", "high"],
                                ordered=True),
        "score": rng.random(n),
    }, index=[f"c{i}" for i in range(n)])
    obs.loc[obs.index[:3], "cell_type"] = np.nan
    adata = ad.AnnData(X=np.zeros((n, 1), np.float32), obs=obs)
    adata.obsm["spatial"] = rng.uniform(0, 1000, (n, 3))
    adata.write_h5ad(path)
    return obs


class TestCategoricals:

    def test_h5ad_categories_are_on_the_array(self, tmp_path):
        from zarr_vectors_tools.convert.ingest.h5ad import ingest_h5ad

        _write_h5ad(tmp_path / "c.h5ad")
        ingest_h5ad(tmp_path / "c.h5ad", tmp_path / "s.zv", (250.0,) * 3)
        meta = _attr_meta(tmp_path / "s.zv", "cell_type")
        # The keys the Neuroglancer viewer reads to label codes.
        assert meta["encoding"] == "dictionary"
        assert meta["categories"] == ["T", "B", "NK"]
        assert meta["_FillValue"] == -1
        assert _attr_meta(tmp_path / "s.zv", "grade")["ordered"] is True
        assert "encoding" not in _attr_meta(tmp_path / "s.zv", "score")

    def test_h5ad_export_restores_the_categoricals_exactly(self, tmp_path, capsys):
        ad = pytest.importorskip("anndata")
        obs = _write_h5ad(tmp_path / "c.h5ad")
        assert _cli(capsys, "convert", tmp_path / "c.h5ad", tmp_path / "s.zv",
                    "--chunk-shape", "250,250,250")[0] == 0
        assert _cli(capsys, "convert", tmp_path / "s.zv", tmp_path / "out.h5ad")[0] == 0
        back = ad.read_h5ad(tmp_path / "out.h5ad").obs
        for column in ("cell_type", "grade"):
            assert list(back[column].cat.categories) == list(obs[column].cat.categories)
            assert back[column].cat.ordered == obs[column].cat.ordered
            pd.testing.assert_series_equal(
                back[column].astype(object), obs[column].astype(object), check_names=False,
            )

    def test_table_text_columns_are_dictionary_encoded(self, tmp_path, cloud, capsys):
        source = tmp_path / "cells.csv"
        regions = np.where(np.arange(len(cloud)) % 2, "CTX", "HY, lateral")
        pd.DataFrame({"x": cloud[:, 0], "y": cloud[:, 1], "z": cloud[:, 2],
                      "region": regions}).to_csv(source, index=False)
        assert _cli(capsys, "convert", source, tmp_path / "t.zv", "--format", "table",
                    "--position-columns", "x,y,z", "--chunk-shape", "50,50,50")[0] == 0
        meta = _attr_meta(tmp_path / "t.zv", "region")
        assert meta["encoding"] == "dictionary"
        assert sorted(meta["categories"]) == ["CTX", "HY, lateral"]
        # A CSV export writes the labels, quoted where they hold the delimiter.
        assert _cli(capsys, "convert", tmp_path / "t.zv", tmp_path / "out.csv",
                    "--attribute", "region")[0] == 0
        back = pd.read_csv(tmp_path / "out.csv")
        assert set(back["region"]) == {"CTX", "HY, lateral"}


# ---------------------------------------------------------------------------
# 7. Draco is refused at ingest
# ---------------------------------------------------------------------------

class TestDraco:

    def test_obj_refuses_draco(self, tmp_path):
        from zarr_vectors_tools.convert.ingest.obj import ingest_obj

        source = tmp_path / "m.obj"
        source.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
        with pytest.raises(IngestError, match="cannot read them back"):
            ingest_obj(source, tmp_path / "s.zv", (5.0,) * 3, encoding="draco")
        assert not (tmp_path / "s.zv").exists()

    def test_stl_refuses_draco(self, tmp_path):
        from zarr_vectors_tools.convert.ingest.stl import ingest_stl

        source = tmp_path / "m.stl"
        source.write_text("solid t\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\n"
                          "vertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid t\n")
        with pytest.raises(IngestError, match="cannot read them back"):
            ingest_stl(source, tmp_path / "s.zv", (5.0,) * 3, encoding="draco")


# ---------------------------------------------------------------------------
# 8. --object-id on a store without object ids
# ---------------------------------------------------------------------------

class TestObjectIdsOnAStoreWithout:

    def test_the_error_says_what_is_missing(self, tmp_path, cloud, capsys):
        write_points(str(tmp_path / "s.zv"), cloud.astype(np.float32), chunk_shape=(50.0,) * 3)
        with pytest.raises(ExportError, match="no object ids"):
            export_csv(tmp_path / "s.zv", tmp_path / "out.csv", object_ids=[3])
        rc, out = _cli(capsys, "convert", tmp_path / "s.zv", tmp_path / "o.csv",
                       "--object-id", "3")
        assert rc != 0 and "no object ids" in out and "sid_ndim" not in out


# ---------------------------------------------------------------------------
# 9 and 10. Install hints name the extra; edgelist needs none
# ---------------------------------------------------------------------------

class TestInstallHints:

    @pytest.mark.parametrize("module, extra, call", [
        ("laspy", "las", "zarr_vectors_tools.convert.ingest.las:ingest_las"),
        ("plyfile", "ply", "zarr_vectors_tools.convert.ingest.ply:ingest_ply"),
        ("networkx", "graph", "zarr_vectors_tools.convert.ingest.graphml:ingest_graphml"),
        ("trx", "trx", "zarr_vectors_tools.convert.ingest.trx:ingest_trx"),
    ])
    def test_a_missing_reader_names_the_extra(self, tmp_path, monkeypatch, module, extra, call):
        import importlib

        monkeypatch.setitem(sys.modules, module, None)
        if module == "trx":
            monkeypatch.setitem(sys.modules, "trx.trx_file_memmap", None)
        mod_name, func = call.split(":")
        ingest = getattr(importlib.import_module(mod_name), func)
        with pytest.raises(IngestError, match=rf"zarr-vectors-tools\[{extra}\]"):
            ingest(tmp_path / "missing.dat", tmp_path / "s.zv", (5.0,) * 3)

    def test_edgelist_needs_no_extra(self):
        assert FORMAT_REGISTRY["edgelist"].extra is None

    def test_edgelist_metrics_name_the_graph_extra(self, tmp_path, monkeypatch):
        from zarr_vectors_tools.convert.ingest.edgelist import ingest_edgelist

        (tmp_path / "e.csv").write_text("source,target\n0,1\n")
        (tmp_path / "n.csv").write_text("node_id,x,y,z\n0,0,0,0\n1,1,1,1\n")
        monkeypatch.setitem(sys.modules, "networkx", None)
        with pytest.raises(IngestError, match=r"zarr-vectors-tools\[graph\]"):
            ingest_edgelist(tmp_path / "e.csv", tmp_path / "n.csv", tmp_path / "s.zv",
                            (5.0,) * 3, compute_degree=True)


# ---------------------------------------------------------------------------
# 11. TRK --num-chunks lands near the request
# ---------------------------------------------------------------------------

def _fanning_trk(path: Path, n: int = 300) -> None:
    """Streamlines that all start in a 10 mm cube and fan out ~80 mm.

    Sized from their first vertices alone, the extent is the cube's, and a
    64-chunk request became a grid of thousands of cells.
    """
    nib = pytest.importorskip("nibabel")
    rng = np.random.default_rng(1)
    lines = []
    for _ in range(n):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        steps = np.arange(1, 41)[:, None] * 2.0 * direction
        lines.append((rng.uniform(95, 105, 3) + steps).astype(np.float32))
    header = {
        nib.streamlines.Field.VOXEL_SIZES: (1.0, 1.0, 1.0),
        nib.streamlines.Field.DIMENSIONS: (200, 200, 200),
        nib.streamlines.Field.VOXEL_TO_RASMM: np.eye(4),
        nib.streamlines.Field.VOXEL_ORDER: "RAS",
    }
    nib.streamlines.save(nib.streamlines.Tractogram(lines, affine_to_rasmm=np.eye(4)),
                         str(path), header=header)


class TestTrkNumChunks:

    def test_the_scan_samples_first_middle_and_last(self, tmp_path):
        nib = pytest.importorskip("nibabel")
        from zarr_vectors_tools.convert.ingest.trk_parallel import (
            build_offset_index,
            parse_trk_header,
        )

        _fanning_trk(tmp_path / "f.trk", n=20)
        index = build_offset_index(tmp_path / "f.trk", parse_trk_header(tmp_path / "f.trk"))
        lines = nib.streamlines.load(str(tmp_path / "f.trk")).streamlines
        expected = np.stack([np.stack([s[0], s[len(s) // 2], s[-1]]) for s in lines])
        # On-disk voxmm is RAS + half a voxel for this identity header.
        np.testing.assert_allclose(index["sample_points"], expected + 0.5, atol=1e-4)

    def test_fitted_edges_give_the_requested_cells(self):
        from zarr_vectors_tools.convert.ingest.trk_parallel import _fit_chunk_shape

        bounds = ([-47.7, -49.6, -49.0], [151.1, 148.2, 150.4])
        edges = _fit_chunk_shape(bounds, 64)
        cells = [
            int(np.floor(hi / e) - np.floor(lo / e)) + 1
            for lo, hi, e in zip(bounds[0], bounds[1], edges)
        ]
        assert int(np.prod(cells)) == 64

    @pytest.mark.parametrize("requested", [64, 125])
    def test_the_grid_is_near_the_request(self, tmp_path, requested):
        from zarr_vectors.building import level_grid_layout

        from zarr_vectors_tools.convert.ingest.trk_parallel import ingest_trk_parallel

        _fanning_trk(tmp_path / "f.trk")
        out = tmp_path / "f.zv"
        ingest_trk_parallel(str(tmp_path / "f.trk"), str(out), num_chunks=requested,
                            workers=1, build_multiscale=False, progress=False)
        zv = json.loads((out / "zarr.json").read_text())["attributes"]["zarr_vectors"]
        _origin, grid = level_grid_layout(
            (zv["bounds"][0], zv["bounds"][1]), tuple(zv["chunk_shape"]),
        )
        total = int(np.prod(grid))
        assert requested / 2 <= total <= requested * 1.5, (grid, total)


# ---------------------------------------------------------------------------
# 12. Ctrl-C stops a parallel ingest promptly
# ---------------------------------------------------------------------------

_INTERRUPTED_INGEST = textwrap.dedent('''
    import sys, time
    import zarr_vectors_tools.convert.ingest.trk_parallel as tp
    from zarr_vectors_tools.convert.ingest._parallel import process_pool_executor

    real = tp._phase_a_worker

    def slow_phase_a(index, shared):
        time.sleep(4)  # a large part's binning
        return real(index, shared)

    tp._phase_a_worker = slow_phase_a
    try:
        with process_pool_executor(2) as ex:
            tp.ingest_trk_parallel(sys.argv[1], sys.argv[2], num_chunks=27, n_parts=8,
                                   workers=2, executor=ex, build_multiscale=False,
                                   progress=True)
    except KeyboardInterrupt:
        print("aborted", flush=True)
        sys.exit(130)
''')


@pytest.mark.slow
def test_sigint_stops_a_parallel_trk_ingest_promptly(tmp_path):
    """Eight 4-second parts on two workers: a shutdown that waited for the
    queue took ~12 s after the interrupt; the pool is now abandoned at once."""
    _fanning_trk(tmp_path / "f.trk", n=200)
    script = tmp_path / "ingest.py"
    script.write_text(_INTERRUPTED_INGEST)
    proc = subprocess.Popen(
        [sys.executable, "-u", str(script), str(tmp_path / "f.trk"), str(tmp_path / "f.zv")],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        for line in proc.stdout:
            if line.startswith("Phase A"):
                break
        time.sleep(1.0)
        proc.send_signal(signal.SIGINT)
        started = time.monotonic()
        output, _ = proc.communicate(timeout=30)
        elapsed = time.monotonic() - started
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    assert proc.returncode == 130, output
    assert elapsed < 4.0, f"took {elapsed:.1f} s to stop"


# ---------------------------------------------------------------------------
# 13. Exports add no headers (what the headers package now says)
# ---------------------------------------------------------------------------

def test_an_export_adds_no_header(tmp_path):
    pytest.importorskip("nibabel")
    from zarr_vectors_tools.convert.export.trk import export_trk
    from zarr_vectors_tools.convert.ingest.trx import ingest_trx
    from zarr_vectors_tools.headers.registry import HeaderRegistry

    _write_trx(tmp_path / "s.trx")
    ingest_trx(tmp_path / "s.trx", tmp_path / "s.zv", (20.0,) * 3)
    before = sorted(HeaderRegistry(str(tmp_path / "s.zv")).available_formats)
    export_trk(tmp_path / "s.zv", tmp_path / "out.trk")
    assert sorted(HeaderRegistry(str(tmp_path / "s.zv")).available_formats) == before


# ---------------------------------------------------------------------------
# 14. The lines ingest reports its line count
# ---------------------------------------------------------------------------

def test_lines_summary_reports_the_line_count(tmp_path, capsys):
    source = tmp_path / "l.csv"
    source.write_text("x0,y0,z0,x1,y1,z1\n0,0,0,1,1,1\n5,5,5,6,6,6\n2,2,2,3,3,3\n")
    rc, out = _cli(capsys, "convert", source, tmp_path / "l.zv", "--format", "lines",
                   "--chunk-shape", "4,4,4")
    assert rc == 0, out
    assert "line_count: 3" in out


# ---------------------------------------------------------------------------
# 15. OBJ/STL options on the CLI
# ---------------------------------------------------------------------------

_TWO_OBJECTS = (
    "o first\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
    "o second\nv 5 5 5\nv 6 5 5\nv 5 6 5\nf 4 5 6\n"
)
_TWO_FACETS = (
    "solid t\n"
    "facet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\n"
    "endloop\nendfacet\n"
    "facet normal 0 0 1\nouter loop\nvertex 1 0 0\nvertex 1 1 0\nvertex 0 1 0\n"
    "endloop\nendfacet\nendsolid t\n"
)


class TestMeshFlags:

    def test_split_objects_gives_one_object_per_group(self, tmp_path, capsys):
        source = tmp_path / "m.obj"
        source.write_text(_TWO_OBJECTS)
        rc, out = _cli(capsys, "convert", source, tmp_path / "s.zv", "--chunk-shape", "4,4,4",
                       "--split-objects")
        assert rc == 0, out
        assert "object_count: 2" in out

    @pytest.mark.parametrize("flags, vertices", [
        ((), 4), (("--no-merge-vertices",), 6), (("--merge-tolerance", "0"), 4),
    ])
    def test_stl_vertex_welding(self, tmp_path, capsys, flags, vertices):
        source = tmp_path / "m.stl"
        source.write_text(_TWO_FACETS)
        rc, out = _cli(capsys, "convert", source, tmp_path / "s.zv", "--chunk-shape", "4,4,4",
                       *flags)
        assert rc == 0, out
        assert f"vertex_count: {vertices}" in out

    @pytest.mark.parametrize("source, flag", [
        ("m.stl", "--split-objects"), ("m.obj", "--no-merge-vertices"),
    ])
    def test_a_flag_for_the_other_format_is_refused(self, tmp_path, capsys, source, flag):
        path = tmp_path / source
        path.write_text(_TWO_FACETS if source.endswith(".stl") else _TWO_OBJECTS)
        rc, out = _cli(capsys, "convert", path, tmp_path / "s.zv", "--chunk-shape", "4,4,4",
                       flag)
        assert rc != 0 and f"{flag} applies to" in out

    def test_tolerance_without_welding_is_refused(self, tmp_path, capsys):
        source = tmp_path / "m.stl"
        source.write_text(_TWO_FACETS)
        rc, out = _cli(capsys, "convert", source, tmp_path / "s.zv", "--chunk-shape", "4,4,4",
                       "--no-merge-vertices", "--merge-tolerance", "1")
        assert rc != 0 and "--merge-tolerance" in out


# ---------------------------------------------------------------------------
# Point exports get --bbox right on a store with object ids and small bins
# ---------------------------------------------------------------------------

@pytest.fixture
def quickstart_store(tmp_path, capsys) -> tuple[Path, np.ndarray]:
    """The quickstart's table: 20 clusters x 500 points, ids and 10-unit bins."""
    rng = np.random.default_rng(0)
    group = np.repeat(np.arange(20), 500)
    centre = rng.uniform(100, 900, size=(20, 3))[group]
    xyz = centre + rng.normal(0, 30, size=(group.size, 3))
    pd.DataFrame({"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2], "group": group,
                  "intensity": rng.random(group.size)}).to_csv(tmp_path / "cells.csv",
                                                                index=False)
    store = tmp_path / "cells.zv"
    rc, out = _cli(capsys, "convert", tmp_path / "cells.csv", store, "--format", "table",
                   "--position-columns", "x,y,z", "--object-id-column", "group",
                   "--chunk-shape", "250,250,250", "--bin-shape", "10,10,10")
    assert rc == 0, out
    return store, xyz.astype(np.float32)


class TestBboxOnAnIdsAndBinsStore:

    BOX = ([260.0] * 3, [490.0] * 3)

    def _truth(self, xyz):
        return int(np.all((xyz >= self.BOX[0]) & (xyz <= self.BOX[1]), axis=1).sum())

    def test_csv(self, quickstart_store, tmp_path, capsys):
        store, xyz = quickstart_store
        assert self._truth(xyz) == 46
        rc, out = _cli(capsys, "convert", store, tmp_path / "box.csv",
                       "--bbox", "260,260,260,490,490,490")
        assert rc == 0, out
        rows = np.loadtxt(tmp_path / "box.csv", delimiter=",", skiprows=1, ndmin=2)
        assert len(rows) == 46
        assert np.all((rows >= 260 - 1e-3) & (rows <= 490 + 1e-3))

    def test_csv_with_object_ids(self, quickstart_store, tmp_path):
        store, xyz = quickstart_store
        export_csv(store, tmp_path / "box.csv", bbox=self.BOX, object_ids=list(range(20)))
        rows = np.loadtxt(tmp_path / "box.csv", delimiter=",", skiprows=1, ndmin=2)
        assert len(rows) == 46

    def test_ply(self, quickstart_store, tmp_path, monkeypatch):
        from zarr_vectors_tools.convert.export.ply import export_ply

        monkeypatch.setitem(sys.modules, "plyfile", types.ModuleType("plyfile"))
        store, _xyz = quickstart_store
        summary = export_ply(store, tmp_path / "box.ply", bbox=self.BOX, binary=False)
        assert summary["vertex_count"] == 46

    def test_h5ad(self, quickstart_store, tmp_path):
        ad = pytest.importorskip("anndata")
        from zarr_vectors_tools.convert.export.h5ad import export_h5ad

        store, _xyz = quickstart_store
        export_h5ad(store, tmp_path / "box.h5ad", bbox=self.BOX)
        assert ad.read_h5ad(tmp_path / "box.h5ad").n_obs == 46


# ---------------------------------------------------------------------------
# --sparsity-strategy attribute where the pyramid cannot take it
# ---------------------------------------------------------------------------

class TestSparsityAttribute:

    def test_refused_for_trk(self, tmp_path, capsys):
        _fanning_trk(tmp_path / "f.trk", n=10)
        rc, out = _cli(capsys, "convert", tmp_path / "f.trk", tmp_path / "f.zv",
                       "--coarsen", "2", "--sparsity", "2",
                       "--sparsity-strategy", "attribute", "--sparsity-attribute", "length")
        assert rc != 0 and "zvtools pyramid" in out
        assert not (tmp_path / "f.zv").exists()

    def test_needs_a_pyramid(self, tmp_path, cloud, capsys):
        source = tmp_path / "in.csv"
        _write_points_text(source, cloud, sep=",", header="x,y,z")
        rc, out = _cli(capsys, "convert", source, tmp_path / "s.zv", "--chunk-shape", "50,50,50",
                       "--sparsity-strategy", "attribute", "--sparsity-attribute", "w")
        assert rc != 0 and "--coarsen" in out
