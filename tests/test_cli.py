"""Tests for the ``zvtools`` CLI (zarr_vectors_tools.cli)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from zarr_vectors.core.store import list_resolution_levels, open_store

from zarr_vectors_tools.cli import build_parser, main
from zarr_vectors_tools.cli._args import (
    build_factors,
    parse_float_list,
    parse_int_list,
    parse_num_chunks,
    parse_shape,
    resolve_format,
)
from zarr_vectors_tools.cli.convert import _maybe_overwrite

# ===================================================================
# Fixtures
# ===================================================================

def _write_csv(path: Path, n: int = 200, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, 400, size=(n, 3))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "z"])
        for p in pts:
            w.writerow([f"{v:.4f}" for v in p])
    return path


def _write_obj(path: Path) -> Path:
    path.write_text("v 0 0 0\nv 10 0 0\nv 0 10 0\nv 5 5 20\nf 1 2 3\nf 1 2 4\n")
    return path


def _write_swc(path: Path) -> Path:
    # id type x y z radius parent
    path.write_text(
        "1 1 0 0 0 1.0 -1\n"
        "2 3 10 0 0 1.0 1\n"
        "3 3 20 0 0 0.8 2\n"
        "4 3 20 10 0 0.6 3\n"
    )
    return path


def _write_trk(path: Path, n: int = 30, seed: int = 0) -> Path:
    """Write a minimal .trk fixture (needs nibabel; call importorskip first)."""
    from nibabel.streamlines import Tractogram
    from nibabel.streamlines.trk import TrkFile

    rng = np.random.default_rng(seed)
    streamlines = [rng.uniform(0, 100, size=(20, 3)).astype(np.float32) for _ in range(n)]
    tfile = TrkFile(Tractogram(streamlines=streamlines, affine_to_rasmm=np.eye(4)))
    # nibabel defaults dimensions to (1,1,1); trk_parallel derives the spatial
    # grid from the header bbox, so give it dims that cover the data extent.
    tfile.header["dimensions"] = np.array([100, 100, 100], dtype=np.int16)
    tfile.header["voxel_sizes"] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    tfile.save(str(path))
    return path


def _levels(store: Path) -> list[int]:
    return list_resolution_levels(open_store(str(store)))


# ===================================================================
# _args unit tests
# ===================================================================

class TestArgHelpers:
    def test_parse_float_list(self):
        assert parse_float_list("8,2,2") == [8.0, 2.0, 2.0]

    def test_parse_int_list(self):
        assert parse_int_list("2,2,2") == [2, 2, 2]

    def test_parse_shape(self):
        assert parse_shape("100,100,100") == (100.0, 100.0, 100.0)

    def test_parse_num_chunks_total(self):
        assert parse_num_chunks("5000") == 5000

    def test_parse_num_chunks_per_axis(self):
        assert parse_num_chunks("5,13,8") == (5, 13, 8)

    def test_build_factors_zip(self):
        assert build_factors([8.0, 2.0], [1.0, 4.0]) == [(8.0, 1.0), (2.0, 4.0)]

    def test_build_factors_none(self):
        assert build_factors(None, None) is None

    def test_build_factors_length_mismatch(self):
        with pytest.raises(SystemExit):
            build_factors([2.0, 2.0], [2.0])

    def test_resolve_format_from_extension(self):
        assert resolve_format("a.obj", None).name == "obj"
        assert resolve_format("a.trk", None).name == "trk"
        assert resolve_format("a.csv", None).name == "csv"
        assert resolve_format("a.LAS", None).name == "las"  # case-insensitive

    def test_resolve_format_explicit_override(self):
        assert resolve_format("a.csv", "lines").name == "lines"
        assert resolve_format("a.csv", "edgelist").name == "edgelist"

    def test_resolve_format_unknown_extension(self):
        with pytest.raises(SystemExit):
            resolve_format("a.bar", None)

    def test_parser_builds(self):
        build_parser()  # must not raise


# ===================================================================
# convert
# ===================================================================

class TestConvert:
    def test_csv_points_level0(self, tmp_path):
        src = _write_csv(tmp_path / "pts.csv")
        out = tmp_path / "pts.zv"
        assert main(["convert", str(src), str(out), "--chunk-shape", "100,100,100"]) == 0
        assert _levels(out) == [0]

    def test_csv_with_sparsity_pyramid(self, tmp_path):
        src = _write_csv(tmp_path / "pts.csv")
        out = tmp_path / "pts.zv"
        rc = main([
            "convert", str(src), str(out), "--chunk-shape", "100,100,100",
            "--coarsen", "2,2", "--sparsity", "2,4", "--chunk-scale", "2,2",
        ])
        assert rc == 0
        assert _levels(out) == [0, 1, 2]

    def test_obj_mesh_auto_detect(self, tmp_path):
        src = _write_obj(tmp_path / "tri.obj")
        out = tmp_path / "tri.zv"
        assert main(["convert", str(src), str(out), "--chunk-shape", "50,50,50"]) == 0
        assert 0 in _levels(out)

    def test_swc_skeleton(self, tmp_path):
        src = _write_swc(tmp_path / "n.swc")
        out = tmp_path / "n.zv"
        assert main(["convert", str(src), str(out), "--chunk-shape", "50,50,50"]) == 0
        assert 0 in _levels(out)

    def test_missing_chunk_shape_errors(self, tmp_path):
        src = _write_csv(tmp_path / "pts.csv")
        with pytest.raises(SystemExit):
            main(["convert", str(src), str(tmp_path / "o.zv")])

    def test_pyramid_length_mismatch_errors(self, tmp_path):
        src = _write_csv(tmp_path / "pts.csv")
        with pytest.raises(SystemExit):
            main([
                "convert", str(src), str(tmp_path / "o.zv"),
                "--chunk-shape", "100,100,100", "--coarsen", "2,2", "--sparsity", "2",
            ])

    def test_unknown_extension_errors(self, tmp_path):
        (tmp_path / "x.bar").write_text("nope")
        with pytest.raises(SystemExit):
            main(["convert", str(tmp_path / "x.bar"), str(tmp_path / "o.zv"),
                  "--chunk-shape", "1,1,1"])


# ===================================================================
# pyramid / validate / info
# ===================================================================

class TestUtilities:
    def _make_store(self, tmp_path) -> Path:
        src = _write_csv(tmp_path / "pts.csv")
        out = tmp_path / "pts.zv"
        main(["convert", str(src), str(out), "--chunk-shape", "100,100,100"])
        return out

    def test_pyramid_subcommand(self, tmp_path):
        out = self._make_store(tmp_path)
        rc = main(["pyramid", str(out), "--coarsen", "2,2", "--sparsity", "2,4",
                   "--chunk-scale", "2,2"])
        assert rc == 0
        assert _levels(out) == [0, 1, 2]

    def test_pyramid_requires_factors(self, tmp_path):
        out = self._make_store(tmp_path)
        with pytest.raises(SystemExit):
            main(["pyramid", str(out)])

    def test_validate(self, tmp_path):
        out = self._make_store(tmp_path)
        assert main(["validate", str(out)]) == 0

    def test_info(self, tmp_path, capsys):
        out = self._make_store(tmp_path)
        assert main(["info", str(out)]) == 0
        assert "resolution levels" in capsys.readouterr().out


# ===================================================================
# trk (needs a fixture writer; skip if nibabel absent)
# ===================================================================

class TestTrk:
    def test_trk_convert_serial_with_pyramid(self, tmp_path):
        pytest.importorskip("nibabel")
        trk = _write_trk(tmp_path / "s.trk")
        out = tmp_path / "s.zv"
        rc = main([
            "convert", str(trk), str(out), "--num-chunks", "27", "--workers", "1",
            "--coarsen", "2,2", "--sparsity", "1,2", "--chunk-scale", "2,2",
            "--coarsen-mode", "decimate", "--sparsity-strategy", "random",
        ])
        assert rc == 0
        assert _levels(out) == [0, 1, 2]

    def test_trk_length_strategy_auto_computes_length(self, tmp_path):
        # --sparsity-strategy length WITHOUT --compute-length must still work:
        # the CLI auto-enables length computation for streamlines.
        pytest.importorskip("nibabel")
        trk = _write_trk(tmp_path / "s.trk")
        out = tmp_path / "s.zv"
        rc = main([
            "convert", str(trk), str(out), "--num-chunks", "27", "--workers", "1",
            "--coarsen", "1,1", "--sparsity", "2,4", "--coarsen-mode", "decimate",
            "--sparsity-strategy", "length",
        ])
        assert rc == 0
        assert _levels(out) == [0, 1, 2]


# ===================================================================
# --overwrite
# ===================================================================

class TestOverwrite:
    def _convert(self, src, out, *extra):
        return main(["convert", str(src), str(out), "--chunk-shape", "100,100,100", *extra])

    def test_reconvert_with_overwrite_succeeds(self, tmp_path):
        # End-to-end: re-converting onto an existing store with --overwrite works.
        src = _write_csv(tmp_path / "pts.csv")
        out = tmp_path / "pts.zv"
        assert self._convert(src, out) == 0
        assert self._convert(src, out, "--overwrite") == 0
        assert _levels(out) == [0]

    def test_maybe_overwrite_removes_store_only_when_flagged(self, tmp_path):
        src = _write_csv(tmp_path / "pts.csv")
        out = tmp_path / "pts.zv"
        self._convert(src, out)
        assert out.exists()
        _maybe_overwrite(out, overwrite=False)   # no-op without the flag
        assert out.exists()
        _maybe_overwrite(out, overwrite=True)    # removes the store
        assert not out.exists()

    def test_overwrite_refuses_non_store_dir(self, tmp_path):
        out = tmp_path / "not_a_store"
        out.mkdir()
        (out / "important.txt").write_text("keep me")
        with pytest.raises(SystemExit):
            _maybe_overwrite(out, overwrite=True)
        assert (out / "important.txt").exists()  # guard prevented deletion
