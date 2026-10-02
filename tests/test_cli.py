"""Tests for the ``zvtools`` CLI (zarr_vectors_tools.cli)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from zarr_vectors.building import list_resolution_levels, open_store

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


def _write_smooth_trk(path: Path, n: int = 200, npts: int = 60, seed: int = 0) -> Path:
    """A .trk of SMOOTH streamlines — the shape real tractography has.

    ``_write_trk`` emits uniform-random points, which carry ~full entropy and
    do not compress.  Real streamlines advance ~1 voxel per step, so
    consecutive coordinates are highly correlated; use this whenever a test
    depends on compressibility rather than just structure.
    """
    from nibabel.streamlines import Tractogram
    from nibabel.streamlines.trk import TrkFile

    rng = np.random.default_rng(seed)
    streamlines = []
    for _ in range(n):
        start = rng.uniform(10, 90, size=3)
        steps = rng.normal(0, 1.0, size=(npts - 1, 3))
        pos = np.concatenate([[start], start + np.cumsum(steps, axis=0)])
        streamlines.append(np.clip(pos, 0.5, 99.5).astype(np.float32))
    tfile = TrkFile(Tractogram(streamlines=streamlines, affine_to_rasmm=np.eye(4)))
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

    def test_build_factors_one_side_defaults_to_one(self):
        assert build_factors([2.0, 2.0], None) == [(2.0, 1.0), (2.0, 1.0)]
        assert build_factors(None, [4.0]) == [(1.0, 4.0)]

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

# Every test here runs a full TRK ingest; together they are the largest
# single block of suite time.
@pytest.mark.slow
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

    def test_trk_synthetic_attributes_all_levels(self, tmp_path):
        """--object-attr / --vertex-attr generate colorable test attributes,
        present and correct at level 0 AND every coarser pyramid level."""
        pytest.importorskip("nibabel")
        from zarr_vectors.building import (
            VERTICES,
            get_resolution_level,
            list_chunk_keys,
            read_chunk_attributes,
            read_chunk_vertices,
            read_object_attributes,
        )

        trk = _write_smooth_trk(tmp_path / "s.trk", n=200, npts=60)
        out = tmp_path / "s.zv"
        rc = main([
            "convert", str(trk), str(out), "--num-chunks", "27", "--workers", "1",
            "--coarsen", "2,2", "--sparsity", "1,2", "--sparsity-strategy", "random",
            "--object-attr", "orientation", "--object-attr", "tortuosity",
            "--object-attr", "vertex_count",
            "--vertex-attr", "arc_length", "--vertex-attr", "z",
            "--vertex-attr", "tangent", "--attr-seed", "5",
        ])
        assert rc == 0
        levels = _levels(out)
        assert levels == [0, 1, 2]

        root = open_store(str(out), mode="r+")
        for li in levels:
            lvl = get_resolution_level(root, li)
            # object attributes carried to every level
            orient = np.asarray(read_object_attributes(lvl, "orientation"))
            tort = np.asarray(read_object_attributes(lvl, "tortuosity"))
            vcount = np.asarray(read_object_attributes(lvl, "vertex_count"))
            assert orient.shape == (200, 3)
            assert vcount.dtype == np.uint32
            fin = np.isfinite(tort)
            assert np.all(tort[fin] >= 1.0 - 1e-4)
            norms = np.linalg.norm(orient, axis=1)
            nz = norms > 1e-6
            np.testing.assert_allclose(norms[nz], 1.0, atol=1e-4)

            # vertex attributes present, aligned, and in range at every level
            arc_vals, tan_ok = [], True
            for c in list_chunk_keys(lvl, VERTICES):
                cc = tuple(int(x) for x in c)
                vg = read_chunk_vertices(lvl, cc, dtype=np.float32, ndim=3)
                a_arc = read_chunk_attributes(lvl, "arc_length", cc, dtype=np.float32)
                a_z = read_chunk_attributes(lvl, "z", cc, dtype=np.float32)
                a_tan = read_chunk_attributes(lvl, "tangent", cc, dtype=np.float32, ncols=3)
                for f in range(len(vg)):
                    v = np.asarray(vg[f])
                    assert len(a_arc[f]) == len(v)
                    arc_vals.append(np.asarray(a_arc[f]).ravel())
                    # z attribute equals the vertex z coordinate at every level
                    np.testing.assert_allclose(
                        np.asarray(a_z[f]).ravel(), v[:, 2], atol=1e-3)
                    tn = np.linalg.norm(np.asarray(a_tan[f]).reshape(len(v), 3), axis=1)
                    tan_ok = tan_ok and bool(np.all((np.abs(tn - 1.0) < 1e-3) | (tn < 1e-6)))
            arc = np.concatenate(arc_vals)
            assert 0.0 <= arc.min() and arc.max() <= 1.0 + 1e-5
            assert tan_ok

    def test_vertex_attr_rejected_for_non_trk(self, tmp_path):
        src = _write_csv(tmp_path / "p.csv")
        with pytest.raises(SystemExit, match="only apply to trk"):
            main(["convert", str(src), str(tmp_path / "o.zv"),
                  "--chunk-shape", "100,100,100", "--vertex-attr", "z"])

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

    def test_trk_compressor_reaches_the_vertices_array(self, tmp_path):
        """--compressor must land on `vertices`, which is the whole point.

        create_store warm-creates vertices/vertex_fragments, and a chunk
        array's codec pipeline is fixed at creation — so a compressor applied
        only to the ingest's own create_* calls silently skips the largest
        array in the store (every later create_vertices_array short-circuits
        on the existing one).  Assert the codec is actually ON vertices, not
        merely that the ingest accepted the flag.

        Uses SMOOTH streamlines: `_write_trk` emits uniform-random points,
        which are near-incompressible, and at 600 vertices per-chunk framing
        overhead makes zstd *larger*.  Real tract coordinates advance ~1 voxel
        per step, so they compress ~2.4x — this fixture reproduces that shape
        so the size assertion means something.
        """
        pytest.importorskip("nibabel")
        sizes = {}
        for comp in ("none", "zstd"):
            trk = _write_smooth_trk(tmp_path / f"{comp}.trk")
            out = tmp_path / f"{comp}.zv"
            rc = main([
                "convert", str(trk), str(out), "--num-chunks", "27",
                "--workers", "1", "--compressor", comp,
            ])
            assert rc == 0
            meta = json.loads((out / "0" / "vertices" / "zarr.json").read_text())
            names = [c.get("name") for c in meta["codecs"]]
            if comp == "none":
                assert names == ["vlen-bytes"], names
            else:
                assert "zstd" in names, names
            sizes[comp] = sum(
                f.stat().st_size for f in (out / "0" / "vertices").rglob("*")
                if f.is_file()
            )

        # And it must actually shrink the payload, not just relabel it.
        assert sizes["zstd"] < sizes["none"], sizes

    def test_trk_compressor_roundtrips_identically(self, tmp_path):
        """Compression must be lossless: same polylines, same vertices."""
        pytest.importorskip("nibabel")
        from zarr_vectors.types.polylines import read_polylines

        out = {}
        for comp in ("none", "zstd"):
            trk = _write_trk(tmp_path / f"rt_{comp}.trk")
            dest = tmp_path / f"rt_{comp}.zv"
            assert main([
                "convert", str(trk), str(dest), "--num-chunks", "27",
                "--workers", "1", "--compressor", comp,
            ]) == 0
            d = read_polylines(str(dest), level=0)
            out[comp] = (d["polyline_count"], d["vertex_count"])
        assert out["zstd"] == out["none"], out

    def test_trk_compressor_applies_to_every_pyramid_level(self, tmp_path):
        """Coarser levels create their OWN arrays, so they need the codec too.

        A chunk array's pipeline is fixed at creation and level 0's codec does
        not propagate — so the compressor has to be forwarded through
        build_pyramid -> coarsen_level -> the strategy.  Without that the
        pyramid levels silently stay raw while level 0 is compressed.
        """
        pytest.importorskip("nibabel")

        trk = _write_smooth_trk(tmp_path / "pyr.trk")
        out = tmp_path / "pyr.zv"
        rc = main([
            "convert", str(trk), str(out), "--num-chunks", "27",
            "--workers", "1", "--compressor", "zstd",
            "--coarsen", "1,1", "--sparsity", "2,2",
            "--coarsen-mode", "decimate", "--sparsity-strategy", "random",
        ])
        assert rc == 0
        levels = _levels(out)
        assert levels == [0, 1, 2], levels
        for lvl in levels:
            meta = json.loads(
                (out / str(lvl) / "vertices" / "zarr.json").read_text()
            )
            names = [c.get("name") for c in meta["codecs"]]
            assert "zstd" in names, f"level {lvl} left uncompressed: {names}"


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
        with pytest.raises(SystemExit, match="--overwrite"):
            _maybe_overwrite(out, overwrite=False)   # refused without the flag
        assert out.exists()
        _maybe_overwrite(out, overwrite=False, resume=True)   # resume continues into it
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


# ===================================================================
# --shard / shard subcommand
# ===================================================================



def _vertices_sharded(store: Path, level: int = 0) -> bool:
    meta = json.loads((Path(store) / str(level) / "vertices" / "zarr.json").read_text())
    return "sharding_indexed" in [c["name"] for c in meta["codecs"]]


class TestSharding:
    def _multi_chunk_csv(self, tmp_path) -> Path:
        # ~200 occupied chunks so sharding actually packs multiple cells/shard.
        return _write_csv(tmp_path / "pts.csv", n=600)

    def test_convert_shard_produces_sharded_vertices(self, tmp_path):
        src = self._multi_chunk_csv(tmp_path)
        out = tmp_path / "pts.zv"
        rc = main(["convert", str(src), str(out), "--chunk-shape", "40,40,40", "--shard", "2"])
        assert rc == 0
        assert _levels(out) == [0]
        assert _vertices_sharded(out), "vertices must carry the sharding_indexed codec"

    def test_shard_subcommand_roundtrips(self, tmp_path):
        src = self._multi_chunk_csv(tmp_path)
        out = tmp_path / "pts.zv"
        main(["convert", str(src), str(out), "--chunk-shape", "40,40,40"])
        assert not _vertices_sharded(out)                 # unsharded by default

        assert main(["shard", str(out), "--shape", "2"]) == 0
        assert _vertices_sharded(out)                     # now sharded

        assert main(["shard", str(out), "--unshard"]) == 0
        assert not _vertices_sharded(out)                 # back to one file per chunk

    def test_sharded_store_reads_back_identically(self, tmp_path):
        # Sharding must not change coordinates — only the on-disk layout.
        from zarr_vectors.types.points import read_points

        src = self._multi_chunk_csv(tmp_path)
        a, b = tmp_path / "flat.zv", tmp_path / "shard.zv"
        main(["convert", str(src), str(a), "--chunk-shape", "40,40,40"])
        main(["convert", str(src), str(b), "--chunk-shape", "40,40,40", "--shard", "2"])

        pa = np.asarray(read_points(str(a))["positions"])
        pb = np.asarray(read_points(str(b))["positions"])
        pa = pa[np.lexsort(pa.T)]
        pb = pb[np.lexsort(pb.T)]
        assert np.allclose(pa, pb, atol=1e-4)


class TestApplyAffineScope:
    def test_apply_affine_rejected_for_non_trk(self, tmp_path):
        # --apply-affine only applies to trk; other inputs must reject it
        # (no source affine / already in world space), not silently ignore it.
        src = _write_csv(tmp_path / "pts.csv")
        with pytest.raises(SystemExit):
            main(["convert", str(src), str(tmp_path / "o.zv"),
                  "--chunk-shape", "50,50,50", "--apply-affine"])


# ===================================================================
# Options that do not apply are refused, not ignored
# ===================================================================

class TestRefusedOptions:
    @pytest.mark.parametrize("extra", [
        ["--compressor", "zstd"],          # only the trk ingest applies a codec
        ["--num-chunks", "8"],             # trk sizes its grid this way
        ["--n-parts", "4"],
        ["--compute-endpoints"],           # streamlines only
        ["--nodes", "n.csv"],              # edgelist only
    ])
    def test_csv_refuses(self, tmp_path, extra):
        src = _write_csv(tmp_path / "pts.csv")
        with pytest.raises(SystemExit, match="applies to"):
            main(["convert", str(src), str(tmp_path / "o.zv"),
                  "--chunk-shape", "100,100,100", *extra])

    def test_table_refuses_knn(self, tmp_path):
        # Was forwarded to ingest_table, which has no such argument.
        src = tmp_path / "t.csv"
        src.write_text("id,x,y,z\na,1,2,3\nb,4,5,6\n")
        with pytest.raises(SystemExit, match="--knn-distance-k"):
            main(["convert", str(src), str(tmp_path / "o.zv"), "--format", "table",
                  "--position-columns", "x,y,z", "--chunk-shape", "10,10,10",
                  "--knn-distance-k", "2"])

    @pytest.mark.parametrize("coarsen,sparsity", [("0.5,2", "1,1"), ("2,2", "1,0.25")])
    def test_factors_below_one(self, coarsen, sparsity):
        with pytest.raises(SystemExit, match=">= 1"):
            build_factors(parse_float_list(coarsen), parse_float_list(sparsity))


# ===================================================================
# SWC: one object per tree
# ===================================================================

def _write_swc_forest(path: Path, trees: int = 3, nodes: int = 40) -> Path:
    lines, nid = [], 0
    for t in range(trees):
        root = nid + 1
        lines.append(f"{root} 1 {t * 30.0} 0 0 2.0 -1")
        nid = root
        for k in range(1, nodes):
            nid += 1
            lines.append(f"{nid} 3 {t * 30.0 + k * 0.5} {k * 0.7} {k * 0.2} 1.0 {nid - 1}")
    path.write_text("\n".join(lines) + "\n")
    return path


class TestSwcForest:
    def test_each_tree_is_an_object(self, tmp_path):
        from zarr_vectors.building import get_resolution_level, read_all_object_manifests

        out = tmp_path / "f.zv"
        src = _write_swc_forest(tmp_path / "f.swc")
        assert main(["convert", str(src), str(out), "--chunk-shape", "50,50,50"]) == 0
        level0 = get_resolution_level(open_store(str(out)), 0)
        assert len(read_all_object_manifests(level0)) == 3


# ===================================================================
# merge stages a file that needs a chunk shape on the target's grid
# ===================================================================

def _write_grouped_table(path: Path, rows: int, start: int = 0) -> Path:
    rng = np.random.default_rng(start)
    lines = ["cell,x,y,z,group"]
    for i in range(rows):
        x, y, z = rng.uniform(0, 300, 3)
        lines.append(f"c{start + i},{x:.3f},{y:.3f},{z:.3f},{i % 3}")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_merge_table_into_store(tmp_path):
    table = ["--format", "table", "--position-columns", "x,y,z",
             "--key-column", "cell", "--object-id-column", "group"]
    target = tmp_path / "t.zv"
    assert main(["convert", str(_write_grouped_table(tmp_path / "a.csv", 20)),
                 str(target), "--chunk-shape", "100,100,100", *table]) == 0
    extra = _write_grouped_table(tmp_path / "b.csv", 6, start=100)
    # A table needs its column options, which only convert takes: refused
    # with that advice, and the advice works.
    assert main(["merge", str(target), str(extra), "--format", "table"]) == 1
    staged = tmp_path / "b.zv"
    assert main(["convert", str(extra), str(staged),
                 "--chunk-shape", "100,100,100", *table]) == 0
    assert main(["merge", str(target), str(staged), "--pyramid", "drop"]) == 0
    meta = json.loads((target / "0" / "zarr.json").read_text())
    assert meta["attributes"]["zarr_vectors_level"]["vertex_count"] == 26


def test_merge_refuses_points_without_objects(tmp_path):
    target = tmp_path / "t.zv"
    assert main(["convert", str(_write_csv(tmp_path / "a.csv", n=20)), str(target),
                 "--chunk-shape", "100,100,100"]) == 0
    extra = _write_csv(tmp_path / "b.csv", n=5, seed=1)
    assert main(["merge", str(target), str(extra)]) == 1   # IngestError, cleanly
    meta = json.loads((target / "0" / "zarr.json").read_text())
    assert meta["attributes"]["zarr_vectors_level"]["vertex_count"] == 20


# ===================================================================
# zvtools pyramid over an existing pyramid
# ===================================================================

def _multiscale_paths(store: Path) -> list[str]:
    attrs = json.loads((store / "zarr.json").read_text())["attributes"]
    return [d["path"] for d in attrs["multiscales"][0]["datasets"]]


class TestPyramidReplace:
    def _store(self, tmp_path):
        out = tmp_path / "p.zv"
        assert main(["convert", str(_write_csv(tmp_path / "p.csv", n=400)), str(out),
                     "--chunk-shape", "100,100,100", "--bin-shape", "10,10,10",
                     "--coarsen", "2,2,2", "--sparsity", "1,1,1",
                     "--cross-level-storage", "none"]) == 0
        return out

    def test_refused_without_replace(self, tmp_path):
        out = self._store(tmp_path)
        with pytest.raises(SystemExit, match="--replace"):
            main(["pyramid", str(out), "--coarsen", "2", "--sparsity", "1"])
        assert _levels(out) == [0, 1, 2, 3]

    def test_replace_leaves_no_stale_level(self, tmp_path):
        out = self._store(tmp_path)
        assert main(["pyramid", str(out), "--coarsen", "4", "--sparsity", "1",
                     "--cross-level-storage", "none", "--replace"]) == 0
        assert _levels(out) == [0, 1]
        assert _multiscale_paths(out) == ["0", "1"]


def test_convert_refuses_existing_output(tmp_path):
    # Writing into an existing store used to leave its old chunks behind.
    out = tmp_path / "pts.zv"
    assert main(["convert", str(_write_csv(tmp_path / "a.csv", n=300)), str(out),
                 "--chunk-shape", "100,100,100"]) == 0
    with pytest.raises(SystemExit, match="already exists"):
        main(["convert", str(_write_csv(tmp_path / "b.csv", n=5, seed=3)), str(out),
              "--chunk-shape", "100,100,100"])
    meta = json.loads((out / "0" / "zarr.json").read_text())
    assert meta["attributes"]["zarr_vectors_level"]["vertex_count"] == 300


def test_edgelist_attributes_follow_their_edges(tmp_path, monkeypatch):
    import zarr_vectors_tools.convert.ingest.edgelist as edgelist

    written = {}
    real = edgelist.write_graph

    def capture(*args, **kwargs):
        written.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(edgelist, "write_graph", capture)
    nodes = tmp_path / "nodes.csv"
    nodes.write_text("node_id,x,y,z\n100,1,1,1\n101,2,2,2\n102,3,3,3\n")
    edges = tmp_path / "edges.csv"
    # The middle edge names a node that is not in the node table.
    edges.write_text("source,target,w\n100,101,1\n100,999,2\n101,102,3\n")
    edgelist.ingest_edgelist(edges, nodes, tmp_path / "g.zv", (10.0, 10.0, 10.0))
    attrs = written.get("edge_attributes") or written.get("link_attributes")
    assert np.asarray(attrs["w"]).tolist() == [1.0, 3.0]


# ===================================================================
# Pyramid fixes: point duplicates, attribute sparsity, per-level counts
# ===================================================================

def _grouped_points_store(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    group = np.repeat(np.arange(40), 100)
    xyz = rng.uniform(50, 350, (40, 3))[group] + rng.normal(0, 25, (group.size, 3))
    src = tmp_path / "g.csv"
    src.write_text("x,y,z,group\n" + "\n".join(
        f"{x:.3f},{y:.3f},{z:.3f},{g}" for (x, y, z), g in zip(xyz, group)
    ) + "\n")
    out = tmp_path / "g.zv"
    assert main(["convert", str(src), str(out), "--format", "table",
                 "--position-columns", "x,y,z", "--object-id-column", "group",
                 "--chunk-shape", "100,100,100", "--bin-shape", "10,10,10",
                 "--coarsen", "2,2,2", "--sparsity", "1,1,1",
                 "--cross-level-storage", "none"]) == 0
    return out


class TestPointPyramid:
    def test_no_object_stores_a_bin_twice(self, tmp_path):
        from zarr_vectors.types.points import read_points

        out = _grouped_points_store(tmp_path)
        for level in (1, 2, 3):
            for oid in (0, 7, 31):
                got = read_points(str(out), level=level, object_ids=[oid])
                pos = np.asarray(got["positions"])
                assert len(pos) == len(np.unique(pos, axis=0)), (level, oid)

    def test_vertex_count_is_what_a_read_returns(self, tmp_path):
        from zarr_vectors.types.points import read_points

        out = _grouped_points_store(tmp_path)
        for level in (1, 2, 3):
            meta = json.loads((out / str(level) / "zarr.json").read_text())
            rows = len(read_points(str(out), level=level)["positions"])
            assert meta["attributes"]["zarr_vectors_level"]["vertex_count"] == rows


class TestAttributeSparsity:
    def _tracts(self, tmp_path):
        pytest.importorskip("nibabel")
        out = tmp_path / "t.zv"
        assert main(["convert", str(_write_smooth_trk(tmp_path / "t.trk", n=120, npts=40)),
                     str(out), "--num-chunks", "27", "--compute-length"]) == 0
        return out

    def test_keeps_the_highest_values(self, tmp_path):
        from zarr_vectors.building import (
            get_resolution_level,
            read_all_object_manifests,
            read_object_attributes,
        )

        out = self._tracts(tmp_path)
        assert main(["pyramid", str(out), "--coarsen", "1", "--sparsity", "4",
                     "--rdp-tolerance", "0.5", "--sparsity-strategy", "attribute",
                     "--sparsity-attribute", "length"]) == 0
        root = open_store(str(out))
        length = np.asarray(read_object_attributes(get_resolution_level(root, 0), "length"))
        manifests = read_all_object_manifests(get_resolution_level(root, 1))
        kept = {i for i, m in enumerate(manifests) if m}
        assert kept == set(np.argsort(-length)[: len(kept)].tolist())
        assert len(kept) == 30

    def test_needs_the_attribute_named(self, tmp_path):
        out = self._tracts(tmp_path)
        with pytest.raises(SystemExit, match="go together"):
            main(["pyramid", str(out), "--coarsen", "1", "--sparsity", "2",
                  "--sparsity-strategy", "attribute"])
        assert main(["pyramid", str(out), "--coarsen", "1", "--sparsity", "2",
                     "--sparsity-strategy", "attribute",
                     "--sparsity-attribute", "no_such_column"]) == 1
        assert _levels(out) == [0]


def test_build_pyramid_refuses_fractions_and_reports_its_method(tmp_path):
    from zarr_vectors_tools.multiresolution.coarsen import build_pyramid

    out = tmp_path / "p.zv"
    assert main(["convert", str(_write_csv(tmp_path / "p.csv")), str(out),
                 "--chunk-shape", "100,100,100"]) == 0
    with pytest.raises(ValueError, match=">= 1"):
        build_pyramid(str(out), factors=[(0.5, 1)])
    assert _levels(out) == [0]
    pytest.importorskip("nibabel")
    tracts = tmp_path / "t.zv"
    assert main(["convert", str(_write_smooth_trk(tmp_path / "t.trk", n=40, npts=30)),
                 str(tracts), "--num-chunks", "8"]) == 0
    result = build_pyramid(str(tracts), factors=[(1, 2)], rdp_tolerances=[0.5])
    assert result["method"] == result["level_specs"][0]["method"] != "per_object"


def test_coarse_levels_count_their_own_vertices(tmp_path):
    from zarr_vectors.building import get_resolution_level, read_object_attributes

    pytest.importorskip("nibabel")
    out = tmp_path / "t.zv"
    assert main(["convert", str(_write_smooth_trk(tmp_path / "t.trk", n=80, npts=50)),
                 str(out), "--num-chunks", "8", "--object-attr", "vertex_count",
                 "--coarsen", "1,1", "--sparsity", "2,2", "--rdp-tolerance", "0.5,1"]) == 0
    root = open_store(str(out))
    for level in (0, 1, 2):
        level_group = get_resolution_level(root, level)
        counts = np.asarray(read_object_attributes(level_group, "vertex_count"))
        meta = json.loads((out / str(level) / "zarr.json").read_text())
        assert int(counts.sum()) == meta["attributes"]["zarr_vectors_level"]["vertex_count"]
        if level:
            assert (counts == 0).sum() > 0      # dropped objects count 0, not "missing"


def test_mesh_decimate_without_scipy_leaves_attributes_off(tmp_path, monkeypatch):
    import zarr_vectors_tools.multiresolution.strategies.mesh_decimate_level as mdl
    from zarr_vectors_tools.convert.ingest.obj import ingest_obj
    from zarr_vectors_tools.multiresolution.coarsen import build_pyramid

    t = (1 + 5 ** 0.5) / 2
    verts = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
                      [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]])
    faces = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
             [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
             [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    obj = tmp_path / "ico.obj"
    obj.write_text("".join(f"v {a} {b} {c}\n" for a, b, c in verts * 10 + 50)
                   + "".join(f"vn {a} {b} {c}\n" for a, b, c in verts)
                   + "".join(f"f {a + 1}//{a + 1} {b + 1}//{b + 1} {c + 1}//{c + 1}\n"
                             for a, b, c in faces))
    out = tmp_path / "ico.zv"
    ingest_obj(obj, out, (100.0, 100.0, 100.0))
    monkeypatch.setattr(mdl, "_have_scipy", lambda: False)
    with pytest.warns(UserWarning, match="not carried"):
        build_pyramid(str(out), factors=[(1.5, 1)], method="mesh_decimate")
    assert not (out / "1" / "vertex_attributes").exists()


def test_coarse_point_levels_carry_attributes(tmp_path):
    from zarr_vectors.types.points import read_points

    rng = np.random.default_rng(2)
    group = np.repeat(np.arange(10), 60)
    xyz = rng.uniform(20, 280, (10, 3))[group] + rng.normal(0, 15, (group.size, 3))
    src = tmp_path / "a.csv"
    src.write_text("x,y,z,group,intensity,label\n" + "\n".join(
        f"{x:.3f},{y:.3f},{z:.3f},{g},{x / 1000:.6f},{g % 3}"
        for (x, y, z), g in zip(xyz, group)
    ) + "\n")
    out = tmp_path / "a.zv"
    assert main(["convert", str(src), str(out), "--format", "table",
                 "--position-columns", "x,y,z", "--object-id-column", "group",
                 "--chunk-shape", "100,100,100", "--bin-shape", "10,10,10",
                 "--coarsen", "2", "--sparsity", "1", "--cross-level-storage", "none"]) == 0
    level1 = read_points(str(out), level=1, attribute_names=["intensity", "label"])
    attrs = level1["vertex_attributes"]
    # A float column is the bin's mean, as the position is: intensity = x/1000.
    np.testing.assert_allclose(attrs["intensity"], level1["positions"][:, 0] / 1000, atol=1e-5)
    # Codes keep a value that occurred, not a blend.
    assert set(np.unique(attrs["label"]).tolist()) <= {0, 1, 2}
