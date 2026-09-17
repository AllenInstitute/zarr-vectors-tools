"""Streamlines go into a store and come back out with everything attached.

The TRK and TRX writers used to emit positions and nothing else, so a
tractogram could be ingested but never handed back to the tools that made it:
per-point scalars, per-streamline properties, bundle names and the reference
image were all lost on the way out.  Each test here ingests a file, exports
it, and loads the result with the format's own reader.

Two ingest bugs found on the way are pinned too: the serial TRK ingester
never stored its header, and the parallel one read the property count from
the wrong byte, which garbled the offset scan of any file with properties.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
from nibabel.streamlines import Field, Tractogram  # noqa: E402
from nibabel.streamlines.trk import TrkFile  # noqa: E402
from zarr_vectors.exceptions import ExportError  # noqa: E402

from zarr_vectors_tools.convert.export.trk import export_trk  # noqa: E402
from zarr_vectors_tools.convert.ingest._trk_names import decode_trk_names  # noqa: E402

AFFINE = np.array(
    [[-1.25, 0, 0, 90], [0, 1.25, 0, -126], [0, 0, 1.25, -72], [0, 0, 0, 1]],
    dtype=np.float32,
)


def _tractogram(seed: int = 0):
    rng = np.random.default_rng(seed)
    # Inside the 145x174x145 grid at 1.25 mm, in RAS, so both TRK spaces
    # hold them without clipping.
    streamlines = [
        (rng.uniform(-60, 60, (n, 3)) + [0, -20, 0]).astype(np.float32)
        for n in (6, 11, 4, 9)
    ]
    data_per_point = {
        "fa": [rng.random((len(s), 1)).astype(np.float32) for s in streamlines],
        "rgb": [rng.random((len(s), 3)).astype(np.float32) for s in streamlines],
    }
    data_per_streamline = {
        "weight": np.arange(4, dtype=np.float32)[:, None],
        "colour": rng.random((4, 3)).astype(np.float32),
    }
    return streamlines, data_per_point, data_per_streamline


def _write_trk(path: Path, seed: int = 0):
    streamlines, dpp, dps = _tractogram(seed)
    header = {
        Field.VOXEL_TO_RASMM: AFFINE,
        Field.VOXEL_SIZES: (1.25, 1.25, 1.25),
        Field.DIMENSIONS: (145, 174, 145),
        Field.VOXEL_ORDER: "LAS",
    }
    TrkFile(
        Tractogram(streamlines, data_per_point=dpp, data_per_streamline=dps,
                   affine_to_rasmm=np.eye(4)),
        header=header,
    ).save(str(path))
    return streamlines, dpp, dps


def _by_first_point(streamlines):
    """Order-independent key: stores do not promise file order on export."""
    return sorted(range(len(streamlines)), key=lambda i: tuple(np.round(streamlines[i][0], 2)))


def _assert_trk_matches(out: Path, streamlines, dpp, dps) -> None:
    loaded = nib.streamlines.load(str(out))
    got = list(loaded.streamlines)
    assert len(got) == len(streamlines)
    order_in, order_out = _by_first_point(streamlines), _by_first_point(got)
    for i, j in zip(order_in, order_out):
        np.testing.assert_allclose(got[j], streamlines[i], atol=2e-3)
        for name in dpp:
            np.testing.assert_allclose(
                loaded.tractogram.data_per_point[name][j], dpp[name][i], atol=1e-6,
            )
        for name in dps:
            np.testing.assert_allclose(
                loaded.tractogram.data_per_streamline[name][j], dps[name][i], atol=1e-6,
            )
    np.testing.assert_allclose(loaded.header[Field.VOXEL_TO_RASMM], AFFINE, atol=1e-6)
    np.testing.assert_allclose(loaded.header[Field.VOXEL_SIZES], (1.25, 1.25, 1.25))
    assert tuple(loaded.header[Field.DIMENSIONS]) == (145, 174, 145)
    assert loaded.header[Field.VOXEL_ORDER] in (b"LAS", "LAS")


# ===================================================================
# The TRK header name tables
# ===================================================================

class TestTrkNames:

    @staticmethod
    def _table(*slots: bytes) -> bytes:
        raw = bytearray(1000)
        for i, slot in enumerate(slots):
            raw[38 + i * 20: 38 + i * 20 + len(slot)] = slot
        return bytes(raw)

    def test_a_multi_column_slot_is_one_name(self) -> None:
        raw = self._table(b"fa", b"rgb\x003")
        assert decode_trk_names(raw, 38, 4) == [("fa", 1), ("rgb", 3)]

    def test_trackvis_style_one_slot_per_column(self) -> None:
        raw = self._table(b"fa", b"md")
        assert decode_trk_names(raw, 38, 2) == [("fa", 1), ("md", 1)]

    def test_unnamed_columns_are_still_addressable(self) -> None:
        raw = self._table(b"fa")
        assert decode_trk_names(raw, 38, 3) == [("fa", 1), ("scalar_1", 1), ("scalar_2", 1)]

    def test_a_width_past_the_column_count_is_clipped(self) -> None:
        raw = self._table(b"rgb\x009")
        assert decode_trk_names(raw, 38, 3) == [("rgb", 3)]


# ===================================================================
# TRK in, TRK out
# ===================================================================

class TestTrkRoundTrip:

    def test_serial_ingest(self, tmp_path: Path) -> None:
        from zarr_vectors_tools.convert.ingest.trk import ingest_trk

        source = tmp_path / "in.trk"
        streamlines, dpp, dps = _write_trk(source)
        store = tmp_path / "s.zv"
        ingest_trk(str(source), str(store), (50.0, 50.0, 50.0))

        out = tmp_path / "out.trk"
        summary = export_trk(str(store), str(out))
        assert summary["attributes_carried"] == ["fa", "rgb"]
        assert summary["object_attributes_carried"] == ["colour", "weight"]
        assert summary["space"] == "rasmm"
        _assert_trk_matches(out, streamlines, dpp, dps)

    @pytest.mark.parametrize("register", [False, True], ids=["voxmm", "rasmm"])
    def test_parallel_ingest(self, tmp_path: Path, register: bool) -> None:
        from zarr_vectors_tools.convert.ingest.trk_parallel import ingest_trk_parallel

        source = tmp_path / "in.trk"
        streamlines, dpp, dps = _write_trk(source)
        store = tmp_path / "s.zv"
        ingest_trk_parallel(
            str(source), str(store), num_chunks=8, n_parts=2, workers=1,
            register_to_rasmm=register, build_multiscale=False, progress=False,
        )

        out = tmp_path / "out.trk"
        summary = export_trk(str(store), str(out))
        assert summary["space"] == ("rasmm" if register else "voxmm")
        assert summary["attributes_carried"] == ["fa", "rgb"]
        _assert_trk_matches(out, streamlines, dpp, dps)

    def test_parallel_ingest_reads_a_file_with_properties(self, tmp_path: Path) -> None:
        """The property count used to be read from byte 236, not 238."""
        from zarr_vectors.types.polylines import read_polylines

        from zarr_vectors_tools.convert.ingest.trk_parallel import (
            ingest_trk_parallel,
            parse_trk_header,
        )

        source = tmp_path / "in.trk"
        streamlines, _dpp, _dps = _write_trk(source)
        header = parse_trk_header(source)
        assert header["n_properties"] == 4
        assert header["property_names"] == [("colour", 3), ("weight", 1)]

        store = tmp_path / "s.zv"
        ingest_trk_parallel(
            str(source), str(store), num_chunks=8, n_parts=2, workers=1,
            build_multiscale=False, progress=False,
        )
        result = read_polylines(str(store))
        counts = sorted(sum(len(f) for f in p) for p in result["polylines"])
        assert counts == sorted(len(s) for s in streamlines)

    def test_attribute_selection(self, tmp_path: Path) -> None:
        from zarr_vectors_tools.convert.ingest.trk import ingest_trk

        source = tmp_path / "in.trk"
        _write_trk(source)
        store = tmp_path / "s.zv"
        ingest_trk(str(source), str(store), (50.0, 50.0, 50.0))

        out = tmp_path / "some.trk"
        summary = export_trk(
            str(store), str(out), attribute_names=["rgb"], object_attribute_names=[],
        )
        assert summary["attributes_carried"] == ["rgb"]
        assert summary["object_attributes_carried"] == []
        loaded = nib.streamlines.load(str(out))
        assert list(loaded.tractogram.data_per_point) == ["rgb"]
        assert not loaded.tractogram.data_per_streamline

        with pytest.raises(ExportError, match="not present"):
            export_trk(str(store), str(tmp_path / "x.trk"), attribute_names=["nope"])
        with pytest.raises(ExportError, match="not present"):
            export_trk(str(store), str(tmp_path / "x.trk"), object_attribute_names=["nope"])

    def test_more_scalars_than_trk_holds_names_the_way_out(self, tmp_path: Path) -> None:
        from zarr_vectors.types.polylines import write_polylines

        store = tmp_path / "many.zv"
        lines = [np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.float32)]
        write_polylines(
            str(store), lines, chunk_shape=(10.0, 10.0, 10.0), geometry_type="streamline",
            vertex_attributes={f"s{i}": [np.ones(3, np.float32)] for i in range(11)},
        )
        with pytest.raises(ExportError, match="attribute_names"):
            export_trk(str(store), str(tmp_path / "x.trk"))


# ===================================================================
# TRX in, TRX out
# ===================================================================

def _trx():
    return pytest.importorskip("trx.trx_file_memmap")


def _write_trx(path: Path):
    from nibabel.streamlines.array_sequence import ArraySequence

    trx_memmap = _trx()
    rng = np.random.default_rng(3)
    streamlines = [rng.uniform(0, 40, (n, 3)).astype(np.float32) for n in (3, 5, 4, 6)]
    positions = np.concatenate(streamlines)
    lengths = np.array([len(s) for s in streamlines], np.uint32)
    offsets = np.concatenate([[0], np.cumsum(lengths[:-1])]).astype(np.uint32)

    trx = trx_memmap.TrxFile(nb_vertices=len(positions), nb_streamlines=len(streamlines))
    trx.header["VOXEL_TO_RASMM"] = np.diag([2.0, 2.0, 2.0, 1.0]).tolist()
    trx.header["DIMENSIONS"] = [91, 109, 91]
    trx.streamlines._data = positions
    trx.streamlines._offsets = offsets
    trx.streamlines._lengths = lengths

    def _seq(values):
        seq = ArraySequence()
        seq._data = values
        seq._offsets, seq._lengths = offsets.astype(np.int64), lengths.astype(np.int64)
        return seq

    fa = rng.random((len(positions), 1)).astype(np.float32)
    trx.data_per_vertex = {"fa": _seq(fa)}
    weight = np.array([0.5, 1.5, 2.5, 3.5], np.float32)
    trx.data_per_streamline = {"weight": weight}
    trx.groups = {"CST_L": np.array([0, 2], np.uint32), "AF_R": np.array([1, 3], np.uint32)}
    trx.data_per_group = {
        "CST_L": {"mean_fa": np.array([[0.4]], np.float32)},
        "AF_R": {"mean_fa": np.array([[0.6]], np.float32)},
    }
    trx_memmap.save(trx, str(path))
    trx.close()
    return streamlines, fa, weight


class TestTrxRoundTrip:

    def _ingest(self, tmp_path: Path):
        from zarr_vectors_tools.convert.ingest.trx import ingest_trx

        source = tmp_path / "in.trx"
        streamlines, fa, weight = _write_trx(source)
        store = tmp_path / "s.zv"
        ingest_trx(str(source), str(store), (20.0, 20.0, 20.0))
        return store, streamlines, fa, weight

    def test_bundle_names_are_kept_up_the_pyramid(self, tmp_path: Path) -> None:
        import zarr_vectors as zv

        from zarr_vectors_tools.multiresolution.coarsen import build_pyramid

        store, *_ = self._ingest(tmp_path)
        build_pyramid(str(store), factors=[(2.0, 1.0)])
        dataset = zv.open(str(store), mode="r")
        for level in (0, 1):
            assert set(dataset.level(level).groups.names()) == {"CST_L", "AF_R"}
        assert sorted(dataset.level(0).groups["AF_R"].members.tolist()) == [1, 3]

    def test_everything_comes_back(self, tmp_path: Path) -> None:
        from zarr_vectors_tools.convert.export.trx import export_trx

        store, streamlines, fa, weight = self._ingest(tmp_path)
        out = tmp_path / "out.trx"
        summary = export_trx(str(store), str(out))
        assert summary["groups_carried"] == ["AF_R", "CST_L"]

        loaded = _trx().load(str(out))
        try:
            assert np.allclose(loaded.header["VOXEL_TO_RASMM"], np.diag([2.0, 2.0, 2.0, 1.0]))
            assert list(loaded.header["DIMENSIONS"]) == [91, 109, 91]
            got = [np.asarray(s).copy() for s in loaded.streamlines]
            offsets = np.concatenate([[0], np.cumsum([len(s) for s in streamlines])])
            got_offsets = np.concatenate([[0], np.cumsum([len(g) for g in got])])
            order_in, order_out = _by_first_point(streamlines), _by_first_point(got)
            index_map = dict(zip(order_in, order_out))
            got_fa = np.asarray(loaded.data_per_vertex["fa"]._data).reshape(-1)
            got_weight = np.asarray(loaded.data_per_streamline["weight"]).reshape(-1)
            for i, j in index_map.items():
                np.testing.assert_array_equal(got[j], streamlines[i])
                np.testing.assert_allclose(
                    got_fa[got_offsets[j]:got_offsets[j + 1]],
                    fa[offsets[i]:offsets[i + 1], 0],
                )
                assert got_weight[j] == pytest.approx(weight[i])
            groups = {k: sorted(np.asarray(v).tolist()) for k, v in loaded.groups.items()}
            assert groups == {
                "CST_L": sorted([index_map[0], index_map[2]]),
                "AF_R": sorted([index_map[1], index_map[3]]),
            }
            assert float(np.asarray(loaded.data_per_group["AF_R"]["mean_fa"]).ravel()[0]) == (
                pytest.approx(0.6)
            )
        finally:
            loaded.close()

    def test_group_ids_limit_the_groups_written(self, tmp_path: Path) -> None:
        import zarr_vectors as zv

        from zarr_vectors_tools.convert.export.trx import export_trx

        store, *_ = self._ingest(tmp_path)
        # Row order is whatever order the TRX reader listed the groups in, so
        # address the bundle by name.
        cst = zv.open(str(store), mode="r").level(0).groups["CST_L"].id
        out = tmp_path / "cst.trx"
        summary = export_trx(str(store), str(out), group_ids=[cst])
        assert summary["groups_carried"] == ["CST_L"]
        assert summary["streamline_count"] == 2

    def test_a_voxmm_trk_store_exports_trx_in_ras(self, tmp_path: Path) -> None:
        from zarr_vectors_tools.convert.export.trx import export_trx
        from zarr_vectors_tools.convert.ingest.trk_parallel import ingest_trk_parallel

        source = tmp_path / "in.trk"
        streamlines, _dpp, _dps = _write_trk(source)
        store = tmp_path / "s.zv"
        ingest_trk_parallel(
            str(source), str(store), num_chunks=8, n_parts=2, workers=1,
            build_multiscale=False, progress=False,
        )
        out = tmp_path / "out.trx"
        export_trx(str(store), str(out))
        loaded = _trx().load(str(out))
        try:
            got = [np.asarray(s).copy() for s in loaded.streamlines]
            np.testing.assert_allclose(loaded.header["VOXEL_TO_RASMM"], AFFINE, atol=1e-6)
        finally:
            loaded.close()
        for i, j in zip(_by_first_point(streamlines), _by_first_point(got)):
            np.testing.assert_allclose(got[j], streamlines[i], atol=2e-3)


# ===================================================================
# From the command line
# ===================================================================

class TestCli:

    def test_attribute_flags_choose_what_is_written(self, tmp_path: Path) -> None:
        from zarr_vectors_tools.cli import main
        from zarr_vectors_tools.convert.ingest.trk import ingest_trk

        source = tmp_path / "in.trk"
        _write_trk(source)
        store = tmp_path / "s.zv"
        ingest_trk(str(source), str(store), (50.0, 50.0, 50.0))
        out = tmp_path / "out.trk"
        assert main([
            "convert", str(store), str(out),
            "--attribute", "fa", "--object-attribute", "weight",
        ]) == 0
        loaded = nib.streamlines.load(str(out))
        assert list(loaded.tractogram.data_per_point) == ["fa"]
        assert list(loaded.tractogram.data_per_streamline) == ["weight"]

    def test_object_attribute_is_refused_where_it_cannot_apply(self, tmp_path: Path) -> None:
        from zarr_vectors.types.points import write_points

        from zarr_vectors_tools.cli import main

        store = tmp_path / "p.zv"
        write_points(str(store), np.ones((3, 3), np.float32), chunk_shape=(5.0, 5.0, 5.0))
        with pytest.raises(SystemExit, match="--object-attribute"):
            main(["convert", str(store), str(tmp_path / "p.csv"), "--object-attribute", "x"])
