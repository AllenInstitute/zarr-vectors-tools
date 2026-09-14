"""``zvtools convert`` reads Neuroglancer precomputed skeleton layers.

The two ingesters have their own tests, driven by an in-memory reader.  What
is tested here is the seam: that a layer is recognised from a directory or a
URL, that its ``info`` picks the ingester, that the CLI's options reach it or
are refused by name, and that the real readers open a layer on disk.  The
layers are written with cloud-volume's own skeleton encoder (and mapbuffer
for ``.frags``), so they are what Neuroglancer would read.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from zarr_vectors_tools.cli import main
from zarr_vectors_tools.cli._args import resolve_format
from zarr_vectors_tools.ingest.precomputed import frag_bounds, layer_url

RADIUS = [{"id": "radius", "data_type": "float32", "num_components": 1}]


def _encode(segment_id: int, vertices: np.ndarray) -> bytes:
    from cloudvolume import Skeleton

    n = len(vertices)
    skeleton = Skeleton(
        np.asarray(vertices, dtype=np.float32),
        np.array([[i, i - 1] for i in range(1, n)], dtype=np.uint32),
        radii=np.linspace(1.0, 2.0, n, dtype=np.float32),
        segid=segment_id,
        extra_attributes=RADIUS,
    )
    return skeleton.to_precomputed()


def _read_back(store: Path, segment_id: int, level: int = 0):
    from zarr_vectors.types import skeletons as sk

    return sk.read_skeleton_by_segment_id(str(store), segment_id, level=level)


def _same_points(a: np.ndarray, b: np.ndarray) -> bool:
    a = np.unique(np.round(np.asarray(a, dtype=np.float64), 2), axis=0)
    b = np.unique(np.round(np.asarray(b, dtype=np.float64), 2), axis=0)
    return a.shape == b.shape and np.allclose(a, b)


# ===================================================================
# Layers on disk
# ===================================================================

#: Plain layer: three segments, one of them spanning two 1000-nm chunks.
PLAIN_SEGMENTS = {
    720575940000000001: np.array(
        [[100, 100, 100], [300, 150, 120], [600, 200, 140]], dtype=np.float32),
    720575940000000002: np.array(
        [[800, 400, 100], [950, 420, 110], [1200, 450, 120], [1500, 480, 130]],
        dtype=np.float32),
    720575940000000003: np.array(
        [[200, 1300, 400], [250, 1350, 420]], dtype=np.float32),
}


@pytest.fixture
def plain_layer(tmp_path: Path) -> Path:
    pytest.importorskip("cloudvolume")
    layer = tmp_path / "skeletons"
    (layer / "segment_properties").mkdir(parents=True)
    (layer / "info").write_text(json.dumps({
        "@type": "neuroglancer_skeletons",
        "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        "vertex_attributes": RADIUS,
        "segment_properties": "segment_properties",
    }))
    ids = sorted(PLAIN_SEGMENTS)
    (layer / "segment_properties" / "info").write_text(json.dumps({
        "@type": "neuroglancer_segment_properties",
        "inline": {
            "ids": [str(i) for i in ids],
            "properties": [
                {"id": "label", "type": "label", "values": ["a", "b", "c"]},
            ],
        },
    }))
    for segment_id, vertices in PLAIN_SEGMENTS.items():
        (layer / str(segment_id)).write_bytes(_encode(segment_id, vertices))
    return layer


#: Spatially indexed layer: 8x8x40 nm voxels, 256x256x100-voxel chunks, so
#: each .frags file covers 2048x2048x4000 nm.  The block starts away from the
#: origin, as a real cutout does.
RESOLUTION = (8.0, 8.0, 40.0)
CHUNK_NM = (2048.0, 2048.0, 4000.0)
KEY_A = "512-768_256-512_100-200.frags"    # starts at (4096, 2048, 4000) nm
KEY_B = "768-1024_256-512_100-200.frags"   # the next chunk along x
FRAG_SEGMENTS = {
    KEY_A: {
        11: np.array([[4500, 2500, 5000], [4700, 2600, 5100], [5000, 2700, 5200]]),
    },
    KEY_B: {
        22: np.array([[6500, 3000, 6000], [6800, 3100, 6100]]),
    },
}


def _write_frags_layer(root: Path, frags_dir: str = "") -> Path:
    pytest.importorskip("cloudvolume")
    pytest.importorskip("mapbuffer")
    from mapbuffer import MapBuffer

    root.mkdir(parents=True)
    (root / "info").write_text(json.dumps({
        "@type": "neuroglancer_skeletons",
        "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        "vertex_attributes": RADIUS,
        "spatial_index": {"resolution": list(RESOLUTION), "chunk_size": list(CHUNK_NM)},
    }))
    target = root / frags_dir if frags_dir else root
    target.mkdir(parents=True, exist_ok=True)
    for key, segments in FRAG_SEGMENTS.items():
        buffer = MapBuffer({
            segment_id: _encode(segment_id, vertices)
            for segment_id, vertices in segments.items()
        })
        (target / key).write_bytes(buffer.tobytes())
    return root


@pytest.fixture
def frags_layer(tmp_path: Path) -> Path:
    return _write_frags_layer(tmp_path / "frags_layer")


# ===================================================================
# Recognising a layer
# ===================================================================

class TestDetection:

    @pytest.mark.parametrize("source", [
        "gs://flywire_v141_m783/skeletons_mip_1",
        "precomputed://gs://allen_neuroglancer_ccf/Mouselight",
        "https://example.org/layer",
    ])
    def test_a_url_is_a_precomputed_layer(self, source: str) -> None:
        assert resolve_format(source, None).name == "precomputed"

    def test_a_directory_holding_an_info_file_is_a_precomputed_layer(
        self, tmp_path: Path,
    ) -> None:
        (tmp_path / "info").write_text("{}")
        assert resolve_format(tmp_path, None).name == "precomputed"

    def test_layer_url_drops_the_neuroglancer_scheme(self) -> None:
        assert layer_url("precomputed://gs://bucket/layer/") == "gs://bucket/layer"

    def test_layer_url_turns_a_directory_into_a_file_url(self, tmp_path: Path) -> None:
        assert layer_url(tmp_path) == "file://" + tmp_path.resolve().as_posix()

    def test_precomputed_flags_are_refused_for_other_formats(
        self, tmp_path: Path,
    ) -> None:
        source = tmp_path / "p.csv"
        source.write_text("x,y,z\n1,2,3\n")
        with pytest.raises(SystemExit, match="--anchor applies to precomputed"):
            main(["convert", str(source), str(tmp_path / "o.zv"),
                  "--chunk-shape", "5,5,5", "--anchor", "0,0,0"])

    def test_a_volume_layer_points_at_its_skeletons(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        pytest.importorskip("cloudfiles")
        layer = tmp_path / "seg"
        layer.mkdir()
        (layer / "info").write_text(json.dumps({
            "@type": "neuroglancer_multiscale_volume", "type": "segmentation",
            "scales": [], "skeletons": "skeletons_mip_2",
        }))
        assert main(["convert", str(layer), str(tmp_path / "o.zv")]) == 1
        assert "skeletons_mip_2" in capsys.readouterr().err

    def test_off_grid_frags_are_refused(self) -> None:
        from zarr_vectors_tools.ingest.precomputed_skeletons import SkeletonInfo

        info = SkeletonInfo(base_url="mem://x", resolution_nm=RESOLUTION,
                            chunk_size_nm=CHUNK_NM)
        assert frag_bounds(info, [KEY_A, KEY_B]) == (
            [4096.0, 2048.0, 4000.0], [8192.0, 4096.0, 8000.0],
        )
        with pytest.raises(ValueError, match="not on the"):
            frag_bounds(info, [KEY_A, "600-856_256-512_100-200.frags"])


# ===================================================================
# A layer with no spatial index
# ===================================================================

class TestPlainLayer:

    def test_every_segment_comes_across_with_its_attributes(
        self, plain_layer: Path, tmp_path: Path,
    ) -> None:
        from zarr_vectors.building import (
            get_resolution_level,
            open_store,
            read_object_attributes,
        )

        store = tmp_path / "plain.zv"
        assert main(["convert", str(plain_layer), str(store),
                     "--chunk-shape", "1000,1000,1000"]) == 0

        for segment_id, vertices in PLAIN_SEGMENTS.items():
            skeleton = _read_back(store, segment_id)
            assert skeleton is not None, segment_id
            assert _same_points(skeleton["positions"], vertices)
            assert "radius" in skeleton["attributes"]
        level0 = get_resolution_level(open_store(str(store)), 0)
        labels = read_object_attributes(level0, "label")
        assert sorted(v.decode().rstrip("\x00") for v in labels.tolist()) == ["a", "b", "c"]

    def test_segment_id_selects_segments(
        self, plain_layer: Path, tmp_path: Path,
    ) -> None:
        store = tmp_path / "one.zv"
        wanted, *others = sorted(PLAIN_SEGMENTS)
        assert main(["convert", str(plain_layer), str(store),
                     "--chunk-shape", "1000,1000,1000",
                     "--segment-id", str(wanted)]) == 0
        assert _read_back(store, wanted) is not None
        assert all(_read_back(store, other) is None for other in others)

    def test_coarsen_and_sparsity_build_the_pyramid(
        self, plain_layer: Path, tmp_path: Path,
    ) -> None:
        from zarr_vectors.building import list_resolution_levels, open_store

        store = tmp_path / "pyramid.zv"
        assert main(["convert", str(plain_layer), str(store),
                     "--chunk-shape", "1000,1000,1000",
                     "--coarsen", "2", "--sparsity", "1"]) == 0
        assert len(list_resolution_levels(open_store(str(store)))) == 2

    def test_chunk_shape_is_required(
        self, plain_layer: Path, tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        assert main(["convert", str(plain_layer), str(tmp_path / "o.zv")]) == 1
        assert "no spatial index" in capsys.readouterr().err

    def test_frags_options_are_refused(
        self, plain_layer: Path, tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        assert main(["convert", str(plain_layer), str(tmp_path / "o.zv"),
                     "--chunk-shape", "1000,1000,1000", "--anchor", "0,0,0"]) == 1
        assert "anchor" in capsys.readouterr().err

    def test_a_file_url_reads_the_same_layer(
        self, plain_layer: Path, tmp_path: Path,
    ) -> None:
        store = tmp_path / "url.zv"
        url = "precomputed://" + layer_url(plain_layer)
        assert main(["convert", url, str(store),
                     "--chunk-shape", "1000,1000,1000"]) == 0
        assert _read_back(store, min(PLAIN_SEGMENTS)) is not None


# ===================================================================
# A layer with a spatial index
# ===================================================================

class TestSpatiallyIndexedLayer:

    def test_every_listed_chunk_comes_across(
        self, frags_layer: Path, tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        store = tmp_path / "frags.zv"
        assert main(["convert", str(frags_layer), str(store)]) == 0
        out = capsys.readouterr().out
        assert "listed 2 .frags chunks" in out
        assert "layer: spatial_index" in out
        for segments in FRAG_SEGMENTS.values():
            for segment_id, vertices in segments.items():
                skeleton = _read_back(store, segment_id)
                assert skeleton is not None, segment_id
                # Stored aligned to the block, read back in world nanometres.
                assert _same_points(skeleton["positions"], vertices)
                assert "radius" in skeleton["attributes"]

    def test_anchor_selects_a_block(
        self, frags_layer: Path, tmp_path: Path,
    ) -> None:
        store = tmp_path / "block.zv"
        assert main(["convert", str(frags_layer), str(store),
                     "--anchor", "512,256,100", "--counts", "1,1,1"]) == 0
        assert _read_back(store, 11) is not None
        assert _read_back(store, 22) is None

    def test_frags_dir_names_where_the_chunks_are(self, tmp_path: Path) -> None:
        layer = _write_frags_layer(tmp_path / "nested", frags_dir="frags")
        assert main(["convert", str(layer), str(tmp_path / "missing.zv")]) == 1
        store = tmp_path / "nested.zv"
        assert main(["convert", str(layer), str(store),
                     "--frags-dir", "frags"]) == 0
        assert _read_back(store, 22) is not None

    def test_a_pyramid_is_built_inline(
        self, frags_layer: Path, tmp_path: Path,
    ) -> None:
        from zarr_vectors.building import list_resolution_levels, open_store

        store = tmp_path / "pyramid.zv"
        assert main(["convert", str(frags_layer), str(store),
                     "--coarsen", "2,2", "--sparsity", "1,1",
                     "--chunk-scale", "2,2"]) == 0
        assert len(list_resolution_levels(open_store(str(store)))) == 3

    @pytest.mark.parametrize("flags, message", [
        (["--chunk-shape", "100,100,100"], "leave"),
        (["--segment-id", "11"], "segment_ids"),
        (["--compressor", "zstd"], "--compressor"),
        (["--anchor", "1,2"], "three integers"),
    ])
    def test_options_that_cannot_apply_are_refused(
        self, frags_layer: Path, tmp_path: Path,
        capsys: pytest.CaptureFixture[str], flags: list[str], message: str,
    ) -> None:
        try:
            rc = main(["convert", str(frags_layer), str(tmp_path / "o.zv"), *flags])
        except SystemExit as exc:
            assert message in str(exc)
            return
        assert rc == 1
        assert message in capsys.readouterr().err

    @pytest.mark.slow
    def test_worker_processes_open_the_layer_themselves(
        self, frags_layer: Path, tmp_path: Path,
    ) -> None:
        serial = tmp_path / "serial.zv"
        parallel = tmp_path / "parallel.zv"
        assert main(["convert", str(frags_layer), str(serial)]) == 0
        assert main(["convert", str(frags_layer), str(parallel),
                     "--workers", "2"]) == 0
        for segment_id in (11, 22):
            a, b = _read_back(serial, segment_id), _read_back(parallel, segment_id)
            assert np.array_equal(a["positions"], b["positions"])
