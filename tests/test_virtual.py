"""Virtual zarr-vectors stores: tractograms read as zarr-vectors stores without
converting them.  The store must be a valid 0.9.2 store whose streamlines
read back as nibabel reads the file, and the server must answer keys,
ranges and listings as Neuroglancer asks for them."""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
from nibabel.streamlines import Field, TckFile, Tractogram, TrkFile  # noqa: E402

from zarr_vectors_tools.virtual import cell_size, fragment_index, open_virtual  # noqa: E402
from zarr_vectors_tools.virtual.serve import make_handler  # noqa: E402


def _lines(n, seed=0):
    rng = np.random.default_rng(seed)
    return [(rng.uniform(-60, 60, 3) + rng.normal(0, 2, (int(rng.integers(2, 40)), 3)).cumsum(0)).astype(np.float32) for _ in range(n)]


@pytest.fixture
def trk(tmp_path):
    lines = _lines(40)
    fa = [np.random.default_rng(1).uniform(0, 1, (len(s), 1)).astype(np.float32) for s in lines]
    affine = np.diag([-2.0, 2.0, 2.5, 1.0])
    affine[:3, 3] = [90, -126, -72]
    header = {Field.VOXEL_TO_RASMM: affine, Field.VOXEL_ORDER: "LAS", Field.DIMENSIONS: np.array([91, 109, 91], np.int16), Field.VOXEL_SIZES: np.array([2, 2, 2.5], np.float32)}
    path = tmp_path / "tracts.trk"
    TrkFile(Tractogram(lines, data_per_point={"fa": fa}, affine_to_rasmm=np.eye(4)), header=header).save(str(path))
    return path


def _write(store, root):
    for key in store.keys():
        data = store.get(key)
        assert data is not None, key
        (root / key).parent.mkdir(parents=True, exist_ok=True)
        (root / key).write_bytes(data)


def test_fragment_index_and_cell_size():
    assert len(fragment_index([5])) == 44
    assert len(fragment_index(list(range(1, 8)))) == 140
    assert len(fragment_index([])) == 16
    assert cell_size(182) == 20 and cell_size(0) == 16


@pytest.mark.parametrize("kind", ["trk", "tck"])
def test_store_is_valid_and_reads_back_as_nibabel(tmp_path, trk, kind):
    from zarr_vectors.types.polylines import read_polylines
    from zarr_vectors.validate import validate

    path = trk
    if kind == "tck":
        path = tmp_path / "tracts.tck"
        TckFile(Tractogram(_lines(30, seed=5), affine_to_rasmm=np.eye(4))).save(str(path))
    expected = list(nib.streamlines.load(str(path)).streamlines)
    store = open_virtual(path)
    root = tmp_path / "virtual.zarrvectors"
    _write(store, root)
    assert validate(str(root), level=3).ok
    back = read_polylines(str(root))
    lines = [np.concatenate([np.asarray(f) for f in p]) for p in back["polylines"]]
    assert len(lines) == len(expected)
    for got, want in zip(lines, expected):
        np.testing.assert_allclose(got, want, atol=1e-4)
    if kind == "trk":
        assert store.list("0/vertex_attributes/") == ["fa/", "zarr.json"]
        assert store.list("0/object_attributes/") == ["length/", "n_points/", "zarr.json"]


def test_server_answers_keys_ranges_and_listings(trk):
    store = open_virtual(trk)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler({"tracts.trk": store}))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}/tracts.trk/"
    try:
        with urllib.request.urlopen(base + "zarr.json") as r:
            assert json.loads(r.read())["attributes"]["zarr_vectors"]["geometry_types"] == ["streamline"]
            assert r.headers["Access-Control-Allow-Origin"] == "*"
        cell = next(k for k in store.keys() if "/vertices/c/" in k)
        request = urllib.request.Request(base + cell, headers={"Range": "bytes=8-19"})
        with urllib.request.urlopen(request) as r:
            assert r.status == 206
            assert r.read() == store.get(cell)[8:20]
        with urllib.request.urlopen(base + "0/object_attributes/") as r:
            assert r.headers["Content-Type"] == "text/html"
            assert 'href="length/"' in r.read().decode()
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "0/vertices/c/999/999/999")
    finally:
        server.shutdown()
        server.server_close()
