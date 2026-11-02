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


def test_mesh_store_is_valid_and_keeps_every_face(tmp_path):
    from zarr_vectors.types.meshes import read_mesh
    from zarr_vectors.validate import validate

    # A cube of quads (12 triangles) and a tetrahedron, as two .obj objects, off the origin.
    cube = np.array([[x, y, z] for z in (0, 2) for y in (0, 2) for x in (0, 2)], float) - [1, 1, 1]
    quads = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]]
    tet = np.array([[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0], [0, 0, 1.5]]) + 5
    lines = ["o cube"] + [f"v {x} {y} {z}" for x, y, z in cube] + ["f " + " ".join(str(i + 1) for i in q) for q in quads]
    lines += ["o tetrahedron"] + [f"v {x} {y} {z}" for x, y, z in tet] + ["f -4 -2 -3", "f -4 -3 -1", "f -4 -1 -2", "f -3 -2 -1"]
    path = tmp_path / "two.obj"
    path.write_text("\n".join(lines) + "\n")
    store = open_virtual(path)
    assert store.count == 2
    root = tmp_path / "mesh.zarrvectors"
    _write(store, root)
    assert validate(str(root), level=3).ok
    mesh = read_mesh(str(root))
    assert mesh["face_count"] == 16 and mesh["vertex_count"] == 12
    offset = json.loads(store.get("zarr.json"))["attributes"]["zarr_vectors"]["coordinate_offset"]
    corners = np.asarray(mesh["vertices"], float) + offset
    assert sorted(map(tuple, corners.round(4))) == sorted(map(tuple, np.concatenate([cube, tet]).round(4)))


def test_swc_store_is_valid_and_keeps_every_edge(tmp_path):
    from zarr_vectors.types.graphs import read_graph
    from zarr_vectors.validate import validate

    # Two trees; the second lists a child before its root, and branches.
    swc = """# two neurons
1 1 10 10 10 3 -1
2 3 14 11 10 1 1
3 3 19 12 11 0.9 2
4 3 16 15 10 0.8 2
203 4 38 44 22 0.6 207
208 4 45 41 23 0.6 207
205 1 40 30 20 2.5 -1
207 4 41 39 22 0.8 205
"""
    path = tmp_path / "forest.swc"
    path.write_text(swc)
    store = open_virtual(path)
    assert store.count == 2
    root = tmp_path / "forest.zarrvectors"
    _write(store, root)
    assert validate(str(root), level=3).ok
    graph = read_graph(str(root))
    offset = np.asarray(json.loads(store.get("zarr.json"))["attributes"]["zarr_vectors"]["coordinate_offset"], float)
    p = np.asarray(graph["positions"], float) + offset
    got = sorted(tuple(sorted([tuple(p[a]), tuple(p[b])])) for a, b in np.asarray(graph["edges"]).reshape(-1, 2))
    nodes = {int(r[0]): (tuple(map(float, r[2:5])), int(r[6])) for r in (l.split() for l in swc.splitlines() if l and not l.startswith("#"))}
    want = sorted(tuple(sorted([pos, nodes[q][0]])) for pos, q in nodes.values() if q in nodes)
    assert got == want
    ids = np.frombuffer(store.get("0/object_attributes/segment_id/c/0"), "<u8")[:2]
    assert ids.tolist() == [1, 2]


def test_point_table_store_reads_back_with_its_columns(tmp_path):
    from zarr_vectors.types.points import read_points
    from zarr_vectors.validate import validate

    path = tmp_path / "cells.csv"
    path.write_text("id,x_um,y_um,z_um,volume,type\n1,1.5,2.5,3.5,10,neuron\n2,-4,20,9,20,glia\n3,33,17,5,30,neuron\n")
    store = open_virtual(path)
    root = tmp_path / "cells.zarrvectors"
    _write(store, root)
    assert validate(str(root), level=3).ok
    attrs = json.loads(store.get("zarr.json"))["attributes"]
    assert attrs["multiscales"][0]["axes"][0]["unit"] == "micrometer"
    assert store.list("0/vertex_attributes/") == ["id/", "volume/", "zarr.json"]
    points = read_points(str(root), attribute_names=["volume"])
    xyz = np.asarray(points["positions"], float) + attrs["zarr_vectors"]["coordinate_offset"]
    assert xyz.round(3).tolist() == [[1.5, 2.5, 3.5], [-4, 20, 9], [33, 17, 5]]
    assert np.asarray(points["vertex_attributes"]["volume"]).ravel().tolist() == [10, 20, 30]
    # Each point an object, so a region can select them.
    assert np.frombuffer(store.get("0/object_attributes/segment_id/c/0"), "<u8")[:3].tolist() == [1, 2, 3]


def test_swc_folder_is_one_store_with_each_neurons_file(tmp_path):
    from zarr_vectors.validate import validate

    for i in range(3):
        (tmp_path / f"cell_{i}.swc").write_text(f"1 1 {10 * i} 0 0 2 -1\n2 3 {10 * i + 5} 1 0 1 1\n3 3 {10 * i + 9} 2 1 1 2\n")
    (tmp_path / "broken.swc").write_text("1 1 x 0 0 1 -1\n")
    store = open_virtual(tmp_path / "@all.swc")
    assert store.count == 3
    root = tmp_path / "neurons.zarrvectors"
    _write(store, root)
    assert validate(str(root), level=3).ok
    import zarr

    assert list(zarr.open_array(str(root / "0/object_attributes/file"), mode="r")[:]) == ["cell_0.swc", "cell_1.swc", "cell_2.swc"]
