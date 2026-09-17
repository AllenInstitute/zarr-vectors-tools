"""Precomputed mesh layers come in as one mesh object per segment.

The layers are built in memory (``mem://``), since a legacy layer's
``<id>:0`` manifests cannot be named on Windows.  Each segment is a box cut
in two at a chunk face, the way a chunked meshing pipeline writes it, so the
seam's four vertices are in both fragments.  What has to hold, for the
legacy layout and both multi-resolution ones: the box comes back closed
(Euler characteristic 2) once the seam is welded, in nanometres, under its
own segment id; a coarser level of detail reads the coarser mesh; segment
properties become object attributes; and the store round-trips through the
precomputed exporter.
"""

from __future__ import annotations

import struct
import uuid
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cloudvolume")
from cloudfiles import CloudFiles  # noqa: E402
from zarr_vectors.building import (  # noqa: E402
    get_resolution_level,
    open_store,
    read_object_attributes,
)
from zarr_vectors.types.meshes import write_mesh  # noqa: E402

from zarr_vectors_tools.convert.ingest.precomputed import ingest_precomputed  # noqa: E402
from zarr_vectors_tools.convert.ingest.precomputed_meshes import (  # noqa: E402
    ingest_precomputed_meshes,
    list_mesh_segment_ids,
    weld_vertices,
)

BITS = 10
Q = 2 ** BITS - 1          # one fragment's quantisation range, and its chunk size
SCALE = np.array([4.0, 4.0, 40.0])
SEGMENTS = (7, 9, 12, 30)


def _box(xs, ys, zs):
    """A closed box surface with a ring of vertices at each of ``xs``.

    Returns ``(vertices, triangles)``: two end caps, and each side split
    into one quad per span between consecutive ``xs``.
    """
    corners = [(y, z) for y, z in ((ys[0], zs[0]), (ys[1], zs[0]), (ys[1], zs[1]), (ys[0], zs[1]))]
    vertices = np.array([(x, y, z) for x in xs for y, z in corners], dtype=np.float64)
    ring = lambda i: [4 * i + k for k in range(4)]  # noqa: E731
    tris = []
    first, last = ring(0), ring(len(xs) - 1)
    tris += [[first[0], first[2], first[1]], [first[0], first[3], first[2]]]
    tris += [[last[0], last[1], last[2]], [last[0], last[2], last[3]]]
    for i in range(len(xs) - 1):
        a, b = ring(i), ring(i + 1)
        for k in range(4):
            k1 = (k + 1) % 4
            tris += [[a[k], a[k1], b[k1]], [a[k], b[k1], b[k]]]
    return vertices, np.array(tris, dtype=np.int64)


def _split(vertices, faces, x_seam):
    """Two fragments, left and right of ``x_seam``, each with its own vertices."""
    out = []
    for side in (np.all(vertices[faces][:, :, 0] <= x_seam, axis=1),
                 np.all(vertices[faces][:, :, 0] >= x_seam, axis=1)):
        used, local = np.unique(faces[side], return_inverse=True)
        out.append((vertices[used], local.reshape(-1, 3)))
    return out


def _edges(faces):
    """``(edge, faces using it)`` for each distinct edge."""
    return np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
                                             faces[:, [2, 0]]]), axis=1),
                     axis=0, return_counts=True)


def _euler(vertices, faces):
    return len(vertices) - len(_edges(faces)[0]) + len(faces)


def _open_edges(faces):
    return int(np.sum(_edges(faces)[1] == 1))


def _stored_box(segment):
    """Stored-model box for ``segment``: it straddles the x = Q chunk face."""
    shift = 10 * SEGMENTS.index(segment)
    return _box((500, Q, 1523), (100 + shift, 400 + shift), (100, 300))


# ===================================================================
# Layer builders
# ===================================================================

def _legacy_layer(properties=None) -> str:
    url = f"mem://meshes/{uuid.uuid4().hex}/mesh"
    cf = CloudFiles(url)
    info = {"@type": "neuroglancer_legacy_mesh"}
    for segment in SEGMENTS:
        vertices, faces = _stored_box(segment)
        names = []
        for i, (v, f) in enumerate(_split(vertices * SCALE, faces, Q * SCALE[0])):
            name = f"{segment}:0:{i}"
            cf.put(name, struct.pack("<I", len(v)) + v.astype("<f4").tobytes()
                   + f.astype("<u4").tobytes(), compress=None)
            names.append(name)
        cf.put_json(f"{segment}:0", {"fragments": names})
    if properties:
        info["segment_properties"] = "segment_properties"
        cf.put_json("segment_properties/info", {
            "@type": "neuroglancer_segment_properties", "inline": properties,
        })
    cf.put_json("info", info)
    return url


def _draco(vertices, faces):
    import DracoPy

    return DracoPy.encode(
        vertices.astype(np.float32), faces.astype(np.uint32), quantization_bits=BITS,
        quantization_range=Q, quantization_origin=[0, 0, 0], preserve_order=True,
    )


def _multilod_segment(segment):
    """``(fragment data, manifest)``: lod 0 in two fragments, lod 1 in one."""
    vertices, faces = _stored_box(segment)
    (left_v, left_f), (right_v, right_f) = _split(vertices, faces, Q)
    right_v = right_v - [Q, 0, 0]              # local to grid cell (1, 0, 0)
    coarse_v, coarse_f = _box((250, 761), (50, 200), (50, 150))  # lod 1: cell 2Q
    fragments = [_draco(left_v, left_f), _draco(right_v, right_f), _draco(coarse_v, coarse_f)]
    positions = [np.array([[0, 0, 0], [1, 0, 0]]), np.array([[0, 0, 0]])]
    sizes = [[len(fragments[0]), len(fragments[1])], [len(fragments[2])]]
    manifest = (
        np.array([Q, Q, Q, 0, 0, 0], "<f4").tobytes() + np.array([2], "<u4").tobytes()
        + np.array([1, 2], "<f4").tobytes() + np.zeros(6, "<f4").tobytes()
        + np.array([2, 1], "<u4").tobytes()
        + b"".join(np.asarray(p, "<u4").tobytes(order="F") + np.asarray(s, "<u4").tobytes()
                   for p, s in zip(positions, sizes))
    )
    return b"".join(fragments), manifest


def _multilod_info(**extra):
    return {
        "@type": "neuroglancer_multilod_draco", "vertex_quantization_bits": BITS,
        "transform": [SCALE[0], 0, 0, 0, 0, SCALE[1], 0, 0, 0, 0, SCALE[2], 0],
        "lod_scale_multiplier": 1.0, **extra,
    }


def _multilod_layer() -> str:
    url = f"mem://meshes/{uuid.uuid4().hex}/mesh"
    cf = CloudFiles(url)
    for segment in SEGMENTS:
        data, manifest = _multilod_segment(segment)
        cf.put(str(segment), data, compress=None)
        cf.put(f"{segment}.index", manifest, compress=None)
    cf.put_json("info", _multilod_info())
    return url


def _sharded_layer() -> str:
    from cloudvolume.datasource.precomputed.sharding import (
        ShardingSpecification,
        synthesize_shard_files,
    )

    spec = ShardingSpecification(
        type="neuroglancer_uint64_sharded_v1", preshift_bits=0, hash="identity",
        minishard_bits=1, shard_bits=1, minishard_index_encoding="raw",
        data_encoding="raw",
    )
    data, manifest_sizes = {}, {}
    for segment in SEGMENTS:
        fragments, manifest = _multilod_segment(segment)
        data[segment] = fragments + manifest
        manifest_sizes[segment] = len(manifest)
    url = f"mem://meshes/{uuid.uuid4().hex}/mesh"
    cf = CloudFiles(url)
    for name, blob in synthesize_shard_files(spec, data, manifest_sizes).items():
        cf.put(name, blob, compress=None)
    cf.put_json("info", _multilod_info(sharding=spec.to_dict()))
    return url


LAYERS = {"legacy": _legacy_layer, "multilod": _multilod_layer, "multilod_sharded": _sharded_layer}


def _objects(store: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """``{segment id: (vertices, faces)}`` for every object at level 0.

    Split by the exporter's object reader: core's ``read_mesh`` does not
    filter by object.
    """
    from zarr_vectors.building import read_all_object_manifests

    from zarr_vectors_tools.convert.export.precomputed import (
        _coordinate_offset,
        _manifest_meshes,
    )

    root = open_store(str(store))
    level0 = get_resolution_level(root, 0)
    segments = np.asarray(read_object_attributes(level0, "segment_id"), dtype=np.int64)
    pairs = list(enumerate(segments.tolist()))
    return {
        segment: (vertices, faces)
        for _oid, segment, vertices, faces in _manifest_meshes(
            store, level0, 0, pairs, True, read_all_object_manifests(level0),
            _coordinate_offset(root),
        )
    }


def _same_rows(a, b):
    np.testing.assert_allclose(np.unique(np.asarray(a, float).round(3), axis=0),
                               np.unique(np.asarray(b, float).round(3), axis=0))


# ===================================================================
# Reading each layout
# ===================================================================

@pytest.mark.parametrize("layout", sorted(LAYERS))
def test_each_segment_is_a_closed_box_under_its_own_id(layout: str, tmp_path: Path) -> None:
    url = LAYERS[layout]()
    assert list_mesh_segment_ids(url, CloudFiles(url).get_json("info")) == list(SEGMENTS)

    store = tmp_path / "meshes.zv"
    summary = ingest_precomputed_meshes(url, store, (8192.0, 8192.0, 8192.0), progress=False)
    assert summary["layout"] == layout
    assert summary["object_count"] == len(SEGMENTS)
    assert summary["vertices_welded"] == 4 * len(SEGMENTS)
    assert summary["empty_segments"] == []

    objects = _objects(store)
    assert sorted(objects) == list(SEGMENTS)
    for segment, (vertices, faces) in objects.items():
        expected, _faces = _stored_box(segment)
        assert (len(vertices), len(faces)) == (12, 20)
        assert _euler(vertices, faces) == 2 and _open_edges(faces) == 0
        _same_rows(vertices, expected * SCALE)


def test_without_welding_the_seam_stays_open(tmp_path: Path) -> None:
    store = tmp_path / "open.zv"
    summary = ingest_precomputed_meshes(
        _legacy_layer(), store, (8192.0,) * 3, weld=False, progress=False,
    )
    assert summary["vertices_welded"] == 0
    vertices, faces = _objects(store)[SEGMENTS[0]]
    # Two open halves, each with a four-edge rim along the seam.
    assert len(vertices) == 16 and _open_edges(faces) == 8


@pytest.mark.parametrize("layout", ["multilod", "multilod_sharded"])
def test_a_coarser_level_of_detail(layout: str, tmp_path: Path) -> None:
    url = LAYERS[layout]()
    store = tmp_path / "lod1.zv"
    ingest_precomputed_meshes(url, store, (8192.0,) * 3, lod=1, segment_ids=[9, 30],
                              progress=False)
    objects = _objects(store)
    assert sorted(objects) == [9, 30]
    vertices, faces = objects[9]
    expected, _faces = _box((250, 761), (50, 200), (50, 150))
    # lod 1 cells are twice lod 0's: stored = 2Q * (0 + q / Q) = 2q.
    _same_rows(vertices, expected * 2 * SCALE)
    assert _euler(vertices, faces) == 2

    with pytest.raises(ValueError, match="lod 2"):
        ingest_precomputed_meshes(url, tmp_path / "lod2.zv", (8192.0,) * 3, lod=2,
                                  progress=False)


@pytest.mark.parametrize("layout", sorted(LAYERS))
def test_a_segment_the_layer_lacks_is_refused(layout: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no mesh for segment id"):
        ingest_precomputed_meshes(LAYERS[layout](), tmp_path / "x.zv", (8192.0,) * 3,
                                  segment_ids=[9, 404], progress=False)


def test_a_legacy_layer_has_one_level_of_detail(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only lod 0"):
        ingest_precomputed_meshes(_legacy_layer(), tmp_path / "x.zv", (8192.0,) * 3,
                                  lod=1, progress=False)


def test_segment_properties_become_object_attributes(tmp_path: Path) -> None:
    url = _legacy_layer(properties={
        "ids": ["30", "7", "12", "9", "99"],
        "properties": [
            {"id": "label", "type": "label", "values": ["d", "a", "c", "b", "zz"]},
            {"id": "volume", "type": "number", "data_type": "float32",
             "values": [4.0, 1.0, 3.0, 2.0, 9.0]},
        ],
    })
    store = tmp_path / "props.zv"
    summary = ingest_precomputed_meshes(url, store, (8192.0,) * 3, segment_ids=[7, 12, 30],
                                        batch_size=1, read_workers=3, progress=False)
    assert summary["properties"] == ["label", "volume"]
    level0 = get_resolution_level(open_store(str(store)), 0)
    assert read_object_attributes(level0, "segment_id").tolist() == [7, 12, 30]
    labels = [bytes(v).rstrip(b"\0").decode() for v in read_object_attributes(level0, "label")]
    assert labels == ["a", "c", "d"]
    np.testing.assert_allclose(read_object_attributes(level0, "volume"), [1.0, 3.0, 4.0])


def test_weld_drops_collapsed_and_repeated_faces() -> None:
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [5, 5, 5]], float)
    faces = np.array([[0, 1, 2], [0, 3, 2], [2, 1, 0], [0, 1, 3]])
    welded, out, dropped = weld_vertices(vertices, faces)
    assert out.tolist() == [[0, 1, 2]] and dropped == 3
    np.testing.assert_allclose(welded, vertices[:3])


# ===================================================================
# Round trip, pyramid, dispatch and CLI
# ===================================================================

def test_a_store_round_trips_through_the_precomputed_exporter(tmp_path: Path) -> None:
    from zarr_vectors_tools.convert.export.precomputed import export_precomputed_meshes

    parts = [_box((0, 50, 90), (i * 100, i * 100 + 40), (0, 30)) for i in range(3)]
    offsets = np.cumsum([0] + [len(v) for v, _f in parts[:-1]])
    source = tmp_path / "source.zv"
    write_mesh(
        str(source),
        np.concatenate([v for v, _f in parts]).astype(np.float32),
        np.concatenate([f + o for (_v, f), o in zip(parts, offsets)]),
        chunk_shape=(64.0, 64.0, 64.0),
        object_ids=np.repeat(np.arange(3), [len(v) for v, _f in parts]),
    )
    layer = f"mem://roundtrip/{uuid.uuid4().hex}"
    exported = export_precomputed_meshes(source, layer, unit="nanometer")

    back = tmp_path / "back.zv"
    ingest_precomputed_meshes(f"{layer}/{exported['directory']}", back, (64.0,) * 3,
                              progress=False)
    objects = _objects(back)
    assert sorted(objects) == exported["segment_ids"]
    for (vertices, faces), segment in zip(parts, exported["segment_ids"]):
        got_vertices, got_faces = objects[segment]
        _same_rows(got_vertices, vertices)
        assert len(got_faces) == len(faces)


def test_the_store_takes_a_mesh_pyramid(tmp_path: Path) -> None:
    from zarr_vectors_tools.multiresolution.coarsen import build_pyramid

    store = tmp_path / "pyramid.zv"
    ingest_precomputed_meshes(_multilod_layer(), store, (2048.0,) * 3, progress=False)
    result = build_pyramid(str(store), factors=[(2.0, 1.0)])
    assert result["levels_created"] == 1
    level1 = get_resolution_level(open_store(str(store)), 1)
    assert level1 is not None


def test_the_precomputed_dispatcher_reads_mesh_layers(tmp_path: Path) -> None:
    url = _multilod_layer()
    summary = ingest_precomputed(url, tmp_path / "d.zv", (8192.0,) * 3, segment_ids=[12],
                                 progress=False)
    assert summary["layout"] == "multilod" and summary["object_count"] == 1

    with pytest.raises(ValueError, match="strides, drop_interior_below do not apply"):
        ingest_precomputed(url, tmp_path / "e.zv", (8192.0,) * 3, strides=[2],
                           drop_interior_below=3)
    with pytest.raises(ValueError, match="pass chunk_shape"):
        ingest_precomputed(url, tmp_path / "f.zv")


def test_a_segmentation_volume_names_its_mesh_directory(tmp_path: Path) -> None:
    root = f"mem://volumes/{uuid.uuid4().hex}"
    CloudFiles(root).put_json("info", {
        "@type": "neuroglancer_multiscale_volume", "type": "segmentation",
        "scales": [], "mesh": "mesh_mip_2_err_40",
    })
    with pytest.raises(ValueError, match="its meshes are at .*mesh_mip_2_err_40"):
        ingest_precomputed(root, tmp_path / "v.zv", (8192.0,) * 3)


class TestCli:

    def test_convert_a_mesh_layer_with_a_pyramid(self, tmp_path: Path, capsys) -> None:
        from zarr_vectors_tools.cli import main

        out = tmp_path / "cli.zv"
        assert main([
            "convert", _sharded_layer(), str(out), "--chunk-shape", "2048,2048,2048",
            "--segment-id", "7", "--segment-id", "30", "--lod", "0",
            "--coarsen", "2", "--sparsity", "1",
        ]) == 0
        printed = capsys.readouterr().out
        assert "ingested precomputed (mesh)" in printed
        assert "layout: multilod_sharded" in printed
        assert sorted(_objects(out)) == [7, 30]
        assert get_resolution_level(open_store(str(out)), 1) is not None

    def test_skeleton_only_flags_are_refused(self, tmp_path: Path) -> None:
        from zarr_vectors_tools.cli import main

        with pytest.raises(SystemExit, match="--anchor does not apply to a precomputed mesh"):
            main(["convert", _multilod_layer(), str(tmp_path / "x.zv"),
                  "--chunk-shape", "8192,8192,8192", "--anchor", "0,0,0"])

    def test_chunk_shape_is_required(self, tmp_path: Path) -> None:
        from zarr_vectors_tools.cli import main

        with pytest.raises(SystemExit, match="--chunk-shape"):
            main(["convert", _legacy_layer(), str(tmp_path / "x.zv")])

    def test_lod_is_refused_for_other_formats(self, tmp_path: Path) -> None:
        from zarr_vectors_tools.cli import main

        source = tmp_path / "p.csv"
        source.write_text("x,y,z\n1,2,3\n")
        with pytest.raises(SystemExit, match="--lod applies to precomputed"):
            main(["convert", str(source), str(tmp_path / "o.zv"),
                  "--chunk-shape", "5,5,5", "--lod", "1"])
