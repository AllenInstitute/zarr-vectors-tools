"""Skeleton and mesh stores go back out as Neuroglancer precomputed layers.

What matters is that the layer is the one Neuroglancer, CloudVolume and igneous
read, so every check reads it back with cloud-volume itself:
``CloudVolume(layer).skeleton.get(id)`` and ``.mesh.get(id)`` must return the
object the store holds, at level 0 and at a coarser level.  The oracle for a
coarse level is the store read independently of the exporter's own splitting
(core's per-segment reader, or an object's manifest vertices).

Skeleton layers are written to local directories.  Mesh layers go to
cloud-files' in-memory ``mem://`` backend, because a legacy mesh manifest is
named ``<id>:0`` and Windows cannot hold that as a file; one test covers the
local directory on each platform.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest
from zarr_vectors.building import (
    get_resolution_level,
    open_store,
    read_object_vertices,
    write_object_attributes,
)
from zarr_vectors.exceptions import ExportError
from zarr_vectors.types import skeletons as sk
from zarr_vectors.types.graphs import read_graph, write_graph
from zarr_vectors.types.meshes import read_mesh, write_mesh

from zarr_vectors_tools.convert.export.precomputed import (
    export_precomputed,
    export_precomputed_meshes,
    export_precomputed_skeletons,
)
from zarr_vectors_tools.convert.ingest.precomputed import layer_url

pytest.importorskip("cloudvolume")


def _volume(layer: str | Path):
    from cloudvolume import CloudVolume

    return CloudVolume(layer_url(layer))


def _mem_layer() -> str:
    return f"mem://zv-precomputed-export/{uuid.uuid4().hex}"


def _json(layer: str | Path, path: str):
    from cloudfiles import CloudFiles

    return CloudFiles(layer_url(layer)).get_json(path)


def _keys(values) -> list[tuple[float, ...]]:
    """Positions as hashable rows, rounded past float32 noise."""
    return [tuple(p) for p in np.round(np.asarray(values, dtype=np.float64), 2).tolist()]


def _points(values) -> set[tuple[float, ...]]:
    return set(_keys(values))


def _segments(positions, edges) -> set[frozenset]:
    """Edges as unordered coordinate pairs, independent of vertex numbering."""
    p = np.round(np.asarray(positions, dtype=np.float64), 2)
    return {frozenset((tuple(p[a]), tuple(p[b]))) for a, b in np.asarray(edges).reshape(-1, 2)}


def _triangles(vertices, faces) -> set[tuple]:
    """Triangles as coordinate triples, rotated to a canonical start.

    A rotation keeps the winding, so a flipped triangle would not match.
    """
    v = np.round(np.asarray(vertices, dtype=np.float64), 2)
    out = set()
    for face in np.asarray(faces).reshape(-1, 3):
        corners = [tuple(v[i]) for i in face]
        start = corners.index(min(corners))
        out.add(tuple(corners[start:] + corners[:start]))
    return out


# ===================================================================
# EM skeleton stores (written by the precomputed ingesters)
# ===================================================================

#: A straight neurite crossing the x = 573440 nm chunk boundary, so at level
#: 0 it is two fragments joined only by a cross-chunk link, and a small
#: branching tree inside one chunk.
LINE = 720575940000000123
TREE = 720575940000000456
CHUNK_NM = (16384.0, 16384.0, 20480.0)


def _em_reader(segments: dict[int, dict]):
    from zarr_vectors_tools.convert.ingest.precomputed_skeletons import (
        InMemoryFragsReader,
        SkeletonInfo,
    )

    info = SkeletonInfo(
        base_url="mem://x", resolution_nm=(32.0, 32.0, 40.0), chunk_size_nm=CHUNK_NM,
        vertex_attributes=[{"id": "radius", "data_type": "float32", "num_components": 1}],
    )
    return InMemoryFragsReader(info, {"17910-18422_10448-10960_3088-3600.frags": segments})


def _em_segments() -> dict[int, dict]:
    xs = np.arange(573300.0, 573600.0, 20.0)
    line = np.stack([xs, np.full_like(xs, 335000.0), np.full_like(xs, 130000.0)], 1)
    tree = np.array([
        [574000, 336000, 131000], [574100, 336000, 131000],
        [574200, 336100, 131000], [574200, 335900, 131000],
    ])
    return {
        LINE: {
            "vertices": line.astype(np.float32),
            "edges": np.array([[i, i - 1] for i in range(1, len(xs))]),
            "radius": np.linspace(1.0, 2.0, len(xs)).astype(np.float32),
        },
        TREE: {
            "vertices": tree.astype(np.float32),
            "edges": np.array([[1, 0], [2, 1], [3, 1]]),
            "radius": np.array([5, 6, 7, 8], dtype=np.float32),
        },
    }


def _ingest_em(path: Path, segments: dict[int, dict]) -> str:
    from zarr_vectors_tools.convert.ingest.precomputed_skeletons import run_ingest

    store = str(path)
    run_ingest(
        _em_reader(segments), store, ["17910-18422_10448-10960_3088-3600.frags"],
        bounds_nm=([573120.0, 334336.0, 123520.0], [589504.0, 350720.0, 144000.0]),
        strides=[2], chunk_scale_factors=[2], sparsity_factors=[1.0],
        align=False, progress=False,
    )
    return store


@pytest.fixture
def em_store(tmp_path: Path) -> str:
    store = _ingest_em(tmp_path / "em.zv", _em_segments())
    # Objects are numbered in segment-id order, so LINE is object 0.
    level0 = get_resolution_level(open_store(store, mode="r+"), 0)
    write_object_attributes(level0, "label", np.array([b"line", b"tree"], dtype="S8"))
    write_object_attributes(level0, "length_nm", np.array([280.0, 300.0]))
    return store


class TestEmSkeletons:

    @pytest.mark.parametrize("level", [0, 1])
    def test_each_segment_reads_back_as_the_store_holds_it(
        self, em_store: str, tmp_path: Path, level: int,
    ) -> None:
        layer = tmp_path / f"layer{level}"
        summary = export_precomputed(em_store, layer, level=level)
        assert summary["segment_ids"] == [LINE, TREE]
        assert summary["attributes_carried"] == ["radius"]

        volume = _volume(layer)
        for segment in (LINE, TREE):
            stored = sk.read_skeleton_by_segment_id(em_store, segment, level=level)
            skeleton = volume.skeleton.get(segment)
            assert np.allclose(skeleton.vertices, stored["positions"])
            assert np.allclose(skeleton.radius, stored["attributes"]["radius"])
            # Every edge the store's reader returns is there, and each skeleton
            # is one tree: the line's cross-chunk edge, which that reader does
            # not return at level 0, has been added back.
            assert _segments(stored["positions"], stored["edges"]) <= _segments(
                skeleton.vertices, skeleton.edges,
            )
            assert len(skeleton.edges) == len(skeleton.vertices) - 1
        assert len(volume.skeleton.get(LINE).vertices) == (15 if level == 0 else 8)

    def test_level_zero_matches_the_source_neurons(self, em_store: str, tmp_path: Path) -> None:
        export_precomputed(em_store, tmp_path / "layer")
        volume = _volume(tmp_path / "layer")
        for segment, source in _em_segments().items():
            skeleton = volume.skeleton.get(segment)
            assert _points(skeleton.vertices) == _points(source["vertices"])
            assert _segments(skeleton.vertices, skeleton.edges) == _segments(
                source["vertices"], source["edges"],
            )

    def test_segment_properties_come_from_the_object_attributes(
        self, em_store: str, tmp_path: Path,
    ) -> None:
        layer = tmp_path / "layer"
        summary = export_precomputed(em_store, layer, level=1)
        assert summary["properties"] == ["label", "length_nm"]

        info = _json(layer, "info")
        assert info["type"] == "segmentation"
        assert info["skeletons"] == "skeletons"
        assert "placeholder" in info["scales"][0]["key"]
        skeleton_info = _json(layer, "skeletons/info")
        assert skeleton_info["@type"] == "neuroglancer_skeletons"
        properties = _json(layer, f"skeletons/{skeleton_info['segment_properties']}/info")
        inline = properties["inline"]
        assert inline["ids"] == [str(LINE), str(TREE)]
        by_id = {p["id"]: p for p in inline["properties"]}
        assert by_id["label"] == {"id": "label", "type": "label", "values": ["line", "tree"]}
        assert by_id["length_nm"]["type"] == "number"
        assert by_id["length_nm"]["values"] == [280.0, 300.0]

    def test_the_layer_ingests_back_into_the_same_store(
        self, em_store: str, tmp_path: Path,
    ) -> None:
        from zarr_vectors.building import read_object_attributes

        from zarr_vectors_tools.convert.ingest.precomputed import ingest_precomputed

        layer = tmp_path / "layer"
        export_precomputed(em_store, layer)
        back = tmp_path / "back.zv"
        ingest_precomputed(layer / "skeletons", back, chunk_shape=CHUNK_NM, progress=False)
        for segment in (LINE, TREE):
            before = sk.read_skeleton_by_segment_id(em_store, segment)
            after = sk.read_skeleton_by_segment_id(str(back), segment)
            assert _points(after["positions"]) == _points(before["positions"])
            assert "radius" in after["attributes"]
        level0 = get_resolution_level(open_store(str(back)), 0)
        labels = read_object_attributes(level0, "label")
        assert [v.decode().rstrip("\x00") for v in labels.tolist()] == ["line", "tree"]

    def test_segment_ids_select_segments(self, em_store: str, tmp_path: Path) -> None:
        layer = tmp_path / "one"
        summary = export_precomputed(em_store, layer, segment_ids=[TREE])
        assert summary["segment_ids"] == [TREE]
        assert _json(layer, "skeletons/segment_properties/info")["inline"]["ids"] == [str(TREE)]
        assert not (layer / "skeletons" / str(LINE)).exists()

    def test_object_ids_index_the_level(self, em_store: str, tmp_path: Path) -> None:
        summary = export_precomputed(em_store, tmp_path / "one", object_ids=[1])
        assert summary["segment_ids"] == [TREE]

    def test_segment_zero_is_refused(self, tmp_path: Path) -> None:
        segments = _em_segments()
        segments[0] = segments.pop(TREE)
        store = _ingest_em(tmp_path / "zero.zv", segments)
        with pytest.raises(ExportError, match="background"):
            export_precomputed(store, tmp_path / "layer")


# ===================================================================
# Graph-kind skeleton stores (no segment ids)
# ===================================================================

def _random_trees() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two branching trees far apart, each spanning several 8-unit chunks."""
    rng = np.random.default_rng(3)
    positions, edges, object_ids = [], [], []
    offset = 0
    for oid, start in enumerate(([2.0, 2.0, 2.0], [60.0, 60.0, 20.0])):
        n = 40
        positions.append(np.cumsum(rng.normal(0, 1.5, (n, 3)), axis=0) + start)
        parents = [max(0, i - 1 - 3 * (i % 5 == 0)) for i in range(1, n)]
        edges.append(np.stack([np.arange(1, n), parents], axis=1) + offset)
        object_ids.append(np.full(n, oid))
        offset += n
    return (
        np.concatenate(positions).astype(np.float32),
        np.concatenate(edges).astype(np.int64),
        np.concatenate(object_ids),
    )


@pytest.fixture
def graph_store(tmp_path: Path):
    positions, edges, object_ids = _random_trees()
    radius = np.linspace(1.0, 3.0, len(positions)).astype(np.float32)
    store = str(tmp_path / "trees.zv")
    write_graph(
        store, positions, edges, chunk_shape=(8.0, 8.0, 8.0), kind="skeleton",
        object_ids=object_ids, vertex_attributes={"radius": radius},
    )
    return store, positions, edges, object_ids, radius


class TestGraphSkeletons:

    def test_objects_are_segments_numbered_from_one(self, graph_store, tmp_path: Path) -> None:
        store, positions, edges, object_ids, radius = graph_store
        layer = tmp_path / "layer"
        summary = export_precomputed_skeletons(store, layer)
        assert summary["segment_ids"] == [1, 2]
        assert summary["unit_assumed"] is True

        volume = _volume(layer)
        for oid in (0, 1):
            mine = np.flatnonzero(object_ids == oid)
            local = {int(g): i for i, g in enumerate(mine)}
            own_edges = [(local[a], local[b]) for a, b in edges if a in local]
            skeleton = volume.skeleton.get(oid + 1)
            assert _points(skeleton.vertices) == _points(positions[mine])
            assert _segments(skeleton.vertices, skeleton.edges) == _segments(
                positions[mine], own_edges,
            )
            expected = dict(zip(_keys(positions[mine]), radius[mine]))
            for vertex, r in zip(_keys(skeleton.vertices), skeleton.radius):
                assert r == pytest.approx(expected[vertex])

    def test_a_coarse_level(self, graph_store, tmp_path: Path) -> None:
        from zarr_vectors_tools.multiresolution.coarsen import build_pyramid

        store = graph_store[0]
        build_pyramid(store, factors=[(2.0, 1.0)], chunk_scale_factors=[2], method="per_object")
        layer = tmp_path / "coarse"
        summary = export_precomputed(store, layer, level=1)
        assert summary["segment_ids"] == [1, 2]

        coarse = read_graph(store, level=1)
        level1 = get_resolution_level(open_store(store), 1)
        volume = _volume(layer)
        for oid in (0, 1):
            own = _points(np.concatenate(read_object_vertices(level1, oid)))
            keep = [
                (a, b) for a, b in coarse["edges"]
                if _points(coarse["positions"][[a, b]]) <= own
            ]
            skeleton = volume.skeleton.get(oid + 1)
            assert _points(skeleton.vertices) == own
            assert _segments(skeleton.vertices, skeleton.edges) == _segments(
                coarse["positions"], keep,
            )

    def test_attribute_widths_and_types(self, tmp_path: Path) -> None:
        store = str(tmp_path / "wide.zv")
        positions = np.array([[1, 1, 1], [12, 1, 1], [14, 1, 1]], dtype=np.float32)
        normal = np.arange(9, dtype=np.float32).reshape(3, 3)
        write_graph(
            store, positions, np.array([[1, 0], [2, 1]]), chunk_shape=(10.0,) * 3,
            kind="skeleton",
            vertex_attributes={"normal": normal, "count": np.array([1, 2, 3], dtype=np.int64)},
        )
        layer = tmp_path / "layer"
        summary = export_precomputed(store, layer)
        # A (V, 3) attribute keeps its three components; an int64 one has no
        # precomputed type and is left out unless asked for.
        assert summary["attributes_carried"] == ["normal"]
        assert summary["attributes_skipped"] == ["count"]
        assert _json(layer, "skeletons/info")["vertex_attributes"] == [
            {"id": "normal", "data_type": "float32", "num_components": 3},
        ]
        skeleton = _volume(layer).skeleton.get(1)
        by_vertex = dict(zip(_keys(positions), normal.tolist()))
        for vertex, value in zip(_keys(skeleton.vertices), skeleton.normal.tolist()):
            assert value == by_vertex[vertex]
        with pytest.raises(ExportError, match="int64"):
            export_precomputed(store, tmp_path / "named", attribute_names=["count"])

    def test_unit_scales_to_nanometres(self, graph_store, tmp_path: Path) -> None:
        store, positions, _edges, object_ids, _radius = graph_store
        summary = export_precomputed(store, tmp_path / "um", unit="micrometer")
        assert summary["nm_per_unit"] == 1000.0
        skeleton = _volume(tmp_path / "um").skeleton.get(2)
        assert _points(skeleton.vertices) == _points(positions[object_ids == 1] * 1000.0)

    def test_a_second_export_adds_to_the_layer(self, graph_store, tmp_path: Path) -> None:
        store = graph_store[0]
        level0 = get_resolution_level(open_store(store, mode="r+"), 0)
        write_object_attributes(level0, "label", np.array([b"first", b"second"], dtype="S8"))
        layer = tmp_path / "layer"
        export_precomputed(store, layer, object_ids=[1])
        summary = export_precomputed(store, layer, object_ids=[0])
        assert summary["properties"] == ["label"]

        inline = _json(layer, "skeletons/segment_properties/info")["inline"]
        assert inline["ids"] == ["2", "1"]
        assert inline["properties"][0]["values"] == ["second", "first"]
        volume = _volume(layer)
        assert len(volume.skeleton.get(1).vertices) == 40
        assert len(volume.skeleton.get(2).vertices) == 40

    @pytest.mark.parametrize("kwargs, message", [
        ({"object_ids": [9]}, "not at level 0"),
        ({"segment_ids": [99]}, "numbered object id \\+ 1"),
        ({"object_ids": [0], "segment_ids": [1]}, "not both"),
        ({"attribute_names": ["thickness"]}, "not at this level"),
        ({"object_attribute_names": ["label"]}, "not at level 0"),
        ({"unit": "furlong"}, "not a length unit"),
        ({"level": 3}, "no level 3"),
    ])
    def test_refusals(self, graph_store, tmp_path: Path, kwargs, message) -> None:
        with pytest.raises(ExportError, match=message):
            export_precomputed(graph_store[0], tmp_path / "layer", **kwargs)

    def test_a_layer_with_other_attributes_is_refused(self, graph_store, tmp_path: Path) -> None:
        layer = tmp_path / "layer"
        export_precomputed(graph_store[0], layer, object_ids=[0])
        with pytest.raises(ExportError, match="carries vertex attributes"):
            export_precomputed(graph_store[0], layer, object_ids=[1], attribute_names=[])

    def test_a_layer_that_is_not_a_segmentation_is_refused(
        self, graph_store, tmp_path: Path,
    ) -> None:
        layer = tmp_path / "skeletons_only"
        layer.mkdir()
        (layer / "info").write_text(json.dumps({"@type": "neuroglancer_skeletons"}))
        with pytest.raises(ExportError, match="not a segmentation layer"):
            export_precomputed(graph_store[0], layer)


# ===================================================================
# Mesh stores
# ===================================================================

def _grid_surfaces(n_obj: int = 3, per_side: int = 10, spacing: float = 90.0):
    """Open triangulated sheets far apart, each crossing several chunks."""
    rng = np.random.default_rng(0)
    vertices, faces, object_ids = [], [], []
    offset = 0
    for oid in range(n_obj):
        gx, gy = np.meshgrid(np.arange(per_side), np.arange(per_side))
        x = gx.ravel() * spacing + oid * 1900.0
        y = gy.ravel() * spacing + (oid % 2) * 1500.0
        z = 500.0 + rng.normal(0, 40.0, x.size)
        vertices.append(np.stack([x, y, z], axis=1))
        idx = np.arange(per_side * per_side).reshape(per_side, per_side)
        a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
        c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
        faces.append(np.concatenate([np.stack([a, b, c], 1), np.stack([b, d, c], 1)]) + offset)
        object_ids.append(np.full(per_side * per_side, oid))
        offset += per_side * per_side
    return (
        np.concatenate(vertices).astype(np.float32),
        np.concatenate(faces).astype(np.int64),
        np.concatenate(object_ids),
    )


@pytest.fixture
def mesh_store(tmp_path: Path):
    vertices, faces, object_ids = _grid_surfaces()
    store = str(tmp_path / "sheets.zv")
    write_mesh(
        store, vertices, faces, chunk_shape=(400.0, 400.0, 400.0), object_ids=object_ids,
        object_attributes={"area": np.array([1.5, 2.5, 3.5])},
    )
    return store, vertices, faces, object_ids


class TestMeshes:

    def test_each_object_reads_back_as_written(self, mesh_store) -> None:
        store, vertices, faces, object_ids = mesh_store
        layer = _mem_layer()
        summary = export_precomputed(store, layer)
        assert summary["segment_ids"] == [1, 2, 3]
        assert summary["face_count"] == len(faces)

        volume = _volume(layer)
        for oid in range(3):
            own = object_ids[faces[:, 0]] == oid
            mesh = volume.mesh.get(oid + 1, remove_duplicate_vertices=False)
            assert _points(mesh.vertices) == _points(vertices[object_ids == oid])
            assert _triangles(mesh.vertices, mesh.faces) == _triangles(vertices, faces[own])
        properties = _json(layer, "mesh/segment_properties/info")["inline"]
        assert properties["ids"] == ["1", "2", "3"]
        assert properties["properties"][0]["values"] == [1.5, 2.5, 3.5]

    def test_a_coarse_level(self, mesh_store) -> None:
        from zarr_vectors_tools.multiresolution.coarsen import build_pyramid

        store = mesh_store[0]
        build_pyramid(store, factors=[(2.0, 1.0)], chunk_scale_factors=[2])
        layer = _mem_layer()
        export_precomputed_meshes(store, layer, level=1)

        coarse = read_mesh(store, level=1)
        level1 = get_resolution_level(open_store(store), 1)
        volume = _volume(layer)
        for oid in range(3):
            own = _points(np.concatenate(read_object_vertices(level1, oid)))
            keep = [f for f in coarse["faces"] if _points(coarse["vertices"][f]) <= own]
            mesh = volume.mesh.get(oid + 1, remove_duplicate_vertices=False)
            assert _points(mesh.vertices) == own
            assert len(keep) > 0
            assert _triangles(mesh.vertices, mesh.faces) == _triangles(coarse["vertices"], keep)

    def test_quads_are_triangulated(self, tmp_path: Path) -> None:
        store = str(tmp_path / "quad.zv")
        vertices = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], dtype=np.float32)
        write_mesh(store, vertices, np.array([[0, 1, 2, 3]]), chunk_shape=(100.0,) * 3)
        layer = _mem_layer()
        assert export_precomputed(store, layer)["face_count"] == 2
        mesh = _volume(layer).mesh.get(1, remove_duplicate_vertices=False)
        assert _triangles(mesh.vertices, mesh.faces) == _triangles(
            vertices, [[0, 1, 2], [0, 2, 3]],
        )

    def test_obj_object_names_become_labels(self, tmp_path: Path) -> None:
        from zarr_vectors_tools.convert.ingest.obj import ingest_obj

        source = tmp_path / "two.obj"
        source.write_text(
            "o first\nv 0 0 0\nv 10 0 0\nv 0 10 0\nf 1 2 3\n"
            "o second\nv 50 50 50\nv 60 50 50\nv 50 60 50\nf 4 5 6\n"
        )
        store = str(tmp_path / "two.zv")
        ingest_obj(source, store, (100.0, 100.0, 100.0), auto_object_id=True)
        layer = _mem_layer()
        export_precomputed(store, layer)
        inline = _json(layer, "mesh/segment_properties/info")["inline"]
        assert inline["properties"] == [
            {"id": "label", "type": "label", "values": ["first", "second"]},
        ]

    def test_draco_stores_are_refused(self, tmp_path: Path) -> None:
        pytest.importorskip("DracoPy")
        vertices, faces, object_ids = _grid_surfaces(n_obj=2)
        store = str(tmp_path / "draco.zv")
        write_mesh(store, vertices, faces, chunk_shape=(400.0,) * 3,
                   object_ids=object_ids, encoding="draco")
        with pytest.raises(ExportError, match="Draco"):
            export_precomputed(store, _mem_layer())

    def test_attribute_names_are_refused_for_meshes(self, mesh_store) -> None:
        with pytest.raises(ExportError, match="per-vertex attributes"):
            export_precomputed(mesh_store[0], _mem_layer(), attribute_names=["radius"])

    def test_a_skeleton_store_is_not_a_mesh(self, graph_store) -> None:
        with pytest.raises(ExportError, match="not a mesh"):
            export_precomputed_meshes(graph_store[0], _mem_layer())

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows file names only")
    def test_a_local_directory_is_refused_on_windows(self, mesh_store, tmp_path: Path) -> None:
        with pytest.raises(ExportError, match="does not allow ':'"):
            export_precomputed(mesh_store[0], tmp_path / "layer")

    @pytest.mark.skipif(sys.platform == "win32", reason="'<id>:0' is not a Windows file name")
    def test_a_local_directory(self, mesh_store, tmp_path: Path) -> None:
        store, vertices, _faces, object_ids = mesh_store
        export_precomputed(store, tmp_path / "layer")
        mesh = _volume(tmp_path / "layer").mesh.get(2, remove_duplicate_vertices=False)
        assert _points(mesh.vertices) == _points(vertices[object_ids == 1])


def test_other_geometries_are_refused(tmp_path: Path) -> None:
    from zarr_vectors.types.points import write_points

    store = str(tmp_path / "points.zv")
    write_points(store, np.zeros((3, 3), dtype=np.float32) + 1, chunk_shape=(10.0,) * 3)
    with pytest.raises(ExportError, match="writes skeleton, graph and mesh stores"):
        export_precomputed(store, tmp_path / "layer")


class TestCli:

    def test_a_local_layer_with_format_and_unit(self, graph_store, tmp_path: Path) -> None:
        from zarr_vectors_tools.cli import main

        store = graph_store[0] if isinstance(graph_store, tuple) else graph_store
        out = tmp_path / "layer"
        assert main([
            "convert", str(store), str(out), "--format", "precomputed",
            "--unit", "micrometer",
        ]) == 0
        info = json.loads((out / "info").read_text())
        assert info["type"] == "segmentation"

    def test_a_url_output_needs_no_format(self, em_store: str, tmp_path: Path) -> None:
        from zarr_vectors_tools.cli._args import resolve_export_format

        assert resolve_export_format("gs://bucket/layer", None).name == "precomputed"
        assert resolve_export_format(tmp_path / "layer", "precomputed").name == "precomputed"
