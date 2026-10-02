"""Ingest fixes from the viewer review: what each store carries for the viewer.

* A plain precomputed skeleton layer keeps every neuron one tree at every
  pyramid level, whatever order its skeletonizer listed vertices and edges in.
* ``mesh_decimate`` nests every level's chunk grid in the one below and
  survives a collapse that moves a vertex past the store's bounds.
* Every store with objects carries ``object_attributes/vertex_count`` and
  ``fragment_attributes/segment_id`` from its ingest on.
* ``--method`` picks the pyramid coarsener from the CLI.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
from zarr_vectors.building import (
    get_level_chunk_shape,
    get_resolution_level,
    level_grid_layout,
    list_resolution_levels,
    open_store,
    read_all_object_manifests,
    read_chunk_fragment_attributes,
    read_chunk_vertices,
    read_level_metadata,
    read_object_attributes,
    read_root_metadata,
    read_vertex_fragment_index,
)
from zarr_vectors.types.graphs import read_graph

from zarr_vectors_tools.cli import main
from zarr_vectors_tools.multiresolution.skeleton_layout import (
    LAYOUT_LINKED,
    LAYOUT_SPLIT,
    skeleton_layout,
)

# ===================================================================
# Helpers
# ===================================================================


def _components(n: int, edges: np.ndarray) -> int:
    """Connected components of ``n`` vertices joined by undirected ``edges``."""
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in np.asarray(edges, dtype=np.int64).reshape(-1, 2).tolist():
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return len({find(i) for i in range(n)})


def _levels(store: Path) -> list[int]:
    return sorted(int(v) for v in list_resolution_levels(open_store(str(store))))


def _object_columns(store: Path, level: int) -> dict:
    """What a level holds of the two viewer columns, checked against its fragments."""
    level_group = get_resolution_level(open_store(str(store)), level)
    manifests = read_all_object_manifests(level_group)
    out: dict = {"objects": sum(1 for m in manifests if m)}
    counts = np.zeros(len(manifests), dtype=np.int64)
    has_ids = level_group.array_exists("fragment_attributes/segment_id")
    named = (
        np.asarray(read_object_attributes(level_group, "segment_id")).astype(np.uint64)
        if level_group.array_exists("object_attributes/segment_id") else None
    )
    wrong_ids = 0
    for oid, manifest in enumerate(manifests):
        for chunk, fragment in manifest:
            chunk = tuple(int(c) for c in chunk)
            index = read_vertex_fragment_index(level_group, chunk)
            counts[oid] += len(index.indices(int(fragment)))
            if has_ids:
                column = read_chunk_fragment_attributes(
                    level_group, "segment_id", chunk, dtype=np.uint64,
                )
                want = int(named[oid]) if named is not None else oid
                wrong_ids += len(column) < len(index) or int(column[fragment]) != want
    if level_group.array_exists("object_attributes/vertex_count"):
        stored = np.asarray(read_object_attributes(level_group, "vertex_count"))
        out["vertex_count_dtype"] = str(stored.dtype)
        out["vertex_count_exact"] = bool(np.array_equal(stored.astype(np.int64), counts))
    if has_ids:
        out["segment_ids_wrong"] = int(wrong_ids)
    return out


# ===================================================================
# S2: plain precomputed skeletons stay one tree per neuron
# ===================================================================


def _shuffled_tree(rng: np.random.Generator, n: int) -> dict:
    """A random tree with vertices in no order and edges pointing either way."""
    parent = np.array([-1] + [int(rng.integers(max(0, i - 6), i)) for i in range(1, n)])
    pos = np.zeros((n, 3))
    for i in range(1, n):
        pos[i] = pos[parent[i]] + rng.normal(0, 400, 3)
    edges = np.array([[i, parent[i]] for i in range(1, n)])
    perm = rng.permutation(n)
    new_of_old = np.argsort(perm)
    edges = new_of_old[edges]
    flip = rng.random(len(edges)) < 0.5
    edges[flip] = edges[flip][:, ::-1]
    return {
        "vertices": (pos[perm] + 5000).astype(np.float32),
        "edges": edges,
        "radius": rng.uniform(1, 5, n).astype(np.float32),
    }


class _PlainReader:
    """Offline stand-in for PlainPrecomputedReader."""

    def __init__(self, n_segments: int = 3, n_vertices: int = 300, seed: int = 1) -> None:
        from zarr_vectors_tools.convert.ingest.precomputed_plain_skeletons import (
            PlainSkeletonInfo,
        )

        rng = np.random.default_rng(seed)
        self.info = PlainSkeletonInfo(
            base_url="mem://plain", transform=np.eye(4),
            vertex_attributes=[{"id": "radius", "data_type": "float32", "num_components": 1}],
        )
        self.segment_ids = [720575940000000101 + i for i in range(n_segments)]
        self.segment_properties_raw = None
        self._skeletons = {s: _shuffled_tree(rng, n_vertices) for s in self.segment_ids}

    def read_skeleton(self, seg_id: int) -> dict:
        return self._skeletons[int(seg_id)]


@pytest.fixture(scope="module", params=[2, 1], ids=["chunk_scale_2", "chunk_scale_1"])
def plain_store(request, tmp_path_factory) -> Path:
    from zarr_vectors_tools.convert.ingest.precomputed_plain_skeletons import (
        run_ingest_plain,
    )

    store = tmp_path_factory.mktemp("plain") / "plain.zv"
    scale = request.param
    run_ingest_plain(
        _PlainReader(), store, chunk_shape_nm=(1000.0, 1000.0, 1000.0),
        strides=[2, 2], chunk_scale_factors=[scale, scale],
        sparsity_factors=[1.0, 1.0], progress=False,
    )
    return store


class TestPlainPrecomputedSkeletons:
    def test_the_store_is_marked_with_the_layout_it_holds(self, plain_store: Path) -> None:
        assert skeleton_layout(str(plain_store)) == LAYOUT_LINKED

    def test_every_level_holds_one_tree_per_neuron(self, plain_store: Path) -> None:
        for level in _levels(plain_store):
            graph = read_graph(str(plain_store), level=level)
            n = graph["node_count"]
            assert len(graph["edges"]) == n - 3, level
            assert _components(n, graph["edges"]) == 3, level

    def test_exports_and_metrics_see_one_tree_per_neuron(
        self, plain_store: Path, tmp_path: Path,
    ) -> None:
        from cloudvolume import Skeleton

        from zarr_vectors_tools.algorithms.skeleton_metrics import compute_skeleton_metrics
        from zarr_vectors_tools.convert.export.precomputed import (
            export_precomputed_skeletons,
        )
        from zarr_vectors_tools.convert.export.swc import export_swc

        attrs = [{"id": "radius", "data_type": "float32", "num_components": 1}]
        for level in _levels(plain_store):
            layer = tmp_path / f"layer{level}"
            summary = export_precomputed_skeletons(plain_store, str(layer), level=level)
            for segment in summary["segment_ids"]:
                data = (layer / "skeletons" / str(segment)).read_bytes()
                skeleton = Skeleton.from_precomputed(
                    data, segid=segment, vertex_attributes=attrs,
                )
                assert _components(len(skeleton.vertices), skeleton.edges) == 1
            swc = export_swc(plain_store, tmp_path / f"swc{level}", level=level,
                             object_ids=[0, 1, 2])
            assert swc["root_count"] == 3
            metrics = compute_skeleton_metrics(plain_store, level=level, write=False)
            assert metrics["component_count"].tolist() == [1, 1, 1]


def _frags_reader(face_copies: bool):
    """Two .frags chunks holding one straight neuron each side of the x face.

    With ``face_copies`` each side ends at igneous's copy of the face vertex
    (face + half a voxel), which is in both chunks.
    """
    from zarr_vectors_tools.convert.ingest.precomputed_skeletons import (
        InMemoryFragsReader,
        SkeletonInfo,
        enumerate_frag_keys,
    )

    info = SkeletonInfo(
        base_url="mem://x", resolution_nm=(32.0, 32.0, 40.0),
        chunk_size_nm=(16384.0, 16384.0, 20480.0),
    )
    keys = enumerate_frag_keys(info, (17398, 10448, 3088), (2, 1, 1))
    face = 17910 * 32 + 16
    y, z = 340000.0, 130000.0
    left = np.array([[560000, y, z], [565000, y, z], [570000, y, z], [face, y, z]], np.float32)
    right = np.array([[face, y, z], [576000, y, z], [580000, y, z]], np.float32)
    if not face_copies:
        left = left[:-1]
    seg = 720575940000000999
    chain = lambda v: np.array([[i, i - 1] for i in range(1, len(v))])  # noqa: E731
    reader = InMemoryFragsReader(info, {
        keys[0]: {seg: {"vertices": left, "edges": chain(left)}},
        keys[1]: {seg: {"vertices": right, "edges": chain(right)}},
    })
    bounds = ([17398 * 32.0, 10448 * 32.0, 3088 * 40.0],
              [18422 * 32.0, 10960 * 32.0, 3600 * 40.0])
    return reader, keys, bounds, seg


class TestFragsLayout:
    def test_aligned_frags_are_the_split_layout(self, tmp_path: Path) -> None:
        from zarr_vectors_tools.convert.ingest.precomputed_skeletons import run_ingest

        reader, keys, bounds, seg = _frags_reader(face_copies=True)
        store = tmp_path / "frags.zv"
        run_ingest(reader, store, keys, bounds_nm=bounds, strides=[2, 2],
                   chunk_scale_factors=[2, 2], sparsity_factors=[1.0, 1.0],
                   progress=False)
        assert skeleton_layout(str(store)) == LAYOUT_SPLIT
        # The face copies are the cross-chunk join: zero-length links.
        level0 = get_resolution_level(open_store(str(store)), 0)
        from zarr_vectors_tools.algorithms._links import read_cross_links

        links = read_cross_links(level0, delta=0)
        assert len(links) == 1
        (ca, va), (cb, vb) = links[0]
        flat_a = np.concatenate(read_chunk_vertices(level0, tuple(ca)))
        flat_b = np.concatenate(read_chunk_vertices(level0, tuple(cb)))
        assert tuple(ca) != tuple(cb)
        np.testing.assert_array_equal(flat_a[va], flat_b[vb])
        # The exporter reads the split layout by segment id and joins the
        # copies: one tree at every level.
        from cloudvolume import Skeleton

        from zarr_vectors_tools.convert.export.precomputed import (
            export_precomputed_skeletons,
        )

        for level in _levels(store):
            layer = tmp_path / f"layer{level}"
            export_precomputed_skeletons(store, str(layer), level=level)
            skeleton = Skeleton.from_precomputed(
                (layer / "skeletons" / str(seg)).read_bytes(), segid=seg,
            )
            assert _components(len(skeleton.vertices), skeleton.edges) == 1, level

    def test_unaligned_frags_are_the_linked_layout(self, tmp_path: Path) -> None:
        from zarr_vectors_tools.convert.ingest.precomputed_skeletons import run_ingest

        reader, keys, bounds, _seg = _frags_reader(face_copies=False)
        store = tmp_path / "frags.zv"
        run_ingest(reader, store, keys, bounds_nm=bounds, strides=[2],
                   chunk_scale_factors=[2], sparsity_factors=[1.0], align=False,
                   progress=False)
        assert skeleton_layout(str(store)) == LAYOUT_LINKED


# ===================================================================
# S5: mesh_decimate grids nest, and a vertex past the bounds is kept
# ===================================================================


def _icosphere(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    t = (1 + 5 ** 0.5) / 2
    verts = [np.array(v, float) / np.linalg.norm(v) for v in (
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
        [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1])]
    faces = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9],
             [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2],
             [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10],
             [8, 6, 7], [9, 8, 1]]
    for _ in range(subdivisions):
        cache: dict = {}

        def mid(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]

        new = []
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = new
    return np.array(verts), np.array(faces)


def _write_obj(path: Path, spheres: list[tuple[tuple[float, float, float], float]],
               subdivisions: int = 4) -> Path:
    v, f = _icosphere(subdivisions)
    lines, faces = [], []
    for k, (centre, radius) in enumerate(spheres):
        lines.append(f"o sphere{k}\n")
        lines += [f"v {a:.6f} {b:.6f} {c:.6f}\n" for a, b, c in v * radius + centre]
        faces += [f"f {a + 1 + k * len(v)} {b + 1 + k * len(v)} {c + 1 + k * len(v)}\n"
                  for a, b, c in f]
    path.write_text("".join(lines + faces))
    return path


def _chunk_shapes(store: Path) -> list[tuple[float, ...]]:
    root = open_store(str(store))
    meta = read_root_metadata(root)
    return [tuple(float(c) for c in get_level_chunk_shape(meta, read_level_metadata(root, lv)))
            for lv in _levels(store)]


class TestMeshDecimate:
    @pytest.mark.parametrize("chunk_scale, factors", [
        (None, [(4, 1), (4, 1)]),
        ([2, 2], [(4, 1), (4, 1)]),
        ([2, 2, 2], [(2, 1), (2, 1), (2, 1)]),
        ([2, 1, 2], [(8, 1), (2, 1), (2, 1)]),
    ])
    def test_a_vertex_past_the_bounds_is_filed_in_the_grid(
        self, tmp_path: Path, chunk_scale, factors,
    ) -> None:
        from zarr_vectors_tools.multiresolution.coarsen import build_pyramid

        obj = _write_obj(tmp_path / "sphere.obj", [((150.0, 150.0, 150.0), 50.0)])
        store = tmp_path / "sphere.zv"
        assert main(["convert", str(obj), str(store), "--chunk-shape", "50,50,50"]) == 0
        build_pyramid(str(store), factors=factors, chunk_scale_factors=chunk_scale,
                      method="mesh_decimate", cross_level_storage="none")

        root = open_store(str(store))
        lo, hi = (np.asarray(b, float) for b in read_root_metadata(root).bounds)
        outside = 0
        for level, shape in zip(_levels(store)[1:], _chunk_shapes(store)[1:]):
            level_group = get_resolution_level(root, level)
            origin, extent = level_grid_layout((lo.tolist(), hi.tolist()), shape)
            for chunk, manifest in {
                tuple(int(c) for c in cc): 1
                for m in read_all_object_manifests(level_group) for cc, _f in m
            }.items():
                assert all(o <= c < o + e for c, o, e in zip(chunk, origin, extent))
                positions = np.concatenate(read_chunk_vertices(level_group, chunk))
                outside += int(np.any((positions < lo) | (positions > hi), axis=1).sum())
        # The decimation's own positions, not clamped to the bounds.
        assert outside > 0

    def test_each_level_nests_in_the_one_below(self, tmp_path: Path) -> None:
        from zarr_vectors_tools.multiresolution.coarsen import build_pyramid

        obj = _write_obj(tmp_path / "pair.obj", [((60.0, 60.0, 60.0), 40.0),
                                                  ((300.0, 260.0, 220.0), 40.0)])
        store = tmp_path / "pair.zv"
        assert main(["convert", str(obj), str(store), "--chunk-shape", "100,100,100",
                     "--split-objects"]) == 0
        build_pyramid(str(store), factors=[(4, 1), (4, 1)], chunk_scale_factors=[2, 2],
                      method="mesh_decimate", cross_level_storage="none")
        shapes = _chunk_shapes(store)
        # Grown by the scale asked for, past the data's extent (about 300).
        assert shapes == [(100.0,) * 3, (200.0,) * 3, (400.0,) * 3]
        for finer, coarser in zip(shapes, shapes[1:]):
            ratio = np.asarray(coarser) / np.asarray(finer)
            np.testing.assert_array_equal(ratio, np.round(ratio))
            assert (ratio >= 1).all()

    def test_coarse_levels_carry_the_object_columns(self, tmp_path: Path) -> None:
        obj = _write_obj(tmp_path / "pair.obj", [((60.0, 60.0, 60.0), 40.0),
                                                  ((200.0, 160.0, 120.0), 40.0)], 3)
        store = tmp_path / "pair.zv"
        assert main(["convert", str(obj), str(store), "--chunk-shape", "50,50,50",
                     "--split-objects", "--coarsen", "4,4", "--sparsity", "1,1",
                     "--chunk-scale", "2,2", "--method", "mesh_decimate"]) == 0
        for level in _levels(store):
            columns = _object_columns(store, level)
            assert columns == {"objects": 2, "vertex_count_dtype": "uint32",
                               "vertex_count_exact": True, "segment_ids_wrong": 0}, level


# ===================================================================
# S6 / S7: vertex_count and fragment segment ids from every ingest
# ===================================================================


def _write_inputs(root: Path) -> dict[str, list[str]]:
    """Small inputs and the convert arguments for each object-bearing format."""
    rng = np.random.default_rng(0)
    root.mkdir(parents=True, exist_ok=True)
    out: dict[str, list[str]] = {}

    pts = rng.uniform(0, 300, (400, 3))
    (root / "points.csv").write_text("x,y,z,oid\n" + "".join(
        f"{a:.3f},{b:.3f},{c:.3f},{i % 7}\n" for i, (a, b, c) in enumerate(pts)))
    out["table"] = [str(root / "points.csv"), "--format", "table", "--position-columns",
                    "x,y,z", "--object-id-column", "oid", "--chunk-shape", "100,100,100"]

    import anndata
    import pandas as pd

    adata = anndata.AnnData(
        X=rng.random((200, 2)).astype(np.float32),
        obs=pd.DataFrame({"cluster": pd.Categorical(rng.choice(["a", "b", "c"], 200))},
                         index=[f"c{i}" for i in range(200)]),
    )
    adata.obsm["spatial"] = rng.uniform(0, 300, (200, 3))
    adata.write_h5ad(root / "cells.h5ad")
    out["h5ad"] = [str(root / "cells.h5ad"), "--object-id-column", "cluster",
                   "--chunk-shape", "100,100,100"]

    ends = rng.uniform(0, 300, (150, 6))
    ends[:, 3:] = ends[:, :3] + rng.normal(0, 30, (150, 3))
    (root / "lines.csv").write_text("x0,y0,z0,x1,y1,z1\n" + "".join(
        ",".join(f"{v:.3f}" for v in row) + "\n" for row in ends))
    out["lines"] = [str(root / "lines.csv"), "--format", "lines", "--chunk-shape", "100,100,100"]

    nodes = rng.uniform(0, 300, (120, 3))
    (root / "nodes.csv").write_text("node_id,x,y,z\n" + "".join(
        f"{i},{a:.3f},{b:.3f},{c:.3f}\n" for i, (a, b, c) in enumerate(nodes)))
    (root / "edges.csv").write_text("source,target\n" + "".join(
        f"{i},{(i * 7 + 3) % 120}\n" for i in range(120)))
    out["edgelist"] = [str(root / "edges.csv"), "--format", "edgelist", "--nodes",
                       str(root / "nodes.csv"), "--chunk-shape", "100,100,100"]

    graphml = ['<?xml version="1.0"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
               '<key id="x" for="node" attr.name="x" attr.type="double"/>',
               '<key id="y" for="node" attr.name="y" attr.type="double"/>',
               '<key id="z" for="node" attr.name="z" attr.type="double"/>',
               '<graph edgedefault="undirected">']
    for i, (a, b, c) in enumerate(nodes[:60]):
        graphml.append(f'<node id="n{i}"><data key="x">{a}</data><data key="y">{b}</data>'
                       f'<data key="z">{c}</data></node>')
    graphml += [f'<edge source="n{i}" target="n{(i + 1) % 60}"/>' for i in range(60)]
    graphml += ["</graph>", "</graphml>"]
    (root / "graph.graphml").write_text("\n".join(graphml))
    out["graphml"] = [str(root / "graph.graphml"), "--chunk-shape", "100,100,100"]

    import nibabel as nib

    streamlines = [np.cumsum(rng.normal(0, 3, (40, 3)), axis=0).astype(np.float32) + 50
                   for _ in range(30)]
    tractogram = nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4))
    nib.streamlines.save(tractogram, str(root / "tracts.tck"))
    out["tck"] = [str(root / "tracts.tck"), "--chunk-shape", "20,20,20"]
    header = {nib.streamlines.Field.VOXEL_TO_RASMM: np.eye(4),
              nib.streamlines.Field.DIMENSIONS: (200, 200, 200),
              nib.streamlines.Field.VOXEL_SIZES: (1.0, 1.0, 1.0)}
    nib.streamlines.save(nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4)),
                         str(root / "tracts.trk"), header=header)
    out["trk"] = [str(root / "tracts.trk"), "--num-chunks", "8"]

    swc = []
    nid = 0
    for tree in range(3):
        start = nid + 1
        for k in range(50):
            nid += 1
            parent = -1 if k == 0 else (nid - 1 if k % 10 else start)
            x, y, z = 30 + 60 * tree + 3 * k, 30 + 2 * k, 40
            swc.append(f"{nid} 3 {x} {y} {z} 1.0 {parent}")
    (root / "trees.swc").write_text("\n".join(swc) + "\n")
    out["swc"] = [str(root / "trees.swc"), "--chunk-shape", "50,50,50"]

    _write_obj(root / "pair.obj", [((60.0, 60.0, 60.0), 40.0), ((200.0, 160.0, 120.0), 40.0)], 2)
    out["obj"] = [str(root / "pair.obj"), "--chunk-shape", "50,50,50"]
    out["obj_objects"] = [str(root / "pair.obj"), "--chunk-shape", "50,50,50",
                          "--split-objects"]

    v, f = _icosphere(2)
    v = (v * 40 + 60).astype(np.float32)
    with open(root / "ball.stl", "wb") as handle:
        handle.write(b"\0" * 80 + struct.pack("<I", len(f)))
        for a, b, c in f:
            handle.write(struct.pack("<3f", 0, 0, 0) + v[a].tobytes() + v[b].tobytes()
                         + v[c].tobytes() + b"\0\0")
    out["stl"] = [str(root / "ball.stl"), "--chunk-shape", "50,50,50"]

    # A legacy precomputed mesh layer of two segments.
    layer = root / "mesh"
    layer.mkdir()
    (layer / "info").write_text(json.dumps({"@type": "neuroglancer_legacy_mesh"}))
    for seg, centre in ((7, 60.0), (9, 200.0)):
        verts = (v - 60 + centre).astype("<f4")
        (layer / f"{seg}:0:0").write_bytes(
            struct.pack("<I", len(verts)) + verts.tobytes() + f.astype("<u4").tobytes())
        (layer / f"{seg}:0").write_text(json.dumps({"fragments": [f"{seg}:0:0"]}))
    out["precomputed_mesh"] = [str(layer), "--chunk-shape", "50,50,50"]

    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from _surface_fixtures import write_freesurfer_subject, write_gifti_subject

    out["gifti"] = [str(write_gifti_subject(root / "gifti_subject")), "--chunk-shape",
                    "20,20,20"]
    out["freesurfer"] = [str(write_freesurfer_subject(root / "fs_subject")),
                         "--chunk-shape", "20,20,20"]
    return out


@pytest.fixture(scope="module")
def inputs(tmp_path_factory) -> dict[str, list[str]]:
    return _write_inputs(tmp_path_factory.mktemp("inputs"))


_FORMATS = ["table", "h5ad", "lines", "edgelist", "graphml", "tck", "trk", "swc", "obj",
            "obj_objects", "stl", "precomputed_mesh", "gifti", "freesurfer"]


class TestObjectColumns:
    @pytest.mark.parametrize("name", _FORMATS)
    def test_every_ingest_with_objects_writes_both(
        self, inputs: dict[str, list[str]], tmp_path: Path, name: str,
    ) -> None:
        if name == "precomputed_mesh":
            pytest.importorskip("cloudvolume")
        args = inputs[name]
        store = tmp_path / f"{name}.zv"
        assert main(["convert", args[0], str(store), *args[1:]]) == 0
        columns = _object_columns(store, 0)
        assert columns["objects"] > 0
        assert columns["vertex_count_dtype"] == "uint32"
        assert columns["vertex_count_exact"] is True
        assert columns["segment_ids_wrong"] == 0

    def test_precomputed_skeletons_write_both(self, plain_store: Path) -> None:
        assert _object_columns(plain_store, 0) == {
            "objects": 3, "vertex_count_dtype": "uint32", "vertex_count_exact": True,
            "segment_ids_wrong": 0,
        }

    def test_a_point_cloud_without_objects_gets_neither(self, tmp_path: Path) -> None:
        csv = tmp_path / "xyz.csv"
        csv.write_text("x,y,z\n1,2,3\n40,50,60\n")
        store = tmp_path / "xyz.zv"
        assert main(["convert", str(csv), str(store), "--chunk-shape", "100,100,100"]) == 0
        level0 = get_resolution_level(open_store(str(store)), 0)
        assert not level0.array_exists("object_attributes/vertex_count")
        assert not level0.array_exists("fragment_attributes/segment_id")

    def test_per_object_vertex_count_false_skips_the_count(self, tmp_path: Path) -> None:
        from zarr_vectors_tools.convert.ingest.csv_points import ingest_csv

        csv = tmp_path / "p.csv"
        csv.write_text("x,y,z\n1,2,3\n40,50,60\n150,150,150\n")
        store = tmp_path / "p.zv"
        ingest_csv(csv, store, (100.0, 100.0, 100.0), object_ids=np.array([0, 0, 1]),
                   per_object_vertex_count=False)
        level0 = get_resolution_level(open_store(str(store)), 0)
        assert not level0.array_exists("object_attributes/vertex_count")
        assert level0.array_exists("fragment_attributes/segment_id")

    def test_sparse_object_ids_count_per_object(self, tmp_path: Path) -> None:
        from zarr_vectors_tools.convert.ingest.csv_points import ingest_csv

        csv = tmp_path / "p.csv"
        csv.write_text("x,y,z\n1,2,3\n40,50,60\n150,150,150\n160,150,150\n170,150,150\n")
        store = tmp_path / "p.zv"
        ingest_csv(csv, store, (100.0, 100.0, 100.0), object_ids=np.array([2, 2, 5, 5, 5]))
        assert _object_columns(store, 0)["vertex_count_exact"] is True
        counts = read_object_attributes(get_resolution_level(open_store(str(store)), 0),
                                        "vertex_count")
        # The writer numbers ids 2 and 5 densely, as objects 0 and 1.
        assert np.asarray(counts).tolist() == [2, 3]

    def test_a_skeleton_level_counts_dropped_trees_as_zero(
        self, inputs: dict[str, list[str]], tmp_path: Path,
    ) -> None:
        args = inputs["swc"]
        store = tmp_path / "swc.zv"
        assert main(["convert", args[0], str(store), *args[1:], "--coarsen", "2",
                     "--sparsity", "2"]) == 0
        level1 = get_resolution_level(open_store(str(store)), 1)
        counts = np.asarray(read_object_attributes(level1, "vertex_count"))
        manifests = read_all_object_manifests(level1)
        dropped = [oid for oid, m in enumerate(manifests) if not m]
        assert dropped
        assert all(int(counts[oid]) == 0 for oid in dropped)
        assert _object_columns(store, 1)["vertex_count_exact"] is True


    def test_a_trk_vertex_count_property_gives_way_to_the_count(self, tmp_path: Path) -> None:
        import nibabel as nib

        rng = np.random.default_rng(3)
        lines = [np.cumsum(rng.normal(0, 3, (n, 3)), axis=0).astype(np.float32) + 50
                 for n in (5, 9, 12)]
        header = {nib.streamlines.Field.VOXEL_TO_RASMM: np.eye(4),
                  nib.streamlines.Field.DIMENSIONS: (200, 200, 200),
                  nib.streamlines.Field.VOXEL_SIZES: (1.0, 1.0, 1.0)}
        tractogram = nib.streamlines.Tractogram(
            lines, affine_to_rasmm=np.eye(4),
            data_per_streamline={"vertex_count": np.array([[1.0], [2.0], [3.0]])},
        )
        nib.streamlines.save(tractogram, str(tmp_path / "t.trk"), header=header)
        store = tmp_path / "t.zv"
        assert main(["convert", str(tmp_path / "t.trk"), str(store), "--num-chunks", "8"]) == 0
        counts = read_object_attributes(get_resolution_level(open_store(str(store)), 0),
                                        "vertex_count")
        assert counts.dtype == np.uint32
        assert np.asarray(counts).tolist() == [5, 9, 12]


# ===================================================================
# Skeleton levels record their scale (OME multiscales) and stride
# ===================================================================


def test_skeleton_levels_record_their_scale_and_stride(
    inputs: dict[str, list[str]], tmp_path: Path,
) -> None:
    from zarr_vectors_tools.multiresolution.coarsen import read_coarsening_record
    from zarr_vectors_tools.multiresolution.refresh import rebuild_pyramid_from_level

    args = inputs["swc"]
    store = tmp_path / "s.zv"
    assert main(["convert", args[0], str(store), *args[1:], "--coarsen", "4,2",
                 "--sparsity", "1,1", "--chunk-scale", "2,4"]) == 0
    root = open_store(str(store))
    # Each level is as coarse as its stride or its chunk growth, whichever is more.
    assert [tuple(read_level_metadata(root, lv).bin_ratio) for lv in (1, 2)] == [
        (4, 4, 4), (16, 16, 16)]
    assert [read_coarsening_record(root, lv)["stride"] for lv in (1, 2)] == [4, 2]
    meta = json.loads((store / "zarr.json").read_text())["attributes"]
    scales = [d["coordinateTransformations"][0]["scale"]
              for d in meta["multiscales"][0]["datasets"]]
    assert scales == [[1.0] * 3, [4.0] * 3, [16.0] * 3]

    def snapshot() -> list[np.ndarray]:
        out = []
        for lv in (1, 2):
            level_group = get_resolution_level(open_store(str(store)), lv)
            out.append(np.unique(np.concatenate([
                np.concatenate(read_chunk_vertices(level_group, tuple(cc)))
                for m in read_all_object_manifests(level_group) for cc, _f in m
            ]), axis=0))
        return out

    before = snapshot()
    rebuild_pyramid_from_level(open_store(str(store), mode="r+"), 0,
                               coarsen_factors={1: 4, 2: 2}, sparsity_strategy="length")
    for old, new in zip(before, snapshot()):
        np.testing.assert_array_equal(old, new)


# ===================================================================
# CLI --method
# ===================================================================


class TestMethodFlag:
    def test_choices_are_the_registry(self) -> None:
        from zarr_vectors_tools.cli import build_parser
        from zarr_vectors_tools.multiresolution import coarsen

        parser = build_parser()
        for command in ("convert", "pyramid"):
            sub = parser._subparsers._group_actions[0].choices[command]  # noqa: SLF001
            action = next(a for a in sub._actions if a.dest == "method")  # noqa: SLF001
            assert set(action.choices) == {"auto", *coarsen._COARSENERS}  # noqa: SLF001
            assert action.default == "auto"

    def test_pyramid_builds_with_edge_collapse(
        self, inputs: dict[str, list[str]], tmp_path: Path,
    ) -> None:
        args = inputs["obj_objects"]
        store = tmp_path / "m.zv"
        assert main(["convert", args[0], str(store), *args[1:]]) == 0
        assert main(["pyramid", str(store), "--coarsen", "4,4", "--sparsity", "1,1",
                     "--chunk-scale", "2,2", "--method", "mesh_decimate"]) == 0
        root = open_store(str(store))
        assert [read_level_metadata(root, lv).coarsening_method for lv in (1, 2)] == [
            "mesh_quadric_collapse"] * 2

    def test_convert_builds_with_edge_collapse(
        self, inputs: dict[str, list[str]], tmp_path: Path,
    ) -> None:
        args = inputs["obj"]
        store = tmp_path / "m.zv"
        assert main(["convert", args[0], str(store), *args[1:], "--coarsen", "4",
                     "--sparsity", "1", "--method", "mesh_decimate"]) == 0
        level1 = read_level_metadata(open_store(str(store)), 1)
        assert level1.coarsening_method == "mesh_quadric_collapse"

    @pytest.mark.parametrize("extra, message", [
        (["--method", "mesh_decimate", "--coarsen", "4", "--sparsity", "1"],
         "coarsens mesh stores, not a skeleton store"),
        (["--method", "per_object", "--coarsen", "2", "--sparsity", "1"],
         "not a skeleton store; use skeleton"),
        (["--method", "skeleton"], "pass --coarsen and --sparsity"),
    ])
    def test_a_method_that_does_not_fit_is_refused_before_the_ingest(
        self, inputs: dict[str, list[str]], tmp_path: Path, extra, message,
    ) -> None:
        args = inputs["swc"]
        store = tmp_path / "s.zv"
        with pytest.raises(SystemExit) as refused:
            main(["convert", args[0], str(store), *args[1:], *extra])
        assert message in str(refused.value)
        assert not store.exists()

    @pytest.mark.parametrize("name, extra, message", [
        ("obj", ["--method", "mesh_decimate", "--coarsen", "4", "--sparsity", "2"],
         "set --sparsity to 1"),
        ("obj", ["--method", "mesh_decimate", "--sparsity", "1"], "must be > 1"),
        ("trk", ["--method", "per_object", "--coarsen", "1", "--sparsity", "2"],
         "its pyramid is built inside the ingest"),
        ("tck", ["--method", "per_object", "--coarsen", "1", "--sparsity", "2",
                 "--rdp-tolerance", "1"], "--method per_object has no tolerance"),
        ("table", ["--method", "polyline", "--coarsen", "2", "--sparsity", "2"],
         "not a point_cloud store"),
    ])
    def test_other_refusals(
        self, inputs: dict[str, list[str]], tmp_path: Path, name, extra, message,
    ) -> None:
        args = inputs[name]
        store = tmp_path / "x.zv"
        with pytest.raises(SystemExit) as refused:
            main(["convert", args[0], str(store), *args[1:], *extra])
        assert message in str(refused.value)
        assert not store.exists()

    def test_pyramid_refuses_before_removing_levels(
        self, inputs: dict[str, list[str]], tmp_path: Path,
    ) -> None:
        args = inputs["swc"]
        store = tmp_path / "s.zv"
        assert main(["convert", args[0], str(store), *args[1:], "--coarsen", "2",
                     "--sparsity", "1"]) == 0
        with pytest.raises(SystemExit) as refused:
            main(["pyramid", str(store), "--coarsen", "4", "--sparsity", "1",
                  "--method", "mesh", "--replace"])
        assert "not a skeleton store" in str(refused.value)
        assert _levels(store) == [0, 1]

    def test_export_refuses_method(
        self, inputs: dict[str, list[str]], tmp_path: Path,
    ) -> None:
        args = inputs["obj"]
        store = tmp_path / "m.zv"
        assert main(["convert", args[0], str(store), *args[1:]]) == 0
        with pytest.raises(SystemExit) as refused:
            main(["convert", str(store), str(tmp_path / "m.obj"), "--method", "mesh"])
        assert "--method" in str(refused.value)
