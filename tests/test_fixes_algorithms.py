"""Regression tests: mesh algorithms see every face, graph algorithms every edge.

* Mesh stores keep a face whose corners lie in different chunks in the link
  family's cross-chunk cells.  Every mesh algorithm must include those faces,
  so its answer does not depend on the chunk shape.
* A store whose faces are all quads keeps them as quads; every mesh algorithm
  must split them into triangles rather than refuse or return zeros.
* A skeleton store implies most parent links by vertex order rather than
  storing them; the graph algorithms must see those links.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from zarr_vectors.building import get_resolution_level, open_store
from zarr_vectors.types import skeletons as sk
from zarr_vectors.types.graphs import read_graph, write_graph
from zarr_vectors.types.meshes import read_mesh, write_mesh

from zarr_vectors_tools.algorithms import (
    bfs_distances,
    cast_ray,
    closest_point,
    compute_connected_components,
    compute_k_core,
    compute_label_propagation,
    compute_louvain,
    compute_mean_curvature,
    compute_mesh_summary,
    compute_vertex_normals,
    shortest_path,
)
from zarr_vectors_tools.algorithms._graph_edges import read_edges
from zarr_vectors_tools.algorithms.mesh_query import (
    _closest_point_on_triangle,
    _moller_trumbore,
)
from zarr_vectors_tools.convert.ingest.obj import ingest_obj
from zarr_vectors_tools.convert.ingest.swc import ingest_swc
from zarr_vectors_tools.multiresolution.object_index import build_object_index

# ---------------------------------------------------------------------------
# Meshes that span chunks
# ---------------------------------------------------------------------------


def _icosphere(subdivisions: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """A sphere of radius 10 centred on (20, 20, 20), outward winding."""
    t = (1 + 5 ** 0.5) / 2
    verts = [np.array(p, float) / np.linalg.norm(p) for p in [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
        (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
    ]]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9),
             (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2),
             (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
             (8, 6, 7), (9, 8, 1)]
    for _ in range(subdivisions):
        cache: dict[tuple[int, int], int] = {}

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
            new += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new
    return (np.array(verts) * 10 + 20).astype(np.float32), np.array(faces, dtype=np.int64)


@pytest.fixture(scope="module")
def spheres(tmp_path_factory) -> dict[int, Path]:
    """The same sphere at three chunk edges; at 20 it is cut on every axis."""
    v, f = _icosphere()
    out = {}
    for edge in (20, 40, 1000):
        path = tmp_path_factory.mktemp("sphere") / f"s{edge}.zv"
        result = write_mesh(str(path), v, f, chunk_shape=(float(edge),) * 3)
        if edge == 20:
            assert result["cross_face_count"] > 50, "the test needs faces that span chunks"
        out[edge] = path
    return out


def _by_position(path: Path) -> np.ndarray:
    """Store order -> an order shared by every chunking of the same mesh."""
    return np.lexsort(np.asarray(read_mesh(str(path))["vertices"]).T[::-1])


def test_mesh_summary_does_not_depend_on_chunk_shape(spheres):
    v, f = _icosphere()
    a, b, c = (v[f[:, k]].astype(np.float64) for k in range(3))
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()
    for path in spheres.values():
        s = compute_mesh_summary(path, per_object=True)
        assert s["face_count"] == len(f)
        assert s["vertex_count"] == len(v)
        assert s["edge_count"] == 3 * len(f) // 2
        assert s["euler_characteristic"] == 2
        assert s["surface_area"] == pytest.approx(area, rel=1e-9)
        assert s["volume"] == pytest.approx(
            compute_mesh_summary(spheres[1000])["volume"], rel=1e-9,
        )
        (obj,) = s["per_object"]
        assert obj["face_count"] == len(f)
        assert obj["surface_area"] == pytest.approx(s["surface_area"], rel=1e-9)
        assert obj["volume"] == pytest.approx(s["volume"], rel=1e-9)


def test_normals_and_curvature_do_not_depend_on_chunk_shape(spheres):
    reference = None
    for path in spheres.values():
        order = _by_position(path)
        normals = compute_vertex_normals(path)
        uniform = compute_vertex_normals(path, weighting="uniform")["normals"]
        curvature = compute_mean_curvature(path)
        assert normals["incomplete_boundary_vertices"] == 0
        assert curvature["incomplete_boundary_vertices"] == 0
        got = (normals["normals"][order], uniform[order], curvature["mean_curvature"][order])
        if reference is None:
            reference = got
            continue
        for x, y in zip(got, reference):
            np.testing.assert_allclose(x, y, atol=1e-6)
    # A sphere's normals point away from its centre; its curvature is 1 / r.
    v = np.asarray(read_mesh(str(spheres[20]))["vertices"], dtype=np.float64)
    radial = (v - 20.0) / np.linalg.norm(v - 20.0, axis=1, keepdims=True)
    normals = compute_vertex_normals(spheres[20])["normals"]
    assert np.min(np.einsum("ij,ij->i", normals, radial)) > 0.99
    assert np.median(compute_mean_curvature(spheres[20])["mean_curvature"]) == pytest.approx(
        0.1, rel=0.01,
    )


def test_queries_find_faces_that_span_chunks(spheres):
    v, f = _icosphere()
    a, b, c = (v[f[:, k]].astype(np.float64) for k in range(3))
    rng = np.random.default_rng(3)
    spanning_hits = 0
    for _ in range(60):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        origin = 20.0 + 30.0 * direction
        ts = _moller_trumbore(origin, -direction, a, b, c)
        expected_t = np.nanmin(ts)
        query = rng.uniform(0, 40, 3)
        expected_d = np.sqrt(_closest_point_on_triangle(query, a, b, c)[1].min())
        for path in spheres.values():
            hit = cast_ray(path, origin, -direction, max_distance=100.0)
            assert hit["hit"]
            assert hit["t"] == pytest.approx(expected_t, abs=1e-9)
            near = closest_point(path, query)
            assert near["found"]
            assert near["distance"] == pytest.approx(expected_d, abs=1e-9)
            if path == spheres[20] and hit["face_index"] is None:
                spanning_hits += 1
                assert len({chunk for chunk, _row in hit["corners"]}) > 1
    assert spanning_hits > 0, "some rays should hit a face that spans chunks"


def test_query_finds_a_face_through_a_chunk_holding_none_of_its_corners(tmp_path):
    """One big triangle whose corners sit in three far corner chunks."""
    v = np.array([[1, 1, 5], [59, 1, 5], [1, 59, 5]], dtype=np.float32)
    path = tmp_path / "big.zv"
    write_mesh(str(path), v, np.array([[0, 1, 2]]), chunk_shape=(10.0, 10.0, 10.0))
    near = closest_point(path, np.array([15.0, 15.0, 8.0]))
    assert near["found"] and near["distance"] == pytest.approx(3.0)
    hit = cast_ray(path, np.array([15.0, 15.0, 9.0]), np.array([0.0, 0.0, -1.0]))
    assert hit["hit"] and hit["t"] == pytest.approx(4.0)
    assert compute_mesh_summary(path)["surface_area"] == pytest.approx(0.5 * 58 * 58)


# ---------------------------------------------------------------------------
# Quads
# ---------------------------------------------------------------------------

_CUBE = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float32)
# Outward winding.
_QUADS = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
                   [2, 3, 7, 6], [0, 4, 7, 3], [1, 2, 6, 5]], dtype=np.int64)


def _fanned(quads: np.ndarray) -> np.ndarray:
    return np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]], axis=0)


@pytest.mark.parametrize("edge", [10.0, 0.5])
def test_quad_mesh_summary(tmp_path, edge):
    path = tmp_path / "quads.zv"
    write_mesh(str(path), _CUBE, _QUADS, chunk_shape=(edge,) * 3)
    s = compute_mesh_summary(path, per_object=True)
    assert s["face_count"] == 6
    assert s["edge_count"] == 12
    assert s["euler_characteristic"] == 2
    assert s["surface_area"] == pytest.approx(6.0)
    assert s["volume"] == pytest.approx(1.0)
    assert s["per_object"][0]["face_count"] == 6
    assert s["per_object"][0]["surface_area"] == pytest.approx(6.0)


@pytest.mark.parametrize("edge", [10.0, 0.5])
def test_quad_mesh_matches_its_triangulation(tmp_path, edge):
    quads = tmp_path / "quads.zv"
    tris = tmp_path / "tris.zv"
    write_mesh(str(quads), _CUBE, _QUADS, chunk_shape=(edge,) * 3)
    write_mesh(str(tris), _CUBE, _fanned(_QUADS), chunk_shape=(edge,) * 3)
    for weighting in ("area", "uniform"):
        got = compute_vertex_normals(quads, weighting=weighting)["normals"]
        assert np.abs(got).sum() > 0
        np.testing.assert_allclose(
            got, compute_vertex_normals(tris, weighting=weighting)["normals"], atol=1e-6,
        )
    curvature = compute_mean_curvature(quads)["mean_curvature"]
    assert np.all(curvature > 0)
    np.testing.assert_allclose(
        curvature, compute_mean_curvature(tris)["mean_curvature"], atol=1e-6,
    )
    near = closest_point(quads, np.array([2.0, 0.5, 0.5]))
    assert near["found"] and near["distance"] == pytest.approx(1.0)
    assert len(near["corners"]) == 4
    hit = cast_ray(quads, np.array([-1.0, 0.3, 0.6]), np.array([1.0, 0.0, 0.0]))
    assert hit["hit"] and hit["t"] == pytest.approx(1.0)


def test_all_quad_obj_is_measured(tmp_path):
    obj = tmp_path / "cube.obj"
    obj.write_text(
        "".join(f"v {x} {y} {z}\n" for x, y, z in _CUBE)
        + "".join("f " + " ".join(str(i + 1) for i in q) + "\n" for q in _QUADS)
    )
    path = tmp_path / "cube.zv"
    ingest_obj(obj, path, (10.0, 10.0, 10.0))
    assert read_mesh(str(path))["faces"].shape[1] == 4
    s = compute_mesh_summary(path)
    assert s["face_count"] == 6
    assert s["surface_area"] == pytest.approx(6.0)
    assert s["volume"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Skeleton stores: implied parent links
# ---------------------------------------------------------------------------

def _two_tree_swc(path: Path) -> None:
    """A Y-shaped 21-node tree and a separate 11-node chain."""
    rows = ["1 1 0 0 0 1 -1"]
    rows += [f"{k + 1} 3 {5 * k} 0 0 1 {k}" for k in range(1, 9)]
    rows += [f"{9 + k} 3 {40 + 5 * k} {5 * k} 0 1 {9 if k == 1 else 8 + k}" for k in range(1, 7)]
    rows += [f"{15 + k} 3 {40 + 5 * k} {-5 * k} 0 1 {9 if k == 1 else 14 + k}" for k in range(1, 7)]
    rows += ["22 1 0 100 0 1 -1"]
    rows += [f"{22 + k} 2 0 {100 + 5 * k} 0 1 {21 + k}" for k in range(1, 11)]
    path.write_text("\n".join(rows) + "\n")


def _edge_set(a, b) -> set[tuple[int, int]]:
    return {tuple(sorted(e)) for e in zip(np.asarray(a).tolist(), np.asarray(b).tolist())}


@pytest.mark.parametrize("edge", [20.0, 50.0, 1000.0])
def test_graph_algorithms_see_a_skeletons_implied_links(tmp_path, edge):
    swc = tmp_path / "two.swc"
    _two_tree_swc(swc)
    path = tmp_path / "two.zv"
    ingest_swc(swc, path, (edge,) * 3)

    graph = read_graph(str(path))
    assert graph["edge_count"] == 30
    a, b, _w, n = read_edges(get_resolution_level(open_store(str(path)), 0))
    assert n == 32
    assert _edge_set(a, b) == _edge_set(graph["edges"][:, 0], graph["edges"][:, 1])

    cc = compute_connected_components(path)
    assert cc["n_components"] == 2
    assert sorted(cc["component_sizes"].tolist()) == [11, 21]

    bfs = bfs_distances(path, 0)
    assert int((bfs["distances"] >= 0).sum()) == int(cc["component_sizes"][cc["labels"][0]])

    # The tip of the chain is 10 links from its soma, whichever chunks they cross.
    positions = np.asarray(graph["positions"])
    soma = int(np.flatnonzero((positions == [0, 100, 0]).all(axis=1))[0])
    tip = int(np.flatnonzero((positions == [0, 150, 0]).all(axis=1))[0])
    route = shortest_path(path, soma, tip)
    assert route["cost"] == 10.0 and len(route["path"]) == 11

    assert compute_k_core(path)["max_core"] == 1
    for labels in (compute_label_propagation(path)["labels"], compute_louvain(path)["labels"]):
        # No community spans the two trees.
        trees = cc["labels"]
        for community in np.unique(labels):
            assert len(np.unique(trees[labels == community])) == 1


def _chunked_skeleton(path: Path) -> None:
    """Written chunk by chunk, as precomputed ingest does: one chain per chunk,
    joined end to start by a cross-chunk link whose parent ends its path."""
    root, level = sk.init_skeleton_store(
        str(path), chunk_shape=(20.0, 20.0, 20.0),
        bounds=([0.0, 0.0, 0.0], [40.0, 20.0, 20.0]), ndim=3, attribute_dtypes={},
    )
    chain = np.array([[k, 5, 5] for k in (2, 6, 10, 14, 18)], np.float32)
    edges = np.array([[k, k - 1] for k in range(1, 5)])
    left, anchors_left = sk.write_skeleton_chunk(level, (0, 0, 0), [
        {"segment_id": 7, "positions": chain, "edges": edges, "attributes": {},
         "anchors": {"end": 4}},
    ])
    right, anchors_right = sk.write_skeleton_chunk(level, (1, 0, 0), [
        {"segment_id": 7, "positions": chain + [20, 0, 0], "edges": edges, "attributes": {},
         "anchors": {"start": 0}},
    ])
    sk.write_skeleton_cross_chunk_links(
        level, [(anchors_left["end"], anchors_right["start"])], ndim=3,
    )
    build_object_index(level, left + right, ndim=3)
    sk.finalize_skeleton_store(root)


def test_chunked_skeleton_link_adds_an_edge_without_reparenting(tmp_path):
    path = tmp_path / "chunked.zv"
    _chunked_skeleton(path)
    a, b, _w, n = read_edges(get_resolution_level(open_store(str(path)), 0))
    assert n == 10
    assert _edge_set(a, b) == {(k, k + 1) for k in range(9)}
    cc = compute_connected_components(path)
    assert cc["n_components"] == 1
    assert bfs_distances(path, 0)["distances"].tolist() == list(range(10))


def test_weights_on_a_skeleton_store_name_the_links_they_lack(tmp_path):
    """A Y: only the second branch's link is stored, with its weight; the
    other three are implied and have none."""
    positions = np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0], [15, 5, 0], [15, 6, 0]], np.float32)
    edges = np.array([[1, 0], [2, 1], [3, 2], [4, 2]])
    path = tmp_path / "weighted.zv"
    write_graph(
        str(path), positions, edges, chunk_shape=(100.0,) * 3, kind="skeleton",
        link_attributes={"w": np.array([1.0, 2.0, 3.0, 4.0])},
    )
    assert compute_connected_components(path)["n_components"] == 1
    assert compute_k_core(path)["max_core"] == 1
    # Unit weights when the attribute is absent, as on a graph store.
    assert shortest_path(path, 0, 4, weight="missing")["cost"] == 3.0
    with pytest.raises(ValueError, match="implies"):
        shortest_path(path, 0, 4, weight="w")


def test_explicit_graph_edges_are_the_stored_records(tmp_path):
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 100, size=(60, 3)).astype(np.float32)
    edges = rng.integers(0, 60, size=(150, 2))
    edges = edges[edges[:, 0] != edges[:, 1]]
    path = tmp_path / "g.zv"
    write_graph(str(path), positions, edges, chunk_shape=(30.0,) * 3)
    a, b, _w, _n = read_edges(get_resolution_level(open_store(str(path)), 0))
    graph = read_graph(str(path))
    assert len(a) == graph["edge_count"]
    assert sorted(zip(a.tolist(), b.tolist())) == sorted(map(tuple, graph["edges"].tolist()))
