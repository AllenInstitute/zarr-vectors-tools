"""SWC stores: segment ids, a skeleton pyramid, and exports that stay exact.

An SWC store and a precomputed-ingested store share a links convention but
join a tree across chunks differently: the precomputed ingests duplicate the
vertex on each chunk face, SWC stores keep every vertex once and store a
crossing edge as a link record.  These tests pin down that the pyramid keeps
every SWC tree in one piece at every level, and that the exporters read each
layout with the reader made for it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree
from zarr_vectors.building import (
    create_object_attributes_array,
    get_resolution_level,
    open_store,
    read_all_object_manifests,
    read_chunk_fragment_attributes,
    read_object_attributes,
    write_object_attributes,
)

from zarr_vectors_tools.cli import main
from zarr_vectors_tools.convert.export.swc import export_swc
from zarr_vectors_tools.convert.ingest.swc import ingest_swc
from zarr_vectors_tools.multiresolution.skeleton_graph import split_components
from zarr_vectors_tools.multiresolution.skeleton_layout import (
    LAYOUT_LINKED,
    LAYOUT_SPLIT,
    skeleton_layout,
)
from zarr_vectors_tools.multiresolution.strategies.skeletons import (
    build_skeleton_pyramid,
    coarsen_skeleton_level,
)

CHUNK = "40,40,40"


# ===================================================================
# Helpers
# ===================================================================

def _write_forest(
    path: Path, trees: int = 6, nodes: int = 400, step: float = 2.0,
    extent: float = 200.0, seed: int = 0, comments: tuple[str, ...] = (),
) -> Path:
    """Random-walk trees, roots first and every parent before its child."""
    rng = np.random.default_rng(seed)
    lines = list(comments)
    nid = 0
    for _t in range(trees):
        pos = [rng.uniform(0.2 * extent, 0.8 * extent, 3)]
        parent = [-1]
        ids = [nid + 1]
        nid += 1
        while len(pos) < nodes:
            cur = int(rng.integers(len(pos)))
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            for _ in range(min(int(rng.integers(20, 80)), nodes - len(pos))):
                d = d + 0.35 * rng.normal(size=3)
                d /= np.linalg.norm(d)
                pos.append(pos[cur] + step * d)
                parent.append(cur)
                ids.append(nid + 1)
                nid += 1
                cur = len(pos) - 1
        for i, p in enumerate(pos):
            kind = 1 if parent[i] < 0 else 3
            par = -1 if parent[i] < 0 else ids[parent[i]]
            radius = 3.0 if parent[i] < 0 else 0.5 + (i % 5) * 0.25
            lines.append(f"{ids[i]} {kind} {p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {radius} {par}")
    path.write_text("\n".join(lines) + "\n")
    return path


def _read_swc(path: Path) -> tuple[np.ndarray, list[str]]:
    rows, comments = [], []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            comments.append(line)
        elif line.strip():
            rows.append([float(x) for x in line.split()[:7]])
    return np.asarray(rows), comments


def _parent_rows(data: np.ndarray) -> np.ndarray:
    row = {int(i): k for k, i in enumerate(data[:, 0].astype(np.int64))}
    return np.array([row.get(int(p), -1) for p in data[:, 6].astype(np.int64)])


def _trees(data: np.ndarray) -> list[np.ndarray]:
    """Each tree's rows, in the order its root appears."""
    parent = _parent_rows(data)
    up = np.where(parent < 0, np.arange(len(parent)), parent)
    for _ in range(64):
        nxt = up[up]
        if np.array_equal(nxt, up):
            break
        up = nxt
    return [np.flatnonzero(up == r) for r in np.flatnonzero(parent < 0)]


def _match(src: np.ndarray, out: np.ndarray) -> np.ndarray:
    """Source row of each exported row, by position."""
    dist, rows = cKDTree(src[:, 2:5]).query(out[:, 2:5])
    assert dist.max() < 1e-3
    return rows


def _assert_same_tree(src: np.ndarray, out: np.ndarray) -> None:
    """Same nodes, parents, radius and type; ids may be renumbered."""
    assert len(src) == len(out)
    rows = _match(src, out)
    assert len(set(rows.tolist())) == len(src)
    src_parent, out_parent = _parent_rows(src), _parent_rows(out)
    expected = src_parent[rows]
    got = np.where(out_parent >= 0, rows[out_parent], -1)
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_allclose(out[:, 5], src[rows, 5], atol=1e-3)
    np.testing.assert_array_equal(out[:, 1], src[rows, 1])


def _assert_decimated_from(src: np.ndarray, out: np.ndarray) -> None:
    """Every coarse node is a source node whose coarse parent is its nearest
    surviving source ancestor, and every tip and branch point survives."""
    rows = _match(src, out)
    kept = np.zeros(len(src), dtype=bool)
    kept[rows] = True
    src_parent, out_parent = _parent_rows(src), _parent_rows(out)
    for k, s in enumerate(rows):
        anc = src_parent[s]
        while anc >= 0 and not kept[anc]:
            anc = src_parent[anc]
        assert (rows[out_parent[k]] if out_parent[k] >= 0 else -1) == anc
    alive = np.zeros(len(src), dtype=bool)
    for tree in _trees(src):
        if kept[tree].any():
            alive[tree] = True
    n_children = np.bincount(src_parent[src_parent >= 0], minlength=len(src))
    assert kept[alive & (n_children != 1)].all()


def _level_swc(store: Path, level: int, tmp_path: Path) -> np.ndarray:
    out = tmp_path / f"level{level}.swc"
    assert main(["convert", str(store), str(out), "--level", str(level)]) == 0
    return _read_swc(out)[0]


def _skeleton_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    blob = path.read_bytes()
    nv, ne = (int(x) for x in np.frombuffer(blob[:8], np.uint32))
    vertices = np.frombuffer(blob[8:8 + 12 * nv], np.float32).reshape(-1, 3)
    edges = np.frombuffer(blob[8 + 12 * nv:8 + 12 * nv + 8 * ne], np.uint32).reshape(-1, 2)
    return vertices, edges


@pytest.fixture
def forest(tmp_path: Path) -> Path:
    return _write_forest(tmp_path / "forest.swc", comments=("# source: test", "# units: um"))


# ===================================================================
# Ingest: segment ids and the layout marker
# ===================================================================

class TestIngest:
    def test_segment_ids_number_the_trees_from_one(self, forest: Path, tmp_path: Path) -> None:
        store = tmp_path / "f.zv"
        ingest_swc(forest, store, (40.0, 40.0, 40.0))
        level0 = get_resolution_level(open_store(str(store)), 0)
        np.testing.assert_array_equal(
            read_object_attributes(level0, "segment_id"), np.arange(1, 7, dtype=np.uint64),
        )
        for oid, manifest in enumerate(read_all_object_manifests(level0)):
            for chunk, fragment in manifest:
                for name, expected in (("object_id", oid), ("segment_id", oid + 1)):
                    column = read_chunk_fragment_attributes(
                        level0, name, tuple(chunk), dtype=np.uint64,
                    )
                    assert int(column[fragment]) == expected
        assert skeleton_layout(str(store)) == LAYOUT_LINKED

    def test_plain_precomputed_ingest_records_the_linked_layout(self, tmp_path: Path) -> None:
        from zarr_vectors_tools.convert.ingest.precomputed_plain_skeletons import (
            PlainSkeletonInfo,
            run_ingest_plain,
        )

        class _Reader:
            info = PlainSkeletonInfo(
                base_url="mem://plain", transform=np.eye(4), vertex_attributes=[],
            )
            segment_ids = [11, 12]
            segment_properties_raw = None

            def read_skeleton(self, seg_id: int) -> dict:
                x = np.arange(12, dtype=np.float32) * 150 + seg_id
                vertices = np.column_stack([x, x * 0 + 10, x * 0 + 10]).astype(np.float32)
                return {"vertices": vertices,
                        "edges": np.array([[i, i - 1] for i in range(1, 12)])}

        store = tmp_path / "p.zv"
        run_ingest_plain(_Reader(), store, chunk_shape_nm=(500.0, 500.0, 500.0),
                         strides=[2], progress=False)
        # Every vertex once, crossing edges as [child, parent] records.
        assert skeleton_layout(str(store)) == LAYOUT_LINKED

    def test_stores_without_the_marker_are_told_apart(self, forest: Path, tmp_path: Path) -> None:
        from zarr_vectors.types import skeletons as sk
        from zarr_vectors.types.graphs import write_graph

        # SWC ingest, marker removed: the SWC header says linked.
        swc_store = tmp_path / "s.zv"
        ingest_swc(forest, swc_store, (40.0, 40.0, 40.0))
        root = open_store(str(swc_store), mode="r+")
        attrs = root.attrs.to_dict()
        attrs.pop("zarr_vectors_tools")
        meta_path = swc_store / "zarr.json"
        meta = json.loads(meta_path.read_text())
        meta["attributes"] = attrs
        meta_path.write_text(json.dumps(meta))
        assert skeleton_layout(str(swc_store)) == LAYOUT_LINKED

        # write_graph directly: no header, no fragment segment ids.
        bare = tmp_path / "bare.zv"
        write_graph(str(bare), np.array([[1.0, 1, 1], [60, 1, 1]], np.float32),
                    np.array([[1, 0]]), chunk_shape=(40.0, 40.0, 40.0), kind="skeleton")
        assert skeleton_layout(str(bare)) == LAYOUT_LINKED

        # Written chunk by chunk, as the precomputed ingests do: split.
        chunked = tmp_path / "c.zv"
        root, level0 = sk.init_skeleton_store(
            str(chunked), chunk_shape=(100.0,) * 3, bounds=([0.0] * 3, [100.0] * 3),
            ndim=3, attribute_dtypes={},
        )
        sk.write_skeleton_chunk(level0, (0, 0, 0), [{
            "segment_id": 5, "positions": np.array([[1, 1, 1], [2, 2, 2]], np.float32),
            "edges": np.array([[1, 0]]), "attributes": {},
        }])
        assert skeleton_layout(str(chunked)) == LAYOUT_SPLIT


# ===================================================================
# The pyramid keeps every tree in one piece
# ===================================================================

class TestPyramid:
    def test_every_level_has_one_root_per_tree(self, forest: Path, tmp_path: Path) -> None:
        store = tmp_path / "f.zv"
        assert main(["convert", str(forest), str(store), "--chunk-shape", CHUNK,
                     "--coarsen", "4,4", "--sparsity", "1,1", "--chunk-scale", "2,2"]) == 0
        src = _read_swc(forest)[0]
        somas = src[src[:, 6] == -1]
        counts = []
        for level in (0, 1, 2):
            out = _level_swc(store, level, tmp_path)
            roots = out[out[:, 6] == -1]
            assert len(roots) == 6
            # Each root is a soma, and still typed one: decimation never
            # re-roots a tree, and a kept vertex keeps its own SWC type.
            assert sorted(_match(somas, roots).tolist()) == list(range(6))
            assert (roots[:, 1] == 1).all() and int((out[:, 1] == 1).sum()) == 6
            if level:
                _assert_decimated_from(src, out)
            else:
                _assert_same_tree(src, out)
            counts.append(len(out))
        assert counts[0] > counts[1] > counts[2]

    def test_sparsity_drops_whole_trees(self, forest: Path, tmp_path: Path) -> None:
        store = tmp_path / "f.zv"
        assert main(["convert", str(forest), str(store), "--chunk-shape", CHUNK,
                     "--coarsen", "4,4", "--sparsity", "2,2", "--chunk-scale", "2,2",
                     "--sparsity-strategy", "length"]) == 0
        src = _read_swc(forest)[0]
        for level, n_trees in ((1, 3), (2, 2)):
            out = _level_swc(store, level, tmp_path)
            assert int((out[:, 6] == -1).sum()) == n_trees
            _assert_decimated_from(src, out)

    def test_pyramid_command_builds_on_an_swc_store(self, forest: Path, tmp_path: Path) -> None:
        store = tmp_path / "f.zv"
        assert main(["convert", str(forest), str(store), "--chunk-shape", CHUNK]) == 0
        assert main(["pyramid", str(store), "--coarsen", "4"]) == 0
        out = _level_swc(store, 1, tmp_path)
        assert int((out[:, 6] == -1).sum()) == 6
        _assert_decimated_from(_read_swc(forest)[0], out)

    @pytest.mark.parametrize("scale", [1, 2, 3])
    def test_edges_that_skip_chunks(self, tmp_path: Path, scale: int) -> None:
        # Every edge is longer than a chunk, so each crosses one or more
        # chunk boundaries, and some join chunks that are not neighbours.
        rows, nid = [], 0
        rng = np.random.default_rng(3)
        for t in range(3):
            pos = [np.array([20.0 + 60 * t, 20.0, 20.0])]
            parent = [-1]
            for k in range(1, 25):
                p = int(rng.integers(max(0, k - 3), k))
                pos.append(pos[p] + rng.uniform(-35, 35, 3) + [25, 0, 0])
                parent.append(p)
            for k, p in enumerate(pos):
                par = -1 if parent[k] < 0 else nid + parent[k] + 1
                rows.append(f"{nid + k + 1} 3 {p[0]:.3f} {p[1]:.3f} {p[2]:.3f} 1.0 {par}")
            nid += len(pos)
        src_path = tmp_path / "long.swc"
        src_path.write_text("\n".join(rows) + "\n")
        store = tmp_path / "long.zv"
        ingest_swc(src_path, store, (20.0, 20.0, 20.0))
        build_skeleton_pyramid(store, strides=[2, 2], chunk_scale_factors=[scale, scale])
        src = _read_swc(src_path)[0]
        for level in (1, 2):
            out = tmp_path / f"l{level}.swc"
            summary = export_swc(store, out, level=level)
            assert summary["root_count"] == 3
            _assert_decimated_from(src, _read_swc(out)[0])

    def test_vertex_count_describes_each_level(self, forest: Path, tmp_path: Path) -> None:
        store = tmp_path / "f.zv"
        ingest_swc(forest, store, (40.0, 40.0, 40.0))
        level0 = get_resolution_level(open_store(str(store), mode="r+"), 0)
        create_object_attributes_array(level0, "vertex_count", dtype="int64")
        write_object_attributes(level0, "vertex_count", np.full(6, 400, dtype=np.int64))
        build_skeleton_pyramid(store, strides=[4], chunk_scale_factors=[2],
                               sparsity_factors=[2.0])
        level1 = get_resolution_level(open_store(str(store)), 1)
        out = tmp_path / "objs"
        stored = read_object_attributes(level1, "vertex_count")
        manifests = read_all_object_manifests(level1)
        alive = [oid for oid, m in enumerate(manifests) if m]
        assert len(alive) == 3
        export_swc(store, out, level=1, object_ids=alive)
        for oid in alive:
            rows, _ = _read_swc(out / f"{oid + 1}.swc")
            assert int(stored[oid]) == len(rows) < 400

    def test_attribute_sparsity_keeps_the_largest(self, forest: Path, tmp_path: Path) -> None:
        store = tmp_path / "f.zv"
        ingest_swc(forest, store, (40.0, 40.0, 40.0))
        values = np.array([5.0, 1.0, 9.0, 3.0, 7.0, 2.0])
        coarsen_skeleton_level(
            store, 0, 1, stride=4, sparsity_factor=2.0,
            sparsity_strategy="attribute", attribute_values=values,
        )
        level1 = get_resolution_level(open_store(str(store)), 1)
        alive = [oid for oid, m in enumerate(read_all_object_manifests(level1)) if m]
        assert alive == [0, 2, 4]

    def test_a_store_without_segment_ids_is_refused_clearly(self, tmp_path: Path) -> None:
        from zarr_vectors.types.graphs import write_graph

        store = tmp_path / "bare.zv"
        write_graph(str(store), np.array([[1.0, 1, 1], [60, 1, 1]], np.float32),
                    np.array([[1, 0]]), chunk_shape=(40.0, 40.0, 40.0), kind="skeleton")
        with pytest.raises(ValueError, match="re-ingest"):
            coarsen_skeleton_level(store, 0, 1, stride=2)

    @pytest.mark.slow
    def test_process_pool_matches_serial(self, forest: Path, tmp_path: Path) -> None:
        from tests.test_precomputed_skeletons import _ppool_executor, _tree_sha

        shas = []
        for name, executor in (("serial", None), ("pool", _ppool_executor)):
            store = tmp_path / name / "f.zv"
            ingest_swc(forest, store, (40.0, 40.0, 40.0))
            build_skeleton_pyramid(store, strides=[4, 4], chunk_scale_factors=[2, 2],
                                   sparsity_factors=[1.0, 2.0], executor=executor)
            shas.append({k: v for k, v in _tree_sha(store).items() if k != "zarr.json"})
        assert shas[0] == shas[1]


# ===================================================================
# Exports read each layout with its own reader
# ===================================================================

class TestExport:
    def test_per_object_swc_is_exact(self, forest: Path, tmp_path: Path) -> None:
        store = tmp_path / "f.zv"
        ingest_swc(forest, store, (40.0, 40.0, 40.0))
        out = tmp_path / "objs"
        summary = export_swc(store, out, object_ids=list(range(6)))
        assert [Path(f).name for f in summary["files"]] == [f"{i}.swc" for i in range(1, 7)]
        src = _read_swc(forest)[0]
        for oid, rows in enumerate(_trees(src)):
            got = _read_swc(out / f"{oid + 1}.swc")[0]
            _assert_same_tree(src[rows], got)
            assert got[got[:, 6] == -1][0, 1] == 1  # the root is the soma

    def test_per_object_swc_of_a_coarse_level(self, forest: Path, tmp_path: Path) -> None:
        store = tmp_path / "f.zv"
        ingest_swc(forest, store, (40.0, 40.0, 40.0))
        build_skeleton_pyramid(store, strides=[4], chunk_scale_factors=[2])
        summary = export_swc(store, tmp_path / "objs", level=1, object_ids=list(range(6)))
        assert summary["root_count"] == 6

    @pytest.mark.parametrize("level", [0, 1])
    def test_precomputed_export_writes_one_tree_per_segment(
        self, forest: Path, tmp_path: Path, level: int,
    ) -> None:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        store = tmp_path / "f.zv"
        assert main(["convert", str(forest), str(store), "--chunk-shape", CHUNK,
                     "--coarsen", "4", "--sparsity", "1", "--chunk-scale", "2"]) == 0
        layer = tmp_path / "layer"
        assert main(["convert", str(store), str(layer), "--format", "precomputed",
                     "--level", str(level)]) == 0
        src = _read_swc(forest)[0]
        src_parent = _parent_rows(src)
        for segment, rows in enumerate(_trees(src), start=1):
            vertices, edges = _skeleton_file(layer / "skeletons" / str(segment))
            n = len(vertices)
            graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
            assert len(edges) == n - 1
            assert connected_components(graph, directed=False)[0] == 1
            if level == 0:
                # The edges are the source tree's, vertex for vertex.
                as_rows = np.column_stack([np.zeros((n, 2)), vertices])
                match = rows[_match(src[rows], as_rows)]
                got = {frozenset((int(match[a]), int(match[b]))) for a, b in edges}
                want = {frozenset((int(r), int(src_parent[r]))) for r in rows if src_parent[r] >= 0}
                assert got == want

    def test_swc_export_restores_the_comment_lines(self, forest: Path, tmp_path: Path) -> None:
        store = tmp_path / "f.zv"
        ingest_swc(forest, store, (40.0, 40.0, 40.0))
        out = tmp_path / "out.swc"
        export_swc(store, out)
        expected = ["# source: test", "# units: um", "# SWC exported by zarr-vectors"]
        assert _read_swc(out)[1] == expected
        # A second round trip keeps one provenance line, not two.
        again = tmp_path / "again.zv"
        ingest_swc(out, again, (40.0, 40.0, 40.0))
        export_swc(again, tmp_path / "again.swc")
        assert _read_swc(tmp_path / "again.swc")[1] == expected
        export_swc(store, tmp_path / "objs", object_ids=[2])
        assert _read_swc(tmp_path / "objs" / "3.swc")[1] == expected


def test_split_components_roots_each_tree_where_asked() -> None:
    positions = np.zeros((5, 3))
    # Path 0-1-2-3-4 and the true root is 2.
    edges = np.array([[0, 1], [1, 2], [3, 2], [4, 3]])
    (piece,) = split_components(positions, edges, vertex_ids=np.arange(5), roots=[2])
    ids = piece["vertex_ids"]
    parent = {int(ids[c]): int(ids[p]) for c, p in piece["edges"]}
    assert parent == {1: 2, 3: 2, 0: 1, 4: 3}
    # Without roots, the lowest index is the root, as before.
    (piece,) = split_components(positions, edges, vertex_ids=np.arange(5))
    assert int(piece["vertex_ids"][0]) == 0
