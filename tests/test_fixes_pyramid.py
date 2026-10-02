"""Regression tests for the pyramid fixes: one class per fixed behaviour.

Each test builds a small store, reads it back through core's public building
surface only, and checks what used to go wrong: coarse graph edges that no
level-0 edge produced, an empty ``links/0`` on point levels, cross-level links
that were dropped, misplaced or pointed at another object, line vertices stored
in a chunk that does not contain them, one-vertex "lines", an OME scale of 1 on
levels whose chunks double, a streamline pyramid refused for a missing
``segment_id``, and a docstring claiming per-vertex attributes were dropped.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from zarr_vectors.building import (
    CAP_MULTISCALE_LINKS,
    get_level_chunk_shape,
    get_resolution_level,
    list_chunk_keys,
    list_link_deltas,
    list_resolution_levels,
    open_store,
    read_all_object_manifests,
    read_chunk_attributes,
    read_chunk_fragment_attributes,
    read_chunk_vertices,
    read_level_metadata,
    read_link_arrays,
    read_link_attributes,
    read_root_metadata,
)
from zarr_vectors.types.graphs import write_graph
from zarr_vectors.types.lines import write_lines
from zarr_vectors.types.points import write_points
from zarr_vectors.types.polylines import write_polylines

from tests._source_helpers import stamp_segment_id_from_manifests
from zarr_vectors_tools.cli import main
from zarr_vectors_tools.multiresolution.coarsen import (
    build_pyramid,
    read_coarsening_record,
)
from zarr_vectors_tools.multiresolution.refresh import rebuild_pyramid_from_level
from zarr_vectors_tools.multiresolution.strategies.points import (
    build_point_subset_pyramid,
)

# ---------------------------------------------------------------------------
# Reading a level back as flat arrays
# ---------------------------------------------------------------------------


class Level:
    """One level as flat arrays: rows in chunk order, fragments in order."""

    def __init__(self, store: Path, level: int) -> None:
        root = open_store(str(store))
        lg = get_resolution_level(root, level)
        self.group = lg
        self.chunk_shape = np.asarray(
            get_level_chunk_shape(read_root_metadata(root), read_level_metadata(root, level)),
            dtype=np.float64,
        )
        self.start: dict[tuple, int] = {}
        self.fragment_rows: dict[tuple, np.ndarray] = {}
        positions, chunk_of_row = [], []
        run = 0
        for cc in sorted(tuple(int(c) for c in k) for k in list_chunk_keys(lg)):
            fragments = read_chunk_vertices(lg, cc, dtype=np.float32, ndim=3)
            self.start[cc] = run
            for f, frag in enumerate(fragments):
                self.fragment_rows[(cc, f)] = np.arange(run, run + len(frag))
                positions.append(np.asarray(frag, dtype=np.float64))
                chunk_of_row += [cc] * len(frag)
                run += len(frag)
        self.pos = np.concatenate(positions) if positions else np.zeros((0, 3))
        self.row_chunk = np.asarray(chunk_of_row, dtype=np.int64).reshape(-1, 3)
        self.obj = np.full(len(self.pos), -1, dtype=np.int64)
        self.manifests = read_all_object_manifests(lg)
        for oid, manifest in enumerate(self.manifests):
            for cc, f in manifest:
                self.obj[self.fragment_rows[(tuple(int(c) for c in cc), int(f))]] = oid
        self.present = {o for o, m in enumerate(self.manifests) if m}

    def rows(self, chunks: np.ndarray, vi: np.ndarray) -> np.ndarray:
        """Global rows of ``(chunk, local row)`` endpoints; -1 if unresolvable."""
        counts = defaultdict(int)
        for cc in self.row_chunk.tolist():
            counts[tuple(cc)] += 1
        out = np.full(vi.shape, -1, dtype=np.int64)
        for idx in np.ndindex(*vi.shape):
            cc = tuple(int(c) for c in chunks[idx])
            local = int(vi[idx])
            if cc in self.start and 0 <= local < counts[cc]:
                out[idx] = self.start[cc] + local
        return out

    def edges(self) -> tuple[np.ndarray, np.ndarray]:
        """``(rows, chunks)`` of every ``links/0`` record."""
        ch, vi = read_link_arrays(self.group, delta=0)
        if vi.size == 0:
            return np.zeros((0, 2), dtype=np.int64), np.zeros((0, 2, 3), dtype=np.int64)
        return self.rows(ch, vi), ch


def _components(n: int, edges: np.ndarray) -> np.ndarray:
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges.tolist():
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(i) for i in range(n)])


def _parents(store: Path, level: int) -> tuple[np.ndarray, Level, Level]:
    """``parent[row]`` of every row of ``level`` from its ``links/+1`` family."""
    fine, coarse = Level(store, level), Level(store, level + 1)
    ch, vi = read_link_arrays(fine.group, delta=1)
    parent = np.full(len(fine.pos), -1, dtype=np.int64)
    if vi.size:
        f = fine.rows(ch[:, :1], vi[:, :1])[:, 0]
        c = coarse.rows(ch[:, 1:], vi[:, 1:])[:, 0]
        assert (f >= 0).all() and (c >= 0).all(), "a +1 record does not resolve"
        assert len(np.unique(f)) == len(f), "a vertex has two parents"
        parent[f] = c
    return parent, fine, coarse


# ---------------------------------------------------------------------------
# 1. Graph pyramids map the graph's own edges
# ---------------------------------------------------------------------------


def _graph_store(path: Path) -> Path:
    """A radius graph in many components, some passing within a bin of each other."""
    rng = np.random.default_rng(3)
    pos = rng.uniform(0.0, 200.0, size=(700, 3)).astype(np.float32)
    d = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    a, b = np.nonzero(np.triu(d < 11.0, k=1))
    edges = np.stack([a, b], axis=1)
    write_graph(
        str(path), pos, edges, chunk_shape=(50.0, 50.0, 50.0),
        bin_shape=(5.0, 5.0, 5.0),
        vertex_attributes={"mass": rng.uniform(0, 1, len(pos)).astype(np.float32)},
        link_attributes={
            "length": d[a, b].astype(np.float32),
            "kind": (np.arange(len(edges)) % 3).astype(np.int32),
        },
    )
    return path


class TestGraphPyramidMapsItsOwnEdges:

    @pytest.mark.parametrize("chunk_scale", [1, 2])
    def test_coarse_edges_are_images_of_fine_edges(self, tmp_path: Path, chunk_scale: int) -> None:
        store = _graph_store(tmp_path / "g.zv")
        build_pyramid(
            str(store), factors=[(2.0, 1.0), (2.0, 1.0), (2.0, 1.0)],
            chunk_scale_factors=[chunk_scale] * 3, cross_level_storage="explicit",
        )
        for k in range(3):
            parent, fine, coarse = _parents(store, k)
            assert (parent >= 0).all(), "every vertex of a one-object graph has a parent"
            f_edges, _ = fine.edges()
            c_edges, c_chunks = coarse.edges()

            # Exactly the images, no self-loops, no duplicates, nothing invented.
            img = np.sort(parent[f_edges], axis=1)
            img = {tuple(e) for e in img.tolist() if e[0] != e[1]}
            got = [tuple(e) for e in np.sort(c_edges, axis=1).tolist()]
            assert len(got) == len(set(got)), "duplicate edge records"
            assert all(a != b for a, b in got), "self-loop records"
            assert set(got) == img

            # Each coarse vertex is the centroid of one (component, bin) group.
            comp = _components(len(fine.pos), f_edges)
            bin_shape = np.asarray(read_level_metadata(open_store(str(store)), k + 1).bin_shape)
            bins = np.floor(fine.pos / bin_shape).astype(np.int64)
            groups = defaultdict(set)
            for row, p in enumerate(parent.tolist()):
                groups[p].add((int(comp[row]), *bins[row].tolist()))
            assert all(len(g) == 1 for g in groups.values())
            assert len({next(iter(g)) for g in groups.values()}) == len(groups)
            for p in range(len(coarse.pos)):
                np.testing.assert_allclose(
                    coarse.pos[p], fine.pos[parent == p].mean(axis=0), atol=1e-3,
                )

            # Components are images of components: no merge, no split.
            c_comp = _components(len(coarse.pos), c_edges)
            image = defaultdict(set)
            for row in range(len(fine.pos)):
                image[comp[row]].add(c_comp[parent[row]])
            assert all(len(v) == 1 for v in image.values()), "a component split"
            assert len({next(iter(v)) for v in image.values()}) == len(image), "components merged"

            # Every vertex sits in its chunk, and records crossing a chunk face
            # are written as cross-chunk records.
            assert (np.floor(coarse.pos / coarse.chunk_shape).astype(int) == coarse.row_chunk).all()
            crossing = (c_chunks[:, 0] != c_chunks[:, 1]).any(axis=1)
            ends = coarse.row_chunk[c_edges[crossing]]
            assert (ends[:, 0] != ends[:, 1]).any(axis=1).all()
            if chunk_scale == 1 and k == 0:
                assert crossing.any(), "fixture must have chunk-crossing coarse edges"

    def test_edge_attributes_follow_the_edges(self, tmp_path: Path) -> None:
        store = _graph_store(tmp_path / "g.zv")
        build_pyramid(str(store), factors=[(2.0, 1.0)], cross_level_storage="explicit")
        parent, fine, coarse = _parents(store, 0)
        f_edges, _ = fine.edges()
        c_edges, _ = coarse.edges()
        f_len = read_link_attributes(fine.group, "length", delta=0)
        f_kind = read_link_attributes(fine.group, "kind", delta=0)
        c_len = read_link_attributes(coarse.group, "length", delta=0)
        c_kind = read_link_attributes(coarse.group, "kind", delta=0)
        assert len(c_len) == len(c_kind) == len(c_edges)
        merged = defaultdict(list)
        for i, (a, b) in enumerate(np.sort(parent[f_edges], axis=1).tolist()):
            if a != b:
                merged[(a, b)].append(i)
        for j, edge in enumerate(np.sort(c_edges, axis=1).tolist()):
            members = merged[tuple(edge)]
            assert c_len[j] == pytest.approx(float(np.mean(f_len[members])), rel=1e-5)
            assert c_kind[j] in set(f_kind[members].tolist())

    def test_read_graph_sees_the_mapped_edges(self, tmp_path: Path) -> None:
        from zarr_vectors.types.graphs import read_graph

        store = _graph_store(tmp_path / "g.zv")
        build_pyramid(str(store), factors=[(2.0, 1.0), (2.0, 1.0)], cross_level_storage="none")
        g0 = read_graph(str(store), level=0)
        for level in (1, 2):
            g = read_graph(str(store), level=level)
            lengths = np.linalg.norm(
                g["positions"][g["edges"][:, 0]] - g["positions"][g["edges"][:, 1]], axis=1,
            )
            # An image edge is at most a level-0 edge plus a bin diagonal each end.
            bin_edge = 5.0 * 2 ** level
            longest0 = np.linalg.norm(
                g0["positions"][g0["edges"][:, 0]] - g0["positions"][g0["edges"][:, 1]], axis=1,
            ).max()
            assert lengths.max() <= longest0 + 2 * bin_edge * np.sqrt(3)
            assert len(_np_components(g)) == len(_np_components(g0))


def _np_components(g: dict) -> set:
    return set(_components(len(g["positions"]), np.asarray(g["edges"])).tolist())


# ---------------------------------------------------------------------------
# 2. Point levels carry no links family
# ---------------------------------------------------------------------------


class TestPointLevelsHaveNoLinks:

    def test_no_links_family_on_coarse_point_levels(self, tmp_path: Path) -> None:
        store = tmp_path / "p.zv"
        rng = np.random.default_rng(0)
        write_points(
            str(store), rng.uniform(0, 300, (2000, 3)).astype(np.float32),
            chunk_shape=(100.0, 100.0, 100.0), bin_shape=(10.0, 10.0, 10.0),
            object_ids=np.repeat(np.arange(200), 10),
        )
        build_pyramid(str(store), factors=[(2.0, 2.0), (2.0, 2.0)], cross_level_storage="none")
        root = open_store(str(store))
        for level in (1, 2):
            lg = get_resolution_level(root, level)
            assert list_link_deltas(lg) == []
            assert not (store / str(level) / "links").exists()
            assert "links" not in (read_level_metadata(root, level).arrays_present or [])


# ---------------------------------------------------------------------------
# 3. Cross-level links: one parent per vertex, same object, exact mirror
# ---------------------------------------------------------------------------


def _id_points(path: Path) -> Path:
    rng = np.random.default_rng(11)
    # Negative coordinates: on a chunk-scale-2 grid, coarse chunk -1 and fine
    # chunk -1 share coords but are different places.
    write_points(
        str(path), rng.uniform(-400.0, 400.0, size=(3000, 3)).astype(np.float32),
        chunk_shape=(100.0, 100.0, 100.0), bin_shape=(25.0, 25.0, 25.0),
        object_ids=np.repeat(np.arange(300), 10),
    )
    return path


class TestCrossLevelLinks:

    @pytest.mark.parametrize("chunk_scale", [1, 2])
    def test_every_vertex_links_to_its_own_objects_row(
        self, tmp_path: Path, chunk_scale: int,
    ) -> None:
        store = _id_points(tmp_path / "p.zv")
        build_pyramid(
            str(store), factors=[(2.0, 2.0), (2.0, 2.0)],
            chunk_scale_factors=[chunk_scale] * 2, sparsity_seed=1,
            cross_level_storage="explicit",
        )
        for k in (0, 1):
            parent, fine, coarse = _parents(store, k)
            surviving = np.isin(fine.obj, sorted(coarse.present))
            assert (parent[surviving] >= 0).all(), "a surviving vertex has no parent"
            assert (parent[~surviving] == -1).all()
            assert (coarse.obj[parent[surviving]] == fine.obj[surviving]).all()

            # -1 is the exact inverse of +1, and every endpoint decodes to the
            # chunk that holds its position.
            ch, vi = read_link_arrays(coarse.group, delta=-1)
            c_rows = coarse.rows(ch[:, :1], vi[:, :1])[:, 0]
            f_rows = fine.rows(ch[:, 1:], vi[:, 1:])[:, 0]
            assert (c_rows >= 0).all() and (f_rows >= 0).all(), "a -1 record dangles"
            forward = {(f, int(parent[f])) for f in np.flatnonzero(parent >= 0).tolist()}
            assert set(zip(f_rows.tolist(), c_rows.tolist())) == forward
            assert len(f_rows) == len(forward)
            for lv, rows, chunks in ((coarse, c_rows, ch[:, 0]), (fine, f_rows, ch[:, 1])):
                assert (np.floor(lv.pos[rows] / lv.chunk_shape).astype(int) == chunks).all()

    def test_root_records_what_was_written(self, tmp_path: Path) -> None:
        explicit = _id_points(tmp_path / "e.zv")
        build_pyramid(str(explicit), factors=[(2.0, 1.0)], cross_level_storage="explicit")
        meta = read_root_metadata(open_store(str(explicit)))
        assert (meta.cross_level_depth, meta.cross_level_storage) == (1, "explicit")

        implicit = _id_points(tmp_path / "i.zv")
        build_pyramid(str(implicit), factors=[(2.0, 1.0)], cross_level_storage="implicit")
        meta = read_root_metadata(open_store(str(implicit)))
        assert (meta.cross_level_depth, meta.cross_level_storage) == (1, "implicit")

        # The polyline coarsener writes none, whatever was asked for.
        streamlines = _core_streamlines(tmp_path / "s.zv")
        summary = build_pyramid(
            str(streamlines), factors=[(1.0, 2.0)], cross_level_storage="explicit",
            rdp_tolerances=[0.5],
        )
        meta = read_root_metadata(open_store(str(streamlines)))
        assert (meta.cross_level_depth, meta.cross_level_storage) == (0, "none")
        assert CAP_MULTISCALE_LINKS not in meta.format_capabilities
        assert (summary["cross_level_depth"], summary["cross_level_storage"]) == (0, "none")

        subset = tmp_path / "sub.zv"
        write_points(
            str(subset), np.random.default_rng(0).uniform(0, 100, (800, 3)).astype(np.float32),
            chunk_shape=(50.0, 50.0, 50.0),
        )
        build_point_subset_pyramid(str(subset), levels=2, divisor=4.0, seed=0)
        meta = read_root_metadata(open_store(str(subset)))
        assert (meta.cross_level_depth, meta.cross_level_storage) == (0, "none")

    def test_rebuild_without_links_clears_the_old_ones(self, tmp_path: Path) -> None:
        store = _id_points(tmp_path / "p.zv")
        build_pyramid(str(store), factors=[(2.0, 1.0)], cross_level_storage="explicit")
        assert 1 in list_link_deltas(get_resolution_level(open_store(str(store)), 0))
        build_pyramid(str(store), factors=[(2.0, 1.0)], cross_level_storage="none")
        root = open_store(str(store))
        assert list_link_deltas(get_resolution_level(root, 0)) == []
        assert read_root_metadata(root).cross_level_storage == "none"


# ---------------------------------------------------------------------------
# 4. Lines: vertices in their own chunk, collapsed lines dropped
# ---------------------------------------------------------------------------


def _lines_store(path: Path) -> Path:
    rng = np.random.default_rng(5)
    a = rng.uniform(0.0, 400.0, size=(1500, 3))
    b = a + rng.normal(scale=15.0, size=a.shape)
    write_lines(
        str(path), np.stack([a, b], axis=1).astype(np.float32),
        chunk_shape=(100.0, 100.0, 100.0), bin_shape=(10.0, 10.0, 10.0),
    )
    return path


class TestLinePyramids:

    def test_bins_that_do_not_divide_the_chunk(self, tmp_path: Path) -> None:
        store = _lines_store(tmp_path / "l.zv")
        summary = build_pyramid(
            str(store), factors=[(2.0, 1.0)] * 3, cross_level_storage="explicit",
        )
        root = open_store(str(store))
        for level in (1, 2, 3):
            lv = Level(store, level)
            below = Level(store, level - 1)
            # Bin 80 in a 100 chunk at level 3: still every vertex in its chunk.
            assert (np.floor(lv.pos / lv.chunk_shape).astype(int) == lv.row_chunk).all()
            # No one-vertex lines: a line whose ends merged is dropped and counted.
            sizes = np.bincount(lv.obj[lv.obj >= 0])
            assert set(sizes[sizes > 0].tolist()) == {2}
            collapsed = summary["level_specs"][level - 1]["objects_collapsed"]
            assert len(lv.present) + collapsed == len(below.present)
            assert read_coarsening_record(root, level)["collapsed_objects"] == collapsed
            if level == 3:
                assert collapsed > 0, "fixture must collapse some lines"
            # A line split over two chunks is joined by one record, first
            # vertex to second, within the object.
            edges, _ = lv.edges()
            split = {o for o in lv.present if len(lv.manifests[o]) == 2}
            assert {int(lv.obj[a]) for a, _ in edges.tolist()} == split
            assert len(edges) == len(split)
            for a, b in edges.tolist():
                assert lv.obj[a] == lv.obj[b]
                first = lv.fragment_rows[tuple(lv.manifests[lv.obj[a]][0])]
                assert a in first and b not in first
        for k in (0, 1, 2):
            parent, fine, coarse = _parents(store, k)
            alive = np.isin(fine.obj, sorted(coarse.present))
            assert (parent[alive] >= 0).all()
            assert (coarse.obj[parent[alive]] == fine.obj[alive]).all()


# ---------------------------------------------------------------------------
# 5. OME scale of streamline levels whose chunks grow
# ---------------------------------------------------------------------------


def _core_streamlines(path: Path, n: int = 40, *, segment_ids: bool = True) -> Path:
    """Written by core's ``write_polylines``, which writes no
    ``fragment_attributes/segment_id``; ``segment_ids`` stamps them as an
    ingest does."""
    rng = np.random.default_rng(2)
    lines = []
    for _ in range(n):
        start = rng.uniform(0, 300, 3)
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        t = np.linspace(0, 150, 60)[:, None]
        lines.append((start + t * direction + np.sin(t / 7.0) * 3.0).astype(np.float32))
    write_polylines(str(path), lines, chunk_shape=(100.0, 100.0, 100.0), geometry_type="streamline")
    if segment_ids:
        stamp_segment_id_from_manifests(path)
    return path


def _scales(store: Path) -> list[list[float]]:
    root = open_store(str(store))
    out = []
    for ds in root.attrs.to_dict()["multiscales"][0]["datasets"]:
        transforms = ds["coordinateTransformations"]
        out.append(next(t["scale"] for t in transforms if t["type"] == "scale"))
    return out


class TestStreamlineScale:

    def test_explicit_tolerance_levels_scale_with_their_chunks(self, tmp_path: Path) -> None:
        store = _core_streamlines(tmp_path / "s.zv")
        build_pyramid(
            str(store), factors=[(1.0, 1.0), (1.0, 1.0)], chunk_scale_factors=[2, 2],
            rdp_tolerances=[0.5, 1.0],
        )
        assert _scales(store) == [[1.0] * 3, [2.0] * 3, [4.0] * 3]
        root = open_store(str(store))
        counts = [read_level_metadata(root, lv).vertex_count for lv in (1, 2)]

        # A refresh reads the factor back from the bin and rebuilds the same levels.
        rebuild_pyramid_from_level(open_store(str(store), mode="r+"), 0)
        root = open_store(str(store))
        assert _scales(store) == [[1.0] * 3, [2.0] * 3, [4.0] * 3]
        assert [read_level_metadata(root, lv).vertex_count for lv in (1, 2)] == counts
        assert read_coarsening_record(root, 2)["rdp_tolerance"] == 1.0

    def test_derived_tolerance_keeps_the_factors_bin(self, tmp_path: Path) -> None:
        # The bin sets a derived tolerance, so it must not grow with the chunk.
        store = _core_streamlines(tmp_path / "s.zv")
        build_pyramid(str(store), factors=[(2.0, 1.0)], chunk_scale_factors=[4])
        assert _scales(store)[1] == [2.0] * 3


# ---------------------------------------------------------------------------
# 6. Streamline pyramids on stores written directly by core
# ---------------------------------------------------------------------------


class TestCoreWrittenStreamlines:

    def test_pyramid_stamps_the_missing_segment_ids(self, tmp_path: Path) -> None:
        store = _core_streamlines(tmp_path / "s.zv", segment_ids=False)
        assert main([
            "pyramid", str(store), "--coarsen", "1,1", "--sparsity", "2,2",
            "--rdp-tolerance", "0.5,1", "--chunk-scale", "2,2",
        ]) == 0
        root = open_store(str(store))
        lg = get_resolution_level(root, 0)
        assert "fragment_attributes" in read_level_metadata(root, 0).arrays_present
        for oid, manifest in enumerate(read_all_object_manifests(lg)):
            for cc, f in manifest:
                seg = read_chunk_fragment_attributes(lg, "segment_id", cc, dtype=np.uint64)
                assert int(seg[f]) == oid
        assert list_resolution_levels(root) == [0, 1, 2]


# ---------------------------------------------------------------------------
# 7. build_pyramid carries per-vertex attributes (the subset builder's
#    docstring said it dropped them)
# ---------------------------------------------------------------------------


class TestVertexAttributesCarried:

    def test_bin_mean_for_floats_first_value_otherwise(self, tmp_path: Path) -> None:
        store = tmp_path / "p.zv"
        rng = np.random.default_rng(4)
        pos = rng.uniform(0, 100, (600, 3)).astype(np.float32)
        write_points(
            str(store), pos, chunk_shape=(50.0, 50.0, 50.0), bin_shape=(5.0, 5.0, 5.0),
            object_ids=np.zeros(600, dtype=np.int64),
            vertex_attributes={
                "val": rng.uniform(0, 1, 600).astype(np.float32),
                "code": rng.integers(0, 4, 600).astype(np.int32),
            },
        )
        build_pyramid(str(store), factors=[(4.0, 1.0)], cross_level_storage="explicit")
        parent, fine, coarse = _parents(store, 0)

        def column(lv: Level, name: str, dtype) -> np.ndarray:
            parts = []
            for cc in sorted(lv.start, key=lv.start.get):
                parts += list(read_chunk_attributes(lv.group, name, cc, dtype=dtype))
            return np.concatenate(parts)

        f_val, c_val = column(fine, "val", np.float32), column(coarse, "val", np.float32)
        f_code, c_code = column(fine, "code", np.int32), column(coarse, "code", np.int32)
        for p in range(len(coarse.pos)):
            members = parent == p
            assert c_val[p] == pytest.approx(float(f_val[members].mean()), rel=1e-5)
            assert c_code[p] in set(f_code[members].tolist())
        assert "dropped" not in (build_point_subset_pyramid.__doc__ or "")
