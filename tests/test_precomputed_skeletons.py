"""End-to-end tests for the precomputed-`.frags` skeleton ingest workflow.

These exercise the ETL driver (extract → per-chunk write → object-index
reduce → pyramid) via the in-memory reader, plus cross-chunk-edge recovery
and coordinate alignment.  The reader/driver live in
``zarr_vectors_tools.ingest.precomputed_skeletons``; the format read-back
(``read_skeleton_by_segment_id``) and coarsening it calls live in the core
``zarr_vectors`` package.
"""
from __future__ import annotations

import shutil
import tempfile

import numpy as np
import pytest

from zarr_vectors.core.arrays import read_cross_chunk_links
from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors.types import skeletons as sk


@pytest.fixture
def tmp_store():
    d = tempfile.mkdtemp(suffix=".zv")
    shutil.rmtree(d)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _tree_sha(store_dir):
    import hashlib
    import os
    out = {}
    for root, _dirs, files in os.walk(store_dir):
        for fn in files:
            fp = os.path.join(root, fn)
            h = hashlib.sha256()
            with open(fp, "rb") as f:
                for blk in iter(lambda: f.read(1 << 20), b""):
                    h.update(blk)
            out[os.path.relpath(fp, store_dir)] = h.hexdigest()
    return out


def _ppool_executor(func, items, shared=None):
    """Parallel map over a ProcessPoolExecutor — exercises pickling of the
    top-level L0 + pyramid workers, their payloads, and the shared
    (scattered-once) reader/plan across processes."""
    items = list(items)
    if not items:
        return []
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    with ProcessPoolExecutor(max_workers=2) as ex:
        return list(ex.map(partial(func, shared=shared), items))


def _flywire_cutout_reader():
    """A small flywire-shaped in-memory cutout spanning multiple chunks."""
    from zarr_vectors_tools.ingest.precomputed_skeletons import (
        InMemoryFragsReader, SkeletonInfo, enumerate_frag_keys,
    )
    info = SkeletonInfo(
        base_url="mem://x", resolution_nm=(32.0, 32.0, 40.0),
        chunk_size_nm=(16384.0, 16384.0, 20480.0),
        vertex_attributes=[
            {"id": "radius", "data_type": "float32", "num_components": 1},
        ])
    cs = np.array(info.chunk_size_nm)
    rng = np.random.default_rng(0)
    anchor = (17910, 8912, 3088)
    counts = (2, 2, 1)
    keys = enumerate_frag_keys(info, anchor, counts)
    from zarr_vectors_tools.ingest.precomputed_skeletons import parse_frag_key
    seg = 720575940000000000
    chunks = {}
    for k in keys:
        o = np.array(parse_frag_key(k)) * np.array(info.resolution_nm)
        d = {}
        for _ in range(12):
            sid = seg; seg += 13
            n = int(rng.integers(30, 90))
            st = o + rng.uniform(0.1, 0.9, 3) * cs
            p = np.clip(np.cumsum(rng.normal(0, 200, (n, 3)), 0) + st,
                        o + 10, o + cs - 10).astype(np.float32)
            d[sid] = {"vertices": p,
                      "edges": np.array([[i, i - 1] for i in range(1, n)]),
                      "radius": rng.uniform(50, 400, n).astype(np.float32)}
        chunks[k] = d
    reader = InMemoryFragsReader(info, chunks)
    bounds = ([anchor[a] * info.resolution_nm[a] for a in range(3)],
              [(anchor[a] + counts[a] * 512) * info.resolution_nm[a] for a in range(3)])
    return reader, keys, bounds


def test_run_ingest_parallel_matches_serial(tmp_path):
    """A parallel executor (multi-process L0 + pyramid) must produce a
    byte-identical store to the serial default."""
    from zarr_vectors_tools.ingest.precomputed_skeletons import run_ingest
    reader, keys, bounds = _flywire_cutout_reader()
    a = str(tmp_path / "serial.zv")
    b = str(tmp_path / "parallel.zv")
    args = dict(bounds_nm=bounds, strides=[8, 8], chunk_scale_factors=[2, 2],
                sparsity_factors=[1.0, 2.0], progress=False)
    run_ingest(reader, a, list(keys), **args)                       # serial
    run_ingest(reader, b, list(keys), executor=_ppool_executor, **args)
    sa, sb = _tree_sha(a), _tree_sha(b)
    assert set(sa) == set(sb), set(sa).symmetric_difference(sb)
    mismatched = [p for p in sa if sa[p] != sb[p]]
    assert not mismatched, mismatched


def test_ingest_driver_offline(tmp_store):
    """End-to-end ETL (extract→write→reduce→pyramid) via an in-memory
    .frags reader, with a flywire-shaped spatial index and a phase-offset
    chunk grid (so one zarr chunk collects pieces from several .frags)."""
    from zarr_vectors_tools.ingest.precomputed_skeletons import (
        InMemoryFragsReader, SkeletonInfo, enumerate_frag_keys, run_ingest,
    )

    info = SkeletonInfo(
        base_url="mem://x",
        resolution_nm=(32.0, 32.0, 40.0),
        chunk_size_nm=(16384.0, 16384.0, 20480.0),
        vertex_attributes=[
            {"id": "radius", "data_type": "float32", "num_components": 1},
            {"id": "cross_sectional_area", "data_type": "float32", "num_components": 1},
        ])
    cs_nm = np.array(info.chunk_size_nm)
    rng = np.random.default_rng(0)
    anchor = (17910, 8912, 3088)             # phase-offset from origin 0
    keys = enumerate_frag_keys(info, anchor, (2, 2, 1))
    corners = [(x, y, z)
               for x in (anchor[0], anchor[0] + 512)
               for y in (anchor[1], anchor[1] + 512)
               for z in (anchor[2],)]
    chunks = {}
    seg = 720575940000000000
    all_segs = set()
    for k, (vx, vy, vz) in zip(keys, corners):
        origin = np.array([vx, vy, vz]) * np.array(info.resolution_nm)
        d = {}
        for _ in range(6):
            sid = seg; seg += 13; all_segs.add(sid)
            npts = int(rng.integers(30, 80))
            start = origin + rng.uniform(0.1, 0.9, 3) * cs_nm
            pos = np.clip(np.cumsum(rng.normal(0, 200, (npts, 3)), axis=0) + start,
                          origin + 10, origin + cs_nm - 10).astype(np.float32)
            d[sid] = {
                "vertices": pos,
                "edges": np.array([[i, i - 1] for i in range(1, npts)]),
                "radius": rng.uniform(50, 400, npts).astype(np.float32),
                "cross_sectional_area": rng.uniform(1e3, 1e5, npts).astype(np.float32),
            }
        chunks[k] = d

    reader = InMemoryFragsReader(info, chunks)
    bounds = ([anchor[a] * info.resolution_nm[a] for a in range(3)],
              [(anchor[a] + (2 if a < 2 else 1) * 512) * info.resolution_nm[a]
               for a in range(3)])
    summary = run_ingest(
        reader, tmp_store, keys, bounds_nm=bounds,
        strides=[8, 8], chunk_scale_factors=[2, 2],
        sparsity_factors=[1.0, 2.0], progress=False)

    assert summary["objects"] == len(all_segs)
    # every segment pulls back out by its uint64 id, with both attributes
    for sid in all_segs:
        r0 = sk.read_skeleton_by_segment_id(tmp_store, sid, level=0)
        assert r0 is not None and len(r0["positions"]) > 0
        assert "radius" in r0["attributes"]
        assert "cross_sectional_area" in r0["attributes"]
    # coarsen-only level keeps all objects; sparsity level drops some
    levels = summary["pyramid"]["levels"]
    assert levels[0]["object_count"] == len(all_segs)
    assert levels[1]["object_count"] < len(all_segs)


def test_cross_chunk_edges_merge_fragments_one_level_up(tmp_store):
    """A segment split across a zarr-chunk boundary at level 0 is stored as
    two fragments + one cross-chunk link; one level up (chunks ×2) both
    pieces fall in the same chunk and the link re-merges them into a single
    connected fragment — no proximity heuristic involved."""
    from zarr_vectors_tools.ingest.precomputed_skeletons import (
        InMemoryFragsReader, SkeletonInfo, run_ingest,
    )

    info = SkeletonInfo(
        base_url="mem://x", resolution_nm=(32.0, 32.0, 40.0),
        chunk_size_nm=(16384.0, 16384.0, 20480.0))
    # Straight line in x crossing the origin-0 boundary at 35*16384=573440.
    xs = np.arange(573300.0, 573600.0, 20.0)
    pos = np.stack([xs, np.full_like(xs, 335000.0), np.full_like(xs, 130000.0)], 1
                   ).astype(np.float32)
    edges = np.array([[i, i - 1] for i in range(1, len(xs))])
    seg = 720575940000000123
    key = "17910-18422_10448-10960_3088-3600.frags"
    reader = InMemoryFragsReader(info, {key: {seg: {"vertices": pos, "edges": edges}}})
    bounds = ([573120.0, 334336.0, 123520.0], [589504.0, 350720.0, 144000.0])

    # align=False keeps the origin-0 grid so the line is phase-split into
    # two chunks, producing the cross-chunk edge that drives merging.
    summary = run_ingest(reader, tmp_store, [key], bounds_nm=bounds,
                         strides=[2], chunk_scale_factors=[2],
                         sparsity_factors=[1.0], align=False, progress=False)
    assert summary["level0_cross_chunk_edges"] == 1

    # Level 0: two fragments, link present.
    r0 = sk.read_skeleton_by_segment_id(tmp_store, seg, level=0)
    assert r0["fragment_count"] == 2
    g0 = get_resolution_level(open_store(tmp_store), 0)
    assert len(read_cross_chunk_links(g0, delta=0)) == 1

    # Level 1: merged to one fragment, fully connected (n-1 edges, 1 comp).
    r1 = sk.read_skeleton_by_segment_id(tmp_store, seg, level=1)
    assert r1["fragment_count"] == 1
    n1 = len(r1["positions"])
    assert len(r1["edges"]) == n1 - 1   # a single tree over all vertices
    # endpoints preserved → x-extent still spans the original line
    assert r1["positions"][:, 0].min() < 573440 < r1["positions"][:, 0].max()


def test_alignment_no_split_and_world_roundtrip(tmp_store):
    """With align=True the chunk grid is shifted to the .frag grid, so a
    piece within one .frag stays a single fragment (no phase-split, no
    cross-chunk edges), and reads return absolute world coordinates."""
    from zarr_vectors_tools.ingest.precomputed_skeletons import (
        InMemoryFragsReader, SkeletonInfo, run_ingest,
    )
    info = SkeletonInfo(
        base_url="mem://x", resolution_nm=(32.0, 32.0, 40.0),
        chunk_size_nm=(16384.0, 16384.0, 20480.0))
    # A line that WOULD cross the origin-0 boundary at 573440 (abs coords).
    xs = np.arange(573300.0, 573600.0, 20.0)
    pos = np.stack([xs, np.full_like(xs, 335000.0), np.full_like(xs, 130000.0)], 1
                   ).astype(np.float32)
    edges = np.array([[i, i - 1] for i in range(1, len(xs))])
    seg = 720575940000000123
    key = "17910-18422_10448-10960_3088-3600.frags"
    reader = InMemoryFragsReader(info, {key: {seg: {"vertices": pos, "edges": edges}}})
    bounds = ([573120.0, 334336.0, 123520.0], [589504.0, 350720.0, 144000.0])

    summary = run_ingest(reader, tmp_store, [key], bounds_nm=bounds,
                         strides=[], align=True, progress=False)
    # aligned → the .frag piece is wholly inside one zarr chunk
    assert summary["level0_cross_chunk_edges"] == 0
    r0 = sk.read_skeleton_by_segment_id(tmp_store, seg, level=0)
    assert r0["fragment_count"] == 1
    # reads come back in absolute world coordinates (offset added back)
    assert r0["positions"][:, 0].min() == pytest.approx(573300.0, abs=1.0)
    assert r0["positions"][:, 0].max() == pytest.approx(573580.0, abs=1.0)


def test_coincident_boundary_vertices_become_cross_chunk_edges(tmp_store):
    """Aligned ingest recovers cross-chunk edges from exact coincident
    boundary vertices (as igneous writes them): a segment present in two
    adjacent .frags with a shared boundary vertex gets a level-0
    cross-chunk link, and the two fragments merge into one connected
    fragment one level up."""
    from zarr_vectors_tools.ingest.precomputed_skeletons import (
        InMemoryFragsReader, SkeletonInfo, enumerate_frag_keys, run_ingest,
    )
    info = SkeletonInfo(
        base_url="mem://x", resolution_nm=(32.0, 32.0, 40.0),
        chunk_size_nm=(16384.0, 16384.0, 20480.0))
    keys = enumerate_frag_keys(info, (17398, 10448, 3088), (2, 1, 1))  # kA, kB in x
    # Duplicated boundary vertices sit at the chunk face + 0.5 voxel (the
    # voxel-center convention), exactly as flywire stores them.
    bnd = 17910 * 32 + 16  # face 573120 + 0.5*32 = 573136
    y, z = 340000.0, 130000.0
    seg = 720575940000000999
    shared = [bnd, y, z]
    # A: ... up to the shared boundary vertex
    aA = np.array([[560000, y, z], [565000, y, z], [570000, y, z], shared], np.float32)
    eA = np.array([[i, i - 1] for i in range(1, len(aA))])
    # B: starts at the same shared vertex, continues
    aB = np.array([shared, [576000, y, z], [580000, y, z]], np.float32)
    eB = np.array([[i, i - 1] for i in range(1, len(aB))])
    reader = InMemoryFragsReader(info, {
        keys[0]: {seg: {"vertices": aA, "edges": eA}},
        keys[1]: {seg: {"vertices": aB, "edges": eB}},
    })
    bounds = ([17398 * 32.0, 10448 * 32.0, 3088 * 40.0],
              [18422 * 32.0, 10960 * 32.0, 3600 * 40.0])

    summary = run_ingest(reader, tmp_store, keys, bounds_nm=bounds,
                         strides=[2], chunk_scale_factors=[2],
                         sparsity_factors=[1.0], align=True, progress=False)
    assert summary["level0_cross_chunk_edges"] >= 1

    r0 = sk.read_skeleton_by_segment_id(tmp_store, seg, level=0)
    assert r0["fragment_count"] == 2          # one per .frag, aligned 1:1
    r1 = sk.read_skeleton_by_segment_id(tmp_store, seg, level=1)
    assert r1["fragment_count"] == 1          # merged via the cross-chunk edge
    n1 = len(r1["positions"])
    assert len(r1["edges"]) == n1 - 1         # single connected tree
