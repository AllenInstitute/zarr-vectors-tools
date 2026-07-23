"""Tests for the chunk-local, executor-parallel streamline/polyline coarsener
(:func:`zarr_vectors_tools.multiresolution.strategies.polylines.coarsen_polyline_level`).

This coarsener replaced a whole-level implementation that OOM'd on large
(5M-streamline) stores because it loaded every fragment, manifest, and
per-object vertex sequence into memory at once.  The replacement processes
one target chunk at a time, reconstructing object connectivity purely from
local reads: a fragment's ``segment_id`` (its object id) plus the source
level's *directed* (predecessor -> successor) ``cross_chunk_links``.

These tests focus on properties that are easy to get subtly wrong with that
design:

* predecessor/successor order must survive coarsening (the whole point of
  ``directed=True`` cross-chunk links — a reader needs this for walk-order
  tangents).
* a cross-chunk transition can land in ANY other chunk, not just a face
  neighbor (``split_polyline_at_boundaries`` does not decompose multi-axis
  crossings into single-axis hops) — Phase B must handle that.
* an object visiting the same target chunk more than once must produce one
  fragment per visit, each correctly stitched to its neighbors.
* sparsity must preserve OID stability (dropped objects leave empty
  manifest slots; kept objects keep their original OID).
* a parallel (multi-process) executor must produce byte-identical output to
  the serial default.
"""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from functools import partial

import numpy as np
import pytest

from zarr_vectors.core.arrays import (
    read_all_object_manifests,
    read_chunk_fragment_attributes,
    read_chunk_vertices,
)
from zarr_vectors_tools.algorithms._links import read_cross_links
from zarr_vectors.core.store import get_resolution_level, open_store
from tests._source_helpers import write_polylines_with_segment_id as write_polylines
from zarr_vectors_tools.multiresolution.coarsen import build_pyramid, coarsen_level


# ===================================================================
# Fixtures
# ===================================================================


def _random_walk_streamlines(seed=0, n=8, npts=25, step=6.0, extent=120.0):
    """Continuous random-walk paths — traverse several chunks in relatively
    long strides (unlike i.i.d. uniform noise, which bounces between chunks
    almost every vertex and leaves nothing to decimate)."""
    rng = np.random.default_rng(seed)
    lines = []
    for _ in range(n):
        start = rng.uniform(extent * 0.1, extent * 0.3, 3)
        steps = rng.normal(0, step, size=(npts - 1, 3))
        pos = np.concatenate([[start], start + np.cumsum(steps, axis=0)])
        pos = np.clip(pos, 1.0, extent - 1.0)
        lines.append(pos.astype("f4"))
    return lines


def _ppool_executor(func, items, shared=None):
    """Parallel executor over a ProcessPoolExecutor — exercises pickling of
    the per-target-chunk worker, its plain-data payloads, and the shared
    (scattered-once) plan across real OS processes."""
    items = list(items)
    if not items:
        return []
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=2) as ex:
        return list(ex.map(partial(func, shared=shared), items))


def _tree_sha(store_dir):
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


def _level_object_count(store, level):
    g = get_resolution_level(open_store(str(store)), level)
    return sum(1 for m in read_all_object_manifests(g) if m)


def _level_vertex_count(store, level):
    from zarr_vectors.core.arrays import list_chunk_keys
    g = get_resolution_level(open_store(str(store)), level)
    return sum(
        len(f) for cc in list_chunk_keys(g, "vertices")
        for f in read_chunk_vertices(g, cc, dtype=np.float32, ndim=3)
    )


# ===================================================================
# Directed cross-chunk links (predecessor -> successor order)
# ===================================================================


def test_directed_link_predecessor_successor_order_preserved(tmp_path):
    """A streamline crossing one boundary must leave a coarsened
    cross-chunk-link record with endpoint 0 = the earlier (predecessor)
    chunk and endpoint 1 = the later (successor) chunk, matching walk
    order — required for correct tangent-direction rendering."""
    store = tmp_path / "s.zv"
    # Straight path from (0,0,0)'s chunk into (1,0,0)'s chunk.
    poly = np.linspace([2, 2, 2], [18, 2, 2], 20).astype("f4")
    write_polylines(str(store), [poly], chunk_shape=(10.0, 10.0, 10.0))

    coarsen_level(
        str(store), source_level=0, target_level=1,
        coarsen_factor=1.0, chunk_scale_factor=1, sparsity_factor=1.0,
        coarsen_mode="decimate",
    )

    lvl1 = get_resolution_level(open_store(str(store)), 1)
    records = read_cross_links(lvl1, delta=0)
    assert len(records) == 1
    (ccA, _viA), (ccB, _viB) = records[0]
    assert tuple(ccA) == (0, 0, 0)
    assert tuple(ccB) == (1, 0, 0)

    # Reversing the path must flip which side is predecessor vs successor.
    store2 = tmp_path / "s2.zv"
    write_polylines(str(store2), [poly[::-1].copy()], chunk_shape=(10.0, 10.0, 10.0))
    coarsen_level(
        str(store2), source_level=0, target_level=1,
        coarsen_factor=1.0, chunk_scale_factor=1, sparsity_factor=1.0,
        coarsen_mode="decimate",
    )
    lvl1b = get_resolution_level(open_store(str(store2)), 1)
    records_b = read_cross_links(lvl1b, delta=0)
    assert len(records_b) == 1
    (ccA2, _), (ccB2, _) = records_b[0]
    assert tuple(ccA2) == (1, 0, 0)
    assert tuple(ccB2) == (0, 0, 0)


# ===================================================================
# Diagonal (non-face-adjacent) cross-target jumps
# ===================================================================


def test_diagonal_cross_target_jump_stitches_correctly(tmp_path):
    """A single-segment transition can jump directly between chunks that
    differ in MORE THAN ONE axis (split_polyline_at_boundaries does not
    decompose multi-axis crossings into single-axis hops) — Phase B must
    discover and stitch this pair even though it isn't a grid neighbor."""
    store = tmp_path / "s.zv"
    poly = np.array([[5.0, 5.0, 5.0], [15.0, 15.0, 15.0]], dtype="f4")
    write_polylines(str(store), [poly], chunk_shape=(10.0, 10.0, 10.0))

    root = open_store(str(store))
    lvl0 = get_resolution_level(root, 0)
    # Sanity: the source level really did split into two diagonally
    # adjacent (non-face-neighbor) chunks.
    manifest0 = read_all_object_manifests(lvl0)[0]
    ccs = [cc for cc, _ in manifest0]
    assert ccs == [(0, 0, 0), (1, 1, 1)]

    coarsen_level(
        str(store), source_level=0, target_level=1,
        coarsen_factor=1.0, chunk_scale_factor=1, sparsity_factor=1.0,
        coarsen_mode="decimate",
    )
    lvl1 = get_resolution_level(open_store(str(store)), 1)
    records = read_cross_links(lvl1, delta=0)
    assert len(records) == 1
    (ccA, _), (ccB, _) = records[0]
    assert tuple(ccA) == (0, 0, 0)
    assert tuple(ccB) == (1, 1, 1)

    manifest1 = read_all_object_manifests(lvl1)[0]
    assert len(manifest1) == 2


# ===================================================================
# Multiple excursions: an object visiting the same target chunk twice
# ===================================================================


def test_object_revisiting_same_chunk_yields_two_fragments(tmp_path):
    """chunk A -> chunk B -> chunk A again must produce TWO separate
    fragments for the object in chunk A (two distinct runs), each
    correctly stitched to its own neighbor in B."""
    store = tmp_path / "s.zv"
    poly = np.array([
        [2.0, 2.0, 2.0],    # chunk (0,0,0)
        [8.0, 2.0, 2.0],    # chunk (0,0,0)
        [12.0, 2.0, 2.0],   # chunk (1,0,0)
        [18.0, 2.0, 2.0],   # chunk (1,0,0)
        [8.0, 2.0, 2.0],    # back to chunk (0,0,0)
        [3.0, 2.0, 2.0],    # chunk (0,0,0)
    ], dtype="f4")
    write_polylines(str(store), [poly], chunk_shape=(10.0, 10.0, 10.0))

    coarsen_level(
        str(store), source_level=0, target_level=1,
        coarsen_factor=1.0, chunk_scale_factor=1, sparsity_factor=1.0,
        coarsen_mode="decimate",
    )
    lvl1 = get_resolution_level(open_store(str(store)), 1)
    manifest = read_all_object_manifests(lvl1)[0]
    ccs = [cc for cc, _ in manifest]
    assert ccs.count((0, 0, 0)) == 2
    assert ccs.count((1, 0, 0)) == 1

    records = read_cross_links(lvl1, delta=0)
    assert len(records) == 2  # A->B and B->A, distinct link instances


# ===================================================================
# Sparsity + OID stability
# ===================================================================


def test_sparsity_drops_objects_and_preserves_oid_slots(tmp_path):
    store = tmp_path / "s.zv"
    lines = _random_walk_streamlines(seed=7, n=8, npts=25)
    write_polylines(str(store), lines, chunk_shape=(60.0, 60.0, 60.0))

    summary = coarsen_level(
        str(store), source_level=0, target_level=1,
        coarsen_factor=4.0, chunk_scale_factor=1, sparsity_factor=4.0,
        sparsity_strategy="random", sparsity_seed=1, coarsen_mode="decimate",
    )
    assert summary["source_objects"] == 8
    assert summary["objects_kept"] == 2  # 1/4 of 8

    lvl1 = get_resolution_level(open_store(str(store)), 1)
    manifests = read_all_object_manifests(lvl1)
    assert len(manifests) == 8  # OID slot space preserved, not shrunk
    kept = [oid for oid, m in enumerate(manifests) if m]
    dropped = [oid for oid, m in enumerate(manifests) if not m]
    assert len(kept) == 2
    assert len(dropped) == 6


# ===================================================================
# Multi-level pyramid: vertex counts shrink, objects preserved
# ===================================================================


def test_two_level_pyramid_shrinks_vertices_preserves_objects(tmp_path):
    store = tmp_path / "s.zv"
    lines = _random_walk_streamlines(seed=11, n=10, npts=30)
    write_polylines(str(store), lines, chunk_shape=(60.0, 60.0, 60.0))

    summary = build_pyramid(
        str(store),
        factors=[(4.0, 1.0), (2.0, 1.0)],
        chunk_scale_factors=[1, 2],
        coarsen_mode="decimate",
    )
    assert summary["levels_created"] == 2

    v0 = _level_vertex_count(store, 0)
    v1 = _level_vertex_count(store, 1)
    v2 = _level_vertex_count(store, 2)
    assert v1 < v0
    assert v2 < v1
    assert _level_object_count(store, 1) == 10
    assert _level_object_count(store, 2) == 10

    # segment_id fragment attributes must be present at every level (the
    # new coarsener relies on this to recurse; the old one never wrote it).
    lvl2 = get_resolution_level(open_store(str(store)), 2)
    from zarr_vectors.core.arrays import list_chunk_keys
    for cc in list_chunk_keys(lvl2, "vertices"):
        segs = read_chunk_fragment_attributes(lvl2, "segment_id", cc, dtype=np.uint64)
        n_frags = len(read_chunk_vertices(lvl2, cc, dtype=np.float32, ndim=3))
        assert len(segs) == n_frags


# ===================================================================
# rdp identity: coarsen_factor == 1 must NOT reduce vertices
# ===================================================================


def test_rdp_coarsen_factor_one_preserves_all_vertices(tmp_path):
    """In the default ``rdp`` mode, ``coarsen_factor == 1`` must be a true
    no-op on vertices (matching decimate's stride-1 identity), so a level built
    with ``--coarsen 1`` is a pure sparser subset of *full-resolution*
    streamlines.

    Regression: rdp used to derive a chunk-size epsilon
    (``0.5 * min(chunk_shape) * coarsen_factor``) that stayed non-zero at
    factor 1, silently simplifying every streamline down to ~half-chunk vertex
    spacing even though the caller asked for no vertex coarsening.
    """
    store = tmp_path / "s.zv"
    lines = _random_walk_streamlines(seed=7, n=12, npts=40)
    write_polylines(str(store), lines, chunk_shape=(30.0, 30.0, 30.0))

    v0 = _level_vertex_count(store, 0)
    build_pyramid(
        str(store),
        factors=[(1.0, 1.0)],      # coarsen 1 (no vtx reduction), sparsity 1 (keep all)
        chunk_scale_factors=[1],
        coarsen_mode="rdp",        # the CLI default
    )
    v1 = _level_vertex_count(store, 1)
    assert v1 == v0, f"rdp coarsen_factor=1 dropped vertices: {v0} -> {v1}"
    assert _level_object_count(store, 1) == 12


def test_rdp_coarsen_factor_above_one_still_simplifies(tmp_path):
    """Guard the other side of the identity fast-path: a factor > 1 must still
    reduce vertices, so the ``coarsen_factor <= 1`` no-op didn't disable rdp
    simplification wholesale."""
    store = tmp_path / "s.zv"
    lines = _random_walk_streamlines(seed=7, n=12, npts=40)
    write_polylines(str(store), lines, chunk_shape=(30.0, 30.0, 30.0))

    v0 = _level_vertex_count(store, 0)
    build_pyramid(
        str(store),
        factors=[(4.0, 1.0)],
        chunk_scale_factors=[1],
        coarsen_mode="rdp",
    )
    v1 = _level_vertex_count(store, 1)
    assert v1 < v0


# ===================================================================
# Serial vs. multi-process executor parity
# ===================================================================


def test_serial_matches_process_pool_executor(tmp_path):
    a = tmp_path / "serial.zv"
    b = tmp_path / "parallel.zv"
    lines_a = _random_walk_streamlines(seed=42, n=10, npts=30)
    lines_b = [x.copy() for x in lines_a]
    write_polylines(str(a), lines_a, chunk_shape=(60.0, 60.0, 60.0))
    write_polylines(str(b), lines_b, chunk_shape=(60.0, 60.0, 60.0))

    args = dict(
        factors=[(4.0, 2.0)], chunk_scale_factors=[2],
        coarsen_mode="decimate", sparsity_strategy="random", sparsity_seed=5,
    )
    build_pyramid(str(a), **args)                       # serial (executor=None)
    build_pyramid(str(b), executor=_ppool_executor, **args)

    sa, sb = _tree_sha(a), _tree_sha(b)
    assert set(sa) == set(sb), set(sa).symmetric_difference(sb)
    mismatched = [p for p in sa if sa[p] != sb[p]]
    assert not mismatched, mismatched
