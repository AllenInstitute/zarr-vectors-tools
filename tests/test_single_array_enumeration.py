"""Regression tests for chunk enumeration under the 0.9.0 single-array core.

Core migrated every per-spatial-chunk item class (``vertices``,
``vertex_fragments``, ``links/<delta>``, attribute arrays) to ONE Zarr v3
vlen-bytes array whose shape is the chunk grid (cells at ``<array>/c/i/j/k``).
That array keeps a ``nonempty_chunks`` presence manifest maintained by a
read-modify-write; the tools write per-chunk cells from *parallel worker
processes*, whose manifest RMWs race and can under-report.  The coordinators
call :meth:`Group.rebuild_nonempty_manifests` after each parallel phase to
re-derive the manifest from the on-disk cells.

These tests assert:
1. The tools produce the single-array layout on disk (not per-chunk sub-arrays).
2. A multi-process pyramid enumerates the *same* chunks as a serial one at
   every level (the manifest is complete despite the write race).
3. Negative chunk coordinates (points below the grid origin) enumerate and
   round-trip through the parallel coarsener.
"""

from __future__ import annotations

import json
import os
from functools import partial
from pathlib import Path

import numpy as np
from zarr_vectors.core.arrays import list_chunk_keys
from zarr_vectors.core.store import get_resolution_level, open_store

from tests._source_helpers import write_polylines_with_segment_id as write_polylines
from zarr_vectors_tools.multiresolution.coarsen import build_pyramid


# ------------------------------------------------------------------ helpers
def _ppool_executor(func, items, shared=None):
    """Run the per-target-chunk worker across real OS processes."""
    items = list(items)
    if not items:
        return []
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=3) as ex:
        return list(ex.map(partial(func, shared=shared), items))


def _random_walk_streamlines(seed, n, npts=30, extent=400.0, step=8.0):
    rng = np.random.default_rng(seed)
    lines = []
    for _ in range(n):
        start = rng.uniform(extent * 0.1, extent * 0.6, 3)
        steps = rng.normal(0, step, size=(npts - 1, 3))
        pos = np.concatenate([[start], start + np.cumsum(steps, axis=0)])
        pos = np.clip(pos, 1.0, extent - 1.0)
        lines.append(pos.astype("f4"))
    return lines


def _vmeta(store, level) -> dict:
    return json.loads(
        (Path(store) / str(level) / "vertices" / "zarr.json").read_text()
    )


def _assert_single_array_layout(store, level) -> None:
    vdir = Path(store) / str(level) / "vertices"
    assert (vdir / "zarr.json").is_file(), f"level {level}: no vertices/zarr.json"
    meta = _vmeta(store, level)
    assert meta["node_type"] == "array", f"level {level}: vertices is not one array"
    assert meta["data_type"] == "variable_length_bytes"
    # No dotted per-chunk sub-array directories (the old Option-G explosion).
    subdirs = [d for d in os.listdir(vdir) if (vdir / d).is_dir() and d != "c"]
    assert subdirs == [], f"level {level}: unexpected per-chunk sub-dirs: {subdirs}"


def _vertex_chunk_set(store, level) -> set[tuple[int, ...]]:
    g = get_resolution_level(open_store(str(store)), level)
    return {tuple(int(x) for x in cc) for cc in list_chunk_keys(g, "vertices")}


# ------------------------------------------------------------------ tests
def test_level0_vertices_is_single_array(tmp_path):
    """The tools' polyline writer produces one ``vertices`` array, not a group
    of per-chunk sub-arrays."""
    store = tmp_path / "s.zv"
    lines = _random_walk_streamlines(seed=1, n=30)
    write_polylines(str(store), lines, chunk_shape=(40.0, 40.0, 40.0))
    _assert_single_array_layout(store, 0)
    assert _vertex_chunk_set(store, 0), "no chunks enumerated at level 0"


def test_parallel_pyramid_enumerates_same_chunks_as_serial(tmp_path):
    """Every level of a multi-process pyramid must enumerate exactly the chunks
    a serial build does — proving the manifest survives the parallel write race.

    A wide extent with a small chunk shape yields many target chunks so several
    worker processes contend for each array's shared ``nonempty_chunks``.
    """
    a = tmp_path / "serial.zv"
    b = tmp_path / "parallel.zv"
    lines = _random_walk_streamlines(seed=7, n=40)
    write_polylines(str(a), [x.copy() for x in lines], chunk_shape=(40.0, 40.0, 40.0))
    write_polylines(str(b), [x.copy() for x in lines], chunk_shape=(40.0, 40.0, 40.0))

    args = dict(
        factors=[(4.0, 1.0), (2.0, 1.0)],
        chunk_scale_factors=[1, 2],
        coarsen_mode="decimate",
        sparsity_strategy="random",
        sparsity_seed=3,
    )
    build_pyramid(str(a), **args)                        # serial (executor=None)
    build_pyramid(str(b), executor=_ppool_executor, **args)

    for level in (0, 1, 2):
        serial = _vertex_chunk_set(a, level)
        parallel = _vertex_chunk_set(b, level)
        assert serial, f"level {level}: serial build enumerated no chunks"
        assert parallel == serial, (
            f"level {level}: parallel enumeration diverged\n"
            f"  missing (dropped by race): {serial - parallel}\n"
            f"  extra: {parallel - serial}"
        )
        _assert_single_array_layout(b, level)


def test_negative_coordinates_enumerate_through_parallel_coarsener(tmp_path):
    """Points below the grid origin produce negative chunk coords (addressed via
    ``chunk_grid_origin``); they must enumerate and round-trip through a
    multi-process coarsen level."""
    store = tmp_path / "neg.zv"
    lines = _random_walk_streamlines(seed=9, n=30)
    # Shift well below zero so a chunk of coords is negative on every axis.
    lines = [x - 200.0 for x in lines]
    assert min(float(x.min()) for x in lines) < 0
    write_polylines(str(store), lines, chunk_shape=(40.0, 40.0, 40.0))

    level0 = _vertex_chunk_set(store, 0)
    assert any(min(cc) < 0 for cc in level0), "expected a negative chunk coord"

    build_pyramid(
        str(store),
        factors=[(2.0, 1.0)],
        chunk_scale_factors=[2],
        coarsen_mode="decimate",
        executor=_ppool_executor,
    )

    _assert_single_array_layout(store, 1)
    level1 = _vertex_chunk_set(store, 1)
    assert level1, "coarsened level enumerated no chunks"
    # Coarsening halves the grid: a negative source coord c maps to c // 2,
    # still negative — the origin offset must keep it addressable.
    assert any(min(cc) < 0 for cc in level1), (
        "negative chunk coord lost through coarsening"
    )
