"""Tests for stride decimation of streamline/polyline pyramids.

Covers:
* ``decimate_polyline``: the linear-path decimation primitive.
* ``build_pyramid(..., coarsen_mode="decimate")``: end-to-end pyramid level
  built via uniform stride decimation instead of RDP simplification.
"""

from __future__ import annotations

import numpy as np

from zarr_vectors.core.arrays import read_all_object_manifests, read_chunk_vertices
from zarr_vectors.core.paths import cross_chunk_links_path
from zarr_vectors.core.store import get_resolution_level, open_store, read_level_metadata
from zarr_vectors.types.polylines import write_polylines
from zarr_vectors_tools.multiresolution.coarsen import build_pyramid
from zarr_vectors_tools.multiresolution.strategies.polylines import decimate_polyline


# ===================================================================
# decimate_polyline
# ===================================================================


def test_decimate_polyline_keeps_stride_and_endpoints():
    verts = np.arange(30 * 3, dtype=np.float32).reshape(30, 3)
    out = decimate_polyline(verts, stride=8)
    kept_rows = (out[:, 0] // 3).astype(int).tolist()
    assert kept_rows == [0, 8, 16, 24, 29]


def test_decimate_polyline_stride_le_1_keeps_all():
    verts = np.arange(10 * 3, dtype=np.float32).reshape(10, 3)
    out = decimate_polyline(verts, stride=1)
    assert out.shape == verts.shape
    np.testing.assert_array_equal(out, verts)

    out0 = decimate_polyline(verts, stride=0)
    np.testing.assert_array_equal(out0, verts)


def test_decimate_polyline_short_input_returns_copy():
    verts = np.arange(2 * 3, dtype=np.float32).reshape(2, 3)
    out = decimate_polyline(verts, stride=8)
    np.testing.assert_array_equal(out, verts)
    assert out is not verts


# ===================================================================
# build_pyramid(coarsen_mode="decimate")
# ===================================================================


def _make_streamlines(seed=0, n=20, vpp=41, extent=100.0):
    rng = np.random.default_rng(seed)
    return [rng.uniform(0, extent, (vpp, 3)).astype("f4") for _ in range(n)]


def _build_store(tmp_path, seed=0, n=20, vpp=41, chunk_shape=(50.0, 50.0, 50.0)):
    store = tmp_path / "streamlines.zv"
    write_polylines(
        str(store),
        _make_streamlines(seed=seed, n=n, vpp=vpp),
        chunk_shape=chunk_shape,
    )
    return store


def _make_smooth_streamlines(seed=0, n=20, vpp=41, extent=100.0):
    """Nearly-monotonic paths (small jitter on a straight line) — crosses a
    chunk boundary a handful of times, not on every vertex.  Contrast with
    ``_make_streamlines`` (uniform random per-vertex noise): a streamline
    that bounces between chunks every 1-2 vertices leaves each chunk-local
    visit too short to decimate (its entry/exit vertices must both be kept
    to preserve which chunks the path actually passed through) — that's a
    real, accepted property of the chunk-local coarsener (see
    ``coarsen_polyline_level``'s module docstring), not something these
    "keeps meaningfully reducing vertices" tests should exercise.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, vpp, dtype="f4")[:, None]
    jitter = rng.uniform(-extent * 0.01, extent * 0.01, (n, vpp, 3)).astype("f4")
    lines = []
    for i in range(n):
        start = rng.uniform(0, extent * 0.1, 3).astype("f4")
        end = rng.uniform(extent * 0.9, extent, 3).astype("f4")
        lines.append(start + t * (end - start) + jitter[i])
    return lines


def test_build_pyramid_decimate_mode_thins_vertices_keeps_streamlines(tmp_path):
    store = tmp_path / "streamlines.zv"
    write_polylines(
        str(store),
        _make_smooth_streamlines(seed=3, n=12, vpp=41),
        chunk_shape=(50.0, 50.0, 50.0),
    )

    summary = build_pyramid(
        str(store),
        factors=[(8.0, 1.0)],
        coarsen_mode="decimate",
    )
    assert summary["levels_created"] == 1
    spec = summary["level_specs"][0]
    assert spec["method"] == "polyline_decimate"
    assert spec["objects_kept"] == 12
    assert spec["source_objects"] == 12
    assert spec["stride"] == 8

    root = open_store(str(store))
    lm = read_level_metadata(root, 1)
    assert lm.coarsening_method == "polyline_decimate"
    assert lm.preserves_object_ids is True

    lvl1 = get_resolution_level(root, 1)
    manifests = read_all_object_manifests(lvl1)
    assert len(manifests) == 12
    # Every streamline must still be present, and stride-8 decimation must
    # have meaningfully reduced its vertex count (a near-straight path
    # crosses only a few chunk boundaries, so each chunk-local visit is
    # long enough to actually decimate — unlike the choppy/adversarial
    # case covered by the coarsener's module docstring).
    total_verts = 0
    for oid, manifest in enumerate(manifests):
        assert len(manifest) > 0, f"object {oid} was dropped unexpectedly"
        obj_verts = 0
        for cc, frag_idx in manifest:
            frags = read_chunk_vertices(lvl1, cc, dtype=np.float32, ndim=3)
            obj_verts += len(frags[frag_idx])
        assert 2 <= obj_verts < 41, f"object {oid} was not decimated: {obj_verts} verts"
        total_verts += obj_verts
    assert total_verts < 12 * 41


def test_build_pyramid_decimate_mode_can_drop_streamlines(tmp_path):
    store = _build_store(tmp_path, seed=5, n=20, vpp=41)

    summary = build_pyramid(
        str(store),
        factors=[(8.0, 4.0)],
        coarsen_mode="decimate",
        sparsity_seed=42,
    )
    spec = summary["level_specs"][0]
    assert spec["method"] == "polyline_decimate"
    assert spec["source_objects"] == 20
    assert spec["objects_kept"] == 5  # sparsity_factor=4.0 -> keep 1/4

    root = open_store(str(store))
    lvl1 = get_resolution_level(root, 1)
    manifests = read_all_object_manifests(lvl1)
    assert len(manifests) == 20
    kept = [m for m in manifests if len(m) > 0]
    dropped = [m for m in manifests if len(m) == 0]
    assert len(kept) == 5
    assert len(dropped) == 15


def test_cross_chunk_links_group_stamped_with_sid_ndim_even_when_empty(tmp_path):
    # All streamlines confined to a single (large) chunk, so the coarsened
    # level has zero cross-chunk-spanning links and write_cross_chunk_links
    # (which stamps sid_ndim) is never called; create_cross_chunk_links_array
    # alone must still stamp sid_ndim so readers don't see it as undefined.
    store = tmp_path / "single_chunk.zv"
    write_polylines(
        str(store),
        _make_streamlines(seed=1, n=5, vpp=10, extent=10.0),
        chunk_shape=(1000.0, 1000.0, 1000.0),
    )
    build_pyramid(str(store), factors=[(8.0, 1.0)], coarsen_mode="decimate")

    root = open_store(str(store))
    lvl1 = get_resolution_level(root, 1)
    meta = lvl1.read_array_meta(cross_chunk_links_path(0))
    assert meta.get("sid_ndim") == 3
    assert meta.get("link_width") == 2
