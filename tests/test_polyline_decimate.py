"""Tests for stride decimation of streamline/polyline pyramids.

Covers:
* ``decimate_polyline``: the linear-path decimation primitive.
* ``build_pyramid(..., coarsen_mode="decimate")``: end-to-end pyramid level
  built via uniform stride decimation instead of RDP simplification.
"""

from __future__ import annotations

import numpy as np

from zarr_vectors.constants import CAP_MULTISCALE_LINKS, XLEVEL_EXPLICIT
from zarr_vectors.core.arrays import (
    list_link_deltas,
    read_all_object_manifests,
    read_chunk_vertices,
)
from zarr_vectors.core.paths import links_group_path
from zarr_vectors.core.store import (
    get_resolution_level,
    list_resolution_levels,
    open_store,
    read_level_metadata,
    read_root_metadata,
)
from tests._source_helpers import write_polylines_with_segment_id as write_polylines
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


def test_links_family_group_stamped_with_sid_ndim_even_when_empty(tmp_path):
    # All streamlines confined to a single (large) chunk, so the coarsened
    # level has zero chunk-spanning links and no writer that stamps sid_ndim
    # is ever called; create_links_family alone must still stamp the policy
    # so readers don't see it as undefined.
    store = tmp_path / "single_chunk.zv"
    write_polylines(
        str(store),
        _make_streamlines(seed=1, n=5, vpp=10, extent=10.0),
        chunk_shape=(1000.0, 1000.0, 1000.0),
    )
    build_pyramid(str(store), factors=[(8.0, 1.0)], coarsen_mode="decimate")

    root = open_store(str(store))
    lvl1 = get_resolution_level(root, 1)
    meta = lvl1.read_array_meta(links_group_path(0))
    assert meta.get("sid_ndim") == 3
    assert meta.get("link_width") == 2
    # directed is family-wide and un-flippable now, so pin it too.
    assert meta.get("directed") is True


def test_no_multiscale_capability_when_no_cross_level_arrays_emitted(tmp_path):
    """The token marks delta != 0 arrays; a store without any must not claim it.

    The polyline coarsener never emits inline ±1 cross-level links, so a
    polyline pyramid built with cross_level_depth=1 produces no delta != 0
    array anywhere — regardless of what cross_level_storage asks for.
    Stamping optimistically (before knowing whether anything would be
    emitted) made the store advertise cross-LEVEL links it does not have.
    """
    store = tmp_path / "polyline_depth1.zv"
    write_polylines(
        str(store),
        _make_streamlines(seed=4, n=12, vpp=40, extent=200.0),
        chunk_shape=(50.0, 50.0, 50.0),
    )
    build_pyramid(
        str(store),
        factors=[(4.0, 1.0)],
        coarsen_mode="decimate",
        cross_level_depth=1,
        cross_level_storage=XLEVEL_EXPLICIT,
    )

    root = open_store(str(store))
    deltas = {
        lvl: list_link_deltas(get_resolution_level(root, lvl))
        for lvl in list_resolution_levels(root)
    }
    emitted = any(d != 0 for ds in deltas.values() for d in ds)
    stamped = CAP_MULTISCALE_LINKS in read_root_metadata(root).format_capabilities
    assert stamped == emitted, (
        f"capability={stamped} but delta!=0 arrays present={emitted} "
        f"(per-level deltas: {deltas})"
    )


def test_sparsity_pyramid_is_cumulative_per_level(tmp_path):
    """--sparsity 10 per level drops to 1/10 of the PREVIOUS level each time.

    Regression for the bug where levels 2..N were byte-identical copies of
    level 1: sparsity targeted a fixed fraction of the ORIGINAL count, so a
    constant factor kept the same survivors every level after the first.
    With cumulative sparsity, N -> N/10 -> N/100 -> N/1000.  decimate stride
    1 keeps every vertex, isolating the object-drop behaviour.
    """
    store = tmp_path / "cumulative.zv"
    write_polylines(
        str(store),
        _make_streamlines(seed=7, n=1000, vpp=20, extent=300.0),
        chunk_shape=(150.0, 150.0, 150.0),
    )
    build_pyramid(
        str(store),
        factors=[(1.0, 10.0)] * 3,
        coarsen_mode="decimate",
        sparsity_strategy="random",
        sparsity_seed=0,
    )

    root = open_store(str(store))
    counts = [
        sum(1 for m in read_all_object_manifests(get_resolution_level(root, lvl)) if m)
        for lvl in sorted(list_resolution_levels(root))
    ]
    # 1000 -> 100 -> 10 -> 1 (random keeps exactly round(alive / 10)).
    assert counts == [1000, 100, 10, 1], counts
