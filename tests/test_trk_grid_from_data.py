"""The chunk grid is derived from the geometry, not from the TRK header.

A TRK header's ``dimensions × voxel_size`` is a *declared* field of view and
nothing in the format checks it against the tracts.  Files exist where it is
simply wrong: on a real 5.6M-streamline tractogram the header claimed a
152 mm axis whose points ran to 186 mm, and 780M of its 890M vertices
(87.7%) fell outside the declared box entirely.

The ingest used to build its chunk grid from that box and clamp any vertex
outside it into the nearest edge chunk.  That kept the coordinates intact but
filed them under chunk coords that do not contain them — 96% of the vertices
in one cell were foreign to it — which breaks every spatial query, the
chunk-local coarsener, and a viewer's per-chunk bounds.  It also piled 415M
vertices into a single cell, 4.6 GiB, past the vlen-bytes ceiling (see
test_chunk_cell_limit).

So Phase A now runs before the store is created, bins against unclamped
coords, and reports its exact bbox; the grid follows from that.  The header
box is used only as a fallback for an empty file.
"""

from __future__ import annotations

import numpy as np
import pytest
from zarr_vectors.building import (
    get_resolution_level,
    list_chunk_keys,
    open_store,
    read_chunk_vertices,
)

from zarr_vectors_tools.ingest.trk_parallel import ingest_trk_parallel

from .test_trk_registration import _write_radiological_trk


def _outside_header_box_streamlines(seed=0, n=30, npts=25):
    """Streamlines living well outside a dim=100 header's 0..100 voxmm box —
    the shape of the real file that exposed this."""
    rng = np.random.default_rng(seed)
    return [
        rng.uniform([120.0, -60.0, 10.0], [220.0, 20.0, 180.0], size=(npts, 3)).astype(np.float32)
        for _ in range(n)
    ]


def _root_attrs(store):
    import json
    from pathlib import Path

    meta = json.loads((Path(store) / "zarr.json").read_text())
    return meta["attributes"]["zarr_vectors"]


def _store_chunk_shape(store):
    return np.asarray(_root_attrs(store)["chunk_shape"], dtype=np.float64)


@pytest.mark.parametrize("register", [False, True])
def test_every_vertex_lands_in_a_chunk_that_contains_it(tmp_path, register):
    """The property the clamp violated: a vertex stored under chunk coord c
    must satisfy ``floor(v / chunk_shape) == c``, on every axis."""
    if register:
        pytest.importorskip("nibabel")
    trk = tmp_path / "outside.trk"
    _write_radiological_trk(trk, _outside_header_box_streamlines(), dim=100, vs=1.0)
    out = tmp_path / f"outside_{register}.zv"

    ingest_trk_parallel(
        str(trk), str(out), num_chunks=64, workers=1,
        build_multiscale=False, register_to_rasmm=register, progress=False,
    )

    cs = _store_chunk_shape(out)
    g = get_resolution_level(open_store(str(out)), 0)
    keys = list_chunk_keys(g, "vertices")
    assert keys, "ingest produced no chunks"

    checked = 0
    for cc in keys:
        for verts in read_chunk_vertices(g, cc, dtype=np.float32, ndim=3):
            v = np.asarray(verts, dtype=np.float64)
            assert np.all(np.floor(v / cs).astype(int) == np.asarray(cc)), (
                f"chunk {cc} holds vertices belonging to "
                f"{np.unique(np.floor(v / cs).astype(int), axis=0)}"
            )
            checked += len(v)
    assert checked > 0


def test_store_bounds_cover_the_geometry(tmp_path):
    """The store's declared bounds come from the data, so they contain it —
    the header's box does not even overlap this fixture on two axes."""
    trk = tmp_path / "outside.trk"
    streamlines = _outside_header_box_streamlines()
    _write_radiological_trk(trk, streamlines, dim=100, vs=1.0)
    out = tmp_path / "outside.zv"

    ingest_trk_parallel(
        str(trk), str(out), num_chunks=64, workers=1,
        build_multiscale=False, progress=False,
    )

    zv = _root_attrs(out)
    lo, hi = np.asarray(zv["bounds"][0]), np.asarray(zv["bounds"][1])

    cloud = np.concatenate([np.asarray(s, dtype=np.float64) for s in streamlines])
    assert np.all(lo <= cloud.min(0) + 1e-6)
    assert np.all(hi >= cloud.max(0) - 1e-6)
    # And the header's own box would not have: dim=100, vs=1 declares 0..100.
    assert cloud.max(0)[0] > 100.0, "fixture must exceed the declared FOV"


def test_no_vertices_are_lost(tmp_path):
    """Binning is a partition: dropping the clamp must not drop geometry.
    Stored vertices == input vertices + boundary-split duplicates, and every
    input point must still be present."""
    trk = tmp_path / "outside.trk"
    streamlines = _outside_header_box_streamlines()
    _write_radiological_trk(trk, streamlines, dim=100, vs=1.0)
    out = tmp_path / "outside.zv"

    summary = ingest_trk_parallel(
        str(trk), str(out), num_chunks=64, workers=1,
        build_multiscale=False, progress=False,
    )

    n_in = sum(len(s) for s in streamlines)
    assert summary["vertex_count"] == n_in, "split_polyline_at_boundaries preserves count"
    assert summary["streamline_count"] == len(streamlines)

    g = get_resolution_level(open_store(str(out)), 0)
    stored = np.concatenate([
        v for cc in list_chunk_keys(g, "vertices")
        for v in read_chunk_vertices(g, cc, dtype=np.float32, ndim=3)
    ])
    assert len(stored) == n_in
    want = np.concatenate([np.asarray(s) for s in streamlines])
    assert np.allclose(np.sort(stored, axis=0), np.sort(want, axis=0), atol=1e-4)


def test_chunk_shape_tracks_the_tracts_not_the_declared_fov(tmp_path):
    """chunk_shape is sized from a per-streamline vertex sample, so a header
    declaring a far larger FOV than the tracts occupy no longer inflates it."""
    trk = tmp_path / "small_tracts.trk"
    rng = np.random.default_rng(1)
    # dim=1000 declares a 1000 mm box; the tracts occupy 100 mm of it.
    tracts = [rng.uniform(400.0, 500.0, size=(20, 3)).astype(np.float32) for _ in range(30)]
    _write_radiological_trk(trk, tracts, dim=1000, vs=1.0)
    out = tmp_path / "small_tracts.zv"

    ingest_trk_parallel(
        str(trk), str(out), num_chunks=64, workers=1,
        build_multiscale=False, progress=False,
    )

    cs = _store_chunk_shape(out)
    # Sized to the ~100 mm the tracts span (64 chunks -> 4 per axis -> ~25 mm),
    # not the declared 1000 mm (which would give ~250 mm and one single chunk).
    assert np.all(cs < 100.0), f"chunk_shape {cs} still reflects the declared FOV"
    assert len(list_chunk_keys(get_resolution_level(open_store(str(out)), 0), "vertices")) > 1


def test_each_streamline_reassembles_in_order(tmp_path):
    """A polyline's fragments must concatenate back to the polyline.

    Phase A now writes its segment table sorted by target chunk, and Phase B
    reads one slice of it per part instead of decompressing and masking the
    whole table once per chunk.  The sort has to keep
    ``(chunk, poly_id, index within the polyline)`` order, because that IS
    the order Phase B emits fragments in and therefore the order the object
    manifest records.  Get it wrong and every streamline comes back with its
    segments shuffled -- which reads as plausible geometry.
    """
    from zarr_vectors.types.polylines import read_polylines

    # Long, wandering streamlines so each one crosses several chunks and
    # re-enters some of them.
    streamlines = []
    for i in range(8):
        t = np.linspace(0.0, 1.0, 60)
        streamlines.append(np.column_stack([
            20.0 + 60.0 * t,
            40.0 + 25.0 * np.sin(t * 9.0 + i),
            40.0 + 25.0 * np.cos(t * 7.0 + i),
        ]).astype(np.float32))

    source = tmp_path / "order.trk"
    _write_radiological_trk(source, streamlines, dim=100, vs=1.0)
    store = tmp_path / "order.zv"
    ingest_trk_parallel(
        str(source), str(store), num_chunks=27, n_parts=3, workers=1,
        build_multiscale=False, progress=False,
    )

    result = read_polylines(str(store))
    by_object = {}
    for index, oid in enumerate(result["object_ids"]):
        by_object.setdefault(int(oid), []).extend(result["polylines"][index])

    assert len(by_object) == len(streamlines)
    for oid, fragments in by_object.items():
        got = np.concatenate(fragments, axis=0)
        np.testing.assert_allclose(got, streamlines[oid], atol=1e-4)
