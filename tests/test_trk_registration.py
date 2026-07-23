"""TRK ingest registers streamlines to RASmm world space.

The parallel TRK ingest stores raw TrackVis "voxmm" point data.  On its own
that is *unregistered*: voxmm starts at the voxel-grid corner (0,0,0) and a
radiological ``vox_to_ras`` affine (negative X/Y — a 180° rotation in the XY
plane, the standard LPS→RAS convention) is never applied.  A store built that
way sits shifted to the chunk-grid origin and axis-flipped relative to the
source image, and NGFF transforms cannot fix it (they scale/translate but
cannot rotate).

Opt-in (``register_to_rasmm=True``, CLI ``--apply-affine``) bakes that affine
into the geometry, matching what ``nibabel.streamlines.load(...).streamlines``
returns and producing an identity CRS affine — the layout a viewer that can't
apply an affine (neuroglancer) needs.  The default keeps raw voxmm + the real
affine in CRS.  These tests pin both: opt-in reproduces nibabel's RASmm cloud;
the default keeps raw voxmm.

The fixture is hand-built (not written via nibabel): nibabel normalises any TRK
it writes to a near-identity trackvis affine, which would make the registered
and raw stores indistinguishable and the test vacuous.  A real dsi-studio /
TrackVis file carries a genuine radiological affine, reproduced here.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from zarr_vectors.core.arrays import list_chunk_keys, read_chunk_vertices
from zarr_vectors.core.store import get_resolution_level, open_store

from zarr_vectors_tools.ingest.trk_parallel import ingest_trk_parallel


def _write_radiological_trk(path, streamlines_voxmm, dim=100, vs=1.0):
    """Write a minimal TRK with an LPS voxel order and a radiological
    ``vox_to_ras`` (negative X/Y, origin at the volume centre).  Points are in
    voxmm (voxel_index × voxel_size), i.e. what is physically stored on disk."""
    half = dim * vs / 2.0
    affine = np.array(
        [[-1.0, 0, 0, half], [0, -1.0, 0, half], [0, 0, 1.0, -half], [0, 0, 0, 1.0]],
        dtype=np.float32,
    )
    h = bytearray(1000)
    struct.pack_into("<6s", h, 0, b"TRACK\x00")
    struct.pack_into("<3h", h, 6, dim, dim, dim)
    struct.pack_into("<3f", h, 12, vs, vs, vs)          # voxel_sizes
    struct.pack_into("<3f", h, 24, 0.0, 0.0, 0.0)       # origin
    struct.pack_into("<h", h, 36, 0)                     # n_scalars
    struct.pack_into("<h", h, 238, 0)                    # n_properties
    struct.pack_into("<16f", h, 440, *affine.flatten())  # vox_to_ras
    struct.pack_into("<4s", h, 948, b"LPS\x00")          # voxel_order
    struct.pack_into("<6f", h, 956, 1.0, 0, 0, 0, 1.0, 0)  # image_orientation_patient
    struct.pack_into("<i", h, 988, len(streamlines_voxmm))  # n_count
    struct.pack_into("<i", h, 992, 2)                    # version
    struct.pack_into("<i", h, 996, 1000)                 # hdr_size
    body = bytearray(h)
    for s in streamlines_voxmm:
        body += struct.pack("<i", len(s)) + np.asarray(s, "<f4").tobytes()
    Path(path).write_bytes(body)
    return affine


def _streamlines_voxmm(seed=0, n=25, npts=20, lo=10.0, hi=90.0):
    rng = np.random.default_rng(seed)
    return [rng.uniform(lo, hi, size=(npts, 3)).astype(np.float32) for _ in range(n)]


def _vertex_cloud(store, level=0):
    g = get_resolution_level(open_store(str(store)), level)
    pts = [
        v
        for cc in list_chunk_keys(g, "vertices")
        for v in read_chunk_vertices(g, cc, dtype=np.float32, ndim=3)
    ]
    return np.concatenate(pts, axis=0)


def test_ingest_registers_to_rasmm(tmp_path):
    """register_to_rasmm=True reproduces nibabel's RASmm streamlines (registered)."""
    nib = pytest.importorskip("nibabel")
    trk = tmp_path / "rad.trk"
    _write_radiological_trk(trk, _streamlines_voxmm())

    ref = np.concatenate(
        [np.asarray(s) for s in nib.streamlines.load(str(trk)).streamlines], axis=0
    )
    assert (ref < 0).any(), "radiological fixture must span negative world coords"

    out = tmp_path / "reg.zv"
    ingest_trk_parallel(
        str(trk), str(out), num_chunks=64, workers=1,
        build_multiscale=False, register_to_rasmm=True, progress=False,
    )
    stored = _vertex_cloud(out)

    # The stored cloud is nibabel's RASmm cloud (registered to the source image).
    assert np.allclose(stored.min(0), ref.min(0), atol=1.0)
    assert np.allclose(stored.max(0), ref.max(0), atol=1.0)
    assert (stored < 0).any(), "registered store must carry world-space negatives"


def test_ingest_default_keeps_raw_voxmm(tmp_path):
    """The default (no ``register_to_rasmm``) keeps raw, all-positive voxmm."""
    pytest.importorskip("nibabel")
    trk = tmp_path / "rad.trk"
    _write_radiological_trk(trk, _streamlines_voxmm())

    out = tmp_path / "raw.zv"
    ingest_trk_parallel(  # no register_to_rasmm → default False
        str(trk), str(out), num_chunks=64, workers=1,
        build_multiscale=False, progress=False,
    )
    stored = _vertex_cloud(out)
    # voxmm = voxel_index × voxel_size ≥ 0; the radiological flip is NOT applied.
    assert (stored >= -1e-2).all(), "raw voxmm coordinates must be non-negative"
    assert stored.max() > 1.0


def test_cli_apply_affine_flag(tmp_path):
    """``convert --apply-affine`` bakes RASmm; without it, raw voxmm is kept."""
    nib = pytest.importorskip("nibabel")
    from zarr_vectors_tools.cli import main

    trk = tmp_path / "rad.trk"
    _write_radiological_trk(trk, _streamlines_voxmm())
    ref = np.concatenate(
        [np.asarray(s) for s in nib.streamlines.load(str(trk)).streamlines], axis=0
    )

    baked, raw = tmp_path / "baked.zv", tmp_path / "raw.zv"
    assert main(["convert", str(trk), str(baked),
                 "--num-chunks", "64", "--workers", "1", "--apply-affine"]) == 0
    assert main(["convert", str(trk), str(raw),
                 "--num-chunks", "64", "--workers", "1"]) == 0

    vb, vr = _vertex_cloud(baked), _vertex_cloud(raw)
    # --apply-affine → RASmm world space (matches nibabel, carries negatives)
    assert np.allclose(vb.min(0), ref.min(0), atol=1.0)
    assert (vb < 0).any()
    # default → raw voxmm (non-negative, flip NOT applied)
    assert (vr >= -1e-2).all()
