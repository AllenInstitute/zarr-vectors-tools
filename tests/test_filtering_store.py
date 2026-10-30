"""Filtering streamlines read from a store, and select_streamlines against the oracle.

What has to hold: load_tractogram gives back exactly what was written --
geometry, object attributes with their absent rows, groups -- in the RAS
millimetres select_streamlines uses, voxel-millimetre TRK stores included.
And select_streamlines, which reads only the chunks near a region, finds
the same streamlines as the oracle testing every segment, for regions that
straddle chunk faces.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from zarr_vectors.types.polylines import write_polylines

from zarr_vectors_tools.algorithms import select_streamlines
from zarr_vectors_tools.filtering import (
    Mask,
    Passes,
    Sphere,
    Tractogram,
    evaluate,
    load_tractogram,
    whole_brain_like,
)

CHUNK = (6.0, 6.0, 6.0)


@pytest.fixture(scope="module")
def synthetic() -> Tractogram:
    return whole_brain_like(
        400, seed=5, semi_axes=(15.0, 18.0, 12.0), mean_length=20.0,
        length_range=(3.0, 50.0), step=2.5, n_groups=3,
    )


@pytest.fixture(scope="module")
def store(synthetic: Tractogram, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("filtering") / "tracts.zv"
    lines = [synthetic.polyline(k) for k in range(synthetic.n_objects)]
    write_polylines(
        str(path), lines, chunk_shape=CHUNK, geometry_type="streamline",
        object_attributes={"fa_mean": synthetic.columns["fa_mean"],
                           "label": synthetic.columns["label"]},
        groups={g: members.tolist() for g, members in enumerate(synthetic.groups)},
    )
    return path


def test_load_gives_back_what_was_written(synthetic: Tractogram, store: Path) -> None:
    loaded = load_tractogram(store)
    assert loaded.n_objects == synthetic.n_objects
    np.testing.assert_array_equal(loaded.offsets, synthetic.offsets)
    np.testing.assert_array_equal(loaded.vertices, synthetic.vertices.astype(np.float64))
    np.testing.assert_array_equal(loaded.columns["label"], synthetic.columns["label"])
    fa, present = loaded.column("fa_mean")
    want, want_present = synthetic.column("fa_mean")
    np.testing.assert_array_equal(present, want_present)
    np.testing.assert_array_equal(fa[present], want[want_present])
    np.testing.assert_allclose(loaded.columns["length"], synthetic.columns["length"], rtol=1e-12)
    assert [g.tolist() for g in loaded.groups] == [g.tolist() for g in synthetic.groups]


def test_load_reads_only_the_columns_asked_for(store: Path) -> None:
    loaded = load_tractogram(store, columns=["label", "not_there"])
    assert sorted(loaded.columns) == ["label", "length", "n_vertices"]


def test_load_refuses_what_is_not_streamlines(tmp_path: Path) -> None:
    from zarr_vectors.types.points import write_points

    path = tmp_path / "points.zv"
    write_points(str(path), np.random.default_rng(0).random((10, 3)) * 10, chunk_shape=CHUNK)
    with pytest.raises(ValueError, match="not streamlines"):
        load_tractogram(path)


def _random_mask(rng: np.random.Generator, anchor: np.ndarray) -> Mask:
    shape = tuple(rng.integers(1, 5, size=3))
    voxels = rng.random(shape) < 0.5
    voxels.flat[0] = True
    size = rng.uniform(0.3, 2.0)
    affine = np.diag([size, size, size, 1.0])
    affine[:3, 3] = anchor - size * np.asarray(shape) / 2.0
    return Mask(voxels, affine)


def test_select_streamlines_agrees_with_the_oracle(synthetic: Tractogram, store: Path) -> None:
    loaded = load_tractogram(store)
    rng = np.random.default_rng(8)
    chunk = np.asarray(CHUNK)
    nonempty = 0
    for trial in range(16):
        anchor = loaded.vertices[rng.integers(len(loaded.vertices))]
        if trial % 2:
            # Pull the region onto the nearest chunk face on one axis.
            axis = rng.integers(3)
            anchor = anchor.copy()
            anchor[axis] = np.round(anchor[axis] / chunk[axis]) * chunk[axis]
        mask = _random_mask(rng, anchor)
        for mode in ("path", "endpoints", "both_endpoints"):
            want = np.flatnonzero(evaluate(Passes(mask, mode), loaded))
            got = select_streamlines(store, mask=mask.voxels, mask_affine=mask.affine, mode=mode)
            np.testing.assert_array_equal(got, want, err_msg=f"trial {trial}, {mode}")
            nonempty += mode == "path" and len(want) > 0
    assert nonempty >= 12


def test_a_voxmm_trk_store_loads_in_ras(tmp_path: Path) -> None:
    pytest.importorskip("nibabel")
    from nibabel.streamlines import Field
    from nibabel.streamlines import Tractogram as NibTractogram
    from nibabel.streamlines.trk import TrkFile

    from zarr_vectors_tools.convert.ingest.trk_parallel import ingest_trk_parallel

    trk_affine = np.array(
        [[-2, 0, 0, 40], [0, 2, 0, -40], [0, 0, 2, -40], [0, 0, 0, 1]], dtype=np.float32,
    )
    through = np.column_stack([np.linspace(-30, 30, 13), np.zeros(13), np.zeros(13)])
    away = np.column_stack([np.linspace(-30, 30, 13), np.full(13, 25.0), np.full(13, 25.0)])
    path = tmp_path / "in.trk"
    TrkFile(
        NibTractogram([through.astype(np.float32), away.astype(np.float32)],
                      affine_to_rasmm=np.eye(4)),
        header={Field.VOXEL_TO_RASMM: trk_affine, Field.VOXEL_SIZES: (2, 2, 2),
                Field.DIMENSIONS: (40, 40, 40), Field.VOXEL_ORDER: "LAS"},
    ).save(str(path))
    store = tmp_path / "voxmm.zv"
    ingest_trk_parallel(
        str(path), str(store), num_chunks=8, n_parts=1, workers=1,
        build_multiscale=False, progress=False,
    )
    loaded = load_tractogram(store)
    lines = sorted((loaded.polyline(k) for k in range(loaded.n_objects)), key=lambda p: p[0, 1])
    np.testing.assert_allclose(lines[0], through, atol=1e-4)
    np.testing.assert_allclose(lines[1], away, atol=1e-4)
    # The oracle and select_streamlines read the same RAS box the same way.
    ids = np.flatnonzero(evaluate(Passes(Sphere((0, 0, 0), 3.0)), loaded))
    box = select_streamlines(store, box=([-3, -3, -3], [3, 3, 3]))
    assert ids.tolist() == box.tolist() and len(ids) == 1
