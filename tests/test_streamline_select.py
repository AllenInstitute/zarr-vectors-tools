"""Selecting streamlines by region.

What has to hold: a streamline that passes through a region is found, even
where no vertex lands inside it; the endpoint modes look at the ends only;
a TRK store kept in voxel millimetres answers a RAS mask the same as one
registered at ingest; and only the chunks the region touches are read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from zarr_vectors.types.polylines import write_polylines

from zarr_vectors_tools.algorithms import streamline_select
from zarr_vectors_tools.algorithms.streamline_select import select_streamlines

CHUNK = (10.0, 10.0, 10.0)

#: A 1 mm mask on a 40 mm cube starting at the origin, with a 4 mm ROI cube
#: at voxels 18..21 (18..21 mm) on every axis.
AFFINE = np.eye(4)


def _mask() -> np.ndarray:
    mask = np.zeros((40, 40, 40), dtype=np.uint8)
    mask[18:22, 18:22, 18:22] = 1
    return mask


@pytest.fixture
def store(tmp_path: Path) -> Path:
    lines = [
        # 0: straight through the ROI, ends far outside it
        np.column_stack([np.linspace(2, 38, 13), np.full(13, 20.0), np.full(13, 20.0)]),
        # 1: starts inside the ROI, leaves along y
        np.column_stack([np.full(6, 19.5), np.linspace(19.5, 38, 6), np.full(6, 20.0)]),
        # 2: nowhere near it
        np.column_stack([np.linspace(2, 8, 4), np.full(4, 3.0), np.full(4, 3.0)]),
        # 3: one segment stepping over the ROI -- no vertex inside it, both
        #    ends in chunks the ROI touches
        np.array([[20.0, 12.0, 20.0], [20.0, 28.0, 20.0]]),
        # 4: both ends inside, a loop out and back
        np.array([[19.0, 19.0, 19.0], [19.0, 30.0, 19.0], [21.0, 30.0, 21.0], [21.0, 21.0, 21.0]]),
    ]
    path = tmp_path / "lines.zv"
    write_polylines(
        str(path), [np.asarray(line, np.float32) for line in lines],
        chunk_shape=CHUNK, geometry_type="streamline",
    )
    return path


def test_path_finds_every_streamline_through_the_region(store: Path) -> None:
    ids = select_streamlines(store, mask=_mask(), mask_affine=AFFINE)
    assert ids.tolist() == [0, 1, 3, 4]


def test_endpoint_modes(store: Path) -> None:
    either = select_streamlines(store, mask=_mask(), mask_affine=AFFINE, mode="endpoints")
    both = select_streamlines(store, mask=_mask(), mask_affine=AFFINE, mode="both_endpoints")
    assert either.tolist() == [1, 4]
    assert both.tolist() == [4]


def test_a_box_works_like_a_mask(store: Path) -> None:
    ids = select_streamlines(store, box=([17.5, 17.5, 17.5], [21.5, 21.5, 21.5]))
    assert ids.tolist() == [0, 1, 3, 4]


def test_only_the_chunks_the_region_touches_are_read(
    store: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from zarr_vectors.types import polylines

    seen: list = []
    real = polylines.read_polylines

    def _spy(*args, **kwargs):
        seen.append(kwargs.get("chunks"))
        return real(*args, **kwargs)

    monkeypatch.setattr(polylines, "read_polylines", _spy)
    select_streamlines(store, mask=_mask(), mask_affine=AFFINE)
    # The ROI spans 17.5..21.5 mm, so chunks 1 and 2 on each axis; streamline
    # 2 lives in chunk 0 and is never read.
    assert seen and seen[0]
    assert all(all(int(c) in (1, 2) for c in cc) for cc in seen[0])


def test_a_nifti_file_is_read_with_its_affine(store: Path, tmp_path: Path) -> None:
    nib = pytest.importorskip("nibabel")
    # 2 mm voxels, origin shifted by -2 mm: the same ROI is voxels 10..11.
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = -2.0
    mask = np.zeros((21, 21, 21), dtype=np.uint8)
    mask[10:12, 10:12, 10:12] = 1        # 18..20 mm centres, 17..21 mm extent
    path = tmp_path / "roi.nii.gz"
    nib.save(nib.Nifti1Image(mask, affine), str(path))
    assert select_streamlines(store, mask=path).tolist() == [0, 1, 3, 4]


def test_a_voxmm_trk_store_answers_a_ras_mask(tmp_path: Path) -> None:
    pytest.importorskip("nibabel")
    from nibabel.streamlines import Field, Tractogram
    from nibabel.streamlines.trk import TrkFile

    from zarr_vectors_tools.convert.ingest.trk_parallel import ingest_trk_parallel

    # LAS, 2 mm: stored voxmm coordinates look nothing like RAS.
    trk_affine = np.array(
        [[-2, 0, 0, 40], [0, 2, 0, -40], [0, 0, 2, -40], [0, 0, 0, 1]], dtype=np.float32,
    )
    through = np.column_stack([np.linspace(-30, 30, 13), np.zeros(13), np.zeros(13)])
    away = np.column_stack([np.linspace(-30, 30, 13), np.full(13, 25.0), np.full(13, 25.0)])
    path = tmp_path / "in.trk"
    TrkFile(
        Tractogram([through.astype(np.float32), away.astype(np.float32)],
                   affine_to_rasmm=np.eye(4)),
        header={Field.VOXEL_TO_RASMM: trk_affine, Field.VOXEL_SIZES: (2, 2, 2),
                Field.DIMENSIONS: (40, 40, 40), Field.VOXEL_ORDER: "LAS"},
    ).save(str(path))
    store = tmp_path / "voxmm.zv"
    ingest_trk_parallel(
        str(path), str(store), num_chunks=8, n_parts=1, workers=1,
        build_multiscale=False, progress=False,
    )
    # A RAS box around the origin.
    ids = select_streamlines(store, box=([-3, -3, -3], [3, 3, 3]))
    assert len(ids) == 1
    from zarr_vectors.types.polylines import read_polylines

    chosen = read_polylines(str(store), object_ids=ids.tolist())["polylines"][0]
    assert len(np.concatenate(chosen)) == 13


@pytest.mark.parametrize("kwargs, message", [
    ({}, "exactly one of"),
    ({"box": ([0, 0, 0], [1, 1, 1]), "mask": np.ones((2, 2, 2))}, "exactly one of"),
    ({"box": ([0, 0, 0], [1, 1, 1]), "mode": "middle"}, "mode must be"),
    ({"mask": np.ones((2, 2, 2))}, "mask_affine"),
    ({"box": ([2, 2, 2], [1, 1, 1])}, "below its min corner"),
])
def test_refusals(store: Path, kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        select_streamlines(store, **kwargs)


def test_a_store_without_streamlines_is_refused(tmp_path: Path) -> None:
    from zarr_vectors.types.points import write_points

    path = tmp_path / "points.zv"
    write_points(str(path), np.ones((3, 3), np.float32), chunk_shape=CHUNK)
    with pytest.raises(ValueError, match="not streamlines"):
        select_streamlines(path, box=([0, 0, 0], [5, 5, 5]))


def test_segment_sampling_is_what_finds_a_jump(store: Path) -> None:
    coarse = select_streamlines(
        store, mask=_mask(), mask_affine=AFFINE, sample_spacing=100.0,
    )
    assert 3 not in coarse.tolist()
    assert streamline_select._along_segments(
        np.array([[0.0, 0, 0], [10.0, 0, 0]]), 2.5,
    ).shape == (5, 3)
