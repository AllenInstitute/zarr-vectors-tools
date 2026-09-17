"""Tests for the LINC TRK ingester."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from nibabel.streamlines import Field, Tractogram
from nibabel.streamlines.trk import TrkFile

from zarr_vectors_tools.convert.ingest.linc_trk import ingest_linc_trk

from zarr_vectors.types.polylines import read_polylines


def _write_linc_trk(path: Path) -> None:
    """Write a small TRK containing two LINC labels."""

    streamlines = [
        np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float32),
        np.array([[3, 3, 3], [4, 4, 4]], dtype=np.float32),
        np.array([[5, 5, 5], [6, 6, 6]], dtype=np.float32),
        np.array([[7, 7, 7], [8, 8, 8]], dtype=np.float32),
    ]

    label_ids = np.array([34, 34, 35, 35], dtype=np.int32)
    label_ids_points = np.array([ [[34], [34]], [[34], [34]], [[35], [35]], [[35], [35]] ], dtype=np.int32)

    header = {
        Field.VOXEL_TO_RASMM: np.eye(4, dtype=np.float32),
        Field.VOXEL_SIZES: (1.0, 1.0, 1.0),
        Field.DIMENSIONS: (50, 50, 50),
        Field.VOXEL_ORDER: "RAS",
    }

    tractogram = Tractogram(
        streamlines,
        data_per_streamline={"label_id": label_ids},
        data_per_point={"label_id": label_ids_points},
        affine_to_rasmm=np.eye(4),
    )

    TrkFile(tractogram, header=header).save(str(path))


def test_linc_trk_groups(tmp_path: Path) -> None:
    """LINC TRK labels are written as groups."""

    source = tmp_path / "gpihb.trk"
    _write_linc_trk(source)
    
    lut = tmp_path / "labels.txt"
    lut.write_text(
        "# ID Name R G B A\n"
        "34 GPi-Hb 85 117 203 255\n"
        "35 GPi-Pf 158 252 235 255\n"
    )

    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({"gpihb.trk": 34}))

    store = tmp_path / "store.zarrvectors"

    summary = ingest_linc_trk(
        source,
        store,
        (50, 50, 50),
        lut_path=lut,
        mapping_path=mapping,
    )

    assert summary["group_count"] == 2
    
    result = read_polylines(str(store), group_ids=[35])
    
    assert result["object_ids"] == [2, 3]
    assert np.all(result["vertex_attributes"]["label_id"] == 35)
