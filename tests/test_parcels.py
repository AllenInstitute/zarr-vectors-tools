"""Per-parcel summaries, computed the way FreeSurfer's stats are.

The synthetic subject has a known white surface, thickness and parcellation,
so every row is checked against values computed straight from the source
arrays.  The real check is against a FreeSurfer subject's own
``?h.aparc.stats``: set ``ZV_FREESURFER_SUBJECT`` to a recon-all directory
with ``surf/``, ``label/`` and ``stats/`` to run it.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("nibabel")

from tests._surface_fixtures import (  # noqa: E402
    PARCELS,
    parcels,
    sheet,
    thickness,
    write_freesurfer_subject,
)
from zarr_vectors_tools.algorithms.parcels import (  # noqa: E402
    parcel_at,
    parcel_summary,
    vertex_areas,
)
from zarr_vectors_tools.convert.ingest.freesurfer import ingest_freesurfer  # noqa: E402

CHUNK = (20.0, 20.0, 20.0)


@pytest.fixture
def store(tmp_path: Path) -> Path:
    subject = write_freesurfer_subject(tmp_path / "bert")
    out = tmp_path / "bert.zarrvectors"
    ingest_freesurfer(subject, out, CHUNK, annotations=["aparc"])
    return out


def test_vertex_areas_sum_to_the_mesh_area() -> None:
    vertices, faces = sheet("left")
    v = vertices.astype(np.float64)
    total = 0.5 * np.linalg.norm(
        np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]]), axis=1,
    ).sum()
    assert vertex_areas(vertices, faces).sum() == pytest.approx(total)


def test_every_parcel_matches_the_source_arrays(store: Path) -> None:
    table = parcel_summary(store, "aparc", metrics=["thickness"])
    names = [name for name, _ in PARCELS.values()]
    for side in ("left", "right"):
        vertices, faces = sheet(side)
        n = len(vertices)
        areas = vertex_areas(vertices, faces)
        rows = parcels(n) - 1
        thick = thickness(n).astype(np.float64)
        for k, name in enumerate(names):
            members = rows == k
            row = table.loc[(side, name)]
            assert row.vertex_count == members.sum()
            assert row.surface_area == pytest.approx(areas[members].sum(), rel=1e-5)
            assert row.thickness_mean == pytest.approx(thick[members].mean(), rel=1e-5)
            assert row.thickness_std == pytest.approx(thick[members].std(), rel=1e-4)
            weighted = (thick[members] * areas[members]).sum() / areas[members].sum()
            assert row.thickness_area_weighted_mean == pytest.approx(weighted, rel=1e-5)


def test_defaults_summarise_every_listed_map(store: Path) -> None:
    table = parcel_summary(store, "aparc", hemispheres=["lh"])
    assert set(table.index.get_level_values("hemisphere")) == {"left"}
    assert {"thickness_mean", "curv_mean"} <= set(table.columns)


def test_parcel_at_points(store: Path) -> None:
    vertices, _faces = sheet("right")
    # The store is in scanner space: find a vertex's stored position via the
    # summary's own reader, then nudge it off the surface.
    from zarr_vectors_tools.algorithms.surfaces import read_hemisphere

    mesh = read_hemisphere(store, "right", coords="white")
    target = 7
    point = mesh["vertices"][target] + [0.0, 0.0, 0.05]
    table = parcel_at(store, [point, [1e4, 1e4, 1e4]], "aparc", max_distance=5.0)
    expected = [name for name, _ in PARCELS.values()][int(parcels(len(vertices))[target] - 1)]
    assert table.loc[0, "hemisphere"] == "right"
    assert table.loc[0, "parcel"] == expected
    assert table.loc[0, "vertex"] == target
    assert pd.isna(table.loc[1, "parcel"])


@pytest.mark.parametrize("kwargs, message", [
    ({"parcellation": "destrieux"}, "not a parcellation"),
    ({"parcellation": "aparc", "surface": "midpial"}, "no 'midpial' surface"),
    ({"parcellation": "aparc", "metrics": ["myelin"]}, "not at level 0"),
    ({"parcellation": "aparc", "hemispheres": ["both"]}, "no \\['both'\\] hemisphere"),
])
def test_refusals(store: Path, kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        parcel_summary(store, **kwargs)


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("ZV_FREESURFER_SUBJECT"),
    reason="set ZV_FREESURFER_SUBJECT to a recon-all subject with stats/",
)
def test_matches_freesurfer_aparc_stats(tmp_path: Path) -> None:
    subject = Path(os.environ["ZV_FREESURFER_SUBJECT"])
    store = tmp_path / "subject.zarrvectors"
    ingest_freesurfer(
        subject, store, (40.0, 40.0, 40.0), annotations=["aparc"], morphometry=["thickness"],
    )
    table = parcel_summary(store, "aparc", metrics=["thickness"])
    for side, prefix in (("left", "lh"), ("right", "rh")):
        for line in open(subject / "stats" / f"{prefix}.aparc.stats"):
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split()
            row = table.loc[(side, fields[0])]
            # The stats file rounds area to integers and thickness to 3 places.
            assert row.vertex_count == int(fields[1])
            assert abs(row.surface_area - float(fields[2])) <= 0.5 + 1e-6
            assert abs(row.thickness_mean - float(fields[4])) <= 5e-4 + 1e-9
            assert abs(row.thickness_std - float(fields[5])) <= 5e-4 + 1e-9
