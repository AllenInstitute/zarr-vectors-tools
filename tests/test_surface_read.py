"""Reading a hemisphere back out of a surface store, as any of its surfaces.

A surface store chunks one surface and keeps the others as coordinate
attributes, and orders vertices by chunk.  ``read_hemisphere`` has to hand
them back numbered like the source file, with the faces of the source file,
whichever surface supplies the coordinates -- that is what lets the inflated
or white surface be drawn from a store built on the midthickness.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("nibabel")

from tests._surface_fixtures import parcels, sheet, thickness, write_gifti_subject  # noqa: E402
from zarr_vectors_tools.algorithms.surfaces import read_hemisphere  # noqa: E402
from zarr_vectors_tools.convert.ingest.gifti import ingest_gifti  # noqa: E402

CHUNK = (20.0, 20.0, 20.0)


@pytest.fixture
def store(tmp_path: Path) -> Path:
    source = write_gifti_subject(tmp_path / "gii")
    out = tmp_path / "surf.zarrvectors"
    ingest_gifti(source, out, CHUNK)
    return out


def _same_triangles(a: np.ndarray, b: np.ndarray) -> bool:
    """Faces equal as sets of triangles, whatever the winding start."""
    def canon(faces):
        rolled = []
        for tri in np.asarray(faces).tolist():
            k = tri.index(min(tri))
            rolled.append(tuple(tri[k:] + tri[:k]))
        return sorted(rolled)
    return canon(a) == canon(b)


@pytest.mark.parametrize("hemisphere", ["left", "right", "lh"])
def test_the_geometry_comes_back_in_source_order(store: Path, hemisphere: str) -> None:
    side = {"lh": "left"}.get(hemisphere, hemisphere)
    vertices, faces = sheet(side)
    result = read_hemisphere(store, hemisphere)

    assert result["surface"] == "midthickness"
    np.testing.assert_allclose(result["vertices"], vertices, atol=1e-5)
    assert result["source_vertex"].tolist() == list(range(len(vertices)))
    assert _same_triangles(result["faces"], faces)


def test_an_alternate_surface_keeps_the_faces(store: Path) -> None:
    vertices, faces = sheet("right")
    inflated = read_hemisphere(store, "right", coords="inflated")
    by_attribute = read_hemisphere(store, "right", coords="coords_inflated")

    assert inflated["surface"] == by_attribute["surface"] == "inflated"
    np.testing.assert_allclose(inflated["vertices"], vertices * 1.5, atol=1e-4)
    np.testing.assert_array_equal(inflated["faces"], by_attribute["faces"])
    geometry = read_hemisphere(store, "right")
    np.testing.assert_array_equal(inflated["faces"], geometry["faces"])


def test_attributes_come_back_on_the_same_vertices(store: Path) -> None:
    n = len(sheet("left")[0])
    result = read_hemisphere(store, "left", attributes=["thickness", "aparc"])
    np.testing.assert_allclose(result["attributes"]["thickness"], thickness(n), atol=1e-6)
    np.testing.assert_array_equal(result["attributes"]["aparc"], parcels(n))


@pytest.mark.parametrize("kwargs, message", [
    ({"hemisphere": "left", "coords": "sphere"}, "no 'sphere' surface"),
    ({"hemisphere": "left", "attributes": ["curv"]}, "not at level 0"),
])
def test_what_the_store_lacks_is_named(store: Path, kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        read_hemisphere(store, **kwargs)


def test_a_missing_hemisphere_is_named(tmp_path: Path) -> None:
    source = write_gifti_subject(tmp_path / "gii", hemispheres=("left",))
    out = tmp_path / "left.zarrvectors"
    ingest_gifti(source, out, CHUNK)
    with pytest.raises(ValueError, match="no 'right' hemisphere"):
        read_hemisphere(out, "right")


def test_a_store_that_is_not_a_surface_is_refused(tmp_path: Path) -> None:
    from zarr_vectors.types.points import write_points

    out = tmp_path / "points.zv"
    write_points(str(out), np.ones((3, 3), np.float32), chunk_shape=CHUNK)
    with pytest.raises(ValueError, match="no surface header"):
        read_hemisphere(out, "left")
