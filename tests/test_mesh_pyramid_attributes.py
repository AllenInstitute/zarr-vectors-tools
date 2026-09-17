"""A mesh pyramid must carry its per-vertex scalars, not just its surface.

Neither mesh coarsener read or wrote ``vertex_attributes``, so every level
above 0 lost thickness, curvature, myelin and parcel labels.  For a cortical
surface those scalars are the payload: a coarse level without them cannot be
shaded or parcellated, which makes the pyramid useless for exactly the
zoomed-out view it exists to serve.

The two coarseners carry them differently, and both are exercised here:

* **vertex clustering** merges vertices that share a bin, so a continuous
  quantity averages over the cluster and a categorical one takes the value of
  the member nearest the centroid;
* **quadric collapse** MOVES vertices to the quadric optimum, so there is no
  index to follow and the value comes from the nearest source vertex.

Both must leave a categorical code a value that genuinely occurred.  A mean
of parcels 3 and 7 is parcel 5, which is a different piece of cortex.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from zarr_vectors.building import (
    get_resolution_level,
    list_chunk_keys,
    open_store,
    read_chunk_attributes,
    read_chunk_vertices,
    read_level_metadata,
)
from zarr_vectors.types.meshes import write_mesh

from zarr_vectors_tools.multiresolution.coarsen import build_pyramid

CHUNK = (20.0, 20.0, 20.0)


def _sheet(n: int = 13):
    """A subdivided square sheet with one continuous and one categorical map.

    ``thickness`` rises linearly with x, so a cluster mean stays inside the
    source range.  ``parcel`` is 0 on one half and 1 on the other, so any
    value other than 0 or 1 is an invented label.
    """
    u = np.linspace(0.0, 12.0, n)
    uu, ww = np.meshgrid(u, u, indexing="ij")
    vertices = np.column_stack(
        [uu.ravel(), ww.ravel(), np.zeros(uu.size)],
    ).astype(np.float32)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            faces += [[a, a + 1, a + n], [a + 1, a + n + 1, a + n]]
    thickness = (vertices[:, 0] * 0.1 + 2.0).astype(np.float32)
    parcel = (vertices[:, 0] > 6.0).astype(np.int32)
    return vertices, np.asarray(faces, dtype=np.int64), thickness, parcel


def _read_level_attribute(store: Path, level: int, name: str) -> np.ndarray:
    group = get_resolution_level(open_store(str(store)), level)
    parts = [
        np.asarray(block).ravel()
        for chunk in list_chunk_keys(group, "vertices")
        for block in read_chunk_attributes(group, name, chunk)
    ]
    return np.concatenate(parts) if parts else np.zeros(0)


def _level_vertex_count(store: Path, level: int) -> int:
    group = get_resolution_level(open_store(str(store)), level)
    return sum(
        len(block)
        for chunk in list_chunk_keys(group, "vertices")
        for block in read_chunk_vertices(group, chunk, ndim=3)
    )


@pytest.mark.parametrize("method", ["mesh", "mesh_decimate"])
class TestMeshAttributeCarry:

    def _build(self, tmp_path: Path, method: str) -> Path:
        store = tmp_path / f"{method}.zv"
        vertices, faces, thickness, parcel = _sheet()
        write_mesh(
            str(store), vertices, faces, chunk_shape=CHUNK,
            vertex_attributes={"thickness": thickness, "parcel": parcel},
        )
        build_pyramid(
            str(store), factors=[(2.0, 1.0)], method=method,
            cross_level_depth=0, cross_level_storage="none",
        )
        return store

    def test_the_level_declares_and_holds_both(
        self, tmp_path: Path, method: str,
    ) -> None:
        store = self._build(tmp_path, method)
        meta = read_level_metadata(open_store(str(store)), 1)
        assert "vertex_attributes" in meta.arrays_present

        count = _level_vertex_count(store, 1)
        for name in ("thickness", "parcel"):
            values = _read_level_attribute(store, 1, name)
            assert len(values) == count, (
                f"{name}: {len(values)} values for {count} vertices"
            )

    def test_a_continuous_map_stays_in_range(
        self, tmp_path: Path, method: str,
    ) -> None:
        """Averaging within a cluster cannot leave the source range."""
        store = self._build(tmp_path, method)
        thickness = _read_level_attribute(store, 1, "thickness")
        assert thickness.min() >= 2.0 - 1e-5
        assert thickness.max() <= 3.2 + 1e-5
        # Not all one value: the map must still vary across the sheet.
        assert thickness.max() - thickness.min() > 0.5

    def test_a_categorical_map_keeps_only_real_labels(
        self, tmp_path: Path, method: str,
    ) -> None:
        """A blended parcel code names cortex that is not there."""
        store = self._build(tmp_path, method)
        parcel = _read_level_attribute(store, 1, "parcel")
        assert set(np.unique(parcel).tolist()) <= {0, 1}
        # Both parcels survive; the sheet is half and half.
        assert set(np.unique(parcel).tolist()) == {0, 1}

    def test_a_store_without_attributes_still_coarsens(
        self, tmp_path: Path, method: str,
    ) -> None:
        store = tmp_path / f"bare_{method}.zv"
        vertices, faces, _t, _p = _sheet()
        write_mesh(str(store), vertices, faces, chunk_shape=CHUNK)
        summary = build_pyramid(
            str(store), factors=[(2.0, 1.0)], method=method,
            cross_level_depth=0, cross_level_storage="none",
        )
        assert summary["level_specs"][0]["attributes_carried"] == []
        meta = read_level_metadata(open_store(str(store)), 1)
        assert "vertex_attributes" not in meta.arrays_present
