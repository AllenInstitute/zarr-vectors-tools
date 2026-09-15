"""A cortical sheet stays one closed sheet up the quadric pyramid.

On FreeSurfer's bert, quadric collapse left 65 duplicate triangles per
hemisphere per level, so a closed surface's Euler characteristic read 67
instead of 2 at every coarse level.  The coarser levels now drop repeated
and degenerate triangles before writing.  Checked here on a closed synthetic
sphere, where the Euler characteristic must stay 2, and directly on the
helper.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zarr_vectors_tools.multiresolution.strategies.mesh_decimate_level import (
    _drop_duplicate_faces,
)


def test_duplicates_and_degenerates_are_dropped_keeping_first_winding() -> None:
    faces = np.array([
        [0, 1, 2],
        [2, 1, 0],   # same triangle, other winding
        [1, 2, 0],   # same triangle, rotated
        [0, 0, 3],   # degenerate
        [1, 2, 3],
    ])
    kept = _drop_duplicate_faces(faces)
    assert kept.tolist() == [[0, 1, 2], [1, 2, 3]]


def test_clean_faces_come_back_untouched() -> None:
    faces = np.array([[0, 1, 2], [1, 2, 3]])
    assert _drop_duplicate_faces(faces) is faces


def _icosphere(subdivisions: int = 4, radius: float = 40.0):
    t = (1.0 + 5 ** 0.5) / 2.0
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
             (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9),
             (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2),
             (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
             (8, 6, 7), (9, 8, 1)]
    verts = [np.array(v, float) / np.linalg.norm(v) for v in verts]
    for _ in range(subdivisions):
        cache: dict[tuple[int, int], int] = {}

        def middle(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]

        faces = [
            f for a, b, c in faces
            for f in ((a, middle(a, b), middle(c, a)), (b, middle(b, c), middle(a, b)),
                      (c, middle(c, a), middle(b, c)),
                      (middle(a, b), middle(b, c), middle(c, a)))
        ]
    return (np.asarray(verts) * radius + 60.0).astype(np.float32), np.asarray(faces)


def _euler(faces: np.ndarray) -> int:
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    return len(np.unique(faces)) - len(np.unique(edges, axis=0)) + len(faces)


@pytest.mark.slow
def test_a_closed_sphere_keeps_euler_2_up_the_pyramid(tmp_path: Path) -> None:
    pytest.importorskip("pyfqmr")
    from zarr_vectors.building import list_resolution_levels, open_store
    from zarr_vectors.types.meshes import read_mesh, write_mesh

    from zarr_vectors_tools.multiresolution.coarsen import build_pyramid

    vertices, faces = _icosphere()
    store = tmp_path / "sphere.zv"
    write_mesh(str(store), vertices, faces, chunk_shape=(30.0, 30.0, 30.0))
    assert _euler(faces) == 2

    build_pyramid(str(store), factors=[(4.0, 1.0), (4.0, 1.0)], method="mesh_decimate")
    for level in list_resolution_levels(open_store(str(store))):
        mesh = read_mesh(str(store), level=level)
        coarse = np.asarray(mesh["faces"])
        assert _euler(coarse) == 2, level
        assert len(np.unique(np.sort(coarse, axis=1), axis=0)) == len(coarse), level
