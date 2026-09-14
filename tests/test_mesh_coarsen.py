"""Tests for the mesh coarsener: chunk-local vertex clustering.

Before this strategy existed a mesh store fell through ``select_coarsener_key``
to ``per_object``, which is wrong for meshes in two independent ways: it
``np.concatenate``s every fragment each object's manifest names (on a store
whose objects share chunk-wide fragments that is quadratic — a 65M-vertex
MICrONS mesh store wanted 599 GB to coarsen 0.78 GB of geometry), and it
rebuilds links at a hardcoded ``link_width=2``, which a triangle cannot
survive. So the invariants under test are:

* **M1** Dispatch — a mesh store routes to ``"mesh"``, not ``"per_object"``.
* **M2** Faces survive — the coarse level has triangles, indices in range, and
  no degenerate ones (a face whose corners merged is dropped, not kept).
* **M3** Every face is considered — ``faces_in`` equals the source face count,
  including the cross-chunk ones that carry a ``perm_idx`` column.
* **M4** Per-object reads keep working — the per-object fragments partition the
  level exactly, at every level.
* **M5** It compounds — ``build_pyramid`` over several levels keeps shrinking,
  and object ids are preserved throughout.
* **M6** Coarse geometry is bounded by the source: centroids of source
  vertices cannot leave the source bounding box.
"""

from __future__ import annotations

import numpy as np
import pytest
from zarr_vectors.building import (
    get_resolution_level,
    list_resolution_levels,
    object_count,
    open_store,
    read_object_vertices,
    read_root_metadata,
)
from zarr_vectors.types.meshes import read_mesh, write_mesh

from zarr_vectors_tools.multiresolution.coarsen import build_pyramid, select_coarsener_key
from zarr_vectors_tools.multiresolution.strategies.meshes import (
    COARSEN_MESH_CLUSTER,
    coarsen_mesh_level,
)


def _icosphere_grid(n_obj: int = 4, per_side: int = 14, spacing: float = 90.0):
    """A few small open surfaces, far enough apart to land in several chunks.

    Each object is a triangulated grid displaced onto its own patch of space,
    so the store has real per-object structure and real cross-chunk faces.
    """
    verts, faces, oids = [], [], []
    off = 0
    rng = np.random.default_rng(0)
    for o in range(n_obj):
        gx, gy = np.meshgrid(np.arange(per_side), np.arange(per_side))
        x = gx.ravel() * spacing + o * 1900.0
        y = gy.ravel() * spacing + (o % 2) * 1500.0
        z = 500.0 + rng.normal(0, 40.0, x.size)
        verts.append(np.stack([x, y, z], axis=1).astype(np.float32))
        idx = np.arange(per_side * per_side).reshape(per_side, per_side)
        a = idx[:-1, :-1].ravel()
        b = idx[:-1, 1:].ravel()
        c = idx[1:, :-1].ravel()
        d = idx[1:, 1:].ravel()
        f = np.concatenate([np.stack([a, b, c], 1), np.stack([b, d, c], 1)])
        faces.append(f + off)
        oids.append(np.full(per_side * per_side, o, dtype=np.int64))
        off += per_side * per_side
    return (np.concatenate(verts), np.concatenate(faces).astype(np.int64),
            np.concatenate(oids))


@pytest.fixture
def mesh_store(tmp_path):
    v, f, o = _icosphere_grid()
    path = tmp_path / "meshes.zv"
    lo = np.floor(v.min(0) - 1).tolist()
    hi = np.ceil(v.max(0) + 1).tolist()
    write_mesh(str(path), v, f, chunk_shape=(1000.0,) * 3,
               bin_shape=(250.0,) * 3, bounds=(lo, hi), encoding="raw",
               object_ids=o)
    return path, v, f, o


def test_m1_mesh_store_dispatches_to_the_mesh_coarsener(mesh_store):
    path, _v, _f, _o = mesh_store
    root_meta = read_root_metadata(open_store(str(path), mode="r"))
    assert select_coarsener_key(root_meta) == "mesh"


def test_m2_m3_faces_survive_and_all_are_considered(mesh_store):
    path, _v, f, _o = mesh_store
    summary = coarsen_mesh_level(str(path), 0, 1, coarsen_factor=2.0,
                                 chunk_scale_factor=2)
    assert summary["method"] == COARSEN_MESH_CLUSTER
    # M3: every source face reached the remap, cross-chunk ones included.
    assert summary["faces_in"] == f.shape[0]

    coarse = read_mesh(str(path), level=1)
    cv, cf = coarse["vertices"], coarse["faces"]
    assert cv.shape[0] < _v_count(path, 0)
    # M2: triangles, in range, none degenerate.
    assert cf.shape[0] > 0
    assert cf.shape[1] == 3
    assert cf.min() >= 0
    assert cf.max() < cv.shape[0]
    assert (cf[:, 0] != cf[:, 1]).all()
    assert (cf[:, 1] != cf[:, 2]).all()
    assert (cf[:, 0] != cf[:, 2]).all()
    # The summary and what a reader sees must agree.
    assert summary["vertex_count"] == cv.shape[0]
    assert summary["faces_out"] == cf.shape[0]


def test_m4_per_object_reads_partition_every_level(mesh_store):
    path, _v, _f, o = mesh_store
    build_pyramid(str(path), factors=[(2.0, 1.0), (2.0, 1.0)],
                  chunk_scale_factors=[2, 2])
    root = open_store(str(path), mode="r")
    n_obj = int(o.max()) + 1
    for level in sorted(list_resolution_levels(root)):
        lg = get_resolution_level(root, level)
        assert object_count(lg) == n_obj
        per_object = sum(
            sum(len(part) for part in read_object_vertices(lg, oid))
            for oid in range(n_obj)
        )
        # Fragments partition the level: no vertex is shared or orphaned.
        assert per_object == read_mesh(str(path), level=level)["vertices"].shape[0]


def test_m5_pyramid_shrinks_and_preserves_object_ids(mesh_store):
    path, _v, _f, o = mesh_store
    build_pyramid(str(path), factors=[(2.0, 1.0)] * 3,
                  chunk_scale_factors=[2, 2, 2])
    root = open_store(str(path), mode="r")
    levels = sorted(list_resolution_levels(root))
    assert levels == [0, 1, 2, 3]
    counts = [read_mesh(str(path), level=lv)["vertices"].shape[0] for lv in levels]
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] < counts[0]
    for lv in levels:
        assert object_count(get_resolution_level(root, lv)) == int(o.max()) + 1


def test_m6_coarse_geometry_stays_inside_the_source_bounds(mesh_store):
    path, v, _f, _o = mesh_store
    coarsen_mesh_level(str(path), 0, 1, coarsen_factor=2.0, chunk_scale_factor=2)
    coarse = read_mesh(str(path), level=1)["vertices"]
    # Centroids of source vertices, so bounded by them on every axis.
    assert np.all(coarse.min(axis=0) >= v.min(axis=0) - 1e-3)
    assert np.all(coarse.max(axis=0) <= v.max(axis=0) + 1e-3)


def _v_count(path, level: int) -> int:
    return read_mesh(str(path), level=level)["vertices"].shape[0]
