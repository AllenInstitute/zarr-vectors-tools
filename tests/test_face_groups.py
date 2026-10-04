"""Per-object face groups (``multiresolution/face_groups.py``).

A viewer drawing one object reads its faces with a byte-range read when every
chunk of a level keeps one intra-chunk link group per vertex fragment and
zarr-vectors-py has stamped the level ``fragment_link_groups``.  The
invariants under test:

* **F1** Both mesh coarseners write one group per fragment, every group's
  corners inside its own fragment, and the level ends up stamped.
* **F2** Grouping is invisible to readers that concatenate groups:
  ``read_mesh`` returns the same mesh before and after ``index_mesh_faces``.
* **F3** ``index_mesh_faces`` stamps a level whose groups do not yet follow
  its fragments, a second run has nothing to do, and a dry run writes
  nothing.
* **F4** A face whose corners lie in two fragments has no per-fragment
  grouping, so such a level is never stamped.
"""

from __future__ import annotations

import numpy as np
import pytest
from zarr_vectors.building import (
    get_resolution_level,
    open_store,
    read_chunk_links,
    read_level_metadata,
    read_root_metadata,
    read_vertex_fragment_index,
)
from zarr_vectors.types.meshes import read_mesh, write_mesh

from zarr_vectors_tools.multiresolution.coarsen import build_pyramid
from zarr_vectors_tools.multiresolution.face_groups import (
    HAS_FRAGMENT_LINK_GROUPS,
    index_mesh_faces,
    split_faces_by_fragment,
)

needs_link_groups = pytest.mark.skipif(
    not HAS_FRAGMENT_LINK_GROUPS,
    reason="zarr-vectors-py without fragment_link_groups",
)


def _link_groups():
    return pytest.importorskip("zarr_vectors.core.link_groups")


def _icosphere(n: int):
    t = (1 + 5 ** 0.5) / 2
    v = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
         [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9],
         [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2],
         [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10],
         [8, 6, 7], [9, 8, 1]]
    verts = [np.array(x, float) / np.linalg.norm(x) for x in v]
    for _ in range(n):
        cache: dict = {}

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                cache[key] = len(verts)
                verts.append(m / np.linalg.norm(m))
            return cache[key]

        f = [tri for a, b, c in f for tri in (
            [a, mid(a, b), mid(c, a)], [b, mid(b, c), mid(a, b)],
            [c, mid(c, a), mid(b, c)], [mid(a, b), mid(b, c), mid(c, a)])]
    return np.array(verts), np.array(f)


@pytest.fixture
def spheres(tmp_path):
    """Two touching spheres over several chunks: shared chunks, cross faces."""
    v, f = _icosphere(3)
    centers, radii = [(20.0, 20.0, 20.0), (40.0, 26.0, 22.0)], [12.0, 10.0]
    verts = np.concatenate([v * r + c for c, r in zip(centers, radii)]).astype("float32")
    faces = np.concatenate([f, f + len(v)])
    oids = np.repeat(np.arange(2, dtype=np.int64), len(v))
    path = tmp_path / "spheres.zv"
    write_mesh(str(path), verts, faces, chunk_shape=(16.0,) * 3, object_ids=oids,
               bounds=([0, 0, 0], [64, 48, 40]), encoding="raw")
    return path


def _stamped(path, level):
    return read_level_metadata(open_store(str(path), mode="r"), level).fragment_link_groups


def _assert_per_fragment(path, level):
    link_groups = _link_groups()
    root = open_store(str(path), mode="r")
    assert _stamped(path, level)
    caps = read_root_metadata(root).format_capabilities
    assert link_groups.CAP_FRAGMENT_LINK_GROUPS in caps
    lg = get_resolution_level(root, level)
    name, width = link_groups.intra_links_name(lg)
    checked = 0
    for key in lg.read_array_meta(name)["nonempty_chunks"]:
        cc = tuple(int(x) for x in key.split("."))
        groups = read_chunk_links(lg, cc, delta=0, link_width=width)
        fragments = read_vertex_fragment_index(lg, cc)
        assert len(groups) == fragments.num_fragments, key
        owner = link_groups.fragment_of_rows(fragments)
        for k, g in enumerate(groups):
            g = np.asarray(g).reshape(-1, width)
            assert (owner[g] == k).all(), (key, k)
        checked += 1
    assert checked > 0


def _flatten_to_single_groups(path, level=0):
    """Re-cut every chunk to one group, as writers before this did."""
    link_groups = _link_groups()
    lg = get_resolution_level(open_store(str(path), mode="r+"), level)
    name, width = link_groups.intra_links_name(lg)
    for key in lg.read_array_meta(name)["nonempty_chunks"]:
        cc = tuple(int(x) for x in key.split("."))
        groups = read_chunk_links(lg, cc, delta=0, link_width=width)
        link_groups.write_link_groups(lg, cc, [sum(len(g) for g in groups)])


def _face_set(mesh):
    v = np.round(np.asarray(mesh["vertices"]), 4)
    return {tuple(sorted(map(tuple, v[face]))) for face in np.asarray(mesh["faces"])}


@needs_link_groups
@pytest.mark.parametrize("method,factor", [("mesh_decimate", 4.0), ("mesh", 2.0)])
def test_f1_pyramid_levels_group_faces_per_fragment(spheres, method, factor):
    build_pyramid(str(spheres), factors=[(factor, 1.0)], chunk_scale_factors=[2],
                  method=method)
    _assert_per_fragment(spheres, 0)
    _assert_per_fragment(spheres, 1)


@needs_link_groups
def test_f2_f3_indexing_changes_nothing_a_reader_sees(spheres):
    _flatten_to_single_groups(spheres)
    assert not _stamped(spheres, 0)
    before = read_mesh(str(spheres), level=0)
    first = index_mesh_faces(str(spheres), levels=[0])
    assert first[0]["stamped"] is True
    assert first[0]["regrouped"] > 0 and first[0]["reordered"] == 0
    _assert_per_fragment(spheres, 0)
    after = read_mesh(str(spheres), level=0)
    np.testing.assert_array_equal(before["vertices"], after["vertices"])
    np.testing.assert_array_equal(before["faces"], after["faces"])
    assert _face_set(after) == _face_set(before)
    again = index_mesh_faces(str(spheres), levels=[0])
    assert again[0]["skipped"] == "already stamped"
    rechecked = index_mesh_faces(str(spheres), levels=[0], verify=True)
    assert rechecked[0]["stamped"] and rechecked[0]["regrouped"] == 0


@needs_link_groups
def test_f3_dry_run_writes_nothing(spheres):
    _flatten_to_single_groups(spheres)
    report = index_mesh_faces(str(spheres), levels=[0], dry_run=True)
    assert report[0]["would_stamp"] and report[0]["regrouped"] > 0
    assert not _stamped(spheres, 0)


def test_f4_a_face_across_fragments_has_no_grouping():
    owner = np.array([0, 0, 0, 1, 1, 1])
    inside = np.array([[3, 4, 5], [0, 1, 2], [2, 1, 0]])
    groups = split_faces_by_fragment(inside, owner, 2)
    assert [g.tolist() for g in groups] == [[[0, 1, 2], [2, 1, 0]], [[3, 4, 5]]]
    assert split_faces_by_fragment(np.array([[0, 1, 3]]), owner, 2) is None
    assert split_faces_by_fragment(np.array([[0, 1, 9]]), owner, 2) is None
    assert split_faces_by_fragment(np.array([[0, 1, 2]]), np.array([0, -1, 0]), 1) is None


def test_f4_without_support_nothing_is_stamped(monkeypatch):
    from zarr_vectors_tools.multiresolution import face_groups

    monkeypatch.setattr(face_groups, "_stamp_fragment_link_groups", None)
    monkeypatch.setattr(face_groups, "_index_fragment_link_groups", None)
    assert face_groups.stamp_face_groups(object(), object()) is False
    with pytest.raises(RuntimeError, match="fragment_link_groups"):
        face_groups.index_mesh_faces("unused.zv")
