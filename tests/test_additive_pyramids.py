"""``build_pyramid(refinement="add")``: each object (or point) stored once.

Invariants under test:

* **P1** A streamline pyramid built by object sparsity alone, made
  additive, reads back -- level by level, through the default union -- as
  exactly the replacement pyramid: the same objects with identical vertices
  and attributes.  Level 0 alone holds only the remainder.  It validates,
  and it does not warn.
* **P2** It warns (``AdditiveRefinementWarning``) and proceeds for a mesh,
  for a level that coarsens geometry, and for a level that drops nothing.
* **P3** A nested point-subset pyramid made additive reads back as the
  replacement pyramid, point for point.
* **P4** The command line takes ``--refinement add`` and says loudly what
  happened to level 0; a mesh gets a warning banner.
* **P5** Nothing rebuilds an additive pyramid, and a merge reads all of an
  additive source.
"""

from __future__ import annotations

import shutil
import sys
import warnings

import numpy as np
import pytest
from zarr_vectors.building import (
    get_resolution_level,
    open_store,
    read_level_metadata,
    read_root_metadata,
    stored_object_ids,
)
from zarr_vectors.types.meshes import write_mesh
from zarr_vectors.types.points import read_points, write_points
from zarr_vectors.types.polylines import read_polylines, write_polylines
from zarr_vectors.validate import validate

from zarr_vectors_tools.multiresolution.additive import (
    AdditiveRefinementWarning,
    supports_additive,
)
from zarr_vectors_tools.multiresolution.coarsen import build_pyramid
from zarr_vectors_tools.multiresolution.strategies.points import (
    build_point_subset_pyramid,
)

pytestmark = pytest.mark.skipif(
    not supports_additive(), reason="zarr-vectors-py without additive levels",
)


def _streamlines(path, n=60, seed=3):
    rng = np.random.default_rng(seed)
    lines = []
    for _ in range(n):
        start = rng.uniform(5, 55, 3)
        steps = rng.normal(0, 1.5, (int(rng.integers(8, 40)), 3))
        lines.append((start + np.cumsum(steps, axis=0)).clip(0.5, 63.5).astype("float32"))
    write_polylines(
        str(path), lines, chunk_shape=(16.0,) * 3, bounds=([0, 0, 0], [64, 64, 64]),
        geometry_type="streamline",
        vertex_attributes={"t": [np.linspace(0, 1, len(x)).astype("float32") for x in lines]},
    )


def _objects(path, level, **kw):
    r = read_polylines(str(path), level=level, **kw)
    t = r["vertex_attributes"]["t"]
    out: dict[int, list] = {}
    pos = 0
    for oid, segments in zip(r["object_ids"], r["polylines"]):
        arr = np.concatenate(segments)
        out.setdefault(int(oid), []).append((arr, t[pos:pos + len(arr)]))
        pos += len(arr)
    return {
        k: (np.concatenate([a for a, _ in v]), np.concatenate([b for _, b in v]))
        for k, v in out.items()
    }


def _identical(a, b):
    return set(a) == set(b) and all(
        np.array_equal(a[k][0], b[k][0]) and np.array_equal(a[k][1], b[k][1]) for k in a
    )


def _additive_warnings(record):
    return [w for w in record if issubclass(w.category, AdditiveRefinementWarning)]


@pytest.fixture
def twin_pyramids(tmp_path):
    """The same sparsity pyramid built twice: replacement and additive."""
    def build(chunk_scale=None):
        replace, add = tmp_path / "replace.zv", tmp_path / "add.zv"
        for p in (replace, add):
            shutil.rmtree(p, ignore_errors=True)
        _streamlines(replace)
        shutil.copytree(replace, add)
        options = dict(factors=[(1.0, 2.0), (1.0, 2.0)], sparsity_seed=7,
                       chunk_scale_factors=chunk_scale)
        build_pyramid(str(replace), **options)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            result = build_pyramid(str(add), refinement="add", **options)
        return replace, add, result, _additive_warnings(record)
    return build


# ------------------------------------------------------------------ P1


@pytest.mark.parametrize("chunk_scale", [None, [2, 2]])
def test_p1_a_sparsity_pyramid_reads_back_identical(twin_pyramids, chunk_scale):
    replace, add, result, warned = twin_pyramids(chunk_scale)
    assert not warned, [str(w.message) for w in warned]
    assert result["refinement"] == "add"
    for level in range(3):
        assert _identical(_objects(add, level), _objects(replace, level)), level
    # Level 0 alone is the remainder: what level 1 does not keep.
    own0 = _objects(add, 0, own_level_only=True)
    assert set(own0) == set(_objects(replace, 0)) - set(_objects(replace, 1))
    for oid, value in own0.items():
        np.testing.assert_array_equal(value[0], _objects(replace, 0)[oid][0])
    result5 = validate(str(add), level=5)
    assert result5.ok, result5.errors
    # Stored once: the additive pyramid is the size of level 0.
    stored = sum(read_level_metadata(open_store(str(add)), lv).vertex_count for lv in range(3))
    assert stored == read_level_metadata(open_store(str(replace)), 0).vertex_count


def test_p1_metadata_and_capabilities(twin_pyramids):
    _replace, add, _result, _ = twin_pyramids()
    root = open_store(str(add))
    meta = read_root_metadata(root)
    assert meta.required_capabilities == ["additive_levels"]
    assert "additive_levels" in meta.format_capabilities
    assert [read_level_metadata(root, lv).refinement for lv in range(3)] == [
        "add", "add", "replace",
    ]
    own = [stored_object_ids(get_resolution_level(root, lv)) for lv in range(3)]
    assert not np.intersect1d(own[0], own[1]).size
    assert not np.intersect1d(own[1], own[2]).size


# ------------------------------------------------------------------ P2


def test_p2_a_coarsened_level_warns_and_proceeds(tmp_path):
    path = tmp_path / "c.zv"
    _streamlines(path)
    with pytest.warns(AdditiveRefinementWarning, match="coarsen geometry"):
        build_pyramid(str(path), factors=[(4.0, 2.0)], refinement="add")
    assert read_level_metadata(open_store(str(path)), 0).refinement == "add"


def test_p2_a_level_that_drops_nothing_warns(tmp_path):
    path = tmp_path / "d.zv"
    _streamlines(path)
    with pytest.warns(AdditiveRefinementWarning, match="will be empty"):
        build_pyramid(str(path), factors=[(1.0, 1.0)], refinement="add")
    assert read_level_metadata(open_store(str(path)), 0).vertex_count == 0
    assert validate(str(path), level=5).ok


def test_p2_a_mesh_warns_and_keeps_its_claims(tmp_path):
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from test_face_groups import _icosphere

    rng = np.random.default_rng(1)
    v, f = _icosphere(2)
    verts = np.concatenate([v * 6 + c for c in rng.uniform(10, 54, (8, 3))]).astype("float32")
    faces = np.concatenate([f + i * len(v) for i in range(8)])
    path = tmp_path / "m.zv"
    write_mesh(str(path), verts, faces, chunk_shape=(32.0,) * 3,
               object_ids=np.repeat(np.arange(8), len(v)), bounds=([0, 0, 0], [64, 64, 64]))
    with pytest.warns(AdditiveRefinementWarning, match="mesh"):
        build_pyramid(str(path), factors=[(2.0, 2.0)], method="mesh", refinement="add")
    meta = read_level_metadata(open_store(str(path)), 0)
    assert meta.refinement == "add" and meta.fragment_link_groups and meta.fragments_tile
    assert validate(str(path), level=5).ok


# ------------------------------------------------------------------ P3


def test_p3_a_point_subset_pyramid_reads_back_identical(tmp_path):
    rng = np.random.default_rng(5)
    points = rng.uniform(0, 64, (4000, 3)).astype("float32")
    points[10] = points[11]                       # one duplicate position
    values = np.arange(4000, dtype="float32")

    def rows(path, level, **kw):
        r = read_points(str(path), level=level, attribute_names=["v"], **kw)
        return sorted(map(tuple, np.column_stack(
            [r["positions"], r["vertex_attributes"]["v"]],
        ).tolist()))

    built = {}
    for refinement in ("replace", "add"):
        path = tmp_path / f"{refinement}.zv"
        write_points(str(path), points, chunk_shape=(32.0,) * 3,
                     bounds=([0, 0, 0], [64, 64, 64]), vertex_attributes={"v": values})
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            build_point_subset_pyramid(str(path), levels=2, divisor=4.0,
                                       refinement=refinement)
        assert not _additive_warnings(record)
        built[refinement] = path
    for level in range(3):
        assert rows(built["add"], level) == rows(built["replace"], level), level
    own0 = rows(built["add"], 0, own_level_only=True)
    assert len(own0) == 4000 - 1000
    assert validate(str(built["add"]), level=5).ok


def test_p3_attributes_left_behind_warn(tmp_path):
    path = tmp_path / "geo.zv"
    write_points(str(path), np.random.default_rng(0).uniform(0, 64, (800, 3)).astype("float32"),
                 chunk_shape=(32.0,) * 3, bounds=([0, 0, 0], [64, 64, 64]),
                 vertex_attributes={"v": np.zeros(800, dtype="float32")})
    with pytest.warns(AdditiveRefinementWarning, match="not carried"):
        build_point_subset_pyramid(str(path), levels=1, divisor=4.0,
                                   attributes=False, refinement="add")


# ------------------------------------------------------------------ P4


def _cli(*argv):
    from zarr_vectors_tools.cli import main

    return main(list(argv))


def test_p4_the_cli_builds_an_additive_pyramid(tmp_path, capsys):
    path = tmp_path / "cli.zv"
    _streamlines(path)
    assert _cli("pyramid", str(path), "--sparsity", "2,2", "--refinement", "add") == 0
    err = capsys.readouterr().err
    assert "ADDITIVE pyramid" in err and "Level 0 was rewritten" in err
    assert "WARNING" not in err
    assert read_level_metadata(open_store(str(path)), 0).refinement == "add"
    with pytest.raises(SystemExit, match="additive levels"):
        _cli("pyramid", str(path), "--sparsity", "2", "--replace")


def test_p4_the_cli_warns_for_a_coarsening_build(tmp_path, capsys):
    path = tmp_path / "cli2.zv"
    _streamlines(path)
    assert _cli("pyramid", str(path), "--coarsen", "4", "--sparsity", "2",
                "--refinement", "add") == 0
    err = capsys.readouterr().err
    assert "WARNING: additive pyramid" in err and "coarsen geometry" in err


# ------------------------------------------------------------------ P5


def test_p5_nothing_rebuilds_an_additive_pyramid(twin_pyramids):
    from zarr_vectors_tools.multiresolution.coarsen import coarsen_level
    from zarr_vectors_tools.multiresolution.refresh import rebuild_pyramid_from_level

    _replace, add, _result, _ = twin_pyramids()
    with pytest.raises(ValueError, match="additive"):
        build_pyramid(str(add), factors=[(1.0, 2.0)])
    with pytest.raises(Exception, match="additive"):
        coarsen_level(str(add), 2, 3, sparsity_factor=2.0)
    with pytest.raises(Exception, match="additive"):
        rebuild_pyramid_from_level(open_store(str(add), mode="r+"), 0)
    with pytest.raises(Exception, match="additive"):
        build_point_subset_pyramid(str(add), levels=1)


def test_p5_a_merge_reads_all_of_an_additive_source(twin_pyramids, tmp_path):
    from zarr_vectors_tools.compose import merge_stores

    replace, add, _result, _ = twin_pyramids()
    target = tmp_path / "merged.zv"
    summary = merge_stores(str(target), [str(add)], create=True, pyramid="drop")
    assert summary["objects_added"] == len(_objects(replace, 0))

    def shapes(path):  # a merge assigns new ids, so compare geometry alone
        return sorted(v[0].tobytes() for v in _objects(path, 0).values())

    assert shapes(target) == shapes(replace)
    with pytest.raises(Exception, match="pyramid='keep'"):
        merge_stores(str(add), [str(target)], pyramid="rebuild")


def test_p5_attaching_to_an_additive_level_writes_its_whole_chain(tmp_path):
    from zarr_vectors_tools.convert.ingest.attach import attach_attributes

    rng = np.random.default_rng(9)
    n = 2000
    path = tmp_path / "att.zv"
    write_points(str(path), rng.uniform(0, 64, (n, 3)).astype("float32"),
                 chunk_shape=(32.0,) * 3, bounds=([0, 0, 0], [64, 64, 64]),
                 vertex_attributes={"zv_join_key": np.arange(n, dtype="int64")})
    build_point_subset_pyramid(str(path), levels=2, divisor=4.0, refinement="add")
    doubled = np.arange(n, dtype="float32") * 2
    summary = attach_attributes(str(path), {"twice": doubled})
    assert summary["levels"] == [0, 1, 2] and summary["vertices_matched"] == n
    for level in range(3):
        r = read_points(str(path), level=level, attribute_names=["zv_join_key", "twice"])
        keys = np.asarray(r["vertex_attributes"]["zv_join_key"])
        np.testing.assert_array_equal(r["vertex_attributes"]["twice"], keys * 2)
