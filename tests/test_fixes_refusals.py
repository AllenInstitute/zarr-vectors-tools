"""Requests refused before anything is written or removed, and markers that
must land.

* **R1** Additive pyramids and face indexing refuse up front, naming the
  zarr-vectors-py support they lack; ``refuse_additive`` checks the store
  even without the core's own check.
* **R2** ``zvtools pyramid --replace`` keeps the old levels when the
  strategy, the store's data or the parallel backend would make the build
  fail; ``zvtools convert`` refuses before the ingest.
* **R3** ``merge_stores(pyramid="keep")`` marks every coarse level stale.
* **R4** Store detection reads Zarr v3 ``zarr.json`` and its
  ``zarr_vectors`` attributes.
* **R5** ``--group-id`` on a store without groups says so.
* **R6** OBJ export takes ``--object-id``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from zarr_vectors.building import (
    get_resolution_level,
    list_resolution_levels,
    open_store,
)
from zarr_vectors.types.meshes import write_mesh
from zarr_vectors.types.polylines import write_polylines

from zarr_vectors_tools.cli import main
from zarr_vectors_tools.cli._args import looks_like_store
from zarr_vectors_tools.cli.convert import _maybe_overwrite
from zarr_vectors_tools.multiresolution.additive import (
    refuse_additive,
    supports_additive,
)
from zarr_vectors_tools.multiresolution.coarsen import (
    build_pyramid,
    check_sparsity_strategy,
)
from zarr_vectors_tools.multiresolution.face_groups import HAS_FRAGMENT_LINK_GROUPS


def _streamlines(path: Path, n: int = 30, groups=None) -> Path:
    rng = np.random.default_rng(3)
    lines = []
    for _ in range(n):
        start = rng.uniform(5, 55, 3)
        steps = rng.normal(0, 1.5, (int(rng.integers(8, 30)), 3))
        lines.append((start + np.cumsum(steps, axis=0)).clip(0.5, 63.5).astype("float32"))
    write_polylines(
        str(path), lines, chunk_shape=(16.0,) * 3, bounds=([0, 0, 0], [64, 64, 64]),
        geometry_type="streamline", groups=groups,
    )
    return path


def _pyramid(tmp_path: Path) -> Path:
    store = _streamlines(tmp_path / "s.zv")
    build_pyramid(str(store), factors=[(1.0, 2.0)])
    return store


def _levels(store: Path) -> list[int]:
    return list(list_resolution_levels(open_store(str(store), mode="r")))


def _csv(path: Path) -> Path:
    path.write_text("x,y,z\n1,2,3\n4,5,6\n7,8,9\n")
    return path


# ---- R1 ---------------------------------------------------------------------

@pytest.mark.skipif(supports_additive(), reason="core supports additive levels")
def test_r1_additive_refused_naming_the_missing_core(tmp_path):
    store = _pyramid(tmp_path)
    with pytest.raises(
        SystemExit, match=r"zarr_vectors\.multiresolution\.additive\.make_levels_additive",
    ):
        main(["pyramid", str(store), "--sparsity", "2", "--replace",
              "--refinement", "add"])
    assert _levels(store) == [0, 1]

    out = tmp_path / "p.zv"
    with pytest.raises(SystemExit, match="level_chain"):
        main(["convert", str(_csv(tmp_path / "p.csv")), str(out),
              "--chunk-shape", "16,16,16", "--sparsity", "2",
              "--refinement", "add"])
    assert not out.exists()


def test_r1_refinement_add_needs_pyramid_levels(tmp_path):
    out = tmp_path / "p.zv"
    with pytest.raises(SystemExit, match="--coarsen and --sparsity"):
        main(["convert", str(_csv(tmp_path / "p.csv")), str(out),
              "--chunk-shape", "16,16,16", "--refinement", "add"])
    assert not out.exists()


def test_r1_refuse_additive_checks_required_capabilities(tmp_path):
    store = _pyramid(tmp_path)
    refuse_additive(str(store), "rebuilding")  # an ordinary store passes
    meta = json.loads((store / "zarr.json").read_text())
    meta["attributes"]["zarr_vectors"]["required_capabilities"] = ["additive_levels"]
    (store / "zarr.json").write_text(json.dumps(meta))
    with pytest.raises(Exception, match="additive"):
        refuse_additive(str(store), "rebuilding")
    with pytest.raises(SystemExit):
        main(["pyramid", str(store), "--sparsity", "2", "--replace"])
    assert _levels(store) == [0, 1]


@pytest.mark.skipif(HAS_FRAGMENT_LINK_GROUPS, reason="core has fragment_link_groups")
def test_r1_index_faces_refused_up_front(tmp_path):
    store = _streamlines(tmp_path / "s.zv")
    with pytest.raises(SystemExit, match="index_fragment_link_groups") as exc:
        main(["index-faces", str(store)])
    assert "RuntimeError" not in str(exc.value)


# ---- R2 ---------------------------------------------------------------------

@pytest.mark.parametrize("extra", [
    ["--sparsity-strategy", "spatial_coverage"],
    ["--sparsity-strategy", "length"],          # no object_attributes/length
    ["--sparsity-strategy", "group"],           # no groups
    ["--method", "per_fragment", "--sparsity-strategy", "length"],
])
def test_r2_replace_keeps_old_levels_on_a_refused_strategy(tmp_path, extra, capsys):
    store = _pyramid(tmp_path)
    with pytest.raises(SystemExit):
        main(["pyramid", str(store), "--sparsity", "2", "--replace", *extra])
    assert "removed existing" not in capsys.readouterr().out
    assert _levels(store) == [0, 1]


def test_r2_mesh_refuses_length():
    with pytest.raises(ValueError, match="'mesh' coarsener"):
        check_sparsity_strategy("mesh", "length", [(2.0, 2.0)])
    check_sparsity_strategy("mesh", "length", [(2.0, 1.0)])  # nothing dropped


def test_r2_replace_keeps_old_levels_without_dask(tmp_path):
    try:
        import dask.distributed  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("dask is installed")
    store = _pyramid(tmp_path)
    with pytest.raises(SystemExit, match="dask"):
        main(["pyramid", str(store), "--sparsity", "2", "--replace",
              "--workers", "2", "--workers-backend", "dask"])
    assert _levels(store) == [0, 1]


def test_r2_convert_refuses_strategy_before_ingest(tmp_path):
    out = tmp_path / "m.zv"
    obj = tmp_path / "m.obj"
    obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    with pytest.raises(SystemExit, match="'mesh' coarsener"):
        main(["convert", str(obj), str(out), "--chunk-shape", "16,16,16",
              "--sparsity", "2", "--sparsity-strategy", "length"])
    assert not out.exists()


# ---- R3 ---------------------------------------------------------------------

def test_r3_merge_keep_marks_levels_stale(tmp_path):
    from zarr_vectors_tools.compose import Geometry, GeometrySource, merge_stores
    from zarr_vectors_tools.compose._carry import is_compose_stale

    store = _pyramid(tmp_path)
    source = GeometrySource(
        Geometry(kind="streamline",
                 parts=[np.array([[10, 10, 10], [12, 12, 12]], np.float32)]),
        label="extra",
    )
    summary = merge_stores(str(store), [source], pyramid="keep")
    assert summary.get("stale_levels") == [1], summary
    root = open_store(str(store), mode="r")
    assert is_compose_stale(get_resolution_level(root, 1))
    assert not is_compose_stale(get_resolution_level(root, 0))


# ---- R4 ---------------------------------------------------------------------

def test_r4_store_detection_reads_zarr_json(tmp_path):
    store = _streamlines(tmp_path / "s.zv")
    assert looks_like_store(store)

    v2 = tmp_path / "v2.zarr"
    v2.mkdir()
    (v2 / ".zattrs").write_text("{}")
    assert not looks_like_store(v2)

    plain = tmp_path / "plain.zarr"
    plain.mkdir()
    (plain / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}})
    )
    assert not looks_like_store(plain)
    with pytest.raises(SystemExit, match="not a zarr-vectors store"):
        _maybe_overwrite(plain, overwrite=True)
    assert plain.exists()
    with pytest.raises(SystemExit, match="not a zarr-vectors store"):
        _maybe_overwrite(v2, overwrite=True)
    assert v2.exists()


# ---- R5 ---------------------------------------------------------------------

def test_r5_group_id_on_a_store_without_groups(tmp_path, capsys):
    store = _streamlines(tmp_path / "s.zv")
    assert main(["convert", str(store), str(tmp_path / "x.trk"),
                 "--group-id", "0"]) == 1
    assert "has no groups" in capsys.readouterr().err

    grouped = _streamlines(tmp_path / "g.zv", n=4, groups={0: [0, 1], 1: [2, 3]})
    assert main(["convert", str(grouped), str(tmp_path / "y.trk"),
                 "--group-id", "1"]) == 0


# ---- R6 ---------------------------------------------------------------------

def test_r6_obj_export_takes_object_id(tmp_path):
    store = tmp_path / "m.zv"
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                  [10, 10, 10], [11, 10, 10], [10, 11, 10]], np.float32)
    write_mesh(str(store), v, np.array([[0, 1, 2], [3, 4, 5]]),
               chunk_shape=(32.0,) * 3, bounds=([0, 0, 0], [32, 32, 32]),
               object_ids=np.array([0, 0, 0, 1, 1, 1]))
    out = tmp_path / "one.obj"
    assert main(["convert", str(store), str(out), "--object-id", "1"]) == 0
    verts = [line for line in out.read_text().splitlines() if line.startswith("v ")]
    assert len(verts) == 3
    assert verts[0].startswith("v 10.")
