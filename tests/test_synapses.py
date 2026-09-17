"""Synapse tables join an EM skeleton store by segment id.

A small skeleton store holds two neurons.  The synapse table names them (and
one segment the store does not hold) in voxel coordinates, the way CAVE
exports do.  What has to hold: each synapse lands on the object of its
owning segment, in nanometres; the other side's segment and the synapse id
come along; unmatched synapses follow the chosen policy; and the skeleton
store gains per-neuron counts that a rerun replaces rather than adds to.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from zarr_vectors.building import get_resolution_level, open_store, read_object_attributes
from zarr_vectors.exceptions import IngestError
from zarr_vectors.types.points import read_points

from zarr_vectors_tools.convert.ingest.synapses import ingest_synapses

A, B, STRANGER = 720575940000000011, 720575940000000029, 720575940000009999
RESOLUTION = (4.0, 4.0, 40.0)


@pytest.fixture
def skeletons(tmp_path: Path) -> Path:
    from zarr_vectors_tools.convert.ingest.precomputed_skeletons import (
        InMemoryFragsReader,
        SkeletonInfo,
        run_ingest,
    )

    info = SkeletonInfo(
        base_url="mem://x", resolution_nm=(32.0, 32.0, 40.0),
        chunk_size_nm=(16384.0, 16384.0, 20480.0),
    )
    key = "17910-18422_10448-10960_3088-3600.frags"
    origin = np.array([17910 * 32.0, 10448 * 32.0, 3088 * 40.0])
    segments = {}
    for i, segment in enumerate((A, B)):
        vertices = (origin + 500 + np.column_stack([
            np.arange(6) * 90.0, np.full(6, 300.0 * (i + 1)), np.full(6, 200.0),
        ])).astype(np.float32)
        edges = np.array([[j, j - 1] for j in range(1, 6)])
        segments[segment] = {"vertices": vertices, "edges": edges}
    bounds = (origin.tolist(), (origin + np.array(info.chunk_size_nm)).tolist())
    store = tmp_path / "skeletons.zv"
    run_ingest(InMemoryFragsReader(info, {key: segments}), str(store), [key],
               bounds_nm=bounds, progress=False)
    return store


def _table(path: Path, *, split: bool) -> tuple[Path, pd.DataFrame]:
    rows = pd.DataFrame({
        "id": [101, 102, 103, 104, 105],
        "pre_pt_root_id": [A, A, B, STRANGER, B],
        "post_pt_root_id": [B, B, A, A, STRANGER],
        "size": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    voxels = np.array([[100, 200, 10], [110, 210, 11], [120, 220, 12],
                       [130, 230, 13], [140, 240, 14]])
    if split:
        for axis, column in zip("xyz", voxels.T):
            rows[f"ctr_pt_position_{axis}"] = column
    else:
        rows["ctr_pt_position"] = [f"[{x} {y} {z}]" for x, y, z in voxels]
    rows.to_csv(path, index=False)
    return path, rows.assign(_voxels=list(voxels))


def _points_by_key(store: Path, names: list[str], *, objects: int) -> dict[int, tuple]:
    """``{synapse id: (object id, position, attributes)}``, reading object by object."""
    out = {}
    for oid in range(objects):
        result = read_points(str(store), object_ids=[oid], attribute_names=["zv_join_key", *names])
        attrs = result["vertex_attributes"]
        for i, key in enumerate(attrs.get("zv_join_key", [])):
            out[int(key)] = (oid, result["positions"][i], {n: attrs[n][i] for n in names})
    return out


def _oid(store: Path, segment: int) -> int:
    level0 = get_resolution_level(open_store(str(store)), 0)
    segments = read_object_attributes(level0, "segment_id").astype(np.int64)
    return int(np.searchsorted(segments, segment))


@pytest.mark.parametrize("split", [True, False], ids=["split-columns", "bracket-strings"])
def test_synapses_land_on_their_post_synaptic_neuron(
    skeletons: Path, tmp_path: Path, split: bool,
) -> None:
    table, rows = _table(tmp_path / "synapses.csv", split=split)
    out = tmp_path / "synapses.zv"
    summary = ingest_synapses(
        table, skeletons, out, resolution=RESOLUTION, columns=["size"], unmatched="keep",
    )
    assert summary["rows_read"] == 5
    assert summary["unmatched"] == 1
    assert summary["unassigned_object"] == 2

    points = _points_by_key(out, ["pre_segment_id", "size"], objects=3)
    expected_owner = {101: _oid(skeletons, B), 102: _oid(skeletons, B), 103: _oid(skeletons, A),
                      104: _oid(skeletons, A), 105: 2}
    assert set(points) == set(expected_owner)
    for _, row in rows.iterrows():
        owner, position, attrs = points[int(row["id"])]
        assert owner == expected_owner[int(row["id"])]
        np.testing.assert_allclose(position, np.asarray(row["_voxels"]) * RESOLUTION)
        assert int(attrs["pre_segment_id"]) == row["pre_pt_root_id"]
        assert attrs["size"] == pytest.approx(row["size"])


def test_counts_are_written_and_replaced_not_added(skeletons: Path, tmp_path: Path) -> None:
    table, _rows = _table(tmp_path / "synapses.csv", split=True)
    for attempt in range(2):
        ingest_synapses(table, skeletons, tmp_path / f"s{attempt}.zv", resolution=RESOLUTION)
    level0 = get_resolution_level(open_store(str(skeletons)), 0)
    pre = read_object_attributes(level0, "synapse_pre_count")
    post = read_object_attributes(level0, "synapse_post_count")
    a, b = _oid(skeletons, A), _oid(skeletons, B)
    assert (int(pre[a]), int(pre[b])) == (2, 2)
    assert (int(post[a]), int(post[b])) == (2, 2)


def test_pre_side_and_unmatched_policies(skeletons: Path, tmp_path: Path) -> None:
    table, _rows = _table(tmp_path / "synapses.csv", split=True)
    dropped = ingest_synapses(
        table, skeletons, tmp_path / "pre.zv", side="pre", resolution=RESOLUTION,
        unmatched="drop", write_counts=False,
    )
    assert dropped["synapses_written"] == 4 and dropped["dropped"] == 1
    points = _points_by_key(tmp_path / "pre.zv", ["post_segment_id"], objects=3)
    assert {owner for owner, _p, _a in points.values()} == {_oid(skeletons, A), _oid(skeletons, B)}

    with pytest.raises(IngestError, match="does not hold"):
        ingest_synapses(table, skeletons, tmp_path / "err.zv", resolution=RESOLUTION,
                        unmatched="error", write_counts=False)


@pytest.mark.parametrize("kwargs, message", [
    ({"side": "both"}, "side must be"),
    ({"unmatched": "ignore"}, "unmatched must be"),
    ({"resolution": (4, 4)}, "three values"),
    ({"position": "centre"}, "missing column"),
])
def test_refusals(skeletons: Path, tmp_path: Path, kwargs, message) -> None:
    table, _rows = _table(tmp_path / "synapses.csv", split=True)
    with pytest.raises(IngestError, match=message):
        ingest_synapses(table, skeletons, tmp_path / "x.zv", **kwargs)


def test_a_store_without_segment_ids_is_refused(tmp_path: Path) -> None:
    from zarr_vectors.types.points import write_points

    store = tmp_path / "points.zv"
    write_points(str(store), np.ones((3, 3), np.float32), chunk_shape=(10.0, 10.0, 10.0))
    table, _rows = _table(tmp_path / "synapses.csv", split=True)
    with pytest.raises(IngestError, match="segment_id"):
        ingest_synapses(table, store, tmp_path / "x.zv")


def test_a_parquet_table(skeletons: Path, tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    csv_path, _rows = _table(tmp_path / "synapses.csv", split=True)
    parquet = tmp_path / "synapses.parquet"
    pd.read_csv(csv_path).to_parquet(parquet)
    summary = ingest_synapses(parquet, skeletons, tmp_path / "pq.zv", resolution=RESOLUTION,
                              write_counts=False)
    assert summary["synapses_written"] == 5


def test_cli(skeletons: Path, tmp_path: Path) -> None:
    from zarr_vectors_tools.cli import main

    table, _rows = _table(tmp_path / "synapses.csv", split=False)
    out = tmp_path / "cli.zv"
    assert main([
        "synapses", str(table), str(skeletons), str(out),
        "--resolution", "4,4,40", "--column", "size", "--unmatched", "drop",
    ]) == 0
    assert (out / "zarr.json").exists()
