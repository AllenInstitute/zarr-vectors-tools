"""Point exports stream a batch at a time and write what a whole read would.

The CSV and PLY exporters used to read the entire level into one array
before writing.  They now read and write a batch of chunks (or objects) at a
time.  What has to hold is that the file does not depend on the batch size
-- same rows, same order, same filters -- and that a small batch really does
keep memory down.
"""

from __future__ import annotations

import tracemalloc
from pathlib import Path

import numpy as np
import pytest
from zarr_vectors.types.points import read_points, write_points

from zarr_vectors_tools.convert.export.csv_points import export_csv
from zarr_vectors_tools.convert.export.ply import export_ply


def _store(path: Path, n: int = 3000, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, 100, (n, 3)).astype(np.float32)
    write_points(
        str(path), positions, chunk_shape=(20.0, 20.0, 20.0),
        object_ids=(np.arange(n) % 7).astype(np.int64),
        vertex_attributes={
            "intensity": rng.random(n).astype(np.float32),
            "rgb": rng.random((n, 3)).astype(np.float32),
        },
    )
    return path


@pytest.fixture
def store(tmp_path: Path) -> Path:
    return _store(tmp_path / "points.zv")


@pytest.mark.parametrize("filters", [
    {},
    {"bbox": ([10.0, 10.0, 10.0], [70.0, 60.0, 90.0])},
    {"object_ids": [5, 1, 3]},
    {"chunks": [(0, 0, 0), (1, 2, 3), (4, 4, 4)]},
], ids=["all", "bbox", "objects", "chunks"])
def test_csv_does_not_depend_on_the_batch_size(store: Path, tmp_path: Path, filters) -> None:
    whole = tmp_path / "whole.csv"
    batched = tmp_path / "batched.csv"
    a = export_csv(str(store), str(whole), attribute_names=["intensity", "rgb"],
                   vertex_budget=10**9, **filters)
    b = export_csv(str(store), str(batched), attribute_names=["intensity", "rgb"],
                   vertex_budget=50, **filters)
    assert a == b
    assert whole.read_bytes() == batched.read_bytes()


def test_csv_rows_are_the_whole_level_read(store: Path, tmp_path: Path) -> None:
    out = tmp_path / "p.csv"
    export_csv(str(store), str(out), attribute_names=["rgb"], vertex_budget=100)
    lines = out.read_text().splitlines()
    assert lines[0] == "dim0,dim1,dim2,rgb_0,rgb_1,rgb_2"
    rows = np.array([[float(v) for v in line.split(",")] for line in lines[1:]])
    expected = read_points(str(store), attribute_names=["rgb"])
    np.testing.assert_allclose(rows[:, :3], expected["positions"], atol=1e-5)
    np.testing.assert_allclose(rows[:, 3:], expected["vertex_attributes"]["rgb"], atol=1e-5)


@pytest.mark.parametrize("binary", [True, False], ids=["binary", "ascii"])
def test_ply_matches_the_whole_level_read(store: Path, tmp_path: Path, binary: bool) -> None:
    plyfile = pytest.importorskip("plyfile")
    out = tmp_path / "p.ply"
    summary = export_ply(
        str(store), str(out), attribute_names=["intensity", "rgb"], binary=binary,
        vertex_budget=80,
    )
    data = plyfile.PlyData.read(str(out))["vertex"].data
    expected = read_points(str(store), attribute_names=["intensity", "rgb"])
    assert summary["vertex_count"] == len(data) == len(expected["positions"])
    np.testing.assert_array_equal(
        np.column_stack([data["x"], data["y"], data["z"]]), expected["positions"],
    )
    np.testing.assert_array_equal(data["intensity"], expected["vertex_attributes"]["intensity"])
    np.testing.assert_array_equal(
        np.column_stack([data[f"rgb_{c}"] for c in range(3)]),
        expected["vertex_attributes"]["rgb"],
    )


def test_an_absent_attribute_is_refused_before_writing(store: Path, tmp_path: Path) -> None:
    from zarr_vectors.exceptions import ExportError

    out = tmp_path / "x.csv"
    with pytest.raises(ExportError, match="not present"):
        export_csv(str(store), str(out), attribute_names=["nope"])
    assert not out.exists()


@pytest.mark.slow
def test_a_small_batch_keeps_memory_down(tmp_path: Path) -> None:
    store = _store(tmp_path / "big.zv", n=300_000, seed=1)

    def peak(budget: int) -> int:
        tracemalloc.start()
        export_csv(str(store), str(tmp_path / f"b{budget}.csv"),
                   attribute_names=["rgb"], vertex_budget=budget)
        _current, top = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return top

    whole, batched = peak(10**9), peak(20_000)
    assert batched < whole / 3, (batched, whole)
