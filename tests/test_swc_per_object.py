"""One SWC per skeleton, named after its segment.

``export_swc`` wrote a whole level as one file.  An EM store holds thousands
of segments, and what a user wants from it is one file per neuron.  Two kinds
of store get there differently: an EM skeleton store keeps a segment id per
object and is read one segment at a time; any other skeleton store is read
once and split by its object manifests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from zarr_vectors.exceptions import ExportError
from zarr_vectors.types.graphs import write_graph

from zarr_vectors_tools.cli import main
from zarr_vectors_tools.convert.export.swc import export_swc

CHUNK = (10.0, 10.0, 10.0)


def _read_swc(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        rows.append({
            "id": int(parts[0]),
            "type": int(parts[1]),
            "xyz": (float(parts[2]), float(parts[3]), float(parts[4])),
            "radius": float(parts[5]),
            "parent": int(parts[6]),
        })
    return rows


@pytest.fixture
def two_neurons(tmp_path: Path) -> tuple[Path, np.ndarray, np.ndarray]:
    """Two trees in one store, the first crossing chunks and re-entering one."""
    positions = np.array([
        [1, 1, 1], [12, 1, 1], [14, 1, 1], [5, 5, 1],     # object 0
        [30, 30, 1], [31, 30, 1], [32, 38, 1],            # object 1
    ], dtype=np.float32)
    edges = np.array([[1, 0], [2, 1], [3, 2], [5, 4], [6, 5]], dtype=np.int64)
    object_ids = np.array([0, 0, 0, 0, 1, 1, 1])
    radius = np.arange(1, 8, dtype=np.float32)
    store = tmp_path / "two.zv"
    write_graph(
        str(store), positions, edges, chunk_shape=CHUNK, kind="skeleton",
        object_ids=object_ids, vertex_attributes={"radius": radius},
    )
    return store, positions, radius


class TestGraphSkeletonStore:

    def test_one_single_root_file_per_object(self, two_neurons, tmp_path: Path) -> None:
        store, positions, radius = two_neurons
        out = tmp_path / "neurons"
        summary = export_swc(str(store), str(out), object_ids=[0, 1])

        assert summary["object_count"] == 2
        assert [Path(f).name for f in summary["files"]] == ["0.swc", "1.swc"]
        expected_radius = {tuple(p): r for p, r in zip(positions.tolist(), radius)}
        for name, n_nodes in (("0.swc", 4), ("1.swc", 3)):
            rows = _read_swc(out / name)
            assert len(rows) == n_nodes
            assert sum(1 for row in rows if row["parent"] == -1) == 1
            for row in rows:
                assert row["radius"] == pytest.approx(expected_radius[row["xyz"]])

    def test_edges_stay_within_their_object(self, two_neurons, tmp_path: Path) -> None:
        store, _positions, _radius = two_neurons
        out = tmp_path / "one"
        export_swc(str(store), str(out), object_ids=[1])
        rows = _read_swc(out / "1.swc")
        by_id = {row["id"]: row for row in rows}
        pairs = {(row["xyz"], by_id[row["parent"]]["xyz"]) for row in rows if row["parent"] != -1}
        assert pairs == {
            ((31.0, 30.0, 1.0), (30.0, 30.0, 1.0)),
            ((32.0, 38.0, 1.0), (31.0, 30.0, 1.0)),
        }

    @pytest.mark.parametrize("kwargs, message", [
        ({"object_ids": [0], "chunks": [(0, 0, 0)]}, "one or the other"),
        ({"object_ids": [5]}, "not at level 0"),
    ])
    def test_refusals(self, two_neurons, tmp_path: Path, kwargs, message) -> None:
        store, _p, _r = two_neurons
        with pytest.raises(ExportError, match=message):
            export_swc(str(store), str(tmp_path / "out"), **kwargs)

    def test_a_file_name_is_not_a_directory(self, two_neurons, tmp_path: Path) -> None:
        store, _p, _r = two_neurons
        with pytest.raises(ExportError, match="directory"):
            export_swc(str(store), str(tmp_path / "x.swc"), object_ids=[0])

    def test_cli(self, two_neurons, tmp_path: Path) -> None:
        store, _p, _r = two_neurons
        out = tmp_path / "cli"
        assert main([
            "convert", str(store), str(out), "--format", "swc",
            "--object-id", "0", "--object-id", "1",
        ]) == 0
        assert sorted(p.name for p in out.iterdir()) == ["0.swc", "1.swc"]


class TestEmSkeletonStore:

    def test_files_are_named_by_segment_id(self, tmp_path: Path) -> None:
        from zarr_vectors.building import get_resolution_level, open_store, read_object_attributes
        from zarr_vectors.types import skeletons as sk

        from zarr_vectors_tools.convert.ingest.precomputed_skeletons import (
            InMemoryFragsReader,
            SkeletonInfo,
            run_ingest,
        )

        info = SkeletonInfo(
            base_url="mem://x", resolution_nm=(32.0, 32.0, 40.0),
            chunk_size_nm=(16384.0, 16384.0, 20480.0),
            vertex_attributes=[{"id": "radius", "data_type": "float32", "num_components": 1}],
        )
        key = "17910-18422_10448-10960_3088-3600.frags"
        origin = np.array([17910 * 32.0, 10448 * 32.0, 3088 * 40.0])
        segments = {}
        for i, segment in enumerate((720575940000000011, 720575940000000029)):
            n = 12
            positions = (origin + 500 + np.column_stack([
                np.arange(n) * 90.0, np.full(n, 300.0 * (i + 1)), np.full(n, 200.0),
            ])).astype(np.float32)
            segments[segment] = {
                "vertices": positions,
                "edges": np.array([[j, j - 1] for j in range(1, n)]),
                "radius": np.linspace(10, 20, n).astype(np.float32),
            }
        bounds = (origin.tolist(), (origin + np.array(info.chunk_size_nm)).tolist())
        store = tmp_path / "em.zv"
        run_ingest(
            InMemoryFragsReader(info, {key: segments}), str(store), [key],
            bounds_nm=bounds, progress=False,
        )

        level0 = get_resolution_level(open_store(str(store)), 0)
        by_object = read_object_attributes(level0, "segment_id").astype(np.uint64).tolist()
        out = tmp_path / "em"
        summary = export_swc(str(store), str(out), object_ids=[0, 1])

        assert sorted(Path(f).name for f in summary["files"]) == sorted(
            f"{int(s)}.swc" for s in by_object
        )
        assert summary["attributes_carried"] == ["radius"]
        for segment in segments:
            rows = _read_swc(out / f"{segment}.swc")
            expected = sk.read_skeleton_by_segment_id(str(store), segment)
            assert len(rows) == len(expected["positions"]) == 12
            assert sum(1 for row in rows if row["parent"] == -1) == 1
            assert sorted(row["radius"] for row in rows) == pytest.approx(
                sorted(np.linspace(10, 20, 12)), abs=1e-3,
            )


def test_a_neuron_crossing_a_chunk_boundary_is_one_tree(tmp_path: Path) -> None:
    """Core reads a segment's pieces in each chunk but not the links between
    chunks; without them a neuron spanning two .frags exports with two roots."""
    from zarr_vectors_tools.convert.ingest.precomputed_skeletons import (
        InMemoryFragsReader,
        SkeletonInfo,
        enumerate_frag_keys,
        run_ingest,
    )

    info = SkeletonInfo(
        base_url="mem://x", resolution_nm=(32.0, 32.0, 40.0),
        chunk_size_nm=(16384.0, 16384.0, 20480.0),
    )
    keys = enumerate_frag_keys(info, (17398, 10448, 3088), (2, 1, 1))
    # igneous duplicates the boundary vertex into both .frags, on the face
    # plus half a voxel.
    boundary = 17910 * 32 + 16
    y, z = 340000.0, 130000.0
    segment = 720575940000000999
    first = np.array([[560000, y, z], [565000, y, z], [570000, y, z], [boundary, y, z]],
                     dtype=np.float32)
    second = np.array([[boundary, y, z], [576000, y, z], [580000, y, z]], dtype=np.float32)
    chain = lambda n: np.array([[i, i - 1] for i in range(1, n)])  # noqa: E731
    reader = InMemoryFragsReader(info, {
        keys[0]: {segment: {"vertices": first, "edges": chain(len(first))}},
        keys[1]: {segment: {"vertices": second, "edges": chain(len(second))}},
    })
    bounds = ([17398 * 32.0, 10448 * 32.0, 3088 * 40.0],
              [18422 * 32.0, 10960 * 32.0, 3600 * 40.0])
    store = tmp_path / "crossing.zv"
    run_ingest(reader, str(store), keys, bounds_nm=bounds, progress=False)

    out = tmp_path / "one"
    summary = export_swc(str(store), str(out), object_ids=[0])
    rows = _read_swc(out / f"{segment}.swc")
    assert summary["root_count"] == 1
    assert sum(1 for row in rows if row["parent"] == -1) == 1
    assert len(rows) == 7
