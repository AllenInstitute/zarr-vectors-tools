"""Integration tests: end-to-end ingest → store → export pipelines.

Each test exercises a full pipeline that involves ingest and/or export
workflows. Pure core-API pipelines (lines, parametric, streamlines
without TRK/TCK/TRX) live in the core ``zarr-vectors`` test suite.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class TestPointCloudPipeline:

    def test_full_pipeline(self, tmp_path: Path) -> None:
        from zarr_vectors.types.points import write_points, read_points
        from zarr_vectors.validate import validate
        from zarr_vectors.multiresolution.coarsen import build_pyramid
        from zarr_vectors.core.store import list_resolution_levels, open_store
        from zarr_vectors_tools.export.csv_points import export_csv

        rng = np.random.default_rng(42)
        store = str(tmp_path / "points.zarrvectors")

        positions = rng.uniform(0, 1000, size=(5000, 3)).astype(np.float32)
        intensity = rng.uniform(0, 1, size=5000).astype(np.float32)
        summary = write_points(
            store, positions,
            chunk_shape=(200.0, 200.0, 200.0),
            vertex_attributes={"intensity": intensity},
        )
        assert summary["vertex_count"] == 5000
        assert summary["chunk_count"] > 1

        vr = validate(store, level=4)
        assert vr.ok, vr.summary()

        result = read_points(store)
        assert result["vertex_count"] == 5000

        result_bbox = read_points(
            store,
            bbox=(np.array([0, 0, 0]), np.array([200, 200, 200])),
        )
        assert 0 < result_bbox["vertex_count"] < 5000

        pyr = build_pyramid(store, factors=[(2.0, 1.0)])
        assert pyr["levels_created"] >= 1

        vr5 = validate(store, level=5)
        assert vr5.ok, vr5.summary()

        levels = list_resolution_levels(open_store(store))
        assert len(levels) >= 2
        coarse = read_points(store, level=1)
        assert 0 < coarse["vertex_count"] < 5000

        csv_out = tmp_path / "exported.csv"
        export_csv(store, csv_out)
        assert csv_out.exists()
        lines = csv_out.read_text().strip().split("\n")
        assert len(lines) == 5001  # header + 5000 rows


class TestCSVPipeline:

    def test_csv_round_trip(self, tmp_path: Path) -> None:
        from zarr_vectors.types.points import read_points
        from zarr_vectors_tools.ingest.csv_points import ingest_csv
        from zarr_vectors_tools.export.csv_points import export_csv

        rng = np.random.default_rng(99)
        positions = rng.uniform(0, 100, size=(200, 3))
        temperature = rng.uniform(20, 40, size=200)
        data = np.column_stack([positions, temperature])

        csv_in = tmp_path / "sensor.csv"
        np.savetxt(csv_in, data, delimiter=",",
                   header="x,y,z,temperature", comments="")

        store = str(tmp_path / "sensor.zarrvectors")
        ingest_csv(csv_in, store, (50.0, 50.0, 50.0),
                   position_columns=["x", "y", "z"],
                   attribute_columns=["temperature"])

        result = read_points(store)
        assert result["vertex_count"] == 200

        csv_out = tmp_path / "sensor_out.csv"
        export_csv(store, csv_out)
        exported = np.loadtxt(csv_out, delimiter=",", skiprows=1)
        assert exported.shape[0] == 200


class TestSkeletonPipeline:

    def test_swc_pipeline(self, tmp_path: Path) -> None:
        from zarr_vectors.types.graphs import read_graph
        from zarr_vectors.validate import validate
        from zarr_vectors.multiresolution.strategies.graphs import prune_skeleton
        from zarr_vectors_tools.ingest.swc import ingest_swc
        from zarr_vectors_tools.export.swc import export_swc

        swc_in = tmp_path / "neuron.swc"
        rng = np.random.default_rng(42)
        lines = ["# synthetic neuron"]
        n_nodes = 100
        positions = [[0.0, 0.0, 0.0]]
        parents = [-1]
        for i in range(1, n_nodes):
            p = max(0, i - rng.integers(1, min(i + 1, 4)))
            px, py, pz = positions[p]
            dx, dy, dz = rng.normal(0, 5, size=3)
            positions.append([px + dx, py + dy, pz + dz])
            parents.append(p)

        for i in range(n_nodes):
            comp = 1 if i == 0 else (2 if positions[i][0] < 0 else 3)
            r = 5.0 if i == 0 else rng.uniform(0.5, 3.0)
            x, y, z = positions[i]
            lines.append(f"{i + 1} {comp} {x:.4f} {y:.4f} {z:.4f} {r:.4f} {parents[i] + 1 if parents[i] >= 0 else -1}")

        swc_in.write_text("\n".join(lines))

        store = str(tmp_path / "neuron.zarrvectors")
        summary = ingest_swc(swc_in, store, (100.0, 100.0, 100.0))
        assert summary["node_count"] == n_nodes

        vr = validate(store, level=4)
        assert vr.ok, vr.summary()

        result = read_graph(store)
        assert result["node_count"] == n_nodes

        pruned = prune_skeleton(
            result["positions"], result["edges"],
            min_branch_length=8.0,
        )
        assert pruned["node_count"] <= n_nodes

        swc_out = tmp_path / "neuron_out.swc"
        export_swc(store, swc_out)
        assert swc_out.exists()
        data_lines = [l for l in swc_out.read_text().strip().split("\n")
                      if not l.startswith("#")]
        assert len(data_lines) == n_nodes


class TestMeshPipeline:

    def test_obj_pipeline(self, tmp_path: Path) -> None:
        from zarr_vectors.types.meshes import read_mesh
        from zarr_vectors.validate import validate
        from zarr_vectors.multiresolution.strategies.meshes import coarsen_mesh_cluster
        from zarr_vectors_tools.ingest.obj import ingest_obj
        from zarr_vectors_tools.export.obj import export_obj

        obj_in = tmp_path / "grid.obj"
        lines = ["# 10x10 grid"]
        for iy in range(10):
            for ix in range(10):
                lines.append(f"v {ix} {iy} 0")
        for iy in range(9):
            for ix in range(9):
                v0 = iy * 10 + ix + 1
                lines.append(f"f {v0} {v0 + 1} {v0 + 11}")
                lines.append(f"f {v0} {v0 + 11} {v0 + 10}")
        obj_in.write_text("\n".join(lines))

        store = str(tmp_path / "grid.zarrvectors")
        summary = ingest_obj(obj_in, store, (5.0, 5.0, 5.0))
        assert summary["vertex_count"] == 100
        assert summary["face_count"] == 162

        vr = validate(store, level=4)
        assert vr.ok, vr.summary()

        result = read_mesh(store)
        assert result["vertex_count"] == 100

        coarsened = coarsen_mesh_cluster(
            result["vertices"], result["faces"], 3.0,
        )
        assert coarsened["vertex_count"] < 100
        assert coarsened["face_count"] < 162

        obj_out = tmp_path / "grid_out.obj"
        export_obj(store, obj_out)
        assert obj_out.exists()
        out_v = sum(1 for l in obj_out.read_text().split("\n") if l.startswith("v "))
        assert out_v == 100
