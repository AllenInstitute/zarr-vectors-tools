"""Random-subset pyramid for point clouds.

The general ``build_pyramid`` does not fit an object-less point cloud —
its sparsity factor drops *objects*, and its coarseners write geometry
only — so this builder exists for stores like a spatial-omics cell atlas,
where the coarse levels must stay colourable by gene and metadata.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr_vectors.building as B
from zarr_vectors.exceptions import CoarseningError
from zarr_vectors.types.points import read_points

from zarr_vectors_tools.ingest.cell_table import ingest_table
from zarr_vectors_tools.multiresolution.strategies.points import (
    build_point_subset_pyramid,
)

_BASE_LABEL = 182941331246012878296807398333956011710


@pytest.fixture
def cell_store(tmp_path: Path):
    """A 4,000-cell store with metadata and two 'gene' columns."""
    rng = np.random.default_rng(0)
    n = 4000
    labels = [str(_BASE_LABEL + i) for i in range(n)]
    positions = rng.uniform(0, 100, (n, 3))
    frame = pd.DataFrame(
        {
            "cell_label": labels,
            "x": positions[:, 0], "y": positions[:, 1], "z": positions[:, 2],
            "cluster": rng.integers(0, 40, n),
            "gene_A": rng.random(n).astype(np.float32),
            "gene_B": rng.random(n).astype(np.float32),
        }
    )
    source = tmp_path / "cells.csv"
    frame.to_csv(source, index=False)
    store = tmp_path / "cells.zarr"
    ingest_table(
        source, store, (25.0, 25.0, 25.0),
        position_columns=["x", "y", "z"], key_column="cell_label",
    )
    return store, frame


def keys_at(store, level: int) -> np.ndarray:
    return np.asarray(
        read_points(str(store), level=level, attribute_names=["zv_join_key"])
        ["vertex_attributes"]["zv_join_key"]
    )


class TestSubsetPyramid:

    def test_each_level_is_an_eighth_of_its_parent(self, cell_store) -> None:
        store, _ = cell_store
        result = build_point_subset_pyramid(store, levels=3, divisor=8, seed=0)
        assert result["levels_created"] == 3

        counts = [
            B.read_level_metadata(B.open_store(str(store), "r"), lv).vertex_count
            for lv in range(4)
        ]
        assert counts == [4000, 500, 62, 7]

    def test_levels_nest(self, cell_store) -> None:
        """Level n+1 must be a subset of level n.

        Nesting is what keeps a point's appearance monotonic as a viewer
        zooms — without it a cell could vanish and reappear between levels.
        """
        store, _ = cell_store
        build_point_subset_pyramid(store, levels=3, divisor=8, seed=0)

        previous = set(keys_at(store, 0).tolist())
        for level in (1, 2, 3):
            current = set(keys_at(store, level).tolist())
            assert current <= previous, f"level {level} is not a subset of its parent"
            previous = current

    def test_attributes_are_carried_with_exact_values(self, cell_store) -> None:
        """The whole point: coarse levels stay colourable."""
        store, frame = cell_store
        build_point_subset_pyramid(store, levels=3, divisor=8, seed=0)

        base = read_points(
            str(store), level=0,
            attribute_names=["zv_join_key", "cluster", "gene_A", "gene_B"],
        )
        lookup = {
            int(key): (
                base["positions"][i],
                base["vertex_attributes"]["cluster"][i],
                base["vertex_attributes"]["gene_A"][i],
                base["vertex_attributes"]["gene_B"][i],
            )
            for i, key in enumerate(base["vertex_attributes"]["zv_join_key"])
        }

        for level in (1, 2, 3):
            got = read_points(
                str(store), level=level,
                attribute_names=["zv_join_key", "cluster", "gene_A", "gene_B"],
            )
            attrs = got["vertex_attributes"]
            assert got["vertex_count"] > 0
            for i, key in enumerate(attrs["zv_join_key"]):
                position, cluster, gene_a, gene_b = lookup[int(key)]
                np.testing.assert_allclose(got["positions"][i], position)
                assert attrs["cluster"][i] == cluster
                assert attrs["gene_A"][i] == pytest.approx(gene_a)
                assert attrs["gene_B"][i] == pytest.approx(gene_b)

    def test_level_metadata_records_identity_scale(self, cell_store) -> None:
        """Subsetting removes points without moving them, so the level
        still occupies level 0's coordinate space — bin_ratio stays 1,
        unlike a binning coarsener that rescales the grid."""
        store, _ = cell_store
        build_point_subset_pyramid(store, levels=2, divisor=8, seed=0)

        root = B.open_store(str(store), "r")
        for level in (1, 2):
            meta = B.read_level_metadata(root, level)
            assert meta.coarsening_method == "random_subset"
            assert meta.parent_level == level - 1
            assert meta.bin_ratio == (1, 1, 1)
            assert meta.object_sparsity == pytest.approx(0.125)
            assert "vertex_attributes" in meta.arrays_present

    def test_seed_is_reproducible_and_matters(self, tmp_path: Path) -> None:
        def build(name: str, seed: int) -> np.ndarray:
            rng = np.random.default_rng(1)
            n = 2000
            frame = pd.DataFrame({
                "cell_label": [str(_BASE_LABEL + i) for i in range(n)],
                "x": rng.uniform(0, 100, n),
                "y": rng.uniform(0, 100, n),
                "z": rng.uniform(0, 100, n),
            })
            source = tmp_path / f"{name}.csv"
            frame.to_csv(source, index=False)
            store = tmp_path / f"{name}.zarr"
            ingest_table(
                source, store, (50.0, 50.0, 50.0),
                position_columns=["x", "y", "z"], key_column="cell_label",
            )
            build_point_subset_pyramid(store, levels=1, divisor=8, seed=seed)
            return np.sort(keys_at(store, 1))

        np.testing.assert_array_equal(build("a", 7), build("b", 7))
        assert not np.array_equal(build("c", 7), build("d", 99))

    def test_geometry_only_mode(self, cell_store) -> None:
        store, _ = cell_store
        result = build_point_subset_pyramid(
            store, levels=2, divisor=8, seed=0, attributes=False
        )
        assert result["attributes_carried"] == 0
        meta = B.read_level_metadata(B.open_store(str(store), "r"), 1)
        assert "vertex_attributes" not in meta.arrays_present

    def test_selected_attributes_only(self, cell_store) -> None:
        store, _ = cell_store
        build_point_subset_pyramid(
            store, levels=1, divisor=8, seed=0, attribute_names=["gene_A"]
        )
        got = read_points(str(store), level=1, attribute_names=["gene_A"])
        assert "gene_A" in got["vertex_attributes"]

    def test_unknown_attribute_is_rejected(self, cell_store) -> None:
        store, _ = cell_store
        with pytest.raises(CoarseningError, match="attributes not at level"):
            build_point_subset_pyramid(
                store, levels=1, divisor=8, attribute_names=["nope"]
            )

    def test_stops_when_a_subset_would_be_empty(self, cell_store) -> None:
        """Asking for more levels than the point count supports stops
        early rather than writing empty levels."""
        store, _ = cell_store
        result = build_point_subset_pyramid(store, levels=12, divisor=8, seed=0)
        assert result["levels_created"] < 12
        counts = [
            B.read_level_metadata(B.open_store(str(store), "r"), lv).vertex_count
            for lv in range(1, result["levels_created"] + 1)
        ]
        assert all(c > 0 for c in counts)

    def test_sharded_levels(self, cell_store) -> None:
        """Sharding the new levels is what keeps the file count sane."""
        import zarr
        from zarr_vectors.sharding.io import _is_native_sharded

        store, _ = cell_store
        build_point_subset_pyramid(
            store, levels=2, divisor=8, seed=0, shard_shape=8
        )
        level = B.get_resolution_level(B.open_store(str(store), "r"), 1)
        node = level.zarr_group["vertex_attributes/gene_A"]
        assert isinstance(node, zarr.Array)
        assert _is_native_sharded(node)

    def test_existing_level_is_an_error_without_resume(self, cell_store) -> None:
        store, _ = cell_store
        build_point_subset_pyramid(store, levels=1, divisor=8, seed=0)
        with pytest.raises(CoarseningError, match="already exists"):
            build_point_subset_pyramid(store, levels=1, divisor=8, seed=0)

    def test_resume_refills_an_incomplete_attribute(self, cell_store) -> None:
        """The failure this guards against: a build interrupted mid-batch
        leaves attribute arrays that exist but hold no chunks, so a read
        requesting one comes back short instead of raising."""
        store, frame = cell_store
        build_point_subset_pyramid(store, levels=2, divisor=8, seed=0)

        expected = read_points(
            str(store), level=1, attribute_names=["zv_join_key", "gene_A"]
        )
        wanted = dict(zip(
            np.asarray(expected["vertex_attributes"]["zv_join_key"]).tolist(),
            np.asarray(expected["vertex_attributes"]["gene_A"]).tolist(),
        ))

        level = B.get_resolution_level(B.open_store(str(store), "r+"), 1)
        level.delete_subtree("vertex_attributes/gene_A")
        assert "gene_A" not in level.require_group("vertex_attributes").children()

        build_point_subset_pyramid(store, levels=2, divisor=8, seed=0, resume=True)

        healed = read_points(
            str(store), level=1, attribute_names=["zv_join_key", "gene_A"]
        )
        assert healed["vertex_count"] == expected["vertex_count"]
        for key, value in zip(
            healed["vertex_attributes"]["zv_join_key"],
            healed["vertex_attributes"]["gene_A"],
        ):
            assert value == pytest.approx(wanted[int(key)])

    def test_resume_refuses_mismatched_parameters(self, cell_store) -> None:
        """Resuming with a different seed would misalign every value."""
        store, _ = cell_store
        build_point_subset_pyramid(store, levels=1, divisor=8, seed=0)
        with pytest.raises(CoarseningError, match="different parameters"):
            build_point_subset_pyramid(
                store, levels=1, divisor=4, seed=0, resume=True
            )

    def test_resume_is_a_noop_when_complete(self, cell_store) -> None:
        store, _ = cell_store
        build_point_subset_pyramid(store, levels=2, divisor=8, seed=0)
        before = keys_at(store, 2).tolist()
        build_point_subset_pyramid(store, levels=2, divisor=8, seed=0, resume=True)
        assert keys_at(store, 2).tolist() == before

    def test_store_still_validates(self, cell_store) -> None:
        from zarr_vectors.validate import validate

        store, _ = cell_store
        build_point_subset_pyramid(store, levels=3, divisor=8, seed=0)
        assert bool(getattr(validate(str(store), level=3), "ok", False))
