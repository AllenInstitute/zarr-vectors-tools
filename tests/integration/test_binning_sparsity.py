"""Integration tests for supervoxel binning and object sparsity.

These tests exercise the complete pipeline: write with bin_shape →
validate → read with bbox targeting → build sparsity pyramid →
validate multi-resolution → read coarser levels.

Also verifies backward compatibility — existing tests still pass
without specifying bin_shape or sparsity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


class TestPointCloudSubChunkBins:
    """1. Point cloud with sub-chunk bins."""

    def test_50k_points_with_bins(self, tmp_path: Path) -> None:
        from zarr_vectors.types.points import write_points, read_points
        from zarr_vectors.validate import validate
        from zarr_vectors.core.store import open_store, read_root_metadata

        rng = np.random.default_rng(42)
        store = str(tmp_path / "pts.zv")
        positions = rng.uniform(0, 800, size=(50000, 3)).astype(np.float32)

        # 4x4x4 = 64 bins per chunk
        summary = write_points(
            store, positions,
            chunk_shape=(200., 200., 200.),
            bin_shape=(50., 50., 50.),
        )
        assert summary["vertex_count"] == 50000
        assert summary["bins_per_chunk"] == (4, 4, 4)

        # Metadata
        meta = read_root_metadata(open_store(store))
        assert meta.base_bin_shape == (50., 50., 50.)
        assert meta.bins_per_chunk == (4, 4, 4)

        # Read all
        assert read_points(store)["vertex_count"] == 50000

        # Bbox for single 50³ bin
        result = read_points(store, bbox=(
            np.array([0, 0, 0]), np.array([49.9, 49.9, 49.9])))
        expected = int(np.sum(np.all((positions >= 0) & (positions < 49.9), axis=1)))
        assert result["vertex_count"] == expected

        # Validate L5
        assert validate(store, level=5).ok


class TestStreamlinesWithSparsity:
    """2. Streamlines with sparsity pyramid."""

    def test_streamline_sparsity(self, tmp_path: Path) -> None:
        from zarr_vectors.types.polylines import write_polylines
        from zarr_vectors.validate import validate
        from zarr_vectors_tools.multiresolution.coarsen import build_pyramid

        rng = np.random.default_rng(42)
        store = str(tmp_path / "tracts.zv")
        polys = []
        for _ in range(200):
            nn = rng.integers(20, 60)
            start = rng.uniform(0, 300, size=(1, 3)).astype(np.float32)
            steps = rng.normal(0, 3, size=(nn - 1, 3)).astype(np.float32)
            polys.append(np.clip(
                np.concatenate([start, start + np.cumsum(steps, axis=0)]),
                0, 399,
            ).astype(np.float32))

        write_polylines(
            store, polys,
            chunk_shape=(200., 200., 200.),
            bin_shape=(50., 50., 50.),
        )

        summary = build_pyramid(store, factors=[(2.0, 2.0)])
        assert summary["levels_created"] == 1
        # sparsity_factor=2.0 → keep_frac=0.5 ≈ object_sparsity=0.5.
        # Approximate check: at least one object was dropped.
        assert summary["level_specs"][0]["objects_kept"] < summary["level_specs"][0]["source_objects"]

        assert validate(store, level=5).ok


class TestMeshBinning:
    """3. Mesh with binning only."""

    def test_mesh_coarsening(self, tmp_path: Path) -> None:
        from zarr_vectors.types.meshes import write_mesh, read_mesh
        from zarr_vectors_tools.multiresolution.coarsen import coarsen_level
        from zarr_vectors.validate import validate

        rng = np.random.default_rng(42)
        store = str(tmp_path / "mesh.zv")
        n_verts = 500
        vertices = rng.uniform(0, 200, size=(n_verts, 3)).astype(np.float32)
        faces = rng.integers(0, n_verts, size=(800, 3)).astype(np.int64)

        write_mesh(
            store, vertices, faces,
            chunk_shape=(200., 200., 200.),
            bin_shape=(50., 50., 50.),
        )

        summary = coarsen_level(store, 0, 1, coarsen_factor=2.0)
        assert summary["vertex_count"] > 0
        assert summary["vertex_count"] < n_verts

        r = read_mesh(store, level=0)
        assert r["vertex_count"] == n_verts

        assert validate(store, level=4).ok


class TestSkeletonManualLevels:
    """5. Manual level creation."""

    def test_multiple_manual_ratios(self, tmp_path: Path) -> None:
        from zarr_vectors.types.points import write_points
        from zarr_vectors_tools.multiresolution.coarsen import coarsen_level
        from zarr_vectors.core.store import (
            open_store, list_resolution_levels, list_available_ratios,
        )
        from zarr_vectors.validate import validate

        rng = np.random.default_rng(42)
        store = str(tmp_path / "man.zv")
        positions = rng.uniform(0, 600, size=(2000, 3)).astype(np.float32)

        # chunk=600, bin=50 → 12^3=1728 bins per chunk maximum
        # but chunks are only 600³, so one chunk
        write_points(
            store, positions,
            chunk_shape=(600., 600., 600.),
            bin_shape=(50., 50., 50.),
        )

        # Add levels at ratios (2,2,2) and (6,6,6) — 6 divides 12 per axis
        coarsen_level(store, 0, 1, coarsen_factor=2.0)
        coarsen_level(store, 0, 2, coarsen_factor=6.0)

        ratios = list_available_ratios(open_store(store))
        assert (1, 1, 1) in ratios
        assert (2, 2, 2) in ratios
        assert (6, 6, 6) in ratios

        levels = list_resolution_levels(open_store(store))
        assert len(levels) == 3

        assert validate(store, level=5).ok


class TestBackwardCompat:
    """6. Backward compatibility — step 15 tests pass unchanged."""

    def test_simple_point_cloud(self, tmp_path: Path) -> None:
        from zarr_vectors.types.points import write_points, read_points
        from zarr_vectors.validate import validate

        rng = np.random.default_rng(42)
        store = str(tmp_path / "p.zv")
        positions = rng.uniform(0, 1000, size=(5000, 3)).astype(np.float32)

        write_points(store, positions, chunk_shape=(200., 200., 200.))
        assert read_points(store)["vertex_count"] == 5000
        assert validate(store, level=5).ok

    def test_pyramid_backward_compat(self, tmp_path: Path) -> None:
        from zarr_vectors.types.points import write_points
        from zarr_vectors_tools.multiresolution.coarsen import build_pyramid
        from zarr_vectors.validate import validate

        rng = np.random.default_rng(42)
        store = str(tmp_path / "pyr.zv")
        positions = rng.uniform(0, 1000, size=(10000, 3)).astype(np.float32)

        write_points(store, positions, chunk_shape=(100., 100., 100.))
        summary = build_pyramid(store, factors=[(2.0, 1.0)])
        assert summary["levels_created"] >= 1
        assert validate(store, level=5).ok

    def test_streamlines(self, tmp_path: Path) -> None:
        from zarr_vectors.types.polylines import write_polylines, read_polylines
        from zarr_vectors.validate import validate

        rng = np.random.default_rng(42)
        store = str(tmp_path / "pl.zv")
        polys = [
            rng.uniform(0, 100, size=(30, 3)).astype(np.float32)
            for _ in range(10)
        ]

        write_polylines(store, polys, chunk_shape=(100., 100., 100.))
        assert read_polylines(store)["polyline_count"] == 10
        assert validate(store, level=5).ok


class TestOMEZarrMetadata:
    """7. OME-Zarr multiscale metadata round-trip."""

    def test_multiscale_roundtrip(self, tmp_path: Path) -> None:
        from zarr_vectors.types.points import write_points
        from zarr_vectors_tools.multiresolution.coarsen import build_pyramid
        from zarr_vectors.core.multiscale import (
            write_multiscale_metadata, read_multiscale_metadata,
            get_level_scale, get_level_translation,
        )
        from zarr_vectors.core.store import open_store

        rng = np.random.default_rng(42)
        store = str(tmp_path / "ms.zv")
        positions = rng.uniform(0, 400, size=(5000, 3)).astype(np.float32)

        write_points(
            store, positions,
            chunk_shape=(200., 200., 200.),
            bin_shape=(50., 50., 50.),
        )
        build_pyramid(store, factors=[(2.0, 1.0), (4.0, 1.0)])

        root = open_store(store, mode="r+")
        ms = write_multiscale_metadata(root)
        assert ms[0]["version"] == "0.4"
        assert ms[0]["metadata"]["format"] == "zarr_vectors"
        assert "type" not in ms[0]
        assert len(ms[0]["datasets"]) == 3

        # Verify scales
        assert get_level_scale(root, 0) == [1.0, 1.0, 1.0]
        assert get_level_scale(root, 1) == [2.0, 2.0, 2.0]
        assert get_level_scale(root, 2) == [4.0, 4.0, 4.0]

        # Translation = bin_shape / 2
        assert get_level_translation(root, 1) == [50.0, 50.0, 50.0]
        assert get_level_translation(root, 2) == [100.0, 100.0, 100.0]

        # Round-trip
        ms_read = read_multiscale_metadata(root)
        assert ms_read == ms


class TestValidationRejection:
    """Validation rejects invalid configurations."""

    def test_rejects_non_divisible_bin(self, tmp_path: Path) -> None:
        """chunk_shape not divisible by bin_shape should be rejected."""
        from zarr_vectors.core.metadata import RootMetadata
        from zarr_vectors.exceptions import MetadataError

        with pytest.raises(MetadataError):
            RootMetadata(
                spatial_index_dims=[
                    {"name": "x", "type": "space", "unit": "um"},
                    {"name": "y", "type": "space", "unit": "um"},
                    {"name": "z", "type": "space", "unit": "um"},
                ],
                chunk_shape=(200., 200., 200.),
                bounds=([0, 0, 0], [200, 200, 200]),
                geometry_types=["point_cloud"],
                base_bin_shape=(60., 50., 50.),  # 200/60 = 3.33
            ).validate()

    def test_rejects_invalid_sparsity(self, tmp_path: Path) -> None:
        """object_sparsity must be in (0, 1]."""
        from zarr_vectors.core.metadata import LevelMetadata
        from zarr_vectors.exceptions import MetadataError

        with pytest.raises(MetadataError):
            LevelMetadata(
                level=1, vertex_count=10, arrays_present=["vertices"],
                bin_shape=(100., 100., 100.), object_sparsity=1.5,
                parent_level=0,
            ).validate()

        with pytest.raises(MetadataError):
            LevelMetadata(
                level=1, vertex_count=10, arrays_present=["vertices"],
                bin_shape=(100., 100., 100.), object_sparsity=0.0,
                parent_level=0,
            ).validate()
