"""v0.7 per-level ``chunk_shape``: pyramid levels can override the root."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zarr_vectors.core.metadata import (
    LevelMetadata,
    RootMetadata,
    chunk_scale_factor,
    get_level_chunk_shape,
    validate_level_chunk_shape_against_root,
)
from zarr_vectors.core.store import (
    open_store,
    read_level_metadata,
    read_root_metadata,
)
from zarr_vectors.exceptions import MetadataError
from zarr_vectors_tools.multiresolution.coarsen import build_pyramid
from zarr_vectors.types.points import read_points, write_points


def _minimal_root_meta(chunk_shape=(200.0, 200.0, 200.0)) -> RootMetadata:
    return RootMetadata(
        zv_version="0.7.0",
        bounds=([0.0, 0.0, 0.0], [1000.0, 1000.0, 1000.0]),
        chunk_shape=chunk_shape,
        geometry_types=["point_cloud"],
        spatial_index_dims=[
            {"name": "x", "type": "space"},
            {"name": "y", "type": "space"},
            {"name": "z", "type": "space"},
        ],
    )


# ---------------------------------------------------------------------------
# Helper unit tests (pure)
# ---------------------------------------------------------------------------


def test_get_level_chunk_shape_falls_back_to_root_when_no_override() -> None:
    rm = _minimal_root_meta()
    assert get_level_chunk_shape(rm, None) == (200.0, 200.0, 200.0)

    lm = LevelMetadata(level=0, vertex_count=0, arrays_present=["vertices"])
    assert lm.chunk_shape is None
    assert get_level_chunk_shape(rm, lm) == (200.0, 200.0, 200.0)


def test_get_level_chunk_shape_returns_override_when_set() -> None:
    rm = _minimal_root_meta()
    lm = LevelMetadata(
        level=1,
        vertex_count=0,
        arrays_present=["vertices"],
        bin_shape=(100.0, 100.0, 100.0),
        chunk_shape=(400.0, 400.0, 400.0),
        parent_level=0,
    )
    assert get_level_chunk_shape(rm, lm) == (400.0, 400.0, 400.0)


def test_chunk_scale_factor_root_default_is_all_ones() -> None:
    rm = _minimal_root_meta()
    assert chunk_scale_factor(rm, None) == (1, 1, 1)


def test_chunk_scale_factor_extracts_per_axis_ratio() -> None:
    rm = _minimal_root_meta()
    lm = LevelMetadata(
        level=1,
        vertex_count=0,
        arrays_present=["vertices"],
        bin_shape=(100.0, 100.0, 100.0),
        chunk_shape=(400.0, 200.0, 600.0),  # 2×, 1×, 3×
        parent_level=0,
    )
    assert chunk_scale_factor(rm, lm) == (2, 1, 3)


def test_chunk_scale_factor_rejects_non_integer_ratio() -> None:
    rm = _minimal_root_meta()
    lm = LevelMetadata(
        level=1,
        vertex_count=0,
        arrays_present=["vertices"],
        bin_shape=(100.0, 100.0, 100.0),
        chunk_shape=(300.0, 200.0, 200.0),  # 1.5× on axis 0
        parent_level=0,
    )
    with pytest.raises(MetadataError, match="integer multiple"):
        chunk_scale_factor(rm, lm)


def test_validate_level_chunk_shape_passes_for_nested_grid() -> None:
    rm = _minimal_root_meta()
    lm = LevelMetadata(
        level=1,
        vertex_count=10,
        arrays_present=["vertices"],
        bin_shape=(100.0, 100.0, 100.0),
        chunk_shape=(400.0, 400.0, 400.0),
        parent_level=0,
    )
    # Should not raise.
    validate_level_chunk_shape_against_root(rm, lm)


def test_validate_level_chunk_shape_rejects_bin_not_dividing_chunk() -> None:
    rm = _minimal_root_meta()
    lm = LevelMetadata(
        level=1,
        vertex_count=10,
        arrays_present=["vertices"],
        bin_shape=(150.0, 150.0, 150.0),  # 400 / 150 isn't integer
        chunk_shape=(400.0, 400.0, 400.0),
        parent_level=0,
    )
    with pytest.raises(MetadataError, match="bin_shape"):
        validate_level_chunk_shape_against_root(rm, lm)


def test_validate_level_chunk_shape_is_noop_when_override_absent() -> None:
    rm = _minimal_root_meta()
    lm = LevelMetadata(
        level=0,
        vertex_count=10,
        arrays_present=["vertices"],
    )
    validate_level_chunk_shape_against_root(rm, lm)  # no chunk_shape → no-op


def test_level_metadata_round_trip_with_chunk_shape() -> None:
    lm = LevelMetadata(
        level=1,
        vertex_count=7,
        arrays_present=["vertices"],
        bin_shape=(100.0, 100.0, 100.0),
        chunk_shape=(400.0, 400.0, 400.0),
        parent_level=0,
    )
    d = lm.to_dict()
    inner = d["zarr_vectors_level"]
    assert inner["chunk_shape"] == [400.0, 400.0, 400.0]
    lm2 = LevelMetadata.from_dict(d)
    assert lm2.chunk_shape == (400.0, 400.0, 400.0)


def test_level_metadata_round_trip_without_chunk_shape() -> None:
    lm = LevelMetadata(
        level=1,
        vertex_count=7,
        arrays_present=["vertices"],
        bin_shape=(100.0, 100.0, 100.0),
        parent_level=0,
    )
    d = lm.to_dict()
    assert "chunk_shape" not in d["zarr_vectors_level"]
    lm2 = LevelMetadata.from_dict(d)
    assert lm2.chunk_shape is None


# ---------------------------------------------------------------------------
# End-to-end pyramid with growing chunk_shape
# ---------------------------------------------------------------------------


def test_pyramid_with_chunk_scale_factor_writes_per_level_metadata(tmp_path: Path) -> None:
    """Build a 3-level pyramid where each coarser level doubles chunk_shape.

    Verifies that the per-level ``LevelMetadata.chunk_shape`` is written
    correctly and that :class:`ZVLevel.chunk_shape` returns the
    effective per-level value.
    """
    rng = np.random.default_rng(0)
    pos = rng.uniform(0.0, 1000.0, size=(500, 3)).astype(np.float32)
    obj_ids = rng.integers(0, 5, size=500).astype(np.int64)
    store = str(tmp_path / "store.zarr")

    write_points(
        store, pos,
        chunk_shape=(200.0, 200.0, 200.0),
        bin_shape=(50.0, 50.0, 50.0),
        bounds=([0.0, 0.0, 0.0], [1000.0, 1000.0, 1000.0]),
        object_ids=obj_ids,
    )

    build_pyramid(
        store,
        factors=[(2.0, 1.0), (2.0, 1.0)],
        chunk_scale_factors=[2, 2],
    )

    root = open_store(store, mode="r+")
    rm = read_root_metadata(root)
    assert rm.chunk_shape == (200.0, 200.0, 200.0)

    lm0 = read_level_metadata(root, 0)
    assert lm0.chunk_shape is None  # inherits root

    lm1 = read_level_metadata(root, 1)
    assert lm1.chunk_shape == (400.0, 400.0, 400.0)
    assert chunk_scale_factor(rm, lm1) == (2, 2, 2)

    lm2 = read_level_metadata(root, 2)
    assert lm2.chunk_shape == (800.0, 800.0, 800.0)
    assert chunk_scale_factor(rm, lm2) == (4, 4, 4)


def test_per_level_chunk_shape_round_trips_through_read_points(tmp_path: Path) -> None:
    """A pyramid with growing chunks must still read vertices back
    correctly at each level (the writer used the per-level chunk_shape
    when assigning metanodes to chunks; the reader must use the same)."""
    rng = np.random.default_rng(0)
    pos = rng.uniform(0.0, 1000.0, size=(300, 3)).astype(np.float32)
    store = str(tmp_path / "store.zarr")

    write_points(
        store, pos,
        chunk_shape=(200.0, 200.0, 200.0),
        bin_shape=(50.0, 50.0, 50.0),
        bounds=([0.0, 0.0, 0.0], [1000.0, 1000.0, 1000.0]),
        object_ids=np.arange(300) % 7,
    )
    build_pyramid(
        store,
        factors=[(2.0, 1.0), (2.0, 1.0)],
        chunk_scale_factors=[2, 2],
    )

    # Level 0 has all 300 vertices.
    out0 = read_points(store, level=0)
    assert out0["vertex_count"] == 300

    # Coarser levels have fewer (binned) but more than zero.
    out1 = read_points(store, level=1)
    out2 = read_points(store, level=2)
    assert 0 < out1["vertex_count"] <= 300
    assert 0 < out2["vertex_count"] <= out1["vertex_count"]


def test_chunk_scale_factor_one_writes_no_per_level_override(tmp_path: Path) -> None:
    """The default chunk_scale_factor=1 (or omitted) must leave
    ``LevelMetadata.chunk_shape`` unset — single-level stores and
    pyramids without chunk scaling look exactly like v0.6 on disk
    except for the version bump."""
    rng = np.random.default_rng(0)
    pos = rng.uniform(0.0, 1000.0, size=(200, 3)).astype(np.float32)
    store = str(tmp_path / "store.zarr")

    write_points(
        store, pos,
        chunk_shape=(200.0, 200.0, 200.0),
        bin_shape=(50.0, 50.0, 50.0),
        bounds=([0.0, 0.0, 0.0], [1000.0, 1000.0, 1000.0]),
        object_ids=np.arange(200) % 5,
    )
    build_pyramid(store, factors=[(2.0, 1.0)])  # no chunk_scale_factors

    root = open_store(store, mode="r+")
    lm1 = read_level_metadata(root, 1)
    assert lm1.chunk_shape is None


def test_chunk_scale_factor_rejects_non_positive(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    pos = rng.uniform(0.0, 1000.0, size=(200, 3)).astype(np.float32)
    store = str(tmp_path / "store.zarr")
    write_points(
        store, pos,
        chunk_shape=(200.0, 200.0, 200.0),
        bin_shape=(50.0, 50.0, 50.0),
        bounds=([0.0, 0.0, 0.0], [1000.0, 1000.0, 1000.0]),
        object_ids=np.arange(200) % 5,
    )
    with pytest.raises(Exception, match="positive integers"):
        build_pyramid(
            store,
            factors=[(2.0, 1.0)],
            chunk_scale_factors=[0],
        )


def test_chunk_scale_factor_rank_must_match_ndim(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    pos = rng.uniform(0.0, 1000.0, size=(200, 3)).astype(np.float32)
    store = str(tmp_path / "store.zarr")
    write_points(
        store, pos,
        chunk_shape=(200.0, 200.0, 200.0),
        bin_shape=(50.0, 50.0, 50.0),
        bounds=([0.0, 0.0, 0.0], [1000.0, 1000.0, 1000.0]),
        object_ids=np.arange(200) % 5,
    )
    with pytest.raises(Exception, match="rank"):
        build_pyramid(
            store,
            factors=[(2.0, 1.0)],
            chunk_scale_factors=[(2, 2)],  # rank 2 vs sid_ndim=3
        )
