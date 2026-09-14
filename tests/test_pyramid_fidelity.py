"""Pyramid levels must mean the same thing as the level below them.

Every case here failed silently before — the store was written, nothing
raised, and the damage only showed up in what a viewer drew or in a number
a downstream tool computed:

* A streamline's fragments are concatenated **in manifest order**, so a
  coarse level whose manifest is sorted by chunk hands back a tract whose
  segments are shuffled.  Tangents, lengths and endpoints are all wrong,
  and the geometry still looks plausible.
* ``--sparsity-strategy length`` is what the large-scale docs recommend,
  and it raised on every store that is not a streamline store.
* A level whose declared ``chunk_shape`` disagrees with the coordinates
  stored in it resolves bbox queries to the wrong chunks.
* A "refresh" that rebuilds a level with a different strategy, or with a
  stride it had to invent, is not a refresh.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from zarr_vectors.building import (
    get_resolution_level,
    list_chunk_keys,
    open_store,
    read_all_object_manifests,
    read_chunk_vertices,
    read_level_metadata,
    read_root_metadata,
)
from zarr_vectors.exceptions import EditError
from zarr_vectors.types.points import write_points
from zarr_vectors.types.polylines import read_polylines

from tests._source_helpers import write_polylines_with_segment_id
from zarr_vectors_tools.ingest.trk_parallel import _compute_chunk_shape
from zarr_vectors_tools.multiresolution.coarsen import build_pyramid
from zarr_vectors_tools.multiresolution.refresh import rebuild_pyramid_from_level

CHUNK = (10.0, 10.0, 10.0)


def _concat_path(store: Path, level: int, oid: int = 0) -> np.ndarray:
    result = read_polylines(str(store), level=level)
    index = result["object_ids"].index(oid) if isinstance(
        result["object_ids"], list,
    ) else int(np.flatnonzero(np.asarray(result["object_ids"]) == oid)[0])
    return np.concatenate(result["polylines"][index], axis=0)


# =====================================================================
# Walk order
# =====================================================================


class TestStreamlineWalkOrder:

    def _descending_store(self, tmp_path: Path) -> Path:
        """One streamline walking from high x to low x across three chunks."""
        store = tmp_path / "descending.zv"
        x = np.linspace(25.0, 5.0, 21)
        poly = np.column_stack(
            [x, np.full(21, 5.0), np.full(21, 5.0)],
        ).astype(np.float32)
        write_polylines_with_segment_id(
            str(store), [poly], chunk_shape=CHUNK, geometry_type="streamline",
        )
        return store

    def test_coarse_level_keeps_the_direction_of_travel(
        self, tmp_path: Path,
    ) -> None:
        """The tract runs 25 -> 5 at level 0 and must still at level 1.

        Ordering the coarse manifest by ``(chunk, fragment)`` instead
        returned it as 9->5, 19->10, 25->20: three correct runs, played in
        chunk-coordinate order.
        """
        store = self._descending_store(tmp_path)
        build_pyramid(
            str(store), factors=[(1.0, 1.0)], coarsen_mode="decimate",
            cross_level_depth=0, cross_level_storage="none",
        )

        level0 = _concat_path(store, 0)[:, 0]
        level1 = _concat_path(store, 1)[:, 0]
        assert np.all(np.diff(level0) < 0)
        assert np.all(np.diff(level1) < 0), (
            f"level 1 walks {level1.tolist()}, which is not monotone"
        )
        np.testing.assert_allclose(level0, level1)

    def test_manifest_chunk_order_matches_the_level_below(
        self, tmp_path: Path,
    ) -> None:
        store = self._descending_store(tmp_path)
        build_pyramid(
            str(store), factors=[(1.0, 1.0)], coarsen_mode="decimate",
            cross_level_depth=0, cross_level_storage="none",
        )
        root = open_store(str(store))
        source = read_all_object_manifests(get_resolution_level(root, 0))[0]
        target = read_all_object_manifests(get_resolution_level(root, 1))[0]
        assert [cc for cc, _ in target] == [cc for cc, _ in source]

    def test_order_survives_a_two_level_pyramid(self, tmp_path: Path) -> None:
        """Level 2 is coarsened from level 1's manifests, so it inherits
        whatever order they were written in."""
        store = self._descending_store(tmp_path)
        build_pyramid(
            str(store), factors=[(1.0, 1.0), (1.0, 1.0)],
            chunk_scale_factors=[1, 1], coarsen_mode="decimate",
            cross_level_depth=0, cross_level_storage="none",
        )
        for level in (1, 2):
            walk = _concat_path(store, level)[:, 0]
            assert np.all(np.diff(walk) < 0), f"level {level}: {walk.tolist()}"

    def test_rdp_mode_also_keeps_the_direction(self, tmp_path: Path) -> None:
        store = self._descending_store(tmp_path)
        build_pyramid(
            str(store), factors=[(2.0, 1.0)], coarsen_mode="rdp",
            cross_level_depth=0, cross_level_storage="none",
        )
        level1 = _concat_path(store, 1)[:, 0]
        assert np.all(np.diff(level1) < 0)


# =====================================================================
# Sparsity strategies on the per-object coarsener
# =====================================================================


def _sized_point_store(path: Path, n_objects: int = 40) -> Path:
    """Object ``k`` holds ``k + 1`` points, so sizes are strictly ordered."""
    rng = np.random.default_rng(1)
    positions, ids = [], []
    for k in range(n_objects):
        positions.append(rng.random((k + 1, 3)) * 40.0)
        ids.append(np.full(k + 1, k))
    write_points(
        str(path),
        np.concatenate(positions).astype(np.float32),
        chunk_shape=CHUNK,
        bin_shape=(1.0, 1.0, 1.0),
        object_ids=np.concatenate(ids),
    )
    return path


def _surviving(store: Path, level: int) -> list[int]:
    manifests = read_all_object_manifests(
        get_resolution_level(open_store(str(store)), level),
    )
    return [oid for oid, manifest in enumerate(manifests) if manifest]


class TestPerObjectSparsityStrategies:

    @pytest.mark.parametrize("strategy", ["random", "length", "spatial_coverage"])
    def test_strategy_runs_on_a_point_store(
        self, tmp_path: Path, strategy: str,
    ) -> None:
        """The CLI offers every one of these for a point cloud.

        ``length`` and ``spatial_coverage`` raised "requires 'lengths'
        array" / "requires 'representative_points'" because the coarsener
        passed neither.
        """
        store = _sized_point_store(tmp_path / f"{strategy}.zv")
        build_pyramid(
            str(store), factors=[(2.0, 2.0)], sparsity_strategy=strategy,
            sparsity_seed=0, cross_level_depth=0, cross_level_storage="none",
        )
        assert len(_surviving(store, 1)) == 20

    def test_length_keeps_the_largest_objects(self, tmp_path: Path) -> None:
        store = _sized_point_store(tmp_path / "by_length.zv")
        build_pyramid(
            str(store), factors=[(2.0, 2.0)], sparsity_strategy="length",
            cross_level_depth=0, cross_level_storage="none",
        )
        # Object k has k + 1 vertices, so keeping half must keep 20..39.
        assert _surviving(store, 1) == list(range(20, 40))


# =====================================================================
# Per-level chunk geometry
# =====================================================================


class TestLevelChunkShape:

    @pytest.mark.parametrize("method", ["per_object", "per_fragment"])
    def test_declared_chunk_shape_matches_the_stored_coordinates(
        self, tmp_path: Path, method: str,
    ) -> None:
        """Every vertex must fall in the chunk its key names.

        The per-fragment strategy scaled the ROOT chunk shape once per
        level while dividing SOURCE coordinates, so at level 2 of a
        ``[2, 2]`` pyramid the store declared 20 units where the
        coordinates had been divided by 40.
        """
        store = tmp_path / f"{method}.zv"
        rng = np.random.default_rng(1)
        write_points(
            str(store),
            (rng.random((400, 3)) * 80.0).astype(np.float32),
            chunk_shape=CHUNK,
            bin_shape=(1.0, 1.0, 1.0),
            object_ids=np.repeat(np.arange(40), 10),
        )
        build_pyramid(
            str(store), factors=[(1.0, 1.0), (1.0, 1.0)],
            chunk_scale_factors=[2, 2], method=method,
            cross_level_depth=0, cross_level_storage="none",
        )

        root = open_store(str(store))
        root_meta = read_root_metadata(root)
        for level, expected in ((1, 20.0), (2, 40.0)):
            meta = read_level_metadata(root, level)
            shape = np.asarray(meta.chunk_shape or root_meta.chunk_shape)
            np.testing.assert_allclose(shape, expected)
            group = get_resolution_level(root, level)
            for chunk in list_chunk_keys(group, "vertices"):
                for fragment in read_chunk_vertices(group, chunk, ndim=3):
                    if len(fragment) == 0:
                        continue
                    cells = np.floor(
                        np.asarray(fragment, dtype=np.float64) / shape,
                    ).astype(np.int64)
                    assert np.all(cells == np.asarray(chunk)), (
                        f"level {level}: a vertex in chunk {tuple(chunk)} "
                        f"resolves to {cells[0].tolist()} at chunk_shape "
                        f"{shape.tolist()}"
                    )


# =====================================================================
# Refresh
# =====================================================================


def _wiggly_streamlines(n: int = 6, points: int = 60) -> list[np.ndarray]:
    out = []
    for i in range(n):
        t = np.linspace(0.0, 1.0, points)
        out.append(np.column_stack([
            t * 30.0,
            5.0 + 3.0 * np.sin(t * 12.0 + i),
            5.0 + 2.0 * np.cos(t * 9.0),
        ]).astype(np.float32))
    return out


class TestRefresh:

    def test_decimated_streamline_pyramid_is_reproduced(
        self, tmp_path: Path,
    ) -> None:
        """A refresh must not silently switch the reduction mode.

        ``coarsening_method`` records which mode built the level; without
        reading it the refresh rebuilt a stride-decimated level with
        Douglas-Peucker, taking 114 vertices to 42.
        """
        store = tmp_path / "decimate.zv"
        write_polylines_with_segment_id(
            str(store), _wiggly_streamlines(), chunk_shape=CHUNK,
            geometry_type="streamline",
        )
        build_pyramid(
            str(store), factors=[(4.0, 1.0)], coarsen_mode="decimate",
            cross_level_depth=0, cross_level_storage="none",
        )
        before = read_level_metadata(open_store(str(store)), 1)

        rebuild_pyramid_from_level(open_store(str(store), mode="r+"), 0)

        after = read_level_metadata(open_store(str(store)), 1)
        assert after.coarsening_method == before.coarsening_method
        assert after.vertex_count == before.vertex_count

    def test_skeleton_level_refuses_rather_than_inventing_a_stride(
        self, tmp_path: Path,
    ) -> None:
        """Nothing on a skeleton level records its stride.

        The bin-ratio arithmetic every other method is recovered by reads
        back 1.0 here, which the coarsener takes as "keep anchors only" --
        so the level was silently flattened.  Refusing names the one
        parameter the caller has to supply.
        """
        pytest.importorskip("zarr_vectors.types.skeletons")
        from tests.test_skeleton_coarsen import _write_two_skeletons
        from zarr_vectors_tools.multiresolution.strategies.skeletons import (
            build_skeleton_pyramid,
        )

        store = tmp_path / "skeleton.zv"
        _write_two_skeletons(str(store))
        build_skeleton_pyramid(str(store), strides=[4], chunk_scale_factors=[2])
        before = read_level_metadata(open_store(str(store)), 1).vertex_count

        with pytest.raises(EditError, match="stride"):
            rebuild_pyramid_from_level(open_store(str(store), mode="r+"), 0)

        rebuild_pyramid_from_level(
            open_store(str(store), mode="r+"), 0, coarsen_factors={1: 4},
        )
        after = read_level_metadata(open_store(str(store)), 1).vertex_count
        assert after == before


# =====================================================================
# Chunk sizing
# =====================================================================


class TestComputeChunkShape:

    @pytest.mark.parametrize(
        "extent", [0.4, 3.0, 150.0, 0.0],
    )
    def test_never_returns_a_zero_edge(self, extent: float) -> None:
        """``round(extent / n)`` hits zero on any sub-chunk-count extent.

        Every downstream ``floor(p / chunk_shape)`` then divides by zero.
        """
        shape = _compute_chunk_shape(([0.0, 0.0, 0.0], [extent] * 3), 125)
        assert all(edge > 0 for edge in shape), shape

    def test_ordinary_extent_is_unchanged(self) -> None:
        assert _compute_chunk_shape(
            ([0.0, 0.0, 0.0], [150.0, 180.0, 110.0]), 125,
        ) == (30.0, 30.0, 28.0)
