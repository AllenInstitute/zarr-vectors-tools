"""The pyramid must scale, and mean the same thing, at every level.

These pin the Day-0 defects found in the hackathon review.  Each one cost
either work proportional to the store where none was needed, or a level that
silently disagreed with the parameters it was built from:

* the cross-level finalize pass read every level before discovering it had
  nothing to write, which is the largest allocation in a pyramid build;
* an RDP tolerance derived from chunk size, so a second level at the same
  factor removed nothing;
* ``stride=1`` meaning "delete the interior of every chain" in the skeleton
  coarsener while meaning "change nothing" everywhere else;
* a quadric mesh level stamping a sparsity it never applied;
* two helpers that scanned the whole input once per output group.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
from zarr_vectors.building import (
    open_store,
    read_level_metadata,
    read_root_metadata,
)
from zarr_vectors.types.meshes import write_mesh
from zarr_vectors.types.points import write_points

from tests._source_helpers import write_polylines_with_segment_id
from zarr_vectors_tools.multiresolution import coarsen as coarsen_module
from zarr_vectors_tools.multiresolution.coarsen import build_pyramid

CHUNK = (10.0, 10.0, 10.0)


def _wiggly(n: int = 6, points: int = 80) -> list[np.ndarray]:
    out = []
    for i in range(n):
        t = np.linspace(0.0, 1.0, points)
        out.append(np.column_stack([
            t * 28.0,
            5.0 + 3.0 * np.sin(t * 14.0 + i),
            5.0 + 2.0 * np.cos(t * 11.0 + i),
        ]).astype(np.float32))
    return out


# =====================================================================
# D01 - the finalize pass must not read what it cannot use
# =====================================================================


class TestCrossLevelFinalize:

    def test_default_depth_reads_no_level(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """At depth 1 there is nothing to compose, so nothing to read.

        ``_reconstruct_chunk_assignments`` allocates one int64 per vertex per
        level.  Running it before checking the depth made a whole-brain
        build's peak memory land in a pass whose output was discarded.
        """
        store = tmp_path / "s.zv"
        write_polylines_with_segment_id(
            str(store), _wiggly(), chunk_shape=CHUNK,
            geometry_type="streamline",
        )
        calls: list[int] = []
        original = coarsen_module._reconstruct_chunk_assignments

        def spy(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(
            coarsen_module, "_reconstruct_chunk_assignments", spy,
        )
        build_pyramid(
            str(store), factors=[(2.0, 1.0), (2.0, 1.0)],
            coarsen_mode="decimate",
        )  # default cross_level_depth=1
        assert calls == [], (
            f"{len(calls)} level(s) reconstructed for a depth that composes "
            f"nothing"
        )

    def test_a_depth_that_composes_still_runs(self, tmp_path: Path) -> None:
        """The early return must not disable the feature it guards."""
        store = tmp_path / "p.zv"
        rng = np.random.default_rng(0)
        write_points(
            str(store), (rng.random((300, 3)) * 40).astype(np.float32),
            chunk_shape=CHUNK, bin_shape=(1.0, 1.0, 1.0),
            object_ids=np.repeat(np.arange(30), 10),
        )
        summary = build_pyramid(
            str(store), factors=[(2.0, 1.0), (2.0, 1.0)],
            cross_level_depth=2, cross_level_storage="explicit",
        )
        assert summary["cross_level_depth"] == 2
        root = open_store(str(store))
        assert read_root_metadata(root).cross_level_depth == 2


# =====================================================================
# D02 - the RDP tolerance must compound with the factor
# =====================================================================


class TestRdpCompounding:

    def _level_counts(self, store: Path, levels: int) -> list[int]:
        root = open_store(str(store))
        return [
            int(read_level_metadata(root, lvl).vertex_count)
            for lvl in range(levels)
        ]

    def test_each_level_is_coarser_than_the_one_below(
        self, tmp_path: Path,
    ) -> None:
        """``--coarsen 8,8`` produced a level 2 identical to level 1.

        The tolerance was half the source CHUNK edge times the factor, and
        the chunk edge does not change unless chunk_scale_factor does, so
        every level simplified at the same tolerance and the second pass had
        nothing left to remove (measured 360 -> 42 -> 42 vertices).
        """
        # A real bin, not the chunk-sized default: with a 10-unit bin the
        # first level's tolerance already flattens every fragment to its two
        # endpoints, and no later level can go below that floor whatever the
        # tolerance is.  The floor is the chunk split, not the simplifier.
        store = tmp_path / "rdp.zv"
        write_polylines_with_segment_id(
            str(store), _wiggly(points=200), chunk_shape=CHUNK,
            bin_shape=(1.0, 1.0, 1.0), geometry_type="streamline",
        )
        summary = build_pyramid(
            str(store), factors=[(2.0, 1.0), (2.0, 1.0), (2.0, 1.0)],
            coarsen_mode="rdp",
            cross_level_depth=0, cross_level_storage="none",
        )
        tolerances = [
            spec.get("simplify_epsilon") for spec in summary["level_specs"]
        ]
        assert tolerances == [1.0, 2.0, 4.0], (
            f"tolerances {tolerances} do not double with the factor"
        )
        counts = self._level_counts(store, 4)
        assert counts[1] < counts[0]
        assert counts[2] < counts[1], (
            f"level 2 ({counts[2]}) did not coarsen past level 1 "
            f"({counts[1]}): the tolerance did not compound"
        )
        assert counts[3] < counts[2]

    def test_factor_one_still_changes_nothing(self, tmp_path: Path) -> None:
        store = tmp_path / "identity.zv"
        write_polylines_with_segment_id(
            str(store), _wiggly(), chunk_shape=CHUNK,
            bin_shape=(1.0, 1.0, 1.0), geometry_type="streamline",
        )
        build_pyramid(
            str(store), factors=[(1.0, 1.0)], coarsen_mode="rdp",
            cross_level_depth=0, cross_level_storage="none",
        )
        counts = self._level_counts(store, 2)
        assert counts[1] == counts[0]


# =====================================================================
# D06 - skeleton decimation agrees with every other coarsener
# =====================================================================


class TestSkeletonStrideSemantics:

    def _chain(self, n: int = 12):
        positions = np.column_stack([
            np.arange(n, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
        ])
        edges = np.stack(
            [np.arange(1, n), np.arange(0, n - 1)], axis=1,
        ).astype(np.int64)
        return positions, edges

    def test_stride_one_is_the_identity(self) -> None:
        """It used to keep anchors only, the most aggressive setting.

        ``coarsen_level`` documents a factor of 1.0 as "no aggregation", and
        a refresh that could not recover the stride fell back to exactly
        this value.
        """
        from zarr_vectors_tools.multiresolution.strategies.skeletons import (
            decimate_skeleton,
        )

        positions, edges = self._chain()
        result = decimate_skeleton(positions, edges, stride=1)
        assert len(result["positions"]) == len(positions)
        np.testing.assert_allclose(result["positions"], positions)

    def test_stride_above_one_still_decimates(self) -> None:
        from zarr_vectors_tools.multiresolution.strategies.skeletons import (
            decimate_skeleton,
        )

        positions, edges = self._chain()
        result = decimate_skeleton(positions, edges, stride=4)
        assert len(result["positions"]) < len(positions)
        # Endpoints are anchors and always survive.
        np.testing.assert_allclose(result["positions"][0], positions[0])
        np.testing.assert_allclose(result["positions"][-1], positions[-1])

    def test_requested_chunk_scale_is_honoured(self, tmp_path: Path) -> None:
        """A requested scale of 1 was silently replaced by 2."""
        from zarr_vectors_tools.multiresolution.coarsen import _skeleton_coarsener

        source = inspect.getsource(_skeleton_coarsener)
        assert "chunk_scale_factor if chunk_scale_factor != 1 else 2" not in source


# =====================================================================
# D07 - a level must not advertise a sparsity it did not apply
# =====================================================================


class TestQuadricSparsity:

    def test_a_sparsity_factor_is_refused(self, tmp_path: Path) -> None:
        """It stamped ``object_sparsity`` and dropped nothing.

        The level then claimed half its objects were gone while all of them
        were present, and its own summary contradicted the claim.
        """
        store = tmp_path / "m.zv"
        v = np.array([
            [0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        ], dtype=np.float32)
        f = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], np.int64)
        write_mesh(str(store), v, f, chunk_shape=(10.0, 10.0, 10.0))
        with pytest.raises(ValueError, match="sparsity_factor"):
            build_pyramid(
                str(store), factors=[(4.0, 2.0)], method="mesh_decimate",
                cross_level_depth=0, cross_level_storage="none",
            )


# =====================================================================
# D08 - grouping helpers do one pass, not one per group
# =====================================================================


class TestGroupingHelpers:

    def test_metanodes_group_every_vertex_exactly_once(self) -> None:
        from zarr_vectors_tools.multiresolution.metanodes import generate_metanodes

        rng = np.random.default_rng(3)
        positions = (rng.random((400, 3)) * 10.0).astype(np.float64)
        result = generate_metanodes(positions, 2.0)

        children = result["children"]
        counts = result["metanode_counts"]
        assert sum(len(c) for c in children) == len(positions)
        assert sorted(np.concatenate(children).tolist()) == list(
            range(len(positions))
        )
        for members, count, centre in zip(
            children, counts, result["metanode_positions"],
        ):
            assert len(members) == count
            # Members ascend, as the per-bin scan they replace produced.
            assert list(members) == sorted(members)
            np.testing.assert_allclose(
                centre, positions[members].mean(axis=0), rtol=1e-6,
            )

    def test_stratified_group_selection_is_seed_stable(self) -> None:
        from zarr_vectors_tools.multiresolution.object_selection import (
            select_stratified_by_group,
        )

        rng = np.random.default_rng(4)
        labels = rng.integers(0, 40, size=2000)
        first = select_stratified_by_group(labels, 0.25, seed=7)
        second = select_stratified_by_group(labels, 0.25, seed=7)
        np.testing.assert_array_equal(first, second)
        # Every group keeps at least one member, which is the guarantee.
        assert set(np.unique(labels[first])) == set(np.unique(labels))


# =====================================================================
# D04 - the process pool sends its shared payload once per worker
# =====================================================================


class TestProcessPoolSharing:

    def test_shared_payload_reaches_the_worker(self) -> None:
        from zarr_vectors_tools.ingest._parallel import (
            _call_with_shared,
            _set_shared,
        )

        _set_shared({"scale": 3})
        assert _call_with_shared(_double, 5) == 15
        _set_shared(None)


def _double(item, shared=None):
    """Module level so a spawned worker can import it."""
    return item * (shared or {}).get("scale", 1)
