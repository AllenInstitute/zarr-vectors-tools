"""Per-bundle summaries, and the group attributes they are stored as.

The synthetic tractogram has hand-checkable numbers: straight streamlines of
known length, one 3-4-5 zigzag whose tortuosity is exactly 1.25, and a bundle
stored half one way and half the other so endpoint orientation is visible.
A 16-unit chunk grid makes every streamline cross at least one chunk
boundary, so every length is assembled from more than one fragment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from zarr_vectors.building import (
    get_resolution_level,
    open_store,
    read_groupings_attributes,
    write_groupings,
)
from zarr_vectors.constants import GROUP_ATTRIBUTES, GROUPS
from zarr_vectors.types.polylines import read_polylines

from tests._source_helpers import write_polylines_with_segment_id
from zarr_vectors_tools.algorithms.bundles import (
    ATTRIBUTE_PREFIX,
    bundle_summary,
    read_bundle_summary,
)
from zarr_vectors_tools.convert.ingest._polyline_enrichments import (
    compute_endpoints,
    compute_lengths,
)
from zarr_vectors_tools.multiresolution.coarsen import build_pyramid

STREAMLINES = [
    # CST: along z.  0 runs up, 1 is stored running down, 2 is a zigzag.
    np.array([[10, 10, 10], [10, 10, 25], [10, 10, 40]], np.float32),  # length 30
    np.array([[12, 10, 50], [12, 10, 30], [12, 10, 10]], np.float32),  # length 40
    np.array([[14, 10, 10], [14, 13, 14], [14, 10, 18]], np.float32),  # 5 + 5, chord 8
    # AF: along x.  3 runs +x, 4 is stored running -x.
    np.array([[5, 40, 40], [25, 40, 40], [45, 40, 40]], np.float32),  # length 40
    np.array([[50, 42, 40], [35, 42, 40], [20, 42, 40]], np.float32),  # length 30
]
GROUP_MEMBERS = {
    0: [0, 1, 2],
    1: [3, 4],
    2: [0, 1, 2, 3, 4],
    3: [],
    4: [2, 2],  # listed twice, one streamline
}
GROUP_NAMES = ["CST", "AF", "ALL", "EMPTY", "SINGLE"]
COLUMNS = [
    "group_id", "streamline_count",
    "length_mean", "length_std", "length_min", "length_median", "length_max",
    "tortuosity_mean",
    "start_x", "start_y", "start_z", "end_x", "end_y", "end_z",
]
WRITTEN = [
    "streamline_count", "length_mean", "length_std", "length_min", "length_median",
    "length_max", "tortuosity_mean", "start_centroid", "end_centroid",
]


def _store(
    path: Path,
    *,
    named: bool = True,
    object_attributes: dict[str, np.ndarray] | None = None,
    group_attributes: dict[str, np.ndarray] | None = None,
    groups: dict[int, list[int]] | None = GROUP_MEMBERS,
) -> Path:
    write_polylines_with_segment_id(
        path, STREAMLINES,
        chunk_shape=(16.0, 16.0, 16.0),
        bounds=([0.0, 0.0, 0.0], [64.0, 64.0, 64.0]),
        groups=groups,
        object_attributes=object_attributes,
        group_attributes=group_attributes,
    )
    if named and groups:
        level_group = get_resolution_level(open_store(str(path), mode="r+"), 0)
        meta = dict(level_group.read_array_meta(GROUPS))
        meta["group_names"] = GROUP_NAMES
        level_group.write_array_meta(GROUPS, meta)
    return path


def _level(path: Path, level: int):
    return get_resolution_level(open_store(str(path), mode="r"), level)


def _bundle_attributes(path: Path, level: int) -> list[str]:
    try:
        names = _level(path, level)[GROUP_ATTRIBUTES].children()
    except Exception:  # noqa: BLE001 - no group attributes at all
        return []
    return sorted(n for n in names if n.startswith(ATTRIBUTE_PREFIX))


class TestSummaryValues:

    def test_one_row_per_group_indexed_by_name(self, tmp_path: Path) -> None:
        table = bundle_summary(_store(tmp_path / "s.zv"), write=False)
        assert list(table.index) == GROUP_NAMES
        assert table.index.name == "group"
        assert list(table.columns) == COLUMNS
        assert table["group_id"].tolist() == [0, 1, 2, 3, 4]

    def test_counts_and_lengths(self, tmp_path: Path) -> None:
        cst = bundle_summary(_store(tmp_path / "s.zv"), write=False).loc["CST"]
        assert cst["streamline_count"] == 3
        assert cst["length_mean"] == pytest.approx(80.0 / 3.0)
        assert cst["length_std"] == pytest.approx(float(np.std([30.0, 40.0, 10.0])))
        assert cst["length_min"] == pytest.approx(10.0)
        assert cst["length_median"] == pytest.approx(30.0)
        assert cst["length_max"] == pytest.approx(40.0)

    def test_tortuosity_is_averaged_per_streamline(self, tmp_path: Path) -> None:
        table = bundle_summary(_store(tmp_path / "s.zv"), write=False)
        assert table.loc["CST", "tortuosity_mean"] == pytest.approx((1.0 + 1.0 + 1.25) / 3.0)
        assert table.loc["AF", "tortuosity_mean"] == pytest.approx(1.0)

    def test_endpoints_are_oriented_along_the_bundle_axis(self, tmp_path: Path) -> None:
        table = bundle_summary(_store(tmp_path / "s.zv"), write=False)
        assert table.attrs["endpoint_orientation"] == "principal_axis"
        # Streamline 1 is stored top-down; oriented, every CST start is at z=10.
        cst = table.loc["CST"]
        np.testing.assert_allclose(cst[["start_x", "start_y", "start_z"]], [12, 10, 10])
        np.testing.assert_allclose(cst[["end_x", "end_y", "end_z"]], [12, 10, 36])
        # Streamline 4 is stored running -x; oriented, starts are the low-x ends.
        af = table.loc["AF"]
        np.testing.assert_allclose(af[["start_x", "start_y", "start_z"]], [12.5, 41, 40])
        np.testing.assert_allclose(af[["end_x", "end_y", "end_z"]], [47.5, 41, 40])

    def test_endpoints_as_stored_when_orientation_is_off(self, tmp_path: Path) -> None:
        table = bundle_summary(_store(tmp_path / "s.zv"), write=False, orient_endpoints=False)
        assert table.attrs["endpoint_orientation"] == "as_stored"
        cst = table.loc["CST"]
        np.testing.assert_allclose(cst[["start_x", "start_y", "start_z"]], [12, 10, 70 / 3])
        np.testing.assert_allclose(cst[["end_x", "end_y", "end_z"]], [12, 10, 68 / 3])
        # Orientation moves endpoints between the centroids, never lengths.
        oriented = bundle_summary(tmp_path / "s.zv", write=False)
        pd.testing.assert_series_equal(table["length_mean"], oriented["length_mean"])
        pd.testing.assert_series_equal(table["tortuosity_mean"], oriented["tortuosity_mean"])

    def test_a_streamline_counts_in_every_group_it_is_in(self, tmp_path: Path) -> None:
        table = bundle_summary(_store(tmp_path / "s.zv"), write=False)
        everything = table.loc["ALL"]
        assert everything["streamline_count"] == 5
        assert everything["length_mean"] == pytest.approx(30.0)
        assert everything["length_median"] == pytest.approx(30.0)
        assert table["streamline_count"].tolist() == [3, 2, 5, 0, 1]

    def test_a_streamline_listed_twice_in_one_group_counts_once(self, tmp_path: Path) -> None:
        single = bundle_summary(_store(tmp_path / "s.zv"), write=False).loc["SINGLE"]
        assert single["streamline_count"] == 1
        assert single["length_mean"] == pytest.approx(10.0)
        assert single["length_std"] == 0.0
        assert single["tortuosity_mean"] == pytest.approx(1.25)

    def test_an_empty_group_has_no_statistics(self, tmp_path: Path) -> None:
        empty = bundle_summary(_store(tmp_path / "s.zv"), write=False).loc["EMPTY"]
        assert empty["streamline_count"] == 0
        assert empty.drop(["group_id", "streamline_count"]).isna().all()

    def test_batching_does_not_change_the_answer(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "s.zv")
        pd.testing.assert_frame_equal(
            bundle_summary(store, write=False, batch_size=1),
            bundle_summary(store, write=False),
        )

    def test_unnamed_groups_take_the_catalogue_fallback_name(self, tmp_path: Path) -> None:
        import zarr_vectors as zv

        store = _store(tmp_path / "s.zv", named=False)
        table = bundle_summary(store, write=False)
        assert list(table.index) == [f"group_{i}" for i in range(5)]
        # Addressable under the same name through core's catalogue.
        members = zv.open(str(store), mode="r").level(0).groups["group_1"].members
        assert sorted(members.tolist()) == [3, 4]


class TestStoredAttributes:

    @staticmethod
    def _enrichments(length_scale: float = 1.0) -> dict[str, np.ndarray]:
        start, end = compute_endpoints(STREAMLINES)
        return {
            "length": compute_lengths(STREAMLINES) * np.float32(length_scale),
            "start": start,
            "end": end,
        }

    def test_stored_attributes_match_the_geometry(self, tmp_path: Path) -> None:
        measured = bundle_summary(_store(tmp_path / "a.zv"), write=False)
        stored = bundle_summary(
            _store(tmp_path / "b.zv", object_attributes=self._enrichments()), write=False,
        )
        assert measured.attrs["measured_from"] == ["geometry"]
        assert stored.attrs["measured_from"] == ["attributes"]
        pd.testing.assert_frame_equal(measured, stored)

    def test_stored_attributes_are_what_level_zero_reads(self, tmp_path: Path) -> None:
        # A stored length twice the real one shows which source was read.
        store = _store(tmp_path / "s.zv", object_attributes=self._enrichments(2.0))
        stored = bundle_summary(store, write=False)
        assert stored.loc["CST", "length_mean"] == pytest.approx(160.0 / 3.0)
        measured = bundle_summary(store, write=False, use_stored_attributes=False)
        assert measured.attrs["measured_from"] == ["geometry"]
        assert measured.loc["CST", "length_mean"] == pytest.approx(80.0 / 3.0)

    def test_a_partial_set_of_stored_attributes_reads_geometry(self, tmp_path: Path) -> None:
        attributes = self._enrichments(2.0)
        del attributes["end"]
        table = bundle_summary(
            _store(tmp_path / "s.zv", object_attributes=attributes), write=False,
        )
        assert table.attrs["measured_from"] == ["geometry"]
        assert table.loc["CST", "length_mean"] == pytest.approx(80.0 / 3.0)

    def test_a_malformed_stored_attribute_is_refused_by_name(self, tmp_path: Path) -> None:
        attributes = self._enrichments()
        attributes["length"] = np.ones((len(STREAMLINES), 2), np.float32)
        store = _store(tmp_path / "s.zv", object_attributes=attributes)
        with pytest.raises(ValueError, match="'length'.*use_stored_attributes=False"):
            bundle_summary(store, write=False)
        # The way out the message names works.
        bundle_summary(store, write=False, use_stored_attributes=False)


class TestWrittenAttributes:

    def test_rows_are_written_as_prefixed_group_attributes(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "s.zv")
        table = bundle_summary(store)
        assert table.attrs["levels_written"] == [0]
        assert _bundle_attributes(store, 0) == sorted(ATTRIBUTE_PREFIX + c for c in WRITTEN)

        level_group = _level(store, 0)
        counts = np.asarray(read_groupings_attributes(level_group, "bundle_streamline_count"))
        assert counts.dtype == np.int64
        assert counts.tolist() == [3, 2, 5, 0, 1]
        means = np.asarray(read_groupings_attributes(level_group, "bundle_length_mean"))
        np.testing.assert_allclose(means, table["length_mean"].to_numpy())
        starts = np.asarray(read_groupings_attributes(level_group, "bundle_start_centroid"))
        assert starts.shape == (5, 3)
        np.testing.assert_allclose(starts[0], [12, 10, 10])

    def test_read_bundle_summary_returns_the_same_table(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "s.zv")
        written = bundle_summary(store)
        read = read_bundle_summary(store)
        pd.testing.assert_frame_equal(read, written)
        assert read.attrs["source_level"] == 0
        assert read.attrs["endpoint_orientation"] == "principal_axis"

    def test_write_false_leaves_the_store_alone(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "s.zv")
        bundle_summary(store, write=False)
        assert _bundle_attributes(store, 0) == []
        with pytest.raises(ValueError, match="holds no bundle summary"):
            read_bundle_summary(store)

    def test_rerunning_overwrites_the_rows(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "s.zv")
        bundle_summary(store)
        bundle_summary(store, orient_endpoints=False)
        read = read_bundle_summary(store)
        assert read.attrs["endpoint_orientation"] == "as_stored"
        assert read.loc["CST", "start_z"] == pytest.approx(70 / 3)

    def test_ingested_group_attributes_are_left_alone(self, tmp_path: Path) -> None:
        mean_fa = np.array([0.4, 0.6, 0.5, np.nan, 0.3], np.float32)
        store = _store(tmp_path / "s.zv", group_attributes={"mean_fa": mean_fa})
        bundle_summary(store)
        np.testing.assert_array_equal(
            np.asarray(read_groupings_attributes(_level(store, 0), "mean_fa")), mean_fa,
        )

    def test_a_named_trx_bundle_store(self, tmp_path: Path) -> None:
        pytest.importorskip("nibabel")
        trx_memmap = pytest.importorskip("trx.trx_file_memmap")
        from nibabel.streamlines.array_sequence import ArraySequence  # noqa: F401

        from zarr_vectors_tools.convert.ingest.trx import ingest_trx

        positions = np.concatenate(STREAMLINES)
        lengths = np.array([len(s) for s in STREAMLINES], np.uint32)
        trx = trx_memmap.TrxFile(nb_vertices=len(positions), nb_streamlines=len(lengths))
        trx.header["VOXEL_TO_RASMM"] = np.eye(4).tolist()
        trx.header["DIMENSIONS"] = [64, 64, 64]
        trx.streamlines._data = positions
        trx.streamlines._offsets = np.concatenate([[0], np.cumsum(lengths[:-1])]).astype(
            np.uint32
        )
        trx.streamlines._lengths = lengths
        trx.groups = {"CST": np.array([0, 1, 2], np.uint32), "AF": np.array([3, 4], np.uint32)}
        source = tmp_path / "in.trx"
        trx_memmap.save(trx, str(source))
        trx.close()

        store = tmp_path / "s.zv"
        ingest_trx(
            str(source), str(store), (16.0, 16.0, 16.0),
            compute_length=True, compute_endpoints=True,
        )
        table = bundle_summary(store)
        assert table.attrs["measured_from"] == ["attributes"]
        assert sorted(table.index) == ["AF", "CST"]
        assert table.loc["CST", "streamline_count"] == 3
        assert table.loc["CST", "length_mean"] == pytest.approx(80.0 / 3.0)
        assert table.loc["AF", "start_x"] == pytest.approx(12.5)


class TestPyramidLevels:

    def test_level_zero_rows_are_written_to_every_level(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "s.zv")
        build_pyramid(str(store), factors=[(2.0, 1.0)])
        table = bundle_summary(store)
        assert table.attrs["levels_written"] == [0, 1]
        for level in (0, 1):
            assert _bundle_attributes(store, level) == sorted(
                ATTRIBUTE_PREFIX + c for c in WRITTEN
            )
            read = read_bundle_summary(store, level=level)
            pd.testing.assert_frame_equal(read, table)
            assert read.attrs["source_level"] == 0

    def test_a_pyramid_built_afterwards_carries_the_rows(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "s.zv")
        table = bundle_summary(store)
        build_pyramid(str(store), factors=[(2.0, 1.0)])
        read = read_bundle_summary(store, level=1)
        pd.testing.assert_frame_equal(read, table)
        # The build copies rows but not the record of where they came from.
        assert read.attrs["source_level"] is None

    def test_a_sparsified_level_still_reports_the_whole_bundle(self, tmp_path: Path) -> None:
        import zarr_vectors as zv

        store = _store(tmp_path / "s.zv")
        build_pyramid(str(store), factors=[(1.0, 2.0)], sparsity_seed=0)
        kept = len(zv.open(str(store), mode="r").level(1).groups["ALL"].members)
        assert kept < 5
        bundle_summary(store)
        assert read_bundle_summary(store, level=1).loc["ALL", "streamline_count"] == 5
        # Summarised from its own membership, the level says what it holds.
        own = bundle_summary(store, level=1, write=False)
        assert own.loc["ALL", "streamline_count"] == kept

    def test_a_level_can_be_summarised_from_its_own_geometry(self, tmp_path: Path) -> None:
        # Stored lengths double the real ones.  The coarsener carries them up
        # verbatim, so a level-1 summary that used them would report 2x.
        start, end = compute_endpoints(STREAMLINES)
        attributes = {"length": compute_lengths(STREAMLINES) * 2, "start": start, "end": end}
        store = _store(tmp_path / "s.zv", object_attributes=attributes)
        build_pyramid(str(store), factors=[(2.0, 1.0)])

        table = bundle_summary(store, level=1)
        assert table.attrs["measured_from"] == ["geometry"]
        assert table.attrs["levels_written"] == [1]
        assert _bundle_attributes(store, 0) == []

        level_one = read_polylines(str(store), level=1)
        paths = {
            int(oid): np.concatenate(fragments)
            for oid, fragments in zip(level_one["object_ids"], level_one["polylines"])
        }
        cst = compute_lengths([paths[0], paths[1], paths[2]])
        assert table.loc["CST", "length_mean"] == pytest.approx(float(np.mean(cst)))
        read = read_bundle_summary(store, level=1)
        assert read.attrs["source_level"] == 1
        pd.testing.assert_frame_equal(read, table)


class TestRefusals:

    def test_a_store_without_groups(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "s.zv", groups=None)
        with pytest.raises(ValueError, match="no object groups"):
            bundle_summary(store)
        with pytest.raises(ValueError, match="no object groups"):
            read_bundle_summary(store)

    def test_a_store_that_is_not_streamlines(self, tmp_path: Path) -> None:
        from zarr_vectors.types.points import write_points

        store = tmp_path / "p.zv"
        write_points(
            str(store), np.random.default_rng(0).uniform(0, 10, (20, 3)).astype(np.float32),
            chunk_shape=(5.0, 5.0, 5.0),
        )
        with pytest.raises(ValueError, match="streamline and polyline"):
            bundle_summary(store)

    def test_a_level_the_store_does_not_have(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "s.zv")
        with pytest.raises(ValueError, match="has no level 3"):
            bundle_summary(store, level=3)
        with pytest.raises(ValueError, match="must be an integer"):
            bundle_summary(store, level=1.5)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="has no level 2"):
            read_bundle_summary(store, level=2)

    def test_a_batch_size_below_one(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            bundle_summary(_store(tmp_path / "s.zv"), batch_size=0)

    def test_a_group_naming_an_object_the_level_does_not_hold(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "s.zv")
        write_groupings(
            get_resolution_level(open_store(str(store), mode="r+"), 0),
            {0: [0, 1, 99]},
        )
        with pytest.raises(ValueError, match=r"\[99\]"):
            bundle_summary(store, write=False)

    def test_a_group_naming_an_object_with_no_geometry_there(self, tmp_path: Path) -> None:
        import zarr_vectors as zv

        store = _store(tmp_path / "s.zv")
        build_pyramid(str(store), factors=[(1.0, 2.0)], sparsity_seed=0)
        kept = set(zv.open(str(store), mode="r").level(1).groups["ALL"].members.tolist())
        dropped = sorted(set(range(5)) - kept)
        # The object index still has a slot for a thinned-out streamline, but
        # no fragments; a group claiming it as a member is refused, not skipped.
        write_groupings(
            get_resolution_level(open_store(str(store), mode="r+"), 1),
            {0: [*sorted(kept), dropped[0]]},
        )
        with pytest.raises(ValueError, match=f"no geometry at level 1: \\[{dropped[0]}\\]"):
            bundle_summary(store, level=1, write=False)

    def test_reading_a_summary_its_groups_have_outgrown(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "s.zv")
        bundle_summary(store)
        write_groupings(
            get_resolution_level(open_store(str(store), mode="r+"), 0),
            {0: [0], 1: [1]},
        )
        with pytest.raises(ValueError, match="stale"):
            read_bundle_summary(store)

    def test_writing_to_a_level_without_the_groups(self, tmp_path: Path) -> None:
        store = _store(tmp_path / "s.zv")
        build_pyramid(str(store), factors=[(2.0, 1.0)])
        get_resolution_level(open_store(str(store), mode="r+"), 1).delete_subtree(GROUPS)
        with pytest.raises(ValueError, match="level 1 .* has no groups"):
            bundle_summary(store)
        # Refused before anything was written, even to the level that could take it.
        assert _bundle_attributes(store, 0) == []
        # Either way out the message names works.
        assert bundle_summary(store, write=False).loc["CST", "streamline_count"] == 3
        bundle_summary(store, level=0)
        assert _bundle_attributes(store, 0) != []
