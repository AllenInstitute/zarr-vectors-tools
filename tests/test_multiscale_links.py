"""End-to-end coverage for the merged links layout.

Connectivity is ONE family under ``links/<delta>/<offsets>/``: an
intra-chunk link is just one whose offsets are all zero.  Covers:

- ``links/<delta>/<offsets>`` round-trips, intra and chunk-spanning.
- ``links/<delta>`` being a GROUP, not an array — the distinction several
  readers fail *silently* on.
- The whole-family / cross-only split, which is where a port from the
  pre-merge two-family API double-counts.
- ``link_attributes`` writer/reader, and the enumeration order that is the
  only thing aligning attribute rows to link records.
- Family policy being family-wide and un-flippable.
- ``build_pyramid`` cross-level emission across depth / storage modes.
- The hard schema-version cutoff.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zarr_vectors.constants import (
    CAP_MULTISCALE_LINKS,
    FORMAT_VERSION,
    LINKS,
    XLEVEL_EXPLICIT,
    XLEVEL_IMPLICIT,
    XLEVEL_NONE,
)
from zarr_vectors.core.arrays import (
    create_link_attributes_array,
    create_links_array,
    create_links_family,
    create_vertices_array,
    link_family_policy,
    list_link_deltas,
    list_link_offsets,
    read_chunk_links,
    read_link_attributes,
    read_links,
    write_chunk_links,
    write_chunk_vertices,
    write_link_attributes,
    write_link_cells,
    write_links,
)
from zarr_vectors.core.metadata import RootMetadata
from zarr_vectors.core.paths import (
    link_attributes_path,
    links_group_path,
    links_path,
)
from zarr_vectors.core.store import create_store, get_resolution_level
from zarr_vectors.exceptions import ArrayError, MetadataError

from zarr_vectors_tools.algorithms._links import (
    link_prefetch_plan,
    list_link_cells,
    read_cross_links,
)

INTRA_3D = ((0, 0, 0),)


def _make_level_group(tmp_path: Path):
    """A real resolution level.

    Per-chunk arrays are single multidim zarr arrays shaped to the level's
    chunk grid, so they can only be allocated against a level that knows
    that grid — a bare group is not enough.
    """
    root = create_store(
        str(tmp_path / "store.zv"),
        bounds=([0.0, 0.0, 0.0], [1000.0, 1000.0, 1000.0]),
        chunk_shape=(100.0, 100.0, 100.0),
        geometry_types=["graph"],
        ndim=3,
    )
    return get_resolution_level(root, 0)


# ===================================================================
# delta=0 round-trips
# ===================================================================

class TestDeltaZero:

    def test_intra_chunk_links_round_trip(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        create_links_array(lg, link_width=2, sid_ndim=3)
        write_chunk_vertices(lg, (0, 0, 0), [np.zeros((4, 3), dtype=np.float32)])
        edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
        write_chunk_links(lg, (0, 0, 0), [edges])

        groups = read_chunk_links(lg, (0, 0, 0), link_width=2, delta=0)
        assert len(groups) == 1
        np.testing.assert_array_equal(groups[0], edges)

    def test_chunk_spanning_links_round_trip(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        links = [
            (((0, 0, 0), 4), ((0, 0, 1), 0)),
            (((0, 0, 0), 2), ((1, 0, 0), 1)),
        ]
        write_links(lg, links, sid_ndim=3)
        # read_links enumerates by (offsets segment, cell), NOT input order:
        # placement is a function of the record, so input order is not
        # preserved.  Each record round-trips in its own input endpoint
        # order, which is what perm_idx exists to restore.
        assert set(read_links(lg)) == set(links)

    def test_links_delta_is_a_group_not_an_array(self, tmp_path: Path) -> None:
        """``links/<delta>`` holds one array per offsets segment.

        Naming the group where an array is expected fails silently —
        ``list_chunks`` returns [] and the batch reader skips it — so pin
        the distinction explicitly.
        """
        lg = _make_level_group(tmp_path)
        create_links_array(lg, link_width=2, delta=0, sid_ndim=3)
        family = links_group_path(0)

        assert lg.array_exists(family)
        assert not lg.standalone_array_exists(family)
        assert lg.list_chunks(family) == []
        # The array is one level deeper, under its offsets segment.
        assert lg.standalone_array_exists(links_path(0, INTRA_3D))


# ===================================================================
# The whole-family / cross-only split
# ===================================================================

class TestFamilyVsCross:

    def _mixed_store(self, tmp_path: Path):
        from zarr_vectors.core.store import get_resolution_level, open_store
        from zarr_vectors.types.graphs import write_graph

        # 3 intra edges inside chunk (0,0,0); 2 edges spanning to (2,0,0).
        positions = np.array(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [20, 0, 0], [21, 0, 0]],
            dtype=np.float32,
        )
        edges = np.array(
            [[0, 1], [1, 2], [2, 3], [0, 4], [1, 5]], dtype=np.int64,
        )
        store = tmp_path / "mixed.zv"
        write_graph(str(store), positions, edges, chunk_shape=(10.0, 10.0, 10.0))
        return get_resolution_level(open_store(str(store)), 0)

    def test_read_links_returns_whole_family(self, tmp_path: Path) -> None:
        lg = self._mixed_store(tmp_path)
        assert len(read_links(lg, delta=0)) == 5

    def test_read_cross_links_returns_spanning_only(self, tmp_path: Path) -> None:
        lg = self._mixed_store(tmp_path)
        cross = read_cross_links(lg, delta=0)
        assert len(cross) == 2
        for record in cross:
            assert len({tuple(cc) for cc, _vi in record}) > 1

    def test_intra_is_family_minus_cross(self, tmp_path: Path) -> None:
        """The regression guard for the double-count trap.

        Pre-merge, read_links meant intra-only and had a sibling cross
        reader.  Anyone porting that idiom by unioning a per-chunk
        read_chunk_links loop with read_links counts every intra edge
        twice — silently, with no error.
        """
        lg = self._mixed_store(tmp_path)
        family = read_links(lg, delta=0)
        cross = read_cross_links(lg, delta=0)
        intra = [r for r in family if len({tuple(cc) for cc, _vi in r}) == 1]
        assert len(intra) == 3
        assert len(intra) + len(cross) == len(family)

    def test_list_link_cells_lists_spanning_cells(self, tmp_path: Path) -> None:
        lg = self._mixed_store(tmp_path)
        cells = list_link_cells(lg, delta=0)
        assert cells == [((0, 0, 0), (2, 0, 0))]
        assert list_link_cells(lg, delta=0, involves=(2, 0, 0)) == cells
        assert list_link_cells(lg, delta=0, involves=(9, 9, 9)) == []

    def test_prefetch_plan_names_arrays_not_groups(self, tmp_path: Path) -> None:
        """Every plan entry must resolve to a real array.

        The batch reader skips any plan node that is not an array, so a
        group path here would silently disable prefetch rather than fail.
        """
        lg = self._mixed_store(tmp_path)
        plan = link_prefetch_plan(lg, [(0, 0, 0), (2, 0, 0)])
        assert plan, "plan must cover the link family"
        for path, _keys in plan:
            assert lg.standalone_array_exists(path), f"{path} is not an array"


# ===================================================================
# delta != 0 — cross-pyramid-level links
# ===================================================================

class TestCrossLevelManual:

    def test_links_plus_one(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        # Source vertices at this level; target side conceptually lives
        # at this_level + 1.  Per-chunk single edge group.
        create_links_array(lg, link_width=2, delta=1, sid_ndim=3)
        edges = np.array([[0, 0], [1, 0], [2, 1]], dtype=np.int64)
        write_chunk_links(lg, (0, 0, 0), [edges], delta=1)
        back = read_chunk_links(lg, (0, 0, 0), link_width=2, delta=1)
        assert len(back) == 1
        np.testing.assert_array_equal(back[0], edges)

    def test_spanning_minus_two(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        links = [
            (((1, 0, 0), 0), ((0, 0, 0), 3)),
            (((1, 0, 0), 1), ((0, 0, 1), 7)),
        ]
        write_links(lg, links, sid_ndim=3, delta=-2)
        assert set(read_links(lg, delta=-2)) == set(links)

    def test_listing_helpers(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_links_array(lg, link_width=2, delta=0, sid_ndim=3)
        create_links_array(lg, link_width=2, delta=1, sid_ndim=3)
        create_links_array(lg, link_width=2, delta=-1, sid_ndim=3)
        assert list_link_deltas(lg) == [-1, 0, 1]
        # One array per offsets segment; only the intra one exists so far.
        assert list_link_offsets(lg, 0) == ["0.0.0"]


# ===================================================================
# Family policy is family-wide and un-flippable
# ===================================================================

class TestFamilyPolicy:

    def test_policy_stamped_on_the_group(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_links_family(
            lg, delta=0, link_width=2, sid_ndim=3, directed=True,
        )
        assert link_family_policy(lg, 0) == (2, 3, True, "canonical")

    def test_family_can_be_stamped_without_any_array(self, tmp_path: Path) -> None:
        """A cross-only family must not be forced to materialise an
        intra array purely to carry the group stamp."""
        lg = _make_level_group(tmp_path)
        create_links_family(lg, delta=0, link_width=2, sid_ndim=3)
        assert lg.array_exists(links_group_path(0))
        assert list_link_offsets(lg, 0) == []

    def test_cannot_flip_directed_under_surviving_siblings(
        self, tmp_path: Path,
    ) -> None:
        """Policy is one per (level, delta): every offsets array decodes
        against it, so flipping it would strand the ones already written."""
        lg = _make_level_group(tmp_path)
        create_links_family(
            lg, delta=0, link_width=2, sid_ndim=3, directed=False,
        )
        links = [(((0, 0, 0), 0), ((1, 0, 0), 1))]
        with pytest.raises(ArrayError):
            write_link_cells(
                lg, links, 3, delta=0, link_width=2, directed=True,
            )


# ===================================================================
# Link attributes
# ===================================================================

class TestLinkAttributes:

    LINKS_3 = [
        (((0, 0, 0), 4), ((0, 0, 1), 0)),
        (((0, 0, 0), 2), ((1, 0, 0), 1)),
        (((1, 0, 0), 5), ((1, 0, 1), 0)),
    ]

    def test_round_trip_with_partition(self, tmp_path: Path) -> None:
        """With the writer's partition, attr_data is in INPUT order."""
        lg = _make_level_group(tmp_path)
        partition = write_links(lg, self.LINKS_3, sid_ndim=3, delta=1)

        weights = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        write_link_attributes(
            lg, "weight", weights, num_links=3, delta=1, partition=partition,
        )

        # Each record keeps the weight it was given, whatever order the
        # family enumerates in.
        by_record = dict(zip(read_links(lg, delta=1),
                             read_link_attributes(lg, "weight", delta=1)))
        for record, expected in zip(self.LINKS_3, weights):
            assert by_record[record] == pytest.approx(expected)

    def test_round_trip_without_partition_is_enumeration_order(
        self, tmp_path: Path,
    ) -> None:
        """Without a partition, attr_data must ALREADY be in read_links order.

        The writer can only re-derive on-disk order by reading the records
        back, so it cannot map input positions for you.  Handing it input
        order here silently mis-assigns every row — placement is a function
        of the record, so input order is not enumeration order.
        """
        lg = _make_level_group(tmp_path)
        write_links(lg, self.LINKS_3, sid_ndim=3, delta=1)

        records = read_links(lg, delta=1)
        assert list(records) != list(self.LINKS_3), (
            "fixture must exercise a reordering, else the test proves nothing"
        )
        wanted = {r: float(i) for i, r in enumerate(self.LINKS_3)}
        rows = np.array([wanted[r] for r in records], dtype=np.float32)
        write_link_attributes(lg, "weight", rows, num_links=3, delta=1)

        by_record = dict(zip(read_links(lg, delta=1),
                             read_link_attributes(lg, "weight", delta=1)))
        for record, expected in wanted.items():
            assert by_record[record] == pytest.approx(expected)

    def test_length_invariant_enforced(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        bad = np.array([1.0, 2.0], dtype=np.float32)
        with pytest.raises(ArrayError):
            write_link_attributes(lg, "weight", bad, num_links=3, delta=0)

    def test_path_layout(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_link_attributes_array(
            lg, "weight", dtype="float32", delta=0, sid_ndim=3, link_width=2,
        )
        assert lg.standalone_array_exists(
            link_attributes_path("weight", 0, INTRA_3D)
        )


# ===================================================================
# Path helpers wired against on-disk truth
# ===================================================================

def test_paths_module_matches_disk_layout(tmp_path: Path) -> None:
    lg = _make_level_group(tmp_path)
    create_links_array(lg, link_width=2, delta=2, sid_ndim=3)
    create_link_attributes_array(
        lg, "w", dtype="float32", delta=-1, sid_ndim=3, link_width=2,
    )

    assert lg.standalone_array_exists(links_path(2, INTRA_3D))
    assert lg.standalone_array_exists(link_attributes_path("w", -1, INTRA_3D))


# ===================================================================
# Schema version cutoff
# ===================================================================

def _minimal_root_md(**overrides):
    base = dict(
        spatial_index_dims=[
            {"name": "x", "type": "space", "unit": "um"},
            {"name": "y", "type": "space", "unit": "um"},
            {"name": "z", "type": "space", "unit": "um"},
        ],
        chunk_shape=(50.0, 50.0, 50.0),
        bounds=([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]),
        geometry_types=["point_cloud"],
    )
    base.update(overrides)
    return RootMetadata(**base)


def test_default_zv_version_is_09():
    md = _minimal_root_md()
    md.validate()
    assert md.zv_version == FORMAT_VERSION
    assert FORMAT_VERSION.startswith("0.9")


def test_pre_05_zv_version_rejected():
    md = _minimal_root_md(zv_version="0.4.1")
    with pytest.raises(MetadataError, match="0.4.1"):
        md.validate()


def test_pre_05_zv_version_rejected_on_roundtrip():
    """A wire-format dict with zv_version='0.4.1' must round-trip into
    a RootMetadata that fails validate() — proving the cutoff catches
    both freshly-constructed and freshly-loaded stores."""
    md = _minimal_root_md(zv_version="0.4.1")
    md.zv_version = "0.4.1"
    d = md.to_dict()
    # 0.5.0 from_dict reads axes from the NGFF multiscales block; inject
    # a minimal one so the round trip can complete.
    d["multiscales"] = [{
        "version": "0.4",
        "axes": list(md.spatial_index_dims),
        "datasets": [{"path": "0", "coordinateTransformations": [
            {"type": "scale", "scale": [1.0] * md.sid_ndim},
        ]}],
        "metadata": {"format": "zarr_vectors"},
    }]
    reloaded = RootMetadata.from_dict(d)
    with pytest.raises(MetadataError):
        reloaded.validate()


def test_invalid_cross_level_storage_rejected():
    md = _minimal_root_md()
    md.cross_level_storage = "always"
    with pytest.raises(MetadataError, match="cross_level_storage"):
        md.validate()


def test_invalid_cross_level_depth_rejected():
    md = _minimal_root_md()
    md.cross_level_depth = -2
    with pytest.raises(MetadataError, match="cross_level_depth"):
        md.validate()


# ===================================================================
# build_pyramid integration (cross-level emission)
# ===================================================================

# These integration tests exercise the full graph write → pyramid build
# round-trip.  They depend on the zarr backend; xfail under environments
# where zarr can't import (e.g. a Python build without a numcodecs
# wheel) so the rest of the suite keeps running.
zarr = pytest.importorskip("zarr")

from zarr_vectors.core.store import (  # noqa: E402
    get_resolution_level,
    list_resolution_levels,
    open_store,
    read_root_metadata,
)
from zarr_vectors_tools.multiresolution.coarsen import build_pyramid  # noqa: E402
from zarr_vectors.types.graphs import write_graph  # noqa: E402


def _seed_simple_graph(tmp_path: Path) -> Path:
    """Write a small 3D graph store usable by build_pyramid."""
    store_path = tmp_path / "graph.zarr"
    rng = np.random.default_rng(0)
    n = 64
    positions = rng.uniform(0.0, 100.0, size=(n, 3)).astype(np.float32)
    # Trivial spanning tree edges so the graph is connected enough to
    # exercise both intra- and chunk-spanning links.
    edges = np.stack(
        [np.arange(n - 1, dtype=np.int64), np.arange(1, n, dtype=np.int64)],
        axis=1,
    )
    object_ids = np.zeros(n, dtype=np.int64)
    write_graph(
        store_path,
        positions=positions,
        edges=edges,
        object_ids=object_ids,
        chunk_shape=(40.0, 40.0, 40.0),
        bounds=([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]),
    )
    return store_path


def _delta_dirs(root, level: int, prefix: str = LINKS) -> set[str]:
    lg = get_resolution_level(root, level)
    if not lg.array_exists(prefix):
        return set()
    return {name for name in lg[prefix]}


def test_build_pyramid_depth_zero_emits_no_cross_level(tmp_path: Path) -> None:
    store_path = _seed_simple_graph(tmp_path)
    build_pyramid(
        store_path,
        factors=[(2.0, 1.0), (2.0, 1.0)],
        cross_level_depth=0,
        cross_level_storage=XLEVEL_NONE,
    )
    root = open_store(str(store_path))
    for lvl in list_resolution_levels(root):
        # Only delta=0 (or nothing at all) should exist.
        assert _delta_dirs(root, lvl) <= {"0"}


def test_build_pyramid_depth_zero_omits_multiscale_capability(
    tmp_path: Path,
) -> None:
    """The token marks delta != 0 arrays only.

    Cross-CHUNK links are just a non-zero offsets segment since the merge,
    so a store with plenty of them but no cross-LEVEL arrays must not
    claim it.
    """
    store_path = _seed_simple_graph(tmp_path)
    build_pyramid(
        store_path,
        factors=[(2.0, 1.0), (2.0, 1.0)],
        cross_level_depth=0,
        cross_level_storage=XLEVEL_NONE,
    )
    root = open_store(str(store_path))
    rmeta = read_root_metadata(root)
    assert CAP_MULTISCALE_LINKS not in rmeta.format_capabilities


def test_build_pyramid_explicit_depth_one(tmp_path: Path) -> None:
    store_path = _seed_simple_graph(tmp_path)
    build_pyramid(
        store_path,
        factors=[(2.0, 1.0), (2.0, 1.0)],
        cross_level_depth=1,
        cross_level_storage=XLEVEL_EXPLICIT,
    )
    root = open_store(str(store_path))
    levels = sorted(list_resolution_levels(root))
    assert len(levels) >= 2, "pyramid should have produced at least one coarser level"

    # Root metadata reflects choices + capability is stamped.
    rmeta = read_root_metadata(root)
    assert rmeta.cross_level_depth == 1
    assert rmeta.cross_level_storage == XLEVEL_EXPLICIT
    assert CAP_MULTISCALE_LINKS in rmeta.format_capabilities

    # Fine levels carry +1, coarser levels carry -1.
    has_plus = any("+1" in _delta_dirs(root, lvl) for lvl in levels[:-1])
    has_minus = any("-1" in _delta_dirs(root, lvl) for lvl in levels[1:])
    assert has_plus, "explicit mode must materialize +1 at the finer level"
    assert has_minus, "explicit mode must materialize -1 at the coarser level"


def test_build_pyramid_implicit_only_plus(tmp_path: Path) -> None:
    store_path = _seed_simple_graph(tmp_path)
    build_pyramid(
        store_path,
        factors=[(2.0, 1.0), (2.0, 1.0)],
        cross_level_depth=1,
        cross_level_storage=XLEVEL_IMPLICIT,
    )
    root = open_store(str(store_path))
    # No -N arrays anywhere.
    for lvl in list_resolution_levels(root):
        deltas = _delta_dirs(root, lvl)
        assert all(not d.startswith("-") for d in deltas), \
            f"implicit mode should not emit negative links/<delta> at level {lvl}"


def test_build_pyramid_stamps_num_links(tmp_path: Path) -> None:
    """finalize_links must run, run last, and run unconditionally.

    EVERY family must carry num_links, including one whose records are all
    chunk-aligned: those are written by the per-cell write_chunk_links,
    which maintains no family-wide counts, so only a finalize pass supplies
    them.  Do not skip a family that lacks the key — that is the bug.

    num_links matching the record count also proves every cell it counted is
    discoverable through nonempty_chunks, and that nothing restamped the
    family meta afterwards and dropped the counts.
    """
    store_path = _seed_simple_graph(tmp_path)
    # depth=1 so the inline ±1 cross-level emitter runs; its aligned-only
    # deltas are exactly the case a per-cell writer leaves uncounted.
    build_pyramid(
        store_path,
        factors=[(2.0, 1.0), (2.0, 1.0)],
        cross_level_depth=1,
        cross_level_storage=XLEVEL_EXPLICIT,
    )
    root = open_store(str(store_path))
    seen = 0
    for lvl in list_resolution_levels(root):
        lg = get_resolution_level(root, lvl)
        for delta in list_link_deltas(lg):
            meta = lg.read_array_meta(links_group_path(delta))
            assert "num_links" in meta, (
                f"level {lvl} delta {delta}: finalize_links never stamped "
                f"num_links (segments={list_link_offsets(lg, delta)})"
            )
            assert meta["num_links"] == len(read_links(lg, delta=delta)), (
                f"level {lvl} delta {delta}: num_links disagrees with reader"
            )
            seen += 1
    assert seen >= 3, "fixture must produce delta=0 and cross-level families"


def test_no_cross_chunk_links_directory_anywhere(tmp_path: Path) -> None:
    """The merged layout has no second family; nothing may recreate one."""
    store_path = _seed_simple_graph(tmp_path)
    build_pyramid(store_path, factors=[(2.0, 1.0)])
    assert not list(Path(store_path).rglob("cross_chunk_links"))
