"""End-to-end coverage for the 0.4 multiscale links layout.

Covers:
- ``links/0/<chunk>`` regression (delta=0 writer/reader behavior).
- Manual round-trip of ``links/+1/<chunk>`` and ``cross_chunk_links/+1/data``.
- The new ``cross_chunk_link_attributes`` writer/reader pair.
- ``build_pyramid`` cross-level emission across depth / storage modes.
- The hard schema-version cutoff at 0.4.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zarr_vectors.constants import (
    CAP_MULTISCALE_LINKS,
    CROSS_CHUNK_LINK_ATTRIBUTES,
    CROSS_CHUNK_LINKS,
    FORMAT_VERSION,
    LINKS,
    XLEVEL_EXPLICIT,
    XLEVEL_IMPLICIT,
    XLEVEL_NONE,
)
from zarr_vectors.core.arrays import (
    create_cross_chunk_link_attributes_array,
    create_cross_chunk_links_array,
    create_link_attributes_array,
    create_links_array,
    create_vertices_array,
    list_cross_link_deltas,
    list_link_deltas,
    read_chunk_links,
    read_cross_chunk_link_attributes,
    read_cross_chunk_links,
    write_chunk_link_attributes,
    write_chunk_links,
    write_chunk_vertices,
    write_cross_chunk_link_attributes,
    write_cross_chunk_links,
)
from zarr_vectors.core.metadata import RootMetadata
from zarr_vectors.core.paths import (
    cross_chunk_link_attributes_path,
    cross_chunk_links_path,
    link_attributes_path,
    links_path,
)
from zarr_vectors.core.store import FsGroup
from zarr_vectors.exceptions import ArrayError, MetadataError


def _make_level_group(tmp_path: Path, name: str = "0") -> FsGroup:
    root = FsGroup(tmp_path / "store.zarr", create=True)
    return root.create_group(name)


# ===================================================================
# delta=0 regression (current behavior preserved under new path layout)
# ===================================================================

class TestDeltaZero:

    def test_intra_chunk_links_round_trip(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_vertices_array(lg)
        create_links_array(lg, link_width=2)
        write_chunk_vertices(lg, (0, 0, 0), [np.zeros((4, 3), dtype=np.float32)])
        edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
        write_chunk_links(lg, (0, 0, 0), [edges])

        groups = read_chunk_links(lg, (0, 0, 0), link_width=2, delta=0)
        assert len(groups) == 1
        np.testing.assert_array_equal(groups[0], edges)

    def test_cross_chunk_links_round_trip(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_cross_chunk_links_array(lg)
        links = [
            (((0, 0, 0), 4), ((0, 0, 1), 0)),
            (((0, 0, 0), 2), ((1, 0, 0), 1)),
        ]
        write_cross_chunk_links(lg, links, sid_ndim=3)
        assert read_cross_chunk_links(lg) == links

    def test_paths_have_delta_segment(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_links_array(lg, link_width=2, delta=0)
        create_cross_chunk_links_array(lg, delta=0)
        # The on-disk layout now puts <delta> between the array prefix
        # and the per-chunk subpath / data blob.
        assert lg.array_exists(f"{LINKS}/0")
        assert lg.array_exists(f"{CROSS_CHUNK_LINKS}/0")


# ===================================================================
# delta != 0 — cross-pyramid-level links (manual writer/reader)
# ===================================================================

class TestCrossLevelManual:

    def test_links_plus_one(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        # Source vertices at this level; target side conceptually lives
        # at this_level + 1.  Per-chunk single edge group.
        create_links_array(lg, link_width=2, delta=1)
        edges = np.array([[0, 0], [1, 0], [2, 1]], dtype=np.int64)
        write_chunk_links(lg, (0, 0, 0), [edges], delta=1)
        back = read_chunk_links(lg, (0, 0, 0), link_width=2, delta=1)
        assert len(back) == 1
        np.testing.assert_array_equal(back[0], edges)

    def test_ccl_minus_two(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_cross_chunk_links_array(lg, delta=-2)
        links = [
            (((1, 0, 0), 0), ((0, 0, 0), 3)),
            (((1, 0, 0), 1), ((0, 0, 1), 7)),
        ]
        write_cross_chunk_links(lg, links, sid_ndim=3, delta=-2)
        assert read_cross_chunk_links(lg, delta=-2) == links

    def test_listing_helpers(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_links_array(lg, link_width=2, delta=0)
        create_links_array(lg, link_width=2, delta=1)
        create_links_array(lg, link_width=2, delta=-1)
        create_cross_chunk_links_array(lg, delta=0)
        create_cross_chunk_links_array(lg, delta=2)
        assert list_link_deltas(lg) == [-1, 0, 1]
        assert list_cross_link_deltas(lg) == [0, 2]


# ===================================================================
# Cross-chunk link attributes (NEW in 0.4)
# ===================================================================

class TestCrossChunkLinkAttributes:

    def test_round_trip(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_cross_chunk_links_array(lg, delta=1)
        links = [
            (((0, 0, 0), 4), ((0, 0, 1), 0)),
            (((0, 0, 0), 2), ((1, 0, 0), 1)),
            (((1, 0, 0), 5), ((1, 0, 1), 0)),
        ]
        write_cross_chunk_links(lg, links, sid_ndim=3, delta=1)

        create_cross_chunk_link_attributes_array(
            lg, "weight", dtype="float32", delta=1,
        )
        weights = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        write_cross_chunk_link_attributes(
            lg, "weight", weights, num_links=3, delta=1,
        )

        back = read_cross_chunk_link_attributes(lg, "weight", delta=1)
        np.testing.assert_allclose(back, weights)

    def test_length_invariant_enforced(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        bad = np.array([1.0, 2.0], dtype=np.float32)
        with pytest.raises(ArrayError):
            write_cross_chunk_link_attributes(
                lg, "weight", bad, num_links=3, delta=0,
            )

    def test_path_layout(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_cross_chunk_link_attributes_array(
            lg, "weight", dtype="float32", delta=0,
        )
        assert lg.array_exists(
            f"{CROSS_CHUNK_LINK_ATTRIBUTES}/weight/0"
        )


# ===================================================================
# Path helpers wired against on-disk truth
# ===================================================================

def test_paths_module_matches_disk_layout(tmp_path: Path) -> None:
    lg = _make_level_group(tmp_path)
    create_links_array(lg, link_width=2, delta=2)
    create_link_attributes_array(lg, "w", dtype="float32", delta=-1)
    create_cross_chunk_links_array(lg, delta=3)
    create_cross_chunk_link_attributes_array(lg, "w", delta=-2)

    assert lg.array_exists(links_path(2))
    assert lg.array_exists(link_attributes_path("w", -1))
    assert lg.array_exists(cross_chunk_links_path(3))
    assert lg.array_exists(cross_chunk_link_attributes_path("w", -2))


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


def test_default_zv_version_is_08():
    md = _minimal_root_md()
    md.validate()
    assert md.zv_version == FORMAT_VERSION
    assert FORMAT_VERSION.startswith("0.8")


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
    create_store,
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
    # exercise both intra- and cross-chunk links.
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


def _delta_dirs(root, level: int, prefix: str) -> set[str]:
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
    levels = list_resolution_levels(root)
    for lvl in levels:
        # Only delta=0 (or nothing at all) should exist.
        assert _delta_dirs(root, lvl, LINKS) <= {"0"}
        assert _delta_dirs(root, lvl, CROSS_CHUNK_LINKS) <= {"0"}


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
    has_plus = False
    has_minus = False
    for lvl in levels[:-1]:
        if "+1" in _delta_dirs(root, lvl, LINKS) | _delta_dirs(root, lvl, CROSS_CHUNK_LINKS):
            has_plus = True
    for lvl in levels[1:]:
        if "-1" in _delta_dirs(root, lvl, LINKS) | _delta_dirs(root, lvl, CROSS_CHUNK_LINKS):
            has_minus = True
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
    levels = sorted(list_resolution_levels(root))
    # No -N arrays anywhere.
    for lvl in levels:
        link_deltas = _delta_dirs(root, lvl, LINKS)
        ccl_deltas = _delta_dirs(root, lvl, CROSS_CHUNK_LINKS)
        assert all(not d.startswith("-") for d in link_deltas), \
            f"implicit mode should not emit negative links/<delta> at level {lvl}"
        assert all(not d.startswith("-") for d in ccl_deltas), \
            f"implicit mode should not emit negative cross_chunk_links/<delta> at level {lvl}"


# ===================================================================
# v0.8 kN-array layout — shape, direct lookup, canonical ci
# ===================================================================

@pytest.mark.skip(
    reason="v0.8 partitioned kN-array/leaf CCL layout was replaced by the "
    "cells/directed/duplicate layout in core (PR #26); these on-disk-shape "
    "assertions no longer apply. Tools reads/writes CCL via the cells API."
)
class TestShardedLayoutV1:
    """Cover the on-disk shape introduced by the v0.8 kN-array layout.

    Sharding is a recommended writer-side codec choice, not a
    format-level mandate; readers detect the v0.8 layout structurally
    (presence of ``kK`` sub-arrays) rather than via a discriminator
    attribute.
    """

    def test_no_layout_attr_written_on_group(self, tmp_path: Path) -> None:
        # The format no longer stamps a ``layout`` discriminator — the
        # presence of ``kK`` Array children is the structural signal.
        lg = _make_level_group(tmp_path)
        create_cross_chunk_links_array(lg, delta=0)
        write_cross_chunk_links(
            lg, [(((0, 0, 0), 1), ((1, 0, 0), 2))], sid_ndim=3,
        )
        meta = lg.read_array_meta(cross_chunk_links_path(0))
        assert "layout" not in meta

    def test_structural_detection_of_legacy_blob(self, tmp_path: Path) -> None:
        # Synthesize a legacy single-blob layout and confirm the reader
        # raises with a migration hint when it encounters one.
        import zarr
        from zarr_vectors.core.arrays import read_cross_chunk_links
        from zarr_vectors.core.group import Group
        from zarr_vectors.exceptions import ArrayError

        store_path = tmp_path / "legacy.zarrvectors"
        root = zarr.open_group(store=str(store_path), mode="w")
        level = root.require_group("0")
        ccl_group = level.require_group("cross_chunk_links").require_group("0")
        data_arr = ccl_group.create_array(
            name="data", shape=(8,), chunks=(8,), dtype="uint8",
        )
        data_arr.attrs.update({
            "zv_array": "cross_chunk_links",
            "sid_ndim": 3,
            "level_delta": 0,
            "link_width": 2,
        })
        lg = Group._from_zarr(level)
        with pytest.raises(ArrayError, match="legacy.*data.*blob"):
            read_cross_chunk_links(lg, delta=0)

    def test_kN_arrays_exist_per_distinct_K(self, tmp_path: Path) -> None:
        lg = _make_level_group(tmp_path)
        create_cross_chunk_links_array(lg, delta=0, link_width=3)
        records = [
            # K=2 face: two distinct chunks among three vertices
            [((0, 0, 0), 1), ((1, 0, 0), 2), ((0, 0, 0), 3)],
            # K=3 face: three distinct chunks
            [((0, 0, 0), 4), ((1, 0, 0), 5), ((0, 1, 0), 6)],
        ]
        write_cross_chunk_links(
            lg, records, sid_ndim=3, link_width=3,
        )
        # Both k2 and k3 should exist on disk.
        parent = cross_chunk_links_path(0)
        children = list(lg.zarr_group[parent])
        assert "k2" in children
        assert "k3" in children
        # k1 should NOT exist (no records had K=1).
        assert "k1" not in children

    def test_per_leaf_lookup_round_trips(self, tmp_path: Path) -> None:
        from zarr_vectors.core.arrays import read_cross_chunk_link_leaf
        lg = _make_level_group(tmp_path)
        create_cross_chunk_links_array(lg, delta=0)
        records = [
            (((0, 0, 0), 5), ((1, 0, 0), 2)),
            (((0, 0, 0), 7), ((2, 0, 0), 3)),
        ]
        write_cross_chunk_links(lg, records, sid_ndim=3)
        # Direct cell read: chunks (0,0,0) + (1,0,0) → 1 record back.
        leaf = read_cross_chunk_link_leaf(
            lg, ((0, 0, 0), (1, 0, 0)), delta=0,
        )
        assert len(leaf) == 1
        assert (tuple(leaf[0][0][0]), leaf[0][0][1]) == ((0, 0, 0), 5)
        assert (tuple(leaf[0][1][0]), leaf[0][1][1]) == ((1, 0, 0), 2)

    def test_canonical_ci_for_l2_delta0(self, tmp_path: Path) -> None:
        """A record written with the larger chunk first comes back with
        the smaller chunk as endpoint 0 (ci=[0,1] canonicalization)."""
        lg = _make_level_group(tmp_path)
        create_cross_chunk_links_array(lg, delta=0)
        # B > A in lex order; pass B first.
        a, b = (0, 0, 0), (1, 0, 0)
        write_cross_chunk_links(
            lg, [(((b, 5)), ((a, 2)))], sid_ndim=3,
        )
        out = read_cross_chunk_links(lg, delta=0)
        assert len(out) == 1
        # Endpoint 0 (smaller chunk by lex) should be ((0,0,0), 2),
        # endpoint 1 should be ((1,0,0), 5).
        assert tuple(out[0][0][0]) == a and out[0][0][1] == 2
        assert tuple(out[0][1][0]) == b and out[0][1][1] == 5

    def test_delta_plus_one_preserves_endpoint_order(
        self, tmp_path: Path,
    ) -> None:
        """For delta != 0 the writer must NOT canonicalize; endpoint 0
        belongs to the owning level, the rest to the target level."""
        lg = _make_level_group(tmp_path)
        create_cross_chunk_links_array(lg, delta=1)
        # Pass (larger_chunk, ...) first.  This is endpoint 0 (owning).
        write_cross_chunk_links(
            lg, [(((1, 0, 0), 7), ((0, 0, 0), 3))],
            sid_ndim=3, delta=1,
        )
        out = read_cross_chunk_links(lg, delta=1)
        assert len(out) == 1
        # Endpoint 0 must still be the (1,0,0) side — no canonicalization.
        assert tuple(out[0][0][0]) == (1, 0, 0) and out[0][0][1] == 7
        assert tuple(out[0][1][0]) == (0, 0, 0) and out[0][1][1] == 3

    def test_chunk_origin_attr_present_on_kK_arrays(
        self, tmp_path: Path,
    ) -> None:
        lg = _make_level_group(tmp_path)
        create_cross_chunk_links_array(lg, delta=0)
        write_cross_chunk_links(
            lg, [(((0, 0, 0), 1), ((1, 0, 0), 2))], sid_ndim=3,
        )
        kN_arr = lg.zarr_group[cross_chunk_links_path(0)]["k2"]
        assert "chunk_origin" in kN_arr.attrs

    def test_list_leaves_enumerates_populated_cells(
        self, tmp_path: Path,
    ) -> None:
        from zarr_vectors.core.arrays import list_cross_chunk_link_leaves
        lg = _make_level_group(tmp_path)
        create_cross_chunk_links_array(lg, delta=0)
        records = [
            (((0, 0, 0), 1), ((1, 0, 0), 2)),
            (((0, 0, 0), 3), ((2, 0, 0), 4)),
        ]
        write_cross_chunk_links(lg, records, sid_ndim=3)
        leaves = list_cross_chunk_link_leaves(lg, delta=0)
        # Two cells, both K=2.
        assert len(leaves) == 2
        assert all(len(L) == 2 for L in leaves)
        # Each cell's chunk-tuple list is sorted lex.
        for L in leaves:
            assert L == tuple(sorted(L))
