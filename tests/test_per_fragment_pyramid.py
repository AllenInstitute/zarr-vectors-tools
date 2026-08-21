"""Tests for the per-*fragment* pyramid: reuse-preserving coarsening.

The per-object coarsener re-cuts every object into per-(object, coarse-chunk)
runs, so a fragment two objects shared at level 0 becomes two private fragments
at level 1 — the fragment count tracks object×chunk runs and a coarse level can
end up larger on disk than its source. This strategy exists to hold the
fragment graph still instead, so the invariants under test are:

* **F1** No spike — the level-1 fragment count never exceeds level 0's.
* **F2** Reuse survives — a fragment referenced by k objects at level 0 is
  referenced by the same k objects at level 1.
* **F3** Subset geometry — coarse positions are a subset of source positions
  (decimation, not binning), so no averaged coordinates appear.
* **F4** Per-fragment ratio — ``coarsen_factor=2`` halves each fragment's
  vertex count, endpoints retained.
* **F5** Storage shrinks monotonically, unlike the per-object path.
"""

from __future__ import annotations

import numpy as np
import pytest
from zarr_vectors.building import (
    create_object_index_array,
    create_store,
    create_vertices_array,
    get_resolution_level,
    list_chunk_keys,
    open_store,
    read_all_object_manifests,
    read_chunk_vertices,
    read_level_metadata,
    write_chunk_fragments,
    write_chunk_vertices,
    write_object_index,
)

from zarr_vectors_tools.multiresolution.coarsen import build_pyramid, coarsen_level
from zarr_vectors_tools.multiresolution.strategies.fragments import (
    COARSEN_PER_FRAGMENT,
)

#: Fragment layout of the fixture: four fragments over one chunk's 40 vertices.
_FRAGMENTS = [
    np.arange(0, 20, dtype=np.int64),    # 0
    np.arange(10, 30, dtype=np.int64),   # 1 — overlaps 0
    np.arange(20, 40, dtype=np.int64),   # 2
    np.arange(0, 40, 2, dtype=np.int64),  # 3 — strided across the whole chunk
]

#: Objects, several sharing a fragment. Fragment 1 is referenced 3x, 2 twice.
_OBJECTS = {
    0: [0, 1],
    1: [1, 2],
    2: [1, 3],
    3: [2],
    4: [3],
}


@pytest.fixture
def shared_fragment_store(tmp_path):
    """One chunk, 4 fragments, 5 objects with overlapping manifests."""
    store = tmp_path / "shared.zarrvectors"
    root = create_store(
        store,
        bounds=([0.0, 0.0, 0.0], [50.0, 50.0, 50.0]),
        chunk_shape=(50.0, 50.0, 50.0),
        geometry_types=["polyline"],
    )
    lg = get_resolution_level(root, 0)

    # A line through the chunk, so decimation has something ordered to thin.
    pos = np.stack([
        np.linspace(1.0, 49.0, 40),
        np.full(40, 25.0),
        np.full(40, 25.0),
    ], axis=1).astype(np.float32)

    create_vertices_array(lg, dtype="float32")
    write_chunk_vertices(lg, (0, 0, 0), [pos], dtype=np.float32)
    write_chunk_fragments(
        lg, (0, 0, 0), list(_FRAGMENTS), target="vertex", mode="replace",
    )
    create_object_index_array(lg)
    write_object_index(
        lg,
        {oid: [((0, 0, 0), f) for f in frags] for oid, frags in _OBJECTS.items()},
        sid_ndim=3,
    )
    return store


# ===================================================================
# Helpers
# ===================================================================

def _fragments(store, level):
    lg = get_resolution_level(open_store(str(store), mode="r"), level)
    out = []
    for cc in sorted(list_chunk_keys(lg)):
        for arr in read_chunk_vertices(lg, cc):
            out.append(np.asarray(arr))
    return out


def _refs(store, level):
    lg = get_resolution_level(open_store(str(store), mode="r"), level)
    return [
        (tuple(int(x) for x in cc), int(f))
        for m in read_all_object_manifests(lg) for cc, f in m
    ]


def _stored_rows(store, level):
    lg = get_resolution_level(open_store(str(store), mode="r"), level)
    return sum(
        len(lg.read_bytes("vertices", ".".join(str(int(c)) for c in cc))) // 12
        for cc in list_chunk_keys(lg)
    )


def _positions(store, level, decimals=4):
    return {
        tuple(p) for arr in _fragments(store, level)
        for p in np.round(arr, decimals).tolist()
    }


# ===================================================================
# F1 / F2 — the point of the strategy
# ===================================================================

def test_fragment_count_does_not_spike(shared_fragment_store):
    store = shared_fragment_store
    n0 = len(_fragments(store, 0))
    coarsen_level(str(store), 0, 1, coarsen_factor=2.0, method="per_fragment")
    assert len(_fragments(store, 1)) <= n0 == len(_FRAGMENTS)


def test_reuse_is_preserved_object_for_object(shared_fragment_store):
    store = shared_fragment_store
    coarsen_level(str(store), 0, 1, coarsen_factor=2.0, method="per_fragment")

    lg0 = get_resolution_level(open_store(str(store), mode="r"), 0)
    lg1 = get_resolution_level(open_store(str(store), mode="r"), 1)
    m0 = read_all_object_manifests(lg0)
    m1 = read_all_object_manifests(lg1)

    # Every object keeps the same number of entries...
    for oid in _OBJECTS:
        assert len(m1[oid]) == len(m0[oid]), oid

    # ...and objects that shared a source fragment share its image.
    by_src: dict = {}
    for oid, frags in _OBJECTS.items():
        for pos, f in enumerate(frags):
            by_src.setdefault(f, []).append((oid, pos))
    for src_f, holders in by_src.items():
        images = {tuple(m1[oid][pos]) for oid, pos in holders}
        assert len(images) == 1, (
            f"source fragment {src_f} was shared by {[o for o, _ in holders]} "
            f"but mapped to {len(images)} different coarse fragments"
        )

    # Reuse factor is unchanged.
    r0, r1 = _refs(store, 0), _refs(store, 1)
    assert len(r0) / len(set(r0)) == pytest.approx(len(r1) / len(set(r1)))


def test_level_metadata_declares_shared_fragments(shared_fragment_store):
    store = shared_fragment_store
    coarsen_level(str(store), 0, 1, coarsen_factor=2.0, method="per_fragment")
    lm = read_level_metadata(open_store(str(store), mode="r"), 1)
    assert lm.shared_fragments is True
    assert lm.coarsening_method == COARSEN_PER_FRAGMENT
    assert lm.preserves_object_ids is True
    assert lm.inherited_num_objects == len(_OBJECTS)


# ===================================================================
# F3 / F4 — the coarsening itself
# ===================================================================

def test_root_capabilities_advertise_the_sharing(shared_fragment_store):
    # A consumer branches on CAP_SHARED_FRAGMENTS to decide whether it may
    # dedupe by fragment identity. Leaving it unstamped would say "no" when the
    # answer is yes — the exact inverse of the per-object path's situation.
    from zarr_vectors.constants import CAP_PRESERVED_OBJECT_IDS, CAP_SHARED_FRAGMENTS
    from zarr_vectors.building import read_root_metadata

    store = shared_fragment_store
    coarsen_level(str(store), 0, 1, coarsen_factor=2.0, method="per_fragment")
    caps = read_root_metadata(open_store(str(store), mode="r")).format_capabilities
    assert CAP_SHARED_FRAGMENTS in caps
    assert CAP_PRESERVED_OBJECT_IDS in caps


def test_coarse_positions_are_a_subset_not_centroids(shared_fragment_store):
    store = shared_fragment_store
    coarsen_level(str(store), 0, 1, coarsen_factor=2.0, method="per_fragment")
    p0, p1 = _positions(store, 0), _positions(store, 1)
    assert p1 <= p0, "decimation must not invent averaged coordinates"
    assert p1 < p0, "some vertices must actually be dropped"


@pytest.mark.parametrize("factor", [2.0, 4.0])
def test_each_fragment_is_thinned_by_the_factor(shared_fragment_store, factor):
    store = shared_fragment_store
    coarsen_level(str(store), 0, 1, coarsen_factor=factor, method="per_fragment")
    before = [len(f) for f in _FRAGMENTS]
    after = [a.shape[0] for a in _fragments(store, 1)]
    assert len(after) == len(before)
    for b, a in zip(before, after):
        assert a == max(2, int(np.ceil(b / factor))), (b, a, factor)


def test_endpoints_survive_thinning(shared_fragment_store):
    # Fragment ends are where cross-chunk seams attach; dropping them would
    # silently sever connectivity.
    store = shared_fragment_store
    src = _fragments(store, 0)
    coarsen_level(str(store), 0, 1, coarsen_factor=4.0, method="per_fragment")
    for before, after in zip(src, _fragments(store, 1)):
        assert np.allclose(before[0], after[0])
        assert np.allclose(before[-1], after[-1])


# ===================================================================
# F5 — storage
# ===================================================================

def test_storage_shrinks_monotonically_across_a_pyramid(shared_fragment_store):
    store = shared_fragment_store
    build_pyramid(
        str(store), factors=[(2.0, 1.0), (2.0, 1.0)],
        method="per_fragment", cross_level_depth=0, cross_level_storage="none",
    )
    rows = [_stored_rows(store, lv) for lv in (0, 1, 2)]
    assert rows[0] > rows[1] > rows[2], rows
    counts = [len(_fragments(store, lv)) for lv in (0, 1, 2)]
    assert counts[1] <= counts[0] and counts[2] <= counts[1], counts


def test_sparsity_drops_fragments_no_surviving_object_references(
    shared_fragment_store,
):
    store = shared_fragment_store
    coarsen_level(str(store), 0, 1, coarsen_factor=1.0, sparsity_factor=2.0,
                  sparsity_seed=0, method="per_fragment")
    kept = {f for _cc, f in _refs(store, 1)}
    # Every emitted fragment is referenced; nothing orphaned.
    assert len(_fragments(store, 1)) == len(kept)
    assert len(_fragments(store, 1)) <= len(_FRAGMENTS)


# ===================================================================
# Contrast with the per-object path (documents WHY this exists)
# ===================================================================

def test_per_object_path_spikes_where_this_one_does_not(shared_fragment_store, tmp_path):
    import shutil

    other = tmp_path / "per_object.zarrvectors"
    shutil.copytree(shared_fragment_store, other)

    coarsen_level(str(shared_fragment_store), 0, 1, coarsen_factor=2.0,
                  method="per_fragment")
    coarsen_level(str(other), 0, 1, coarsen_factor=2.0)  # automatic routing

    frag_shared = len(_fragments(shared_fragment_store, 1))
    frag_object = len(_fragments(other, 1))
    r = _refs(other, 1)
    assert frag_shared <= len(_FRAGMENTS)
    assert len(r) == len(set(r)), (
        "per-object coarsening is expected to give every manifest entry its "
        "own fragment — if this ever stops being true, the contrast this "
        "strategy exists for has changed"
    )
    assert frag_shared < frag_object


# ===================================================================
# Cumulative coarsen factors (shared by every strategy)
# ===================================================================

def test_coarsen_factors_compound_across_levels(shared_fragment_store):
    """``coarsen`` is a ratio against the level below, not against the root.

    ``[2, 2]`` must give 2x then 4x. Multiplying the ROOT bin at every level
    instead makes level 2 a near-identity re-bin of level 1, so a three-level
    pyramid silently collapses to two useful ones.
    """
    from zarr_vectors.building import read_level_metadata

    store = shared_fragment_store
    build_pyramid(
        str(store), factors=[(2.0, 1.0), (2.0, 1.0)],
        cross_level_depth=0, cross_level_storage="none",
    )
    root = open_store(str(store), mode="r")
    bins = [read_level_metadata(root, lv).bin_shape for lv in (1, 2)]
    assert bins[1][0] == pytest.approx(bins[0][0] * 2.0), (
        f"level 2 must be 2x level 1, got {bins}"
    )


def test_an_empty_link_family_still_carries_its_counts(tmp_path):
    """A coarse level with no surviving links must stamp ``num_links = 0``.

    ``write_links`` stamps the family counts as a side effect, so a level that
    collapses to a single metavertex — nothing left to link — used to leave the
    family carrying policy but no counts. Compounding factors makes deep levels
    collapse routinely, so this stopped being a corner case.
    """
    from zarr_vectors.building import (
        get_resolution_level,
        links_group_path,
        list_link_deltas,
        list_resolution_levels,
    )
    from zarr_vectors.types.graphs import write_graph

    store = tmp_path / "graph.zarrvectors"
    rng = np.random.default_rng(0)
    n = 64
    write_graph(
        store,
        positions=rng.uniform(0.0, 100.0, size=(n, 3)).astype(np.float32),
        edges=np.stack([np.arange(n - 1), np.arange(1, n)], axis=1).astype(np.int64),
        object_ids=np.zeros(n, dtype=np.int64),
        chunk_shape=(40.0, 40.0, 40.0),
        bounds=([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]),
    )
    # Deep enough that the last level bins the whole volume into one cell.
    build_pyramid(str(store), factors=[(2.0, 1.0), (2.0, 1.0)],
                  cross_level_depth=1, cross_level_storage="explicit")

    root = open_store(str(store), mode="r")
    for lvl in list_resolution_levels(root):
        lg = get_resolution_level(root, lvl)
        for delta in list_link_deltas(lg):
            meta = lg.read_array_meta(links_group_path(delta))
            assert "num_links" in meta, f"level {lvl} delta {delta}"


# ===================================================================
# Group propagation (shared by every coarsener)
# ===================================================================

@pytest.mark.parametrize("method", ["per_fragment", None])
def test_group_taxonomy_reaches_every_level(shared_fragment_store, method):
    """``<N>/groups/<id>`` must answer what ``0/groups/<id>`` answers.

    Groups name object ids and every coarsener preserves object ids, so the
    taxonomy is meaningful at a coarse level unchanged. Without propagation a
    coarse level has an object index but no way to say what any object *is* —
    a reader asking for "the network object" at level 2 gets nothing.
    """
    from zarr_vectors.building import (
        get_resolution_level,
        list_resolution_levels,
        read_all_groupings,
    )

    store = shared_fragment_store
    # Give the fixture a named group taxonomy to carry.
    lg0 = get_resolution_level(open_store(str(store), mode="r+"), 0)
    write_object_index(
        lg0,
        {oid: [((0, 0, 0), f) for f in frags] for oid, frags in _OBJECTS.items()},
        sid_ndim=3,
    )
    from zarr_vectors.building import create_groupings_array, write_groupings

    create_groupings_array(lg0)
    write_groupings(lg0, {0: [0, 1], 1: [2, 3], 2: [4]})

    build_pyramid(
        str(store), factors=[(2.0, 1.0), (2.0, 1.0)], method=method,
        cross_level_depth=0, cross_level_storage="none",
    )

    root = open_store(str(store), mode="r")
    base = [list(g) for g in read_all_groupings(get_resolution_level(root, 0))]
    for lv in list_resolution_levels(root):
        if lv == 0:
            continue
        lg = get_resolution_level(root, lv)
        got = [list(g) for g in read_all_groupings(lg)]
        assert got == base, f"level {lv} taxonomy drifted: {got} != {base}"


def test_sparsified_objects_leave_their_groups(shared_fragment_store):
    """A group must not list an OID sparsification dropped.

    Dropped OIDs keep their slot with an empty manifest, so a stale membership
    would hand a reader an empty object and call it a member.
    """
    from zarr_vectors.building import (
        create_groupings_array,
        get_resolution_level,
        read_all_groupings,
        read_all_object_manifests,
        write_groupings,
    )

    store = shared_fragment_store
    lg0 = get_resolution_level(open_store(str(store), mode="r+"), 0)
    create_groupings_array(lg0)
    write_groupings(lg0, {0: list(_OBJECTS)})

    coarsen_level(str(store), 0, 1, coarsen_factor=1.0, sparsity_factor=2.0,
                  sparsity_seed=0, method="per_fragment")

    lg1 = get_resolution_level(open_store(str(store), mode="r"), 1)
    members = [int(o) for o in read_all_groupings(lg1)[0]]
    manifests = read_all_object_manifests(lg1)
    assert len(members) < len(_OBJECTS), "sparsity should have dropped someone"
    for oid in members:
        assert manifests[oid], f"group lists oid {oid} but its manifest is empty"
