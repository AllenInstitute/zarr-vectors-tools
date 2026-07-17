"""Tests for the per-object pyramid: shared metavertices + ID-stable objects.

Covers the four invariants from the plan:

* **I1** OID-stable lookup — surviving OIDs at level L+1 are the same
  OIDs as at level L; dropped OIDs return an empty manifest.
* **I2** Continuity — manifest(L+1, oid) is the ordered-distinct image
  of manifest(L, oid) under the bin→metavertex map.
* **I3** Chunk-neighbourhood containment —
  ``chunks(L+1, oid) ⊆ neighbouring_chunk_keys(chunks(L, oid), halo=1)``.
* **I4** Shared-metavertex multiplicity — at least one metavertex is
  referenced by ≥ 2 objects' manifests, and the on-disk fragment
  count is strictly less than the sum of manifest-ref counts.
"""

from __future__ import annotations

import numpy as np
import pytest

from zarr_vectors.constants import (
    CAP_PRESERVED_OBJECT_IDS,
    CAP_SHARED_FRAGMENTS,
    COARSEN_PER_OBJECT,
)
from zarr_vectors.core.arrays import (
    read_all_object_manifests,
    read_chunk_vertices,
    read_object_attribute_present_mask,
    read_object_attributes,
)
from zarr_vectors.core.store import (
    get_resolution_level,
    list_resolution_levels,
    open_store,
    read_level_metadata,
    read_root_metadata,
)
from zarr_vectors.lazy.store import open_zv
from zarr_vectors_tools.multiresolution.coarsen import build_pyramid, coarsen_level
from zarr_vectors.spatial.chunking import neighbouring_chunk_keys
from zarr_vectors.types.polylines import write_polylines as _write_polylines_raw

from .conftest import stamp_segment_id_from_manifests


def write_polylines(store_path, polylines, **kwargs):
    """``write_polylines`` + derive ``fragment_attributes/segment_id``.

    zarr-vectors-py 0.8.1's ``write_polylines`` no longer auto-writes
    ``segment_id`` (see ``stamp_segment_id_from_manifests``), which
    ``coarsen_polyline_level`` hard-requires on its source level.
    """
    summary = _write_polylines_raw(store_path, polylines, **kwargs)
    root = open_store(str(store_path), mode="r+")
    stamp_segment_id_from_manifests(get_resolution_level(root, 0))
    return summary


# ===================================================================
# Fixture helpers
# ===================================================================


def _make_streamlines(seed=0, n=20, vpp=30, extent=100.0):
    rng = np.random.default_rng(seed)
    return [
        rng.uniform(0, extent, (vpp, 3)).astype("f4") for _ in range(n)
    ]


def _build_store(tmp_path, seed=0, n=20):
    store = tmp_path / "tr.zv"
    write_polylines(
        str(store),
        _make_streamlines(seed=seed, n=n, vpp=30),
        chunk_shape=(50.0, 50.0, 50.0),
        bin_shape=(5.0, 5.0, 5.0),
    )
    return store


# ===================================================================
# Metadata + capability stamping
# ===================================================================


def test_per_object_level_metadata(tmp_path):
    store = _build_store(tmp_path, seed=1, n=10)
    coarsen_level(str(store), source_level=0, target_level=1,
                  coarsen_factor=2.0, sparsity_factor=2.0, sparsity_seed=42)

    root = open_store(str(store))
    lm = read_level_metadata(root, 1)
    assert lm.preserves_object_ids is True
    # implicit_sequential path writes per-(object, coarsened-chunk)
    # fragments, so fragments aren't shared between objects.
    assert lm.shared_fragments is False
    assert lm.coarsening_method == COARSEN_PER_OBJECT
    assert lm.inherited_num_objects == 10
    assert lm.parent_level == 0

    rm = read_root_metadata(root)
    assert CAP_PRESERVED_OBJECT_IDS in rm.format_capabilities
    assert CAP_SHARED_FRAGMENTS not in rm.format_capabilities


# ===================================================================
# I1 — OID stability
# ===================================================================


def test_oid_stable_lookup(tmp_path):
    store = _build_store(tmp_path, seed=2, n=20)
    coarsen_level(str(store), source_level=0, target_level=1,
                  coarsen_factor=1.5, sparsity_factor=2.0, sparsity_seed=42)

    root = open_store(str(store))
    lvl1 = get_resolution_level(root, 1)
    manifests = read_all_object_manifests(lvl1)

    assert len(manifests) == 20  # inherited OID space
    survivors = {i for i, m in enumerate(manifests) if m}
    drops = {i for i, m in enumerate(manifests) if not m}
    assert len(survivors) + len(drops) == 20
    assert len(survivors) < 20  # something was dropped
    # Dropped OIDs are valid lookups returning [] (not exceptions).
    for oid in drops:
        assert manifests[oid] == []


def test_monotone_oid_drop_across_levels(tmp_path):
    """S(L+1) ⊆ S(L) — once dropped, an OID stays dropped at coarser levels."""
    store = _build_store(tmp_path, seed=3, n=30)
    build_pyramid(
        str(store),
        factors=[(1.5, 2.0), (1.5, 2.0)],
        sparsity_seed=42,
    )

    zv = open_zv(str(store))
    levels = list_resolution_levels(open_store(str(store)))
    assert levels == [0, 1, 2]

    present_sets = {L: set(zv[L].present_oids.tolist()) for L in levels}
    assert present_sets[2] <= present_sets[1] <= present_sets[0]
    # Object_levels for any surviving level-2 OID is a contiguous prefix.
    for oid in present_sets[2]:
        visible = zv.object_levels(oid)
        assert visible == list(range(max(visible) + 1))


# ===================================================================
# I3 — chunk-neighbourhood containment
# ===================================================================


def test_chunk_halo_containment(tmp_path):
    store = _build_store(tmp_path, seed=4, n=20)
    coarsen_level(str(store), source_level=0, target_level=1,
                  coarsen_factor=2.0, sparsity_factor=2.0, sparsity_seed=42)

    root = open_store(str(store))
    lvl0 = get_resolution_level(root, 0)
    lvl1 = get_resolution_level(root, 1)
    m0 = read_all_object_manifests(lvl0)
    m1 = read_all_object_manifests(lvl1)

    for oid in range(len(m1)):
        if not m1[oid]:
            continue
        chunks0 = {cc for cc, _ in m0[oid]}
        halo = set(chunks0)
        for c in chunks0:
            halo.update(neighbouring_chunk_keys(c, halo=1))
        chunks1 = {cc for cc, _ in m1[oid]}
        assert chunks1.issubset(halo), (
            f"OID {oid}: chunks at level 1 {chunks1} exceed level-0 halo {halo}"
        )


# ===================================================================
# I4 — shared metavertices
# ===================================================================


def test_per_object_fragments_layout(tmp_path):
    """On the implicit_sequential path each (object, coarsened-chunk)
    visit is one fragment.  Manifest entries are not shared between
    objects; the on-disk fragment count equals the total manifest
    entry count.  Bin centroids ARE shared across fragments at the
    position level — multiple fragments in one chunk may include a
    row at the same centroid coordinates."""
    store = _build_store(tmp_path, seed=5, n=30)
    coarsen_level(str(store), source_level=0, target_level=1,
                  coarsen_factor=2.0, sparsity_factor=1.0, sparsity_seed=42)

    root = open_store(str(store))
    lvl1 = get_resolution_level(root, 1)
    mans = read_all_object_manifests(lvl1)

    all_refs = [r for m in mans for r in m]
    unique_refs = set(all_refs)
    # Each (chunk, fragment_idx) belongs to exactly one object.
    assert len(all_refs) == len(unique_refs), (
        "Per-(object, chunk) fragments must not be shared between objects."
    )

    from zarr_vectors.core.arrays import list_chunk_keys
    on_disk = 0
    for cc in list_chunk_keys(lvl1):
        fragments = read_chunk_vertices(lvl1, cc, dtype=np.float32, ndim=3)
        on_disk += len(fragments)
    assert on_disk == len(unique_refs)

    # Centroid-position sharing across fragments: at least one pair of
    # fragments inside the same chunk must contain a row at the same
    # coordinates (different streamlines crossing the same bin).
    from zarr_vectors.core.arrays import list_chunk_keys as _lc
    found_position_overlap = False
    for cc in _lc(lvl1):
        fragments = read_chunk_vertices(lvl1, cc, dtype=np.float32, ndim=3)
        if len(fragments) < 2:
            continue
        all_rows = np.concatenate(fragments, axis=0)
        unique_rows = np.unique(all_rows, axis=0)
        if len(unique_rows) < len(all_rows):
            found_position_overlap = True
            break
    assert found_position_overlap, (
        "Expected at least one shared metavertex POSITION across "
        "per-object fragments in some chunk."
    )


# ===================================================================
# I2 — continuity (sketch — full bin-mapped recomputation is heavy;
# we instead verify the de-duplication invariant)
# ===================================================================


def test_consecutive_dedup_in_manifest(tmp_path):
    """Manifests at level 1 never have two consecutive identical refs."""
    store = _build_store(tmp_path, seed=6, n=20)
    coarsen_level(str(store), source_level=0, target_level=1,
                  coarsen_factor=3.0, sparsity_factor=1.0)

    root = open_store(str(store))
    lvl1 = get_resolution_level(root, 1)
    for oid, m in enumerate(read_all_object_manifests(lvl1)):
        for i in range(1, len(m)):
            assert m[i] != m[i - 1], (
                f"OID {oid} has consecutive duplicate ref at i={i}: {m[i]}"
            )


# ===================================================================
# Factor edge cases
# ===================================================================


def test_factors_both_one_is_noop_ish(tmp_path):
    """factors=(1, 1) produces a level identical in object count."""
    store = _build_store(tmp_path, seed=7, n=15)
    coarsen_level(str(store), source_level=0, target_level=1,
                  coarsen_factor=1.0, sparsity_factor=1.0)

    root = open_store(str(store))
    lvl0_mans = read_all_object_manifests(get_resolution_level(root, 0))
    lvl1_mans = read_all_object_manifests(get_resolution_level(root, 1))
    survivors = sum(1 for m in lvl1_mans if m)
    assert survivors == sum(1 for m in lvl0_mans if m)


def test_pure_sparsify(tmp_path):
    """coarsen_factor=1, sparsity_factor=2 — half the OIDs dropped."""
    store = _build_store(tmp_path, seed=8, n=40)
    coarsen_level(str(store), source_level=0, target_level=1,
                  coarsen_factor=1.0, sparsity_factor=2.0, sparsity_seed=42)

    root = open_store(str(store))
    lvl1_mans = read_all_object_manifests(get_resolution_level(root, 1))
    survivors = sum(1 for m in lvl1_mans if m)
    # Bernoulli ~half; the implementation routes via apply_sparsity
    # which exactly halves for the random strategy.
    assert 10 <= survivors <= 30


def test_factors_via_build_pyramid(tmp_path):
    store = _build_store(tmp_path, seed=9, n=30)
    result = build_pyramid(
        str(store),
        factors=[(2.0, 2.0), (2.0, 2.0)],
        sparsity_seed=42,
    )
    assert result["levels_created"] == 2
    assert result["method"] == COARSEN_PER_OBJECT
    for spec in result["level_specs"]:
        assert spec["method"] == COARSEN_PER_OBJECT
        assert spec["preserves_object_ids"] is True


# ===================================================================
# Object attribute present_mask round-trip
# ===================================================================


def test_object_attribute_present_mask_roundtrip(tmp_path):
    """When a parent level has object_attributes, the per-object coarsen
    writes the per-OID present_mask alongside the dense array."""
    from zarr_vectors.core.arrays import (
        create_object_attributes_array,
        write_object_attributes,
    )

    store = _build_store(tmp_path, seed=10, n=20)
    root = open_store(str(store), mode="r+")
    lvl0 = get_resolution_level(root, 0)
    # Inject an object_attribute at level 0.
    obj_attr = np.arange(20, dtype="f4")
    create_object_attributes_array(lvl0, "score")
    write_object_attributes(lvl0, "score", obj_attr)

    coarsen_level(str(store), source_level=0, target_level=1,
                  coarsen_factor=1.0, sparsity_factor=2.0, sparsity_seed=42)

    lvl1 = get_resolution_level(root, 1)
    out = read_object_attributes(lvl1, "score")
    mask = read_object_attribute_present_mask(lvl1, "score")
    assert mask is not None
    assert mask.shape == (20,)
    # Present rows have their source value; absent rows carry the 0.8.1
    # default "absent" sentinel (NaN for floats — write_object_attributes
    # overwrites present_mask=0 rows with fill_value in-band; the old
    # data+present_mask sibling-array layout left them as whatever the
    # caller's zero-filled array already had).
    for oid in range(20):
        if mask[oid]:
            assert out[oid] == oid
        else:
            assert np.isnan(out[oid])
