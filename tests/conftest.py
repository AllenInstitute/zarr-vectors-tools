"""Shared pytest fixtures for zarr-vectors-tools tests."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded numpy random generator for reproducible test data."""
    return np.random.default_rng(42)


def stamp_segment_id_from_manifests(level_group) -> None:
    """Derive and write ``fragment_attributes/segment_id`` from the level's
    ``object_index`` manifests.

    Under zarr-vectors-py 0.8.1, :func:`zarr_vectors.types.polylines.write_polylines`
    no longer auto-writes ``segment_id`` (unlike the pre-0.8.1 writer this
    tools package was originally built against) — callers that need it
    (:func:`zarr_vectors_tools.multiresolution.strategies.polylines.coarsen_polyline_level`
    hard-requires it on its source level) must derive it themselves. Test
    fixtures that build a level via ``write_polylines`` call this afterward
    instead of relying on the removed auto-write.
    """
    from zarr_vectors.core.arrays import (
        create_fragment_attribute_array,
        read_all_object_manifests,
        write_chunk_fragment_attributes,
    )

    per_chunk: dict = defaultdict(dict)
    for oid, manifest in enumerate(read_all_object_manifests(level_group)):
        for cc, fidx in manifest:
            per_chunk[tuple(int(x) for x in cc)][int(fidx)] = oid

    if not per_chunk:
        return
    create_fragment_attribute_array(level_group, "segment_id", dtype="uint64")
    for cc, frag_map in per_chunk.items():
        seg_ids = np.zeros(max(frag_map) + 1, dtype=np.uint64)
        for fidx, oid in frag_map.items():
            seg_ids[fidx] = oid
        write_chunk_fragment_attributes(
            level_group, "segment_id", cc, seg_ids, dtype=np.uint64,
        )
