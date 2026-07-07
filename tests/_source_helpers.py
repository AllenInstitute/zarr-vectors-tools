"""Test source builders that stamp ``fragment_attributes/segment_id``.

Core ``write_polylines`` no longer auto-writes a per-fragment ``segment_id``
(the precomputed_skeletons core did; core >= main only writes fragment
attributes the caller passes). The tools coarseners hard-require
``segment_id`` on the source level, so these helpers reproduce the old
behavior for tests by deriving ``segment_id`` = dense object id from the
object manifests after the write.

Usage in a test module (shadows the core writer, so existing call sites are
unchanged)::

    from tests._source_helpers import write_polylines_with_segment_id as write_polylines
"""

from __future__ import annotations

import numpy as np

from zarr_vectors.core.arrays import (
    create_fragment_attribute_array,
    read_all_object_manifests,
    write_chunk_fragment_attributes,
)
from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors.types.polylines import read_polylines, write_polylines  # noqa: F401  (read_polylines re-exported for tests)


def stamp_segment_id_from_manifests(store_path, level: int = 0) -> None:
    """Write ``fragment_attributes/segment_id`` (= object id) at ``level``.

    Reads the object manifests (``object_id -> [(chunk, fragment_index)]``)
    and, for each chunk, writes a ``segment_id`` array indexed by fragment
    index so fragments of the same object share a segment id across chunks —
    exactly what the coarsener needs to reconstruct object boundaries.
    """
    root = open_store(str(store_path), mode="r+")
    lg = get_resolution_level(root, level)
    manifests = read_all_object_manifests(lg)  # list indexed by object id

    per_chunk: dict[tuple[int, ...], dict[int, int]] = {}
    for oid, entries in enumerate(manifests):
        for chunk, fidx in entries:
            key = tuple(int(c) for c in chunk)
            per_chunk.setdefault(key, {})[int(fidx)] = int(oid)
    if not per_chunk:
        return

    create_fragment_attribute_array(lg, "segment_id", dtype="uint64")
    for chunk, fragmap in per_chunk.items():
        n = max(fragmap) + 1
        seg = np.array([fragmap.get(i, 0) for i in range(n)], dtype=np.uint64)
        write_chunk_fragment_attributes(lg, "segment_id", chunk, seg, dtype=np.uint64)


def write_polylines_with_segment_id(store_path, polylines, **kwargs):
    """``write_polylines`` + stamp per-fragment ``segment_id`` from manifests."""
    write_polylines(str(store_path), polylines, **kwargs)
    stamp_segment_id_from_manifests(store_path)
