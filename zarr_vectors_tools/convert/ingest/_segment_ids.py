"""Stamp ``fragment_attributes/segment_id`` on a freshly written level.

The streamline coarsener reconstructs which object a fragment belongs to
from a per-fragment ``segment_id``, and refuses outright without one --
*"coarsen_polyline_level requires fragment_attributes/segment_id on the
source level"*.  Core's ``write_polylines`` does not write it (core has no
such requirement), so a store ingested from a ``.tck``, ``.trx`` or
``.trk`` through the in-memory readers ingested cleanly and then failed at
the pyramid step, which for ``zvtools convert --coarsen ...`` is after the
expensive part has already succeeded.

The ids are the dense object ids, recovered from the object manifests the
writer just produced, so this needs no cooperation from the writer and no
second pass over the geometry.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def stamp_segment_ids(store_path: str | Path, *, level: int = 0) -> int:
    """Write ``fragment_attributes/segment_id`` for every fragment at ``level``.

    Args:
        store_path: Store to stamp, modified in place.
        level: Resolution level whose manifests define the mapping.

    Returns:
        The number of chunks stamped.
    """
    from zarr_vectors.building import (
        create_fragment_attribute_array,
        get_resolution_level,
        open_store,
        read_all_object_manifests,
        write_chunk_fragment_attributes,
    )

    level_group = get_resolution_level(
        open_store(str(store_path), mode="r+"), level,
    )
    manifests = read_all_object_manifests(level_group)

    per_chunk: dict[tuple[int, ...], dict[int, int]] = {}
    for oid, entries in enumerate(manifests):
        for chunk, fragment_index in entries:
            per_chunk.setdefault(
                tuple(int(c) for c in chunk), {},
            )[int(fragment_index)] = int(oid)
    if not per_chunk:
        return 0

    create_fragment_attribute_array(level_group, "segment_id", dtype="uint64")
    for chunk, fragments in per_chunk.items():
        column = np.zeros(max(fragments) + 1, dtype=np.uint64)
        for fragment_index, oid in fragments.items():
            column[fragment_index] = np.uint64(oid)
        write_chunk_fragment_attributes(
            level_group, "segment_id", chunk, column, dtype=np.uint64,
        )
    return len(per_chunk)
