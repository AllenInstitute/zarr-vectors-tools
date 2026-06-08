"""Object-index reduce step (coordination).

:func:`build_object_index` groups the per-chunk ``(segment_id, chunk, fragment)``
records emitted by the level-0 skeleton writer into a segment-id-preserving dense
object index, writing ``object_index`` + ``object_attributes/segment_id``.

This is *coordination*, not a per-chunk IO primitive (it gathers every chunk's
records to assign the global dense OID space), so it lives in zarr-vectors-tools
alongside the pyramid coordinators rather than in the core data-access SDK.  It
calls only core IO primitives (``write_object_index`` /
``write_object_attributes``).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from zarr_vectors.core.arrays import (
    create_object_attributes_array,
    write_object_attributes,
    write_object_index,
)
from zarr_vectors.typing import ChunkCoords

SEGMENT_ID_ATTR = "segment_id"


def build_object_index(
    level_group,
    records: list[tuple[int, ChunkCoords, int]],
    *,
    ndim: int,
) -> dict[int, int]:
    """Group per-chunk records by segment_id → dense object index.

    Writes ``object_index`` (one manifest per object, dense IDs in
    sorted-segment-id order) and ``object_attributes/segment_id`` (uint64) so
    skeletons can be pulled back out by original ID.

    Args:
        level_group: Level-0 group.
        records: ``(segment_id, chunk_coords, fragment_index)`` from every chunk
            written.
        ndim: Spatial index dim count.

    Returns:
        ``{segment_id: object_id}``.
    """
    # Deterministic order: by segment_id, then chunk, then fragment.
    records = sorted(records, key=lambda r: (int(r[0]), tuple(r[1]), int(r[2])))
    seg_ids = sorted({int(r[0]) for r in records})
    oid_of_seg = {s: i for i, s in enumerate(seg_ids)}

    manifests: dict[int, list[tuple[ChunkCoords, int]]] = defaultdict(list)
    for seg, cc, fidx in records:
        manifests[oid_of_seg[int(seg)]].append((tuple(int(c) for c in cc), int(fidx)))

    write_object_index(
        level_group, dict(manifests), sid_ndim=ndim,
        total_objects=len(seg_ids),
    )
    seg_arr = np.asarray(seg_ids, dtype=np.uint64)
    create_object_attributes_array(level_group, SEGMENT_ID_ATTR, dtype="uint64")
    write_object_attributes(level_group, SEGMENT_ID_ATTR, seg_arr)
    return oid_of_seg
