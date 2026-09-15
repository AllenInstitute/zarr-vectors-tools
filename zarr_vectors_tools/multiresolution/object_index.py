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
from zarr_vectors.building import (
    create_fragment_attribute_array,
    create_object_attributes_array,
    write_chunk_fragment_attributes,
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

    # Materialize per-fragment OID in each chunk. This is required by the
    # skeleton coarsener and avoids global segment_id->OID lookups at runtime.
    create_fragment_attribute_array(level_group, "object_id", dtype="uint64")
    by_chunk: dict[ChunkCoords, list[tuple[int, int]]] = defaultdict(list)
    for seg, cc, fidx in records:
        by_chunk[tuple(int(c) for c in cc)].append((int(fidx), oid_of_seg[int(seg)]))
    for cc, pairs in by_chunk.items():
        if not pairs:
            continue
        max_fidx = max(fidx for fidx, _ in pairs)
        oid_arr = np.full(max_fidx + 1, np.uint64(0), dtype=np.uint64)
        for fidx, oid in pairs:
            oid_arr[int(fidx)] = np.uint64(oid)
        write_chunk_fragment_attributes(
            level_group, "object_id", cc, oid_arr, dtype=np.uint64
        )

    seg_arr = np.asarray(seg_ids, dtype=np.uint64)
    create_object_attributes_array(level_group, SEGMENT_ID_ATTR, dtype="uint64")
    write_object_attributes(level_group, SEGMENT_ID_ATTR, seg_arr)
    return oid_of_seg


def shard_rows_by_object(
    rows: np.ndarray, *, rows_per_shard: int = 250_000,
) -> list[np.ndarray]:
    """Cut object-index rows into shards of whole objects, in object-id order.

    ``rows[:, 0]`` is the object id; the remaining columns are whatever the
    reducer needs (fragment index, chunk coords, ...).  Every row of an
    object lands in the same shard -- a reducer needs all of an object's
    fragments to write its manifest -- and each shard is a contiguous,
    ascending id range of about ``rows_per_shard`` rows, the working set of
    one Phase C task.  Rows of one object keep their input order.

    This is how the parallel coarseners hand their gathered rows to the
    object-index reduce.  They used to bucket rows by ``oid % shards`` into
    one temp file per (target chunk, shard) and read those back one by one;
    at 250k rows a shard the whole level's rows are a few tens of megabytes
    of int64, which is the same order as the cross-chunk anchor arrays the
    same coordinators already gather inline.

    Returns ``[]`` when ``rows`` is empty.
    """
    rows = np.asarray(rows)
    n = int(rows.shape[0])
    if n == 0:
        return []
    order = np.argsort(rows[:, 0], kind="stable")
    rows = rows[order]
    oids = rows[:, 0]
    step = max(1, int(rows_per_shard))
    cuts = [0]
    while cuts[-1] < n:
        nxt = min(n, cuts[-1] + step)
        if nxt < n:
            # Never split an object: advance the cut past its last row.
            nxt = int(np.searchsorted(oids, oids[nxt], side="right"))
        cuts.append(nxt)
    return [rows[a:b] for a, b in zip(cuts[:-1], cuts[1:])]
