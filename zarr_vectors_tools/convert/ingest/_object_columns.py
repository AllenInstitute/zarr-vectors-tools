"""Stamp the per-object columns a viewer reads on a freshly written level.

Two arrays let a viewer treat an object as one thing across chunks:

* ``object_attributes/vertex_count`` (uint32, one row per object) -- what
  the Neuroglancer fork budgets each object by.  Its OBJECT detail focus is
  offered only when every level has the column; the coarseners recount it
  per level, so the ingest has to start it.
* ``fragment_attributes/segment_id`` (uint64, one row per fragment) -- the
  id a fragment is coloured and picked by.  Without it the viewer falls back
  to the fragment's index within its chunk, so one object changes colour
  from chunk to chunk.  The id is the object's ``object_attributes/
  segment_id`` where the store has one (a source segment id), else its
  dense object id: the two spellings the viewer resolves a picked id by.

Both come from what the writer just produced -- the object manifests and
each chunk's fragment index, kilobytes per chunk -- so no vertex is read
back and no writer has to cooperate.  An array the writer already wrote is
left as it is -- except a ``vertex_count`` that is not uint32 (a TRX
``dps`` array of that name, say), which a viewer reading uint32 would
misread, so it is counted again.  A level without objects (a point cloud
with no object ids) gets neither.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["stamp_level_object_columns", "stamp_object_columns"]

VERTEX_COUNT_ATTR = "vertex_count"
SEGMENT_ID_ATTR = "segment_id"


def stamp_object_columns(
    store_path: str | Path,
    *,
    level: int = 0,
    vertex_count: bool = True,
    segment_id: bool = True,
) -> dict[str, bool]:
    """Write ``vertex_count`` and the fragment ``segment_id`` where ``level`` lacks them.

    Args:
        store_path: Store to stamp, modified in place.
        level: Resolution level to stamp.
        vertex_count: Write ``object_attributes/vertex_count``.
        segment_id: Write ``fragment_attributes/segment_id``.

    Returns:
        ``{"vertex_count": written, "segment_id": written}``.
    """
    from zarr_vectors.building import get_resolution_level, open_store

    level_group = get_resolution_level(open_store(str(store_path), mode="r+"), level)
    return stamp_level_object_columns(
        level_group, vertex_count=vertex_count, segment_id=segment_id,
    )


def stamp_level_object_columns(
    level_group: Any,
    *,
    vertex_count: bool = True,
    segment_id: bool = True,
) -> dict[str, bool]:
    """:func:`stamp_object_columns` on a level group opened for writing."""
    from zarr_vectors.building import (
        create_fragment_attribute_array,
        create_object_attributes_array,
        read_all_object_manifests,
        read_object_attributes,
        read_vertex_fragment_index,
        write_chunk_fragment_attributes,
        write_object_attributes,
    )
    from zarr_vectors.constants import FRAGMENT_ATTRIBUTES, OBJECT_ATTRIBUTES

    written = {"vertex_count": False, "segment_id": False}
    want_counts = vertex_count
    if want_counts and level_group.array_exists(f"{OBJECT_ATTRIBUTES}/{VERTEX_COUNT_ATTR}"):
        stored = np.asarray(read_object_attributes(level_group, VERTEX_COUNT_ATTR))
        want_counts = stored.dtype != np.uint32 or stored.ndim != 1
    want_ids = segment_id and not level_group.array_exists(
        f"{FRAGMENT_ATTRIBUTES}/{SEGMENT_ID_ATTR}",
    )
    if not (want_counts or want_ids):
        return written
    try:
        manifests = read_all_object_manifests(level_group)
    except Exception:  # noqa: BLE001 - no object index: a level without objects
        return written
    if not any(manifests):
        return written

    # Per chunk: each fragment's object, and each fragment's vertex count.
    owners: dict[tuple[int, ...], dict[int, int]] = {}
    for oid, manifest in enumerate(manifests):
        for chunk, fragment in manifest:
            owners.setdefault(tuple(int(c) for c in chunk), {})[int(fragment)] = oid
    counts = np.zeros(len(manifests), dtype=np.int64)
    fragment_count: dict[tuple[int, ...], int] = {}
    for chunk, fragments in owners.items():
        index = read_vertex_fragment_index(level_group, chunk)
        fragment_count[chunk] = len(index)
        table = index.ranges()
        for fragment, oid in fragments.items():
            if table is not None:
                counts[oid] += int(table[fragment, 1])
            elif index.is_range(fragment):
                counts[oid] += index.range(fragment)[1]
            else:
                counts[oid] += len(index.indices(fragment))

    if want_counts:
        # Every row is a real count: an object with no geometry here has 0.
        create_object_attributes_array(level_group, VERTEX_COUNT_ATTR, dtype="uint32")
        write_object_attributes(
            level_group, VERTEX_COUNT_ATTR, counts.astype(np.uint32),
            present_mask=np.ones(len(counts), dtype=np.uint8),
        )
        written["vertex_count"] = True

    if want_ids:
        ids = np.arange(len(manifests), dtype=np.uint64)
        if level_group.array_exists(f"{OBJECT_ATTRIBUTES}/{SEGMENT_ID_ATTR}"):
            source = np.asarray(
                read_object_attributes(level_group, SEGMENT_ID_ATTR), dtype=np.uint64,
            ).reshape(-1)
            ids[: min(len(ids), len(source))] = source[: len(ids)]
        create_fragment_attribute_array(level_group, SEGMENT_ID_ATTR, dtype="uint64")
        for chunk, fragments in owners.items():
            # One row per fragment in the chunk, so the viewer, which
            # refuses a short column, takes it; a fragment no object names
            # keeps 0.
            column = np.zeros(fragment_count[chunk], dtype=np.uint64)
            for fragment, oid in fragments.items():
                column[fragment] = ids[oid]
            write_chunk_fragment_attributes(
                level_group, SEGMENT_ID_ATTR, chunk, column, dtype=np.uint64,
            )
        written["segment_id"] = True
    return written
