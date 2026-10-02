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
  The streamline coarsener rebuilds objects from this column, so it asks
  for the dense id whatever the store has (``dense_ids``).

A third, ``fragment_attributes/object_id`` (the dense object id per
fragment), is what the skeleton coarsener reads; it is written on request.

All come from what the writer just produced -- the object manifests and
each chunk's fragment index, kilobytes per chunk -- so no vertex is read
back and no writer has to cooperate.  An array the writer already wrote is
left as it is -- except a ``vertex_count`` that is not uint32 (a TRX
``dps`` array of that name, say), which a viewer reading uint32 would
misread, so it is counted again, and one a coarsener carried from the level
below (``recount``).  A level without objects (a point cloud with no object
ids) gets none of them.

This is the one place these columns are computed; every ingest and every
coarsener that writes them comes through here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "object_vertex_counts",
    "stamp_level_object_columns",
    "stamp_object_columns",
]

VERTEX_COUNT_ATTR = "vertex_count"
SEGMENT_ID_ATTR = "segment_id"
OBJECT_ID_ATTR = "object_id"


def stamp_object_columns(
    store_path: str | Path,
    *,
    level: int = 0,
    vertex_count: bool = True,
    segment_id: bool = True,
    object_id: bool = False,
    dense_ids: bool = False,
) -> dict[str, bool]:
    """Write ``vertex_count`` and the fragment ``segment_id`` where ``level`` lacks them.

    Args:
        store_path: Store to stamp, modified in place.
        level: Resolution level to stamp.
        vertex_count: Write ``object_attributes/vertex_count``.
        segment_id: Write ``fragment_attributes/segment_id``.
        object_id: Write ``fragment_attributes/object_id``.
        dense_ids: Stamp ``segment_id`` with the dense object id even where
            the level has an ``object_attributes/segment_id``.

    Returns:
        ``{"vertex_count": written, "segment_id": written, "object_id": written}``.
    """
    from zarr_vectors.building import get_resolution_level, open_store

    level_group = get_resolution_level(open_store(str(store_path), mode="r+"), level)
    return stamp_level_object_columns(
        level_group, vertex_count=vertex_count, segment_id=segment_id,
        object_id=object_id, dense_ids=dense_ids,
    )


def stamp_level_object_columns(
    level_group: Any,
    *,
    vertex_count: bool = True,
    segment_id: bool = True,
    object_id: bool = False,
    dense_ids: bool = False,
    recount: bool = False,
) -> dict[str, bool]:
    """:func:`stamp_object_columns` on a level group opened for writing.

    ``recount`` counts ``vertex_count`` again even where the level has a
    uint32 one: a coarsener carries the column from the level below, where
    it described that level's vertices, not these.
    """
    from zarr_vectors.building import (
        create_fragment_attribute_array,
        create_object_attributes_array,
        read_all_object_manifests,
        read_object_attributes,
        write_chunk_fragment_attributes,
        write_object_attributes,
    )
    from zarr_vectors.constants import FRAGMENT_ATTRIBUTES, OBJECT_ATTRIBUTES

    written = {"vertex_count": False, "segment_id": False, "object_id": False}
    want_counts = vertex_count
    if (
        want_counts and not recount
        and level_group.array_exists(f"{OBJECT_ATTRIBUTES}/{VERTEX_COUNT_ATTR}")
    ):
        stored = np.asarray(read_object_attributes(level_group, VERTEX_COUNT_ATTR))
        want_counts = stored.dtype != np.uint32 or stored.ndim != 1
    want_ids = segment_id and not level_group.array_exists(
        f"{FRAGMENT_ATTRIBUTES}/{SEGMENT_ID_ATTR}",
    )
    want_oids = object_id and not level_group.array_exists(
        f"{FRAGMENT_ATTRIBUTES}/{OBJECT_ID_ATTR}",
    )
    if not (want_counts or want_ids or want_oids):
        return written
    try:
        manifests = read_all_object_manifests(level_group)
    except Exception:  # noqa: BLE001 - no object index: a level without objects
        return written
    if not any(manifests):
        return written

    owners = _fragment_owners(manifests)
    counts, fragment_count = _count_vertices(level_group, owners, len(manifests))

    if want_counts:
        # Every row is a real count: an object with no geometry here has 0.
        create_object_attributes_array(level_group, VERTEX_COUNT_ATTR, dtype="uint32")
        write_object_attributes(
            level_group, VERTEX_COUNT_ATTR, counts.astype(np.uint32),
            present_mask=np.ones(len(counts), dtype=np.uint8),
        )
        written["vertex_count"] = True

    columns: dict[str, np.ndarray] = {}
    if want_ids:
        ids = np.arange(len(manifests), dtype=np.uint64)
        if not dense_ids and level_group.array_exists(
            f"{OBJECT_ATTRIBUTES}/{SEGMENT_ID_ATTR}",
        ):
            source = np.asarray(
                read_object_attributes(level_group, SEGMENT_ID_ATTR), dtype=np.uint64,
            ).reshape(-1)
            ids[: min(len(ids), len(source))] = source[: len(ids)]
        columns[SEGMENT_ID_ATTR] = ids
    if want_oids:
        columns[OBJECT_ID_ATTR] = np.arange(len(manifests), dtype=np.uint64)
    for name, ids in columns.items():
        create_fragment_attribute_array(level_group, name, dtype="uint64")
        for chunk, (fragments, oids) in owners.items():
            # One row per fragment in the chunk, so the viewer, which
            # refuses a short column, takes it; a fragment no object names
            # keeps 0.
            column = np.zeros(fragment_count[chunk], dtype=np.uint64)
            column[fragments] = ids[oids]
            write_chunk_fragment_attributes(
                level_group, name, chunk, column, dtype=np.uint64,
            )
        written[name] = True
    return written


def object_vertex_counts(level_group: Any) -> np.ndarray:
    """Each object's vertex count at this level, from the fragment indexes."""
    from zarr_vectors.building import read_all_object_manifests

    manifests = read_all_object_manifests(level_group)
    counts, _ = _count_vertices(level_group, _fragment_owners(manifests), len(manifests))
    return counts


def _fragment_owners(
    manifests: list,
) -> dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]]:
    """Per chunk: ``(fragment indices, owning object ids)``."""
    by_chunk: dict[tuple[int, ...], tuple[list[int], list[int]]] = {}
    for oid, manifest in enumerate(manifests):
        for chunk, fragment in manifest:
            fragments, oids = by_chunk.setdefault(tuple(int(c) for c in chunk), ([], []))
            fragments.append(int(fragment))
            oids.append(oid)
    return {
        chunk: (np.asarray(fragments, dtype=np.int64), np.asarray(oids, dtype=np.int64))
        for chunk, (fragments, oids) in by_chunk.items()
    }


def _count_vertices(
    level_group: Any,
    owners: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]],
    n_objects: int,
) -> tuple[np.ndarray, dict[tuple[int, ...], int]]:
    """``(vertices per object, fragments per chunk)`` from the fragment indexes."""
    from zarr_vectors.building import read_vertex_fragment_index
    from zarr_vectors.constants import VERTEX_FRAGMENTS

    from zarr_vectors_tools.algorithms._links import chunk_key_str

    counts = np.zeros(n_objects, dtype=np.int64)
    fragment_count: dict[tuple[int, ...], int] = {}
    # One prefetch of the (small) fragment-index cells for the whole level.
    with level_group.batched_reads(
        [(VERTEX_FRAGMENTS, [chunk_key_str(c) for c in owners])],
    ):
        for chunk, (fragments, oids) in owners.items():
            index = read_vertex_fragment_index(level_group, chunk)
            fragment_count[chunk] = len(index)
            table = index.ranges()
            if table is not None:
                lengths = np.asarray(table[:, 1], dtype=np.int64)
            else:
                lengths = np.asarray([
                    index.range(f)[1] if index.is_range(f) else len(index.indices(f))
                    for f in range(len(index))
                ], dtype=np.int64)
            np.add.at(counts, oids, lengths[fragments])
    return counts, fragment_count
