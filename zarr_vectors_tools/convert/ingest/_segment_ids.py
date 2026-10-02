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
second pass over the geometry.  The work is
:func:`~zarr_vectors_tools.convert.ingest._object_columns.stamp_object_columns`'s;
this is the name the streamline ingests call it by.
"""

from __future__ import annotations

from pathlib import Path

from zarr_vectors_tools.convert.ingest._object_columns import stamp_object_columns


def stamp_segment_ids(store_path: str | Path, *, level: int = 0) -> bool:
    """Write ``fragment_attributes/segment_id`` for every fragment at ``level``.

    Args:
        store_path: Store to stamp, modified in place.
        level: Resolution level whose manifests define the mapping.

    Returns:
        Whether the column was written (a level that already has one, or
        has no objects, is left as it is).
    """
    return stamp_object_columns(
        store_path, level=level, vertex_count=False, dense_ids=True,
    )["segment_id"]
