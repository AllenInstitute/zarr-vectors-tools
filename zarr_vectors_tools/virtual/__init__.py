"""Virtual zarr-vectors stores: a standard file read as a zarr-vectors store
on demand, without converting it.

The file is read once and binned into spatial cells; every key of the store
(``zarr.json``, ``0/vertices/c/1/0/2``, …) is then made when asked.  The
layout matches, chunk for chunk, the one the browser makes from the same
file (ngplatform's Zarr Vectors plugin reads local tractograms this way),
and is a valid format 0.9.2 store (``zarr_vectors.validate(level=3)``).

Tractograms for now: TrackVis ``.trk``, MRtrix ``.tck`` and ``.trx``::

    from zarr_vectors_tools.virtual import open_virtual
    store = open_virtual("tracts.trk")
    store.get("zarr.json"), store.list("0/object_attributes/")

``zvtools virtual serve tracts.trk`` serves them over HTTP to any
Neuroglancer with the zarr-vectors data source (ranges, CORS and folder
listings included).
"""

from __future__ import annotations

from pathlib import Path

from .streamlines import StreamlineStore, cell_size, fragment_index
from .tractograms import open_tck, open_tractogram, open_trk, open_trx

VIRTUAL_SUFFIXES = (".trk", ".tck", ".trx")


def open_virtual(path: str | Path) -> StreamlineStore:
    """A file as a virtual zarr-vectors store, read once now."""
    return StreamlineStore(open_tractogram(path))


__all__ = [
    "StreamlineStore",
    "VIRTUAL_SUFFIXES",
    "cell_size",
    "fragment_index",
    "open_tck",
    "open_tractogram",
    "open_trk",
    "open_trx",
    "open_virtual",
]
