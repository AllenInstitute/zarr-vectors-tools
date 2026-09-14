"""Moving geometry between stores, and between a file and a store.

``ingest`` makes a store from a file.  ``export`` makes a file from a
store.  Neither answers "put this into the store I already have", and
that gap is what this subpackage fills — in both directions:

.. code-block:: text

    file  --merge_stores-->  existing store
    store --merge_stores-->  existing store
    store --split_store -->  many stores

The pieces:

* :mod:`~zarr_vectors_tools.compose.readers` — parse a file into
  geometry, without building a store to hold it.
* :mod:`~zarr_vectors_tools.compose.sources` — one interface over "a
  file" and "another store", including the coordinate transform that
  brings one into the other's frame.
* :mod:`~zarr_vectors_tools.compose.merge` — write a source into an
  existing store, appending rather than replacing.
* :mod:`~zarr_vectors_tools.compose.split` — the inverse, cutting a store
  along its groups, an attribute, or a recorded merge.

A worked example — a bundle atlas, its label column and its lookup table,
joining a tractogram store as named groups::

    from zarr_vectors_tools.compose import (
        GeometrySource, derive_groups, load_lut, merge_stores, read_trk,
    )

    bundles = read_trk("atlas.trk")                      # space as stored
    derive_groups(bundles, "label_id", names=load_lut("atlas.json"))
    merge_stores(
        "tractogram.zarrvectors",
        [GeometrySource(bundles, transform=to_target_frame)],
        pyramid="rebuild",
    )
"""

from __future__ import annotations

from zarr_vectors_tools.compose.merge import merge_stores, plan_merge
from zarr_vectors_tools.compose.readers import (
    Geometry,
    derive_groups,
    load_lut,
    native_formats,
    read_geometry,
    read_tck,
    read_trk,
    read_trk_header,
    read_trx,
    trackvis_to_rasmm,
)
from zarr_vectors_tools.compose.sources import (
    FileSource,
    GeometrySource,
    ObjectBatch,
    Source,
    SourceInfo,
    StoreSource,
    open_source,
)
from zarr_vectors_tools.compose.split import plan_split, split_parts, split_store

__all__ = [
    "FileSource",
    "Geometry",
    "GeometrySource",
    "ObjectBatch",
    "Source",
    "SourceInfo",
    "StoreSource",
    "derive_groups",
    "load_lut",
    "merge_stores",
    "native_formats",
    "open_source",
    "plan_merge",
    "plan_split",
    "read_geometry",
    "read_tck",
    "read_trk",
    "read_trk_header",
    "read_trx",
    "split_parts",
    "split_store",
    "trackvis_to_rasmm",
]
