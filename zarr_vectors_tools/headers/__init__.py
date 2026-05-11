"""Format-specific header preservation for zarr vectors stores.

When data is ingested from a format (TRK, SWC, LAS, OBJ, ...),
the original header metadata is stored in ``/headers/<format>/``
within the zarr vectors store.  On export, headers are read back
to reconstruct format-specific fields for perfect round-tripping.

Headers accumulate — if a TRK is ingested and later exported to
TRX, both ``/headers/trk`` and ``/headers/trx`` will exist.
"""

from zarr_vectors_tools.headers.registry import HeaderRegistry

__all__ = ["HeaderRegistry"]
