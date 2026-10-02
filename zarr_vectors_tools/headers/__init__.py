"""Format-specific header preservation for zarr vectors stores.

When data is ingested from a format (TRK, TRX, SWC, CSV, OBJ, .h5ad, ...),
the source's metadata that the store's arrays cannot hold is kept in
``/headers/<format>/`` within the store.  Exporters read it back to
rebuild format-specific fields: a TRK or TRX export writes the reference
image it came with, a CSV export the column names, an ``.h5ad`` export the
categorical labels and cell barcodes.

Only ingest (and a staged attach, which extends the ``.h5ad`` header)
writes headers; an export reads them and adds none.  A store
built from one format therefore carries that format's header, and an
export to a different format uses whatever it can of it (a TRX export of a
TRK-ingested store takes its reference image from the TRK header).
"""

from zarr_vectors_tools.headers.registry import HeaderRegistry

__all__ = ["HeaderRegistry"]
