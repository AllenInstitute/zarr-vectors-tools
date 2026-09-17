"""Format conversion: files into zarr-vectors stores, and stores back out.

Two submodules, one per direction:

* :mod:`zarr_vectors_tools.convert.ingest` -- one module per readable format
  (``csv_points``, ``trk_parallel``, ``swc``, ``gifti``, ...), each ending in
  a core ``write_*`` call, plus the shared enrichment helpers and the
  process-pool executors (``_parallel``) the parallel paths run on.
* :mod:`zarr_vectors_tools.convert.export` -- one module per writable format
  (``trk``, ``ply``, ``swc``, ``obj``, ...), each reading a store level and
  writing a file.

``zvtools convert`` dispatches to both through the registries in
:mod:`zarr_vectors_tools.cli._args`.  Neither submodule re-exports: import
from the concrete module, e.g.
``from zarr_vectors_tools.convert.ingest.csv_points import ingest_csv``.
"""
