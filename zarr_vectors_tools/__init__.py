"""zarr-vectors-tools: ingest and export workflows for the zarr vectors format.

The core read/write APIs live in the ``zarr_vectors`` package. This package
provides format-conversion workflows that wrap third-party libraries
(nibabel, laspy, plyfile, trimesh, networkx, trx-python), the multi-scale
coordination + compute layer (``zarr_vectors_tools.multiresolution``), and the
example notebooks that demonstrate end-to-end pipelines.

Importing this package registers the multi-scale coordinators with core:

- the coarsener registry (``register_coarsener("skeleton"/"per_object", ...)``)
  is populated by importing ``...multiresolution.coarsen``, so
  ``coarsen_level`` / ``build_pyramid`` dispatch works.

Under zarr-vectors-py 0.8.1, pyramid refresh after ``zarr_vectors.ops.edit``
edits is handled natively by ``zarr_vectors.ops.refresh``/
``rebuild_pyramid_from_level`` — the ``register_pyramid_refresher``
injection slot this package used to fill on 0.8.0 no longer exists, so
there is nothing to wire here for that feature.
"""


def _register_multiscale_with_core() -> None:
    """Wire the tools coordination layer into core's injection slots (idempotent)."""
    # Importing coarsen runs its bottom-of-module register_coarsener(...) calls.
    from zarr_vectors_tools.multiresolution import coarsen as _coarsen  # noqa: F401


_register_multiscale_with_core()
