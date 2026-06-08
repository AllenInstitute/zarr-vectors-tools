"""zarr-vectors-tools: ingest and export workflows for the zarr vectors format.

The core read/write APIs live in the ``zarr_vectors`` package. This package
provides format-conversion workflows that wrap third-party libraries
(nibabel, laspy, plyfile, trimesh, networkx, trx-python), the multi-scale
coordination + compute layer (``zarr_vectors_tools.multiresolution``), and the
example notebooks that demonstrate end-to-end pipelines.

Importing this package registers the multi-scale coordinators with core:

- the coarsener registry (``register_coarsener("skeleton"/"per_object", ...)``)
  is populated by importing ``...multiresolution.coarsen``, so
  ``coarsen_level`` / ``build_pyramid`` dispatch works; and
- the pyramid-refresh hook is filled via
  ``zarr_vectors.ops.register_pyramid_refresher`` so
  ``EditSession(refresh_pyramid=...)`` can re-coarsen after edits.
"""


def _register_multiscale_with_core() -> None:
    """Wire the tools coordination layer into core's injection slots (idempotent)."""
    # Importing coarsen runs its bottom-of-module register_coarsener(...) calls.
    from zarr_vectors_tools.multiresolution import coarsen as _coarsen  # noqa: F401
    from zarr_vectors.ops import register_pyramid_refresher
    from zarr_vectors_tools.multiresolution.refresh import rebuild_pyramid_from_level

    register_pyramid_refresher(rebuild_pyramid_from_level)


_register_multiscale_with_core()
