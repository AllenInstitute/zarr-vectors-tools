"""Executors for the parallel ingest and pyramid paths.

Pass one as ``executor=`` to an ingest or to
:func:`~zarr_vectors_tools.multiresolution.coarsen.build_pyramid`::

    from zarr_vectors_tools.parallel import process_pool_executor

    with process_pool_executor(8) as ex:
        build_pyramid("cells.zv", factors=[(2, 2)], executor=ex)

``process_pool_executor`` needs nothing beyond the standard library;
``dask_executor`` needs the ``parallel`` extra.  Both are what
``zvtools --workers N [--workers-backend dask]`` uses.
"""

from zarr_vectors_tools.convert.ingest._parallel import (
    dask_executor,
    process_pool_executor,
)

__all__ = ["dask_executor", "process_pool_executor"]
