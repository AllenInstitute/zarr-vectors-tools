"""Dask-backed executor for the parallel ingest.

The core (`zarr_vectors`) coarsener and this package's ingest take an injectable
``executor`` — a ``map``-like callable ``executor(func, items) -> list`` that
applies a picklable ``func`` to each item, in order.  The serial default runs
in-process; :func:`dask_executor` returns one backed by a local Dask cluster of
**process** workers (the hot loops are pure-Python / GIL-bound, so threads would
not scale).
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Callable, Iterable


@contextmanager
def dask_executor(workers: int | None = None):
    """Context manager yielding an ``executor(func, items)`` callable backed by
    a local Dask cluster of ``workers`` process workers (default: cores-1).

    Usage::

        with dask_executor(12) as ex:
            run_ingest(..., executor=ex)

    Requires the ``parallel`` extra (``pip install 'zarr-vectors-tools[parallel]'``).
    """
    try:
        from dask.distributed import Client, LocalCluster
    except ImportError as e:  # pragma: no cover - dependency guard
        raise ImportError(
            "dask_executor requires the 'parallel' extra: "
            "pip install 'zarr-vectors-tools[parallel]'"
        ) from e

    n = workers if workers and workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    cluster = LocalCluster(
        n_workers=n, threads_per_worker=1, processes=True, dashboard_address=None,
    )
    client = Client(cluster)
    try:
        def executor(
            func: Callable[..., Any], items: Iterable[Any], shared: Any = None,
        ) -> list:
            """Apply ``func(item, shared=shared)`` to each item in order.

            ``shared`` is data common to every task (e.g. the source reader, or a
            pyramid level's plan).  It is ``scatter``-ed to the workers **once**
            (broadcast) rather than re-pickled into every task payload — without
            this, dense/few-task levels are dominated by serializing the same
            bulk object N times (the per-task ``UserWarning: Sending large
            graph`` regression).
            """
            items = list(items)
            if not items:
                return []
            if shared is None:
                return client.gather(client.map(func, items))
            sh = client.scatter(shared, broadcast=True)
            return client.gather(client.map(func, items, shared=sh))

        yield executor
    finally:
        client.close()
        cluster.close()
