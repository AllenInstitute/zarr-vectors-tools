"""Process-pool executors for the parallel ingest / coarsening.

The core (`zarr_vectors`) coarsener and this package's ingest take an injectable
``executor`` — a ``map``-like callable ``executor(func, items) -> list`` that
applies a picklable ``func`` to each item, in order.  The serial default runs
in-process; :func:`process_pool_executor` returns one backed by the stdlib
``ProcessPoolExecutor`` (no extra dependency), and :func:`dask_executor` returns
one backed by a local Dask cluster (needs the ``parallel`` extra, and adds
``scatter`` broadcast of the shared payload).  Both use **process** workers —
the hot loops are pure-Python / GIL-bound, so threads would not scale.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from typing import Any, cast

#: The per-worker copy of the shared payload, set once at process start by
#: :func:`_set_shared` and read by :func:`_call_with_shared`.  A module global
#: rather than a closure because only module-level names survive pickling to a
#: spawned worker, which is how Windows and macOS start processes.
_SHARED: Any = None


def _set_shared(shared: Any) -> None:
    """Pool initializer: keep ``shared`` for the life of the worker."""
    global _SHARED
    _SHARED = shared


def _call_with_shared(func: Callable[..., Any], item: Any) -> Any:
    """Apply ``func`` to one item with the worker's own copy of ``shared``."""
    return func(item, shared=_SHARED)


@contextmanager
def process_pool_executor(workers: int | None = None):
    """Context manager yielding an ``executor(func, items, shared)`` callable
    backed by a stdlib :class:`concurrent.futures.ProcessPoolExecutor` of
    ``workers`` processes (default: cores-1).  No extra dependency.

    Usage::

        with process_pool_executor(8) as ex:
            build_pyramid(..., executor=ex)

    ``shared`` is sent to each worker ONCE, at pool start, rather than
    re-pickled into every task -- ``pool.map(partial(func, shared=shared))``
    serialises it per item, and the coarseners' shared payload carries a
    per-object keep mask, so a 5M-object store dispatched over 10k target
    chunks pushed tens of gigabytes through the pipe to send the same array
    10k times.  ``dask_executor`` already avoided this with ``scatter``.

    Suitable for the coarsener's per-chunk workers, which already write
    their cells with ``record_presence=False`` and rely on a coordinator
    manifest rebuild.
    """
    from concurrent.futures import ProcessPoolExecutor

    n = workers if workers and workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    # One pool per distinct ``shared`` payload, created lazily: the payload
    # is fixed at worker start, so a caller that dispatches two different
    # phases (each with its own shared dict) gets a pool per phase.  Pools
    # are closed when the context manager exits.
    pools: dict[int, Any] = {}

    def executor(
        func: Callable[..., Any], items: Iterable[Any], shared: Any = None,
    ) -> list:
        items = list(items)
        if not items:
            return []
        key = id(shared)
        pool = pools.get(key)
        if pool is None:
            pool = ProcessPoolExecutor(
                max_workers=n,
                initializer=_set_shared,
                initargs=(shared,),
            )
            pools[key] = pool
        return list(pool.map(_call_with_shared, [func] * len(items), items))

    try:
        yield executor
    finally:
        for pool in pools.values():
            pool.shutdown(wait=True)


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
        n_workers=n, threads_per_worker=1, processes=True, dashboard_address="127.0.0.1:8787",
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
                return cast(list[Any], client.gather(client.map(func, items)))
            # Preferred path: scatter shared data once.
            # If the shared object is not safely serializable (or triggers a
            # transport/protocol size issue), fall back to passing it directly.
            # Callers should keep `shared` lightweight where possible.
            try:
                sh = client.scatter(shared, broadcast=True)
                return cast(
                    list[Any],
                    client.gather(client.map(func, items, shared=sh)),
                )
            except Exception:
                return cast(
                    list[Any],
                    client.gather(client.map(func, items, shared=shared)),
                )

        yield executor
    finally:
        client.close()
        cluster.close()
