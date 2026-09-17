"""Process-pool executors for the parallel ingest / coarsening.

The core (`zarr_vectors`) coarsener and this package's ingest take an injectable
``executor`` — a ``map``-like callable ``executor(func, items, shared) -> list``
that applies a picklable ``func`` to each item, in order.  The serial default
runs in-process; :func:`process_pool_executor` returns one backed by the stdlib
``ProcessPoolExecutor`` (no extra dependency), and :func:`dask_executor` returns
one backed by a local Dask cluster (needs the ``parallel`` extra, and adds
``scatter`` broadcast of the shared payload).  Both use **process** workers —
the hot loops are pure-Python / GIL-bound, so threads would not scale.
"""
from __future__ import annotations

import os
import pickle
import tempfile
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from typing import Any, cast

#: The per-worker copy of the current shared payload, and the token naming
#: it.  Module globals rather than a closure because only module-level names
#: survive pickling to a spawned worker, which is how Windows and macOS start
#: processes.  A worker swaps the payload when a task arrives carrying a
#: token it has not seen, so one pool serves every phase of a run.
_SHARED: Any = None
_SHARED_TOKEN: str | None = None


def _set_shared(shared: Any, token: str | None = None) -> None:
    """Make ``shared`` the worker's current payload (also the pool initializer)."""
    global _SHARED, _SHARED_TOKEN
    _SHARED = shared
    _SHARED_TOKEN = token


def _load_shared(token: str, path: str | None) -> None:
    """Worker side: make the payload ``token`` names current, loading it once.

    ``path`` is the pickle the coordinator wrote for this token, or ``None``
    when the phase has no shared payload at all.
    """
    if _SHARED_TOKEN == token:
        return
    if path is None:
        _set_shared(None, token)
        return
    with open(path, "rb") as f:
        _set_shared(pickle.load(f), token)


def _call_with_shared(
    func: Callable[..., Any], item: Any,
    token: str | None = None, path: str | None = None,
) -> Any:
    """Apply ``func`` to one item with the worker's own copy of ``shared``.

    ``token=None`` keeps whatever payload is current (the pool-initializer
    form); the executor below always names one.
    """
    if token is not None:
        _load_shared(token, path)
    return func(item, shared=_SHARED)


def _map_chunksize(n_items: int, n_workers: int) -> int:
    """Items per IPC message: a few batches per worker, so load stays balanced
    across target chunks of very different sizes, but a run over thousands of
    small chunks is not paying a pickle round-trip for each one."""
    return max(1, min(32, n_items // max(1, n_workers * 4)))


@contextmanager
def process_pool_executor(workers: int | None = None):
    """Context manager yielding an ``executor(func, items, shared)`` callable
    backed by ONE stdlib :class:`concurrent.futures.ProcessPoolExecutor` of
    ``workers`` processes (default: cores-1).  No extra dependency.

    Usage::

        with process_pool_executor(8) as ex:
            build_pyramid(..., executor=ex)

    One pool for the life of the context.  It used to open a fresh pool for
    every distinct ``shared`` payload -- one per phase, three phases per
    pyramid level -- and keep them all open until the context closed, so a
    five-level TRK run with ``--workers 12`` spawned the interpreter (and
    re-imported numpy, zarr and this package) some 200 times and held every
    idle worker's memory to the end.  Spawn is what Windows and macOS use, and
    at a second or two per process it was a visible share of a small run and
    the largest resident-memory item of a large one.

    ``shared`` still reaches each worker ONCE per phase rather than being
    re-pickled into every task: the coordinator writes it to a temp file and
    tags every task with a token; a worker loads the file the first time it
    sees the token and keeps the payload until the token changes.  The
    coarseners' shared payload carries a per-object keep mask, so sending it
    per task pushed tens of gigabytes through the pipe on a 5M-object store.

    Suitable for the coarsener's per-chunk workers, which already write
    their cells with ``record_presence=False`` and rely on a coordinator
    manifest rebuild.
    """
    from concurrent.futures import ProcessPoolExecutor

    n = workers if workers and workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    state: dict[str, Any] = {"pool": None, "seq": 0}
    tag = f"{os.getpid()}-{id(state):x}"

    def executor(
        func: Callable[..., Any], items: Iterable[Any], shared: Any = None,
    ) -> list:
        items = list(items)
        if not items:
            return []
        pool = state["pool"]
        if pool is None:
            pool = state["pool"] = ProcessPoolExecutor(max_workers=n)
        state["seq"] += 1
        token = f"{tag}-{state['seq']}"
        path: str | None = None
        if shared is not None:
            fd, path = tempfile.mkstemp(prefix="zv_shared_", suffix=".pkl")
            with os.fdopen(fd, "wb") as f:
                pickle.dump(shared, f, protocol=pickle.HIGHEST_PROTOCOL)
        try:
            k = len(items)
            return list(pool.map(
                _call_with_shared, [func] * k, items, [token] * k, [path] * k,
                chunksize=_map_chunksize(k, n),
            ))
        finally:
            # Every task of this phase has returned, so every worker that
            # needed the payload has loaded it; nothing reads the file again.
            if path is not None:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    try:
        yield executor
    finally:
        if state["pool"] is not None:
            state["pool"].shutdown(wait=True)


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
