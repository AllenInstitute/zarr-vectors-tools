# Parallel and concurrent workflows

Every parallel path in `zarr-vectors-tools` — ingest and pyramid alike —
goes through a single, deliberately small abstraction: an injectable
**executor**. There is no framework to learn and nothing to configure
globally. You pass an executor in, or you pass `None` and everything runs
serially in-process.

:::{note}
**There is no `asyncio` in this package.** No coroutines, no event loop,
no `async def` entry points. The concurrency model is process-parallel
work over a chunk grid, plus one thread pool for network reads. If you
are looking for an `await`-able API, there isn't one — wrap the
synchronous calls in `asyncio.to_thread` if you need to drive them from
an async application.
:::

## The executor contract

```python
executor(func, items, shared=None) -> list
```

Apply a **picklable** `func` to each item and return the results **in
input order**. That is the whole interface. `None` means "no executor" —
run serially in the calling process.

Order preservation matters more than it looks: every parallel coarsener
in this package defines a serial fallback with identical semantics, so
serial and parallel runs produce the same store *by construction*, not by
luck. The test suite pins this — `test_skeleton_coarsen.py`,
`test_polyline_coarsen_chunk_local.py`, `test_precomputed_skeletons.py`
and `test_single_array_enumeration.py` each assert that serial output
equals parallel output.

## The two backends

Both live in `zarr_vectors_tools/ingest/_parallel.py` and are context
managers. Both default to `max(1, cpu_count() - 1)` workers.

| | `process_pool_executor` | `dask_executor` |
| --- | --- | --- |
| Backing | stdlib `concurrent.futures.ProcessPoolExecutor` | `dask.distributed.LocalCluster` |
| Extra required | none | `parallel` |
| `shared` handling | re-pickled into every task | `client.scatter(broadcast=True)` **once** |
| Dashboard | — | `127.0.0.1:8787` |
| Best for | most work; the default | dense levels with a large `shared` payload |

```python
from zarr_vectors_tools.ingest._parallel import process_pool_executor
from zarr_vectors_tools.multiresolution.coarsen import build_pyramid

# Stdlib backend — no extra dependency.
with process_pool_executor(8) as ex:
    build_pyramid(store_path, factors=[(8.0, 2.0), (8.0, 4.0)], executor=ex)
```

```python
from zarr_vectors_tools.ingest._parallel import dask_executor

# Dask backend — needs pip install 'zarr-vectors-tools[parallel]'
with dask_executor(12) as ex:
    build_pyramid(store_path, factors=[(8.0, 2.0), (8.0, 4.0)], executor=ex)
```

### Why `shared` exists

`shared` is the data common to every task — a source reader, or a
pyramid level's plan. The stdlib backend re-pickles it into each task
payload, so keep it lightweight there.

The dask backend scatters it once. That path is not a micro-optimisation:
without it, levels with few tasks but bulky shared state were dominated
by serialising the same object N times, which surfaced as dask's
`UserWarning: Sending large graph`. The scatter is wrapped in a
`try`/`except` that falls back to per-task passing, so a shared object
that does not survive `scatter` degrades in speed rather than failing.

## Processes, not threads

Both backends use **process** workers. The hot loops — RDP simplification,
forest traversal, fragment binning — are pure Python and GIL-bound, so
threads would not scale. This is a deliberate choice, stated in
`_parallel.py`'s module docstring.

### The one exception: I/O-bound cloud reads

`ingest/precomputed_plain_skeletons.py` uses a `ThreadPoolExecutor` for
fetching skeletons from a precomputed layer. That work is network-bound,
not GIL-bound, so threads are the correct tool. It is a **separate knob**
from the pyramid's process workers:

```python
from zarr_vectors_tools.ingest.precomputed_plain_skeletons import run_ingest_plain

run_ingest_plain(
    source,
    out_store,
    read_workers=32,     # threads — concurrent cloud reads
    pyramid_workers=8,   # processes — coarsening
)
```

Tune `read_workers` against your network and the remote store's rate
limits; tune `pyramid_workers` against your core count. Raising the
former does nothing for coarsening speed, and raising the latter does
nothing for fetch speed.

## From the command line

`zvtools convert` and `zvtools pyramid` both expose the same two flags:

```bash
zvtools convert tracts.trk out.zarrvectors \
    --num-chunks 5000 --workers 12 --workers-backend dask \
    --coarsen 8,8 --sparsity 2,4
```

`--workers-backend` picks `process` (default) or `dask`. Internally
`executor_ctx(workers, backend)` in `cli/_args.py` yields `None` unless
`workers > 1`, so `--workers 1` is genuinely serial rather than a
one-worker pool.

## Concurrency safety: how writers avoid racing

Per-chunk arrays record which chunks are non-empty in a single
**array-wide** attribute. Two workers writing disjoint cells still write
the same `zarr.json`, so a naive parallel write loses keys — the loser's
update silently vanishes.

The protocol that avoids this has two halves:

1. **Workers write with `record_presence=False`.** They touch only their
   own cell data and never update the shared manifest attribute.
2. **A single-process coordinator rebuilds afterwards** —
   `rebuild_nonempty_manifests` from `zarr_vectors_tools/_manifests.py`,
   plus core's `finalize_links`.

`CROSS_LINK_TASK_SHARD_AXIS = 4` (in `multiresolution/constants.py`)
bounds how many endpoint pairs a single cross-link task takes on. It is a
task-granularity knob only — it is explicitly **not** Zarr sharding.

:::{warning}
One residual race is documented rather than fixed: concurrently
*creating* the same offsets array is still time-of-check/time-of-use
unsafe. The workaround is structural — the coordinator pre-creates every
offsets array before dispatching any worker. If you write your own
parallel driver against these internals, preserve that ordering. See
[the upstream findings](../upstream/links-merge-findings.md).
:::

## Writing your own driver

If you drive ingest or coarsening from a script rather than the CLI, two
rules apply.

**Guard the entry point.** Both backends spawn processes, and on Windows
and macOS spawn re-imports the `__main__` module. Without the guard you
get an infinite fork bomb:

```python
if __name__ == "__main__":
    with dask_executor(8) as ex:
        run_ingest(source, out_store, anchor=anchor, executor=ex)
```

`scripts/ingest_cutout_parralel.py` is a working example.

**Levels are sequential; chunks are parallel.** `build_pyramid` cannot
overlap levels — level *i+1* reads level *i*. Parallelism happens *within*
a level, across target chunks. So a pyramid over few, large chunks
parallelises poorly no matter how many workers you give it; if that is
your situation, the fix is a smaller `chunk_shape`, not more workers.

## See also

- [Compressor choice](compressors.md) — the other half of write throughput.
- [Large-scale pipelines](large_scale_pipelines.md) — composing both.
- [Tractography at scale](../ingest/tractography_at_scale.md) — the parallel TRK path.
- [Skeletons in EM](../ingest/em_skeletons.md) — the thread/process split in practice.
- [Building pyramids](../multiresolution/building_pyramids.md)
