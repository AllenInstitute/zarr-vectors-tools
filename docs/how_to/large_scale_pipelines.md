# Large-scale pipelines

Ingesting a whole-brain tractogram or a connectomics cutout is a
different job from converting a test file. This page composes the
individual decisions — chunking, parallelism, compression, pyramid
factors — into working pipelines, and covers what to do when one dies
halfway.

The `scripts/` directory in the repository holds the real drivers these
examples are drawn from.

## The decisions, in order

Three of them are fixed at creation and cannot be revisited without
re-ingesting. Make them deliberately:

| Decision | Fixed at creation? | Reference |
| --- | --- | --- |
| `chunk_shape` / `--num-chunks` | **yes** | [Choosing chunk and bin shape](choose_chunk_and_bin.md) |
| compressor | **yes**, per array | [Choosing a compressor](compressors.md) |
| dtype | **yes** | |
| worker count and backend | no | [Parallel workflows](parallelism.md) |
| pyramid factors | no — rebuildable | [Coarsening versus sparsity](../multiresolution/concepts.md) |

## Pipeline: a large tractogram

```bash
zvtools convert tracts.trk tracts.zarrvectors \
    --num-chunks 5000 \
    --n-parts 48 \
    --workers 12 --workers-backend dask \
    --compressor zstd \
    --coarsen 8,8,8 --sparsity 2,4,8 --sparsity-strategy length
```

TRK is the only format with a fully parallel ingest and an inline pyramid
build, so this single command is the whole pipeline. `--n-parts` sets
file-split granularity and defaults to `4 × workers`; raising it improves
load balance on uneven tractograms at the cost of more intermediate
files.

`--sparsity-strategy length` matters here beyond speed: at level 3 you are
keeping an eighth of the streamlines, and keeping the *longest* eighth
gives a far more recognisable low-resolution view than a random eighth.

See [Tractography at scale](../ingest/tractography_at_scale.md).

## Pipeline: an EM skeleton cutout

Precomputed skeleton ingest is Python-only — it is not in the `zvtools`
format registry. The shape of a driver script:

```python
from zarr_vectors_tools.ingest._parallel import dask_executor
from zarr_vectors_tools.ingest.precomputed_skeletons import (
    PrecomputedFragsReader, enumerate_frag_keys, run_ingest,
)

if __name__ == "__main__":                      # required: workers re-import __main__
    reader = PrecomputedFragsReader("gs://flywire_v141_m783/skeletons_mip_1")
    # `keys` and `bounds_nm` are both derived from a real .frags chunk
    # corner — see the worked example for how they are computed.
    keys, bounds = enumerate_frag_keys(reader.info, anchor, counts), bounds_nm

    with dask_executor(8) as ex:
        run_ingest(
            reader,
            "cutout.zarrvectors",
            keys,
            bounds_nm=bounds,
            drop_interior_below=3,              # discard tiny chunk-interior fragments
            executor=ex,
        )
```

`run_ingest` takes a **reader object and an explicit key list**, not a URL
and a bounding box — the keys are enumerated from an anchor rather than
listed, because a bucket-wide listing on a production EM layer is
prohibitively slow. [Skeletons in EM](../ingest/em_skeletons.md) has the
full worked example, including how `anchor`, `counts` and `bounds_nm` are
computed from the layer's `info`.

:::{warning}
The `if __name__ == "__main__":` guard is **not optional**. Both executor
backends spawn processes, and spawn re-imports the `__main__` module.
Without the guard you get an unbounded fork bomb rather than an error.
:::

The plain (no spatial index) path splits its concurrency in two, because
the two phases are bound by different resources:

```python
from zarr_vectors_tools.ingest.precomputed_plain_skeletons import run_ingest_plain

run_ingest_plain(
    "precomputed://gs://allen_neuroglancer_ccf/Mouselight",
    "mouselight.zarrvectors",
    read_workers=32,      # threads — network-bound cloud reads
    pyramid_workers=8,    # processes — GIL-bound coarsening
)
```

Tune `read_workers` against your network and the remote store's rate
limits; tune `pyramid_workers` against your core count. See [Skeletons in
EM](../ingest/em_skeletons.md).

## Ingest and pyramid as separate stages

For anything long-running, split the two. A pyramid can be rebuilt; an
ingest cannot be resumed:

```bash
# Stage 1 — expensive, do it once.
zvtools convert huge.trk out.zarrvectors --num-chunks 5000 \
    --workers 12 --compressor zstd

# Stage 2 — cheap to redo with different factors.
zvtools pyramid out.zarrvectors --coarsen 8,8 --sparsity 2,4 \
    --workers 12 --compressor zstd
```

:::{warning}
`zvtools pyramid --compressor` defaults to `none` independently of what
level 0 was written with. Omit it here and you get compressed level-0
data under raw coarse levels. Pass it every time.
:::

## Recovering from a failed pyramid

Pyramid builds are the usual casualty of a long run — they are the most
memory-hungry stage, and the OOM killer finds them. Levels are written in
sequence, so completed levels are intact and only the level in progress
is partial.

The recovery is to delete the first incomplete level and rebuild from the
last good one, which is what `scripts/resume_trk_pyramid.py` does. In
Python:

```python
from zarr_vectors_tools.multiresolution.refresh import rebuild_pyramid_from_level

# Levels 0-2 are good; 3 was killed mid-write. Delete level 3 on disk first,
# then rebuild everything above level 2 using each level's stored factors.
rebuild_pyramid_from_level(root, source_level=2)
```

`rebuild_pyramid_from_level` reuses each level's stored `bin_ratio`,
`object_sparsity`, `chunk_shape` and `coarsening_method`, so the result
is equivalent to a from-scratch build rather than an approximation. Pass
`sparsity_strategy=` (and `compressor=`) to match the original build —
neither is recorded on disk. A skeleton pyramid also needs its stride,
which nothing records: pass `coarsen_factors={level: stride}` or the
refresh refuses. See [Refresh](../multiresolution/refresh.md).

To avoid the OOM in the first place: more chunks (smaller `chunk_shape`)
lowers peak memory per task, because the parallel coarseners are
chunk-local with peak memory on the order of one target chunk.

## Why parallelism plateaus

Levels are sequential — level *i+1* reads level *i* — so parallelism
happens **within** a level, across target chunks. Two consequences:

- A store with few, large chunks parallelises poorly no matter how many
  workers you add. The fix is a smaller `chunk_shape`, decided at ingest.
- Adding workers past the per-level chunk count does nothing.

If a pyramid build is not scaling, check `zvtools info` for the chunk
count before adding workers.

## Validate before you rely on it

```bash
zvtools validate out.zarrvectors --level 3
```

Returns `0` on conformance and `1` otherwise, so it drops straight into
CI or the end of a driver script. Worth running after any pipeline that
was interrupted and resumed.

## See also

- [Parallel workflows](parallelism.md)
- [Choosing a compressor](compressors.md)
- [Choosing chunk and bin shape](choose_chunk_and_bin.md)
- [Tractography at scale](../ingest/tractography_at_scale.md)
- [Skeletons in EM](../ingest/em_skeletons.md)
- [Refresh](../multiresolution/refresh.md)
