# Large data

## Workers

```bash
zvtools convert tracts.trk tracts.zv --num-chunks 1000 --workers 8
zvtools pyramid cells.zv --coarsen 2,2 --sparsity 2,2 --workers 8 --workers-backend dask
```

Without `--workers` (or with `--workers 1`) every step runs in one process,
the TRK ingest included. `--workers N` runs N worker processes for:

| Input | Runs in parallel |
|---|---|
| trk | reading the file in parts, writing level-0 chunks, the pyramid |
| precomputed skeletons with a spatial index | reading and writing each `.frags` chunk, the pyramid |
| precomputed skeletons without one | the pyramid (skeletons are fetched by threads, below) |
| any other input | the pyramid only; the ingest itself runs in one process |

`--workers-backend dask` runs a local Dask cluster (`[parallel]` extra) instead
of the default standard-library process pool. `zvtools merge` and `split` run in
one process, a merge's pyramid rebuild included: merge large stores with
`--pyramid drop`, then run `zvtools pyramid --workers N`. Levels are built in
turn and workers share one level's chunks, so few chunks means little speed-up
([Store layout](store_layout.md#chunk-shape), [Pyramids](pyramids.md)).

## Large tractograms

```bash
zvtools convert tracts.trk tracts.zv --num-chunks 1000 --workers 8 --n-parts 64 \
    --scratch-dir scratch/tracts --coarsen 1,1,1 --sparsity 2,2,2 \
    --rdp-tolerance 0.5,1,2 --chunk-scale 2,2,2
```

The TRK ingest sorts the file's byte ranges ("parts") into chunks on scratch
disk, then writes level 0 chunk by chunk: memory scales with a part, not the file.

| Option | Meaning |
|---|---|
| `--num-chunks N` | Target total chunk count (default 125), or `X,Y,Z` per axis. It sets the chunk edge in whole mm, from the first, middle and last point of every streamline, so the grid lands near N: 1000 gave a 6 × 16 × 12 grid (1,152 chunks of 8-9 mm) on a 2.4 M-vertex test file. |
| `--n-parts N` | Parts the file is cut into (default 4 × workers, at least 16). More parts balance uneven files better. |
| `--scratch-dir DIR` | Keep the parts and a progress record (`trk_parallel_run.json`) in DIR, not a temporary directory. DIR is kept afterwards; delete it when done. It took 36 MB for a 28 MB TRK. |
| `--resume` | Rerun the same command with this after a run stops; Ctrl-C stops a run and its workers at once. It reuses the file scan, the parts, a complete level 0 and finished pyramid levels, and rebuilds a half-written level. Changed options are refused. |

## Precomputed layers and Python scripts

`zvtools convert` fetches a precomputed skeleton layer without a spatial index
with 8 threads, 256 skeletons at a time. Python sets both:

```python
from zarr_vectors_tools.convert.ingest.precomputed_plain_skeletons import (
    PlainPrecomputedReader,
    run_ingest_plain,
)
from zarr_vectors_tools.parallel import process_pool_executor

if __name__ == "__main__":
    with process_pool_executor(8) as ex:
        run_ingest_plain(
            PlainPrecomputedReader("layer/skeletons"),  # directory or gs:// URL
            "skeletons.zv",
            chunk_shape_nm=(1_000_000.0, 1_000_000.0, 1_000_000.0),
            read_workers=32,       # threads fetching skeletons
            batch_size=256,        # skeletons held in memory at once
            scratch_dir="scratch", # existing directory for spilled batches
            strides=[2, 2],        # two coarser levels
            executor=ex,           # pyramid worker processes
        )
```

Raise `read_workers` until the network or the server's rate limit is the
bottleneck; `batch_size` sets peak memory, not the result
([Skeletons](convert/skeletons.md)). Worker processes re-import the script that
starts them, so put any parallel ingest or pyramid call under
`if __name__ == "__main__":`. `zarr_vectors_tools.parallel` has
`process_pool_executor(n)` and `dask_executor(n)`; pass either as `executor=`
to an ingest or to `build_pyramid("cells.zv", factors=[(2, 2)], executor=ex)`.

## Writing to the cloud

Build the store on local disk, shard it, then upload it:

```bash
zvtools shard tracts.zv --shape 8        # or convert ... --shard 8
gsutil -m rsync -r tracts.zv gs://my-bucket/tracts.zv
```

Sharding packs neighbouring chunks into one file per array: the 2.4 M-vertex
store above went from 2,343 files to 242 ([Store layout](store_layout.md#sharding);
serving it: [Visualise](visualise.md)).

## Recipes: zvtools run

A recipe lists `zvtools` steps to run in order and resume after a failure:

```yaml
name: two-subjects
steps:
  - convert: {input: data/subject_a.trk, output: out/tracts.zv, num_chunks: 64,
              workers: 4}
  - merge: {target: out/tracts.zv, sources: [data/subject_b.trk],
            on_out_of_bounds: expand, pyramid: drop}
  - pyramid: {store: out/tracts.zv, coarsen: [1, 1], sparsity: [2, 2],
              rdp_tolerance: [0.5, 1], chunk_scale: [2, 2]}
  - validate: {store: out/tracts.zv}
  - shard: {store: out/tracts.zv, shape: 8}
```

```bash
zvtools run tracts.yaml --dry-run   # print and check each step's command
zvtools run tracts.yaml             # run; a rerun skips finished steps
zvtools run tracts.yaml --from 3    # rerun step 3 and everything after it
zvtools run tracts.yaml --force     # rerun every step
```

- Each step is a subcommand (`convert`, `pyramid`, `attach`, `merge`, `split`,
  `validate`, `info`, `shard`, `bundles`, `synapses`). Keys are its long
  options with underscores for dashes, plus its positional arguments by name
  (`input`, `output`, `store`, `target`, `sources`); an unknown key is an error.
- Lists become comma-separated values (`coarsen: [1, 1]` is `--coarsen 1,1`) or
  repeated flags; `true` turns a switch on. Relative paths resolve against the
  recipe's directory (or `workdir:`).
- Progress is kept in `tracts.run.json` beside the recipe. A finished step is
  skipped if its options are unchanged and its outputs exist; after a step
  reruns, all later ones do. The run stops at the first failure.
- Rerunning a `convert` or `pyramid` step it ran before, `zvtools run` adds
  `--overwrite` or `--replace` itself, so its earlier output is replaced.
- Recipes are `.yaml` (needs the `[recipes]` extra), `.toml` or `.json`. The
  repository's `examples/recipes/` has tractography, EM skeleton, surface and
  cell recipes.
