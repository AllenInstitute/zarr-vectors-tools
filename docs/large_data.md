# Large data

Run conversions and pyramids on several processes, resume a long TRK ingest,
upload a store, and chain steps in a recipe.

## Workers

```bash
zvtools convert tracts.trk tracts.zv --num-chunks 1000 --workers 8
zvtools pyramid cells.zv --coarsen 2,2 --sparsity 2,2 --workers 8 --workers-backend dask
```

`--workers N` runs N processes. For TRK and precomputed skeleton layers with a
spatial index, they read and write level 0 as well as the pyramid; for other
inputs, only the pyramid. `--workers-backend dask` uses a local Dask cluster
(`[parallel]` extra). Workers share one level's chunks, so few chunks give
little speed-up.

`zvtools merge` and `split` run in one process. To merge large stores, merge
with `--pyramid drop`, then run `zvtools pyramid --workers N`.

## Large tractograms

```bash
zvtools convert tracts.trk tracts.zv --num-chunks 1000 --workers 8 --n-parts 64 \
    --scratch-dir scratch/tracts --coarsen 1,1,1 --sparsity 2,2,2 \
    --rdp-tolerance 0.5,1,2 --chunk-scale 2,2,2
```

The TRK ingest cuts the file into parts and sorts them into chunks on scratch
disk, so memory scales with a part, not the file.

| Option | Meaning |
|---|---|
| `--n-parts N` | Parts to cut the file into (default 4 × workers, at least 16). |
| `--scratch-dir DIR` | Keep the parts and progress record in DIR. DIR is kept afterwards. |
| `--resume` | Rerun the same command with this after a run stops. It reuses finished work; changed options are refused. |

## Python scripts

`zarr_vectors_tools.parallel` has `process_pool_executor(n)` and
`dask_executor(n)`. Each is a context manager; pass what it yields as
`executor=` to an ingest or to `build_pyramid`. Worker processes re-import the
script, so put the call under `if __name__ == "__main__":`.

```python
from zarr_vectors_tools.multiresolution.coarsen import build_pyramid
from zarr_vectors_tools.parallel import process_pool_executor

if __name__ == "__main__":
    with process_pool_executor(8) as ex:
        build_pyramid("tracts.zv", factors=[(1, 2)], executor=ex)
```

`zvtools convert` fetches a precomputed skeleton layer without a spatial index
with 8 threads, 256 skeletons at a time. To change these, call
`run_ingest_plain` (`convert.ingest.precomputed_plain_skeletons`) with
`read_workers` and `batch_size`; `batch_size` sets peak memory.

## Writing to the cloud

Build the store on local disk, [shard](store_layout.md#sharding) it, then
upload it:

```bash
zvtools shard tracts.zv --shape 8
gsutil -m rsync -r tracts.zv gs://my-bucket/tracts.zv
```

## Recipes: zvtools run

A recipe lists `zvtools` steps to run in order and resume after a failure:

```yaml
name: two-subjects
steps:
  - convert: {input: data/subject_a.trk, output: out/tracts.zv, num_chunks: 64}
  - merge: {target: out/tracts.zv, sources: [data/subject_b.trk], pyramid: drop}
  - pyramid: {store: out/tracts.zv, coarsen: [1, 1], sparsity: [2, 2],
              rdp_tolerance: [0.5, 1], chunk_scale: [2, 2]}
  - shard: {store: out/tracts.zv, shape: 8}
```

```bash
zvtools run tracts.yaml --dry-run   # print and check each step's command
zvtools run tracts.yaml             # run; a rerun skips finished steps
zvtools run tracts.yaml --from 3    # rerun step 3 and everything after it
```

- Each step is a subcommand. Keys are its long options with `_` for `-`, plus
  its positional arguments by name. Lists become comma-separated values or
  repeated flags.
- Relative paths in positional arguments and in `nodes`, `lut`, `transform`
  and `scratch_dir` resolve against the recipe's directory.
- Progress is kept in `tracts.run.json`. A step is skipped if its options are
  unchanged and its outputs exist; after a step reruns, all later ones do.
- Recipes are `.yaml` (`[recipes]` extra), `.toml` or `.json`. The repository's
  `examples/recipes/` has examples.
