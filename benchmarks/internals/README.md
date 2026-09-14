# Internals benchmarks

Notebooks that sweep the `zarr-vectors` write/read/edit path against
**itself** along one axis at a time. Nothing here compares against
another file format — for that, see [`../formats/`](../formats/).

| Notebook | Axis swept | Fixed |
|----------|-----------|-------|
| [`01_size_scaling.ipynb`](01_size_scaling.ipynb) | `N ∈ {10³, 10⁴, 10⁵, 10⁶}` | point cloud, local backend |
| [`02_data_types.ipynb`](02_data_types.ipynb) | geometry type (all six) | `N = 50 000`, local backend |
| [`03_backends.ipynb`](03_backends.ipynb) | backend | point cloud, `N = 100 000`; `local` always, `obstore`/`fsspec` when `ZV_BENCH_S3_URL` is set |
| [`04_pyramid_scaling.ipynb`](04_pyramid_scaling.ipynb) | coarsening factor × build mode | points + graphs, `build_pyramid` |
| [`05_bbox_queries.ipynb`](05_bbox_queries.ipynb) | geometry type | spatial bbox query via `lazy.open_zv` |
| [`06_chunk_shape.ipynb`](06_chunk_shape.ipynb) | `chunk_shape` | points at fixed `N` |
| [`07_compression.ipynb`](07_compression.ipynb) | compressor / codec × `N` | points |
| [`08_edit_operations.ipynb`](08_edit_operations.ipynb) | edit kind, atomicity, `N_edits`, concurrency, index scaling | `N = 50 000` baseline points |

Each notebook follows the same shape: **setup → sweep → table →
plot**. ~10 cells, ~1 plot, no surprises.

Notebook 08 is the widest: `kind ∈ {move_in_chunk, move_cross_chunk,
add, soft_delete}`, `atomic ∈ {True, False}`, `N_edits ∈ {1, 10, 100,
1 000}`. A multi-writer sub-row covers 1 vs 4 cooperating editors on
disjoint chunks via `oid_prefix=`, and an index-scaling sub-row sweeps
`N_OBJECTS ∈ {1k, 10k, 50k}` to document the flatness of the
post-Iter4 fragment→OID inverted index.

Notebooks 01 and 02 average each measurement over `N_RUNS = 10`
repeats and plot the mean with a shaded Student's-t 95 % CI band
(`T95_DF9 = 2.262`, hard-coded to avoid a scipy dependency).
Notebook 03 is still a single-run sketch.

## Running

```bash
pip install -e ".[all]" jupyter matplotlib
jupyter lab benchmarks/internals/
```

Then open one and run all cells. Expected runtime on a laptop:

- `01_size_scaling`: a few minutes (the 1 M case dominates)
- `02_data_types`: ~30 s
- `03_backends`: ~10 s without cloud, longer with
- `08_edit_operations`: ~3–6 minutes (`move_cross_chunk` +
  `atomic=True` at `N_edits=1 000` plus the concurrency sub-row
  dominate)

Only `08_edit_operations.ipynb` is committed with its outputs
intact; the rest are stripped.

## Optional cloud backend benchmarking

Notebook 03 benchmarks the `obstore` and `fsspec` cloud backends
**only when** the `ZV_BENCH_S3_URL` env var is set:

```bash
export ZV_BENCH_S3_URL="s3://my-bucket/zv-bench/"
jupyter lab benchmarks/internals/03_backends.ipynb
```

Both `obstore` and `fsspec` are optional installs on the core
package:

```bash
pip install "zarr-vectors[obstore]"   # preferred
# OR
pip install "zarr-vectors[cloud]"     # fsspec fallback
```

When the env var is unset, the notebook prints a skip note for the
cloud rows and reports a one-row local-only result.

## Caveats

These numbers are machine-dependent and **not benchmarks of
underlying algorithms** — different geometry types do genuinely
different work. Treat them as "what to expect on my machine" sanity
plots, not as cross-format comparisons.

No CI gating, no pytest-benchmark integration, no memory profiling
(disk bytes only). The one automated guard is
`tests/test_perf_writes.py` in **`zarr-vectors-py`**, which asserts
loose wall-clock ceilings on the write/read path to catch
order-of-magnitude regressions.

## Regenerating notebooks

```bash
python benchmarks/internals/_build.py
```

This rewrites all eight `.ipynb` files in place with fresh cell IDs.
Edit the cell templates in `_build.py`, never the notebooks — direct
notebook edits are overwritten on the next build.

> **Rebuilding discards stored outputs.** `_build.py` emits every cell
> with `"outputs": []`, so a rebuild strips the committed results from
> `08_edit_operations.ipynb` (the only notebook here carrying them).
> Re-run that notebook and re-commit it, or `git checkout` it
> afterwards if you only meant to regenerate the others.
