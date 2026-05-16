# Benchmarks

Three small notebooks that compare **loading and filtering** a
zarr-vectors store against the canonical file format for each
geometry type. All three follow the same shape — setup → build
inputs → sweep → table → plot — and exercise reads only (writes
happen once outside the timing loop).

| Notebook | Axis | Fixed | Swept |
|----------|------|-------|-------|
| [`01_size_scaling.ipynb`](01_size_scaling.ipynb) | size (N) | three geometries: points, graph, mesh | `N ∈ {1e3, 1e4, 1e5}` |
| [`02_data_types.ipynb`](02_data_types.ipynb)     | geometry type | `N = 50 000` | all six types, each vs its canonical competitor |
| [`03_filtering.ipynb`](03_filtering.ipynb)       | subset fraction | `N = 100 000`, points + polylines | `fraction ∈ {0.001, 0.01, 0.1, 0.5, 1.0}` |

For every measurement the notebook runs `N_RUNS = 10` repeats and
reports the mean with a Student's-t **95 % CI half-width** (`T95_DF9
= 2.262`). Size-scaling and filtering plots have **log-log axes**
with shaded CI bands; the data-type comparison is a bar chart with
error bars.

## What's compared

| Geometry | Competitor | Reader | Subset op |
| --- | --- | --- | --- |
| point cloud | PLY | `plyfile.PlyData.read` | bbox |
| line | CSV | `pandas.read_csv` | bbox |
| polyline / streamline | TRX | `trx.trx_file_memmap.load` | object_ids (native partial read) |
| graph | GraphML / edge-list CSV | `networkx.read_graphml` / `pd.read_csv` | bbox |
| skeleton | SWC | text parse | bbox |
| mesh | OBJ | pure-Python parser | bbox |

**TRX is the only competitor with a native partial read.** Every
other competitor reads the entire file and filters in numpy, which
is the honest comparison — those formats simply have no other
option.

## Running

```bash
pip install -e ".[all]" jupyter matplotlib
jupyter lab benchmarks/
```

Then open one notebook and run all cells. Expected runtime on a
laptop:

- `01_size_scaling`: ~5 minutes (the 100 K mesh case is the long pole; the pure-Python OBJ parser dominates).
- `02_data_types`: ~1 minute.
- `03_filtering`: ~1 minute.

Sections gated on optional deps (`networkx`, `trx-python`,
`plyfile`) skip gracefully if the package isn't installed. To run
the whole sweep install the `all` extra above.

## Regenerating notebooks

The notebooks are generated from `_build.py`. To edit them, change
the cell templates there and re-run:

```bash
python benchmarks/_build.py
```

This rewrites all three `.ipynb` files in place with fresh cell
IDs. The notebooks themselves should be committed alongside the
source script.

## Caveats

- These numbers are machine-dependent and meant as **"what to
  expect on my machine"** sanity plots. Don't quote them as
  authoritative.
- Wall time and on-disk size only — no memory profiling.
- No CI gating; regressions aren't caught automatically. A
  `tests/test_perf_*.py` style regression suite (along the lines of
  `zarr-vectors-py`'s `test_perf_writes.py`) is a reasonable
  follow-up but is intentionally out of scope here.
- Synthetic data with `SEED = 0` everywhere. Realistic datasets
  will have different sparsity / chunk-occupancy patterns and
  consequently different scaling slopes.
