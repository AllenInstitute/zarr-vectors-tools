# Benchmarks

```{admonition} Stub
:class: warning

Authoritative numbers are not yet published. This page describes
how to run the benchmark notebooks shipped with the repository and
what each one measures. **Results tables and plots will be added in
a future release** once a fixed harness, reference hardware, and
reproducibility protocol have been agreed.
```

The benchmark suite measures **loading** and **filtering** a
zarr-vectors store against the canonical file format for each
geometry type. Writes happen once outside the timing loop; only
reads are benchmarked.

## Notebooks

Three notebooks live under
[`benchmarks/`](https://github.com/Andrew-Keenlyside/zarr-vectors-tools/tree/main/benchmarks)
in the source tree. Each follows the same shape (setup → build
inputs → sweep → table → plot).

| Notebook | Axis swept | Fixed |
|----------|-----------|-------|
| `01_size_scaling.ipynb` | vertex count `N ∈ {10³, 10⁴, 10⁵}` | three geometries: point cloud, graph, mesh |
| `02_data_types.ipynb`   | geometry type (all six) | `N = 50 000`, each vs canonical competitor |
| `03_filtering.ipynb`    | subset fraction `f ∈ {10⁻³, 10⁻², 10⁻¹, 0.5, 1.0}` | `N = 100 000`, point cloud + polylines |

Notebooks 01 and 03 produce **log-log** matplotlib plots with
shaded **95 % CI** bands (`N_RUNS = 10` repeats per measurement,
Student's t with df = 9). Notebook 02 produces a bar chart with
error bars.

## What's compared

| Geometry | Competitor | Reader | Subset op |
| --- | --- | --- | --- |
| point cloud | PLY | `plyfile.PlyData.read` | bbox |
| line | CSV | `pandas.read_csv` | bbox |
| polyline / streamline | TRX | `trx.trx_file_memmap.load` | object_ids (native partial read) |
| graph | GraphML / edge-list CSV | `networkx.read_graphml`, `pd.read_csv` | bbox |
| skeleton | SWC | text parse | bbox |
| mesh | OBJ | pure-Python parser | bbox |

**TRX is the only competitor with a native partial read.** Every
other competitor materialises the full file and applies the filter
in numpy. The notebook-02 results table flags this with a
`supports_partial` column so readers can interpret the bars.

## Running locally

```bash
pip install -e ".[all]" jupyter matplotlib
jupyter lab benchmarks/
```

Then open one notebook and run all cells. Expected runtime on a
laptop:

- `01_size_scaling`: ~5 minutes (the 100 K mesh case dominates).
- `02_data_types`: ~1 minute.
- `03_filtering`: ~1 minute.

## Results

*To be added.* Numbers will be published here once a reproducibility
protocol (hardware, OS, dependency versions, dataset seeds) has been
frozen and the notebooks have been re-run against the locked target.
Until then the notebooks themselves are the only available reference
and they are machine-dependent — do not treat them as published
metrics.

## Caveats

- **Wall time and on-disk size only.** No memory profiling.
- **No CI gating.** Regressions aren't caught automatically.
- **Different geometry types do genuinely different work.** Don't
  cross-compare rows of `02_data_types`.
- **Synthetic data** seeded with `SEED = 0` throughout. Realistic
  datasets will have different sparsity / chunk-occupancy patterns
  and consequently different scaling slopes.

To regenerate the notebooks from the source template:

```bash
python benchmarks/_build.py
```
