# Benchmarks

Two suites, answering two different questions. Both live here rather
than in `zarr-vectors-py` so the core package keeps its dependency
surface at `numpy` / `numcodecs` / `zarr` — the benchmarks need
`pandas`, `matplotlib`, `jupyter`, and (for the format comparisons)
the third-party readers this package already wraps.

| Suite | Question | Contents |
|-------|----------|----------|
| [`paper/`](paper/) | *The three claims the paper makes, and nothing else.* | scripts — 8 panels as separate files + one supplementary table |
| [`formats/`](formats/) | *How does Zarr Vectors compare to the format I use today?* | 3 notebooks — Zarr Vectors reads/filters vs PLY, CSV, TRX, GraphML, SWC, OBJ |
| [`internals/`](internals/) | *How does Zarr Vectors scale along axis X?* | 8 notebooks — size, geometry type, backend, pyramid, bbox query, chunk shape, codec, edit cost |

`formats/` and `internals/` are exploratory: each has its own
`README.md`, its own `_build.py`, and its own `01..N` numbering. They
share no code; the small timing/stats helpers (`_time`, `_store_bytes`,
`_mean_ci95`, `N_RUNS = 10`, `T95_DF9 = 2.262`) are duplicated into each
generated notebook, which is what keeps a notebook runnable standalone.

`paper/` is different in kind — it is a publication artefact, not an
exploration. It is scripts rather than notebooks so that the measurement
can be re-run on reference hardware and the panels regenerated from the
committed CSV without re-measuring; the panels are written one per file
so the figure can be composed wherever the paper is being written. Start there if you want the headline
comparison; start in the other two if you want to understand a
particular axis.

## Running

```bash
pip install -e ".[all]" jupyter matplotlib
jupyter lab benchmarks/
```

Then open a notebook and run all cells. Budget ~5 minutes for the
long ones (`formats/01_size_scaling`, `internals/01_size_scaling`,
`internals/08_edit_operations`).

`paper/` is scripts, not notebooks, so it runs from a shell instead:

```bash
python benchmarks/paper/run_sweep.py     # 10^3..10^6, ~20-30 min
python benchmarks/paper/run_large.py     # 10^3..10^7, ~1.5 h, resumable
python benchmarks/paper/make_figure.py   # seconds
```

The default sweep stops at 10^6 because one more decade roughly triples
its wall time, and it is mostly run to check that nothing regressed.
`run_large.py` drives the same measurement code one decade further, block
by block into `paper/results/large/`, so the long run is opt-in and a
killed one can be resumed. See [`paper/README.md`](paper/README.md).

## Regenerating

Notebooks are generated; never edit the `.ipynb` directly.

```bash
python benchmarks/formats/_build.py     # rewrites formats/*.ipynb
python benchmarks/internals/_build.py   # rewrites internals/*.ipynb
```

## Shared caveats

- Wall time and on-disk bytes only — no memory profiling.
- No CI gating in either suite. The only automated performance guard
  in the project is `tests/test_perf_writes.py` over in
  `zarr-vectors-py`, which asserts loose (~3×) wall-clock ceilings to
  catch order-of-magnitude regressions.
- Synthetic data with `SEED = 0`. Real datasets have different
  sparsity and chunk-occupancy patterns, and therefore different
  scaling slopes.
- Machine-dependent. These are "what to expect on my hardware" sanity
  plots, not authoritative numbers.
