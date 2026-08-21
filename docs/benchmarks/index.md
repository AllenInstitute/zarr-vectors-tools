# Benchmarks

All zarr-vectors benchmark notebooks live in this package rather than
in the core `zarr-vectors-py` repository, so that core keeps its
runtime dependencies at `numpy` / `numcodecs` / `zarr`. The
benchmarks need `pandas`, `matplotlib`, `jupyter`, and — for the
format comparisons — the third-party readers this package already
wraps.

There are three suites under
[`benchmarks/`](https://github.com/AllenInstitute/zarr-vectors-tools/tree/main/benchmarks)
in the source tree:

| Suite | Question it answers | Contents |
|-------|--------------------|----------|
| `benchmarks/paper/` | The three claims the paper makes, and nothing else | scripts: one 8-panel figure, one supplementary table |
| `benchmarks/formats/` | How does ZVF compare to the format I use today? | 3 notebooks |
| `benchmarks/internals/` | How does ZVF scale along one axis? | 8 notebooks |

`formats/` and `internals/` are exploratory notebooks. Both follow the
same shape (setup → build inputs → sweep → table → plot), and both
average each measurement over `N_RUNS = 10` repeats, reporting the mean
with a Student's-t 95 % CI half-width (`T95_DF9 = 2.262`, hard-coded to
avoid a scipy dependency).

## Publication figure — `benchmarks/paper/`

Eight panels covering storage per vertex, the size crossover against
dataset size, bulk write and read, single-object fetch and replace, and
bounding-box query cost swept two ways. Measurement and plotting are
separate scripts (`run_sweep.py` → `measurements.csv` →
`make_figure.py`) so the sweep can be re-run on reference hardware
without a plotting stack and the panels regenerated from the committed
CSV without re-measuring.

The panels are written as **separate files** under `results/panels/`,
one per panel plus a standalone key, rather than pre-composed into a
sheet — composition is left to whoever assembles the figure.
`--combined` still produces the single-sheet version. Every sweep runs
one point per decade from 10³ to 10⁶.

The figure sweeps two different variables and says which is which on
every axis. Storage and the spatial queries scale with **vertex** count;
bulk write, bulk read and single-object fetch scale with **object**
count, each object held at a fixed size (1 vertex per point, 12 per
streamline, 144 per mesh patch), because "how many streamlines" is the
axis a user picks a format along and the one a per-object index is meant
to answer.

Every competitor format there is given its best available
implementation, not a typical one — vectorised binary readers, in-place
single-record overwrites where the layout permits them, and gzipped
variants of every text format — so the comparison is deliberately
conservative. See the suite's `README.md` for the full set of design
rules and caveats.

## Format comparisons — `benchmarks/formats/`

```{admonition} Stub
:class: warning

Authoritative numbers are not yet published for this suite. **Results
tables and plots will be added in a future release** once a fixed
harness, reference hardware, and reproducibility protocol have been
agreed.
```

Measures **loading** and **filtering** a zarr-vectors store against
the canonical file format for each geometry type. Writes happen once
outside the timing loop; only reads are benchmarked.

| Notebook | Axis swept | Fixed |
|----------|-----------|-------|
| `01_size_scaling.ipynb` | vertex count `N ∈ {10³, 10⁴, 10⁵}` | three geometries: point cloud, graph, mesh |
| `02_data_types.ipynb`   | geometry type (all six) | `N = 50 000`, each vs canonical competitor |
| `03_filtering.ipynb`    | subset fraction `f ∈ {10⁻³, 10⁻², 10⁻¹, 0.5, 1.0}` | `N = 100 000`, point cloud + polylines |

Notebooks 01 and 03 produce **log-log** matplotlib plots with shaded
95 % CI bands. Notebook 02 produces a bar chart with error bars.

### What's compared

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

Expected runtime on a laptop: `01_size_scaling` ~5 minutes (the
100 K mesh case dominates, because the OBJ parser is pure Python);
`02_data_types` ~1 minute; `03_filtering` ~1 minute.

## Internals sweeps — `benchmarks/internals/`

Sweeps zarr-vectors against **itself** along one axis at a time. No
competing format is involved.

| Notebook | Axis swept | Fixed |
|----------|-----------|-------|
| `01_size_scaling.ipynb` | vertex count `N ∈ {10³, 10⁴, 10⁵, 10⁶}` | point cloud, local backend |
| `02_data_types.ipynb` | geometry type (all six) | `N = 50 000`, local backend |
| `03_backends.ipynb` | storage backend (`local`, `obstore`, `fsspec`) | point cloud, `N = 100 000` |
| `04_pyramid_scaling.ipynb` | coarsening factor × build mode | points + graphs, via `build_pyramid` |
| `05_bbox_queries.ipynb` | geometry type | spatial bbox query via `lazy.open_zv` |
| `06_chunk_shape.ipynb` | `chunk_shape` | points at fixed `N` |
| `07_compression.ipynb` | compressor / codec × `N` | points |
| `08_edit_operations.ipynb` | edit kind, atomicity, `N_edits`, concurrency, index scaling | `N = 50 000` baseline points |

Notebook 08 additionally covers 1 vs 4 cooperating editors on
disjoint chunks (via `oid_prefix=`) and sweeps `N_OBJECTS ∈ {1k, 10k,
50k}` to document the flatness of the fragment→OID inverted index.
It is the only notebook committed with its outputs intact.

Expected runtime on a laptop: `01_size_scaling` a few minutes (the
1 M case dominates); `02_data_types` ~30 seconds; `03_backends` ~10
seconds without cloud; `08_edit_operations` ~3–6 minutes.

### Vertex scaling — `internals/01_size_scaling`

A point cloud with `N` random vertices in `[0, 1000)³` is written and
read four ways against a pandas / CSV baseline. Each `N` is averaged
over 10 runs; bands are 95 % Student's-t confidence intervals
(df = 9). Numbers below come from a typical SSD-backed workstation —
treat them as *order-of-magnitude* references, since wall times
depend on disk, filesystem, and OS cache state.

![Vertex scaling: write, read all, read one, disk size — zarr-vectors
vs CSV](vertex_scaling_benchmarking.png)

zarr-vectors pays a fixed ~0.4 s setup cost (zarr metadata,
fragment-index sidecars, one array per spatial chunk) regardless of
`N`, then scales sublinearly. CSV scales linearly from the first
point — `to_csv` formats and writes every row sequentially,
`read_csv` scans the byte stream once, and a "random row" read is
`read_csv(..., skiprows=range(1, row_idx + 1), nrows=1)` which still
parses every preceding line before the one it wants. Crossover
points:

| Operation | CSV wins when… | zarr-vectors wins when… |
|-----------|---------------|-------------------------|
| Write     | `N < 10⁵` | `N ≳ 10⁵` |
| Read all  | `N ≲ 10⁶` | `N > 10⁶` (gap closes fast) |
| Read one  | `N < 10⁵` | `N ≳ 10⁵` |
| Disk size | `N < 10⁴` | `N ≳ 10⁴` |

#### Dtype and on-disk encoding

Both sides start from the same `float32` input
(`numpy.random.default_rng().uniform(...).astype(np.float32)`).

| Stage | zarr-vectors | pandas / CSV |
|-------|--------------|--------------|
| On-disk encoding | packed `float32` → Blosc(Zstd, BitShuffle, level=5) | decimal text, one row per line |
| Bytes per `(x, y, z)` row | 12 (pre-compression) | ~30–50 (8–12 chars per float + delimiters) |
| Read result dtype | `float32` ndarray | `float64` `DataFrame` |

Two dtype asymmetries drive the plot. CSV renders each float as a
decimal string (`{:.12g}` by default), so a 16-byte `(x, y, z,
intensity)` row blows up to ~40 bytes of text — and text does not
compress with the density Blosc-BitShuffle gives binary float32. On
read, pandas silently widens every column to `float64` unless
`dtype=` is set; zarr-vectors round-trips the exact `float32` it
stored.

The constant ~0.1 s "Read one" floor for zarr-vectors at every `N` is
the lazy reader (`zarr_vectors.lazy.open_zv`) opening the store,
listing chunks once, and decoding a single Blosc-compressed
`vertices/<i.j.k>` chunk — no scan, no offset table to walk.

### Cloud-backend mode

`internals/03_backends.ipynb` benchmarks the `obstore` and `fsspec`
cloud backends **only when** the `ZV_BENCH_S3_URL` env var is set:

```bash
export ZV_BENCH_S3_URL="s3://my-bucket/zv-bench/"
pip install "zarr-vectors[obstore]"   # preferred
# or
pip install "zarr-vectors[cloud]"     # fsspec fallback
```

Without the env var the cloud rows are skipped and a one-row
local-only result is reported.

## Running locally

```bash
pip install -e ".[all]" jupyter matplotlib
jupyter lab benchmarks/
```

## Results

*Format-comparison numbers to be added.* They will be published here
once a reproducibility protocol (hardware, OS, dependency versions,
dataset seeds) has been frozen and the notebooks re-run against the
locked target. Until then the notebooks themselves are the only
available reference and they are machine-dependent — do not treat
them as published metrics.

## Caveats

- **Wall time and on-disk size only.** No memory profiling.
- **No CI gating** in either suite. The only automated performance
  guard in the project is `tests/test_perf_writes.py` in
  `zarr-vectors-py`, which asserts loose (~3×) wall-clock ceilings on
  the write/read path to catch order-of-magnitude regressions.
- **Different geometry types do genuinely different work.** Don't
  cross-compare rows of either `02_data_types`.
- **Synthetic data** seeded with `SEED = 0` throughout. Realistic
  datasets will have different sparsity / chunk-occupancy patterns
  and consequently different scaling slopes.
- **Not comparable across hardware**, file systems, or cloud regions.
- The CSV baseline in `internals/01` uses pandas' default `read_csv`
  behaviour (float64 output, no chunked reads). A tuned baseline with
  `dtype=np.float32` and `engine='c'` would narrow the read gap
  somewhat but not change the random-access scaling.

To regenerate the notebooks from their source templates:

```bash
python benchmarks/formats/_build.py
python benchmarks/internals/_build.py
```
