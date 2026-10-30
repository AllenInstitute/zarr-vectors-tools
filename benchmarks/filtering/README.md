# Filtering: gate G1

*Is numpy, run inside the browser by Pyodide, fast enough to be the
streamline filter engine of a Neuroglancer-based viewer?*

This is gate G1 of the viewer platform's build guide. The filter plan sets
interactive targets (its C8): an operator flip under 16 ms, a column filter
under 50 ms, a 5 mm sphere under 100 ms, publishing the answer to the GPU
under 20 ms, peak memory under 1.5 GB, and a first answer within 3 s. If
numpy meets them under Pyodide, the platform needs no Rust kernel for those
steps.

## What is measured

`g1.py` builds an `IndexedEngine` (`zarr_vectors_tools.filtering`) over a
tractogram held in memory, then asks it every kind of question. Each region
query gets a fresh region centred on a random vertex, so nothing comes from
the cache; P3 flips the operators of trees whose leaves are all cached. The
brute-force oracle answers one sphere too, as a cross-check (the run records
whether they agree) and as the baseline the index must beat.

| Key | Question |
|---|---|
| `build_s` | building the segment index (the engine's start-up) |
| `sphere_ms`, `box_ms`, `disk_ms`, `mask_ms` | one region of about 5 mm (box 10 mm, mask 10 x 1 mm voxels) |
| `slab_ms` | a 2 mm slab, TrackVis's slice filter |
| `endpoints_ms` | a 5 mm sphere, either end inside |
| `p1_ms` | `length between (p10, p90)` |
| `p3_ms` | re-evaluating a four-leaf tree, every leaf cached |
| `publish_ms` | three answers to one byte per streamline |
| `oracle_sphere_s` | the same sphere, every segment tested |
| `loaded_mb`, `peak_mb` | memory with the set loaded, and at its peak |

Times are compute only: the geometry is already in memory, so nothing is
fetched or decoded. `peak_mb` is the process's peak RSS under CPython, and
the wasm heap under Pyodide, which only grows. Both include loading the set
from `data/`, where `run.py` wrote it beforehand in a separate process, so
generating a synthetic set costs neither runtime anything.

The sets:

- `synth-1e5`, `synth-1e6`: `whole_brain_like(n, seed=0)`. Streamlines with
  1 mm steps and lengths of 20–200 mm (median about 50), curving smoothly
  and folded back into a brain-sized ellipsoid. That gives a whole brain's
  density, about 40 vertices per mm³ at 10^6 streamlines.
- `real`: a store passed as `--real`, cached as `data/real/`. The
  results below used the HiP-CT right-hemisphere iFOD2 tractogram
  (`hip_ct_zarr_vector_.../APEX_hackathon/sub-Ha1_sample-RightHemi_...`):
  79,648 streamlines and 39.4 M vertices at a 0.1 mm step.

## Running

You need an unpacked Pyodide distribution holding `pyodide.mjs`,
`pyodide-lock.json` and the numpy wheel. The release tarball has them all,
and nothing is fetched from a CDN.

```bash
curl -LO https://github.com/pyodide/pyodide/releases/download/314.0.7/pyodide-314.0.7.tar.bz2
tar -xjf pyodide-314.0.7.tar.bz2      # -> ./pyodide/
python benchmarks/filtering/run.py --pyodide-dir ./pyodide [--real path/to/store.zarrvectors]
```

Each set runs in a fresh process per runtime. Results go to
`results/measurements.json` and `results/summary.md`. Run one runtime or set
with `--runtimes cpython` or `--sets synth-1e5`. Under Pyodide the filtering
subpackage is copied in as the package `zvfilter`, since it imports only
numpy and itself; that is how a browser worker would load it.

## Results

See [`results/summary.md`](results/summary.md) for the table and
[`results/measurements.json`](results/measurements.json) for every number
(including p90 and max). They were measured on a 4-core cloud container with
Python 3.13 / numpy 2.5 and Pyodide 314.0.7 (Python 3.14, numpy 2.4) under
Node 22.

RESULTS_PLACEHOLDER

## Caveats

- One machine, one run; the times are for comparing runtimes and sets, not
  absolute.
- Single-threaded throughout, as Pyodide is.
- The engine holds level 0 whole in memory. A browser engine streams
  chunks, so it fetches and decodes as well; that cost is not measured here.
