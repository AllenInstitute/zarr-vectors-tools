# The `zvtools` command line

Installing `zarr-vectors-tools` puts a `zvtools` executable on your path.
It covers the common end-to-end jobs — convert a file, build a pyramid,
check conformance, inspect a store — without writing any Python.

```bash
zvtools --version
zvtools {convert,pyramid,validate,info} ...
```

It is also runnable as a module, which is useful when the script shim is
not on `PATH`:

```bash
python -m zarr_vectors_tools convert cells.csv out.zarrvectors --chunk-shape 100,100,100
```

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | success |
| `1` | an error, printed as `error: TypeName: message` on stderr |
| `130` | interrupted with Ctrl-C (prints `aborted`) |

Ingest, coarsening and validation errors are funnelled into that single
clean one-line form rather than a traceback.

---

## `zvtools convert`

```bash
zvtools convert INPUT OUTPUT [options]
```

Ingest `INPUT` into a new store at `OUTPUT`, optionally building coarser
levels in the same run.

### Format selection

`--format` defaults to `auto`, which resolves from the file extension:

| `--format` | Extensions | Ingest function | Geometry | Extra |
| --- | --- | --- | --- | --- |
| `trk` | `.trk` | `trk_parallel.ingest_trk_parallel` | streamlines | `parallel` |
| `trx` | `.trx` | `trx.ingest_trx` | streamlines | `streamlines` |
| `tck` | `.tck` | `tck.ingest_tck` | streamlines | `streamlines` |
| `swc` | `.swc` | `swc.ingest_swc` | skeleton | — |
| `obj` | `.obj` | `obj.ingest_obj` | mesh | — |
| `stl` | `.stl` | `stl.ingest_stl` | mesh | — |
| `ply` | `.ply` | `ply.ingest_ply` | points | `ply` |
| `las` | `.las`, `.laz` | `las.ingest_las` | points | `las` |
| `csv` | `.csv`, `.xyz` | `csv_points.ingest_csv` | points | — |
| `lines` | *(none)* | `lines.ingest_lines_csv` | lines | — |
| `edgelist` | *(none)* | `edgelist.ingest_edgelist` | graph | `graph` |
| `graphml` | `.graphml` | `graphml.ingest_graphml` | graph | `graph` |

:::{note}
`lines` and `edgelist` register **no extensions**, because a `.csv` file
auto-detects to `csv` (points). Both therefore need an explicit
`--format`:

```bash
zvtools convert segments.csv out.zarrvectors --format lines --chunk-shape 50,50,50
zvtools convert edges.csv   out.zarrvectors --format edgelist --nodes nodes.csv \
    --chunk-shape 50,50,50
```
:::

If an extension does not resolve, the error names every valid `--format`
value. If the format's extra is missing, the error carries the exact
`pip install` command to fix it.

### Options

| Flag | Type | Default | Notes |
| --- | --- | --- | --- |
| `--format` | choice | `auto` | see the table above |
| `--chunk-shape X,Y,Z` | shape | — | **required for every format except `trk`** |
| `--num-chunks N｜X,Y,Z` | int or triple | — | **`trk` only**; target total chunk count, or per-axis counts |
| `--bin-shape X,Y,Z` | shape | — | optional intra-chunk sub-binning |
| `--dtype` | str | `float32` | stored position dtype |
| `--overwrite` | flag | off | replace an existing output store |
| `--compressor` | choice | `none` | `none｜zstd｜blosc` |
| `--workers` | int | — | parallel worker processes |
| `--workers-backend` | choice | `process` | `process｜dask` (`dask` needs the `parallel` extra) |
| `--n-parts` | int | — | `trk`: file-split granularity |
| `--compute-length` | flag | off | streamlines: store per-object length |
| `--compute-endpoints` | flag | off | streamlines: store per-object endpoints |
| `--nodes` | path | — | **required for `edgelist`**: the node CSV |
| `--knn-distance-k` | int | — | points: *k* for the kNN-distance enrichment (needs `points-enrichment`) |

Plus every [pyramid option](#pyramid-options) below.

:::{warning}
`--overwrite` deliberately refuses to remove a directory that does not
contain a `zarr.json` or `.zattrs`. It will not let you point it at your
home directory and delete it. Do not work around this.
:::

### Examples

```bash
# MERFISH transcripts, 100 µm chunks, with per-cell transcript counts.
zvtools convert transcripts.csv cells.zarrvectors --chunk-shape 100,100,100

# Tractogram: ~5000 chunks, 12 dask workers, compressed, with a 2-level pyramid.
zvtools convert tracts.trk tracts.zarrvectors \
    --num-chunks 5000 --workers 12 --workers-backend dask \
    --compressor zstd --coarsen 8,8 --sparsity 2,4

# Neuron morphology from SWC.
zvtools convert neuron.swc neuron.zarrvectors --chunk-shape 50,50,50
```

---

## `zvtools pyramid`

```bash
zvtools pyramid STORE [options]
```

Build coarser resolution levels on an **existing** store. Geometry routing
is automatic — the coarsener is chosen from the store's
`links_convention` and `geometry_types`, so you do not select it by hand.
See [Strategies](../multiresolution/strategies.md).

Both `--coarsen` and `--sparsity` are required here.

| Flag | Type | Default | Notes |
| --- | --- | --- | --- |
| `--cross-level-storage` | choice | *(store default)* | `none｜implicit｜explicit` |
| `--cross-level-depth` | int | *(store default, 1)* | `0` none, `N` up to ±N, `-1` all pairs |
| `--compressor` | choice | `none` | match what level 0 was written with |
| `--workers` | int | — | parallel worker processes |
| `--workers-backend` | choice | `process` | `process｜dask` |

Plus the [pyramid options](#pyramid-options) below.

```bash
zvtools pyramid tracts.zarrvectors \
    --coarsen 8,8,8 --sparsity 2,2,2 \
    --sparsity-strategy length --compressor zstd --workers 8
```

(pyramid-options)=

## Pyramid options

These appear on both `convert` and `pyramid`.

| Flag | Type | Default | Notes |
| --- | --- | --- | --- |
| `--coarsen C1,C2,...` | float list | — | per-level vertex coarsen factor, or decimation stride |
| `--sparsity S1,S2,...` | float list | — | per-level object-drop **divisor** (`1` keep all, `2` half, `8` an eighth) |
| `--chunk-scale K1,K2,...` | int list | — | per-level chunk-size multiplier |
| `--sparsity-strategy` | choice | `random` | `random｜length｜spatial_coverage｜attribute｜point_thinning` |
| `--coarsen-mode` | choice | `rdp` | `rdp｜decimate` — polyline vertex reduction |

`--coarsen` and `--sparsity` must have **equal length**, one entry per
coarser level; a mismatch exits with an error naming both counts. Giving
neither means no pyramid is built.

:::{note}
Passing `--sparsity-strategy length` on a streamline format
auto-enables `--compute-length`, since the strategy needs the lengths it
ranks by. The CLI prints a `note:` when it does this for you.
:::

The two axes are explained in
[Coarsening versus sparsity](../multiresolution/concepts.md).

---

## `zvtools validate`

```bash
zvtools validate STORE [--level N]
```

Run core conformance validation. `--level` is the conformance level `1`–`5`
and defaults to `3`. Prints errors, messages and issues; returns `0` when
the store conforms and `1` otherwise, so it drops straight into CI.

```bash
zvtools validate tracts.zarrvectors --level 4
```

---

## `zvtools info`

```bash
zvtools info STORE
```

Print a store's `zv_version`, `geometry_types`, `links_convention`,
`chunk_shape`, `bounds`, resolution levels, cross-level depth and
storage, and format capabilities.

This is the fastest way to answer "what coarsener will `zvtools pyramid`
pick?" — the `links_convention` and `geometry_types` lines are exactly
the inputs to that decision.

---

## The other CLI

`precomputed_skeletons` carries its own separate argparse entry point for
EM skeleton ingest, which is not exposed through `zvtools`:

```bash
python -m zarr_vectors_tools.ingest.precomputed_skeletons \
    SOURCE OUT_STORE --anchor X Y Z [--counts ...] [--frags-dir ...] \
    [--strides ...] [--chunk-scales ...] [--sparsity ...] \
    [--workers N] [--drop-interior-below N]
```

See [Skeletons in EM](../ingest/em_skeletons.md).

## See also

- [Quickstart](quickstart.md) — the same jobs from Python.
- [Ingest workflows](../ingest/index.md) — per-format detail.
- [Parallel workflows](../how_to/parallelism.md) — `--workers` and `--workers-backend`.
- [Choosing a compressor](../how_to/compressors.md) — `--compressor`.
- [Coarsening versus sparsity](../multiresolution/concepts.md) — `--coarsen` and `--sparsity`.
