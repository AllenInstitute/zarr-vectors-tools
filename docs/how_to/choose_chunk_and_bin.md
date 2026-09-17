# Chunk and bin shape in this package's workflows

:::{important}
**How to choose a chunk and bin shape is documented by the parent
package**, at {zvpy}`Choosing chunk and bin shape <how_to/choose_chunk_and_bin.html>`. That page has the
occupancy arithmetic, the query-shape and anisotropy reasoning, and the
`Layout` surface. Read it first — this page does not repeat it.

What follows is only the part that is specific to `zarr-vectors-tools`:
how those values reach an ingest, and the flags this package adds.
:::

For the format-level definitions, see {zvpy}`chunk shape <spec/chunking/chunk_shape.html>`,
{zvpy}`bin shape <spec/chunking/bin_shape.html>` and
{zvpy}`chunk versus bin <spec/chunking/chunk_vs_bin.html>`.

## How ingests take it

Every ingest function in this package takes `chunk_shape` explicitly and
`bin_shape` optionally, and forwards both to the parent package's writers
unchanged. There is no tools-specific interpretation of either value.

```python
from zarr_vectors_tools.convert.ingest.csv_points import ingest_csv

ingest_csv("cells.csv", "cells.zarrvectors", chunk_shape=(125.0, 125.0, 125.0))
```

The CLI spells the same thing as `--chunk-shape`.

## Let TRK compute it for you

The parallel TRK path is the one exception. It takes `--num-chunks` — a
target chunk *count* — and derives the shape from the data bounds:

```bash
zvtools convert tracts.trk out.zarrvectors --num-chunks 5000
```

You can also pass per-axis counts as `X,Y,Z`. This exists because
tractogram bounds are not obvious before parsing, so asking for a shape up
front is awkward. `--num-chunks` is **TRK-only**; every other format needs
`--chunk-shape`.

## When `bin_shape` matters here

The parent package's advice — leave `bin_shape` unset unless you have a
reason — holds. This package adds exactly one reason to set it:

- `--sparsity-strategy point_thinning` derives its survivor count from
  `bin_shape` rather than from `--sparsity`. If you are using that
  strategy, `bin_shape` is what controls it. See
  [Object selection](../multiresolution/object_selection.md).

## Chunk scale across pyramid levels

Coarser levels usually want larger chunks — they hold less data per unit
volume, so keeping level 0's chunk shape leaves them sparse and
overhead-bound. `--chunk-scale` sets a per-level multiplier on the pyramid
commands in this package:

```bash
zvtools pyramid store.zarrvectors \
    --coarsen 8,8,8 --sparsity 2,2,2 --chunk-scale 2,4,8
```

On skeleton stores `chunk_scale_factor` already defaults to **2**, so you
only need the flag to override that.

## If you got it wrong

Re-ingesting from the source file is the straightforward fix, and the one
this package supports end to end.

The parent package also provides in-place rechunking —
`zarr_vectors.building.rechunk` and `rechunk_by_attribute` — which
re-lays-out an existing store without going back to the source. It is a
coordinator operation: never run it while workers are writing. See
{zvpy}`rechunking <spec/chunking/rechunking.html>` and {zvpy}`the building API <api/building.html>`.

## See also

- **Parent package:** {zvpy}`Choosing chunk and bin shape <how_to/choose_chunk_and_bin.html>` — how to choose the values.
- [Coarsening versus sparsity](../multiresolution/concepts.md) — what the pyramid does with them.
- [Choosing a compressor](compressors.md) — the other creation-time decision.
- [Parallel workflows](parallelism.md) — why chunk count bounds your parallelism.
- [Concepts](../getting_started/concepts.md)
