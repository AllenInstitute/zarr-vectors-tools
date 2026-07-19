# Choosing chunk and bin shape

`chunk_shape` is the one required parameter on almost every ingest, and
the one with the largest effect on how the store performs afterwards. It
cannot be changed later without rewriting the store, so it is worth a few
minutes up front.

## What chunk shape controls

A chunk is the unit of I/O. Everything downstream follows from its size:

| Chunk shape | Consequence |
| --- | --- |
| too large | every spatial query over-fetches; parallel workers have nothing to divide; peak memory per task rises |
| too small | request count and metadata overhead dominate, badly so on cloud object stores |

The useful mental target is **how much data one chunk holds**, not how
big it is in space. Aim for chunks that hold roughly 10⁴–10⁶ vertices.
Below that, per-chunk overhead dominates; above it, queries and workers
get coarse.

## A practical procedure

1. **Find your bounds and vertex count.** Both are in the ingest summary
   dict, or `zvtools info` on an existing store.
2. **Pick a target chunks-per-axis.** For a roughly cubic dataset,
   32–128 chunks per axis is a reasonable starting range.
3. **Divide.** `chunk_shape = extent / chunks_per_axis`.
4. **Sanity-check the occupancy.** `n_vertices / n_chunks` should land in
   the 10⁴–10⁶ band.

```python
import numpy as np

extent = np.array([2000.0, 1500.0, 800.0])   # µm
n_vertices = 50_000_000

chunks_per_axis = 64
chunk_shape = tuple(extent / chunks_per_axis)
print(chunk_shape)                            # (31.25, 23.44, 12.5)
print(n_vertices / chunks_per_axis ** 3)      # ~191 vertices/chunk — too small
```

That result says 64 per axis is too fine for this dataset: 262 144 chunks
holding 191 vertices each is nearly all overhead. Drop to 16 per axis
(4 096 chunks, ~12 000 vertices each) and the numbers land in range.

:::{note}
Occupancy is what matters, not chunk count. A sparse dataset spread over
a large volume needs *larger* chunks than a dense one in the same volume,
because most of its chunks are empty.
:::

## Match the chunk to the query

Occupancy sets the scale; your access pattern sets the aspect ratio.

- **Interactive viewing** — match the chunk to roughly the viewport
  volume at your most common zoom, so a screenful is a handful of
  fetches.
- **Whole-object reads** (fetch one neuron, one tract bundle) — prefer
  larger chunks. Every chunk an object crosses is another fragment to
  reassemble.
- **Anisotropic data** — an EM volume with 4 nm lateral and 40 nm axial
  sampling wants an anisotropic `chunk_shape` too. Cubic chunks in
  *index* space are not cubic in physical space.

## Let TRK compute it for you

The parallel TRK path takes `--num-chunks` — a target chunk *count* — and
derives the shape from the data bounds:

```bash
zvtools convert tracts.trk out.zarrvectors --num-chunks 5000
```

You can also pass per-axis counts as `X,Y,Z`. This exists because
tractogram bounds are not obvious before parsing, so asking for a shape
up front is awkward. `--num-chunks` is **TRK-only**; every other format
needs `--chunk-shape`.

## Bin shape

`bin_shape` is an optional finer grid *inside* each chunk, used as the
spatial-hash bucket size by some accelerators, and as the basis of the
`point_thinning` sparsity strategy.

**Leave it unset unless you have a specific reason.** The writers default
it sensibly. The two reasons to set it:

- you are using `--sparsity-strategy point_thinning`, which derives its
  survivor count from `bin_shape` rather than from `--sparsity`;
- you have profiled a spatial-query workload and found the default bucket
  size wrong for it.

When set, `bin_shape` should divide `chunk_shape` evenly.

## Chunk scale across pyramid levels

Coarser levels usually want larger chunks — they hold less data per unit
volume, so keeping level 0's chunk shape leaves them sparse and
overhead-bound. `--chunk-scale` sets a per-level multiplier:

```bash
zvtools pyramid store.zarrvectors \
    --coarsen 8,8,8 --sparsity 2,2,2 --chunk-scale 2,4,8
```

On skeleton stores `chunk_scale_factor` already defaults to **2**, so you
only need the flag to override that.

## If you got it wrong

Chunk shape is fixed at creation. To change it, re-ingest from the source
file. There is no in-place rechunk — which is the reason to spend the few
minutes on the arithmetic before a long ingest rather than after it.

## See also

- [Coarsening versus sparsity](../multiresolution/concepts.md) — what the pyramid does with these.
- [Choosing a compressor](compressors.md) — the other creation-time decision you cannot revisit.
- [Parallel workflows](parallelism.md) — why chunk count bounds your parallelism.
- [Concepts](../getting_started/concepts.md)
- Parent package: [Choosing chunk and bin](https://zarr-vectors-py.readthedocs.io/en/latest/how_to/choose_chunk_and_bin.html)
