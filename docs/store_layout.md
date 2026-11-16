# Store layout

How to choose the chunk shape, bin shape, compressor and sharding of a store.
The core package defines {zvpy}`chunks and bins <spec/chunking/chunk_vs_bin.html>` and
{zvpy}`sharding <spec/chunking/sharding.html>`.

## Chunk shape

The chunk shape is the edge of one level-0 grid cell per axis, in the input's
units: mm for TRK, TCK and TRX, nm for EM precomputed layers, and the file's
own units otherwise. It is required, except:

- TRK takes `--num-chunks N` (total, default 125) or `--num-chunks X,Y,Z`.
- A precomputed skeleton layer with a spatial index keeps its own grid.

Aim for 10⁴–10⁵ vertices in a typical level-0 chunk; for TRK, `--num-chunks` ≈
total vertices ÷ 10⁵. Tiny chunks cost the viewer many requests; huge ones
load slowly.

:::{warning}
One chunk's data must stay under 4 GiB per array. Only the TRK ingest checks
this; other formats can write a truncated chunk without an error.
:::

```bash
zvtools convert cells.csv cells.zv --chunk-shape 100,100,100
zvtools convert tracts.trk tracts.zv --num-chunks 1000   # ~1e8 vertices
```

## Bin shape

A bin is a sub-cell of a chunk; the chunk shape must be a whole multiple of
it. Bins matter only for a pyramid: point, graph and line levels coarsen in
multiples of the bin edge, and the default streamline tolerance comes from it.
Without `--bin-shape`, the first coarse level of a point store merges each
chunk into a few vertices. Set the bin near the spacing between neighbouring
points:

```bash
zvtools convert cells.csv cells.zv --chunk-shape 100,100,100 --bin-shape 10,10,10
```

## Compressor

Stores are uncompressed by default, which suits viewing: the
[viewer](visualise.md) reads per-chunk arrays raw. `zstd` and `blosc` make a
store smaller, for archives and transfer. `--compressor` works on
`zvtools convert` for TRK input, and on `zvtools pyramid` for the levels it
adds; use the value level 0 was written with.

## Sharding

Sharding packs `N × N × N` neighbouring chunks into one file per array, so a
large store has far fewer files. The data does not change. Use `8` before
uploading a large store.

```bash
zvtools convert tracts.trk tracts.zv --num-chunks 1000 --shard 8   # after writing
zvtools shard tracts.zv --shape 8                                  # an existing store
zvtools shard tracts.zv --unshard                                  # one file per chunk
```

A server for a sharded store must support HTTP Range requests
([serving requirements](visualise.md#serving-requirements)).

## Position dtype

`--dtype` sets the stored position type (default `float32`). Keep `float32`:
the viewer requires it, and precomputed ingest refuses anything else.
