# Store layout

Four settings decide how a store is laid out on disk: the chunk shape, the bin
shape, the compressor and sharding. This page is the one place that recommends
values for them. Pyramid settings are on [Pyramids](pyramids.md).

| Setting | Flag | Default | Recommendation |
|---|---|---|---|
| Chunk shape | `--chunk-shape X,Y,Z` (`--num-chunks` for TRK) | none: required | ~10⁴–10⁵ vertices per level-0 chunk |
| Bin shape | `--bin-shape X,Y,Z` | the chunk shape | near the point spacing, if you will build a pyramid |
| Compressor | `--compressor` (TRK only) | `none` | `none` for any store you want to view |
| Sharding | `--shard N` or `zvtools shard` | unsharded | `8` before uploading a large store |

## Chunk shape

The chunk shape is the edge length of one grid cell at level 0, per axis, **in
the input's coordinate units**: millimetres for TRK, TCK and TRX, nanometres for
EM precomputed layers, and whatever the file uses for CSV, SWC, OBJ and the
rest. Every format requires it except two:

- **TRK** sizes the grid from the file's extent: `--num-chunks N` (total, default
  125) or `--num-chunks X,Y,Z` (per axis).
- **A precomputed skeleton layer with a spatial index** keeps the layer's own
  grid.

A starting point: choose the chunk edge so that a typical level-0 chunk holds
about 10⁴–10⁵ vertices, which is roughly 0.1–1 MB of float32 positions. For
TRK, that means `--num-chunks` ≈ total vertices ÷ 10⁵. A viewer issues about
one request per array per visible chunk, so many tiny chunks cost requests,
while a few huge ones load slowly and waste memory.

:::{warning}
One chunk's data must stay under 4 GiB per array. Only the TRK ingest checks
this; other formats can write a truncated chunk without an error. Keep chunks
far below that limit.
:::

```bash
# Points spanning about 1000 units per axis: 100-unit chunks give a 10 x 10 x 10 grid
zvtools convert cells.csv cells.zv --chunk-shape 100,100,100

# 1 M streamlines x ~100 points = 1e8 vertices -> about 1000 chunks
zvtools convert tracts.trk tracts.zv --num-chunks 1000
```

## Bin shape

A bin is a sub-cell of a chunk. The chunk shape must be a whole multiple of the
bin shape, and when `--bin-shape` is omitted a bin is the whole chunk.

Bins matter only when you build a pyramid. Points, graphs and line stores
coarsen in multiples of the bin edge, and the default streamline simplification
tolerance is derived from it. With no `--bin-shape`, the first coarse level of a
point store already merges each chunk into a few vertices. Set the bin near the
spacing between neighbouring points:

```bash
zvtools convert cells.csv cells.zv --chunk-shape 100,100,100 --bin-shape 10,10,10
```

See [Pyramids](pyramids.md) for what each coarse level does with it.

## Compressor

Stores are written uncompressed by default, and that is the right choice for
any store you want to view. The [Neuroglancer viewer](visualise.md) reads
per-chunk arrays raw, so it cannot display a compressed store.

`zstd` (zarr's default level) and `blosc` (Blosc with zstd and bit-shuffle,
level 5) make a streamline store about 2.4x smaller, which suits archives and
transfer. Two commands take `--compressor`:

- `zvtools convert` for TRK input. Other formats refuse the flag.
- `zvtools pyramid`, for the coarse levels it adds. Use the value level 0 was
  written with.

## Sharding

An unsharded store writes one file per chunk per array, so a large store can
have millions of small files. Sharding packs `N × N × N` neighbouring chunks
into one file per array, which suits cloud upload:

```bash
zvtools convert tracts.trk tracts.zv --num-chunks 1000 --shard 8   # after writing
zvtools shard tracts.zv --shape 8                                  # an existing store
zvtools shard tracts.zv --unshard                                  # back to one file per chunk
```

`8` packs up to 512 chunks into each file. The data does not change, only its
file layout. A server that serves a sharded store to the viewer must support
HTTP Range requests (see [Visualise](visualise.md#serving-requirements)).

## Position dtype

`--dtype` sets the stored position type (default `float32`). Keep `float32` for
anything you want to view: the viewer requires it, and precomputed ingest
refuses anything else.
