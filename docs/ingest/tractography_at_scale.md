# Tractography at scale

`ingest_trk_parallel` converts a TRK file that does not fit in RAM. It
never holds the whole tractogram: the file is sliced into byte ranges,
each worker reads only its own range, and the coordinator only ever
touches index-sized data. This is the path `--format trk` takes on the
CLI, and the only ingest in the package that builds its pyramid inline.

```python
from zarr_vectors_tools.ingest.trk_parallel import ingest_trk_parallel

summary = ingest_trk_parallel(
    "hcp_5M.trk",
    "hcp.zv",
    num_chunks=125,              # target total spatial chunks; grid is near-isotropic
    n_parts=None,                # file slices for Phase A; default 4 × workers
    workers=8,                   # worker processes
    compressor="zstd",           # codec for the level-0 per-chunk arrays
    compute_length=True,         # required by the default "length" sparsity strategy
    compute_endpoints=True,
    build_multiscale=True,       # inline pyramid — no separate build_pyramid call
    pyramid_factors=[(8.0, 1.0), (8.0, 1.0)],   # (coarsen, sparsity) per coarser level
    progress=True,
)
```

Note there is no `chunk_shape` parameter. The grid is derived from the
TRK header's `dim × voxel_size` bounding box divided by `num_chunks`,
which keeps chunks near-isotropic without you having to know the field of
view in advance. Pass an explicit `(nx, ny, nz)` tuple to `num_chunks`
when you do.

## The pipeline

| Phase | Runs | What it does |
| --- | --- | --- |
| 0 | serial | Parse the 1000-byte TRK header with `struct` (no `nibabel`); derive bounds and the chunk grid from `num_chunks`. |
| 1 | serial | Offset-index scan: walk the file recording each streamline's byte offset, point count, and byte span. Partition into `n_parts` byte-balanced, non-overlapping parts. |
| A | parallel over parts | Each worker reads its byte range, bins streamlines into spatial chunks via `split_polyline_at_boundaries`, and writes a `.npz` of segment descriptors plus raw vertices. |
| B | parallel over chunks | Each worker assembles one spatial chunk from all `n_parts` `.npz` files and writes level 0. |
| Coordinator | serial | Rebuild `nonempty_chunks` manifests from the on-disk cells, write the object index, and write boundary-crossing links into `links/0/`. |
| 5 | serial | Store CRS/affine metadata and a `TRKHeader` for round-trip export. |
| 6 | parallel | Build the multiscale pyramid via `build_pyramid`. |

Phase 1's scan is cheap — roughly 17 s for 5M streamlines — because it
reads only each record's length prefix and seeks past the payload.

The manifest rebuild after Phase B is not optional bookkeeping. Phase B
workers each read-modify-write the shared `nonempty_chunks` manifest from
separate processes, so those updates race and under-report even though
every cell file landed on disk. The coordinator re-derives the manifests
single-process before anything downstream enumerates chunks.

## Tuning `num_chunks`, `n_parts`, and `workers`

| Knob | Default | Controls |
| --- | --- | --- |
| `num_chunks` | `125` | Total spatial chunks (or an explicit per-axis triple). Sets chunk *size*, hence read granularity for consumers. |
| `n_parts` | `4 × workers`, minimum 16 | How finely the input file is sliced for Phase A. |
| `workers` | `cpu_count() - 1` | How many processes run at once. |

`n_parts` and `workers` are independent. More parts means better load
balancing — streamline density varies wildly along a TRK file — and the
excess parts simply queue on the executor. Raising `n_parts` does not
raise peak memory, because a part is bounded by its byte range; raising
`workers` does, because that many parts are resident at once.

:::{note}
`max_streamlines=N` ingests only the first N streamlines in on-disk
order. Use it to shake out chunk-grid and pyramid settings on a subset
before committing hours to the full file.
:::

## Links and the merged layout

Streamline connectivity inside a fragment is implicit-sequential, so no
intra-fragment link records are written at all. The only records that
land are the boundary crossings: consecutive segments of one streamline
that fell in different spatial chunks. They are written with
`write_links(..., delta=0)`, i.e. as a non-zero offsets segment of the
ordinary `links/0/` family. There is no separate group for them.

## Compression

`compressor=None` (default) stores raw, so vertices cost exactly
`n_vertices × ndim × itemsize` — the same payload the `.trk` holds.
`"zstd"` or `"blosc"` roughly halves that, measured at about 2.4× on HCP
tract coordinates, at the cost of a slower synchronous encoding write
path.

:::{warning}
A chunk array's codec pipeline is fixed when the array is **created**, so
the compressor has to be set on this call — it cannot be applied
afterwards. It is also passed through to `build_pyramid`, because coarser
levels create their own arrays and level 0's codec does not propagate.
:::

## Intermediates

Phase A's `.npz` files go to a temporary directory by default. Point
`intermediate_dir` at a fast local disk with room for roughly the input
file's size, and set `keep_intermediate=True` when debugging a failed
run. Cleanup errors are swallowed deliberately: by the time cleanup runs
the store is written, so a straggling scratch file must never fail an
otherwise complete ingest.

## See also

- [Tractography](tractography.md) — the in-memory TCK / TRK / TRX readers
- [Parallelism](../how_to/parallelism.md) — executors, workers, and backends
- [Compressors](../how_to/compressors.md)
- [Large-scale pipelines](../how_to/large_scale_pipelines.md)
- [Building pyramids](../multiresolution/building_pyramids.md)
- [Ingest workflows](index.md)
