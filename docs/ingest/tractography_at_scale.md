# Tractography at scale

`ingest_trk_parallel` converts a TRK file that does not fit in RAM. It
never holds the whole tractogram: the file is sliced into byte ranges,
each worker reads only its own range, and the coordinator only ever
touches index-sized data. This is the path `--format trk` takes on the
CLI, and the only ingest in the package that builds its pyramid inline.

```python
from zarr_vectors_tools.convert.ingest.trk_parallel import ingest_trk_parallel

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
geometry — see below — which keeps chunks near-isotropic without you
having to know the field of view in advance. Pass an explicit
`(nx, ny, nz)` tuple to `num_chunks` when you do.

## The pipeline

| Phase | Runs | What it does |
| --- | --- | --- |
| 0 | serial | Parse the 1000-byte TRK header with `struct` (no `nibabel`); resolve the voxmm→RASmm affine if registering. |
| 1 | serial | Offset-index scan: walk the file recording each streamline's byte offset, point count, byte span, and first vertex. Size `chunk_shape` from those vertices; partition into `n_parts` byte-balanced, non-overlapping parts. |
| A | parallel over parts | Each worker reads its byte range, bins streamlines into spatial chunks via `split_polyline_at_boundaries`, and writes a `.npz` of segment descriptors plus raw vertices — and reports the exact bbox of what it wrote. |
| — | serial | Union those bboxes into the store bounds and lay out the chunk grid. Refuse the grid if any chunk would exceed the per-cell ceiling (below). |
| B | parallel over chunks | Each worker assembles one spatial chunk from all `n_parts` `.npz` files and writes level 0. |
| Coordinator | serial | Rebuild `nonempty_chunks` manifests from the on-disk cells, write the object index, and write boundary-crossing links into `links/0/`. |
| 5 | serial | Store CRS/affine metadata and a `TRKHeader` for round-trip export. |
| 6 | parallel | Build the multiscale pyramid via `build_pyramid`. |

Phase 1's scan is cheap — roughly 6 s for 5.6M streamlines — because it
reads only each record's length prefix plus its first vertex, and seeks
past the rest of the payload.

### Why the grid comes from the data

A TRK header's `dimensions × voxel_size` is a *declared* field of view,
and nothing in the format checks it against the tracts. Files exist where
it is simply wrong. On one 10.7 GB tractogram the header declared a
152 mm axis whose points ran to 186 mm, and 780M of its 890M vertices
(87.7%) fell outside the declared box altogether.

So the grid is not built from the header. `chunk_shape` — chunk *size*
only — comes from the bbox of one vertex per streamline, gathered free
during the Phase 1 scan; on that file it landed within 0.12 mm of the
true bbox on every axis. The grid's origin and extent then come from the
exact bboxes Phase A reports, which is why the store is not created until
Phase A has finished.

The alternative is what the ingest used to do: clamp any vertex outside
the header's box into the nearest edge chunk. That preserves coordinates
but files them under a chunk coord that does not contain them, which
breaks spatial queries, the chunk-local coarsener, and a viewer's
per-chunk bounds. On that file it left one cell holding 415M vertices of
which 96% were foreign to it — and, at 4.6 GiB, past the ceiling below.

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

### The 4 GiB per-chunk ceiling

`num_chunks` has a hard floor set by the data, not by taste. Every
per-chunk array is a Zarr v3 `vlen-bytes` array, and that codec records
each cell's byte length as a **uint32** — so a chunk holding 4 GiB or
more of vertices (358M xyz float32 vertices) wraps that header modulo
2<sup>32</sup>. numcodecs range-checks nothing: all the bytes land on
disk, the recorded length becomes `len % 2**32`, and every later read
returns a truncated buffer. Compression does not buy headroom, because
`vlen-bytes` writes the length before any bytes→bytes compressor runs.

Averages hide this. A 10.7 GB tractogram over a 6×6×7 grid averages
42 MB per cell, but tractography is not uniformly dense — only 16 of
those 252 cells were occupied, and one held 415M vertices (4.6 GiB).

The ingest checks for it after Phase A, once binning knows the exact
per-chunk counts and before the store is created, and refuses the grid
with the offending chunk and a suggested `num_chunks`:

```text
spatial grid too coarse for this input: chunk (5, 5, 6) holds 415,322,477
vertices = 4.6 GiB. A single chunk's payload must stay under 4.0 GiB [...]
Re-run with a finer grid — num_chunks (--num-chunks) of about 1,200 or more
```

The suggestion assumes the dense region subdivides evenly; for a tight
bundle, go higher. Chunks that large are worth avoiding on their own
merits anyway — a multi-gigabyte cell is a poor unit of progressive
fetch for any viewer.

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

A directory you name is never deleted, and the ingest keeps a progress
record in it, `trk_parallel_run.json`, with the options each stage ran
with.

## Resuming a run that stopped

On a whole-brain file the offset scan and Phase A take hours, and the
pyramid as long again. A run given `intermediate_dir` records each stage as
it completes, and `resume=True` skips what is recorded and still on disk:

| Stage | On resume |
| --- | --- |
| offset scan | reused if the file's size and modification time are unchanged |
| Phase A | its part files are reused |
| level 0 | reused if it finished; a partial store is removed and rebuilt from the part files |
| pyramid | finished levels are kept; the level that was being written is removed and rebuilt |

```bash
zvtools convert tracts.trk tracts.zarrvectors --num-chunks 5000 --workers 8 \
    --coarsen 8,8 --sparsity 1,4 --scratch-dir /scratch/tracts --resume
```

Pass the same options on every run, including the first. A resume whose
options would change what an earlier stage wrote is refused, and the error
names the options that differ. Pyramid options only decide which levels are
kept. A leftover store at the output path is removed only when the record
shows this ingest created it. In a [recipe](../getting_started/cli.md),
`scratch_dir` and `resume: true` make a rerun after a failure pick up where
it stopped.

## See also

- [Tractography](tractography.md) — the in-memory TCK / TRK / TRX readers
- [Parallelism](../how_to/parallelism.md) — executors, workers, and backends
- [Compressors](../how_to/compressors.md)
- [Large-scale pipelines](../how_to/large_scale_pipelines.md)
- [Building pyramids](../multiresolution/building_pyramids.md)
- [Ingest workflows](index.md)
