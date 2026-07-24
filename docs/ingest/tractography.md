# Tractography

Streamlines from diffusion MRI are polylines: an ordered run of vertices
joined by implicit sequential edges, one object per tract. Three readers
cover the common formats.

| Format | Ingest function | Reader | Extra |
| --- | --- | --- | --- |
| MRtrix TCK | `zarr_vectors_tools.ingest.tck.ingest_tck` | `nibabel` | `streamlines` |
| TrackVis TRK | `zarr_vectors_tools.ingest.trk.ingest_trk` | `nibabel` | `streamlines` |
| TRX | `zarr_vectors_tools.ingest.trx.ingest_trx` | `trx-python` | `streamlines` |

:::{warning}
All three load the whole tractogram into memory. Above roughly a million
streamlines, use `ingest_trk_parallel` instead — see
[Tractography at scale](tractography_at_scale.md). It is also what
`--format trk` on the CLI routes to.
:::

## MRtrix TCK — `ingest_tck`

```python
from zarr_vectors_tools.ingest.tck import ingest_tck

summary = ingest_tck(
    "tracts.tck",
    "tracts.zv",
    (5.0, 5.0, 5.0),               # chunk_shape in mm
    compute_length=True,           # object_attributes["length"]
    compute_endpoints=True,        # object_attributes["start"] and ["end"], (O, 3)
    length_range=(20.0, 250.0),    # drop tracts outside the range, before writing
)
print(summary["dropped_by_length"])
```

TCK carries positions and nothing else — no per-vertex scalars, no
per-streamline properties. Everything you can query downstream comes from
the enrichment options.

## TrackVis TRK — `ingest_trk`

```python
from zarr_vectors_tools.ingest.trk import ingest_trk

ingest_trk(
    "tracts.trk",
    "tracts.zv",
    (5.0, 5.0, 5.0),
    preserve_header=True,          # default — write TRKHeader for round-trip
    compute_length=True,
    length_range=(20.0, 250.0),
)
```

TRK's `data_per_point` becomes per-vertex `attributes/*` and
`data_per_streamline` becomes per-streamline `object_attributes/*`,
automatically, under the names the file uses. `length_range` re-indexes
both alongside the surviving streamlines, so a filtered store stays
internally consistent.

`preserve_header=True` writes a `TRKHeader` holding `voxel_size`,
`dimensions`, `vox_to_ras`, `voxel_order`, and the scalar / property
names, so `export_trk` can rebuild a valid file. It is best-effort: a
malformed header is swallowed and the ingest still succeeds.

### Coordinate conventions

A TRK file stores each point in **voxmm** — `voxel_index × voxel_size`,
millimetres measured along the voxel axes, not scanner axes. The vox→RAS
affine that maps voxmm to scanner space sits in the 1000-byte header.

`nibabel` applies that affine when it loads, so `ingest_trk` writes
positions already in RAS+ millimetres and the affine survives only as
header metadata. The parallel ingester takes the opposite route: it
parses the header itself, **keeps coordinates in voxmm**, and records the
affine as store CRS metadata (`input_space: "voxmm"`, `output_space:
"RASmm"`, `units: "mm"`, plus the flattened 4×4). Nothing is resampled —
a consumer that wants scanner space applies the affine at read time.

:::{note}
The two TRK paths therefore produce stores in *different* coordinate
spaces from the same input file. Do not mix them in one analysis without
checking the store's CRS metadata first.
:::

## TRX — `ingest_trx`

TRX is structurally the closest format to ZVF — both keep positions,
offsets, and per-vertex/per-object data in separate arrays.

```python
from zarr_vectors_tools.ingest.trx import ingest_trx

ingest_trx(
    "tracts.trx",
    "tracts.zv",
    (5.0, 5.0, 5.0),
    mean_scalar=["fa", "md"],      # per-streamline means of two dpv scalars
    compute_endpoints=True,
)
```

| TRX field | ZVF destination |
| --- | --- |
| `positions` | `vertices/` |
| `offsets` | vertex group boundaries (one group per streamline) |
| `dpv/<name>` | `attributes/<name>/` (per-vertex) |
| `dps/<name>` | `object_attributes/<name>/` (per-streamline) |
| `groups/<name>` | `groupings/` |
| `dpg/<name>` | `groupings_attributes/` |
| *(computed)* | `object_attributes/mean_<name>/` via `mean_scalar` |

`mean_scalar` takes a name or a list of names of `dpv` scalars and writes
the per-streamline mean of each. Names absent from the file are skipped
silently — no error, no attribute.

:::{note}
Group *names* are not preserved as strings. Each TRX group becomes an
integer-keyed grouping, and the only grouping attribute written is
`group_id`, a float index into the original name order. Keep the name
list yourself if you need to resolve tract labels later.
:::

When `length_range` drops streamlines, group membership is rebuilt
against the surviving indices rather than being invalidated.

## Shared enrichment options

All three functions accept the same three:

`compute_length`
: Writes `object_attributes["length"]` — the summed Euclidean distance
  along each streamline.

`compute_endpoints`
: Writes `object_attributes["start"]` and `["end"]`, each shape `(O, D)`
  — useful for endpoint-based tract selection without touching vertices.

`length_range=(min, max)`
: Drops streamlines outside the range *before* writing, and reports the
  count as `dropped_by_length` in the summary dict. Lengths are computed
  once and reused by `compute_length`.

If filtering removes everything, the ingest raises `IngestError` rather
than writing an empty store.

## Synthetic attributes for coloring test data (trk CLI)

The `zvtools convert … --format trk` path can *generate* colorable
attributes from geometry alone — handy when a TRK file has no native
scalars but you need test data for attribute coloring. Both flags are
repeatable and are wired into the parallel TRK path only (trx/tck carry
their own native scalars, so the flags are rejected there).

```bash
zvtools convert tracts.trk out.zarrvectors --format trk \
  --object-attr orientation --object-attr tortuosity --object-attr vertex_count \
  --vertex-attr arc_length --vertex-attr z --vertex-attr tangent --attr-seed 0
```

`--object-attr {length,endpoints,orientation,tortuosity,vertex_count}`
: Per-streamline (color *by object*). `length`/`endpoints` are the same
  data as `--compute-length`/`--compute-endpoints`; `orientation` is the
  start→end unit vector `(O, 3)` (DEC RGB); `tortuosity` is length ÷
  endpoint distance (≥ 1); `vertex_count` is points per streamline.

`--vertex-attr {arc_length,x,y,z,random,index,tangent}`
: Per-vertex (color *by vertex*). `arc_length` runs 0→1 along each
  streamline (a head→tail gradient); `x`/`y`/`z` are the coordinate
  value; `index` is 0→1 within each streamline; `tangent` is the
  per-vertex unit direction `(N, 3)` (DEC); `random` is seeded by
  `--attr-seed`.

Every generated attribute is carried through the sparsity pyramid, so
coloring renders at all zoom levels. See the full attribute table in
[Enrichments](../enrichments.md#synthetic-attributes-for-coloring-test-data-trk-cli-only).

## See also

- [Tractography at scale](tractography_at_scale.md) — the parallel TRK pipeline
- [Enrichments → polylines and streamlines](../enrichments.md#polylines-and-streamlines)
- [Export → streamlines](../export/streamlines.md)
- [Headers](../headers.md)
- [Ingest workflows](index.md)
