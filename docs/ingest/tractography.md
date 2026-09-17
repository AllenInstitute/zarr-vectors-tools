# Tractography

Streamlines from diffusion MRI are polylines: an ordered run of vertices
joined by implicit sequential edges, one object per tract. Three readers
cover the common formats.

| Format | Ingest function | Reader | Extra |
| --- | --- | --- | --- |
| MRtrix TCK | `zarr_vectors_tools.convert.ingest.tck.ingest_tck` | `nibabel` | `streamlines` |
| TrackVis TRK | `zarr_vectors_tools.convert.ingest.trk.ingest_trk` | `nibabel` | `streamlines` |
| TRX | `zarr_vectors_tools.convert.ingest.trx.ingest_trx` | `trx-python` | `streamlines` |

:::{warning}
All three load the whole tractogram into memory. Above roughly a million
streamlines, use `ingest_trk_parallel` instead — see
[Tractography at scale](tractography_at_scale.md). It is also what
`--format trk` on the CLI routes to.
:::

## MRtrix TCK — `ingest_tck`

```python
from zarr_vectors_tools.convert.ingest.tck import ingest_tck

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
from zarr_vectors_tools.convert.ingest.trk import ingest_trk

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
`dimensions`, `vox_to_ras`, `voxel_order`, the scalar / property names, and
`space="rasmm"`, so `export_trk` can rebuild a valid file. (Before this was
fixed the header was read with the wrong keys inside a silent `except`, and
no store from `ingest_trk` had one.) A multi-column scalar such as an RGB
colour stays one three-column attribute.

The parallel ingester (`zvtools convert x.trk`) also carries the file's
scalars and properties, under the names in its header, unless
`keep_scalars=False` / `keep_properties=False`. A file name that collides
with a generated attribute (`--vertex-attr`, `--object-attr`) is refused
rather than overwritten.

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

TRX is structurally the closest format to Zarr Vectors — both keep positions,
offsets, and per-vertex/per-object data in separate arrays.

```python
from zarr_vectors_tools.convert.ingest.trx import ingest_trx

ingest_trx(
    "tracts.trx",
    "tracts.zv",
    (5.0, 5.0, 5.0),
    mean_scalar=["fa", "md"],      # per-streamline means of two dpv scalars
    compute_endpoints=True,
)
```

| TRX field | Zarr Vectors destination |
| --- | --- |
| `positions` | `vertices/` |
| `offsets` | vertex group boundaries (one group per streamline) |
| `dpv/<name>` | `attributes/<name>/` (per-vertex) |
| `dps/<name>` | `object_attributes/<name>/` (per-streamline) |
| `groups/<name>` | `groupings/`, one row per group, named after it |
| `dpg/<group>/<name>` | `groupings_attributes/<name>/`, one row per group; NaN where a group lacks the key |
| header `VOXEL_TO_RASMM`, `DIMENSIONS` | `TRXHeader` under `/headers/trx/` |
| *(computed)* | `object_attributes/mean_<name>/` via `mean_scalar` |

`mean_scalar` takes a name or a list of names of `dpv` scalars and writes
the per-streamline mean of each. Names absent from the file are skipped
silently — no error, no attribute.

Group names are kept on the groupings array, where core's group catalogue
reads them and the pyramid carries them to every level:

```python
import zarr_vectors as zv

level = zv.open("tracts.zv", mode="r").level(0)
level.groups.names()               # ("AF_L", "CST_L", ...)
level.groups["CST_L"].members      # object ids
```

Rows follow the order the TRX reader lists the groups in, so address a
bundle by name rather than by row. Stores written before names were kept
have a float `group_id` group attribute instead.

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

## Selecting streamlines by region

`select_streamlines` returns the object ids of the streamlines that meet a
region: a NIfTI mask (with its own affine), or a box in RAS millimetres.
Only the chunks the region touches are read.

```python
from zarr_vectors_tools.algorithms.streamline_select import select_streamlines
from zarr_vectors_tools.convert.export.trk import export_trk

through_cst = select_streamlines("tracts.zv", mask="cst_roi.nii.gz")
ending_in_m1 = select_streamlines("tracts.zv", mask="m1.nii.gz", mode="endpoints")
export_trk("tracts.zv", "cst.trk", object_ids=through_cst.tolist())
```

| `mode` | Selects a streamline when |
| --- | --- |
| `"path"` (default) | any vertex, or any point sampled along a segment, is inside |
| `"endpoints"` | either end is inside |
| `"both_endpoints"` | both ends are inside |

Points along each segment are tested every `sample_spacing` (default half
the mask's smallest voxel), so a simplified coarse level still finds a
streamline that steps over a small region. A TRK store kept in voxel
millimetres is mapped through its header first, so the same RAS mask selects
the same streamlines however the file was ingested. Reading only the
region's chunks misses one case: a segment longer than a chunk whose ends
both lie outside the chunks the region touches. At level 0 that does not
happen; at a heavily simplified level, select at a finer one.

## Bundle summaries

`bundle_summary` gives the first numbers a tract analysis wants, one row per
group: streamline count, length mean / std / min / median / max, mean
tortuosity, and start / end centroids. The rows are also written to the store
as `group_attributes/bundle_*`, so TRX export carries them as `dpg`.

```python
from zarr_vectors_tools.algorithms.bundles import bundle_summary, read_bundle_summary

table = bundle_summary("tracts.zv")               # level 0, written to every level
table.loc["CST_L", ["streamline_count", "length_mean", "start_z"]]
read_bundle_summary("tracts.zv", level=2)         # the same rows at a coarse level
bundle_summary("tracts.zv", level=2)              # level 2's own members and geometry
```

```bash
zvtools bundles tracts.zarrvectors --csv bundles.csv
```

By default a coarse level reports the whole level-0 bundle, not what
thinning left. Streamlines have no direction, so each bundle is turned to
agree with its main axis before endpoints are averaged; "start" is the low
end along that axis. Pass `orient_endpoints=False` (`--as-stored`) for
directed streamlines. Ingesting with `compute_length=True,
compute_endpoints=True` lets level 0 be summarised without reading any
geometry. A summary is a snapshot: edit the store's groups and run it again.

## See also

- [Tractography at scale](tractography_at_scale.md) — the parallel TRK pipeline
- [Enrichments → polylines and streamlines](../enrichments.md#polylines-and-streamlines)
- [Export → streamlines](../export/streamlines.md)
- [Headers](../headers.md)
- [Ingest workflows](index.md)
