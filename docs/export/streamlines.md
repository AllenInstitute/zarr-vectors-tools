# Streamlines

Two targets for the Zarr Vectors `polylines` geometry: TrackVis TRK and TRX. Each
has its own extra, and `streamlines` installs both:

```bash
pip install "zarr-vectors-tools[trk]"          # nibabel only
pip install "zarr-vectors-tools[trx]"          # trx-python only
pip install "zarr-vectors-tools[streamlines]"  # both
```

:::{note}
The extras are split because `nibabel` is pure Python while `trx-python`
ships compiled wheels. Only the `trk` half installs in a WebAssembly
runtime such as Pyodide. Neither exporter imports the other's
dependency: `from zarr_vectors_tools.convert.export.trk import export_trk`
succeeds with `trx-python` absent.
:::

Both exporters write back what the store kept, not just positions:

| Store | TRK | TRX |
| --- | --- | --- |
| per-vertex attributes | scalars (`data_per_point`) | `dpv` |
| per-object attributes | properties (`data_per_streamline`) | `dps` |
| named object groups and their attributes | — | `groups` and `dpg` |
| reference image from the TRK or TRX header | `vox_to_ras`, voxel size, dimensions, voxel order | `VOXEL_TO_RASMM`, `DIMENSIONS` |

Each exporter reassembles an object's stored segments into one streamline,
and raises `ExportError` if the filters leave nothing to export.

## Which attributes are written

`attribute_names` (per-vertex) and `object_attribute_names` (per-object)
choose them:

- `None`, the default, writes every numeric attribute. Text attributes,
  such as labels stored as fixed-width bytes, cannot go in a streamline
  file; they are skipped and listed under `attributes_skipped` in the
  summary.
- `[]` writes none.
- A list writes exactly those. A name the level does not have, or one that
  is not numeric, raises `ExportError` naming it.

TRK holds at most ten scalar names and ten property names, each up to 20
characters. An export that asks for more raises `ExportError` and suggests
narrowing the selection.

## Coordinate space

A store keeps streamlines in one of two spaces, and its TRK header records
which:

- `voxmm`: TrackVis voxel millimetres, as the file had them. This is the
  default for `zvtools convert x.trk` (the parallel ingest).
- `rasmm`: RAS millimetres. This is what `--apply-affine`, the serial
  `ingest_trk`, TRX and TCK produce.

Both exporters convert from the stored space, so a file written from
either kind of store loads in the same place. The summary's `space`
reports which one the store was in. Stores written before the header
recorded it fall back to the `crs` the parallel ingest stamps.

## TrackVis TRK — `export_trk`

```python
from zarr_vectors_tools.convert.export.trk import export_trk

summary = export_trk(
    "tracts.zv",                # store_path
    "tracts.trk",               # output_path
    level=0,                    # resolution level to read
    object_ids=[3, 5, 17],      # keep only these streamlines
    group_ids=None,             # keep only these groupings
    chunks=None,                # chunk whitelist — see the warning below
    attribute_names=None,       # every numeric per-vertex attribute
    object_attribute_names=["length"],
)
summary["streamline_count"]
summary["attributes_carried"]         # e.g. ["fa", "rgb"]
summary["object_attributes_carried"]  # ["length"]
```

The header comes from the store's `TRKHeader`. A store ingested from TRX
takes its reference image from the `TRXHeader` instead, with the voxel
order derived from the affine. `affine=` is for a store with neither: the
positions are then treated as voxel coordinates of that affine, and with
no affine at all the identity is used.

## TRX — `export_trx`

```python
from zarr_vectors_tools.convert.export.trx import export_trx

summary = export_trx(
    "tracts.zv",
    "tracts.trx",
    level=0,
    object_ids=None,
    group_ids=[2],              # also limits the groups written to these
    chunks=None,
)
summary["groups_carried"]       # e.g. ["CST_L"]
```

A group lists output streamlines, so each member object is mapped through
the streamlines actually written, and a group with none is left out. Group
names are the store's (`zvtools convert` of a TRX keeps the bundle names),
or `group_<id>` for an unnamed row. Group attributes become that group's
`dpg`. NaN values are skipped, because that is how the TRX ingest fills a
key one group did not have.

## From the command line

```bash
# Everything the store kept.
zvtools convert tracts.zarrvectors out.trk

# Only FA per point and length per streamline.
zvtools convert tracts.zarrvectors out.trk --attribute fa --object-attribute length

# One bundle to TRX, with its name and per-group data.
zvtools convert atlas.zarrvectors cst.trx --group-id 2
```

## The segment-level `chunks` caveat

:::{warning}
On both `export_trk` and `export_trx` the `chunks` filter selects at the
**segment** level, not the object level. A streamline crossing a chunk
boundary is stored as several segments, and each surviving contiguous run
becomes its own streamline in the output. The returned
`streamline_count` can therefore exceed the number of source objects, and
individual streamlines are cut short at the boundary of the whitelist.
:::

```python
# Objects here are whole streamlines — counts match the source.
by_object = export_trk("tracts.zv", "a.trk", object_ids=[7])

# Chunks here cut object 7 into however many segments it spans.
by_chunk = export_trk("tracts.zv", "b.trk", chunks=[(0, 0, 0), (0, 0, 1)])
by_chunk["streamline_count"] >= by_object["streamline_count"]   # possibly much greater
```

Use `object_ids` or `group_ids` whenever you need whole objects; reach
for `chunks` only when you genuinely want a spatial slab and can tolerate
cut streamlines. Per-object attributes follow the object, so every run
cut from one object carries that object's values.

## See also

- [Export overview](index.md) — shared call shape and the `level=` parameter.
- [Ingest → tractography](../ingest/tractography.md) — the symmetric direction.
- [Tractography at scale](../ingest/tractography_at_scale.md) — the parallel TRK path.
- [Headers](../headers.md) — `TRKHeader`, `TRXHeader` and the `HeaderRegistry` API.
