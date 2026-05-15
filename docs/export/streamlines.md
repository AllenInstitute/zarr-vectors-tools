# Streamlines

Two export targets for the ZVF polyline / streamline geometry: TrackVis
TRK and TRX. Both require neuroimaging libraries.

## TrackVis TRK — `export_trk`

```python
from zarr_vectors_tools.export import export_trk

export_trk(
    store_path="tracts.zv",
    output_path="tracts.trk",
    affine=None,                # uses identity if None
)
```

Requires `nibabel` (install with `pip install "zarr-vectors-tools[streamlines]"`).

Returns `{"streamline_count": int, "vertex_count": int}`.

**Notable options**

`affine`
: 4×4 voxel-to-RAS affine matrix. If `None`, uses identity. If the
  store was ingested with `ingest_trk(..., preserve_header=True)`, the
  original TRK affine is automatically loaded from
  [`TRKHeader`](../headers.md) when this argument is omitted.

`level`, `object_ids`, `group_ids`, `chunks`
: Standard filters. The `chunks` filter is special for streamlines: it
  filters at the **segment** level — only vertex groups stored in the
  listed chunks are emitted, and each surviving contiguous run is
  written as its own streamline. The output `streamline_count` can
  therefore exceed the source object count when a long streamline is
  cut at chunk boundaries.

## TRX — `export_trx`

```python
from zarr_vectors_tools.export import export_trx

export_trx(
    store_path="tracts.zv",
    output_path="tracts.trx",
    object_ids=[3, 5, 17],
)
```

Requires `trx-python` (install with `pip install "zarr-vectors-tools[streamlines]"`).

Returns `{"streamline_count": int, "vertex_count": int}`.

TRX is a direct round-trip target for ZVF streamlines — every
`attributes/*` (per-vertex) array becomes a `dpv` field, every
`object_attributes/*` array becomes a `dps` field, every `groupings/*`
group becomes a TRX `groups` entry, and every `groupings_attributes/*`
becomes a `dpg`.

`level`, `object_ids`, `group_ids`, `chunks` filters behave exactly as
they do for `export_trk`.

## See also

- [Ingest → polylines and streamlines](../ingest/polylines_streamlines.md)
