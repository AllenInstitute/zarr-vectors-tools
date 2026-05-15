# Polylines and streamlines

A **polyline** is an ordered sequence of vertices connected by implicit
sequential edges. **Streamlines** (in the diffusion-MRI sense) are
polylines with neuroimaging-specific metadata. Three ingest functions
cover the common file formats.

| Format | Function | Reader | Per-vertex data | Per-streamline data |
| --- | --- | --- | --- | --- |
| MRtrix TCK | `ingest_tck` | `nibabel` | – | – |
| TrackVis TRK | `ingest_trk` | `nibabel` | `data_per_point` | `data_per_streamline` |
| TRX | `ingest_trx` | `trx-python` | `dpv/*` | `dps/*` |

All three share the same enrichment options: `compute_length`,
`compute_endpoints`, `length_range`.

## MRtrix TCK — `ingest_tck`

```python
from zarr_vectors_tools.ingest import ingest_tck

ingest_tck(
    input_path="tracts.tck",
    output_path="tracts.zv",
    chunk_shape=(5.0, 5.0, 5.0),
    compute_length=True,
    length_range=(20.0, 250.0),     # drop too-short / too-long tracts
)
```

TCK stores streamlines in scanner (RAS) millimetre coordinates with
**no per-vertex attributes** — only positions. The format itself is
minimal; everything you get downstream is the result of the enrichment
options listed below.

## TrackVis TRK — `ingest_trk`

```python
from zarr_vectors_tools.ingest import ingest_trk

ingest_trk(
    input_path="tracts.trk",
    output_path="tracts.zv",
    chunk_shape=(5.0, 5.0, 5.0),
    preserve_header=True,
    compute_endpoints=True,
)
```

TRK carries a voxel-to-RAS affine and (optionally) per-vertex scalars
(`data_per_point`) and per-streamline properties (`data_per_streamline`)
inline. `preserve_header=True` round-trips the affine + scalar/property
names via [`TRKHeader`](../headers.md).

## TRX — `ingest_trx`

```python
from zarr_vectors_tools.ingest import ingest_trx

ingest_trx(
    input_path="tracts.trx",
    output_path="tracts.zv",
    chunk_shape=(5.0, 5.0, 5.0),
    mean_scalar=["fa", "md"],
)
```

TRX is the closest format to ZVF for streamlines. The mapping is direct:

| TRX | ZVF |
| --- | --- |
| `positions` | `vertices/` |
| `offsets` | vertex-group boundaries |
| `dpv/<name>` | `attributes/<name>/` |
| `dps/<name>` | `object_attributes/<name>/` |
| `groups/<name>` | `groupings/` |
| `dpg/<name>` | `groupings_attributes/` |

**TRX-only option**

`mean_scalar`
: Name (or list of names) of `dpv` scalars whose per-streamline mean
  should be written to `object_attributes["mean_<name>"]`. Names not
  present in the source are skipped silently.

## Shared enrichment options

`compute_length`
: Writes `object_attributes["length"]` — total path length (sum of
  segment Euclidean distances).

`compute_endpoints`
: Writes `object_attributes["start"]` and `["end"]` — first and last
  vertex of each polyline, shape `(O, D)`.

`length_range = (min, max)`
: Drop streamlines whose total length is outside the range before
  writing. The summary dict reports the count as `dropped_by_length`.

## See also

- [Enrichments](../enrichments.md#polylines-and-streamlines)
- [Export → TRK / TRX](../export/streamlines.md)
- Parent: [Polyline spec](https://zarr-vectors.readthedocs.io/en/latest/spec/geometry_types/polyline.html)
