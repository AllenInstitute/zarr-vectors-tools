# Point clouds

Three formats land here: comma-separated text (`ingest_csv`), the LAS /
LAZ binary point-cloud format (`ingest_las`), and PLY (`ingest_ply`,
points-only for v0; mesh PLY is a future addition). All three write to
the **point cloud** ZVF geometry.

## CSV / XYZ — `ingest_csv`

Plain text with `delimiter`-separated coordinates and optional
attribute columns. Headers are recognised by name; without a header the
first `ndim` columns are positions and the rest are attributes.

```python
from zarr_vectors_tools.ingest import ingest_csv

ingest_csv(
    input_path="lidar.csv",
    output_path="lidar.zv",
    chunk_shape=(50.0, 50.0, 50.0),
    auto_detect_columns=True,
    knn_distance_k=8,
)
```

**Notable options**

`auto_detect_columns`
: Heuristically resolve `x/y/z` plus `r/g/b`, `intensity`, `label`,
  `confidence` from the header row. Falls back to positional inference
  if names don't match.

`normalise`
: Centre and rescale positions to `[-1, 1]`. The offset and scale are
  stored on the [`CSVHeader`](../headers.md) so the export round-trip
  recovers the original coordinates.

`drop_na`, `drop_duplicates`
: Pre-filter rows.

`knn_distance_k`
: When set, writes `attributes["knn_distance"]` — the mean Euclidean
  distance to the `k` nearest neighbours. Requires `scipy`.

`object_ids` + `per_object_vertex_count`
: Supply a `(N,)` integer array of per-vertex object IDs. With
  `per_object_vertex_count=True`, `object_attributes["vertex_count"]`
  is also written.

## LAS / LAZ — `ingest_las`

```python
from zarr_vectors_tools.ingest import ingest_las

ingest_las(
    input_path="scan.laz",
    output_path="scan.zv",
    chunk_shape=(10.0, 10.0, 10.0),
    include_attributes=True,
)
```

Requires `laspy` (install with `pip install "zarr-vectors-tools[las]"`).
`include_attributes=True` picks up every standard LAS field that's
present in the file:

| LAS field | ZVF attribute name | dtype |
| --- | --- | --- |
| `intensity` | `intensity` | float32 |
| `classification` | `classification` | float32 (cast from int) |
| `red`, `green`, `blue` | `color` | float32, shape `(N, 3)` |
| `gps_time` | `gps_time` | float32 |

`knn_distance_k`, `object_ids`, and `per_object_vertex_count` work the
same as for `ingest_csv`.

## PLY (points) — `ingest_ply`

```python
from zarr_vectors_tools.ingest import ingest_ply

ingest_ply(
    input_path="cloud.ply",
    output_path="cloud.zv",
    chunk_shape=(1.0, 1.0, 1.0),
    include_attributes=True,
)
```

Requires `plyfile`. Positions are detected from `x/y/z` or `X/Y/Z`
properties; the first three properties are used as a fallback. With
`include_attributes=True`, every non-position vertex property is
written as a per-vertex attribute under its PLY name.

`knn_distance_k`, `object_ids`, and `per_object_vertex_count` behave
identically to the other two ingesters.

## See also

- [Enrichments](../enrichments.md#point-clouds) — full attribute name list.
- [Export → PLY / CSV](../export/point_clouds.md)
- Parent: [Point cloud spec](https://zarr-vectors.readthedocs.io/en/latest/spec/geometry_types/point_cloud.html)
