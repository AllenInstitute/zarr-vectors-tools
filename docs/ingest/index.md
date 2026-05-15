# Ingest workflows

Every ingest function in `zarr_vectors_tools.ingest` reads a single file
format and writes a Zarr Vectors store. They all follow the same shape:

```python
result = ingest_<format>(
    input_path,         # path to the source file
    output_path,        # path for the new ZVF store
    chunk_shape,        # required spatial chunk size
    bin_shape=None,     # optional supervoxel bin size
    # ... format-specific options ...
)
```

Each one returns the summary dict from the underlying `zarr-vectors`
writer (`write_points`, `write_mesh`, `write_graph`, `write_polylines`,
or `write_lines`), augmented with any enrichment counters such as
`dropped_by_length` or `dropped_by_na`.

## Format matrix

| Format | Ingest function | ZVF geometry | Third-party deps | Auto-derived attributes |
| --- | --- | --- | --- | --- |
| CSV / XYZ point cloud | `ingest_csv` | point cloud | none (pure Python) | `knn_distance` *(scipy)*, `vertex_count` |
| LAS / LAZ | `ingest_las` | point cloud | `laspy` | `intensity`, `classification`, `color`, `gps_time`, `knn_distance`, `vertex_count` |
| PLY (points) | `ingest_ply` | point cloud | `plyfile` | every non-position PLY property, `knn_distance`, `vertex_count` |
| CSV line segments | `ingest_lines_csv` | line | none | `length` |
| MRtrix TCK | `ingest_tck` | polyline / streamline | `nibabel` | `length`, `start`, `end` |
| TrackVis TRK | `ingest_trk` | polyline / streamline | `nibabel` | TRK `data_per_point` + `data_per_streamline`, `length`, `start`, `end` |
| TRX | `ingest_trx` | polyline / streamline | `trx-python` | TRX `dpv/dps/groups/dpg`, `length`, `start`, `end`, `mean_<scalar>` |
| Edge-list CSV | `ingest_edgelist` | graph | `pandas` *(+ optional `networkx`, `cudf`)* | `degree`, `component`, `clustering` |
| GraphML | `ingest_graphml` | graph | `networkx` | `degree`, `component`, `clustering` |
| SWC | `ingest_swc` | skeleton (tree) | none | `radius`, `compartment`, `topological_depth`, `strahler`, `node_kind` |
| OBJ | `ingest_obj` | mesh | none | `normal` (if `vn` lines present), per-vertex `object_ids` (if `auto_object_id=True`) |
| STL (ASCII + binary) | `ingest_stl` | mesh | none | per-face normals from the STL |

Geometry-specific pages cover each format in depth:

- [Point clouds](point_clouds.md) — CSV, LAS, PLY
- [Lines](lines.md) — line-segment CSV
- [Polylines and streamlines](polylines_streamlines.md) — TCK, TRK, TRX
- [Graphs](graphs.md) — edgelist CSV, GraphML
- [Skeletons](skeletons.md) — SWC
- [Meshes](meshes.md) — OBJ, STL

The auto-derived attribute names listed in the matrix are referenced in
full at [Enrichments](../enrichments.md).

## Headers and round-trip

Several formats carry metadata that isn't part of the geometry itself:
TRK voxel-to-RAS affines, SWC `coordinate_space` comments, OBJ object
names, CSV normalisation parameters. These are preserved into
`/headers/<format>/` on the store so the matching `export_*` can
recover them. See [Headers](../headers.md).

## See also

- [Quickstart](../getting_started/quickstart.md)
- [Concepts](../getting_started/concepts.md)
- Parent package: [Geometry types](https://zarr-vectors.readthedocs.io/en/latest/spec/geometry_types/index.html)
