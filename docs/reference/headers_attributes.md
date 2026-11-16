# Headers and attributes

The format headers each ingest keeps in a store, and the attributes each
ingest writes.

## Format headers

An ingest keeps source metadata that the geometry cannot hold as JSON in the
attributes of `headers/<format>/zarr.json`. Exports to the same family of
formats read it; they never add a header.

```python
from zarr_vectors_tools.headers import HeaderRegistry

reg = HeaderRegistry("tracts.zv")
reg.available_formats      # ['trk']
trk = reg.get("trk")       # TRKHeader
trk.space, trk.voxel_size  # ('voxmm', (2.0, 2.0, 2.0))
```

| Header | Written by | Holds | Read by |
|---|---|---|---|
| `trk` | trk; Python `ingest_trk`, `ingest_linc_trk` | `voxel_size`, `dimensions`, `vox_to_ras`, `voxel_order`, `scalar_names`, `property_names`, `n_count`, `origin`, `version`, `space` (`voxmm` or `rasmm`) | trk export; trx export (voxmm to RAS, and the reference when there is no `trx` header) |
| `trx` | trx | `voxel_to_rasmm`, `dimensions`, `dpv_names`, `dps_names`, `dpg_names` | trx export; trk export of a TRX-ingested store |
| `h5ad` | h5ad, table; extended by `zvtools attach` | `spatial_key`, `spatial_ndim`, `spatial_columns`, `n_obs`, `n_vars`, `obs_names`, `gene_names`, `obs_attrs`, `gene_attrs`, `categories`, `dtypes`, `row_attr`, `obs_index` (up to 200,000 rows), `object_id_column`, `object_id_categories`, `layer`, `position_names` | h5ad export; csv export (position column names) |
| `surface` | gifti, freesurfer; extended by CIFTI attach | `source`, `space`, `geometry`, per-hemisphere `object_id`, `structure`, `n_vertices`, `n_faces`; `alternates`, `scalars`, `label_tables`, `key_attribute`, `c_ras`, `applied_transforms`, `series` | gifti export (required); precomputed export |
| `swc` | swc | `comment_lines`, `coordinate_space`, `scaling` | swc export |
| `obj` | obj with `--split-objects` | `object_names`, `group_names`, `mtllib` | precomputed export (segment labels) |
| `csv` | csv | `column_names`, `delimiter`, `has_header_row`, `position_columns`, `attribute_columns`; `normalise_offset`, `normalise_scale` (Python `normalise=True`) | csv export (position column names) |
| `graph` | Python `ingest_edgelist`, `ingest_graphml` with `compute_summary=True` | `node_count`, `edge_count`, `is_directed`, `mean_degree`, `n_components`, `largest_component_size` | nothing |

tck, ply, las, stl, lines, precomputed layers and `zvtools synapses` write no
header. `zvtools merge` stores a second header of the same format as
`<format>@<source>`, such as `trk@subject_b` ([Merge](../compose.md)).

## Attributes each ingest writes

Paths are per level: `<level>/vertex_attributes/<name>` and
`<level>/object_attributes/<name>`. ×3 marks three columns per row. "Python"
marks options only the Python function has.

| Input | Vertex attributes | Object attributes |
|---|---|---|
| trk | the file's scalars (float32); `--vertex-attr` `arc_length`, `index`, `x`, `y`, `z`, `random` (float32), `tangent` (float32 ×3) | the file's properties (float32); `length` (float32, `--compute-length`); `start`, `end` (float32 ×3, `--compute-endpoints`); `--object-attr` `orientation` (float32 ×3), `tortuosity` (float32) |
| tck | none | `length`, `start`, `end` as for trk |
| trx | each `dpv` array (the file's dtype; float16 as float32) | each `dps` array (the file's dtype); `length`, `start`, `end` as for trk; `mean_<name>` (Python `mean_scalar=`). Groups become named groups, `dpg` arrays group attributes (float32) |
| LINC trk (Python) | the file's scalars | `label_id` (int64); one group per label, with `name` and `color` (uint8 ×4) |
| swc | `radius`, `compartment` (float32); Python `topological_depth`, `strahler`, `node_kind` (float32) | `segment_id` (uint64): 1 to N in root order |
| precomputed skeletons | the layer's vertex attributes, in their data types | `segment_id` (uint64); without a spatial index, each segment property (text as 256-byte strings, tags as a uint64 bit mask) |
| precomputed meshes | none | `segment_id` (uint64) and the segment properties |
| csv | each non-position column (float32); `knn_distance` (float32) | none |
| table | each non-position, non-key column: numbers as read, text and categories as int32 codes (labels in the array's `categories`, -1 for missing), booleans uint8, dates int64; `zv_join_key` (int64, `--key-column`); `table_row` (int64) | none |
| h5ad | each obs column, as for table; `gene_<name>` (float32); `h5ad_row` (int64); `zv_join_key` (int64); `knn_distance` | none |
| ply | each non-position vertex property (float32); `knn_distance` | none |
| las | `intensity`, `classification` (float32), `gps_time` (float64), `color` (float32 ×3); `knn_distance` | none |
| lines | none | each column after the six endpoints (float32); `length` (float32, `--compute-length`) |
| edgelist, graphml | each numeric node column (float32); Python `degree`, `component`, `clustering` (float32) | none; edge columns go to `link_attributes/<name>` (float32) |
| obj | `normal` (float32 ×3), when the file has one `vn` per `v` | none |
| stl | none | none |
| gifti, freesurfer | each map (float32), each parcellation (int32 codes), `coords_<surface>` (float32 ×3), `zv_join_key` (int64) | none; one object per hemisphere, in a group `left` or `right` |
| `zvtools synapses` | `pre_segment_id` or `post_segment_id` (int64), each `--column` (float32), `zv_join_key` (int64) | `segment_id` (uint64); the skeleton store gains `synapse_pre_count` and `synapse_post_count` (int64) |

`zvtools attach` adds vertex attributes encoded as for table and h5ad, or as
for gifti from CIFTI. Coarser levels: [Pyramids](../pyramids.md).

Every store with objects also gets two arrays the viewer reads:

| Array | Type | Holds |
|---|---|---|
| `object_attributes/vertex_count` | uint32 | The object's vertex count at that level. Python `per_object_vertex_count=False` (point ingests) skips it. |
| `fragment_attributes/segment_id` | uint64 | One per fragment: the object's `segment_id`, else its object id. The viewer colours and picks by it. |

Skeleton stores also get `fragment_attributes/object_id`.
