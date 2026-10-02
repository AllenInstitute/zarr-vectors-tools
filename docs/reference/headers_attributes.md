# Headers and attributes

## Format headers

An ingest keeps metadata from the source file that the geometry cannot hold,
as JSON in the attributes of `headers/<format>/zarr.json`. Exporters back to the
same family of formats read it.

```python
from zarr_vectors_tools.headers import HeaderRegistry

reg = HeaderRegistry("tracts.zv")
reg.available_formats      # ['trk']
trk = reg.get("trk")       # TRKHeader
trk.space, trk.voxel_size  # ('voxmm', (1.0, 1.0, 1.0))
trk.affine                 # 4 x 4 voxel-to-RAS array
```

| Header | Written by | Holds | Read by |
|---|---|---|---|
| `trk` | trk; Python `ingest_trk`, `ingest_linc_trk` | `voxel_size`, `dimensions`, `vox_to_ras`, `voxel_order`, `scalar_names`, `property_names`, `n_count`, `origin`, `version`, `space` of the stored positions (`voxmm` or `rasmm`) | trk export (header fields; voxmm positions written back unchanged); trx export (voxmm to RAS, and the reference when there is no `trx` header) |
| `trx` | trx | `voxel_to_rasmm`, `dimensions`, `dpv_names`, `dps_names`, `dpg_names` | trx export (reference image); trk export of a TRX-ingested store |
| `h5ad` | h5ad, table; extended by `zvtools attach` | `spatial_key`, `spatial_columns`, `n_obs`, `n_vars`, original obs/gene names and their attribute names, `categories`, source `dtypes`, `row_attr`, `obs_index` (identifiers, up to 200,000 rows), `object_id_column`, `layer`, `position_names` (a table's position columns) | h5ad export: names, categoricals, row order, `obs_names`; csv export: position column names |
| `surface` | gifti, freesurfer; extended by `zvtools attach` (CIFTI) | `source`, `space`, `geometry`, per hemisphere `object_id`, CIFTI `structure`, `n_vertices`, `n_faces`; `alternates` (`coords_<surface>`), `scalars`, parcellation `label_tables`, `key_attribute`, `c_ras` | gifti export (required); precomputed export (unit millimetre, hemisphere as segment label) |
| `swc` | swc | `comment_lines`: the file's `#` lines | swc export writes them back, then its own `# SWC exported by zarr-vectors` line |
| `obj` | obj with `--split-objects` | `object_names` from `o`/`g` lines | precomputed export (segment labels); obj export does not restore them |
| `csv` | csv | columns, delimiter, header row or not, `position_columns`; with Python `normalise=True`, `normalise_offset` and `normalise_scale` | csv export: position column names. It writes normalised coordinates as stored |
| `graph` | Python `ingest_edgelist` / `ingest_graphml` with `compute_summary=True` | `node_count`, `edge_count`, `is_directed`, `mean_degree`, `n_components`, `largest_component_size` | nothing |

tck, ply, las, stl, lines, precomputed layers and `zvtools synapses` write no
header. ply export reads none either. Exports read headers and never add one. `zvtools merge` copies headers; a second
header of the same format is stored as `<format>@<source>`, after the source's
file or store name, e.g. `trk@subject_b` ([Merge](../compose.md)).

## Attributes each ingest writes

Paths are per level: `<level>/vertex_attributes/<name>` and
`<level>/object_attributes/<name>`. An attribute marked ×3 has three columns
per row (row shape `(3,)` in its metadata); all others hold one value per row.
The viewer colours by single-column attributes only
([Visualise](../visualise.md)). "Python" marks options that only the Python
function has.

| Input | Vertex attributes | Object attributes |
|---|---|---|
| trk | the file's scalars (float32; a multi-column scalar stays one attribute); `--vertex-attr` `arc_length`, `index`, `x`, `y`, `z`, `random` (float32), `tangent` (float32 ×3) | the file's properties (float32); `length` (float32, `--compute-length`); `start`, `end` (float32 ×3, `--compute-endpoints`); `--object-attr` `orientation` (float32 ×3), `tortuosity` (float32) |
| tck | none | `length`, `start`, `end` as for trk |
| trx | each `dpv` array (the file's dtype, float16 as float32; its own column count) | each `dps` array (the file's dtype; one value per row when the array has one column); `length`, `start`, `end` as for trk; `mean_<name>` (Python `mean_scalar=`). TRX groups become named groups, `dpg` arrays group attributes (float32) |
| LINC trk (Python `ingest_linc_trk`) | the file's scalars | `label_id` (int64); one group per label, with `name` and `color` (uint8 ×4) from the LUT |
| swc | `radius`, `compartment` (float32); Python `compute_topological_depth`, `compute_strahler`, `compute_node_kind` add `topological_depth`, `strahler`, `node_kind` (float32) | `segment_id` (uint64): 1 to N, one object per tree in root order |
| precomputed skeletons | the layer's vertex attributes, in the layer's data types | `segment_id` (uint64); without a spatial index also each segment property: text as 256-byte strings, numbers in their data type, tags as a uint64 bit mask |
| precomputed meshes | none | `segment_id` (uint64) and the segment properties |
| csv | each non-position column (float32, named from the header row); `knn_distance` (float32, `--knn-distance-k`) | none |
| table | each non-position, non-key column: numbers as read (e.g. float64, int64), text and categories as int32 codes dictionary-encoded (labels in the array's `categories`, -1 for missing), booleans uint8, dates int64; `zv_join_key` (int64, `--key-column`); `table_row` (int64) | none |
| h5ad | each obs column, encoded as for table; `gene_<name>` (float32, `--gene`); `h5ad_row` (int64); `zv_join_key` (int64); `knn_distance` | none |
| ply | each non-position vertex property (float32, one attribute each); `knn_distance` | none |
| las | `intensity`, `classification` (float32), `gps_time` (float64), `color` (float32 ×3); `knn_distance` | none |
| lines | none | each column after the six endpoint columns (float32); `length` (float32, `--compute-length`) |
| edgelist, graphml | each numeric node column or node key (float32); Python `compute_degree`, `compute_component`, `compute_clustering` add `degree`, `component`, `clustering` (float32) | none; edge columns go to `link_attributes/<name>` (float32) |
| obj | `normal` (float32 ×3) when the file has one `vn` per `v` | none |
| stl | none (facet normals are not kept) | none |
| gifti, freesurfer | each map (float32), each parcellation (int32 codes; names in the `surface` header), `coords_<surface>` (float32 ×3), `zv_join_key` (int64) | none; one object per hemisphere, in a group named `left` or `right` |
| `zvtools synapses` | `pre_segment_id` or `post_segment_id` (int64), each `--column` (float32), `zv_join_key` (int64) | `segment_id` (uint64); the precomputed skeleton store gains `synapse_pre_count` and `synapse_post_count` (int64) |

`zvtools attach` adds vertex attributes encoded as for table and h5ad, or CIFTI
maps as for gifti ([Single-cell and spatial tables](../convert/single_cell.md),
[Cortical surfaces](../convert/surfaces.md)). For attributes at coarser levels, see
[Pyramids](../pyramids.md).

Every store with objects (all but a point cloud ingested without object ids) also
gets two columns the viewer reads:

- `object_attributes/vertex_count` (uint32): the object's vertices at that level. The
  viewer budgets objects by it. Python `per_object_vertex_count=False` (point
  ingests) skips it.
- `fragment_attributes/segment_id` (uint64), one per fragment: the object's
  `segment_id` where it has one, else its object id. The viewer colours and picks by
  it, so an object keeps one colour across chunks; streamline and skeleton pyramids
  read it too. Skeleton stores add `fragment_attributes/object_id`.
