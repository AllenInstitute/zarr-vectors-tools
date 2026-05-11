# zarr-vectors-tools

> [!NOTE]
> This package is under development and will change.

**Ingest and export workflows for the Zarr Vector Format (ZVF).**

`zarr-vectors-tools` is the companion workflow package to [`zarr-vectors-py`](https://github.com/Andrew-Keenlyside/zarr-vectors-py). The core read/write APIs (chunking, sharding, multiresolution, spatial binning) live in the `zarr_vectors` package. This package adds format-conversion workflows that wrap third-party readers and writers, plus example notebooks demonstrating end-to-end pipelines.

*Aligned to the Zarr Vectors specification by Forrest Collman, Allen Institute for Brain Sciences.*
[Specification](https://github.com/AllenInstitute/zarr_vectors)

---

## Install

```bash
pip install zarr-vectors-tools
```

## Supported formats

| Format    | Ingest | Export | Geometry        | Backend    |
|-----------|:------:|:------:|-----------------|------------|
| CSV       | yes    | yes    | points          | numpy      |
| LAS/LAZ   | yes    |        | points          | laspy      |
| PLY       | yes    | yes    | points / meshes | plyfile    |
| OBJ       | yes    | yes    | meshes          | pure-python |
| STL       | yes    |        | meshes          | pure-python |
| SWC       | yes    | yes    | skeletons       | numpy      |
| GraphML   | yes    |        | graphs          | networkx   |
| TRK       | yes    | yes    | streamlines     | nibabel    |
| TCK       | yes    |        | streamlines     | nibabel    |
| TRX       | yes    | yes    | streamlines     | trx-python |

The CSV-based ingesters cover three geometries: `ingest_csv` (points), `ingest_lines_csv` (line segments), and `ingest_edgelist` (graphs via a separate node-position CSV + edge CSV; supports cuDF acceleration via `use_cudf=True`).

## Optional enrichments

Every ingester accepts opt-in keyword arguments that compute derived attributes at ingest time. All default to off — existing calls are unchanged.

**CSV / LAS / PLY points** (`ingest_csv`, `ingest_las`, `ingest_ply`):

| kwarg | effect |
|---|---|
| `auto_detect_columns` (CSV only) | Detect `x/y/z`, `r/g/b`, `intensity`, `label`, ... from the header. |
| `drop_na` (CSV only) | Drop rows with NaN in any position column. |
| `drop_duplicates` (CSV only) | Drop rows with duplicate position triples. |
| `normalise` (CSV only) | Centre + scale positions to `[-1, 1]`; offset/scale stored in the CSV header for round-trip. |
| `knn_distance_k` | Compute mean kNN distance per vertex (`attributes["knn_distance"]`). Needs `scipy` (extra `[points-enrichment]`). |
| `per_object_vertex_count` | Add `object_attributes["vertex_count"]`. Requires `object_ids`. |

**Lines** (`ingest_lines_csv`):

| kwarg | effect |
|---|---|
| `compute_length` | Per-segment length as `line_attributes["length"]`. |
| `drop_zero_length` | Drop segments shorter than `1e-9`. |
| `drop_na`, `drop_duplicates` | Same semantics as CSV points. |

**Streamlines** (`ingest_trk`, `ingest_tck`, `ingest_trx`):

| kwarg | effect |
|---|---|
| `compute_length` | Per-streamline path length to `object_attributes["length"]`. |
| `compute_endpoints` | Start / end points to `object_attributes["start"]` and `["end"]`. |
| `length_range` | Filter streamlines whose length is outside `(min, max)`. |
| `mean_scalar` (TRX only) | For each named per-vertex scalar, store its per-streamline mean. |

**Skeletons** (`ingest_swc`):

| kwarg | effect |
|---|---|
| `compute_topological_depth` | Edge count from soma per node. |
| `compute_strahler` | Strahler stream order. |
| `compute_node_kind` | Soma / branch / continuation / terminal classification. |

**Graphs** (`ingest_graphml`, `ingest_edgelist`):

| kwarg | effect |
|---|---|
| `compute_degree` | Per-node degree. |
| `compute_component` | Per-node connected-component label. |
| `compute_clustering` | Per-node clustering coefficient. |
| `compute_summary` | Graph-level metrics stored in a `GraphHeader` (workaround until core grows a per-graph attribute slot). |

**Meshes** (`ingest_obj`):

| kwarg | effect |
|---|---|
| `auto_object_id` | Parse `o` / `g` directives; assign per-vertex `object_ids` and store names in `OBJHeader.object_names`. |

Install the enrichment extras as needed:

```bash
pip install zarr-vectors-tools[points-enrichment]   # scipy, for kNN
pip install zarr-vectors-tools[gpu]                 # cuDF, RAPIDS-only
```
