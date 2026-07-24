# Enrichments

Several ingest functions can compute extra attributes during import.
Some are derived directly from the source format (LAS `intensity`, SWC
`radius`), others are pure computations (`length`, `knn_distance`,
`strahler`). All land in the standard ZVF attribute slots — per-vertex
under `attributes/<name>/`, per-object under
`object_attributes/<name>/` — so they're accessible to any reader of
the store, not just this package.

This page is the canonical list of attribute names and the ingest
options that produce them.

## Point clouds

| Attribute | Slot | Type | Source | Triggered by |
| --- | --- | --- | --- | --- |
| `intensity` | per-vertex | float32 | LAS field | `ingest_las(include_attributes=True)` |
| `classification` | per-vertex | float32 | LAS field | `ingest_las(include_attributes=True)` |
| `color` | per-vertex | float32, `(N, 3)` | LAS RGB triple | `ingest_las(include_attributes=True)` |
| `gps_time` | per-vertex | float32 | LAS field | `ingest_las(include_attributes=True)` |
| *(every PLY property)* | per-vertex | dtype of PLY column | PLY vertex element | `ingest_ply(include_attributes=True)` |
| `knn_distance` | per-vertex | float32 | computed | `ingest_csv(knn_distance_k=k)`, `ingest_las(knn_distance_k=k)`, `ingest_ply(knn_distance_k=k)` — requires `scipy` |
| `vertex_count` | per-object | int64 | computed | any of the three point ingesters with `per_object_vertex_count=True` + `object_ids` |

`knn_distance` is the mean Euclidean distance to each point's `k`
nearest neighbours.

## Lines

| Attribute | Slot | Type | Source | Triggered by |
| --- | --- | --- | --- | --- |
| `length` | per-line | float32 | computed | `ingest_lines_csv(compute_length=True)` |
| *(CSV columns past `2*ndim`)* | per-line | float | CSV columns | `ingest_lines_csv(attribute_columns=...)` |

## Polylines and streamlines

| Attribute | Slot | Type | Source | Triggered by |
| --- | --- | --- | --- | --- |
| `length` | per-streamline | float32 | computed | any of `ingest_tck/trk/trx(compute_length=True)` |
| `start` | per-streamline | float32, `(O, D)` | computed | any of the three with `compute_endpoints=True` |
| `end` | per-streamline | float32, `(O, D)` | computed | any of the three with `compute_endpoints=True` |
| *(TRK `data_per_point` names)* | per-vertex | float32 | TRK reader | `ingest_trk` (auto, when the TRK file has them) |
| *(TRK `data_per_streamline` names)* | per-streamline | float32 | TRK reader | `ingest_trk` (auto) |
| *(TRX `dpv/<name>`)* | per-vertex | dtype of TRX field | TRX reader | `ingest_trx` (auto) |
| *(TRX `dps/<name>`)* | per-streamline | dtype of TRX field | TRX reader | `ingest_trx` (auto) |
| `mean_<scalar>` | per-streamline | float32 | computed mean of dpv | `ingest_trx(mean_scalar="name")` |

### Synthetic attributes for coloring test data (trk CLI only)

The `zvtools convert … --format trk` path can also *generate* colorable
attributes from geometry alone, for exercising attribute coloring when a
TRK file carries no native scalars. Both flags are repeatable; the
`random` generators are seeded by `--attr-seed` (default 0). All are
carried through the pyramid to every level.

| Attribute | Slot | Type | Triggered by |
| --- | --- | --- | --- |
| `length` | per-streamline | float32 | `--object-attr length` (= `--compute-length`) |
| `start` / `end` | per-streamline | float32, `(O, 3)` | `--object-attr endpoints` (= `--compute-endpoints`) |
| `orientation` | per-streamline | float32, `(O, 3)` | `--object-attr orientation` — start→end unit vector (DEC RGB) |
| `tortuosity` | per-streamline | float32 | `--object-attr tortuosity` — length ÷ endpoint distance (≥ 1) |
| `vertex_count` | per-streamline | uint32 | `--object-attr vertex_count` |
| `arc_length` | per-vertex | float32 | `--vertex-attr arc_length` — 0→1 along each streamline |
| `x` / `y` / `z` | per-vertex | float32 | `--vertex-attr x\|y\|z` — the coordinate value |
| `random` | per-vertex | float32 | `--vertex-attr random` — uniform [0, 1), seeded |
| `index` | per-vertex | float32 | `--vertex-attr index` — 0→1 within each streamline |
| `tangent` | per-vertex | float32, `(N, 3)` | `--vertex-attr tangent` — per-vertex unit direction (DEC) |

## Graphs

| Attribute | Slot | Type | Source | Triggered by |
| --- | --- | --- | --- | --- |
| `degree` | per-node | int32 | computed | `ingest_edgelist(compute_degree=True)`, `ingest_graphml(compute_degree=True)` |
| `component` | per-node | uint32 | computed (weakly-connected) | `compute_component=True` on either graph ingest |
| `clustering` | per-node | float32 | networkx clustering coefficient | `compute_clustering=True` on either graph ingest |
| *(node CSV attribute columns)* | per-node | various | edgelist node CSV | `ingest_edgelist(node_attribute_columns=...)` |
| *(edge CSV attribute columns)* | per-edge | various | edgelist edge CSV | `ingest_edgelist(edge_attribute_columns=...)` |

## Skeletons (SWC)

| Attribute | Slot | Type | Source | Triggered by |
| --- | --- | --- | --- | --- |
| `radius` | per-node | float32 | SWC column 5 | `ingest_swc` (always) |
| `compartment` | per-node | int32 | SWC column 2 (type code) | `ingest_swc` (always) |
| `topological_depth` | per-node | float32 | computed BFS depth from soma | `ingest_swc(compute_topological_depth=True)` |
| `strahler` | per-node | float32 | computed Strahler order | `ingest_swc(compute_strahler=True)` |
| `node_kind` | per-node | float32 | computed (0=soma, 1=branch, 2=continuation, 3=terminal) | `ingest_swc(compute_node_kind=True)` |

:::{note}
`ingest_swc`'s docstring advertises `uint16` / `uint8` / `uint8` for
these three. The implementation casts all of them to `float32` before
writing, which is what the table above reflects. Cast on read if you need
the integer semantics.
:::

## Meshes

| Attribute | Slot | Type | Source | Triggered by |
| --- | --- | --- | --- | --- |
| `normal` | per-vertex | float32, `(N, 3)` | OBJ `vn` lines | `ingest_obj` (auto, when `vn` lines are present) |

:::{warning}
STL per-face normals are **parsed but not stored**. Both STL parsers
return them, but `ingest_stl` assigns them to a local `face_normals` and
never passes them to `write_mesh`, so they do not reach the store.
Recompute them from the geometry with
[`compute_vertex_normals`](algorithms/mesh_attributes.md) if you need
them.
:::

Algorithms can also persist attributes via `write_back=True`:

| Attribute | Slot | Type | Source | Triggered by |
| --- | --- | --- | --- | --- |
| `vertex_normal` | per-vertex | float32, `(N, 3)` | computed | `compute_vertex_normals(write_back=True)` |
| `mean_curvature` | per-vertex | float32, `(N,)` | computed | `compute_mean_curvature(write_back=True)` |
| `component_label` | per-vertex | uint32 | computed | `compute_connected_components(write_back=True)` |

## See also

- [Ingest workflows](ingest/index.md) — full format matrix with deps.
- [Algorithms](algorithms/index.md) — which functions write back.
