# Graphs and line segments: GraphML, edge lists, lines

```bash
zvtools convert network.graphml network.zv --chunk-shape 250,250,250
zvtools convert edges.csv connectome.zv --format edgelist --nodes nodes.csv --chunk-shape 250,250,250
zvtools convert segments.csv segments.zv --format lines --chunk-shape 250,250,250 --compute-length
```

A graph store holds nodes at positions and edges between them; the whole graph
is one object. A lines store holds independent two-point segments, one object
each. `edgelist` and `lines` have no extension, so they must be named with
`--format` ([Converting files](index.md)). Choose the chunk shape, in the file's
units, from how densely nodes or segments fill space, not from the size of one
node: see [Store layout](../store_layout.md#chunk-shape).

## GraphML

Needs `pip install "zarr-vectors-tools[graph]"` (networkx).

- Positions are the node attributes `x`, `y`, `z`. A node missing one gets 0 on
  that axis, so a GraphML with no layout collapses to the origin.
- Every other node attribute becomes a per-vertex attribute and every edge
  attribute a per-edge attribute, both float32. The names are read from the
  first node and the first edge; text attributes are skipped.
- Other position names: Python `ingest_graphml(..., position_attrs=("cx", "cy", "cz"))`.

## Edge list

Two CSVs: the edge file is INPUT, the node file is `--nodes`.

| File | Required columns | Other columns |
|---|---|---|
| edges (INPUT) | `source`, `target` | each becomes a per-edge attribute (float32) |
| nodes (`--nodes`) | `node_id`, `x`, `y`, `z` | each becomes a per-node attribute (float32) |

Edges find their nodes by `node_id` value, not by row. Text columns are skipped.
Check that every `source` and `target` is in the node file: an edge naming a
missing node is dropped without a count. From the CLI, edge lists need no extra.

Other column names, column choice, filters and graph metrics are Python-only:

```python
from zarr_vectors_tools.convert.ingest.edgelist import ingest_edgelist

summary = ingest_edgelist(
    "edges.csv", "nodes.csv", "metrics.zv", (250.0, 250.0, 250.0),
    source_col="source", target_col="target", node_id_col="node_id",
    position_columns=("x", "y", "z"),
    drop_duplicates=True,     # repeated (source, target) pairs and node ids
    compute_degree=True,      # per-node attributes; these need [graph]
    compute_component=True,
    compute_clustering=True,
)
print(summary["edge_count"], summary["dropped_duplicate_edges"])
```

`ingest_graphml` takes the same `compute_*` options; `component` is the
connected-component label (weakly connected for a directed GraphML).
`compute_summary=True` records node, edge and component counts in the store
header. `ingest_edgelist` also takes `edge_attribute_columns`,
`node_attribute_columns` and `drop_na`.

## Line segments

One segment per row: the first six columns are the endpoints
`x0,y0,z0,x1,y1,z1`, and every further column becomes a per-segment attribute
(float32) under its header name. A header row is required.

`--compute-length` adds `length`, the distance between the endpoints. In Python,
`ingest_lines_csv` also takes `ndim=2` (four endpoint columns), `delimiter`,
`has_header`, `attribute_columns`, `drop_na`, `drop_duplicates` and
`drop_zero_length`.

For chains of connected points, use a streamline or skeleton format:
[Streamlines](streamlines.md), [Skeletons](skeletons.md).

## Pyramids and export

Graphs and lines coarsen by binning, like points; see [Pyramids](../pyramids.md).

No GraphML, edge-list or lines writer exists. A graph store exports as a
Neuroglancer precomputed skeleton layer ([Skeletons](skeletons.md)); `--unit`
names the coordinate unit when the store records none:

```bash
zvtools convert connectome.zv connectome_skeletons --format precomputed --unit micrometer
```
