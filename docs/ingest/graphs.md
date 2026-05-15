# Graphs

Two ingest functions cover the common graph file formats: a paired-CSV
edge-list, and GraphML. Both write to the **graph** ZVF geometry and
support the same set of optional node-level enrichments.

## Edge-list CSV — `ingest_edgelist`

Two CSV files: one for edges (`source`, `target`, optional weight /
attributes) and one for node positions (`node_id`, `x`, `y`, `z`,
optional node attributes).

```python
from zarr_vectors_tools.ingest import ingest_edgelist

ingest_edgelist(
    edges_path="edges.csv",
    nodes_path="nodes.csv",
    output_path="graph.zv",
    chunk_shape=(100.0, 100.0, 100.0),
    compute_degree=True,
    compute_component=True,
)
```

**Notable options**

`source_col`, `target_col`, `node_id_col`
: Column names. Defaults: `"source"`, `"target"`, `"node_id"`.

`position_columns`
: Tuple of node-CSV column names for coordinates. Default `("x", "y", "z")`.

`edge_attribute_columns`, `node_attribute_columns`
: Filter which non-id, non-position columns to keep. Default: all.

`use_cudf`
: Read both CSVs on GPU via RAPIDS cuDF. Requires the `gpu` extra and a
  working CUDA setup. Falls back to pandas otherwise (pandas is a core
  dependency).

`compute_degree`, `compute_component`, `compute_clustering`
: Auto-derive node attributes. See the shared list below.

## GraphML — `ingest_graphml`

```python
from zarr_vectors_tools.ingest import ingest_graphml

ingest_graphml(
    input_path="network.graphml",
    output_path="network.zv",
    chunk_shape=(1.0, 1.0, 1.0),
    position_attrs=("x", "y", "z"),
    compute_clustering=True,
)
```

Requires `networkx`. Node positions must be stored as node attributes
named by `position_attrs`.

## Shared enrichment options (both ingests)

`compute_degree`
: Writes `node_attributes["degree"]` (in + out for directed graphs).

`compute_component`
: Writes 0-indexed connected-component label to
  `node_attributes["component"]`. Uses **weakly** connected components
  for directed graphs.

`compute_clustering`
: Writes the per-node clustering coefficient to
  `node_attributes["clustering"]`. Requires `networkx`.

`compute_summary`
: Writes a [`GraphHeader`](../headers.md) with node / edge counts, mean
  degree, and connected-component summary stats. (Workaround until a
  per-graph attribute slot lands in the core spec.)

## See also

- [Algorithms → graph clustering and components](../algorithms/index.md)
- [Enrichments](../enrichments.md#graphs)
- Parent: [Graph spec](https://zarr-vectors.readthedocs.io/en/latest/spec/geometry_types/graph.html)
