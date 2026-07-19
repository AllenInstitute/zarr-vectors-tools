# Graphs

A ZVF **graph** is nodes at explicit positions plus arbitrary edges — the
natural shape for a connectome, where each node is a neuron or region at
its centroid and each edge is a measured connection. Two ingests:
`ingest_edgelist` (a pair of CSVs) and `ingest_graphml` (via networkx).
Both need the `graph` extra for the enrichment options.

## Edge-list CSV — `ingest_edgelist`

Two files. The edge CSV holds `source`/`target` plus anything else; the
node CSV holds `node_id` plus coordinates plus anything else. Node IDs
are matched by value, not by row order.

```python
from zarr_vectors_tools.ingest.edgelist import ingest_edgelist

# edges.csv:  source,target,synapse_count
# nodes.csv:  node_id,x,y,z,cell_type
summary = ingest_edgelist(
    "edges.csv",                          # first positional
    "nodes.csv",                          # second positional
    "connectome.zv",
    (10_000.0, 10_000.0, 10_000.0),       # chunk_shape in nm
    source_col="source", target_col="target", node_id_col="node_id",
    position_columns=("x", "y", "z"),
    edge_attribute_columns=["synapse_count"],
    node_attribute_columns=["cell_type"],
    drop_na=True,          # drops NaN source/target edges and NaN-position nodes
    drop_duplicates=True,  # dedupes edges on (source, target), nodes on node_id
    compute_degree=True,
    compute_component=True,
    compute_clustering=True,
    compute_summary=True,
)
print(summary["dropped_duplicate_edges"], summary["dropped_na_nodes"])
```

Edges referencing a node ID absent from the node CSV are dropped
silently — there is no counter for them, so validate your node table if
the edge count comes back lower than expected.

:::{warning}
`edgelist` registers no file extension, so the CLI cannot auto-detect it
and `--nodes` is a hard requirement:

```bash
zvtools convert edges.csv connectome.zv \
    --format edgelist --nodes nodes.csv --chunk-shape 10000,10000,10000
```

Omitting `--nodes` exits with an error; omitting `--format` ingests
`edges.csv` as a point cloud.
:::

### GPU CSV reads

`use_cudf=True` reads both CSVs through RAPIDS cuDF and converts back to
pandas for the rest of the pipeline. It needs the `gpu` extra and a
working CUDA environment — install per the RAPIDS instructions, since
`cudf` does not come from PyPI in the usual way. Without it the call
raises `IngestError` rather than falling back, so leave it `False`
(the default) unless the CSVs are large enough that parsing dominates.

## GraphML — `ingest_graphml`

```python
from zarr_vectors_tools.ingest.graphml import ingest_graphml

ingest_graphml(
    "network.graphml",
    "network.zv",
    (1.0, 1.0, 1.0),
    position_attrs=("x", "y", "z"),   # node attribute names holding coordinates
    compute_degree=True,
    compute_clustering=True,
)
```

Requires `networkx`. Positions must already be node attributes named by
`position_attrs`; any node missing one of them gets `0.0` on that axis
rather than an error, so a GraphML with no layout produces a graph
collapsed at the origin.

Remaining node attributes and all edge attributes are carried across
automatically, typed from the *first* node and the *first* edge — a
sample, not a scan. Attributes that will not cast to float are skipped.

## Shared enrichment options

| Option | Writes | Notes |
| --- | --- | --- |
| `compute_degree` | `degree` per node | In + out for directed graphs. |
| `compute_component` | `component` per node | 0-indexed label; **weakly** connected for directed graphs. |
| `compute_clustering` | `clustering` per node | networkx clustering coefficient; directed graphs are converted to undirected first. |
| `compute_summary` | a `GraphHeader` | Node/edge counts, mean degree, component count, largest component size. |

`compute_summary` writes to `/headers/graph/` rather than to a per-graph
attribute slot — a workaround until the core API grows one. It is
best-effort, like all header writes, so a failure there does not fail the
ingest.

:::{note}
`ingest_edgelist` builds a fresh `networkx.Graph` purely to compute these
— it is always undirected there, so `component` is plain connected
components. `ingest_graphml` uses the graph as loaded and honours
directedness.
:::

## See also

- [Enrichments → graphs](../enrichments.md#graphs)
- [Algorithms → graph components](../algorithms/graph_components.md)
- [Algorithms → graph clustering](../algorithms/graph_clustering.md)
- [Skeletons](skeletons.md) — the tree-shaped special case
- [Ingest workflows](index.md)
