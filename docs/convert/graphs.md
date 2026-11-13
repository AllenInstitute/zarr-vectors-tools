# Graphs and line segments: GraphML, edge lists, lines

Convert GraphML files, edge lists and line-segment tables to graph and lines
stores. Chunk shape and `--overwrite` are in [Converting files](index.md).

| Format | In / out | Extra | Input |
|---|---|---|---|
| `graphml` | in | `graph` | node attributes `x`, `y`, `z` give positions (0 when missing) |
| `edgelist` | in | — | edge CSV with `source`, `target`; node CSV (`--nodes`) with `node_id`, `x`, `y`, `z` |
| `lines` | in | — | CSV with a header; first six columns `x0,y0,z0,x1,y1,z1` |

```bash
zvtools convert edges.csv connectome.zv --format edgelist --nodes nodes.csv --chunk-shape 250,250,250
```

```python
from zarr_vectors_tools.convert.ingest.edgelist import ingest_edgelist

ingest_edgelist("edges.csv", "nodes.csv", "connectome.zv", (250.0, 250.0, 250.0),
                drop_duplicates=True)
```

Edges find their nodes by `node_id`, not by row. An edge naming a missing node
is dropped, with a warning; the summary counts them as
`dropped_edges_missing_node`. Other column names, column choice and graph
metrics (`compute_degree`, `compute_component`, `compute_clustering`; these
need the `graph` extra) are Python-only.

| Option | Format | Effect |
|---|---|---|
| `--nodes FILE` | edgelist | The node CSV. |
| `--compute-length` | lines | Add `length`, the distance between the endpoints. |

## What the store holds

- **Graph:** nodes as vertices and edges as links; the whole graph is one
  object. Other node columns or attributes become vertex attributes, edge
  columns or attributes link attributes, all float32. Text is skipped.
- **Lines:** one object per two-point segment. Further columns become object
  attributes (float32).

Graphs and lines coarsen by binning, like points; see [Pyramids](../pyramids.md).
For chains of connected points, use [streamlines](streamlines.md) or
[skeletons](skeletons.md).

## Export

No GraphML, edge-list or lines writer exists. CSV export writes a graph's
nodes as points ([Points](points.md#export-to-csv-or-ply)). A graph store also
exports as a precomputed skeleton layer:

```bash
zvtools convert connectome.zv connectome_skeletons --format precomputed --unit micrometer
```
