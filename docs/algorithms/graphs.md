# Graph algorithms

Search, connected components and communities on a graph store, made from
GraphML or an edge list (see [Graphs](../convert/graphs.md)), or on a skeleton
store (see [Skeletons](../convert/skeletons.md)). The examples use
`net.graphml`: 15 vertices forming two 6-vertex cliques joined by one edge,
plus a separate 3-vertex path. Each edge has a numeric `length` attribute.

```bash
zvtools convert net.graphml net.zv --chunk-shape 50,50,50
```

Each function reads all of the level's edges, including those between
chunks, into memory. On a skeleton store, the edges are the parent links,
including those the store implies by node order instead of storing. For
per-neuron measures, such as cable length or Strahler order, use
[skeleton metrics](streamlines_skeletons.md#skeleton-metrics).

## Vertex numbers

Vertices are numbered in store order (chunk by chunk), not in the input
file's order. Row `i` of core's `read_graph(...)["positions"]` is vertex `i`,
so you can find a vertex by its position:

```python
import numpy as np
from zarr_vectors.types.graphs import read_graph

positions = read_graph("net.zv")["positions"]          # row i = vertex i
source = int(np.flatnonzero((positions == [10, 10, 10]).all(axis=1))[0])
target = int(np.flatnonzero((positions == [85, 60, 60]).all(axis=1))[0])
```

Every per-vertex result (`distances`, `labels`, `coreness`) uses this order.

## Search

```python
from zarr_vectors_tools.algorithms import bfs_distances, shortest_path

bfs = bfs_distances("net.zv", source)
bfs["distances"]      # [ 0  1  1  1  1  1 -1 -1 -1  2  3  3  3  3  3]  hops; -1 = not reached
bfs["predecessors"]   # [-1  0  0  0  0  0 -1 -1 -1  5  9  9  9  9  9]

def remaining(v):     # straight-line distance: a lower bound on the rest of the path
    return float(np.linalg.norm(positions[v] - positions[target]))

shortest_path("net.zv", source, target, weight="length", heuristic=remaining)
# {'path': [0, 5, 9, 14], 'cost': 125.0, 'visited': 12}
```

| Signature | Argument | Meaning |
| --- | --- | --- |
| `bfs_distances(store, source, *, level=0, max_distance=None)` | `max_distance` | stop after this many hops; vertices further away stay `-1` |
| `shortest_path(store, source, target, *, level=0, weight=None, heuristic=None)` | `weight` | edge attribute giving each edge's cost; `None` costs 1 per edge |
| | `heuristic` | `vertex -> float`, a lower bound on the remaining cost; turns Dijkstra into A* |

- `predecessors[v]` is the lowest-numbered neighbour of `v` one hop nearer
  the source.
- `shortest_path` returns `{'path': [], 'cost': inf}` when the target cannot
  be reached. `visited` counts the vertices taken off the queue.
- `weight` names an edge attribute. Numeric GraphML edge data and extra
  edge-list columns are stored as edge attributes under their own names. If
  the store has no attribute of that name, every edge weighs 1, without error.
  A skeleton store keeps no attribute for the links it implies, so naming an
  attribute it does store raises `ValueError`.

## Connected components

```python
from zarr_vectors_tools.algorithms import compute_connected_components

cc = compute_connected_components("net.zv", write_back=True)
cc["labels"]                   # [0 0 0 0 0 0 1 1 1 0 0 0 0 0 0]  uint32, per vertex
cc["n_components"]             # 2
cc["component_sizes"]          # [12  3]
cc["largest_component_size"]   # 12
```

Edges are treated as undirected; components are numbered in order of their
lowest vertex. `write_back=True` (default `False`) stores `labels` as
`0/vertex_attributes/component_label`, to colour by in the [viewer](../visualise.md).

## Communities

```python
from zarr_vectors_tools.algorithms import (
    compute_k_core, compute_label_propagation, compute_louvain,
)

compute_k_core("net.zv")["coreness"]     # [5 5 5 5 5 5 1 1 1 5 5 5 5 5 5]

lpa = compute_label_propagation("net.zv", max_iter=20, seed=0)
lpa["labels"], lpa["converged"]          # [0 0 0 0 0 0 1 2 1 3 3 3 3 3 3], False

louvain = compute_louvain("net.zv", seed=0)
louvain["labels"], round(louvain["modularity"], 3)   # [0 0 0 0 0 0 1 1 1 2 2 2 2 2 2], 0.525
```

| Function | Returns | Options (defaults) |
| --- | --- | --- |
| `compute_k_core` | `coreness` (uint32 per vertex), `max_core`, `core_sizes` | `level=0` |
| `compute_label_propagation` | `labels`, `n_communities`, `community_sizes`, `iterations`, `converged` | `max_iter=20`, `seed=0` |
| `compute_louvain` | `labels`, `modularity`, `n_communities`, `community_sizes`, `iterations` | `weight=None`, `max_iter=10`, `seed=0` |

- A vertex's coreness is the largest *k* for which it belongs to a subgraph
  where every vertex has at least *k* neighbours.
- Label propagation is synchronous: every vertex takes its neighbours' most
  common label at the same time, with ties broken by `seed`. On some shapes,
  such as the 3-vertex path above, labels swap back and forth and never
  settle. Check `converged`.
- Louvain (Blondel et al. 2008) maximises modularity. Its `weight` is an edge
  strength: a heavier edge holds its two ends together. This is the opposite
  of `shortest_path`, where `weight` is a cost.
- Labels are numbered in order of each community's lowest vertex. None of
  the three writes to the store.

## Your own algorithms

`build_adjacency` returns the level's adjacency lists, with each neighbour
paired with its edge weight:

```python
from zarr_vectors.building import get_resolution_level, open_store
from zarr_vectors_tools.algorithms.graph_search import build_adjacency

adj, n = build_adjacency(get_resolution_level(open_store("net.zv"), 0), weight_attr="length")
adj[5]   # [(9, 75.0), (0, 25.0), (1, 20.0), (2, 15.0), (3, 10.0), (4, 5.0)]
```
