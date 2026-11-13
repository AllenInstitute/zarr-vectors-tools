# Graph algorithms

Search, connected components and communities on a graph store
([Graphs](../convert/graphs.md)) or a skeleton store
([Skeletons](../convert/skeletons.md)). Each function reads all of the level's
edges into memory. On a skeleton store, the edges are the parent links.

| Function | What it does | Returns |
| --- | --- | --- |
| `bfs_distances(store, source, *, level=0, max_distance=None)` | Hop count from `source`; stops after `max_distance` hops. | `distances` (-1: not reached), `predecessors` |
| `shortest_path(store, source, target, *, level=0, weight=None, heuristic=None)` | Dijkstra; A* when `heuristic(vertex)` gives a lower bound on the remaining cost. `weight` names the edge attribute holding each edge's cost (default 1). | `path` (`[]` if unreachable), `cost` (`inf` if unreachable), `visited` |
| `compute_connected_components(store, *, level=0, write_back=False)` | Component label per vertex, edges undirected. `write_back=True` stores `vertex_attributes/component_label`. | `labels`, `n_components`, `component_sizes`, `largest_component_size` |
| `compute_k_core(store, *, level=0)` | Coreness: the largest *k* for which the vertex is in a subgraph where every vertex has at least *k* neighbours. | `coreness`, `max_core`, `core_sizes` |
| `compute_label_propagation(store, *, level=0, max_iter=20, seed=0)` | Communities by synchronous label propagation. Labels can oscillate; check `converged`. | `labels`, `n_communities`, `community_sizes`, `iterations`, `converged` |
| `compute_louvain(store, *, level=0, weight=None, max_iter=10, seed=0)` | Communities that maximise modularity. Here `weight` is an edge strength, not a cost. | `labels`, `modularity`, `n_communities`, `community_sizes`, `iterations` |

Vertices are numbered in store order (chunk by chunk), not in the input file's
order. Row `i` of `read_graph(store)["positions"]` is vertex `i`. Labels are
numbered in order of each group's lowest vertex. If the store has no edge
attribute named by `weight`, every edge weighs 1.

The example store holds two 6-vertex cliques joined by one edge, and a separate
3-vertex path. Each edge has a `length` attribute.

```python
from zarr_vectors.types.graphs import read_graph
from zarr_vectors_tools.algorithms import compute_connected_components, shortest_path

positions = read_graph("net.zv")["positions"]   # row i = vertex i
shortest_path("net.zv", 0, 14, weight="length")
# {'path': [0, 5, 6, 14], 'cost': 95.0, 'visited': 12}
compute_connected_components("net.zv")["labels"]
# [0 0 0 0 0 0 0 0 0 0 1 1 1 0 0]
```

For your own algorithms, `build_adjacency(level_group, *, weight_attr=None)` in
`zarr_vectors_tools.algorithms.graph_search` returns each vertex's neighbours,
paired with their edge weights, and the vertex count. Get `level_group` from
`zarr_vectors.building.get_resolution_level(open_store(path), 0)`. For per-neuron measures
such as cable length, see [skeleton metrics](streamlines_skeletons.md#skeleton-metrics).
