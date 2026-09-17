# Graph search

Frontier search over a chunked graph (or skeleton) store. Both
algorithms read the level's links once per call into an array adjacency
(compressed sparse rows: `indptr`, `indices`, `weights`).

At format {{ zv_version }} connectivity is a single family, so the adjacency
comes from **one whole-family `read_link_arrays(delta=0)`** rather than a
per-chunk pass followed by a separate cross-chunk merge. Edge weights come
from `read_link_attributes(level_group, weight_attr, delta=0)`, aligned
row-for-row with the links by their shared `(segment, cell)` enumeration
order; a length mismatch falls back to unit weights.

:::{warning}
Older code and notes describe merging cross-chunk edges in from
`read_cross_chunk_links`. That reader no longer exists, and reproducing
the pattern against the merged family double-counts every intra-chunk
edge. See the double-count trap in [Algorithms](index.md).
:::

## `bfs_distances` — unweighted BFS

```python
from zarr_vectors_tools.algorithms import bfs_distances

result = bfs_distances("graph.zv", source=0, max_distance=5)
result["distances"]      # (N,) int32 — -1 for unreached
result["predecessors"]   # (N,) int64 — lowest-numbered parent; -1 for source and unreached
```

`source`
: Global vertex index to start from.

`max_distance`
: Optional cutoff. Vertices beyond stay at `-1`. `None` runs until the
  connected component is exhausted.

## `shortest_path` — Dijkstra / A\*

```python
from zarr_vectors_tools.algorithms import shortest_path

# Plain Dijkstra
res = shortest_path("graph.zv", source=0, target=42, weight="cost")
res["path"]      # list[int]
res["cost"]      # float, inf if unreachable
res["visited"]   # int — nodes popped from the queue

# A* with an admissible heuristic
def euclid_to_target(v):
    ...
res = shortest_path("graph.zv", source=0, target=42, heuristic=euclid_to_target)
```

`source`, `target`
: Global vertex indices.

`weight`
: Name of an edge attribute to use as the per-edge cost. `None`
  (default) means unit weights, which makes Dijkstra equivalent to BFS.
  Every edge, intra- or cross-chunk, reads its weight from
  `link_attributes/<weight>/0/`; without that family, weights are
  silently 1.

`heuristic`
: Optional admissible lower-bound function `node_index -> float`. When
  supplied the search becomes A*.

## Algorithm notes

Both functions build the full adjacency once. Memory is **O(N + E)**: every
edge contributes two entries (undirected), a parallel edge one pair per
copy, and a self-loop two entries on its vertex.

- **BFS** expands a whole frontier per step with numpy: gather the
  frontier's neighbours, keep the unvisited ones, and give each its
  lowest-numbered parent. The distances equal a queue BFS's, and the
  predecessors do not depend on edge order.
- **Dijkstra / A\*** keeps a `heapq`, over plain lists taken from the
  adjacency arrays, and stops when the target is popped. A stale entry
  (one pushed before a shorter route was found) is skipped on pop. A node
  can be reopened, so an admissible heuristic that is not consistent still
  finds the shortest path. Path reconstruction follows the predecessor
  chain from `target` back to `source`.

`build_adjacency` still returns the Python adjacency lists for callers that
want them. The algorithms here and in
[Graph clustering](graph_clustering.md) use the array form directly.

## See also

- [Algorithms index](index.md) — write_back / per_object matrix.
- [Graph clustering](graph_clustering.md)
- Parent: [Graph spec](https://zarr-vectors.readthedocs.io/en/latest/spec/geometry_types/graph.html)
