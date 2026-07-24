# Graph search

Frontier search over a chunked graph (or skeleton) store. Both
algorithms in this module use the same in-memory adjacency map, built
once per call via the internal `build_adjacency` helper.

At format {{ zvf_version }} connectivity is a single family, so
`build_adjacency` performs **one whole-family `read_links(delta=0)`**
rather than a per-chunk pass followed by a separate cross-chunk merge.
Edge weights come from `read_link_attributes(level_group, weight_attr,
delta=0)`, aligned row-for-row with the links by shared `(segment, cell)`
enumeration order; a length mismatch falls back to unit weights.

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
result["predecessors"]   # (N,) int64 — -1 for source and unreached
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
  Cross-chunk edges read their weight from
  `cross_chunk_link_attributes/<weight>/0/` when that array exists;
  otherwise they silently fall back to unit weight.

`heuristic`
: Optional admissible lower-bound function `node_index -> float`. When
  supplied the search becomes A*.

## Algorithm notes

Both functions materialise the full adjacency once. Memory cost is
**O(N + E)** in the number of nodes plus edges; for a graph with `E`
intra-chunk edges plus `C` cross-chunk edges, every edge contributes
two entries to the adjacency list (undirected). BFS itself is the
standard double-ended queue; the Dijkstra path uses `heapq` with
stale-entry filtering on pop. Path reconstruction follows the
predecessor pointer chain from `target` back to `source`.

The shared `build_adjacency` helper is also called by every algorithm
in [Graph clustering](graph_clustering.md) — the in-memory adjacency
that LPA, Louvain, and k-core need is identical to what the search
routines build, so a single code path covers both.

## See also

- [Algorithms index](index.md) — write_back / per_object matrix.
- [Graph clustering](graph_clustering.md)
- Parent: [Graph spec](https://zarr-vectors.readthedocs.io/en/latest/spec/geometry_types/graph.html)
