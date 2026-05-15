# Graph clustering

Three algorithms answering "what are the modules / dense regions in
this network?":

- `compute_k_core` — Batagelj-Zaversnik degree-peeling.
- `compute_label_propagation` — synchronous LPA (Raghavan-Albert-Kumara 2007).
- `compute_louvain` — greedy modularity optimisation (Blondel et al. 2008).

All three materialise the in-memory adjacency once via
`graph_search.build_adjacency`. LPA and Louvain need that because they
touch every edge per iteration; k-core could stream but uniformity
wins.

## `compute_k_core`

```python
from zarr_vectors_tools.algorithms import compute_k_core

result = compute_k_core("graph.zv")

result["coreness"]      # (N,) uint32 — per-vertex k-core membership
result["max_core"]      # int
result["core_sizes"]    # (max_core+1,) int64
```

The k-coreness of vertex *v* is the largest *k* such that *v* belongs
to a subgraph where every vertex has degree ≥ *k*.

### Algorithm notes

A min-heap keyed by `(current_degree, vertex)` peels lowest-degree
vertices one at a time. Stale entries (whose degree was decremented
since the push) are filtered on pop. Complexity is **O((N + E) log N)**;
memory is the adjacency list plus one heap entry per vertex.

## `compute_label_propagation`

```python
from zarr_vectors_tools.algorithms import compute_label_propagation

result = compute_label_propagation("graph.zv", max_iter=20, seed=0)

result["labels"]           # (N,) uint32 — community label per vertex
result["n_communities"]    # int
result["iterations"]       # int
result["converged"]        # bool
result["community_sizes"]  # (n_communities,) int64
```

`max_iter`
: Maximum number of synchronous rounds. Convergence is typically 5–20.

`seed`
: RNG seed used to break ties when multiple neighbour labels are
  equally popular.

### Algorithm notes

Each vertex starts in its own community. In each round, every vertex
adopts the most frequent label among its neighbours, breaking ties via
a seeded RNG. Iterates until labels stabilise or `max_iter` rounds have
passed. Time per round is **O(N + E)**; total is bounded by `max_iter`
rounds.

## `compute_louvain`

```python
from zarr_vectors_tools.algorithms import compute_louvain

result = compute_louvain("graph.zv", weight="cost", max_iter=10)

result["labels"]           # (N,) uint32 — level-0 community per vertex
result["modularity"]       # float — final modularity Q
result["n_communities"]    # int
result["iterations"]       # int — outer rounds executed
result["community_sizes"]  # (n_communities,) int64
```

`weight`
: Edge-attribute name. `None` (default) means unit weights. Cross-chunk
  edges use weights from `cross_chunk_link_attributes/<weight>/0/`
  when that array exists; otherwise unit weight (silent fallback).

`max_iter`
: Maximum number of outer Phase-1 + Phase-2 rounds.

`seed`
: RNG seed for tie-breaking in the local-move order.

### Algorithm notes

The classic two-phase Blondel et al. loop: a local-move phase
maximises modularity gain by moving vertices between communities, then
each community is contracted into a super-node and the loop recurses.
Stops when a full Phase-2 round yields modularity gain < `1e-6` or
`max_iter` outer rounds have elapsed. The reported `labels` are the
level-0 dendrogram collapsed back to the original vertices via
successive `compact` remaps.

## See also

- [Algorithms index](index.md)
- [Graph search](graph_search.md) — shares `build_adjacency`.
- [Graph components](graph_components.md) — coarser partition: every
  community is a subset of one connected component.
