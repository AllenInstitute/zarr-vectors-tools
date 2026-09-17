# Graph clustering

Three algorithms answering "what are the modules / dense regions in
this network?":

- `compute_k_core` — degree peeling.
- `compute_label_propagation` — synchronous LPA (Raghavan-Albert-Kumara 2007).
- `compute_louvain` — greedy modularity optimisation (Blondel et al. 2008).

All three read the level's links once, as arrays, into an in-memory
adjacency (compressed sparse rows). LPA and Louvain need all of it because
they touch every edge per iteration; k-core could stream, but uniformity
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

Peeling in batches: at the current *k*, every live vertex of degree at
most *k* is removed in one numpy step, and its live neighbours' degrees
drop. Those that fall to *k* or below form the next batch. When none do,
*k* rises to the smallest live degree. Coreness is unique, so removing a
batch at once gives the same answer as removing its vertices one at a time.
Each batch costs its vertices' edges, so the total is **O(N + E)** plus one
numpy step per batch. Memory is the adjacency plus a few per-vertex arrays.

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

### LPA algorithm notes

Each vertex starts in its own community. In each round, every vertex
adopts the most frequent label among its neighbours, breaking ties at random
among the tied labels with a seeded RNG. A vertex with no neighbours keeps
its label. Iterates until labels stabilise or `max_iter` rounds have passed.

A round is one sort of the edge ends by `(vertex, neighbour label)`: count
each run, then take each vertex's largest count, with a random draw
breaking ties. Time per round is **O(E log E)** in numpy. Labels are
numbered in order of each community's lowest vertex.

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
: Edge-attribute name. `None` (default) means unit weights. Every edge,
  intra- or cross-chunk, reads its weight from `link_attributes/<weight>/0/`;
  without that family, unit weight (silent fallback).

`max_iter`
: Maximum number of outer Phase-1 + Phase-2 rounds.

`seed`
: RNG seed for tie-breaking in the local-move order.

### Louvain algorithm notes

The classic two-phase Blondel et al. loop: a local-move phase
maximises modularity gain by moving vertices between communities, then
each community is contracted into a super-node and the loop recurses.
Stops when a full Phase-2 round yields modularity gain < `1e-6` or
`max_iter` outer rounds have elapsed. The reported `labels` are the
level-0 dendrogram collapsed back to the original vertices via
successive `compact` remaps.

The local-move phase is sequential by definition, since each move changes
the gains of the moves after it. It still visits one vertex at a time, over
plain lists taken from the adjacency arrays. The modularity and the
contraction into super-nodes are array operations: a `bincount` over
community pairs.

## See also

- [Algorithms index](index.md)
- [Graph search](graph_search.md) — reads the same adjacency.
- [Graph components](graph_components.md) — coarser partition: every
  community is a subset of one connected component.
