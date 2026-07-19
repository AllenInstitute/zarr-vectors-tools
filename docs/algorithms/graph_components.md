# Graph components

`compute_connected_components` labels every node in an undirected graph
(or skeleton) with the 0-indexed identifier of its connected component.

## Usage

```python
from zarr_vectors_tools.algorithms import compute_connected_components

result = compute_connected_components("graph.zv", write_back=True)

result["labels"]                  # (N,) uint32
result["n_components"]            # int
result["largest_component_size"]  # int
result["component_sizes"]         # (n_components,) int64, indexed by label
```

`level`
: Resolution level to operate on.

`write_back`
: When `True`, the per-vertex `labels` array is persisted under
  `attributes/component_label/` via `ZVWriter.add_node_attribute_sync`
  so later reads / other algorithms / downstream tools can use it
  without recomputing.

## Algorithm notes

**Union-find** with path compression and union-by-rank over a single
whole-family `read_links(level_group, delta=0)` pass. At format 0.9.0
connectivity is one family under `links/0/<offsets>/`, and an
intra-chunk edge is just one whose endpoints share a chunk — so that
single read replaces both the old per-chunk intra loop and the separate
cross-chunk pass. Doing both would union every intra-chunk edge twice
and silently double its degree; see the double-count warning on the
[algorithms index](index.md). Reads are batched through
`link_prefetch_plan`, which names one array per offsets segment rather
than the `links/0` group. Memory scales with the **node count**, so the
algorithm runs against stores larger than RAM. After every edge is
unioned, the disjoint-set roots are compacted into contiguous 0-indexed
labels via `np.unique(..., return_inverse=True)`.

## See also

- [Algorithms index](index.md)
- [Graph search](graph_search.md) — uses the same global-index mapping.
- Parent: [Links](https://zarr-vectors.readthedocs.io/en/latest/spec/object_model/links.html)
