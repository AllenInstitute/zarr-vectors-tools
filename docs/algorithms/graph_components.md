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

Per-chunk **union-find** with path compression and union-by-rank, plus
a single global pass over the `cross_chunk_links/0/` array to merge
endpoints that span chunks. Memory scales with the **node count** —
edge arrays are loaded one chunk at a time, never all at once, so the
algorithm runs against stores larger than RAM. After every edge is
unioned, the disjoint-set roots are compacted into contiguous
0-indexed labels via `np.unique(..., return_inverse=True)`.

## See also

- [Algorithms index](index.md)
- [Graph search](graph_search.md) — uses the same global-index mapping.
- Parent: [Cross-chunk links](https://zarr-vectors.readthedocs.io/en/latest/spec/object_model/cross_chunk_links.html)
