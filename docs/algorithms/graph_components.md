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
  `attributes/component_label/` via `zarr_vectors_tools._attributes.write_vertex_attribute`
  so later reads / other algorithms / downstream tools can use it
  without recomputing.

## Algorithm notes

The level's links are read once, as arrays: `read_link_arrays(level_group,
delta=0)` gives every record's endpoint chunks and chunk-local indices, and
`link_endpoints_to_rows` turns them into global vertex numbers. At format
0.9.0 connectivity is one family under `links/0/<offsets>/`, and an
intra-chunk edge is just one whose endpoints share a chunk, so that single
read covers intra- and cross-chunk edges alike. Reading the cross-chunk
records a second time would double them; see the double-count warning on the
[algorithms index](index.md).

The components come from a **union-find over the whole edge list at once**.
Each round points the larger root of every edge at the smaller one, then
shortcuts every vertex to its root, and drops the edges whose ends now share
one. Roots only decrease, so it stops when a round changes nothing. Every
step is a numpy pass, with no Python call per edge or per vertex.

- **Labels.** Components are numbered in order of their lowest vertex, so
  vertex 0 is in component 0 and the numbering does not depend on the order
  the edges are stored in.
- **Memory.** Holds the edge list (two `int64` per edge) and one `int64` per
  vertex, on top of the link reader's own arrays.
- **Time.** Reading the links dominates. On a 300,000-vertex, 306,000-edge
  store on a local Windows disk, the read took 197 s and the components
  0.28 s. The edge-at-a-time implementation this replaced spent about the
  same on the read. The read cost is mostly opening one file per chunk and
  link array; a sharded store packs those into far fewer files.

## See also

- [Algorithms index](index.md)
- [Graph search](graph_search.md) — reads the same edge arrays.
- Parent: [Links](https://zarr-vectors.readthedocs.io/en/latest/spec/object_model/links.html)
