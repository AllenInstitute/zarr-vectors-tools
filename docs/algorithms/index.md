# Algorithms

Eleven streaming algorithms across two domains: graphs (and skeletons,
which are graphs with a tree convention) and triangle meshes. They all
read directly from a chunked Zarr Vectors store — no full
materialisation — and a subset can write their results back to the
same store via `ZVWriter.add_node_attribute_sync`.

## Matrix

| Function | Domain | Returns | `write_back` | `per_object` | Constraints |
| --- | --- | --- | --- | --- | --- |
| `bfs_distances` | graph | distances, predecessors | – | – | – |
| `shortest_path` | graph | path, cost, visited | – | – | edge weights via attribute name |
| `compute_connected_components` | graph | labels, sizes | ✓ → `attributes/component_label/` | – | undirected |
| `compute_k_core` | graph | coreness, sizes | – | – | – |
| `compute_label_propagation` | graph | labels, sizes | – | – | – |
| `compute_louvain` | graph | labels, modularity | – | – | edge weights via attribute name |
| `compute_mesh_summary` | mesh | area, volume, Euler χ | – | ✓ → per-object dicts | triangle meshes (`link_width=3`) |
| `compute_vertex_normals` | mesh | normals | ✓ → `attributes/vertex_normal/` | – | triangle meshes |
| `compute_mean_curvature` | mesh | curvature | ✓ → `attributes/mean_curvature/` | – | triangle meshes |
| `closest_point` | mesh | hit position, chunk, face | – | – | triangle meshes |
| `cast_ray` | mesh | hit t, position, chunk, face | – | – | triangle meshes |

## Calling convention

All algorithms take a store path as the first argument and accept
`level=0` (the resolution level to operate on) as a keyword:

```python
from zarr_vectors_tools.algorithms import (
    compute_connected_components,
    compute_mesh_summary,
    shortest_path,
)

cc      = compute_connected_components("graph.zv", write_back=True)
summary = compute_mesh_summary("mesh.zv", per_object=True)
path    = shortest_path("graph.zv", source=0, target=42, weight="cost")
```

## Domain pages

- [Graph search](graph_search.md) — `bfs_distances`, `shortest_path`
- [Graph components](graph_components.md) — `compute_connected_components`
- [Graph clustering](graph_clustering.md) — `compute_k_core`,
  `compute_label_propagation`, `compute_louvain`
- [Mesh summary](mesh_summary.md) — `compute_mesh_summary`
- [Mesh attributes](mesh_attributes.md) — `compute_vertex_normals`,
  `compute_mean_curvature`
- [Mesh queries](mesh_query.md) — `closest_point`, `cast_ray`

## Cross-chunk handling

ZVF stores edges and faces that span chunk boundaries in a separate
`cross_chunk_links/<delta>/` array. Algorithms route cross-chunk
records through the public `read_cross_chunk_links` reader so they
participate correctly:

- Graph search, components, clustering: cross-chunk **edges** carry
  full weights via `read_cross_chunk_link_attributes` when a
  `weight=<attr>` is requested; otherwise they fall back to unit
  weights.
- Mesh summary: cross-chunk **face** records (arity ≥ 3) contribute
  their boundary edges to the Euler-characteristic dedup set but their
  per-face area / volume is excluded. The returned
  `excluded_cross_face_edges` counter quantifies the gap.
- Mesh attributes (normals, curvature): boundary vertices appearing in
  any cross-chunk record are counted in
  `incomplete_boundary_vertices`; their values are computed from
  intra-chunk faces only.
- Mesh queries (closest_point, cast_ray): cross-chunk faces are not
  tested; for typical meshes this is a negligible minority.

## See also

- [`ZVWriter.add_node_attribute_sync`](https://zarr-vectors.readthedocs.io/en/latest/api/lazy.html)
  — the write-back surface used by the four functions above.
