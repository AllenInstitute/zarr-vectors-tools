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

At format 0.9.0, connectivity is **one** family per level: every record
lives under `links/<delta>/<offsets>/`, and an intra-chunk record is
simply one whose offsets are all zero. There is no separate
`cross_chunk_links/` group to read — a record that spans chunks is
distinguished by its offsets, not by its location.

:::{warning}
**The double-count trap.** Before the links merge, `read_links` meant
*intra-chunk only* and had a sibling `read_cross_chunk_links`, so the
idiom was "loop `read_chunk_links` over every chunk, then add
`read_cross_chunk_links`". `read_links` now returns the **whole family**,
so that idiom unions every intra-chunk edge twice — silently doubling
every degree, with no error and no exception. Code ported from the
pre-0.9 shape must do exactly one of:

- call `read_links(level_group, delta=0)` alone and **drop** the
  per-chunk loop; or
- keep the per-chunk `read_chunk_links` loop and add
  `read_cross_links(level_group, delta=0)` — never bare `read_links`.
:::

The supported filters live in `zarr_vectors_tools.algorithms._links`:

| Helper | Returns | Use for |
| --- | --- | --- |
| `read_cross_links` | Every record under `links/<delta>` spanning ≥ 2 chunks | Pairing with a per-chunk `read_chunk_links` loop |
| `link_prefetch_plan` | `(array_path, [chunk_key])` entries, one per offsets segment | Batching reads before a whole-family decode |
| `list_link_cells` | Chunk tuples of every populated cross-chunk cell | Bucketing cells to target chunks without decoding records (`delta=0` only) |

`read_cross_links` is the **sole** definition of the cross filter — that
is why the module does not re-export a bare `read_links`. It filters on
decoded chunk identity rather than on the offsets segment name, because
`store="duplicate"` and `perm_idx` make the segment-to-record mapping
non-obvious, and `read_links` is the only public reader that reverses
`perm_idx` back to input order.

:::{note}
`links/<delta>` is a **group**, not an array — its children are one array
per offsets segment. Naming the group where an array is expected fails
*silently*: `list_chunks` returns `[]` for it and the batch reader skips
any prefetch entry that is not an array, so every read falls back to a
serial GET with no warning. That is what `link_prefetch_plan` exists for:
it enumerates the segments so the plan names arrays and prefetch actually
happens.
:::

How each algorithm resolves this:

- **Graph search, components, clustering**: one whole-family
  `read_links` and no per-chunk loop. Weights come from
  `read_link_attributes(level_group, weight, delta=0)`, which enumerates
  in the same `(segment, cell)` order as `read_links` — that shared order
  is the only thing aligning row *i* to record *i*, so a length mismatch
  is treated as a partial or stale write and the weights are dropped back
  to unit.
- **Mesh summary**: per-chunk `read_chunk_links` for the area / volume /
  edge accumulation, then `read_cross_links` for the boundary records.
  Cross-chunk **face** records (arity ≥ 3) contribute their consecutive
  endpoint pairs to the Euler-characteristic dedup set, but their
  per-face area / volume is excluded; `excluded_cross_face_edges`
  quantifies the gap.
- **Mesh attributes** (normals, curvature): per-chunk faces drive the
  computation, and `read_cross_links` identifies boundary vertices, which
  are counted in `incomplete_boundary_vertices`. Their values are
  computed from intra-chunk faces only.
- **Mesh queries** (`closest_point`, `cast_ray`): per-chunk
  `read_chunk_links` only — cross-chunk faces are not tested. For typical
  meshes this is a negligible minority.

## See also

- [Multiresolution](../multiresolution/index.md) — building the levels these algorithms read.
- [Cross-level links](../multiresolution/cross_level_links.md) — the `delta != 0` families in the same layout.
- [`ZVWriter.add_node_attribute_sync`](https://zarr-vectors.readthedocs.io/en/latest/api/lazy.html)
  — the write-back surface used by the four functions above.
