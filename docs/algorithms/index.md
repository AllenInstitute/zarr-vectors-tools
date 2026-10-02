# Algorithms

`zarr_vectors_tools.algorithms` measures and queries a store where it is. Each
function reads the chunks it needs from one [level](../concepts.md) and
returns arrays, dicts or pandas tables. Some can also write their result into
the store as an attribute.

```python
from zarr_vectors_tools.algorithms import compute_skeleton_metrics

metrics = compute_skeleton_metrics("neurons.zv")   # one row per neuron; also stored
```

Everything here is Python, except bundle summaries, which also have a
command: `zvtools bundles STORE`.

| Function | Geometry | Computes | Writes to the store | Page |
| --- | --- | --- | --- | --- |
| `bfs_distances` | graph, skeleton | hop count and parent from one vertex | no | [Graphs](graphs.md) |
| `shortest_path` | graph, skeleton | Dijkstra or A* path and cost | no | [Graphs](graphs.md) |
| `compute_connected_components` | graph, skeleton | component label per vertex | `write_back=False`; if true: `vertex_attributes/component_label` | [Graphs](graphs.md) |
| `compute_k_core` | graph, skeleton | coreness per vertex | no | [Graphs](graphs.md) |
| `compute_label_propagation` | graph, skeleton | community per vertex | no | [Graphs](graphs.md) |
| `compute_louvain` | graph, skeleton | community per vertex, modularity | no | [Graphs](graphs.md) |
| `compute_mesh_summary` | mesh | area, volume, Euler characteristic; per object on request | no | [Meshes](meshes.md) |
| `compute_vertex_normals` | mesh | unit normal per vertex | `write_back=False`; if true: `vertex_attributes/vertex_normal` | [Meshes](meshes.md) |
| `compute_mean_curvature` | mesh | mean curvature per vertex | `write_back=False`; if true: `vertex_attributes/mean_curvature` | [Meshes](meshes.md) |
| `closest_point` | mesh | nearest point on the surface | no | [Meshes](meshes.md) |
| `cast_ray` | mesh | first surface hit along a ray | no | [Meshes](meshes.md) |
| `select_streamlines` | streamlines | ids of streamlines through, or ending in, a mask or box | no | [Streamlines, skeletons](streamlines_skeletons.md) |
| `bundle_summary` | streamlines with groups | per group: count, length statistics, tortuosity, endpoint centroids | `write=True`: `group_attributes/bundle_*` | [Streamlines, skeletons](streamlines_skeletons.md) |
| `read_bundle_summary` | streamlines with groups | the rows `bundle_summary` stored | no | [Streamlines, skeletons](streamlines_skeletons.md) |
| `compute_skeleton_metrics` | skeleton | per object: cable length, node, leaf, branch and component counts, Strahler order, extent | `write=True`: `object_attributes/<metric>` | [Streamlines, skeletons](streamlines_skeletons.md) |
| `SegmentLink` | stores with segment ids | the same segment's object in each store | no | [Streamlines, skeletons](streamlines_skeletons.md) |
| `read_hemisphere` | cortical surface | one hemisphere in the source file's vertex order | no | [Streamlines, skeletons](streamlines_skeletons.md) |
| `parcel_summary` | cortical surface | per parcel: vertex count, area, map mean and standard deviation | no | [Streamlines, skeletons](streamlines_skeletons.md) |
| `parcel_at` | cortical surface | the parcel nearest each point | no | [Streamlines, skeletons](streamlines_skeletons.md) |

Every function takes the store path first (`SegmentLink` takes several) and
`level=` as a keyword argument, defaulting to `0`. The exception is
`bundle_summary`: it defaults to `level=None`, which computes from level 0 and
writes the same rows to every level. A writing function writes to the level it
was given: `0/vertex_attributes/component_label` for level 0. Re-running it
replaces the attribute.

## Objects that span chunks

Every function follows objects across chunk boundaries: an edge or a triangle
whose ends lie in different chunks counts once, like any other. Results do not
depend on the chunk shape.
