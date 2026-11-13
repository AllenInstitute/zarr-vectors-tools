# Algorithms

`zarr_vectors_tools.algorithms` measures and queries a store in place. Each
function reads the chunks it needs from one [level](../concepts.md) and returns
arrays, dicts or pandas tables. Bundle summaries also have a command,
`zvtools bundles STORE`.

| Function | Geometry | Computes |
| --- | --- | --- |
| `bfs_distances` | [graph, skeleton](graphs.md) | hop count and parent from one vertex |
| `shortest_path` | [graph, skeleton](graphs.md) | Dijkstra or A* path and cost |
| `compute_connected_components` | [graph, skeleton](graphs.md) | component label per vertex |
| `compute_k_core` | [graph, skeleton](graphs.md) | coreness per vertex |
| `compute_label_propagation` | [graph, skeleton](graphs.md) | community per vertex |
| `compute_louvain` | [graph, skeleton](graphs.md) | community per vertex, modularity |
| `compute_mesh_summary` | [mesh](meshes.md) | area, volume, Euler characteristic |
| `compute_vertex_normals` | [mesh](meshes.md) | unit normal per vertex |
| `compute_mean_curvature` | [mesh](meshes.md) | mean curvature per vertex |
| `closest_point` | [mesh](meshes.md) | nearest point on the surface |
| `cast_ray` | [mesh](meshes.md) | first surface hit along a ray |
| `select_streamlines` | [streamlines](streamlines_skeletons.md) | ids of streamlines through, or ending in, a region |
| `bundle_summary` | [streamlines with groups](streamlines_skeletons.md) | per group: count, length, tortuosity, endpoint centroids |
| `read_bundle_summary` | [streamlines with groups](streamlines_skeletons.md) | the stored `bundle_summary` rows |
| `compute_skeleton_metrics` | [skeleton](streamlines_skeletons.md) | per object: cable length, counts, Strahler order, extent |
| `SegmentLink` | [stores with segment ids](streamlines_skeletons.md) | the same segment's object in each store |
| `read_hemisphere` | [cortical surface](streamlines_skeletons.md) | one hemisphere in source vertex order |
| `parcel_summary` | [cortical surface](streamlines_skeletons.md) | per parcel: vertex count, area, map statistics |
| `parcel_at` | [cortical surface](streamlines_skeletons.md) | the parcel nearest each point |

Every function takes the store path first (`SegmentLink` takes several) and
`level=0` as a keyword.
`compute_connected_components`, `compute_vertex_normals` and
`compute_mean_curvature` store their result as a vertex attribute with
`write_back=True`. `compute_skeleton_metrics` and `bundle_summary` store theirs
by default. A function writes to the level it reads, and a rerun replaces the
attribute; `bundle_summary` reads level 0 and writes to every level.

Every function follows objects across chunk boundaries: an edge or triangle
whose ends lie in different chunks counts once. Results do not depend on the
chunk shape.
