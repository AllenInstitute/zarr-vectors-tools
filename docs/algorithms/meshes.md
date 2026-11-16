# Mesh algorithms

Surface area, volume, normals, curvature and spatial queries on a mesh store
([Meshes](../convert/meshes.md)). Each function splits a quad (a, b, c, d) into
the triangles (a, b, c) and (a, c, d).

| Function | What it does | Returns |
| --- | --- | --- |
| `compute_mesh_summary(store, *, level=0, per_object=False)` | Totals for the level. `volume` is signed and means something only for a closed, consistently wound mesh. | `surface_area`, `volume`, `face_count`, `vertex_count`, `edge_count`, `euler_characteristic` (V − E + F), `excluded_cross_face_edges` (always 0); `per_object` rows when asked |
| `compute_vertex_normals(store, *, level=0, weighting="area", write_back=False)` | Unit normal per vertex. `weighting="uniform"` counts faces equally. `write_back=True` stores `vertex_attributes/vertex_normal`. | `normals` (V × 3, float32) |
| `compute_mean_curvature(store, *, level=0, write_back=False)` | Unsigned mean curvature per vertex (cotangent Laplace–Beltrami). `write_back=True` stores `vertex_attributes/mean_curvature`. | `mean_curvature` (V, float32) |
| `closest_point(store, query, *, level=0, max_distance=None, max_expansion_rings=4)` | Nearest point on the surface, searching outward one ring of chunks at a time. | `found`, `position`, `distance` (`inf` if not found), `chunk_key`, `face_index`, `corners` |
| `cast_ray(store, origin, direction, *, level=0, max_distance=None)` | First surface hit along the ray. A zero `direction` raises `ValueError`. | `hit`, `t` (distance along the ray; `inf` on a miss), `position`, `chunk_key`, `face_index`, `corners` |

Per-vertex rows are in store order, as `read_mesh` returns vertices. In the
query results, `corners` names the face found, each corner as `(chunk, index
in chunk)`. `face_index` is `None` when the face's corners lie in more than one
chunk.

The example is a sphere of radius 10 centred on (20, 20, 20), with 642 vertices
and 1,280 triangles, ingested with `--chunk-shape 20,20,20`:

```python
import numpy as np
from zarr_vectors_tools.algorithms import closest_point, compute_mean_curvature, compute_mesh_summary

s = compute_mesh_summary("sphere.zv")
s["surface_area"], s["volume"], s["euler_characteristic"]   # 1250.65, 4152.74, 2
np.median(compute_mean_curvature("sphere.zv")["mean_curvature"])   # 0.0998: 1 / radius
closest_point("sphere.zv", np.array([45.0, 20.0, 20.0]))["position"]   # [30. 20. 20.]
```
