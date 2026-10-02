# Mesh algorithms

Surface area, volume, normals, curvature and spatial queries on a mesh store,
made from OBJ, STL or precomputed meshes (see [Meshes](../convert/meshes.md)).
The examples use `sphere.obj`, a sphere of radius 10 centred on (20, 20, 20),
with 642 vertices and 1,280 triangles:

```bash
zvtools convert sphere.obj sphere.zv --chunk-shape 20,20,20
```

At this chunk shape the sphere is cut through its centre on all three axes, so
270 of its 1,280 triangles have corners in more than one chunk. Every function
includes those triangles, so the results below are the same for any chunk
shape.

Faces can be triangles or quads (an OBJ whose faces are all quads is stored as
quads). Each function splits a quad (a, b, c, d) into the triangles (a, b, c)
and (a, c, d).

## Summary

```python
from zarr_vectors_tools.algorithms import compute_mesh_summary

summary = compute_mesh_summary("sphere.zv", per_object=True)
```

| Key | Example | Meaning |
| --- | --- | --- |
| `surface_area` | `1250.65` | sum of triangle areas, in square store units |
| `volume` | `4152.74` | signed volume; meaningful only for a closed, consistently wound mesh |
| `face_count` | `1280` | faces; a quad counts once |
| `vertex_count` | `642` | vertices at the level |
| `edge_count` | `1920` | distinct face edges; a quad has 4 |
| `euler_characteristic` | `2` | V − E + F; 2 for a closed surface without holes |
| `excluded_cross_face_edges` | `0` | always 0; kept for compatibility |
| `per_object` | `[{'object_id': 0, 'surface_area': 1250.65, ...}]` | `object_id`, `surface_area`, `volume`, `face_count` and `vertex_count` per object; only with `per_object=True` |

## Normals and curvature

```python
import numpy as np
from zarr_vectors_tools.algorithms import compute_mean_curvature, compute_vertex_normals

normals = compute_vertex_normals("sphere.zv", write_back=True)
normals["normals"].shape                    # (642, 3), float32 unit vectors

curvature = compute_mean_curvature("sphere.zv", write_back=True)
np.median(curvature["mean_curvature"])      # 0.09976691: 1 / radius
```

| Function | Result key | Written as (with `write_back=True`) | Option |
| --- | --- | --- | --- |
| `compute_vertex_normals` | `normals`, `(V, 3)` float32 | `vertex_attributes/vertex_normal`, 3 columns | `weighting="area"` (larger faces count more) or `"uniform"` |
| `compute_mean_curvature` | `mean_curvature`, `(V,)` float32 | `vertex_attributes/mean_curvature` | none |

`write_back` defaults to `False`. Rows are in store order, the order in which
core's `read_mesh` returns vertices. Mean curvature uses the cotangent
Laplace–Beltrami operator (Meyer et al. 2003). It is unsigned: a sphere of
radius *r* gives 1/*r* everywhere. Both functions also return
`incomplete_boundary_vertices`, always 0, kept for compatibility.

:::{warning}
`vertex_normal` has three columns. The viewer currently stops drawing chunks
that carry a multi-column vertex attribute (see [Visualise](../visualise.md)),
so keep `write_back=False` for normals on a store you plan to view.
:::

## Spatial queries

```python
import numpy as np
from zarr_vectors_tools.algorithms import cast_ray, closest_point

near = closest_point("sphere.zv", np.array([45.0, 20.0, 20.0]), max_distance=50.0)
near["found"], near["position"], near["distance"]   # True, [30. 20. 20.], 15.0
near["face_index"], near["corners"]   # None, (((1, 1, 1), 8), ((1, 0, 0), 63), ((1, 1, 0), 65))

hit = cast_ray("sphere.zv", origin=np.array([0.0, 21.0, 22.0]),
               direction=np.array([1.0, 0.0, 0.0]), max_distance=100.0)
hit["hit"], hit["t"], hit["position"]   # True, 10.277057420924843, [10.27705742 21. 22.]
```

| Function | Options (defaults) | Returns |
| --- | --- | --- |
| `closest_point(store, query, ...)` | `max_distance=None`, `max_expansion_rings=4`, `level=0` | `found`, `position`, `distance`, `chunk_key`, `face_index`, `corners` |
| `cast_ray(store, origin, direction, ...)` | `max_distance=None`, `level=0` | `hit`, `t` (distance along the ray), `position`, `chunk_key`, `face_index`, `corners` |

- `closest_point` searches the chunks around the query, one ring of chunks at a
  time, up to `max_expansion_rings` rings. If nothing lies within
  `max_distance`, it returns `found=False` and `distance=inf`.
- `cast_ray` normalises `direction` and raises `ValueError` if it is zero. It
  walks the chunks along the ray and stops at the first hit. A miss returns
  `hit=False` and `t=inf`.
- `corners` names the face that was found: each corner as `(chunk, index in
  chunk)`, in the face's winding order.
- `chunk_key` is the chunk holding the face's first corner. `face_index` is
  the face's number among that chunk's own faces, or `None` when the face's
  corners lie in more than one chunk.
