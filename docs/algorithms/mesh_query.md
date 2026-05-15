# Mesh queries

Two spatial queries on a chunked triangle mesh: closest-point on the
surface, and first-hit ray intersection. Both localise candidate
chunks via `zarr_vectors.spatial.chunking.chunks_intersecting_bbox`
and test only the chunks they need.

## `closest_point`

```python
import numpy as np
from zarr_vectors_tools.algorithms import closest_point

result = closest_point(
    "mesh.zv",
    query=np.array([5.0, 1.0, 2.0]),
    max_distance=10.0,
)

result["found"]        # bool
result["position"]     # (3,) float64 — closest point on the mesh
result["distance"]     # float — Euclidean distance
result["chunk_key"]    # tuple | None — chunk containing the winning face
result["face_index"]   # int | None — face index local to chunk_key
```

`query`
: `(3,)` query point in store coordinates.

`max_distance`
: Optional cap on the search radius. When `None`, the search expands
  rings of neighbouring chunks until a hit is found or all occupied
  chunks have been visited.

`max_expansion_rings`
: Safety cap on the ring expansion (default 4). Each ring widens the
  candidate bbox by one chunk size.

### Algorithm notes

Eberly 2001 closest-point-on-triangle, vectorised across each chunk's
face array. Search expands rings outward from the query's home chunk
until a hit is closer than the next ring's bounding box can offer —
the standard early-out for ring-expansion searches.

Cross-chunk faces are not tested. For meshes where most faces are
intra-chunk this is negligible; for high-cross-chunk meshes the
boundary may be off by at most one chunk's diameter.

## `cast_ray`

```python
import numpy as np
from zarr_vectors_tools.algorithms import cast_ray

result = cast_ray(
    "mesh.zv",
    origin=np.array([0.0, 0.0, 0.0]),
    direction=np.array([1.0, 0.0, 0.0]),  # auto-normalised
    max_distance=100.0,
)

result["hit"]          # bool
result["t"]            # float — ray parameter at hit; inf on miss
result["position"]     # (3,) float64 — hit position
result["chunk_key"]    # tuple | None
result["face_index"]   # int | None — local to chunk_key
```

`origin`, `direction`
: `(3,)` arrays. `direction` is normalised internally; passing
  `(0, 0, 0)` raises `ValueError`.

`max_distance`
: Optional upper bound on hit distance.

### Algorithm notes

Chunk traversal via 3D **DDA** (digital differential analyser) along
the ray direction. Each occupied chunk along the path runs a
vectorised Möller-Trumbore intersection on its triangles; the first
chunk that produces a hit wins. Halt criterion: along some axis the
current chunk position has permanently exited the occupied-chunk
bounding box.

As with closest-point, cross-chunk faces are not tested.

## See also

- [Algorithms index](index.md)
- [Mesh summary](mesh_summary.md) — global stats over the same store.
- Parent: [`chunks_intersecting_bbox`](https://zarr-vectors.readthedocs.io/en/latest/api/spatial.html)
