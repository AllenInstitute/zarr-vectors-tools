# Mesh summary

`compute_mesh_summary` streams every chunk's faces and reports
surface area, signed-tetrahedron volume, face count, edge count,
vertex count, and the Euler characteristic `V − E + F` for a triangle
mesh store. With `per_object=True` it additionally walks each object's
manifest to attribute the per-fragment contributions to a specific
object ID.

## Usage

### Global summary

```python
from zarr_vectors_tools.algorithms import compute_mesh_summary

result = compute_mesh_summary("mesh.zv")

result["surface_area"]                # float — sum of per-face triangle areas
result["volume"]                      # float — signed-tetrahedron sum
result["face_count"]                  # int
result["vertex_count"]                # int
result["edge_count"]                  # int — deduplicated intra + cross
result["euler_characteristic"]        # int — V - E + F
result["excluded_cross_face_edges"]   # int — cross-chunk face boundary edges
```

`volume` is only physically meaningful for closed, consistently-wound
meshes. The Euler characteristic is exact only when the store has no
cross-chunk faces; the `excluded_cross_face_edges` counter quantifies
how many face-boundary edges contributed to the dedup set but couldn't
be attributed to a specific face.

### Per-object summary

```python
result = compute_mesh_summary("mesh.zv", per_object=True)

for entry in result["per_object"]:
    entry["object_id"]
    entry["surface_area"]
    entry["volume"]
    entry["face_count"]
    entry["vertex_count"]
```

`per_object=True` reads every object manifest from `object_index/`,
fetches each fragment via `read_fragment`, and pairs it with the
matching `read_chunk_links` entry (the v0.6 fragment-index layout
aligns vertex and link groups index-for-index per chunk). A per-chunk
link-group cache avoids re-decoding for objects that share chunks.

Under the v0.6 `CAP_SHARED_FRAGMENTS` capability, multiple objects
sharing a fragment will each receive that fragment's contribution —
this is the intended semantic; the per-object totals are not expected
to sum to the global totals if shared fragments are present.

## Algorithm notes

- **Surface area**: half the L2 norm of `(v1 - v0) × (v2 - v0)`,
  summed across all triangles.
- **Volume**: signed tetrahedron from origin, `(v0 · (v1 × v2)) / 6`,
  summed.
- **Edge count**: every triangle contributes three edges to a dedup
  set keyed on `((chunk_key, local_index), (chunk_key, local_index))`.
  Records spanning two or more chunks, of arity ≥ 3, contribute their
  consecutive endpoint pairs to the same set. Those come from
  `read_cross_links(level_group, delta=0)` and **not** from a bare
  `read_links`: the per-chunk `read_chunk_links` loop above has already
  consumed every intra-chunk record from the same `links/0/<offsets>/`
  family, so a whole-family read would double-count them. See the
  double-count warning on the [algorithms index](index.md).
- **Triangle meshes only** (`link_width=3`). Quad / polygon support
  requires fan-triangulation; not implemented in v0.

## See also

- [Algorithms index](index.md)
- [Mesh attributes](mesh_attributes.md) — per-vertex normals / curvature.
- [Mesh queries](mesh_query.md)
- Parent: [Mesh spec](https://zarr-vectors.readthedocs.io/en/latest/spec/geometry_types/mesh.html)
