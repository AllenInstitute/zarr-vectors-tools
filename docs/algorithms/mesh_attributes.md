# Mesh attributes

Two per-vertex mesh attributes derived directly from the chunked
store: vertex normals and mean curvature. Both accept `write_back=True`
to persist the result back to the store via
`ZVWriter.add_node_attribute_sync` so subsequent reads (or downstream
viewers) can use the attribute without recomputing.

## `compute_vertex_normals`

```python
from zarr_vectors_tools.algorithms import compute_vertex_normals

result = compute_vertex_normals(
    "mesh.zv",
    weighting="area",
    write_back=True,
)

result["normals"]                       # (N, 3) float32 — unit normals in global order
result["incomplete_boundary_vertices"]  # int — vertices on cross-chunk edges
```

`weighting`
: `"area"` (default) sums each incident face's un-normalised normal
  (so faces with more area contribute more). `"uniform"` sums
  unit-length face normals.

`write_back`
: When `True`, the result is persisted under
  `attributes/vertex_normal/` via
  `ZVWriter.add_node_attribute_sync("vertex_normal", normals, dtype=np.float32)`.

## `compute_mean_curvature`

Cotangent Laplace–Beltrami mean curvature (Meyer et al. 2003). The
per-vertex output is `‖H(v)‖ / 2` where
`H(v) = (1 / (2 A_v)) Σ_w (cot α_vw + cot β_vw)(x_v − x_w)`.

```python
from zarr_vectors_tools.algorithms import compute_mean_curvature

result = compute_mean_curvature("mesh.zv", write_back=True)

result["mean_curvature"]                # (N,) float32
result["incomplete_boundary_vertices"]  # int
```

`write_back`
: Persists under `attributes/mean_curvature/`.

## Algorithm notes

Both functions stream intra-chunk faces, accumulating per-vertex
quantities using `np.add.at` against a globally-indexed buffer. Vertex
indices are mapped to global indices via
`chunk_local_to_global_offsets` (the public helper from
`zarr_vectors.spatial.boundary`).

Cross-chunk **faces** lose face identity in the v0 storage layout, so
their contributions to normals / curvature are not added — vertices
appearing in any cross-chunk record are counted as
`incomplete_boundary_vertices` so callers can quantify the effect.
For typical meshes the boundary is a small fraction of the surface
and the resulting error is negligible; for meshes that are heavily
subdivided across chunks, treat boundary-vertex values as approximate.

The Voronoi area in the mean-curvature denominator uses the
barycentric approximation (face_area / 3 per vertex). The full
obtuse-triangle Voronoi case introduces special-case handling that
isn't essential for v0 and isn't implemented.

## See also

- [Algorithms index](index.md)
- [Mesh summary](mesh_summary.md) — global area / volume / Euler χ.
- Parent: [`ZVWriter` API](https://zarr-vectors.readthedocs.io/en/latest/api/lazy.html)
