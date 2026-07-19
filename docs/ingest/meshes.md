# Meshes

Two mesh ingests, both pure Python with no extra to install: Wavefront
OBJ and STL. Both write the **mesh** ZVF geometry. PLY is points-only in
this package — see [Point clouds](point_clouds.md).

## Wavefront OBJ — `ingest_obj`

```python
from zarr_vectors_tools.ingest.obj import ingest_obj

ingest_obj(
    "model.obj",
    "model.zv",
    (1.0, 1.0, 1.0),
    encoding="raw",              # or "draco"
    draco_quantization_bits=11,  # only read when encoding="draco"
    auto_object_id=True,         # split by o/g directives
)
```

Triangles and quads are stored as-is: an all-triangle file gets link
width 3, an all-quad file link width 4. Faces with more than four
vertices are fan-triangulated as they are parsed. If the file mixes
triangles and quads, every quad is split into two triangles so the whole
mesh lands at link width 3 — a uniform-quad mesh keeps its quads, a mixed
one does not.

Face indices accept the `v/vt/vn` form (only the vertex index is used)
and negative indices, resolved relative to the vertices seen so far.
Vertex normals from `vn` lines become `attributes["normal"]`, but only
when their count exactly matches the vertex count.

`auto_object_id=True` tracks `o <name>` and `g <name>` directives and
gives each subsequent vertex the integer ID of the most recently declared
name. The name list is preserved in `OBJHeader.object_names` so export
can restore the directives.

:::{note}
Object IDs are assigned to **vertices in declaration order**, not by
which faces reference them. A file that declares all vertices up front
and only then opens its groups will put every vertex in object 0.
:::

## STL — `ingest_stl`

```python
from zarr_vectors_tools.ingest.stl import ingest_stl

ingest_stl(
    "model.stl",
    "model.zv",
    (0.1, 0.1, 0.1),
    merge_vertices=True,     # default
    merge_tolerance=1e-6,
    encoding="raw",
)
```

ASCII and binary are auto-detected by sniffing the first 80 bytes for the
`solid` magic, so you do not declare which you have.

STL stores every triangle as three independent vertices with no sharing
at all. `merge_vertices=True` (the default) recovers the shared topology
by rounding positions onto a `merge_tolerance` grid and deduplicating —
the *original* coordinate of the first occurrence is kept, not the
rounded one, so nothing is quantised in the output. Set
`merge_tolerance=0` for exact-match deduplication only.

:::{warning}
Leave `merge_vertices` on unless you have a specific reason not to.
Without it the mesh has no connected topology — every triangle is an
island — and the algorithms that walk edges (vertex normals, curvature,
connected components) return meaningless results.
:::

Per-face normals are parsed out of the STL but are not written to the
store; recompute vertex normals from the merged geometry instead.

## See also

- [Algorithms → mesh attributes](../algorithms/mesh_attributes.md)
- [Algorithms → mesh summary](../algorithms/mesh_summary.md)
- [Export → meshes](../export/meshes.md)
- [Enrichments → meshes](../enrichments.md#meshes)
- [Ingest workflows](index.md)
