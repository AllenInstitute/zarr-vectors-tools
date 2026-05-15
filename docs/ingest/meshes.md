# Meshes

Two mesh ingests today: Wavefront OBJ and STL (ASCII or binary). Both
write to the **mesh** ZVF geometry, both are pure Python with no
external dependencies. Mesh PLY ingest is a future addition; the
current PLY ingest is points-only (see [Point clouds](point_clouds.md)).

## Wavefront OBJ — `ingest_obj`

```python
from zarr_vectors_tools.ingest import ingest_obj

ingest_obj(
    input_path="model.obj",
    output_path="model.zv",
    chunk_shape=(1.0, 1.0, 1.0),
    auto_object_id=True,        # split by o/g directives
)
```

The parser supports triangle and quad faces directly; polygons with
more than four vertices are fan-triangulated. Vertex normals (`vn`
lines) are read into `attributes["normal"]` if present.

**Notable options**

`encoding`
: `"raw"` (default) or `"draco"` — controls the mesh encoding used by
  the underlying `write_mesh` writer.

`draco_quantization_bits`
: Only relevant when `encoding="draco"`.

`auto_object_id`
: Parse `o <name>` and `g <name>` directives. Each subsequent vertex
  inherits the most recently declared name's integer ID; the name list
  is preserved in [`OBJHeader.object_names`](../headers.md) for export.

## STL — `ingest_stl`

```python
from zarr_vectors_tools.ingest import ingest_stl

ingest_stl(
    input_path="model.stl",
    output_path="model.zv",
    chunk_shape=(0.1, 0.1, 0.1),
    merge_vertices=True,
    merge_tolerance=1e-6,
)
```

STL stores each triangle as three independent vertices — there's no
explicit face sharing. `merge_vertices=True` (default) deduplicates
vertices that lie within `merge_tolerance` of each other, recovering
the shared-vertex topology that downstream algorithms (closest-point,
vertex normals) need to behave correctly.

Both ASCII and binary STL files are auto-detected and parsed.

**Notable options**

`encoding`
: `"raw"` or `"draco"`.

`merge_vertices` / `merge_tolerance`
: As described above. Set `merge_vertices=False` only if you have a
  specific reason to preserve the duplicated-vertex layout.

## See also

- [Algorithms → mesh](../algorithms/index.md) — summary, normals,
  curvature, closest-point, ray casting.
- [Export → OBJ](../export/meshes.md)
- Parent: [Mesh spec](https://zarr-vectors.readthedocs.io/en/latest/spec/geometry_types/mesh.html)
