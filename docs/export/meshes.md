# Meshes

The ZVF mesh geometry round-trips to Wavefront OBJ via `export_obj`.

## Wavefront OBJ — `export_obj`

```python
from zarr_vectors_tools.export import export_obj

export_obj(
    store_path="model.zv",
    output_path="model.obj",
    object_ids=[0, 1],
)
```

Returns `{"vertex_count": int, "face_count": int}`.

The exporter writes a 1-indexed OBJ with `v` (vertex) and `f` (face)
lines. If the source store was ingested with `ingest_obj(...,
auto_object_id=True)` the [`OBJHeader.object_names`](../headers.md) is
used to emit `o <name>` directives in the output so groups round-trip.

**Notable options**

`level`
: Resolution level to export.

`bbox`
: `(low_corner, high_corner)` bounding box filter.

`object_ids`
: Object ID filter; AND-ed with `bbox` and `chunks`.

`chunks`
: Whitelist of chunk-coordinate tuples. Faces spanning a listed and an
  unlisted chunk are dropped.

## See also

- [Ingest → OBJ and STL](../ingest/meshes.md)
- [Mesh algorithms](../algorithms/index.md) — compute attributes on a
  store before exporting (surface area, normals, mean curvature, …).
