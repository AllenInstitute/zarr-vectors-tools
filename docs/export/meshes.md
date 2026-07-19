# Meshes

The ZVF `meshes` geometry writes out to Wavefront OBJ via `export_obj`.
No third-party dependency.

```python
from zarr_vectors_tools.export.obj import export_obj

summary = export_obj(
    "model.zv",                 # store_path
    "model.obj",                # output_path
    level=0,                    # resolution level to read
    bbox=([-100.0] * 3, [100.0] * 3),   # (low_corner, high_corner)
    object_ids=[0, 1],          # keep only these objects; AND-ed with bbox
    chunks=None,                # chunk whitelist; also AND-ed
)
summary["vertex_count"]         # → int
summary["face_count"]           # → int
```

The output is a plain 1-indexed OBJ: two `#` comment lines, one `v` line
per vertex, one `f` line per face. Two-dimensional vertices are padded to
`z = 0.0`; faces are written with whatever arity the store holds, so
quads stay quads. Raises `ExportError` if the filters leave no vertices.

:::{note}
Only geometry is written. Normals (`vn`), texture coordinates (`vt`),
material references (`mtllib` / `usemtl`), and `o` / `g` grouping
directives are not emitted, even when the store has an `OBJHeader`
holding the original `mtllib` and object names — pull those from the
[`HeaderRegistry`](../headers.md) if you need to reattach them.
:::

## Filtering with `chunks`

A face is only written when all of its vertices survive the filter, so a
face spanning a listed and an unlisted chunk is dropped. Cutting a mesh
by `chunks` therefore leaves an open boundary along the chunk seam rather
than a clean cross-section.

```python
# Whole objects — watertight if the source was.
export_obj("model.zv", "objects.obj", object_ids=[0, 1])

# A spatial slab — faces straddling the seam are gone, leaving holes.
export_obj("model.zv", "slab.obj", chunks=[(0, 0, 0), (0, 0, 1)])
```

## See also

- [Export overview](index.md) — shared call shape and the `level=` parameter.
- [Ingest → meshes](../ingest/meshes.md) — the symmetric direction, OBJ and STL.
- [Mesh algorithms](../algorithms/index.md) — compute surface area, normals, or curvature on the store before exporting.
- [Multiresolution](../multiresolution/index.md) — build decimated levels for preview exports.
