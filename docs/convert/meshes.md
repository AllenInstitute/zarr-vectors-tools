# Meshes

Convert OBJ and STL files and Neuroglancer precomputed mesh layers to mesh
stores, and export them back. Chunk shape, `--overwrite` and the export
options are in [Converting files](index.md).

| Format | In / out | Extra | Units |
|---|---|---|---|
| `obj` | in, out | — | the file's |
| `stl` | in | — | the file's |
| `precomputed` | in, out | `precomputed` | nm |

```bash
zvtools convert model.obj model.zv --chunk-shape 50,50,50
```

```python
from zarr_vectors_tools.convert.ingest.obj import ingest_obj

ingest_obj("model.obj", "model.zv", (50.0, 50.0, 50.0), auto_object_id=True)
```

| Option | Format | Effect |
|---|---|---|
| `--split-objects` | obj | One object per `o` or `g` name (Python `auto_object_id=True`). Default: the whole file is one object. |
| `--merge-tolerance D` | stl | Weld corners that round to the same point on a grid of spacing D (default `1e-6`, file units). |
| `--no-merge-vertices` | stl | Keep every triangle's corners separate. |
| `--segment-id ID` | precomputed | Read only this segment (repeatable). Default: all. |
| `--lod N` | precomputed | Level of detail of a multi-resolution layer; `0` (default) is the finest. |

## What the store holds

- Triangles, or quads when every face of an OBJ is a quad. Larger faces are
  split into triangles.
- OBJ vertex normals as `normal`, when there is one `vn` per `v`. Texture
  coordinates, materials and STL facet normals are dropped.
- With `--split-objects`, the `o` and `g` names in the store's `obj` header.
- For a precomputed layer, one object per segment, with `segment_id` and the
  segment properties as object attributes. Fragment seams are welded.

## Neuroglancer precomputed mesh layers

```bash
zvtools convert seg/mesh cells.zv --chunk-shape 32000,32000,32000 --segment-id 7
```

INPUT is the mesh directory (local, or `gs://`, `s3://`, `https://`,
`file://`), not the segmentation layer above it. Legacy and multi-resolution
layers, plain or sharded, are read. `--chunk-shape` is required, in nm. Memory
holds the selected segments until they are written, so `--segment-id` bounds it.

## Export

```bash
zvtools convert model.zv model_out.obj
```

OBJ export writes vertices and faces only. `--object-id` keeps those objects.
`--bbox` keeps faces whose corners are all inside, so a box that cuts the mesh
leaves an open edge.

Precomputed export writes a segmentation layer with a single-resolution
`mesh/` in nm, one segment per object. Segment ids come from `segment_id`, else
object id + 1. Pass `--unit micrometer` if the store records no unit and is not
in nm.

## Pyramids

Use `--method mesh_decimate`, which keeps a closed surface closed, with
`--sparsity 1`:

```bash
zvtools pyramid model.zv --coarsen 4,4 --sparsity 1,1 --chunk-scale 2,2 --method mesh_decimate
```

Carrying per-vertex attributes to coarser levels needs the `mesh` extra. Choosing values:
[Pyramids](../pyramids.md).
