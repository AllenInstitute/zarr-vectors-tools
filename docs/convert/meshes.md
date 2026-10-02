# Meshes

```bash
zvtools convert model.obj model.zv --chunk-shape 50,50,50
zvtools convert part.stl part.zv --chunk-shape 50,50,50
```

`--chunk-shape` is in the mesh's own coordinate units: for a model in micrometres,
`50,50,50` gives 50 µm chunks ([choosing it](../store_layout.md); shared options:
[Convert](index.md)). OBJ and STL need no extra.

## OBJ

Triangles and quads are stored as they are; larger faces are fan-triangulated, and a
file mixing triangles and quads is stored as triangles. Only the vertex index of a
`v/vt/vn` corner is read (negative indices work); texture coordinates and materials
are dropped. Vertex normals become the vertex attribute `normal` when there is one
`vn` per `v`. That attribute currently stops chunks rendering in the viewer
([Visualise](../visualise.md)): for a model meant for viewing, drop the normals first
with `grep -v '^vn ' model.obj > model_plain.obj`.

By default the whole file is one [object](../concepts.md). `--split-objects` gives one
object per `o` or `g` name (each vertex takes the id of the last such line above it) and
keeps the names in the store's `obj` header:

```bash
zvtools convert model.obj model_objects.zv --chunk-shape 50,50,50 --split-objects
```

In Python the same is `ingest_obj(..., auto_object_id=True)`.

## STL

ASCII and binary STL are detected automatically. STL repeats each corner once per
triangle, so the ingest welds corners that round to the same point of a grid
`--merge-tolerance` fine (default `1e-6`, file units): a 5,120-triangle sphere gives
2,562 vertices, or 15,360 disconnected ones with `--no-merge-vertices`. Facet normals are
not stored.

```bash
zvtools convert part.stl part_welded.zv --chunk-shape 50,50,50 --merge-tolerance 1e-4
```

In Python these are `ingest_stl(..., merge_tolerance=1e-4)` and `merge_vertices=False`.
Both readers store meshes raw: `encoding="draco"` is refused, since a Draco store could
not be read back.

## Neuroglancer precomputed mesh layers

```bash
zvtools convert seg/mesh cells.zv --chunk-shape 32000,32000,32000
zvtools convert seg/mesh two_cells.zv --chunk-shape 32000,32000,32000 --segment-id 7 --segment-id 9 --lod 1
```

INPUT is the mesh directory (local, or `gs://`, `s3://`, `https://`, `file://`, with or
without `precomputed://`), not the segmentation layer above it. Legacy and
multi-resolution Draco layouts, plain or sharded, are read. Needs `[precomputed]`.

| Option | Meaning |
| --- | --- |
| `--chunk-shape X,Y,Z` | required, in nanometres: a mesh layer has no grid to inherit |
| `--segment-id ID` | read only this segment; repeatable. Default: every segment |
| `--lod N` | multi-resolution layers: level of detail, `0` (default) the finest |

Each segment becomes one object, in ascending segment id, kept as the object attribute
`segment_id`; inline segment properties become object attributes too. Fragment seams
are welded, so a closed mesh stays closed. Memory holds the selected segments until
written (about 20 bytes per vertex, 24 per triangle): `--segment-id` bounds it.

## Export

```bash
zvtools convert model.zv model_out.obj
zvtools convert model.zv corner.obj --bbox 0,0,0,100,100,100
zvtools convert model_objects.zv ball_b --format precomputed --object-id 1
```

OBJ export writes vertices and faces from `--level N` (default 0); quads stay quads,
and no normals, groups or materials are written. `--bbox X0,Y0,Z0,X1,Y1,Z1` keeps the
faces whose corners are all inside, so a box that cuts the mesh leaves an open edge.
OBJ export cannot select objects, so `--object-id` is refused.

Precomputed export (`--format precomputed`, or a URL as OUTPUT; needs `[precomputed]`)
writes a segmentation layer with a legacy `mesh/`: one level of detail, in nanometres,
one segment per object (object N is segment N + 1 unless the store has `segment_id`).
Pass `--unit micrometer` (or `millimeter`) if the store records no unit and is not in
nm. Segment properties work as for [skeletons](skeletons.md). On Windows, write to a bucket.

## Pyramids

`zvtools pyramid`, or the same flags on `convert`, adds coarser levels. For meshes pass
`--method mesh_decimate`, edge-collapse decimation, which keeps a closed surface closed:

```bash
zvtools pyramid model_objects.zv --coarsen 4,4 --sparsity 1,1 --chunk-scale 2,2 \
    --method mesh_decimate
```

Each `--coarsen` value (> 1) is how many times smaller a level is than the one below. On
a model of two spheres (5,124 vertices, 50-unit chunks) the command above gave
5,124 → 1,284 → 324 vertices in 100- and 200-unit chunks, with no open or non-manifold
edges. `--sparsity` must be 1: decimation keeps every object. Object attributes and
segment ids carry to every level; per-vertex attributes come from the nearest source
vertex, which needs scipy (`[mesh]`; without it they are left off, with a warning).
In Python: `build_pyramid(..., factors=[(4, 1), (4, 1)], chunk_scale_factors=[2, 2],
method="mesh_decimate")`.

Without `--method`, levels are built by vertex clustering, the one mesh method that
also drops objects (`--sparsity`). On the same model `--coarsen 2,2` left 16 open and
40 non-manifold edges and stray triangles. Keep `--chunk-scale 2,...` with either
method: the viewer never refines a level that keeps every object and the chunk size of
the level below ([Visualise](../visualise.md)). Choosing values: [Pyramids](../pyramids.md).
