# Concepts

A Zarr Vectors **store** holds vector geometry, from point clouds to meshes, cut
into a spatial grid so that a reader loads only the region and level of detail
it needs. Terms in bold are defined in the [glossary](#glossary). This is the
[quickstart](quickstart.md)'s `cells.zv`, with levels 1 and 2 trimmed:

```text
cells.zv
├── 0
│   ├── object_index
│   ├── vertex_attributes
│   ├── vertex_fragments
│   └── vertices
├── 1
├── 2
└── headers
```

Each numbered directory is a **level**: `0/` holds the full-resolution data and
the coarser levels form the **pyramid**. A level cuts space into **chunks**, each
split into **bins**. Every per-chunk array, such as `vertices/`, keeps one file
per chunk at `c/<i>/<j>/<k>` (or packs many into a **shard**), so a reader
fetches only the chunks it needs.

## Objects and attributes

An **object** that crosses chunk boundaries is cut into **fragments**, one or
more per chunk it touches. Its **manifest** lists them, so reading one object
reads only its own chunks; in `cells.zv`, 18 of the 20 objects span more than
one chunk. Objects can be collected into **groups**, and **links** join vertices
into edges and mesh triangles.

Attributes sit beside the geometry in a level's directory: one value per vertex
in `vertex_attributes/<name>`, per object in `object_attributes/<name>`, or per
fragment in `fragment_attributes/<name>`. Metadata from the source file, such as
a TRK affine, is kept under `headers/<format>/`.
[Headers and attributes](reference/headers_attributes.md) lists what each ingest
writes.

## What this package adds

The core package, [zarr-vectors](https://github.com/AllenInstitute/zarr-vectors-py),
defines Zarr Vectors format 0.9 and does the low-level reading and writing of
stores. zarr-vectors-tools adds readers and writers for file formats, pyramid
builders for each geometry, algorithms that run over a store, and
[merge and split](compose.md) of stores.

## Glossary

Store
: A Zarr v3 directory or bucket prefix holding one dataset; named `*.zv` in these docs.

Level
: One resolution of the data, at `<n>/`. Level 0 is full resolution; higher levels are coarser.

Chunk
: One cell of a level's spatial grid, `--chunk-shape` wide in the input's coordinate units.

Bin
: A sub-cell of a chunk (`--bin-shape`, default the whole chunk). Chunks are whole multiples of it.

Object
: One item, such as a streamline, neuron, mesh or cluster of points, with an integer ID.

Fragment
: A piece of one object stored in one chunk; an object has one or more per chunk it touches.

Segment ID
: The source's own ID for an object (an EM segment ID, or an SWC tree's number), in `object_attributes/segment_id`.

Vertex attribute
: A value per vertex, in `<level>/vertex_attributes/<name>`: a scalar, or a short row such as RGB.

Object attribute
: A value per object, in `<level>/object_attributes/<name>`, such as a streamline's `length`.

Group
: A named set of objects, such as a TRX group, in `<level>/groups`. See `zvtools split --by groups`.

Link
: An edge or mesh triangle, in `<level>/links/<delta>/<offsets>/`. `<delta>` is the level gap (`0` within
  a level); `<offsets>` locate the other vertices' chunks. Streamlines store only chunk-crossing steps.

Manifest
: The list of chunks and fragments that hold one object, in `<level>/object_index/manifests`.

Pyramid
: Levels 1 to N, each built from the one below with fewer vertices, fewer objects or larger chunks.

Sparsity
: Per coarser level, the divisor of the object count: `--sparsity 2,2` keeps 1/2, then 1/4, of the objects.

Coarsen factor
: Per coarser level, how much each object's geometry is simplified (`--coarsen`). See [Pyramids](pyramids.md).

Chunk scale
: Per coarser level, the chunk-edge multiplier: `--chunk-scale 2,2` gives 2×, then 4×, the level-0 edge.

voxmm
: TRK's native coordinates (voxel position × voxel size, in mm), kept unless you pass `--apply-affine`.

RAS
: Right-anterior-superior: the neuroimaging world convention, in mm (+x right, +y anterior, +z superior).

LOD
: Level of detail: a viewer draws coarse levels zoomed out, finer ones zoomed in. See [Visualise](visualise.md).

Shard
: One file holding many chunks of an array (`--shard 8`: 8×8×8 chunks). See [Store layout](store_layout.md#sharding).
