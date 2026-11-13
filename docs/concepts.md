# Concepts

This page shows how a store written by zarr-vectors-tools is laid out and
defines the terms the tools add. The format itself (levels, chunks, bins,
objects, fragments, manifests, links and attributes) belongs to the core
package, zarr-vectors-py: see its {zvpy}`core concepts <getting_started/concepts.html>`
and the {zvpy}`specification <spec/index.html>`.

## A store on disk

This is the [quickstart](quickstart.md)'s `cells.zv`, with levels 1 and 2
collapsed:

```text
cells.zv
├── 0
│   ├── fragment_attributes
│   ├── object_attributes
│   ├── object_index
│   ├── vertex_attributes
│   ├── vertex_fragments
│   └── vertices
├── 1
├── 2
└── headers
```

Each numbered directory is a level: `0/` holds the full-resolution data, and
`1/` and `2/` are the coarser levels of the pyramid. The
{zvpy}`directory structure <spec/layout/directory_structure.html>` page
describes every array.

## What this package adds

The core package reads and writes stores. zarr-vectors-tools adds:

- readers and writers for file formats ([Convert](convert/index.md));
- pyramid builders for each geometry ([Pyramids](pyramids.md));
- algorithms that run over a store ([Algorithms](algorithms/index.md));
- [merge and split](compose.md) of stores.

Every ingest also writes what a viewer and a later export need:

- `object_attributes/vertex_count` (vertices per object) and
  `fragment_attributes/segment_id` (the ID each fragment is coloured and picked
  by), in every level of a store with objects;
- the source file's own metadata, such as a TRK affine, under
  `headers/<format>/`.

[Headers and attributes](reference/headers_attributes.md) lists what each
ingest writes.

## Glossary

Terms of the format (store, level, chunk, bin, object, fragment, manifest,
link, group, vertex and object attribute) are defined in the core package's
{zvpy}`core concepts <getting_started/concepts.html>`. These docs name stores
`*.zv`.

Segment ID
: The source's own ID for an object (an EM segment ID, or an SWC tree's number), in `object_attributes/segment_id`.

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
