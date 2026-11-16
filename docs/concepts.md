# Concepts

What zarr-vectors-tools adds to a store, and the terms these docs use. The
format itself belongs to the core package: see its
{zvpy}`concepts <explanation/concepts.html>` and
{zvpy}`specification <spec/index.html>`.

## A store on disk

The [quickstart](quickstart.md)'s `cells.zv`, with levels 1 and 2 collapsed:

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

`0/` is full resolution; `1/` and `2/` are coarser levels of the pyramid. The
{zvpy}`directory structure <spec/layout/directory_structure.html>` page
describes every array.

The core package reads and writes stores. zarr-vectors-tools adds file
[conversion](convert/index.md), [pyramids](pyramids.md),
[algorithms](algorithms/index.md), and [merge and split](compose.md). Every
ingest writes `object_attributes/vertex_count` and
`fragment_attributes/segment_id` in every level of a store with objects, and
the source file's metadata under `headers/<format>/`; [Headers and attributes](reference/headers_attributes.md)
lists them.

## Glossary

Format terms (level, chunk, bin, object, fragment, link) are defined in the
core {zvpy}`concepts <explanation/concepts.html>`. These docs name stores
`*.zv`.

Segment ID
: The source's own ID for an object (an EM segment ID, or an SWC tree's number), in `object_attributes/segment_id`.

Sparsity
: Per coarser level, the divisor of the object count: `--sparsity 2,2` keeps 1/2, then 1/4, of the objects.

Coarsen factor
: Per coarser level, how much each object's geometry is simplified (`--coarsen`).

Chunk scale
: Per coarser level, the chunk-edge multiplier: `--chunk-scale 2,2` gives 2×, then 4×, the level-0 edge.

voxmm
: TRK's native coordinates (voxel position × voxel size, in mm), kept unless you pass `--apply-affine`.

RAS
: Right-anterior-superior: the neuroimaging world convention, in mm.

Shard
: One file holding many chunks of an array. See [Store layout](store_layout.md#sharding).
