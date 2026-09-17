# Meshes

Three mesh ingests, all writing the **mesh** Zarr Vectors geometry:
Wavefront OBJ and STL, which are pure Python with no extra to install, and
Neuroglancer precomputed mesh layers, which need the `precomputed` extra. PLY
is points-only in this package — see [Point clouds](point_clouds.md).

## Wavefront OBJ — `ingest_obj`

```python
from zarr_vectors_tools.convert.ingest.obj import ingest_obj

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
from zarr_vectors_tools.convert.ingest.stl import ingest_stl

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

## Neuroglancer precomputed — `ingest_precomputed_meshes`

A precomputed mesh layer holds one mesh per segment. Each segment becomes
one object, numbered by the rank of its segment id, and the ids are stored
ascending as the `segment_id` object attribute, as the EM skeleton ingests
do. `export_precomputed` and lookups by segment id work on the result.

```bash
zvtools convert gs://bucket/segmentation/mesh cells.zv \
    --chunk-shape 32000,32000,32000 --segment-id 720575940000000011 --lod 1
```

```python
from zarr_vectors_tools.convert.ingest.precomputed_meshes import ingest_precomputed_meshes

ingest_precomputed_meshes(
    "gs://bucket/segmentation/mesh",   # the mesh directory, with its own info
    "cells.zv",
    (32000, 32000, 32000),              # nm; there is no source grid to inherit
    segment_ids=[720575940000000011],   # default: every segment in the layer
    lod=1,                              # multi-resolution layers; 0 is the finest
)
```

Point it at the mesh directory, not the segmentation layer above it. A
segmentation layer is refused with a message naming its mesh and skeleton
directories.

| Layer `info` | Layout | Levels of detail | Segments listed from |
| --- | --- | --- | --- |
| `neuroglancer_legacy_mesh` | a `<id>:0` manifest and its fragment files | one | the manifest names |
| `neuroglancer_multilod_draco` | a `<id>.index` manifest and a `<id>` file of Draco fragments | several | the manifest names |
| the same, with `sharding` | manifests and fragments packed into `.shard` files | several | each shard's minishard indices |

- **Welding.** Chunked meshing pipelines write a vertex on a chunk face once
  for each fragment that touches it. `weld=True` (the default) merges
  vertices with identical coordinates within a segment, then drops faces
  that collapse or repeat. A segment's mesh then comes back closed where
  the source mesh was closed. With `weld=False`, the fragments do not
  share vertices across their seams.
- **Segment properties.** When the mesh `info` names `segment_properties`,
  its inline properties become object attributes (strings as `S256`,
  numbers in their own dtype, tags as a bitmask). Pass
  `segment_properties=False` to skip them.
- **Empty segments.** A segment with no faces at the chosen `lod` is left
  out and listed under `empty_segments` in the summary.
- **Pyramids.** The pyramid is built afterwards, as for any mesh store:
  `--coarsen` on the command line, or `build_pyramid` in Python. See
  [Building pyramids](../multiresolution/building_pyramids.md).
- **Memory.** Reads are threaded, `batch_size` segments at a time. The store
  is written in one call, though, so memory holds every ingested segment's
  vertices and faces at once, about 20 bytes per vertex and 24 per
  triangle. Select segments to bound it.
- **Local layers.** A legacy layer's `<id>:0` manifests cannot exist on a
  Windows filesystem, so read those layers from a bucket there.
  Multi-resolution layers are fine locally.

## See also

- [Algorithms → mesh attributes](../algorithms/mesh_attributes.md)
- [Algorithms → mesh summary](../algorithms/mesh_summary.md)
- [Export → meshes](../export/meshes.md)
- [Enrichments → meshes](../enrichments.md#meshes)
- [Cortical surfaces](surfaces.md) — GIFTI, FreeSurfer and CIFTI
- [Ingest workflows](index.md)
