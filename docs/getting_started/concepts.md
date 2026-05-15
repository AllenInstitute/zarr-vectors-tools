# Concepts

`zarr-vectors-tools` is organised around three workflows that compose
over a shared on-disk format:

```text
┌──────────┐   ┌────────────────┐   ┌──────────────┐   ┌──────────┐
│ source   │──▶│ ingest_<fmt>() │──▶│ Zarr Vectors │──▶│ export_  │──▶ output
│ file     │   │ + enrichments  │   │   store      │   │  <fmt>() │   file
└──────────┘   └────────────────┘   └──────────────┘   └──────────┘
                                            ▲
                                            │
                                  ┌──────────────────────┐
                                  │ compute_*() algorithm │
                                  │ (optional, may write  │
                                  │  attributes back)     │
                                  └──────────────────────┘
```

Every public entry point in this package either:

1. **Reads** a file format and writes a ZVF store (`zarr_vectors_tools.ingest`),
2. **Computes** over a ZVF store and optionally writes results back
   (`zarr_vectors_tools.algorithms`), or
3. **Reads** a ZVF store and writes a file format
   (`zarr_vectors_tools.export`).

## Relationship to `zarr-vectors-py`

The on-disk format, the chunk encoding, the spatial index, and the
multiresolution pyramid all live in
[`zarr-vectors`](https://zarr-vectors.readthedocs.io/). That package
defines:

- The `write_points`, `write_mesh`, `write_graph`, `write_polylines`,
  `write_lines` writers that this package calls under the hood.
- The fragment-index encoding (v0.6) that lets multiple objects share
  vertices within a chunk.
- The `ZVWriter` post-hoc mutation handle that
  `compute_connected_components(..., write_back=True)` and the
  mesh-attribute write-back paths route through.

This package is a layer on top that adds:

- **File-format wrappers** — every ingest module wraps a specific
  third-party reader (`laspy`, `nibabel`, `plyfile`, `networkx`,
  `trx-python`) or a pure-Python parser (OBJ, STL, SWC) and feeds the
  result into the appropriate ZVF writer.
- **Enrichment helpers** — per-vertex / per-object attributes derived
  during ingest (vertex counts, polyline lengths, k-nearest-neighbour
  distances, Strahler order, …). See [Enrichments](../enrichments.md).
- **Streaming algorithms** — graph search, connected components,
  community detection, mesh summary, vertex normals, mean curvature,
  closest-point and ray queries. None materialise the whole store in
  memory; all walk the chunk grid (or the object manifests) on demand.
- **Header round-trip** — format-specific metadata (TRK affines, SWC
  comments, OBJ object names, CSV normalisation parameters, …) is
  preserved in a sidecar `headers/<format>/` group so re-exporting
  recovers it. See [Headers](../headers.md).

## Geometry types and which ingest goes with which

ZVF defines seven geometry types. Each file format you can ingest maps
to exactly one:

| Geometry type | Ingest modules |
| --- | --- |
| point cloud | `ingest_csv`, `ingest_las`, `ingest_ply` |
| line | `ingest_lines_csv` |
| polyline / streamline | `ingest_tck`, `ingest_trk`, `ingest_trx` |
| graph | `ingest_edgelist`, `ingest_graphml` |
| skeleton (tree) | `ingest_swc` |
| mesh | `ingest_obj`, `ingest_stl`, `ingest_ply` *(mesh PLY future)* |

The matching export modules cover the formats most users round-trip back
to: CSV, PLY, TRK, TRX, SWC, OBJ.

## Chunk shape and bin shape

Both ingest writers take a required `chunk_shape` and an optional
`bin_shape`. These belong to the underlying ZVF spec — see the parent
package's `concepts <https://zarr-vectors.readthedocs.io/en/latest/getting_started/concepts.html>`__
page for the full mental model. Short version:

- `chunk_shape` = physical extent of one Zarr chunk, in store coords.
  Controls how I/O is parallelised and how spatial queries shard.
- `bin_shape` = optional supervoxel grid inside each chunk. When set, it
  becomes the spatial-hash bucket size used for some accelerators.

For most ingests, set `chunk_shape` to roughly the working spatial
resolution and leave `bin_shape` unset; the writers default it
sensibly.

## What lives where on disk

A ZVF store written by an ingest module looks like:

```text
my_store.zv/
├── zarr.json                 # Zarr v3 group marker
├── .zattrs                   # ZVF root metadata (geometry type, dims, …)
├── 0/                        # resolution level 0 (only level for v0 writes)
│   ├── vertices/             # per-chunk vertex blobs
│   ├── vertex_fragments/     # per-chunk fragment index (v0.6)
│   ├── links/0/              # intra-chunk edges or faces
│   ├── link_fragments/       # per-chunk link fragment index
│   ├── cross_chunk_links/0/  # edges/faces that span chunk boundaries
│   ├── attributes/<name>/    # per-vertex attributes (e.g. knn_distance)
│   ├── object_attributes/    # per-object attributes (e.g. vertex_count, length)
│   ├── object_index/         # object manifests (which fragments per object)
│   └── headers/<format>/     # preserved format-specific metadata
└── ...
```

You don't need to touch any of this directly — the ingest, algorithm,
and export functions handle every read/write through the
`zarr-vectors-py` public API.

## See also

- [Quickstart](quickstart.md) — concrete code for each workflow.
- [Ingest workflows](../ingest/index.md)
- [Algorithms](../algorithms/index.md)
- [Export workflows](../export/index.md)
- Parent package: [zarr-vectors concepts](https://zarr-vectors.readthedocs.io/en/latest/getting_started/concepts.html)
