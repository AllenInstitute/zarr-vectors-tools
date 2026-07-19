# Concepts

`zarr-vectors-tools` is organised around four workflows that compose over
a shared on-disk format:

```text
┌──────────┐   ┌────────────────┐   ┌──────────────┐   ┌──────────┐
│ source   │──▶│ ingest_<fmt>() │──▶│ Zarr Vectors │──▶│ export_  │──▶ output
│ file     │   │ + enrichments  │   │   store      │   │  <fmt>() │   file
└──────────┘   └────────────────┘   └──────┬───────┘   └──────────┘
                                           │ ▲
                            ┌──────────────┘ └──────────────┐
                            ▼                               │
                  ┌───────────────────┐        ┌────────────────────────┐
                  │ build_pyramid()   │        │ compute_*() algorithm  │
                  │ coarser levels    │        │ (may write attributes  │
                  └───────────────────┘        │  back to the store)    │
                                               └────────────────────────┘
```

Every public entry point in this package either:

1. **Reads** a file format and writes a store (`zarr_vectors_tools.ingest`),
2. **Coarsens** a store into additional resolution levels
   (`zarr_vectors_tools.multiresolution`),
3. **Computes** over a store and optionally writes results back
   (`zarr_vectors_tools.algorithms`), or
4. **Reads** a store and writes a file format (`zarr_vectors_tools.export`).

New to the format itself? Read
[Getting started with Zarr Vectors](zarr_vectors.md) first.

## Relationship to `zarr-vectors-py`

The on-disk format, the chunk encoding, the spatial index, links, and
lazy access all live in
[`zarr-vectors`](https://zarr-vectors-py.readthedocs.io/en/latest). That
package provides:

- The `write_points`, `write_mesh`, `write_graph`, `write_polylines` and
  `write_lines` writers that this package calls under the hood.
- The fragment encoding that lets several objects share a chunk.
- The `ZVWriter` mutation handle that
  `compute_connected_components(..., write_back=True)` and the
  mesh-attribute write-back paths route through.
- A **basic** multiresolution layer — `per_object` binning plus `random`
  object selection — and a plug-in strategy registry.

This package adds:

- **File-format wrappers** — each ingest module wraps a third-party reader
  (`laspy`, `nibabel`, `plyfile`, `networkx`, `trx-python`,
  `cloud-volume`) or a pure-Python parser (OBJ, STL, SWC) and feeds the
  result into the appropriate writer.
- **Enrichment helpers** — per-vertex and per-object attributes derived
  during ingest: vertex counts, polyline lengths, kNN distances, Strahler
  order. See [Enrichments](../enrichments.md).
- **The rich multiresolution layer** — skeleton and polyline coarseners,
  spatial-coverage / length / attribute / point-thinning selection,
  cross-level links. See [Multiresolution](../multiresolution/index.md).
- **Streaming algorithms** — graph search, connected components, community
  detection, mesh summary, vertex normals, mean curvature, closest-point
  and ray queries. None materialise the whole store; all walk the chunk
  grid or the object manifests on demand.
- **The `zvtools` CLI**. See [The `zvtools` CLI](cli.md).

### How the multiresolution split works

Importing `zarr_vectors_tools` has a side effect: it registers this
package's coarseners (`skeleton`, `polyline`) and selectors
(`spatial_coverage`, `length`, `attribute`, `point_thinning`) into
`zarr_vectors.multiresolution.registry`.

That is what lets core dispatch into tools —
`zarr_vectors.multiresolution.coarsen_level(method="skeleton")` works
once tools has been imported — **without core taking a dependency on
tools**. The registration degrades silently on a core too old to have the
registry.

One thing is deliberately *not* wired up: pyramid refresh.
`EditSession(refresh_pyramid=...)` uses core's basic refresher. To get the
rich re-coarsening after an edit, call
[`rebuild_pyramid_from_level`](../multiresolution/refresh.md) yourself.

## Geometry types and which ingest goes with which

| Geometry type | Ingest functions |
| --- | --- |
| point cloud | `ingest_csv`, `ingest_las`, `ingest_ply` |
| line | `ingest_lines_csv` |
| polyline / streamline | `ingest_tck`, `ingest_trk`, `ingest_trk_parallel`, `ingest_trx` |
| graph | `ingest_edgelist`, `ingest_graphml` |
| skeleton (tree) | `ingest_swc`, `run_ingest`, `run_ingest_plain` |
| mesh | `ingest_obj`, `ingest_stl` |

The export modules cover the formats most users round-trip back to: CSV,
PLY, TRK, TRX, SWC, OBJ.

:::{warning}
Neither `zarr_vectors_tools.ingest` nor `zarr_vectors_tools.export` has
re-exports — their `__init__.py` files are empty. Always import from the
concrete module:

```python
from zarr_vectors_tools.ingest.csv_points import ingest_csv   # correct
from zarr_vectors_tools.ingest import ingest_csv              # ImportError
```

`zarr_vectors_tools.algorithms` **does** re-export, so
`from zarr_vectors_tools.algorithms import bfs_distances` is fine.
:::

## Chunk shape and bin shape

Ingest writers take a required `chunk_shape` and an optional `bin_shape`.

- `chunk_shape` — the physical extent of one Zarr chunk in store
  coordinates. Controls how I/O is parallelised and how spatial queries
  shard.
- `bin_shape` — an optional supervoxel grid *inside* each chunk, used as
  the spatial-hash bucket size by some accelerators.

For most ingests, set `chunk_shape` to roughly the working spatial
resolution and leave `bin_shape` unset. See
[Choosing chunk and bin shape](../how_to/choose_chunk_and_bin.md).

The `trk` path is the exception: it takes `--num-chunks` (a target chunk
*count*) instead of an explicit shape, and derives the shape from the
data bounds.

## What lives where on disk

A store written at format version {{ zvf_version }} looks like:

```text
my_store.zarrvectors/
├── zarr.json                      # Zarr v3 group marker
├── .zattrs                        # root metadata (geometry types, bounds, axes, …)
├── 0/                             # resolution level 0
│   ├── vertices/                  # per-chunk vertex blobs
│   ├── vertex_fragments/          # per-chunk fragment index
│   ├── links/                     # ALL connectivity — one family
│   │   ├── 0/                     # delta 0: offsets groups live below
│   │   │   ├── <offsets>/         # one array per offsets segment
│   │   │   └── ...
│   │   └── ...
│   ├── link_fragments/
│   ├── attributes/<name>/         # per-vertex attributes
│   ├── object_attributes/<name>/  # per-object attributes
│   ├── object_index/              # object manifests (which fragments per object)
│   └── fragment_attributes/
├── 1/                             # coarser level, same layout
├── 2/
└── headers/<format>/              # preserved format-specific metadata
```

:::{warning}
**There is no `cross_chunk_links/` group at format {{ zvf_version }}.**
Connectivity was merged into a single family under
`links/<delta>/<offsets>/`, where an intra-chunk link is simply one whose
offsets are all zero. If you are porting code or reading older notes that
mention `cross_chunk_links/`, see the double-count trap in
[Algorithms](../algorithms/index.md) before you trust it.

Note also that `links/<delta>` is a **group**, not an array. Naming the
group where an array is expected fails *silently* — `list_chunks` returns
`[]` and prefetch quietly does nothing.
:::

You should not need to touch any of this directly. The ingest, pyramid,
algorithm and export functions handle every read and write through the
`zarr-vectors` public API.

## Headers

Several formats carry metadata that is not part of the geometry: TRK
voxel-to-RAS affines, SWC `coordinate_space` comments, OBJ object names,
CSV normalisation parameters. Ingest preserves these into a
`headers/<format>/` group on the store.

:::{warning}
Recovering a header on export is **manual**. No exporter reads
`headers/` automatically — in particular `export_trk(affine=None)` writes
an **identity** affine, not the affine that was stored at ingest, which
silently produces a misregistered tractogram. Read the header yourself
and pass it in. See [Headers](../headers.md) and
[Exporting streamlines](../export/streamlines.md).
:::

## See also

- [Getting started with Zarr Vectors](zarr_vectors.md) — the format primer.
- [Quickstart](quickstart.md) — concrete code for each workflow.
- [The `zvtools` CLI](cli.md)
- [Modules](../modules/index.md) — what each subpackage owns.
- [Coarsening versus sparsity](../multiresolution/concepts.md)
- Parent package: [zarr-vectors concepts](https://zarr-vectors-py.readthedocs.io/en/latest/getting_started/concepts.html)
