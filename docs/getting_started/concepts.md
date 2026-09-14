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

:::{important}
This page describes **this package's** workflows only. The format they
operate on — stores, chunks, bins, fragments, links, the object model — is
owned and documented by
[`zarr-vectors-py`](https://zarr-vectors-py.readthedocs.io/en/latest).
Start at {zvpy}`Core concepts <getting_started/concepts.html>`, and see
[How this package relates to `zarr-vectors-py`](zarr_vectors.md) for the
division of labour.
:::

## Relationship to `zarr-vectors-py`

The format itself, the chunk encoding, the spatial index, links, and lazy
access all live in the parent package and are documented there. What this
package consumes from it:

- The store-creating writers and per-chunk write helpers on the supported
  `zarr_vectors.building` surface ({zvpy}`the building API <api/building.html>`), which every
  ingest path and every write-back path here routes through.
- The `zarr_vectors.api` surface ({zvpy}`the data API <api/api.html>`) for reading.
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
(`spatial_coverage`, `length`, `attribute`, `point_thinning`) into the
parent package's strategy registry, through its supported
`zarr_vectors.building.register_coarsen_strategy` and
`register_selection_strategy` entry points.

That is what lets the parent package dispatch into tools **without taking
a dependency on it**. Ask what an installation has with
`zarr_vectors.coarsen_methods()`:

```python
import zarr_vectors as zv

zv.coarsen_methods()          # ('per_object',)

import zarr_vectors_tools     # noqa: F401

zv.coarsen_methods()          # ('per_object', 'polyline', 'skeleton')
```

The registration degrades silently on a core too old to have the
registry.

One thing is deliberately *not* wired up: pyramid refresh. The parent
package's editing path (`ds.editing(refresh_pyramid=True)`) uses its own
basic refresher. To get the rich re-coarsening after an edit, call
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
Both are format-level concepts owned by the parent package — what they
mean, how they interact, and how to size them are specified at
{zvpy}`chunk shape <spec/chunking/chunk_shape.html>`,
{zvpy}`bin shape <spec/chunking/bin_shape.html>` and
{zvpy}`chunk versus bin <spec/chunking/chunk_vs_bin.html>`, with the sizing reasoning at
{zvpy}`Choosing chunk and bin shape <how_to/choose_chunk_and_bin.html>`.

What is specific to this package is only how those values are *supplied*
to an ingest, and the one path that derives them for you:

- Every ingest function takes `chunk_shape` explicitly, and `bin_shape`
  optionally.
- The `trk` path is the exception: it takes `--num-chunks` (a target chunk
  *count*) instead of an explicit shape, and derives the shape from the
  data bounds.

See [Choosing chunk and bin shape](../how_to/choose_chunk_and_bin.md) for
the flags this package adds on top.

## What lives where on disk

The on-disk layout is specified by the parent package, not here. Read it
at {zvpy}`Directory structure <spec/layout/directory_structure.html>`, with the two metadata
documents at {zvpy}`root metadata <spec/layout/root_metadata.html>` and
{zvpy}`level metadata <spec/layout/level_groups.html>`, and connectivity at
{zvpy}`Links <spec/object_model/links.html>`.

This release targets on-disk format version {{ zv_version }}.

You should not need to touch any of it directly. The ingest, pyramid,
algorithm and export functions in this package handle every read and write
through the supported `zarr_vectors.api` and `zarr_vectors.building`
surfaces.

:::{warning}
One layout fact is worth repeating here because it changes how *algorithm*
code in this package must be written: at format {{ zv_version }} all
connectivity lives in a single `links/` family, so a read returns the whole
family. The old idiom of reading per-chunk links and then adding
cross-chunk links counts every intra-chunk edge twice. See the double-count
trap in [Algorithms](../algorithms/index.md).
:::

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

- [How this package relates to `zarr-vectors-py`](zarr_vectors.md) — the division of labour.
- [Quickstart](quickstart.md) — concrete code for each workflow.
- [The `zvtools` CLI](cli.md)
- [Modules](../modules/index.md) — what each subpackage owns.
- [Coarsening versus sparsity](../multiresolution/concepts.md)
- **Parent package:** {zvpy}`Core concepts <getting_started/concepts.html>` — the format itself.
- **Parent package:** {zvpy}`the specification <spec/index.html>` — the normative specification.
