# Modules

`zarr-vectors-tools` is about 12 000 lines across 57 modules, organised
into six subpackages. This page is the map: what each one owns, its main
entry points, and where to read more.

| Subpackage | Owns | Narrative | API |
| --- | --- | --- | --- |
| `ingest` | file format → store | [Ingest](../ingest/index.md) | [`api/ingest`](../api/ingest.rst) |
| `multiresolution` | coarser levels, pyramids | [Multiresolution](../multiresolution/index.md) | [`api/multiresolution`](../api/multiresolution.rst) |
| `algorithms` | streaming graph and mesh compute | [Algorithms](../algorithms/index.md) | [`api/algorithms`](../api/algorithms.rst) |
| `export` | store → file format | [Export](../export/index.md) | [`api/export`](../api/export.rst) |
| `headers` | format metadata preservation | [Headers](../headers.md) | [`api/headers`](../api/headers.rst) |
| `cli` | the `zvtools` command line | [CLI](../getting_started/cli.md) | [`api/cli`](../api/cli.rst) |

---

## `zarr_vectors_tools.ingest`

Reads a source file and writes a store. Nineteen modules — one per
format, plus shared enrichment helpers.

Each ingest function takes `(input_path, output_path, chunk_shape, ...)`
and returns the underlying writer's summary dict, augmented with
enrichment counters such as `dropped_by_length` or `dropped_by_na`.

| Module | Entry point | Format |
| --- | --- | --- |
| `csv_points` | `ingest_csv` | CSV / XYZ points |
| `las` | `ingest_las` | LAS / LAZ |
| `ply` | `ingest_ply` | PLY (points) |
| `lines` | `ingest_lines_csv` | 6-column segment CSV |
| `tck` | `ingest_tck` | MRtrix TCK |
| `trk` | `ingest_trk` | TrackVis TRK (serial) |
| `trk_parallel` | `ingest_trk_parallel` | TrackVis TRK (parallel, inline pyramid) |
| `trx` | `ingest_trx` | TRX |
| `swc` | `ingest_swc` | SWC skeletons |
| `precomputed_skeletons` | `run_ingest` | precomputed `.frags` with a spatial index |
| `precomputed_plain_skeletons` | `run_ingest_plain` | precomputed, no spatial index |
| `obj` | `ingest_obj` | OBJ meshes |
| `stl` | `ingest_stl` | STL meshes (ASCII + binary) |
| `edgelist` | `ingest_edgelist` | edge-list CSV pair |
| `graphml` | `ingest_graphml` | GraphML |

Supporting modules: `_parallel` (the two executor backends),
`_point_enrichments`, `_polyline_enrichments`, `_tree_enrichments`.

:::{warning}
`ingest/__init__.py` is **empty** — there are no re-exports. Import from
the concrete module: `from zarr_vectors_tools.ingest.csv_points import
ingest_csv`.
:::

---

## `zarr_vectors_tools.multiresolution`

The largest subsystem, and the one with no counterpart in core. Builds
coarser resolution levels by reducing along two orthogonal axes —
**coarsening** (fewer vertices per object) and **sparsity** (fewer
objects). Start at [Coarsening versus
sparsity](../multiresolution/concepts.md).

| Module | Role |
| --- | --- |
| `coarsen` | orchestrator: `build_pyramid`, `coarsen_level`, coarsener registry, cross-level links |
| `object_selection` | the five sparsity strategies and `apply_sparsity` |
| `refresh` | `rebuild_pyramid_from_level` — re-coarsen after edits |
| `metanodes` | `generate_metanodes` — vertex binning with `mean｜sum｜first｜count` aggregation |
| `object_index` | `build_object_index` — the reduce step |
| `skeleton_graph` | `split_components` — vertices + edges → rooted tree pieces |
| `coarsen_implicit` | pure, I/O-free helpers for the `implicit_sequential` path |
| `_forest` | vectorised batched forest operations |
| `constants` | `COARSEN_SKELETON`, `CROSS_LINK_TASK_SHARD_AXIS` |

`strategies/` holds the per-geometry coarseners: `skeletons`,
`polylines`, `points`, `meshes`, `graphs`.

:::{note}
`_forest` is worth knowing about if you are profiling. Its
`_nearest_special_ancestor` uses path-doubling (Hillis–Steele), which is
O(log height) rather than O(height). The docstring records that a scipy
formulation benchmarked ~1.75× *slower* and naive relaxation ~10× slower.
`tests/test_forest_vectorized.py` pins it to a scalar reference.
:::

---

## `zarr_vectors_tools.algorithms`

Streaming compute over a store. Nothing here materialises the whole store
— everything walks the chunk grid or the object manifests on demand.

Unlike `ingest` and `export`, this subpackage **does** re-export, with an
`__all__` of 11 functions:

```python
from zarr_vectors_tools.algorithms import bfs_distances, compute_louvain
```

| Module | Functions |
| --- | --- |
| `graph_search` | `bfs_distances`, `shortest_path`, `build_adjacency` |
| `graph_components` | `compute_connected_components` |
| `graph_clustering` | `compute_k_core`, `compute_label_propagation`, `compute_louvain` |
| `mesh_summary` | `compute_mesh_summary` |
| `mesh_attributes` | `compute_vertex_normals`, `compute_mean_curvature` |
| `mesh_query` | `closest_point`, `cast_ray` |

Two internal modules matter to anyone writing their own algorithm:

- **`_links`** — the offsets-aware link readers. This is the single
  definition of the cross-chunk filter, and the module that documents the
  double-count trap. Use `read_cross_links` rather than composing
  `read_links` with a per-chunk loop.
- **`_chunk_neighbours`** — `neighbouring_chunk_keys`, a shim for a core
  helper that does not exist yet.

---

## `zarr_vectors_tools.export`

Store → file. Six exporters, all sharing the signature shape
`(store_path, output_path, *, level=0, ...)` and all returning a summary
dict.

| Module | Entry point | Output |
| --- | --- | --- |
| `csv_points` | `export_csv` | CSV |
| `ply` | `export_ply` | PLY |
| `trk` | `export_trk` | TrackVis TRK |
| `trx` | `export_trx` | TRX |
| `swc` | `export_swc` | SWC |
| `obj` | `export_obj` | OBJ |

`level=` selects the resolution level, so exporting from a coarser level
gives you a decimated file — often the quickest way to get a previewable
subset.

:::{warning}
`export/__init__.py` is also **empty**. And no exporter reads the
`headers/` group automatically — see [Export](../export/index.md).
:::

---

## `zarr_vectors_tools.headers`

Preserves format-specific metadata that is not part of the geometry, in a
`headers/<format>/` group on the store.

- `registry.HeaderRegistry` — `available_formats`, `has`, `get`, `add`,
  `remove`. Accepts a store path or an open group.
- `formats` — the `Header` base class plus `TRKHeader`, `NIfTIHeader`,
  `SWCHeader`, `LASHeader`, `OBJHeader`, `CSVHeader`, `GraphHeader`, and
  `header_from_dict`.

Ingest writes these; reading them back on export is manual.

---

## `zarr_vectors_tools.cli`

The `zvtools` entry point, declared as
`zvtools = "zarr_vectors_tools.cli:main"`.

| Module | Role |
| --- | --- |
| `__init__` | `build_parser`, `main`, the argparse tree, the error funnel |
| `_args` | parsers, `FORMAT_REGISTRY`, `resolve_format`, `load_ingest_func`, `executor_ctx` |
| `convert` | the `convert` subcommand |
| `pyramid` | the `pyramid`, `validate` and `info` subcommands |

`_args.FORMAT_REGISTRY` is the **authoritative** format table — extension
mapping, ingest module, and required extra all live there.

---

## Root modules

**`__init__.py`** — import side effects only. It registers this package's
coarseners and selectors into `zarr_vectors.multiresolution.registry`, so
core can dispatch into tools without depending on tools. Degrades
silently on a core too old to have the registry. It does **not** define
`__version__`; the CLI reads the version via `importlib.metadata`.

**`_manifests.py`** — `rebuild_nonempty_manifests` and
`per_chunk_array_paths`, a workaround for a core method that does not
exist yet. Its docstring documents the manifest race that makes the
parallel write protocol necessary; see [Parallel
workflows](../how_to/parallelism.md).

---

## Shims awaiting upstream

Four pieces exist only until core grows an equivalent. If you are reading
the source and wondering why something is reimplemented here, this is
why:

| Shim | Waiting on |
| --- | --- |
| `_manifests.rebuild_nonempty_manifests` | `Group.rebuild_nonempty_manifests` |
| `algorithms._chunk_neighbours.neighbouring_chunk_keys` | `zarr_vectors.spatial.chunking.neighbouring_chunk_keys` |
| `algorithms._links.list_link_cells` | a public `list_link_cells`; currently only implemented for `delta == 0` |
| `headers.registry.HeaderRegistry` | a public `HeaderRegistry` shim in `zarr-vectors-py` |

[Upstream findings](../upstream/links-merge-findings.md) is the formal
write-up of these and of the concurrency work.

## See also

- [Getting started with Zarr Vectors](../getting_started/zarr_vectors.md)
- [Concepts](../getting_started/concepts.md)
- [API reference](../api/index.rst)
