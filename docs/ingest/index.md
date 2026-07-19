# Ingest workflows

Each ingest function reads one source format and writes a Zarr Vectors
store. There are no re-exports — `zarr_vectors_tools/ingest/__init__.py`
is empty, so every import names its module in full.

```python
# Always import from the module, never from the package.
from zarr_vectors_tools.ingest.csv_points import ingest_csv

result = ingest_csv(
    "transcripts.csv",          # source file
    "transcripts.zv",           # path for the new store (must not exist)
    (50.0, 50.0, 50.0),         # chunk_shape — required, positional
    bin_shape=None,             # optional supervoxel bin size
    # ... format-specific options ...
)
result["vertex_count"]          # summary dict from the underlying writer
```

Two entry points break that shape: `ingest_edgelist` takes an edge CSV
*and* a node CSV before `output_path`, and `ingest_trk_parallel` takes no
`chunk_shape` at all — it derives the grid from `num_chunks`.

The returned dict is whatever the `zarr-vectors` writer produced
(`write_points`, `write_lines`, `write_polylines`, `write_graph`,
`write_mesh`), plus enrichment counters such as `dropped_by_length`,
`dropped_na`, or `dropped_duplicates`.

## Format matrix

Generated from `FORMAT_REGISTRY` in
`zarr_vectors_tools/cli/_args.py` — the single place that maps a
`--format` key to extensions, module, entry function, and extra.

| Format | CLI `--format` key | Ingest function | ZVF geometry | Extra required |
| --- | --- | --- | --- | --- |
| CSV / XYZ point cloud | `csv` *(`.csv`, `.xyz`)* | `zarr_vectors_tools.ingest.csv_points.ingest_csv` | points | none |
| LAS / LAZ | `las` *(`.las`, `.laz`)* | `zarr_vectors_tools.ingest.las.ingest_las` | points | `las` |
| PLY (points) | `ply` *(`.ply`)* | `zarr_vectors_tools.ingest.ply.ingest_ply` | points | `ply` |
| CSV line segments | `lines` *(no extension)* | `zarr_vectors_tools.ingest.lines.ingest_lines_csv` | lines | none |
| TrackVis TRK | `trk` *(`.trk`)* | `zarr_vectors_tools.ingest.trk_parallel.ingest_trk_parallel` | streamlines | `parallel` |
| TRX | `trx` *(`.trx`)* | `zarr_vectors_tools.ingest.trx.ingest_trx` | streamlines | `streamlines` |
| MRtrix TCK | `tck` *(`.tck`)* | `zarr_vectors_tools.ingest.tck.ingest_tck` | streamlines | `streamlines` |
| SWC | `swc` *(`.swc`)* | `zarr_vectors_tools.ingest.swc.ingest_swc` | skeleton | none |
| Edge-list CSV pair | `edgelist` *(no extension)* | `zarr_vectors_tools.ingest.edgelist.ingest_edgelist` | graph | `graph` |
| GraphML | `graphml` *(`.graphml`)* | `zarr_vectors_tools.ingest.graphml.ingest_graphml` | graph | `graph` |
| Wavefront OBJ | `obj` *(`.obj`)* | `zarr_vectors_tools.ingest.obj.ingest_obj` | mesh | none |
| STL | `stl` *(`.stl`)* | `zarr_vectors_tools.ingest.stl.ingest_stl` | mesh | none |

Install an extra with `pip install "zarr-vectors-tools[las]"`, or
`[all]` for the lot. Python 3.11 or newer is required.

:::{warning}
`lines` and `edgelist` register **no file extensions**, so they can never
be auto-detected — the CLI requires an explicit `--format`. A `.csv`
input with no `--format` resolves to `csv`, i.e. a point cloud, silently
and successfully. Pass `--format lines` or `--format edgelist` when the
CSV is not points.
:::

`trk` is the only registry entry with `inline_pyramid=True`: the ingest
builds the multiscale pyramid itself rather than leaving it to a
follow-up `build_pyramid` call. See
[Tractography at scale](tractography_at_scale.md).

## Ingests not in the CLI registry

Three entry points are reachable from Python only:

| Function | Source | Extra |
| --- | --- | --- |
| `zarr_vectors_tools.ingest.precomputed_skeletons.run_ingest` | Precomputed skeleton layer with a `spatial_index` (`.frags`) | `precomputed` |
| `zarr_vectors_tools.ingest.precomputed_plain_skeletons.run_ingest_plain` | Precomputed skeleton layer with no spatial index | `precomputed` |
| `zarr_vectors_tools.ingest.trk.ingest_trk` | TRK, serial whole-file read via `nibabel` | `streamlines` |

`ingest_trk` is the serial sibling of `ingest_trk_parallel`. `--format trk`
on the CLI always routes to the parallel path; reach for the serial one
in Python when the file is small enough to hold in RAM and you want the
`nibabel` reader's `data_per_point` / `data_per_streamline` handling.

## Geometry pages

- [Point clouds](point_clouds.md) — CSV/XYZ, LAS/LAZ, PLY
- [Lines](lines.md) — line-segment CSV
- [Tractography](tractography.md) — TCK, TRK, TRX
- [Tractography at scale](tractography_at_scale.md) — the parallel TRK pipeline
- [Skeletons](skeletons.md) — SWC
- [Skeletons in EM](em_skeletons.md) — precomputed / CloudVolume sources
- [Graphs](graphs.md) — edge-list CSV, GraphML
- [Meshes](meshes.md) — OBJ, STL

Attribute names produced by the enrichment options are listed in full at
[Enrichments](../enrichments.md).

## Headers and round-trip

Several formats carry metadata that is not part of the geometry: TRK
voxel-to-RAS affines, SWC comment lines, OBJ object names, CSV
normalisation offset and scale. These are written to `/headers/<format>/`
so the matching `export_*` can recover them. Header preservation is
best-effort — a failure there never fails the ingest. See
[Headers](../headers.md).

## See also

- [Quickstart](../getting_started/quickstart.md)
- [The `zvtools` CLI](../getting_started/cli.md)
- [Choosing chunk and bin shapes](../how_to/choose_chunk_and_bin.md)
- [Enrichments](../enrichments.md)
