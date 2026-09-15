# Export workflows

Every function in `zarr_vectors_tools.convert.export` reads a Zarr Vectors store
and writes one file format. They all share the same call shape and all
return a summary dict.

From the shell they are reached through `zvtools convert` with a store as
the input:

```bash
zvtools convert cloud.zv cloud.ply --attribute intensity
zvtools convert tracts.zv cst.trk --level 2 --group-id 4
```

See [the CLI reference](../getting_started/cli.md#export-options) for the
flags. The Python API below is what the command calls, and takes filters
the command does not expose (`chunks=`, and the format-specific options).

```python
# The subpackage does NOT re-export; always import the module directly.
from zarr_vectors_tools.convert.export.ply import export_ply

result = export_ply(
    "cloud.zv",             # store_path — source Zarr Vectors store
    "cloud.ply",            # output_path — file to write
    level=0,                # resolution level to read from
    bbox=([-100.0] * 3, [100.0] * 3),   # optional filters, AND-ed together
    object_ids=[3, 5],
)
result["vertex_count"]      # every exporter returns a summary dict
```

:::{warning}
`zarr_vectors_tools/convert/export/__init__.py` is empty. `from
zarr_vectors_tools.convert.export import export_ply` raises `ImportError` —
import from the module (`zarr_vectors_tools.convert.export.ply`) every time.
:::

## Format matrix

| Source geometry | Export function | Output format | Supported filters | Extra required |
| --- | --- | --- | --- | --- |
| points | `zarr_vectors_tools.convert.export.csv_points.export_csv` | CSV / XYZ text | `bbox`, `object_ids`, `chunks` | none |
| points | `zarr_vectors_tools.convert.export.ply.export_ply` | PLY (binary or ASCII) | `bbox`, `object_ids`, `chunks` | `ply` |
| points | `zarr_vectors_tools.convert.export.h5ad.export_h5ad` | AnnData `.h5ad` | `bbox`, `chunks` *(see note)* | `h5ad` |
| polylines | `zarr_vectors_tools.convert.export.trk.export_trk` | TrackVis TRK | `object_ids`, `group_ids`, `chunks`, `attribute_names`, `object_attribute_names` | `trk` |
| polylines | `zarr_vectors_tools.convert.export.trx.export_trx` | TRX | `object_ids`, `group_ids`, `chunks`, `attribute_names`, `object_attribute_names` | `trx` |
| graphs (trees) | `zarr_vectors_tools.convert.export.swc.export_swc` | SWC, or one file per object | `chunks`, `object_ids` | none |
| surfaces | `zarr_vectors_tools.convert.export.gifti.export_gifti` | GIFTI directory (`.surf` / `.shape` / `.func` / `.label.gii`) | `hemispheres`, `surfaces`, `attribute_names` | `surfaces` |
| skeletons, graphs | `zarr_vectors_tools.convert.export.precomputed.export_precomputed_skeletons` | Neuroglancer precomputed skeleton layer (directory or bucket) | `object_ids`, `segment_ids`, `attribute_names`, `object_attribute_names` | `precomputed` |
| meshes | `zarr_vectors_tools.convert.export.precomputed.export_precomputed_meshes` | Neuroglancer legacy mesh layer (bucket; not a local directory on Windows) | `object_ids`, `segment_ids`, `object_attribute_names` | `precomputed` |
| meshes | `zarr_vectors_tools.convert.export.obj.export_obj` | Wavefront OBJ | `bbox`, `object_ids`, `chunks` | none |

Install an extra with `pip install "zarr-vectors-tools[trk]"`.

Filters AND together: `bbox=(...)` *and* `object_ids=[3, 5]` means
"objects 3 and 5, intersected with the bounding box". Passing `None`
(the default) disables that filter.

:::{note}
`export_h5ad` accepts `object_ids` only alongside `attribute_names=[]`.
Core's object-filtered read path drops vertex attributes, which for an
`.h5ad` would mean silently writing an empty `obs`; it raises instead.
See [Single-cell and spatial omics](single_cell.md).
:::

## Exporting from a coarser level

`level=` picks which pyramid level the exporter reads. Level `0` is full
resolution; every level above it is progressively decimated, so
`level=2` writes a much smaller file with the same spatial extent.

```python
from zarr_vectors_tools.convert.export.obj import export_obj

# Full-resolution mesh — the real artefact, potentially huge.
export_obj("model.zv", "model_full.obj", level=0)

# Level 2 of the pyramid — a decimated preview, seconds instead of minutes.
summary = export_obj("model.zv", "model_preview.obj", level=2)
summary["face_count"]   # far lower than the level-0 count
```

No decimation happens at export time — the coarsening was done once when
the pyramid was built, so a coarse export is cheap to read and cheap to
write. Requesting a `level` that the store does not have raises
`ExportError`. See [multiresolution](../multiresolution/index.md) for how
levels are built.

## The `chunks` filter

`chunks` takes a whitelist of chunk-coordinate tuples and keeps only data
physically stored in those chunks. It is the cheapest filter — it skips
whole chunks rather than reading and masking them — but it selects on
storage layout, not on geometry.

:::{warning}
On `export_trk` and `export_trx` the `chunks` filter selects at the
**segment** level. A polyline that crosses a chunk boundary is stored as
several segments, and each surviving contiguous run is written as its own
streamline. The returned `streamline_count` can therefore **exceed** the
number of source objects. If you need whole objects, filter with
`object_ids` instead.
:::

`export_swc` and `export_obj` have the mirror-image caveat: an edge or
face spanning a listed and an unlisted chunk is dropped, which can split
one tree or surface into several disconnected pieces.

## Memory

`export_csv` and `export_ply` read and write a batch of chunks at a time
(`vertex_budget`, default four million points), so memory holds one batch
and the file is identical to a whole-level read. The streamline, mesh,
AnnData and whole-level SWC exporters still read the whole level before
writing. Size the machine for the level you export, or export a coarser
level or a subset (`object_ids`, `bbox`). See
[Which paths are memory-bounded](../how_to/large_scale_pipelines.md#which-paths-are-memory-bounded).

## Format headers

Ingest preserves format-specific metadata that the Zarr Vectors geometry model
cannot hold — TRK affines, SWC comment lines, OBJ object names, CSV
normalisation parameters — under `/headers/<format>/` on the store.

The streamline and surface exporters read those headers back themselves:

| Exporter | Header it reads | What it restores |
| --- | --- | --- |
| `export_trk` | `trk`, or `trx` | reference image (`vox_to_ras`, voxel size, dimensions, voxel order), and the stored coordinate space |
| `export_trx` | `trx`, or `trk` | `VOXEL_TO_RASMM`, `DIMENSIONS`; voxmm positions are converted to RAS |
| `export_gifti` | `surface` | hemispheres, surface names, label tables, dataspace |

The other exporters do not consult `/headers/`; read a header back with
`HeaderRegistry` when you need its values.

See [headers](../headers.md) for the registry API and the per-format
dataclasses.

## Geometry pages

- [Point clouds](point_clouds.md) — CSV, PLY
- [Streamlines](streamlines.md) — TRK, TRX
- [Skeletons](skeletons.md) — SWC
- [Meshes](meshes.md) — OBJ
- [Cortical surfaces](surfaces.md) — GIFTI

## See also

- [Ingest workflows](../ingest/index.md) — the symmetric direction.
- [Headers](../headers.md) — what ingest preserved and how to read it back.
- [Multiresolution](../multiresolution/index.md) — building the levels that `level=` selects.
- [Quickstart](../getting_started/quickstart.md) — end-to-end round trip.
