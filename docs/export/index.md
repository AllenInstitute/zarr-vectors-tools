# Export workflows

Every function in `zarr_vectors_tools.export` reads a Zarr Vectors store
and writes one file format. They all share the same call shape and all
return a summary dict.

```python
# The subpackage does NOT re-export; always import the module directly.
from zarr_vectors_tools.export.ply import export_ply

result = export_ply(
    "cloud.zv",             # store_path — source ZVF store
    "cloud.ply",            # output_path — file to write
    level=0,                # resolution level to read from
    bbox=([-100.0] * 3, [100.0] * 3),   # optional filters, AND-ed together
    object_ids=[3, 5],
)
result["vertex_count"]      # every exporter returns a summary dict
```

:::{warning}
`zarr_vectors_tools/export/__init__.py` is empty. `from
zarr_vectors_tools.export import export_ply` raises `ImportError` —
import from the module (`zarr_vectors_tools.export.ply`) every time.
:::

## Format matrix

| Source geometry | Export function | Output format | Supported filters | Extra required |
| --- | --- | --- | --- | --- |
| points | `zarr_vectors_tools.export.csv_points.export_csv` | CSV / XYZ text | `bbox`, `object_ids`, `chunks` | none |
| points | `zarr_vectors_tools.export.ply.export_ply` | PLY (binary or ASCII) | `bbox`, `object_ids`, `chunks` | `ply` |
| polylines | `zarr_vectors_tools.export.trk.export_trk` | TrackVis TRK | `object_ids`, `group_ids`, `chunks` | `trk` |
| polylines | `zarr_vectors_tools.export.trx.export_trx` | TRX | `object_ids`, `group_ids`, `chunks` | `trx` |
| graphs (trees) | `zarr_vectors_tools.export.swc.export_swc` | SWC | `chunks` | none |
| meshes | `zarr_vectors_tools.export.obj.export_obj` | Wavefront OBJ | `bbox`, `object_ids`, `chunks` | none |

Install an extra with `pip install "zarr-vectors-tools[trk]"`.

Filters AND together: `bbox=(...)` *and* `object_ids=[3, 5]` means
"objects 3 and 5, intersected with the bounding box". Passing `None`
(the default) disables that filter.

## Exporting from a coarser level

`level=` picks which pyramid level the exporter reads. Level `0` is full
resolution; every level above it is progressively decimated, so
`level=2` writes a much smaller file with the same spatial extent.

```python
from zarr_vectors_tools.export.obj import export_obj

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

## Format headers

Ingest preserves format-specific metadata that the ZVF geometry model
cannot hold — TRK affines, SWC comment lines, OBJ object names, CSV
normalisation parameters — under `/headers/<format>/` on the store.

:::{warning}
The exporters do **not** read `/headers/` themselves. Nothing in
`zarr_vectors_tools.export` touches `HeaderRegistry`, so metadata does
not round-trip automatically: read the header back yourself and pass the
values in as arguments where the exporter accepts them (currently only
`export_trk(affine=...)`).
:::

```python
from zarr_vectors_tools.headers import HeaderRegistry
from zarr_vectors_tools.export.trk import export_trk

reg = HeaderRegistry("tracts.zv")
header = reg.get("trk")                 # TRKHeader written at ingest time
export_trk("tracts.zv", "out.trk", affine=header.affine)
```

See [headers](../headers.md) for the registry API and the per-format
dataclasses.

## Geometry pages

- [Point clouds](point_clouds.md) — CSV, PLY
- [Streamlines](streamlines.md) — TRK, TRX
- [Skeletons](skeletons.md) — SWC
- [Meshes](meshes.md) — OBJ

## See also

- [Ingest workflows](../ingest/index.md) — the symmetric direction.
- [Headers](../headers.md) — what ingest preserved and how to read it back.
- [Multiresolution](../multiresolution/index.md) — building the levels that `level=` selects.
- [Quickstart](../getting_started/quickstart.md) — end-to-end round trip.
