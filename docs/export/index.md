# Export workflows

Every export function in `zarr_vectors_tools.export` reads a Zarr
Vectors store and writes a single file format. They all follow the same
shape:

```python
result = export_<format>(
    store_path,         # path to the source ZVF store
    output_path,        # path for the new file
    level=0,            # resolution level
    bbox=None,          # optional spatial filter
    object_ids=None,    # optional object ID filter
    chunks=None,        # optional chunk whitelist
    # ... format-specific options ...
)
```

Filters AND together: `bbox=([-10]*3, [10]*3)` *and* `object_ids=[3, 5]`
means "objects 3 and 5, intersected with the bounding box". When
`chunks` is supplied, the chunk filter is applied at the segment level
for polyline / mesh exports — see the format-specific pages for what
that means in practice.

## Format matrix

| Format | Export function | Source ZVF geometry | Third-party deps |
| --- | --- | --- | --- |
| CSV / XYZ | `export_csv` | point cloud | none |
| PLY (points) | `export_ply` | point cloud | `plyfile` |
| TrackVis TRK | `export_trk` | polyline / streamline | `nibabel` |
| TRX | `export_trx` | polyline / streamline | `trx-python` |
| SWC | `export_swc` | skeleton (tree) | none |
| Wavefront OBJ | `export_obj` | mesh | none |

Geometry-specific pages:

- [Point clouds](point_clouds.md) — CSV, PLY
- [Streamlines](streamlines.md) — TRK, TRX
- [Skeletons](skeletons.md) — SWC
- [Meshes](meshes.md) — OBJ

## Header round-trip

Where a format has a preserved [`Header`](../headers.md) sitting in
`/headers/<format>/` on the store (TRK affines, SWC `coordinate_space`,
OBJ object names, CSV normalisation parameters), the matching export
reads it back automatically. No need to pass it through explicitly.

## See also

- [Ingest workflows](../ingest/index.md) — the symmetric direction.
- [Quickstart](../getting_started/quickstart.md) — end-to-end round trip.
