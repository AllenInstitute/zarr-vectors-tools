# Headers

Format-specific metadata that doesn't fit into the ZVF geometry model
itself — TRK voxel-to-RAS affines, SWC `coordinate_space` comments,
OBJ object-name lists, CSV normalisation parameters — is preserved
alongside the data so the matching `export_*` can recover the original
file's metadata.

## Storage

Headers live under `/headers/<format>/.zattrs` inside the store as
JSON-serialisable dicts. One header per format, never more.

```text
my_store.zv/
├── 0/                 # resolution level 0
│   └── ...
└── headers/
    ├── trk/.zattrs    # TRKHeader serialised
    ├── swc/.zattrs    # SWCHeader serialised
    └── obj/.zattrs    # OBJHeader serialised
```

## `HeaderRegistry`

The registry is the public API for reading, writing, listing, and
removing headers. The ingest functions write headers automatically;
you generally only need the registry to inspect what was preserved or
to clear it.

```python
from zarr_vectors_tools.headers import HeaderRegistry

reg = HeaderRegistry("tracts.zv")

reg.available_formats      # ['trk']
reg.has("trk")             # True
trk = reg.get("trk")       # → TRKHeader
trk.affine                 # (4, 4) np.ndarray, or None
trk.scalar_names           # ['fa', 'md']

reg.remove("trk")          # drop the header
```

Pass either a store path or an already-open `FsGroup` root handle to
the constructor; the path form opens the store with `mode="r+"`.

## Format-specific header classes

All header classes inherit from `Header` and implement `to_dict()` /
`from_dict()` for JSON serialisation.

### `TRKHeader`

Captures the TrackVis TRK file header so it can be reconstructed by
`export_trk`:

| Field | Type | Meaning |
| --- | --- | --- |
| `voxel_size` | `(float, float, float)` | TRK voxel size |
| `dimensions` | `(int, int, int)` | TRK image dimensions |
| `vox_to_ras` | `list[float]` | flattened 4×4 affine (16 floats) |
| `voxel_order` | `str` | e.g. `"LAS"` |
| `n_scalars`, `scalar_names` | int, list[str] | per-vertex data field names |
| `n_properties`, `property_names` | int, list[str] | per-streamline data field names |
| `n_count` | int | original streamline count |

`TRKHeader.affine` is a convenience `@property` that returns the
`(4, 4)` numpy array, or `None` if no affine was preserved.

### `NIfTIHeader`

Spatial-reference header for any geometry type with a known coordinate
system: affine, dimensions, voxel sizes, qform/sform codes,
`xyzt_units`. Useful when ingesting into a NIfTI-derived coordinate
frame.

### `SWCHeader`

`comment_lines` (the `#`-prefixed text at the top of an SWC file),
`coordinate_space` (e.g. `"RAS micron"`), `scaling` (per-axis float
triple). Restored as comments at the top of any `export_swc` output.

### `LASHeader`

`version`, `point_format`, `point_count`, `scale`, `offset`,
`min_bound`, `max_bound`, `crs_wkt`. Stored on ingest so a future
`export_las` (not in v0) can reconstruct the original LAS header.

### `OBJHeader`

`mtllib`, `object_names`, `group_names`. When `ingest_obj(...,
auto_object_id=True)` runs, the parsed object names from `o <name>`
directives are stored here so `export_obj` can round-trip them as
matching `o` directives in the output.

### `CSVHeader`

`column_names`, `delimiter`, `position_columns`, `attribute_columns`,
`has_header_row`. Also stores `normalise_offset` and `normalise_scale`
when `ingest_csv(..., normalise=True)` was used, so `export_csv` can
invert the normalisation.

### `GraphHeader`

Summary statistics for the source graph: `node_count`, `edge_count`,
`is_directed`, `mean_degree`, `n_components`, `largest_component_size`.
Written by `ingest_edgelist` and `ingest_graphml` when
`compute_summary=True`.

## See also

- [Ingest workflows](ingest/index.md) — each format page notes whether
  `preserve_header` is on or off by default.
- [Export workflows](export/index.md) — every export reads the
  matching header automatically.
