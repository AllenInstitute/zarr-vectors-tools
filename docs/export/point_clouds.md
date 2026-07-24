# Point clouds

Two targets for the ZVF `points` geometry: delimited text via `export_csv`
and PLY via `export_ply`.

## CSV / XYZ — `export_csv`

```python
from zarr_vectors_tools.export.csv_points import export_csv

summary = export_csv(
    "cloud.zv",                 # store_path
    "cloud.csv",                # output_path
    level=0,                    # resolution level to read
    bbox=([-100.0] * 3, [100.0] * 3),   # (low_corner, high_corner)
    object_ids=[3, 5],          # keep only these objects; AND-ed with bbox
    chunks=None,                # whitelist of chunk-coordinate tuples
    delimiter=",",              # column separator
    header=True,                # write a header row
    attribute_names=["intensity"],      # extra columns; None = positions only
)
summary["vertex_count"]         # → int
```

Position columns are named `dim0`, `dim1`, … by the number of dimensions
in the store, not `x`/`y`/`z`. An attribute is written as one column
under its own name if it is 1-D, and as `<name>_0`, `<name>_1`, … if it
has multiple channels. Values are written through `numpy.savetxt` with
`fmt="%.6f"`, so every column — including integer attributes such as
`classification` — lands as a fixed six-decimal float.

:::{note}
`attribute_names` is both the filter and the column order: an attribute
present in the store but absent from the list is not written, and a name
in the list that the store does not carry is silently skipped rather than
raising.
:::

## PLY — `export_ply`

```python
from zarr_vectors_tools.export.ply import export_ply

summary = export_ply(
    "cloud.zv",
    "cloud.ply",
    level=0,
    bbox=None,
    object_ids=None,
    chunks=None,
    attribute_names=["intensity", "colour"],
    binary=True,                # True → binary PLY, False → ASCII
)
summary["vertex_count"]
```

Needs the `ply` extra: `pip install "zarr-vectors-tools[ply]"`. The
import is attempted lazily, so a missing `plyfile` surfaces as
`ExportError` from the call rather than at import time.

Stores of three dimensions or fewer get the conventional `x`, `y`, `z`
property names; anything higher-dimensional falls back to `dim0`,
`dim1`, …. Every property — positions and attributes alike — is written
as `f4`, so a float64 store loses precision and integer attributes become
floats on the way out.

## See also

- [Export overview](index.md) — shared call shape, `level=`, and the `chunks` filter.
- [Ingest → point clouds](../ingest/point_clouds.md) — the symmetric direction.
- [Headers](../headers.md) — `CSVHeader` and `LASHeader` hold what ingest preserved.
