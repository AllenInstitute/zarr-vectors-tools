# Point clouds

Two export targets for the ZVF point-cloud geometry: CSV / XYZ text and
PLY (binary or ASCII).

## CSV / XYZ — `export_csv`

```python
from zarr_vectors_tools.export import export_csv

export_csv(
    store_path="cloud.zv",
    output_path="cloud.csv",
    bbox=([-100.0]*3, [100.0]*3),
    attribute_names=["intensity", "classification"],
)
```

Returns `{"vertex_count": int}`.

**Notable options**

`level`
: Resolution level to export. Defaults to `0` (full resolution).

`bbox`
: `(low_corner, high_corner)` tuple. Only vertices inside are kept.

`object_ids`
: List of object IDs to keep. AND-ed with `bbox` / `chunks`.

`chunks`
: Whitelist of chunk-coordinate tuples; only data stored in those
  chunks is exported.

`delimiter`, `header`
: Output formatting controls. `header=True` (default) writes a header
  row with `dim0/dim1/dim2` plus any attribute columns.

`attribute_names`
: List of attribute names to include as extra columns. Default `None`
  means positions only. Multi-channel attributes (e.g. `color`) expand
  to `<name>_0`, `<name>_1`, … columns.

If the source store was ingested with `ingest_csv(..., normalise=True)`,
the offset/scale in the [`CSVHeader`](../headers.md) is applied during
export so the output coordinates match the original file.

## PLY — `export_ply`

```python
from zarr_vectors_tools.export import export_ply

export_ply(
    store_path="cloud.zv",
    output_path="cloud.ply",
    attribute_names=["color", "intensity"],
    binary=True,
)
```

Requires `plyfile` (install with `pip install "zarr-vectors-tools[ply]"`).

Returns `{"vertex_count": int}`.

`binary`
: `True` (default) writes binary PLY; `False` writes ASCII.

All other options (`level`, `bbox`, `object_ids`, `chunks`,
`attribute_names`) match `export_csv`.

## See also

- [Ingest → point clouds](../ingest/point_clouds.md)
