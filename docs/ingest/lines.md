# Lines

A **line** in ZVF is a single, independent two-endpoint segment — not a
chain of segments (those are polylines). This package currently has one
line ingest: `ingest_lines_csv`.

## CSV line segments — `ingest_lines_csv`

Six-column (3D) or four-column (2D) CSV: first `2 * ndim` columns are
endpoint coordinates `(x0, y0, z0, x1, y1, z1)`, remaining columns become
per-line attributes.

```python
from zarr_vectors_tools.ingest import ingest_lines_csv

ingest_lines_csv(
    input_path="segments.csv",
    output_path="segments.zv",
    chunk_shape=(10.0, 10.0, 10.0),
    compute_length=True,
)
```

**Notable options**

`ndim`
: 2 or 3 (default 3). Determines how many columns are positions.

`attribute_columns`
: Filter which non-position columns to keep. Default: all of them.

`compute_length`
: Writes `line_attributes["length"]` — the Euclidean distance between
  each segment's endpoints.

`drop_zero_length`
: Drop segments shorter than `1e-9` before writing.

`drop_na`, `drop_duplicates`
: Pre-filter rows.

The returned summary dict includes a `dropped_*` counter for each
filter that was active.

## See also

- [Enrichments](../enrichments.md#lines)
- Parent: [Line spec](https://zarr-vectors.readthedocs.io/en/latest/spec/geometry_types/line.html)
