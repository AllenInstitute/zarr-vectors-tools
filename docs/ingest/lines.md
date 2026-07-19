# Lines

A **line** in ZVF is a single, independent two-endpoint segment — not a
chain of segments (those are polylines, see
[Tractography](tractography.md)). One ingest covers it:
`ingest_lines_csv`.

## CSV line segments — `ingest_lines_csv`

The first `2 * ndim` columns are endpoint coordinates in
`x0,y0,z0,x1,y1,z1` order — six columns in 3D, four in 2D. Everything
after them is a per-line attribute.

```python
from zarr_vectors_tools.ingest.lines import ingest_lines_csv

# segments.csv:  x0,y0,z0,x1,y1,z1,weight
#                12.0,4.5,0.0,18.2,4.9,0.0,0.73
summary = ingest_lines_csv(
    "segments.csv",
    "segments.zv",
    (10.0, 10.0, 10.0),
    attribute_columns=["weight"],  # header names; default is every non-position column
    compute_length=True,           # object_attributes["length"], Euclidean endpoint distance
    drop_zero_length=True,         # drop segments below 1e-9 long
    drop_na=True,                  # drop rows with a NaN endpoint coordinate
    drop_duplicates=True,          # drop rows whose full endpoint pair repeats
)
print(summary["dropped_zero_length"], summary["dropped_na"])
```

The filters run in a fixed order — `drop_na`, then `drop_duplicates`,
then `drop_zero_length` — and each writes its own `dropped_*` counter
into the summary dict. `compute_length` reuses the lengths already
computed for `drop_zero_length`, so enabling both costs nothing extra.

`attribute_columns` takes header *names*, so it only works with
`has_header=True`. Without a header, non-position columns are named
`col6`, `col7`, … by index.

:::{warning}
`lines` registers no file extension. A `.csv` input auto-detects to
`csv`, which ingests it as a **point cloud** using only the first three
columns — no error, wrong store. On the CLI you must pass `--format
lines` explicitly:

```bash
zvtools convert segments.csv segments.zv --format lines --chunk-shape 10,10,10
```
:::

If every row is filtered out, the function raises `IngestError` rather
than writing an empty store.

## See also

- [Enrichments → lines](../enrichments.md#lines)
- [Tractography](tractography.md) — for connected vertex chains
- [Ingest workflows](index.md)
