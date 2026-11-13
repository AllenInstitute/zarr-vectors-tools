# Point clouds: CSV, PLY, LAS

Convert CSV, PLY and LAS point clouds to point stores, and export them to CSV
or PLY. Chunk shape, `--overwrite` and the export options are in
[Converting files](index.md).

| Format | In / out | Extra | Positions | Vertex attributes |
|---|---|---|---|---|
| `csv`, `.xyz` | in, out | — | the first three columns | every other column, under its header name (`col3`, `col4`, … without a header) |
| `ply` | in, out | `ply` | `x`, `y`, `z` of the `vertex` element | every other vertex property |
| `las`, `.laz` | in | `las` | X, Y, Z | `intensity`, `classification`, `color` (3 columns), `gps_time` |

```bash
zvtools convert points.csv points.zv --chunk-shape 250,250,250
```

```python
import pandas as pd
from zarr_vectors_tools.convert.ingest.csv_points import ingest_csv

labels = pd.read_csv("points.csv")["label"].to_numpy()   # one object id per row
ingest_csv("points.csv", "labelled.zv", (250.0, 250.0, 250.0),
           attribute_columns=["intensity"], object_ids=labels)
```

Object ids and column choice are Python-only. `ingest_ply` and `ingest_las`
also take `object_ids`. Without object ids a point cloud has no objects, so
pyramid sparsity has nothing to drop.

| Option | Effect |
|---|---|
| `--delimiter D` | csv input: `,`, `';'`, `tab` or `whitespace`. Default: from the first row. |
| `--knn-distance-k K` | Add `knn_distance`: each point's mean distance to its K nearest neighbours. Needs the `points-enrichment` extra. Also for h5ad. |

## What the store holds

- One point per row or vertex. In a CSV, a first row that is not all numbers is
  the header. Every column must hold numbers; for text columns use
  [`--format table`](single_cell.md#keyed-tables).
- Attributes are float32, except LAS `gps_time` (float64). Other LAS fields and
  a PLY's faces are dropped.
- With object ids, each object's `vertex_count`.
- Every name and dtype: [Headers and attributes](../reference/headers_attributes.md).

At georeferenced coordinates, float32 positions lose precision: near
4×10⁶ m they step by 0.25 m. The summary then warns; `--dtype float64` keeps
the file's resolution, but the viewer reads float32 only.

## Export to CSV or PLY

```bash
zvtools convert points.zv points_out.csv --attribute intensity
```

CSV keeps the source's position column names, else `x,y,z`. A 3-column
attribute becomes `color_0,color_1,color_2`, and a category column is written
as its labels. A `.xyz` output has spaces and no header. PLY is binary; it
refuses category columns. `export_ply(..., binary=False)` writes ASCII.
