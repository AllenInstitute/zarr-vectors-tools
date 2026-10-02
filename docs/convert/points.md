# Point clouds: CSV, PLY, LAS

```bash
zvtools convert points.csv points.zv --chunk-shape 250,250,250
zvtools convert cloud.ply cloud.zv --chunk-shape 250,250,250
zvtools convert scan.las scan.zv --chunk-shape 250,250,250
```

Each row or vertex becomes one point. PLY needs the `ply` extra, LAS the `las` extra.
The chunk shape is in the file's units; see [Store layout](../store_layout.md#chunk-shape).

## What each reader keeps

| Format | Positions | Per-vertex attributes |
|---|---|---|
| `csv`, `.xyz` | the first three columns | every other column (float32), under its header name, or `col3`, `col4`, … without a header |
| `ply` | `x`, `y`, `z` of the `vertex` element (else `X`, `Y`, `Z`, else the first three properties) | every other vertex property (float32) |
| `las`, `.laz` | X, Y, Z | `intensity`, `classification` and `color` (red, green, blue) as float32; `gps_time` as float64 |

Every name and dtype is listed on [Headers and attributes](../reference/headers_attributes.md).

- **csv**: a first row that is not all numbers is the header; otherwise every
  row is a point. The delimiter comes from the first row too: a comma if it has
  one, else runs of spaces or tabs, so a space-delimited `.xyz` needs no flags.
  `--delimiter` overrides it: `,`, `';'`, `tab`, or `whitespace`. Every column
  must hold numbers; for text columns use `--format table`.
- **ply**: only the `vertex` element is read. A mesh PLY's faces are dropped.
- **las**: other point fields (return number, scan angle, …) are not kept.
  `gps_time` keeps the file's full precision. `color` is one 3-column
  attribute, which the viewer cannot colour by yet; see
  [Visualise](../visualise.md). Positions are float32 by default, which steps
  by 0.25 m at georeferenced coordinates near 4×10⁶ m; the summary then lists
  a warning, and `--dtype float64` keeps the file's resolution (the viewer
  reads float32 only).

## csv or table?

`csv` is for numeric point lists: x, y, z first, numbers after. Use
`--format table` when positions are named columns, rows carry an identifier you
will join other files on, you need object ids, or columns hold text. See
[Single cell](single_cell.md#keyed-tables).

## kNN distance

```bash
zvtools convert points.csv dense.zv --chunk-shape 250,250,250 --knn-distance-k 8
```

Adds the per-vertex attribute `knn_distance`: each point's mean distance to its 8
nearest neighbours, in coordinate units. Small values mark dense regions. It
needs `pip install "zarr-vectors-tools[points-enrichment]"` (scipy) and is
accepted for `csv`, `ply`, `las` and `h5ad` input.

## From Python

```python
import pandas as pd
from zarr_vectors_tools.convert.ingest.csv_points import ingest_csv

labels = pd.read_csv("points.csv")["label"].to_numpy()   # one id per row, file order
ingest_csv(
    "points.csv", "labelled.zv", (250.0, 250.0, 250.0),
    attribute_columns=["intensity"],
    object_ids=labels,
)
```

Object ids and column choice are Python-only. `ingest_csv` reads the header
and delimiter as the CLI does; `has_header=` and `delimiter=` override them. It
also takes `position_columns`, `drop_na`, `drop_duplicates` and `normalise`;
`ingest_ply` and `ingest_las` take `object_ids` and `include_attributes=False`.
With `object_ids`, each object's `vertex_count` (uint32) is stored too;
`per_object_vertex_count=False` skips it. A point cloud without object ids has no
objects, so pyramid sparsity has nothing to drop; see [Pyramids](../pyramids.md).

## Export to CSV or PLY

```bash
zvtools convert points.zv points_out.csv --attribute intensity --attribute label
zvtools convert points.zv box.ply --bbox 0,0,0,500,500,500 --attribute intensity
zvtools convert points.zv points.tsv --format csv --delimiter tab
zvtools convert points.zv points_out.xyz
```

- Positions only by default. `--attribute NAME` (repeatable) adds a column. A
  name the level does not have stops the export with
  `ExportError: attribute(s) not present at level 0: ['colour']`.
- **CSV** position columns take the names the source gave them (a CSV's
  header, a table's `--position-columns`), else `x,y,z`; then each attribute.
  A multi-column attribute becomes `color_0,color_1,color_2`. Numbers have six
  decimals; a category column (from a table or h5ad) is written as its labels.
  A `.xyz` output is space-delimited with no header row.
- **PLY** is binary little-endian: `x`, `y`, `z` as float32, then attributes,
  float32 except float64 ones such as `gps_time`, which stay double. A category
  column is refused, since a PLY property holds numbers only. ASCII PLY:
  `export_ply(..., binary=False)` in Python.
- `--object-id` needs a store written with object ids; on one without, the
  export stops with `ExportError: this store has no object ids at level 0`.
- `--level N` exports a coarser level, attributes included: a float column
  holds the bin's mean, any other column one merged point's value
  ([Pyramids](../pyramids.md)).
