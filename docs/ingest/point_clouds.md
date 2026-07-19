# Point clouds

Three source formats write the **points** ZVF geometry: delimited text
(`ingest_csv`), LAS/LAZ (`ingest_las`), and PLY (`ingest_ply`,
points-only — mesh PLY is not yet wired up). All three share the same
`object_ids`, `knn_distance_k`, and `per_object_vertex_count` options.

## CSV / XYZ — `ingest_csv`

The worked example is a MERFISH-style transcript table: one row per
detected transcript, with a position in micrometres, the gene it was
called as, and the ID of the cell it was assigned to. Every transcript
is a vertex; every cell is an object.

```python
import numpy as np
import pandas as pd

from zarr_vectors_tools.ingest.csv_points import ingest_csv

# transcripts.csv:  x,y,z,gene_id,cell_id
#                   1043.2,880.7,12.0,417,90211
# object_ids is a separate (N,) array, not a CSV column — read it first,
# in the same row order ingest_csv will parse.
cell_ids = pd.read_csv("transcripts.csv")["cell_id"].to_numpy(dtype=np.int64)

summary = ingest_csv(
    "transcripts.csv",
    "merfish.zv",
    (100.0, 100.0, 10.0),           # chunk_shape in µm — thin in z (tissue sections)
    auto_detect_columns=True,       # resolve x/y/z from the header row
    attribute_columns=["gene_id"],  # per-transcript attribute; keeps cell_id out
    object_ids=cell_ids,            # groups transcripts into per-cell objects
    per_object_vertex_count=True,   # object_attributes["vertex_count"] = transcripts/cell
    knn_distance_k=8,               # attributes["knn_distance"] — needs points-enrichment
    drop_na=True,                   # discard rows with a NaN coordinate
)
print(summary["dropped_na"])
```

`auto_detect_columns=True` only fires when `position_columns` is `None`.
It matches header names case-insensitively against `x`/`pos_x`/`posx`/`px`
(and the `y`/`z` equivalents), and picks up `r`/`g`/`b`, `intensity`,
`label`, `class`, `tissue`, `confidence` as attributes. If it cannot
resolve all `ndim` axes it returns nothing and the function falls back to
"first `ndim` columns are positions" — silently. Pass `position_columns`
explicitly when the header is unusual.

Setting `attribute_columns` yourself is what keeps `cell_id` from being
duplicated as a per-vertex attribute: the default is *every* non-position
column.

:::{note}
Attribute columns are read through `np.loadtxt` and cast to `float32`, so
a gene *name* column fails the parse and raises `IngestError`. Map gene
names to integer IDs before ingest and keep the lookup table beside the
store.
:::

Other options worth knowing:

| Option | Effect |
| --- | --- |
| `ndim` | Spatial dimensionality, default `3`. |
| `delimiter`, `has_header`, `skip_rows` | Text parsing. |
| `normalise` | Centre on the centroid and divide by the maximum absolute coordinate so positions sit in `[-1, 1]`. Offset and scale go to `CSVHeader` for round-trip export. |
| `drop_duplicates` | Drop rows whose position triple repeats an earlier row. |

`knn_distance_k` writes each point's mean Euclidean distance to its `k`
nearest neighbours. It needs `scipy` — `pip install
"zarr-vectors-tools[points-enrichment]"`. On a transcript cloud that is a
cheap local-density proxy: dense neighbourhoods give small values.

`per_object_vertex_count=True` raises `IngestError` unless `object_ids`
is supplied. Note that `drop_na` and `drop_duplicates` filter
`object_ids` alongside the positions, so the counts stay consistent.

## LAS / LAZ — `ingest_las`

```python
from zarr_vectors_tools.ingest.las import ingest_las

ingest_las(
    "scan.laz",
    "scan.zv",
    (10.0, 10.0, 10.0),
    include_attributes=True,   # default — pull the standard LAS fields
    knn_distance_k=8,
)
```

Requires `laspy` — `pip install "zarr-vectors-tools[las]"`. LAZ goes
through the same `laspy.read` call. With `include_attributes=True`, each
standard field present in the file becomes a per-vertex attribute:

| LAS field | ZVF attribute | dtype on disk |
| --- | --- | --- |
| `intensity` | `intensity` | float32 |
| `classification` | `classification` | float32 (cast from int32) |
| `red`, `green`, `blue` | `color` | float32, shape `(N, 3)` |
| `gps_time` | `gps_time` | float32 |

There is no CSV-style column selection here — it is all fields or none.

## PLY (points) — `ingest_ply`

```python
from zarr_vectors_tools.ingest.ply import ingest_ply

ingest_ply(
    "cloud.ply",
    "cloud.zv",
    (1.0, 1.0, 1.0),
    include_attributes=True,
)
```

Requires `plyfile` — `pip install "zarr-vectors-tools[ply]"`. Positions
come from the `vertex` element's `x`/`y`/`z` or `X`/`Y`/`Z` properties;
failing both, the first three properties are used and `ndim` is inferred
from them. With `include_attributes=True` every remaining vertex property
is written under its own PLY name, cast to float32 — properties that will
not cast are skipped rather than raising.

## See also

- [Enrichments → point clouds](../enrichments.md#point-clouds)
- [Export → point clouds](../export/point_clouds.md)
- [Choosing chunk and bin shapes](../how_to/choose_chunk_and_bin.md)
- [Ingest workflows](index.md)
