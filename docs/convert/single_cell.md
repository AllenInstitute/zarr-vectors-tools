# Single-cell and spatial tables: h5ad, table, attach

```bash
zvtools convert cells.h5ad cells.zv --chunk-shape 500,500,500 --gene Gad1 --gene Pvalb
```

Each cell becomes one point. h5ad input and output need the `h5ad` extra;
`--format table` needs none. The chunk shape is in the coordinates' units
([Store layout](../store_layout.md#chunk-shape)).

## AnnData (.h5ad)

Positions come from an `obsm` array. Without `--spatial-key`, the first of
`spatial`, `X_spatial`, `spatial_fov`, `X_umap`, `X_tsne`, `X_pca` present is
used, so true coordinates win over an embedding. Up to its first three columns
become x, y, z, and `--chunk-shape` needs one entry per column used. The viewer
shows 3D stores only ([Visualise](../visualise.md)).

```bash
zvtools convert cells.h5ad umap.zv --spatial-key X_umap --chunk-shape 5,5 --no-obs
```

| Option | Effect |
|---|---|
| `--spatial-key KEY` | The `obsm` key holding the coordinates. |
| `--spatial-columns I,J[,K]` | Which columns of it to use (default: the first up to 3). |
| `--obs-column NAME` | Store only these `obs` columns (repeatable). Default: all of them. |
| `--no-obs` | Store no `obs` columns. |
| `--gene NAME` | Store this gene's expression as `gene_NAME` (repeatable). Default: no genes. |
| `--layer NAME` | Read expression from `layers[NAME]` instead of `X`. |
| `--object-id-column NAME` | Group cells into objects by this `obs` column. |
| `--backed` | Leave `X` on disk while reading. Use it on large files with few `--gene`. |
| `--drop-na` | Drop cells with NaN coordinates. Without it they stop the ingest. |
| `--knn-distance-k K` | Add `knn_distance`; see [Points](points.md#knn-distance). |

Numeric `obs` columns keep their dtype. Text and categorical columns are stored
as int32 codes that carry their labels (dictionary-encoded), so the viewer
shows `T`, `B`, `NK` rather than 0, 1, 2; missing values are code -1. Booleans
are stored as uint8. Two
more attributes are written: `h5ad_row` (the source row, which export
uses to restore order) and `zv_join_key` (the hashed `obs_names`, which
`zvtools attach` joins on). Every name is listed on
[Headers and attributes](../reference/headers_attributes.md).

With `--object-id-column`, object ids are the column's codes: category order for
a categorical, order of first appearance for text, the values themselves for an
integer column. Missing values are refused.

Python-only `ingest_h5ad` options: `store_row_index=False` and
`store_join_key=False` skip the two attributes above; `preserve_obs_index`
inlines the cell barcodes in the header so export restores `obs_names`, up to
`max_obs_index` cells (default 200,000). With `--object-id-column`, each
object's cell count is stored as `vertex_count`; `per_object_vertex_count=False`
skips it.

## Keyed tables

For a delimited table with named position columns, a row identifier or text columns:

```bash
zvtools convert coords.csv atlas.zv --format table \
    --position-columns x,y,z --key-column cell_label --chunk-shape 500,500,500
```

| Option | Effect |
|---|---|
| `--position-columns X,Y[,Z]` | Required. Coordinate columns in axis order; `--chunk-shape` needs as many entries. |
| `--key-column NAME` | Row identifier, hashed into `zv_join_key` so later files can be attached. |
| `--column NAME` | Store only these columns (repeatable). Default: every column but positions and key. |
| `--delimiter D` | Column separator: `,` (default), `tab`, `';'`, or `whitespace` for runs of spaces and tabs. |
| `--object-id-column NAME` | Group rows into objects. |

Rows with NaN coordinates are always dropped (`dropped_na` in the summary).
Columns are encoded as for h5ad `obs`, and each point gets `table_row`, its row
after that drop. Python-only `ingest_table` options: `preserve_index`
inlines the key values in the header so h5ad export restores them as
`obs_names`, up to `max_index` rows (default 200,000).

## Adding columns: zvtools attach

`zvtools attach STORE FILE` adds per-vertex attributes from a table or an h5ad
to a store, matching rows to points by key. The file's row order and row set
need not match the store.

```bash
zvtools attach atlas.zv metadata.csv --key-column cell_label --column cluster --column score
zvtools attach atlas.zv expression.h5ad --gene Htr7 --gene Npy
```

Each run reports `vertices_matched` and `vertices_unmatched`. Rows with no point
are ignored. Points with no row get NaN (floats) or -1 (codes); `--missing error`
refuses instead, before anything is written. The store needs a join key: ingest it from h5ad, or as a table
with `--key-column`.

| Option | Effect |
|---|---|
| `--key-column NAME` | The file's identifier column. Required for a table; h5ad defaults to the `obs` index. |
| `--key-attribute NAME` | The store attribute holding the key. Default `zv_join_key`. |
| `--column NAME` | Columns to add (repeatable): any column of a table (default all but the key), `obs` columns of an h5ad. |
| `--gene NAME` | h5ad expression to add as `gene_NAME`, matched on `var_names`, then `gene_symbol`. `--gene-by COL` picks the `var` column; `--layer` reads a layer. |
| `--missing fill\|error` | What to do with points that have no row. |
| `--overwrite` | Replace attributes that exist; otherwise they are refused. |
| `--shard N` | Write the new arrays sharded. |
| `--level N` | Level to write into. Default 0. |

Unsharded, each attribute costs one file per chunk. When adding many columns,
pass the same `--shard N` to `convert` and every `attach`
([Store layout](../store_layout.md#sharding)):

```bash
zvtools convert coords.csv atlas_s.zv --format table --position-columns x,y,z \
    --key-column cell_label --chunk-shape 500,500,500 --shard 8
zvtools attach atlas_s.zv expression.h5ad --gene Htr7 --gene Npy --shard 8
```

The viewer shows the first 32 vertex attributes; name others with `#attributes=` ([Visualise](../visualise.md)).

## Export to h5ad

```bash
zvtools convert cells.zv roundtrip.h5ad
zvtools convert atlas.zv slab.h5ad --bbox 0,0,0,1000,1000,1000 \
    --attribute cluster --attribute gene_Htr7 --attribute table_row
```

Every vertex attribute is written unless `--attribute` names some. Genes go to
`X` under their gene names, `obs` columns to `obs` with their categories
(order, ordered flag and missing values as in the source), multi-column
attributes to `obsm`, positions to `obsm` under the source key.
The row attribute (`h5ad_row` or `table_row`) restores the source order and
`obs_names`; when you pass `--attribute`, include it, or cells are named `0`,
`1`, … in store order. `--object-id` selects by `--object-id-column` codes.
`--level 1` exports a coarser level with its attributes; categories there come
from one cell per bin ([Pyramids](../pyramids.md)).

A CSV export of a table store names the position columns as
`--position-columns` did and writes category columns as their labels:

```bash
zvtools convert atlas.zv atlas.csv --attribute cluster
```
