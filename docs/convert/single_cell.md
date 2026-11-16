# Single-cell and spatial tables: h5ad, table, attach

Convert AnnData files and keyed tables to point stores, add columns with
`zvtools attach`, and export to h5ad or CSV. Chunk shape, `--overwrite` and
the export options are in [Converting files](index.md).

| Format | In / out | Extra | Positions |
|---|---|---|---|
| `h5ad` | in, out | `h5ad` | an `obsm` array, up to its first three columns |
| `table` | in (out as `csv`) | — | the named `--position-columns` |

```bash
zvtools convert cells.h5ad cells.zv --chunk-shape 500,500,500 --gene Gad1 --gene Pvalb
```

```python
from zarr_vectors_tools.convert.ingest.h5ad import ingest_h5ad

ingest_h5ad("cells.h5ad", "cells.zv", (500.0, 500.0, 500.0), genes=["Gad1", "Pvalb"])
```

Each cell or row becomes one point. Without `--spatial-key`, the first of
`spatial`, `X_spatial`, `spatial_fov`, `X_umap`, `X_tsne`, `X_pca` present is
used. `--chunk-shape` needs one entry per position column. The viewer shows 3D
stores only.

| Option | Format | Effect |
|---|---|---|
| `--spatial-key KEY` | h5ad | The `obsm` key holding the coordinates. |
| `--spatial-columns I,J[,K]` | h5ad | Which of its columns to use. |
| `--obs-column NAME` / `--no-obs` | h5ad | Store only these `obs` columns (repeatable), or none. Default: all. |
| `--gene NAME` | h5ad | Store this gene's expression as `gene_NAME` (repeatable). Default: none. |
| `--layer NAME` | h5ad | Read expression from `layers[NAME]`, not `X`. |
| `--backed` | h5ad | Leave `X` on disk; for large files with few genes. |
| `--drop-na` | h5ad | Drop cells with NaN coordinates. Without it they stop the ingest. |
| `--position-columns X,Y[,Z]` | table | Required. Coordinate columns in axis order. |
| `--key-column NAME` | table | Row identifier, hashed into `zv_join_key` for `attach`. |
| `--column NAME` | table | Store only these columns (repeatable). |
| `--delimiter D` | table | `,` (default), `tab`, `';'` or `whitespace`. |
| `--object-id-column NAME` | both | Group cells into objects by this column. |
| `--knn-distance-k K` | h5ad | Add `knn_distance`; see [Points](points.md). |

## What the store holds

- Numeric columns keep their dtype. Text and categorical columns become int32
  codes that carry their labels; missing values are -1. Booleans become uint8.
- `h5ad_row` or `table_row`: the source row, which export uses to restore order.
- `zv_join_key`: the hashed `obs_names` or `--key-column`, which `attach` joins on.
- With `--object-id-column`, each object's `vertex_count`.

A table ingest drops rows with NaN coordinates. Every name:
[Headers and attributes](../reference/headers_attributes.md).

## Keyed tables

Use `--format table` when positions are named columns, rows carry an
identifier, or columns hold text:

```bash
zvtools convert coords.csv atlas.zv --format table \
    --position-columns x,y,z --key-column cell_label --chunk-shape 500,500,500
```

## Adding columns: zvtools attach

`zvtools attach STORE FILE` adds vertex attributes from a table or an h5ad,
matching rows to points by key. Row order need not match. The store needs a
`zv_join_key`.

```bash
zvtools attach atlas.zv metadata.csv --key-column cell_label --column cluster
```

Points with no row get NaN (floats) or -1 (codes); `--missing error` refuses
instead. `--gene`, `--layer`, `--overwrite` and `--level` also work; see the
[CLI reference](../reference/cli.md). Each attribute costs one file per chunk,
so when adding many, pass the same `--shard N` to `convert` and every `attach`.

## Export to h5ad

```bash
zvtools convert cells.zv roundtrip.h5ad
```

Genes go to `X`, `obs` columns to `obs` with their categories, positions to
`obsm`. The row attribute restores the source order and `obs_names`: if you
pass `--attribute`, include it. A CSV export writes category columns as their
labels.
