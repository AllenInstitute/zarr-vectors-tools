# Single-cell and spatial omics

`export_h5ad` reads the Zarr Vectors `points` geometry and writes an AnnData
`.h5ad`. It is the inverse of
[`ingest_h5ad`](../ingest/single_cell.md), and it is also how a store
assembled by [staged attach](../ingest/single_cell.md#staged-attach-datasets-split-across-files)
is turned back into a single analysable file.

Needs the `h5ad` extra: `pip install "zarr-vectors-tools[h5ad]"`.

```python
from zarr_vectors_tools.convert.export.h5ad import export_h5ad

summary = export_h5ad(
    "xenium.zv",                # store_path
    "xenium.h5ad",              # output_path
    level=0,                    # resolution level to read
    bbox=None,                  # optional filters, AND-ed together
    chunks=None,
    attribute_names=None,       # None = every attribute at this level
    spatial_key=None,           # obsm key; None = the header's, else "spatial"
    restore_order=True,         # sort back into source obs order
    decode_categoricals=True,   # rebuild pandas.Categorical from codes
    compression="gzip",
)
summary["vertex_count"], summary["n_vars"], summary["order_restored"]
```

## How attributes are routed

| Vertex attribute | Lands in |
| --- | --- |
| recorded in the header as a gene | `X`, with the gene as a `var_name` |
| recorded in the header as an `obs` column | `obs`, under its original label |
| categorical (header has its levels) | `obs` as a `pandas.Categorical` |
| multi-channel (2-D) | `obsm[name]` — no single `obs` column fits it |
| `h5ad_row` / `table_row`, `zv_join_key` | nothing — Zarr Vectors bookkeeping |
| anything else | `obs`, under its stored name |

Positions always become `obsm[spatial_key]`.

The row-index and join-key attributes are deliberately dropped: one exists
to restore ordering, the other so further files can be staged in. Neither
is data, and surfacing them would put a column of hashes in every exported
`obs`.

## Order and identity

Zarr Vectors orders vertices by spatial chunk, not by source row. With the
row-index attribute present (written by default at ingest) `export_h5ad`
sorts on it and `order_restored` comes back `True`, so an
ingest → export round-trip returns cells in their original order. When the
header also carries the `obs` index, `obs_names` are restored to the
original barcodes; otherwise they become `"0"`, `"1"`, … .

A store written by some other ingester exports fine — every attribute
becomes a numeric `obs` column under its stored name, `order_restored` is
`False`, and `X` has zero columns.

## Filters

`bbox` and `chunks` behave as they do everywhere else in the export API.

:::{warning}
`object_ids` is the exception. Core's object-filtered read path does not
carry vertex attributes, so combining `object_ids=[...]` with attributes
would silently produce an `.h5ad` with an empty `obs`. `export_h5ad`
raises `ExportError` instead:

```python
# Refused — the attributes would come back empty
export_h5ad("cells.zv", "out.h5ad", object_ids=[0])

# Fine — positions only, filtered by object
export_h5ad("cells.zv", "out.h5ad", object_ids=[0], attribute_names=[])
```

Use `bbox=` or `chunks=` when you need attributes alongside a subset.
:::

## Exporting from a coarser level

Coarser pyramid levels carry positions but not attributes, so a level-1
export yields geometry with an empty `obs` and `order_restored=False`:

```python
export_h5ad("cells.zv", "coarse.h5ad", level=1)
# {'vertex_count': 125, 'n_vars': 0, 'obs_columns': 0, ...}
```

## See also

- [Ingest `.h5ad` and staged attach](../ingest/single_cell.md)
- [Point clouds](point_clouds.md) — CSV and PLY targets for the same geometry
