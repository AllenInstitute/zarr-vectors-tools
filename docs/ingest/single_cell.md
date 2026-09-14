# Single-cell and spatial omics

Two related workflows, both writing the **points** Zarr Vectors geometry with one
vertex per cell:

- **[`ingest_h5ad`](#anndata-h5ad-ingest-h5ad)** — a self-contained
  AnnData `.h5ad` whose `obsm` already holds the coordinates.
- **[Staged attach](#staged-attach-datasets-split-across-files)** — a
  dataset split across several files, imported one file at a time.

Both need the `h5ad` extra:

```bash
pip install "zarr-vectors-tools[h5ad]"
```

## AnnData `.h5ad` — `ingest_h5ad`

An `.h5ad` is a cell-by-gene table with side tables. The only part of it
that is *spatial* is an `obsm` embedding, and that becomes the point
cloud. Everything else rides along as per-vertex attributes.

```python
from zarr_vectors_tools.ingest.h5ad import ingest_h5ad

summary = ingest_h5ad(
    "xenium.h5ad",
    "xenium.zv",
    (100.0, 100.0, 10.0),          # chunk_shape in µm, one entry per axis
    spatial_key="auto",            # obsm key; "auto" resolves it (below)
    obs_columns=None,              # None = every obs column; [] = none
    genes=["EPCAM", "PTPRC"],      # expression columns, opt-in
    object_id_column="cell_type",  # groups cells into Zarr Vectors objects
)
print(summary["spatial_key"], summary["vertex_count"])
```

`spatial_key="auto"` tries `spatial`, `X_spatial`, `spatial_fov`,
`X_umap`, `X_tsne`, `X_pca` in that order, so true spatial coordinates win
over an embedding when both are present. A 2D embedding stays 2D — pass a
2-tuple `chunk_shape` for it.

`genes` is opt-in because expression is wide: a Zarr Vectors attribute is one array
per name, so storing 20 000 genes would mean 20 000 arrays. `obs` columns
default to all, since there are usually few and they are what colouring is
driven by.

### What is stored, and how it round-trips

| AnnData | Zarr Vectors |
| --- | --- |
| `obsm[spatial_key]` | vertex positions |
| numeric `obs` column | vertex attribute, dtype preserved |
| categorical / string `obs` column | int32 codes + labels in the header |
| `bool` `obs` column | uint8 + dtype in the header |
| `X[:, gene]` / `layers[layer][:, gene]` | `gene_<name>` attribute |
| `obs_names` (barcodes) | header, when under `max_obs_index` |
| source row index | `h5ad_row` attribute |
| hashed `obs_names` | `zv_join_key` attribute |

Zarr Vectors orders vertices by spatial chunk, not by source row, so exporting
without the `h5ad_row` attribute would return the cells permuted. It is
stored by default and [`export_h5ad`](../export/single_cell.md) sorts on
it, making the round-trip exact:

```python
from zarr_vectors_tools.export.h5ad import export_h5ad

export_h5ad("xenium.zv", "round_trip.h5ad")
```

:::{note}
`datetime64` `obs` columns are stored as int64 nanoseconds and stay that
way on export — AnnData has no h5ad serialisation for datetimes, so
handing one back would produce a file `write_h5ad` refuses. The original
dtype is recorded in the header.
:::

## Staged attach: datasets split across files

Large atlases rarely ship as one file. The Allen Brain Cell Atlas MERFISH
releases are the worked example: coordinates in one CSV, per-cell metadata
in another, expression in a multi-gigabyte `.h5ad` with **no `obsm` at
all** — and each file covering a different subset of cells in a different
order.

```text
ccf_coordinates.csv   2 616 328 cells   cell_label, x, y, z, parcellation_index
cell_metadata.csv     2 846 908 cells   cell_label, cluster_alias, donor_sex, ...
*-log2.h5ad           4 167 870 cells × 1 122 genes, obsm empty
```

Merging those into a single `.h5ad` first would mean materialising an
18.7 GB dense matrix. Instead, create the store from whichever file has
the coordinates, then stage each remaining file in as attributes.

### 1. Create the store from the coordinate table

`ingest_table` is the CSV ingester for tables whose rows carry a stable
identifier. Unlike `ingest_csv` it reads through pandas, so string and
categorical columns work, and it hashes the identifier into the
`zv_join_key` attribute that later files join against.

```bash
zvtools convert ccf_coordinates.csv abca1.zv --format table \
    --position-columns x,y,z --key-column cell_label \
    --chunk-shape 0.5,0.5,0.5
```

```text
ingested table (points)
  vertex_count: 2616328
  chunk_count: 2742
  key_column: cell_label
```

`--format table` is required: `.csv` auto-detects to the numeric-only
`csv` ingester, which cannot read a 39-digit `cell_label`.

### 2. Stage in the metadata table

```bash
zvtools attach abca1.zv cell_metadata.csv --key-column cell_label \
    --column cluster_alias --column donor_sex --column brain_section_label
```

```text
  vertices_matched: 2616328
  vertices_unmatched: 0
  rows_available: 2846908
```

The table has 230 580 more rows than the store has points; those rows are
simply unused. Points with *no* row in the incoming file are filled
(`NaN` for floats, `-1` for category codes) and counted in
`vertices_unmatched` rather than failing the import. Pass
`--missing error` to make an unmatched point fatal instead.

### 3. Stage in expression

```bash
zvtools attach abca1.zv Zhuang-ABCA-1-log2.h5ad \
    --gene Htr7 --gene Slc17a7 --gene Pvalb
```

Genes match against `var_names` first and then `gene_symbol`, so Ensembl
IDs and symbols both work without converting. Only the requested columns
are read — on the file above that is ~36 s at 1.8 GB peak RSS, against
18.7 GB to densify `X`.

Repeat as needed: each run adds attributes to the same store, so a panel
can be built up over several sessions rather than in one pass.

### 4. Export the assembled result

```python
from zarr_vectors_tools.export.h5ad import export_h5ad

# A spatial subset, with everything staged in so far
export_h5ad("abca1.zv", "slab.h5ad", bbox=([5.0, 3.0, 2.0], [5.5, 3.5, 2.5]))
```

Staged columns are registered in the store header as they are attached, so
export emits them under their original labels with categoricals decoded —
indistinguishable from columns written at ingest.

### The join key

Identifiers are reduced to int64 by `hash_keys` (FNV-1a over the UTF-8
bytes) because Zarr Vectors attributes are numeric, and because the atlas's own
`cell_label` is a 39-digit value that no integer column can hold. The hash
ignores the zero padding that fixed-width numpy byte-string arrays carry,
so the same label hashes identically whether it arrives in an `S39` or an
`S64` column.

A store must have been ingested **with a key** to be attached onto. Both
`ingest_table(key_column=...)` and `ingest_h5ad` (which hashes `obs_names`
by default) write one; `attach` fails with a clear message otherwise.

For the case where the incoming file shares the store's *row order* — more
columns from the file the store was built from — pass `keys=None` and
point `key_attribute` at the row-index attribute instead:

```python
from zarr_vectors_tools.ingest.attach import attach_attributes

attach_attributes(
    "abca1.zv",
    {"score": scores},          # in the source file's row order
    keys=None,
    key_attribute="table_row",
)
```

### Sharding — decide it up front

:::{warning}
Pass `--shard` on **every** attach when staging a wide panel. Unsharded, a
Zarr Vectors store costs one file per spatial chunk *per attribute*, so the file
count is `chunks × attributes`. On the atlas store above — 2,742 chunks and
1,122 genes — that is **3.1 million files**, and `zvtools shard` afterwards
has to read every one of them back: measured at ~3 arrays/min, roughly
6 hours.
:::

```bash
zvtools convert ccf_coordinates.csv abca1.zv --format table \
    --position-columns x,y,z --key-column cell_label \
    --chunk-shape 0.5,0.5,0.5 --shard 12
zvtools attach abca1.zv cell_metadata.csv --key-column cell_label \
    --column cluster_alias --shard 12
zvtools attach abca1.zv Zhuang-ABCA-1-log2.h5ad --gene Htr7 --shard 12
```

`--shard 12` packs 12×12×12 chunks per file, taking the same store to
~6 shards per array — about 8,000 files instead of 3.1 million. Keep the
value identical across attaches so the store stays uniform.

If a store is already unsharded, `zvtools shard` still fixes it, but the
repack is single-threaded and array-by-array. Since `shard_store` takes an
explicit `arrays=` list, skips arrays already sharded at the target shape,
and touches no shared metadata, it partitions safely across processes —
which is how the atlas store was repacked in minutes rather than hours.

### Cost

Attach is chunk-wise: peak memory is one column of the incoming file plus
one spatial chunk, never the whole store. Measured on the files above:

| Step | Cells | Wall clock | Peak RSS |
| --- | --- | --- | --- |
| `convert` coordinates | 2 616 328 | 11 s | — |
| `attach` metadata (4 columns) | 2 846 908 rows | 33 s | — |
| `attach` 4 genes from a 2 GB `.h5ad` | 4 167 870 rows | 36 s | 1.8 GB |

## See also

- [Export to `.h5ad`](../export/single_cell.md)
- [Point clouds](point_clouds.md) — the numeric-only CSV, LAS, PLY paths
- [Headers](../headers.md) — what round-trip metadata is kept
