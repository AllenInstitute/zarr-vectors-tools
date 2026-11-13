# Merge and split stores

`zvtools merge TARGET SOURCES...` appends the objects of files and stores to an
existing store without rewriting what is there. `zvtools split STORE OUTDIR`
writes one new store per group, attribute value or merged source. The
[CLI reference](reference/cli.md) lists every flag.

## Merge

```bash
zvtools merge tracts.zv subject_b.trk --dry-run    # plan only: ids, fit
zvtools merge tracts.zv subject_b.trk --on-out-of-bounds expand
```

Incoming objects get new ids after the target's last one. Merge carries:

| What | How |
|---|---|
| Object attributes | Appended. A missing column is filled with NaN (float), or the type's minimum (signed int) or maximum (unsigned int). `length`, `start`, `end` and `vertex_count` are computed from the incoming geometry when the target has them. |
| Vertex attributes | The target's set, filled the same way. A name new to a non-empty target is skipped. |
| Groups | Renumbered. A name the target has is extended; `--group-prefix b_` keeps incoming groups apart. |
| Headers | Copied. A second header of one format is kept as `<format>@<source>`; exporters read only the first. |
| Provenance | Recorded, for `split --by provenance`. |

Streamlines, lines and point clouds with object ids can be merged; meshes,
skeletons, graphs and mixed geometries cannot. TRK, TCK and TRX files are read
directly. Other files are converted on the target's chunk shape first. A point
cloud without object ids, or a format that needs options such as `table`, is
refused: convert it first, then merge the store.

| Option | Meaning |
|---|---|
| `--on-out-of-bounds` | For a source outside the target's chunks: `raise` (default), `skip` the objects outside, or `expand` the grid upwards. A source below the target's first chunk is refused. |
| `--transform FILE` | Move every source by a 4 × 4 affine (`.npy`, `.json` or 16 comma-separated numbers). |
| `--space ras` | Read TRK sources in RAS, for a target converted with `--apply-affine` (default: voxmm). |
| `--source-attr NAME` | Store each new object's source index as an object attribute. |
| `--create --cell-size X,Y,Z` | Make a missing target with these chunks. |
| `--group-by NAME --lut FILE` | Turn a per-streamline label into named groups, using a JSON `{"name": code}` table (TRK, TCK and TRX only). |

```bash
zvtools merge atlas.zv atlas.trk --create --cell-size 10,10,10 \
    --group-by label_id --lut atlas_lut.json
```

### Pyramid after a merge

`--pyramid rebuild` (default) rebuilds each coarser level with its recorded
settings. Pass the `--sparsity-strategy` the levels were built with, because
it is not recorded. `drop` removes the levels; `keep` leaves them out of date and marks each one
with `"compose_stale": true` in the `zarr_vectors_tools` block of its level
attributes.
For a different pyramid, merge with `--pyramid drop` and run `zvtools pyramid`
([Pyramids](pyramids.md)).

## Split

```bash
zvtools split atlas.zv bundles/ --dry-run     # part names and object counts
zvtools split atlas.zv bundles/               # bundles/atlas_cst.zv, ...
zvtools split tracts.zv parts/ --by provenance
```

Each part, `OUTDIR/<store>_<part>.zv`, keeps attributes, groups and headers.

| Option | Meaning |
|---|---|
| `--by groups` | One part per named group (default). |
| `--by attribute --attribute NAME` | One part per value of an object attribute; `--lut FILE` names the values. |
| `--by provenance` | One part per merged source, plus `original`. |
| `--level N` | Split level N instead of level 0. |
| `--pyramid rebuild` | Rebuild the parent's coarser levels for each part (level 0 splits only). Default `drop`. |

## Python

```python
from zarr_vectors_tools.compose import FileSource, merge_stores, split_store

merge_stores("tracts.zv", [FileSource("subject_b.trk")],
             on_out_of_bounds="expand", pyramid="drop")
split_store("atlas.zv", "bundles", by="groups")
```

Most keywords are the CLI options with `_` for `-` (`create`, `cell_size`,
`group_prefix`, `on_out_of_bounds`, `pyramid`; for `split_store` also `by`,
`attribute`, `level`, `bounds`, `min_objects`, `overwrite`). These differ:

| CLI | Python |
|---|---|
| `--source-attr NAME` | `merge_stores(source_attribute=NAME)` |
| `--pyramid-coarsen`, `--pyramid-sparsity` | `pyramid_factors=[(coarsen, sparsity), ...]`, one pair per level |
| `--sparsity-strategy`, `--coarsen-mode` | `pyramid_options={"sparsity_strategy": ..., "coarsen_mode": ...}` |
| `--lut FILE` (split) | `split_store(names=load_lut(FILE))` |
| `--format`, `--transform`, `--space` (merge) | per source: `FileSource(path, format=..., transform=..., space=...)` (`space` for TRK) |
| `--group-by NAME --lut FILE` (merge) | per source: `derive_groups(read_geometry(path, fmt), NAME, names=load_lut(FILE))` in a `GeometrySource` |
| `--dry-run` | `plan_merge`, `plan_split` |
