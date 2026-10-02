# Merge and split stores

`zvtools merge TARGET SOURCES...` appends the objects of files and stores to an
existing store without rewriting what is there. `zvtools split STORE OUTDIR`
writes one new store per group, attribute value or merged source.

## Merge

```bash
zvtools merge tracts.zv subject_b.trk --dry-run    # plan only: ids, fit
zvtools merge tracts.zv subject_b.trk --on-out-of-bounds expand
```

Incoming objects get new ids after the target's last one (the `id_offset`
reported per source). With them, merge carries:

| What | How |
|---|---|
| Object attributes | Appended. A column one side lacks is filled with NaN (float), the type's minimum (signed int) or maximum (unsigned int). `length`, `start`, `end` and `vertex_count` are instead computed from the incoming geometry when the target has them. |
| Vertex attributes | The target's set; a name the source lacks is filled the same way, in the target column's type. A name new to a non-empty target is skipped and listed under `skipped_vertex_attributes`. |
| Groups | Renumbered to the new ids. A name the target already has is extended; `--group-prefix b_` keeps incoming groups separate (`b_cst`). |
| Headers | Copied. A second header of one format is kept as `<format>@<source>`, after the source's file or store name (`trk@subject_b`); exporters read only the first. |
| Provenance | Sources and id offsets are recorded for `split --by provenance`. |

Streamlines, lines and point clouds with object ids can be merged; meshes,
skeletons, graphs and mixed geometry types are refused. Category codes (table,
h5ad) are copied as numbers, not re-coded between sources. TRK, TCK and TRX
files are read directly; other formats are first converted on the target's
chunk shape (or `--cell-size` with `--create`). A point cloud with no object ids
is refused (there are no objects to move), as is a format that needs extra
options, such as `table`. Convert those first, then merge the store:

```bash
zvtools convert cells_b.csv cells_b.zv --format table --position-columns x,y,z \
    --key-column cell_id --object-id-column sample --chunk-shape 250,250,250
zvtools merge cells.zv cells_b.zv --source-attr source
```

`--source-attr NAME` stores each new object's source index (float32, 0 for this
merge's first source; NaN for older objects). `--create` makes a missing target
with chunks of `--cell-size X,Y,Z` (default: the first store source's, else 1/64
of the sources' extent per axis). A created target is stored like the first store
source (compression, bins per chunk), else uncompressed with one bin per chunk,
as `convert` writes it. `--format` names file formats.

### Fitting the grid

`--on-out-of-bounds` handles a source outside the target's chunks: `raise`
(default) refuses and prints the overhang per axis, `skip` drops the objects
outside, `expand` adds chunks above the grid. The grid starts at the target's
first chunk, which can lie below coordinate 0. `expand` cannot add chunks below
it, because every chunk written is numbered from it: a source reaching below is
refused before anything is written. `--transform b_to_a.npy` moves every
source by a 4 × 4 affine (`.npy`, `.json` nested list, or 16 comma-separated
numbers, row-major). TRK sources are read in stored voxmm coordinates, matching
a target converted without `--apply-affine`; for one converted with it, pass
`--space ras`.

### Label columns as named groups

```bash
zvtools merge atlas.zv atlas.trk --create --cell-size 10,10,10 \
    --group-by label_id --lut atlas_lut.json
```

`--group-by` turns a per-streamline label into groups, named from a JSON
`{"name": code}` `--lut` minus file extensions: `{"cst.trk": 21, "af.trk": 22,
"ifof.trk": 23}` gives `cst`, `af`, `ifof`; other codes become `label_id_<code>`.
`--group-by`, `--lut` and `--space` apply to TRK, TCK and TRX files; they are
refused for stores and other formats.

### Pyramid after a merge

`--pyramid rebuild` (default) rebuilds each coarser level with the settings it
records (coarsening, chunk scale, tolerance, sparsity) and `--sparsity-strategy`,
which is not recorded: pass the original (default `random`; `group` keeps every
named group). `length` needs the target to have it (convert with
`--compute-length`); incoming streamlines get theirs computed.
`--pyramid-coarsen` with `--pyramid-sparsity` (and
`--coarsen-mode`) build a different pyramid, and are refused with `drop` or
`keep`. `drop` removes the levels, `keep` leaves them marked stale. For any other change, merge with `--pyramid drop` and
run `zvtools pyramid` with the new flags ([Pyramids](pyramids.md)).

## Split

```bash
zvtools split atlas.zv bundles/ --dry-run     # part names and object counts
zvtools split atlas.zv bundles/               # bundles/atlas_cst.zv, ...
zvtools split tracts.zv parts/ --by provenance
zvtools split atlas.zv by_label/ --by attribute --attribute label_id --lut atlas_lut.json
```

Each part, `OUTDIR/<store>_<part>.zv`, keeps attributes, groups and headers.
Characters other than letters, digits, `.`, `-` and `_` in a part name become
`_`, and a name used twice gets `_2`.

| Option | Meaning |
|---|---|
| `--by groups` | One part per named group (default). |
| `--by attribute --attribute NAME` | One part per value of an object attribute; `--lut FILE` names the values. |
| `--by provenance` | One part per source an earlier merge added, named after its file or store (`tracts_subject_b.zv`), plus `original`. |
| `--level N` | Split level N instead of level 0. |
| `--bounds source` | Keep the parent's grid (default). `fit` allocates only the chunks a part's contents need; they are still the parent's chunks. |
| `--min-objects N`, `--overwrite` | Skip parts with fewer than N objects; replace parts that exist (otherwise an error). |
| `--pyramid rebuild` | Give each part the parent's coarser levels, rebuilt from the part with each level's recorded settings (coarsening, chunk scale, tolerance, sparsity). Needs a split of level 0. Default `drop`: level 0 only. |
| `--pyramid-coarsen`, `--pyramid-sparsity` | Build these levels instead of the parent's, as for merge; implies `--pyramid rebuild`. `--sparsity-strategy` and `--coarsen-mode` apply as for merge. |

## Python

```python
import numpy as np
from zarr_vectors_tools.compose import FileSource, merge_stores, plan_merge, split_store

shift = np.eye(4)
shift[:3, 3] = -25.0   # subject_b's frame -> the target's
print(plan_merge("tracts.zv", [FileSource("subject_b.trk", transform=shift)]))
merge_stores("tracts.zv", [FileSource("subject_b.trk", transform=shift)],
             on_out_of_bounds="expand", pyramid="drop")
split_store("atlas.zv", "bundles", by="groups",
            pyramid="rebuild", pyramid_factors=[(1, 2)])
```

Keywords follow the CLI options, plus `pyramid_factors=[(coarsen, sparsity)]`,
`pyramid_options={"sparsity_strategy": ..., "coarsen_mode": ...}` and
`split_store(..., by="objects", parts={name: [ids]})`. `StoreSource(path, transform=...)`,
`GeometrySource` and `plan_split(store, by=...)` also exist. `plan_merge` closes
its sources, so build new ones for the merge.
