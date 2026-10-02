# Converting files

`zvtools convert INPUT OUTPUT` moves data between a file and a Zarr Vectors
store. INPUT sets the direction:

- A file, a directory or a URL is **ingested** into a new store at OUTPUT. The
  format comes from INPUT's extension or `--format`. A directory is a FreeSurfer
  subject, a directory of `.gii` files or a precomputed layer (holds `info`); a
  URL is a precomputed layer.
- A store (a directory holding `zarr.json`) is **exported** to the file at
  OUTPUT. The format comes from OUTPUT's extension or `--format`.

```bash
zvtools convert points.csv points.zv --chunk-shape 250,250,250   # ingest
zvtools convert points.zv points.ply                             # export
```

:::{warning}
`table`, `lines` and `edgelist` have no extension: name them with `--format`.
Without it, a `.csv` file is read as `csv` points (first three columns as x, y, z), with no error.
:::

## Formats

| Format | Extensions | In / out | Geometry | Extra | Page |
|---|---|---|---|---|---|
| `csv` | `.csv`, `.xyz` | in, out | points | — | [Points](points.md) |
| `ply` | `.ply` | in, out | points | `ply` | [Points](points.md) |
| `las` | `.las`, `.laz` | in | points | `las` | [Points](points.md) |
| `h5ad` | `.h5ad` | in, out | points | `h5ad` | [Single cell](single_cell.md) |
| `table` | none | in | points | — | [Single cell](single_cell.md) |
| `trk` | `.trk` | in, out | streamlines | `trk` | [Streamlines](streamlines.md) |
| `trx` | `.trx` | in, out | streamlines | `trx` | [Streamlines](streamlines.md) |
| `tck` | `.tck` | in | streamlines | `trk` | [Streamlines](streamlines.md) |
| `swc` | `.swc` | in, out | skeleton | — | [Skeletons](skeletons.md) |
| `precomputed` | a URL, or a directory with `info` | in, out | skeleton or mesh | `precomputed` | [Skeletons](skeletons.md), [Meshes](meshes.md) |
| `obj` | `.obj` | in, out | mesh | — | [Meshes](meshes.md) |
| `stl` | `.stl` | in | mesh | — | [Meshes](meshes.md) |
| `gifti` | `.gii`, or a directory of them | in, out | surface | `surfaces` | [Surfaces](surfaces.md) |
| `freesurfer` | a subject directory | in | surface | `surfaces` | [Surfaces](surfaces.md) |
| `graphml` | `.graphml` | in | graph | `graph` | [Graphs](graphs.md) |
| `edgelist` | none | in | graph | — | [Graphs](graphs.md) |
| `lines` | none | in | line segments | — | [Graphs](graphs.md) |

Install an extra with `pip install "zarr-vectors-tools[las]"` (`[all]` for every
one); a missing one stops the command with an error naming the extra to install.

## Ingest options

| Option | Effect |
|---|---|
| `--chunk-shape X,Y,Z` | Level-0 chunk edge per axis, in the input's coordinate units. Required except for trk (`--num-chunks`) and a precomputed skeleton layer with a spatial index. See [Store layout](../store_layout.md#chunk-shape). |
| `--bin-shape X,Y,Z` | Bin edge inside a chunk. The chunk must be a whole multiple of it. Default: the chunk shape. See [](../store_layout.md#bin-shape). |
| `--dtype` | Stored position dtype. Default `float32`, the only one the viewer reads. See [](../store_layout.md#position-dtype). |
| `--overwrite` | Delete the store at OUTPUT first; refuses a directory that is not a store. Without it, give OUTPUT a new path. |
| `--coarsen`, `--sparsity`, `--chunk-scale`, … | Build coarser levels right after the ingest. See [Pyramids](../pyramids.md). |
| `--shard N` | Pack chunks into shard files after writing. See [](../store_layout.md#sharding). |
| `--workers N` | Parallel processes for trk and precomputed ingest and for the pyramid build. See [Large data](../large_data.md). |

An option that does not apply to the input format is refused, not ignored:

```text
$ zvtools convert points.csv points.zv --chunk-shape 250,250,250 --compressor zstd
error: --compressor applies to trk input — not 'csv'
```

## Export options

| Option | Effect | Formats |
|---|---|---|
| `--level N` | Pyramid level to export. Default 0, the finest. | all |
| `--object-id ID` | Only these objects (repeatable). | all but `gifti` |
| `--bbox X0,Y0,Z0,X1,Y1,Z1` | Only what falls in the box: min corner, then max corner. | `csv`, `ply`, `h5ad`, `obj` |
| `--attribute NAME` | Per-vertex attributes to write (repeatable). `csv` and `ply` write none unless named; `h5ad`, `trk` and `trx` write every one. A name the level lacks is an error. | all but `obj`, `swc` |
| `--group-id ID`, `--object-attribute NAME` | Only these streamline groups; per-object attributes to write (both repeatable). | `trk`, `trx`; `--object-attribute` also `precomputed` |
| `--delimiter D` | Column separator: `,` (default; a space for `.xyz`), `tab`, or any character. | `csv` |
| `--hemisphere`, `--surface`, `--prefix` | Which surfaces to write, and a file-name prefix. | `gifti` |
| `--segment-id ID`, `--unit UNIT` | Segments to write; coordinate unit when the store records none. | `precomputed` |

Any other option is refused. Export replaces an existing OUTPUT file; `gifti` and
`precomputed` write a directory, so pass `--format` (or a URL for `precomputed`).
Every flag is listed in the [CLI reference](../reference/cli.md).

## Python entry points

Import each function from its module; the packages re-export nothing. Ingest
functions take `(input, output, chunk_shape, **options)` and return a summary
dict; `ingest_edgelist` takes the node CSV second, and `ingest_trk_parallel`
takes `num_chunks` instead of a chunk shape. Export functions take
`(store, output, *, level=0, **options)`.

| Format | `zarr_vectors_tools.convert.ingest.` | `zarr_vectors_tools.convert.export.` |
|---|---|---|
| `csv` / `ply` / `las` | `csv_points.ingest_csv` / `ply.ingest_ply` / `las.ingest_las` | `csv_points.export_csv` / `ply.export_ply` / — |
| `h5ad` / `table` | `h5ad.ingest_h5ad` / `cell_table.ingest_table` | `h5ad.export_h5ad` / — |
| `trk` / `trx` / `tck` | `trk_parallel.ingest_trk_parallel` / `trx.ingest_trx` / `tck.ingest_tck` | `trk.export_trk` / `trx.export_trx` / — |
| `swc` / `precomputed` | `swc.ingest_swc` / `precomputed.ingest_precomputed` | `swc.export_swc` / `precomputed.export_precomputed` |
| `obj` / `stl` | `obj.ingest_obj` / `stl.ingest_stl` | `obj.export_obj` / — |
| `gifti` / `freesurfer` | `gifti.ingest_gifti` / `freesurfer.ingest_freesurfer` | `gifti.export_gifti` / — |
| `graphml` / `edgelist` / `lines` | `graphml.ingest_graphml` / `edgelist.ingest_edgelist` / `lines.ingest_lines_csv` | — |
