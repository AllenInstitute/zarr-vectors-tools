# Converting files

`zvtools convert INPUT OUTPUT` moves data between a file and a Zarr Vectors
store. If INPUT is a file, a directory or a URL, it is **ingested** into a new
store at OUTPUT. If INPUT is a store (a directory holding `zarr.json`), it is
**exported** to the file at OUTPUT. The format comes from the file's extension,
or from `--format`.

```bash
zvtools convert points.csv points.zv --chunk-shape 250,250,250   # ingest
zvtools convert points.zv points.ply                             # export
```

:::{warning}
`table`, `lines` and `edgelist` have no extension: name them with `--format`.
Without it, a `.csv` file is read as `csv` points, with no error.
:::

## Formats

| Format | Input | In / out | Extra | Page |
|---|---|---|---|---|
| `csv` | `.csv`, `.xyz` | in, out | — | [Points](points.md) |
| `ply` | `.ply` | in, out | `ply` | [Points](points.md) |
| `las` | `.las`, `.laz` | in | `las` | [Points](points.md) |
| `h5ad` | `.h5ad` | in, out | `h5ad` | [Single cell](single_cell.md) |
| `table` | none | in | — | [Single cell](single_cell.md) |
| `trk` | `.trk` | in, out | `trk` | [Streamlines](streamlines.md) |
| `trx` | `.trx` | in, out | `trx` | [Streamlines](streamlines.md) |
| `tck` | `.tck` | in | `trk` | [Streamlines](streamlines.md) |
| `swc` | `.swc` | in, out | — | [Skeletons](skeletons.md) |
| `precomputed` | a URL, or a directory with `info` | in, out | `precomputed` | [Skeletons](skeletons.md), [Meshes](meshes.md) |
| `obj` | `.obj` | in, out | — | [Meshes](meshes.md) |
| `stl` | `.stl` | in | — | [Meshes](meshes.md) |
| `gifti` | `.gii`, or a directory of them | in, out | `surfaces` | [Surfaces](surfaces.md) |
| `freesurfer` | a subject directory | in | `surfaces` | [Surfaces](surfaces.md) |
| `graphml` | `.graphml` | in | `graph` | [Graphs](graphs.md) |
| `edgelist` | none | in | — | [Graphs](graphs.md) |
| `lines` | none | in | — | [Graphs](graphs.md) |

Install an extra with `pip install "zarr-vectors-tools[las]"` (`[all]` for
every one). A missing extra stops the command with an error that names it.

From Python, ingest functions live in
`zarr_vectors_tools.convert.ingest.<format>` and export functions in
`zarr_vectors_tools.convert.export.<format>`; the packages re-export nothing.
The modules for `csv` and `table` are `csv_points` and `cell_table`. The
functions are `ingest_<format>` and `export_<format>`, except
`trk_parallel.ingest_trk_parallel` and `lines.ingest_lines_csv`.

## Ingest options

| Option | Effect |
|---|---|
| `--chunk-shape X,Y,Z` | Level-0 chunk edge per axis, in the input's units. Required except for trk (`--num-chunks`) and a precomputed skeleton layer with a spatial index. See [Store layout](../store_layout.md#chunk-shape). |
| `--bin-shape X,Y,Z` | Bin edge inside a chunk; must divide the chunk shape. Default: the chunk shape. See [Bin shape](../store_layout.md#bin-shape). |
| `--dtype` | Stored position dtype. Default `float32`, the only one the viewer reads. See [Position dtype](../store_layout.md#position-dtype). |
| `--overwrite` | Delete the store at OUTPUT first. Refuses a directory that is not a store. |
| `--coarsen`, `--sparsity`, `--chunk-scale`, … | Build coarser levels after the ingest. See [Pyramids](../pyramids.md). |
| `--shard N` | Pack chunks into shard files. See [Sharding](../store_layout.md#sharding). |
| `--workers N` | Parallel processes for trk and precomputed ingest and the pyramid build. See [Large data](../large_data.md). |

An option that does not apply to the format is refused:

```text
$ zvtools convert points.csv points.zv --chunk-shape 250,250,250 --compressor zstd
error: --compressor applies to trk input — not 'csv'
```

## Export options

| Option | Effect | Formats |
|---|---|---|
| `--level N` | Pyramid level to export. Default 0, the finest. | all |
| `--object-id ID` | Only these objects (repeatable). | all but `gifti`, `obj` |
| `--bbox X0,Y0,Z0,X1,Y1,Z1` | Only what falls in the box (min corner, max corner). | `csv`, `ply`, `h5ad`, `obj` |
| `--attribute NAME` | Vertex attributes to write (repeatable). `csv` and `ply` write none unless named; `h5ad`, `trk` and `trx` write all. A missing name is an error. | all but `obj`, `swc` |
| `--group-id ID` | Only these streamline groups (repeatable). | `trk`, `trx` |
| `--object-attribute NAME` | Object attributes to write (repeatable). | `trk`, `trx`, `precomputed` |
| `--delimiter D` | `,` (default; a space for `.xyz`), `tab`, or any character. | `csv` |
| `--hemisphere`, `--surface`, `--prefix` | Surfaces to write, and a file-name prefix. | `gifti` |
| `--segment-id ID`, `--unit UNIT` | Segments to write; unit when the store records none. | `precomputed` |

Export replaces an existing OUTPUT file. `gifti` and `precomputed` write a
directory, so pass `--format` (or a URL for `precomputed`). Every flag is in
the [CLI reference](../reference/cli.md).
