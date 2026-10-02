# Streamlines

```bash
zvtools convert tracts.trk tracts.zv
```

TRK, TCK and TRX files become streamline stores: one object per streamline,
its points as vertices. TRK and TCK need the `[trk]` extra (nibabel), TRX the
`[trx]` extra (trx-python); see [Install](../install.md).

## TRK

`zvtools convert` streams a TRK file in byte ranges and never holds the whole
tractogram, so files larger than RAM work. The file's per-point scalars
become vertex attributes and its per-streamline properties object
attributes, under the names in the header.

```bash
zvtools convert tracts.trk tracts.zv --num-chunks 64 --workers 8 \
    --scratch-dir scratch --resume
```

| Option | Default | Effect |
| --- | --- | --- |
| `--num-chunks N` or `X,Y,Z` | `125` | Target total chunk count, or chunks per axis. The ingest picks the chunk edge in mm, prints it as `chunk_shape`, and the grid comes out close to N. TRK refuses `--chunk-shape`. |
| `--workers N` | one process | Worker processes. Without it the ingest runs serially. |
| `--n-parts N` | `max(4 × workers, 16)`, at most the streamline count | Byte ranges the file is cut into for the parallel pass. |
| `--scratch-dir DIR` | a temporary directory | Keep the part files and a progress record here. |
| `--resume` | off | Reuse what an earlier run with the same options finished in `--scratch-dir`. |
| `--apply-affine` | off | Store RAS mm instead of voxmm; see [Coordinates](#coordinates). |
| `--compute-length` | off | `object_attributes/length`, in mm. |
| `--compute-endpoints` | off | `object_attributes/start` and `end`, three columns each. |

Pass `--resume` and the same options on every run, including the first. A
rerun reuses the file scan, the part files, a finished level 0 and finished
pyramid levels, and refuses options that would change them. Choosing
`--num-chunks`, `--workers` and `--n-parts`: [Store layout](../store_layout.md),
[Large data](../large_data.md).

`--object-attr` and `--vertex-attr` (repeatable, TRK only) generate test and
demo attributes from the geometry, for trying colour-by on a file with no
scalars of its own:

```bash
zvtools convert tracts.trk demo.zv --object-attr tortuosity --vertex-attr arc_length
```

Per streamline: `length`, `endpoints`, `orientation` (start-to-end unit
vector, 3 columns), `tortuosity` (length ÷ endpoint distance). Each
streamline's `vertex_count` is stored without asking, as for every store with
objects ([attributes](../reference/headers_attributes.md)). Per vertex: `arc_length` (0 to 1 along the streamline), `x`,
`y`, `z`, `index`, `random` (seeded by `--attr-seed`), `tangent` (3 columns;
the viewer colours by single-column attributes, see [Visualise](../visualise.md)).

From Python, `zarr_vectors_tools.convert.ingest.trk_parallel.ingest_trk_parallel`
adds `max_streamlines` (ingest the first N only, for a trial run). It runs
serially unless given an `executor`, and unlike the CLI it builds a pyramid
by default (`pyramid_factors=[(8, 1), (8, 1)]`); pass
`build_multiscale=False` or factors of your own.

### Pyramid

The levels are built inside the ingest. What the values do, and how to
choose others: [Pyramids](../pyramids.md).

```bash
zvtools convert tracts.trk tracts.zv \
    --coarsen 1,1,1 --sparsity 2,2,2 --rdp-tolerance 0.5,1,2 --chunk-scale 2,2,2
```

## Coordinates

A TRK file stores points in **voxmm**: voxel index × voxel size, in mm along
the reference image's voxel axes. The header's vox-to-RAS affine maps them to
scanner **RAS mm**. TCK and TRX files store RAS mm. Every streamline store
records millimetres as its axis unit, which the viewer's scale bar and a
precomputed export read.

| Input | Stored positions | Affine kept in |
| --- | --- | --- |
| `zvtools convert x.trk` | voxmm, as in the file | TRK header (`space: voxmm`) and the store's `crs` |
| `zvtools convert x.trk --apply-affine` | RAS mm | TRK header (`space: rasmm`); `crs` is the identity |
| TCK | RAS mm | — |
| TRX | RAS mm | TRX header (`VOXEL_TO_RASMM`, `DIMENSIONS`) |
| Python `ingest_trk`, `ingest_linc_trk` | RAS mm | TRK header |

The viewer draws stored positions as they are; it does not apply the store's
affine. Use `--apply-affine` when the tracts must line up with data in RAS
mm, such as a TCK or TRX store or surfaces in scanner space. Keep the default
to keep the file's own coordinates. The exporters convert from either space.
Headers in full: [Headers and attributes](../reference/headers_attributes.md).

## TCK and TRX

```bash
zvtools convert tracts.tck tck.zv --chunk-shape 20,20,20 --compute-length
zvtools convert bundles.trx bundles.zv --chunk-shape 20,20,20 --compute-endpoints
```

Both readers load the whole file into memory, in one process, and need
`--chunk-shape` in mm. TCK carries positions only. TRX `dpv` become vertex
attributes, `dps` object attributes (one value per streamline when the array
has one column), `dpg` group attributes, and `groups` named groups, kept on
every pyramid level (`zv.open("bundles.zv").level(0).groups.names()` gives
`('CST_L', 'AF_R')`). `dpv` and `dps` keep the file's dtype, so a `uint32`
count exports as the same integers; `float16` is stored as float32.
The pyramid flags are the same as for TRK; the levels are built after the
ingest. From Python, `ingest_tck` and `ingest_trx` also take `length_range`,
and `ingest_trx` takes `mean_scalar`:

```python
from zarr_vectors_tools.convert.ingest.trx import ingest_trx

summary = ingest_trx(
    "bundles.trx", "bundles_25mm.zv",
    (20.0, 20.0, 20.0),           # chunk_shape, mm
    length_range=(25.0, 250.0),   # drop shorter and longer streamlines, mm
    mean_scalar="fa",             # object_attributes/mean_fa from dpv "fa"
)
summary["dropped_by_length"]      # 64
```

## LINC TRK

`ingest_linc_trk` (Python only, `[trk]` extra) reads a TRK whose
`data_per_streamline["label_id"]` holds an atlas label per streamline and
makes one group per label. It loads the whole file, in one process;
positions are RAS mm.

```python
from zarr_vectors_tools.convert.ingest.linc_trk import ingest_linc_trk

summary = ingest_linc_trk(
    "my_bundle.trk", "my_bundle.zv",
    (20.0, 20.0, 20.0),               # chunk_shape, mm
    lut_path="linc_lut.txt",          # lines of "ID NAME R G B A"
    mapping_path="linc_mapping.json", # {"my_bundle.trk": 34}
    compute_length=True,
)
```

- The input's file name must be in the mapping, and its label must occur in
  the file. Every label in the file needs a LUT line, such as
  `34 Left-Hippocampus 220 20 60 255`; names may contain spaces and `#`
  starts a comment. Integral float labels such as `34.0` are accepted.
- The group id is the label id: `level.groups.by_id(34).members`. The LUT
  name and RGBA colour become group attributes `name` and `color`; the group
  itself is unnamed, so TRX export calls it `group_34`.
- `compute_endpoints` and `length_range` are accepted too; groups are built
  from the streamlines that remain.

## Export to TRK and TRX

```bash
zvtools convert tracts.zv out.trk
zvtools convert tracts.zv three.trk --object-id 0 --object-id 1 --object-id 2 \
    --attribute fa --object-attribute weight
zvtools convert bundles.zv cst.trx --group-id 0
```

Vertex attributes become TRK scalars or TRX `dpv`, object attributes TRK
properties or TRX `dps`; TRX also gets the groups and their attributes. The
reference image comes from the store's TRK header, else its TRX header, else
the identity (Python `export_trk(..., affine=...)` supplies one).

- By default every numeric attribute is written; `--attribute` and
  `--object-attribute` choose. Text attributes are skipped and listed as
  `attributes_skipped`. TRK holds at most 10 scalars and 10 properties, with
  names of up to 20 characters.
- `--group-id` takes group ids, in the order of `groups.names()`; TRX export
  then writes only those groups. `--level N` exports a pyramid level.
- Positions are converted from the stored space, so a TRK exported from a
  voxmm or a RAS mm store of the same file matches the original.

Region selection, bundle summaries: [Streamlines and skeletons](../algorithms/streamlines_skeletons.md).
