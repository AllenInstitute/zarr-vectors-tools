# Streamlines

Convert TRK, TCK and TRX tractograms to streamline stores and back. Chunk
shape, `--overwrite` and the export options are in
[Converting files](index.md).

| Format | In / out | Extra | Positions | Memory |
|---|---|---|---|---|
| `trk` | in, out | `trk` | voxmm (RAS mm with `--apply-affine`) | streams the file; larger than RAM works |
| `tck` | in | `trk` | RAS mm | loads the whole file |
| `trx` | in, out | `trx` | RAS mm | loads the whole file |

```bash
zvtools convert tracts.trk tracts.zv --num-chunks 64 --workers 8
```

```python
from zarr_vectors_tools.convert.ingest.trk_parallel import ingest_trk_parallel

ingest_trk_parallel("tracts.trk", "tracts.zv", num_chunks=64, build_multiscale=False)
```

TRK takes `--num-chunks`, not `--chunk-shape`. TCK and TRX need
`--chunk-shape` in mm. The CLI builds a pyramid only when given `--coarsen` or
`--sparsity`, and its `--sparsity-strategy` default is `random`. In Python,
`ingest_trk_parallel` runs serially unless given an `executor`, builds a
pyramid unless `build_multiscale=False`, and keeps the longest streamlines
(`sparsity_strategy="length"`).

| Option | Format | Effect |
|---|---|---|
| `--num-chunks N` or `X,Y,Z` | trk | Target chunk count (default 125), or chunks per axis. The chunk edge it picks is printed as `chunk_shape`. |
| `--workers N`, `--n-parts N` | trk | Worker processes, and the byte ranges the file is cut into. See [Large data](../large_data.md). |
| `--scratch-dir DIR`, `--resume` | trk | Keep part files here; with `--resume`, a rerun with the same options reuses them. Pass both on every run, including the first. |
| `--apply-affine` | trk | Store RAS mm instead of voxmm; see [Coordinates](#coordinates). |
| `--compute-length` | all | `object_attributes/length`, in mm. |
| `--compute-endpoints` | all | `object_attributes/start` and `end`, 3 columns each. |
| `--object-attr`, `--vertex-attr` | trk | Generated demo attributes (`tortuosity`, `arc_length`, …) for files with no scalars of their own. |

## What the store holds

- One object per streamline; its points are the object's vertices.
- TRK scalars and TRX `dpv` as vertex attributes; TRK properties and TRX `dps`
  as object attributes. TRX keeps its dtypes (float16 becomes float32). A TRK
  property of one value per streamline is one column, shape (N,).
- TRX `groups` as named groups and `dpg` as group attributes, on every level.
- The TRK or TRX header. Every streamline ingest records millimetres as the
  axis unit.
- Every name: [Headers and attributes](../reference/headers_attributes.md).

## Coordinates

A TRK file stores points in **voxmm**: voxel index × voxel size. Its
vox-to-RAS affine maps them to scanner **RAS mm**. TCK and TRX store RAS mm.
The viewer draws stored positions as they are. Use `--apply-affine` when TRK
tracts must line up with data in RAS mm, such as TCK or TRX stores or surfaces.
Exporters convert from either space.

## LINC TRK

`ingest_linc_trk` (Python only) reads a TRK whose
`data_per_streamline["label_id"]` holds an atlas label, and makes one group per
label. Its `lut_path` file has lines `ID NAME R G B A`; its `mapping_path` JSON
maps the file name to its label, such as `{"my_bundle.trk": 34}`. The group id
is the label; its name and colour become group attributes.

## Export to TRK and TRX

```bash
zvtools convert bundles.zv cst.trx --group-id 0
```

Vertex attributes become TRK scalars or TRX `dpv`; object attributes become
TRK properties or TRX `dps`. Text attributes are skipped. TRK holds at most 10
scalars and 10 properties, with names of up to 20 characters. The reference
image comes from the store's TRK header, else its TRX header, else the identity.

Region selection and bundle summaries:
[Streamlines and skeletons](../algorithms/streamlines_skeletons.md).
