# Install

```bash
pip install zarr-vectors-tools
zvtools --version          # zvtools (zarr-vectors-tools 0.3.0)
```

This needs Python 3.11 or newer. It installs the `zvtools` command and the core
package, `zarr-vectors`.

## Optional extras

The base install converts CSV point tables (`csv`, `table`), line CSVs
(`lines`), edge-list CSVs (`edgelist`), SWC, OBJ and STL. Other formats and
features need an extra:

```bash
pip install "zarr-vectors-tools[trk]"         # one extra
pip install "zarr-vectors-tools[trk,h5ad]"    # several
pip install "zarr-vectors-tools[all]"         # every extra except gpu
```

| Extra | Enables | Installs |
|---|---|---|
| `trk` | TRK and TCK ingest; TRK export | nibabel |
| `trx` | TRX ingest and export | trx-python |
| `streamlines` | `trk` and `trx` together | nibabel, trx-python |
| `surfaces` | GIFTI and FreeSurfer ingest; GIFTI export; CIFTI through `zvtools attach` | nibabel |
| `las` | LAS and LAZ ingest | laspy, lazrs |
| `ply` | PLY ingest and export | plyfile |
| `h5ad` | AnnData `.h5ad` ingest and export | anndata |
| `graph` | GraphML ingest; per-node metrics in Python `ingest_edgelist` | networkx |
| `points-enrichment` | `--knn-distance-k` on point ingest | scipy |
| `mesh` | per-vertex attributes on `mesh_decimate` pyramid levels; quadric decimation in `coarsen_mesh_quadric` | pyfqmr, scipy |
| `precomputed` | Neuroglancer precomputed skeleton and mesh ingest (multi-resolution mesh layers also need DracoPy); precomputed export | cloud-volume, mapbuffer, cloud-files |
| `parallel` | `--workers-backend dask` | dask[distributed] |
| `recipes` | YAML recipes for `zvtools run` | pyyaml |
| `gpu` | GPU CSV reading in Python `ingest_edgelist(..., use_cudf=True)`. Install RAPIDS first | cudf |
| `all` | every extra above except `gpu` | |

If an extra is missing, the command stops with an error that names it, for
example `laspy is required for LAS/LAZ ingest. Install with: pip install
'zarr-vectors-tools[las]'`.

## The core package

This package builds on [zarr-vectors](https://github.com/AllenInstitute/zarr-vectors-py),
which defines the Zarr Vectors format and reads and writes stores; pip installs
it as a dependency. The core must write Zarr Vectors format 0.9 and provide the
`zarr_vectors.building` API, or `import zarr_vectors_tools` fails. If an
installed core lacks either, install the latest one from GitHub (below).

## Install from GitHub

```bash
pip install "git+https://github.com/AllenInstitute/zarr-vectors-tools"
pip install "zarr-vectors-tools[trk] @ git+https://github.com/AllenInstitute/zarr-vectors-tools"
pip install "git+https://github.com/AllenInstitute/zarr-vectors-py"   # the core package
```
