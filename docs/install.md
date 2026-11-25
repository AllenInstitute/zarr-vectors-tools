# Install

```bash
pip install zarr-vectors-tools
zvtools --version          # zvtools (zarr-vectors-tools 0.3.0)
```

This needs Python 3.11 or newer. It installs the `zvtools` command and the core
package, [zarr-vectors](https://github.com/AllenInstitute/zarr-vectors-py).

## Optional extras

The base install converts CSV point tables, line CSVs, edge-list CSVs, SWC, OBJ
and STL. Other formats need an extra:

```bash
pip install "zarr-vectors-tools[trk,h5ad]"    # one or more extras
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
| `mesh` | per-vertex attributes on `mesh_decimate` pyramid levels; `coarsen_mesh_quadric` | pyfqmr, scipy |
| `precomputed` | Neuroglancer precomputed ingest and export (multi-resolution mesh layers also need DracoPy) | cloud-volume, mapbuffer, cloud-files |
| `parallel` | `--workers-backend dask` | dask[distributed] |
| `recipes` | YAML recipes for `zvtools run` | pyyaml |
| `gpu` | GPU CSV reading in `ingest_edgelist(..., use_cudf=True)`; install RAPIDS first | cudf |

A command that needs a missing extra stops with an error naming it.

## Core package version

The core must write Zarr Vectors format 0.10 and provide `zarr_vectors.building`,
or `import zarr_vectors_tools` fails. If it does not, install the latest core
from GitHub:

```bash
pip install "git+https://github.com/AllenInstitute/zarr-vectors-py"
pip install "git+https://github.com/AllenInstitute/zarr-vectors-tools"
```
