# Installation

`zarr-vectors-tools` requires Python ≥ 3.10 and pulls in `zarr-vectors`,
`numpy`, and `pandas` as core dependencies. Every file format and every
optional enrichment is gated behind an extra so a minimal install stays
slim.

## Base install

```bash
pip install zarr-vectors-tools
```

This is enough for CSV point clouds, CSV line segments, the SWC and OBJ
ingest paths (pure-Python parsers), and every algorithm in
`zarr_vectors_tools.algorithms`.

## Optional extras

Pick the extras that match the file formats you need:

| Extra | Installs | Enables |
| --- | --- | --- |
| `las` | `laspy>=2.4` | LAS / LAZ point cloud ingest |
| `ply` | `plyfile>=1.0` | PLY point cloud ingest **and** export |
| `streamlines` | `nibabel>=5.0`, `trx-python>=0.3` | TRK, TCK, TRX streamline ingest and TRK/TRX export |
| `graph` | `networkx>=3.0` | GraphML ingest and the `compute_degree` / `compute_clustering` enrichments on edgelist + graphml ingest |
| `points-enrichment` | `scipy>=1.10` | `knn_distance` attribute enrichment for point clouds |
| `gpu` | `cudf` | GPU-accelerated edgelist CSV reading (requires a RAPIDS-compatible CUDA environment) |
| `all` | union of `las`, `ply`, `streamlines`, `graph`, `points-enrichment` | Everything except `gpu` |

Install a single extra:

```bash
pip install "zarr-vectors-tools[streamlines]"
```

Install everything (recommended for exploration):

```bash
pip install "zarr-vectors-tools[all]"
```

The `gpu` extra requires a working RAPIDS install — follow the
[RAPIDS installation guide](https://docs.rapids.ai) for your CUDA
version rather than letting pip resolve it.

## Development install

```bash
git clone https://github.com/Andrew-Keenlyside/zarr-vectors-tools
cd zarr-vectors-tools
pip install -e ".[all,dev]"
```

The `dev` extra adds `pytest`, `ruff`, and `jupyter`.

## Verifying the install

```python
import zarr_vectors_tools
from zarr_vectors_tools.algorithms import compute_connected_components
```

If `zarr-vectors-tools` imported but a specific ingest fails with
`ImportError: No module named 'laspy'` (or similar), install the matching
extra — the error message names the missing package.

## See also

- [Quickstart](quickstart.md) — write your first store end-to-end.
- [Concepts](concepts.md) — how ingest, algorithms, and export compose.
