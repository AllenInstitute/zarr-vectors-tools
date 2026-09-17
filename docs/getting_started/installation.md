# Installation

`zarr-vectors-tools` requires **Python ≥ 3.11** and pulls in
`zarr-vectors`, `zarr>=3.0`, `numpy` and `pandas` as core dependencies.
Every file format and every optional enrichment is gated behind an extra,
so a minimal install stays slim.

## Base install

```bash
pip install zarr-vectors-tools
```

That covers CSV point clouds, CSV line segments, the pure-Python SWC,
OBJ and STL parsers, the whole `multiresolution` layer, and every
algorithm in `zarr_vectors_tools.algorithms`.

:::{warning}
**The core dependency is not yet on PyPI at the version this package
needs.** `zarr-vectors-tools` requires the merged
`links/<delta>/<offsets>/` layout — on-disk format {{ zv_version }} — in
which connectivity is a single family and there is no
`cross_chunk_links/` to fall back to.

That format ships in the current core *development* line, whose package
version is still `0.2.x`. Format version and package version are
independent, so the `zarr-vectors>=0.2.1.dev0` pin can only exclude the
feature-less `0.2.0` release; it cannot by itself guarantee a
merged-links core. **The real guarantee is an editable install of a core
working tree that carries it:**

```bash
git clone https://github.com/BRIDGE-Neuroscience/zarr-vectors-py
pip install -e ./zarr-vectors-py
pip install -e ./zarr-vectors-tools
```

If ingest fails with missing-attribute errors around links or manifests,
this is almost certainly why. Check with `zvtools info STORE` — the
`zv_version` line should start with `0.9`.
:::

## Optional extras

| Extra | Installs | Enables |
| --- | --- | --- |
| `las` | `laspy>=2.4` | LAS / LAZ point cloud ingest |
| `ply` | `plyfile>=1.0` | PLY point cloud ingest **and** export |
| `trk` | `nibabel>=5.0` | TCK / TRK ingest; TRK export. Pure Python — installs under Pyodide |
| `trx` | `trx-python>=0.3` | TRX ingest and export |
| `streamlines` | `trk` + `trx` | both of the above (back-compat alias) |
| `graph` | `networkx>=3.0` | GraphML ingest and the `degree` / `clustering` enrichments |
| `points-enrichment` | `scipy>=1.10` | the `knn_distance` attribute on point clouds |
| `mesh` | `pyfqmr>=0.2`, `scipy>=1.10` | quadric mesh decimation in the pyramid coarseners |
| `precomputed` | `cloud-volume>=8.0`, `mapbuffer>=0.7`, `cloud-files>=4.0` | precomputed / CloudVolume EM skeleton ingest |
| `parallel` | `dask[distributed]>=2024.0` | the dask executor backend (`--workers-backend dask`) |
| `gpu` | `cudf` | GPU edge-list CSV reading; needs a RAPIDS CUDA environment |
| `all` | everything above **except `gpu`** | |
| `dev` | `pytest>=7.0`, `ruff>=0.4`, `jupyter>=1.0` | the test and lint toolchain |

```bash
pip install "zarr-vectors-tools[trk]"              # one extra
pip install "zarr-vectors-tools[all]"              # everything but gpu
```

:::{note}
The stdlib process-pool backend needs **no** extra — `--workers 8` works
on a base install. The `parallel` extra is only for
`--workers-backend dask`. See [Parallel
workflows](../how_to/parallelism.md).
:::

The `gpu` extra needs a working RAPIDS install. Follow the [RAPIDS
installation guide](https://docs.rapids.ai) for your CUDA version rather
than letting pip resolve `cudf` on its own.

## Development install

```bash
git clone https://github.com/AllenInstitute/zarr-vectors-tools
cd zarr-vectors-tools
pip install -e ".[all,dev]"
```

### Tests and lint

The suite runs in two tiers. End-to-end TRK ingests and tests that start
worker processes carry the `slow` marker; everything else is the fast
tier:

```bash
pytest -m "not slow"            # fast tier: what CI runs on every push
pytest                          # full tier: everything, ~30 minutes
ruff check zarr_vectors_tools   # lint the package (tests/ has known cosmetic violations)
```

A new test that runs a full ingest or opens a process pool belongs in the
slow tier: mark it `@pytest.mark.slow`.

CI (`.github/workflows/tests.yml`) runs lint and the fast tier on every
pull request and on pushes to the long-lived branches, on Ubuntu with
Python 3.11 and 3.13 and on Windows with 3.13. The full tier runs nightly
on Ubuntu and Windows, or on demand from the Actions tab. CI installs
`zarr-vectors` from git at the commit pinned in that workflow's `CORE_SHA`
(see the warning above for why); bump it there when this package starts
to need a newer core. An optional reader that will not install on a runner
is flagged in the run summary, and its tests skip.

## Verifying the install

```python
import zarr_vectors_tools
from zarr_vectors_tools.algorithms import compute_connected_components
```

and from the terminal:

```bash
zvtools --version
```

If the package imports but a specific ingest fails with `ImportError: No
module named 'laspy'` or similar, install the matching extra — the CLI
error message names the exact `pip install` command.

## See also

- [Getting started with Zarr Vectors](zarr_vectors.md) — what the format is for.
- [Quickstart](quickstart.md) — write your first store end-to-end.
- [The `zvtools` CLI](cli.md)
