# Quickstart

Four things end to end: convert a file, build a pyramid, run an
algorithm, export a subset. From the terminal first, then the same work
from Python.

## From the terminal

```bash
# 1. Convert a CSV point cloud into a store, with 100-unit chunks.
zvtools convert cells.csv cells.zarrvectors --chunk-shape 100,100,100

# 2. Add two coarser levels: each halves the object count and
#    reduces vertex detail eightfold.
zvtools pyramid cells.zarrvectors --coarsen 8,8 --sparsity 2,2

# 3. Check what you built.
zvtools info cells.zarrvectors

# 4. Confirm it conforms to the spec.
zvtools validate cells.zarrvectors --level 3
```

Full flag reference: [The `zvtools` CLI](cli.md).

## From Python

:::{warning}
`zarr_vectors_tools.ingest` and `zarr_vectors_tools.export` have **empty**
`__init__.py` files — there are no re-exports. Import from the concrete
module or you will get an `ImportError`:

```python
from zarr_vectors_tools.ingest.csv_points import ingest_csv   # correct
from zarr_vectors_tools.ingest import ingest_csv              # ImportError
```

`zarr_vectors_tools.algorithms` is the exception — it does re-export.
:::

### Ingest a point cloud

```python
from zarr_vectors_tools.ingest.csv_points import ingest_csv

result = ingest_csv(
    "cells.csv",
    "cells.zarrvectors",
    chunk_shape=(100.0, 100.0, 100.0),
    auto_detect_columns=True,      # find x/y/z and attribute columns by name
    drop_na=True,                  # discard rows with missing coordinates
    knn_distance_k=8,              # per-point kNN distance (needs [points-enrichment])
    per_object_vertex_count=True,  # per-object vertex counts
)
print(result["vertex_count"])
```

`ingest_csv` returns the underlying writer's summary dict, augmented with
enrichment counters such as `dropped_by_na`.

### Build a pyramid

```python
from zarr_vectors_tools.multiresolution.coarsen import build_pyramid

build_pyramid(
    "cells.zarrvectors",
    # One (coarsen, sparsity) tuple per coarser level.
    # coarsen: vertex reduction factor.  sparsity: object-drop divisor.
    factors=[(8.0, 2.0), (8.0, 2.0)],
    sparsity_strategy="spatial_coverage",  # keep a spatially even spread
    sparsity_seed=42,
)
```

Both factors are ≥ 1, and `1.0` opts out of that axis. The two are
genuinely different operations — see [Coarsening versus
sparsity](../multiresolution/concepts.md).

### Run an algorithm

```python
from zarr_vectors_tools.algorithms import compute_connected_components

summary = compute_connected_components(
    "graph.zarrvectors",
    write_back=True,   # store the labels as attributes/component_label/
)
print(summary["n_components"], summary["largest_component_size"])
```

Algorithms stream over the chunk grid rather than loading the store, so
this works on stores much larger than memory.

### Export a subset

```python
from zarr_vectors_tools.export.ply import export_ply

export_ply(
    "cells.zarrvectors",
    "preview.ply",
    level=2,                                  # a coarse level = a small preview
    bbox=((0, 0, 0), (500, 500, 500)),        # spatial filter
    attribute_names=["knn_distance"],
    binary=True,
)
```

Exporting from a coarser `level` is the quickest way to get something
small enough to open in a viewer.

## A skeleton round trip

```python
from zarr_vectors_tools.ingest.swc import ingest_swc
from zarr_vectors_tools.algorithms import compute_connected_components
from zarr_vectors_tools.export.swc import export_swc

# 1. Ingest an SWC skeleton, deriving Strahler order and node kinds.
ingest_swc(
    "neuron.swc",
    "neuron.zarrvectors",
    chunk_shape=(50.0, 50.0, 50.0),
    compute_strahler=True,
    compute_node_kind=True,
)

# 2. Label connected components and persist them on the store.
compute_connected_components("neuron.zarrvectors", write_back=True)

# 3. Write SWC back out.
export_swc("neuron.zarrvectors", "neuron_labelled.swc")
```

:::{warning}
This is a *geometry* round trip, not a byte-for-byte one. `export_swc`
reconstructs the parent column from the stored edges, but synthesises the
radius (`1.0`) and compartment type rather than restoring the values SWC
carried in. Header comments preserved at ingest are not read back
automatically either. See [Headers](../headers.md).
:::

## A tractography example

Streamlines are the case where the defaults matter most, because the
files are large:

```bash
zvtools convert tracts.trk tracts.zarrvectors \
    --num-chunks 5000 \
    --workers 12 --workers-backend dask \
    --compressor zstd \
    --coarsen 8,8 --sparsity 2,4 --sparsity-strategy length
```

Four things differ from the point-cloud case:

- `--num-chunks` instead of `--chunk-shape`. TRK is the only format that
  takes a target chunk *count* and derives the shape from the data bounds.
- `--workers` — TRK is the only format with a fully parallel ingest, and
  it builds its pyramid inline. See [Tractography at
  scale](../ingest/tractography_at_scale.md).
- `--compressor zstd` roughly halves the store (~2.4× measured on HCP
  tract coordinates). See [Choosing a
  compressor](../how_to/compressors.md).
- `--sparsity-strategy length` keeps the longest streamlines, which are
  usually the ones worth seeing at low resolution. It auto-enables
  `--compute-length`, and the CLI prints a `note:` when it does.

## Where to go next

| If you want to… | Go to |
| --- | --- |
| understand the format | [Getting started with Zarr Vectors](zarr_vectors.md) |
| see how the pieces compose | [Concepts](concepts.md) |
| convert a specific format | [Ingest workflows](../ingest/index.md) |
| tune a pyramid | [Coarsening versus sparsity](../multiresolution/concepts.md) |
| go faster | [Parallel workflows](../how_to/parallelism.md) |
| find a function | [Modules](../modules/index.md) |

## See also

- [Installation](installation.md)
- [The `zvtools` CLI](cli.md)
- [Examples](../examples.md) — the notebooks under `examples/`.
