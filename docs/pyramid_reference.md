# Pyramid reference

The Python side of [Pyramids](pyramids.md): which coarsener handles each
geometry, the options `build_pyramid` takes beyond the CLI's, and the builders
that only exist in Python. Exact signatures are in the
[API reference](api/multiresolution.rst).

## Coarseners

`build_pyramid` picks a coarsener from the store's geometry. Pass `method=` (CLI
`--method`) to choose one yourself; `coarsener_keys()` lists them.

| `method` | Chosen for | `--coarsen` value is | Sparsity strategies |
|---|---|---|---|
| `per_object` | points, graphs, lines | multiplier of the bin edge (graphs merge vertices only within a connected component) | random, length, spatial_coverage, point_thinning, group, attribute |
| `polyline` | streamlines | sets the Douglas–Peucker tolerance (rdp), or a stride (decimate) | random, length, group, attribute |
| `skeleton` | skeletons | vertex stride along each branch | length (random becomes length), attribute |
| `mesh` | meshes | multiplier of the median edge length (vertex clustering) | random, attribute |
| `mesh_decimate` | meshes, by `--method` / `method=` only (recommended) | on-disk size reduction per level, > 1 | none |
| `per_fragment` | by `--method` / `method=` only | multiplier of the bin edge, per chunk fragment | random, attribute |

`zarr_vectors_tools.multiresolution.coarsen.register_coarsener(key, fn)` adds a
coarsener under a new key. Each coarse level records its own parameters (bin,
chunk, method, and the RDP tolerance or skeleton stride), which is what
`rebuild_pyramid_from_level` re-runs a level with.

## `build_pyramid` options not on the CLI

| Option | Effect |
|---|---|
| `method` | coarsener key from the table above |
| `sparsity_seed` | seed for random selection, so a rebuild keeps the same objects |
| `executor` | run chunks in parallel: `zarr_vectors_tools.parallel.process_pool_executor(n)` or `dask_executor(n)`, as a context manager (see [Large data](large_data.md)) |

`build_pyramid` returns a summary with one entry per level in `level_specs`.
Its `cross_level_storage` and `cross_level_depth` describe the links between
levels that were actually written, as the store's root metadata records them:
`"none"` and 0 for geometries whose coarsener writes none.

## Builders outside `build_pyramid`

| Function | Use it for |
|---|---|
| `multiresolution.strategies.points.build_point_subset_pyramid(store, levels=5, divisor=8.0, seed=0)` | point clouds without object ids: each level is a random 1/divisor of the one above, with every per-vertex attribute carried over |
| `multiresolution.strategies.skeletons.build_skeleton_pyramid(store, strides=[8, 8], chunk_scale_factors=[2, 2])` | skeletons, with skeleton-only options: `attr_agg`, `drop_interior_below` |
| `multiresolution.strategies.mesh_decimate_level.build_mesh_pyramid_to_floor(store, 4.0)` | meshes: adds 4x-smaller levels until the surface cannot shrink further or `min_faces` (default 256) is reached |
| `multiresolution.refresh.rebuild_pyramid_from_level(root, 0)` | rebuilding every level above a level that changed, with the parameters each level was built with; takes an open root group |

All paths are under `zarr_vectors_tools.`. For example:

```python
from zarr_vectors.building import open_store
from zarr_vectors_tools.multiresolution.refresh import rebuild_pyramid_from_level
from zarr_vectors_tools.multiresolution.strategies.mesh_decimate_level import (
    build_mesh_pyramid_to_floor,
)

# Re-run every coarse level after editing level 0 in place.
rebuild_pyramid_from_level(open_store("cells.zv", mode="r+"), 0)

# A 10,242-vertex sphere reached 5120, 1280 and 320 faces before min_faces stopped it.
build_mesh_pyramid_to_floor("sphere.zv", 4.0, verbose=False)
```

## Helpers that `build_pyramid` does not use

These work on arrays in memory, not on stores, and are not called by
`build_pyramid`: `coarsen_points`, `coarsen_points_store`
(`strategies.points`), `coarsen_graph` (`strategies.graphs`), `simplify_skeleton`, `decimate_skeleton`
(`strategies.skeletons`), `coarsen_mesh_cluster`, `coarsen_mesh_quadric`
(`strategies.meshes`, `[mesh]` extra), `decimate_mesh` (`strategies.mesh_decimate`)
and the selection functions in `multiresolution.object_selection`. Use them to
experiment, or to write your own coarsener.
