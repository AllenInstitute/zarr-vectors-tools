# Pyramid reference

Coarseners, `build_pyramid` options and Python-only builders. For choosing
values, see [Pyramids](pyramids.md); for signatures, the
[API reference](api/multiresolution.rst). Paths are under `zarr_vectors_tools.`.

## Coarseners

`build_pyramid` picks a coarsener from the store's geometry. `method=` (CLI
`--method`) picks one yourself; `coarsener_keys()` lists them.

| `method` | Chosen for | `--coarsen` value is | Sparsity strategies |
|---|---|---|---|
| `per_object` | points, graphs, lines | multiplier of the bin edge | random, length, spatial_coverage, point_thinning, group, attribute |
| `polyline` | streamlines | sets the Douglas–Peucker tolerance (rdp), or a stride (decimate) | random, length, group, attribute |
| `skeleton` | skeletons | vertex stride along each branch | length (random becomes length), attribute |
| `mesh` | meshes | multiplier of the median edge length (vertex clustering) | random, attribute |
| `mesh_decimate` | meshes, by `method` only (recommended) | size reduction per level, > 1 | none |
| `per_fragment` | by `method` only | multiplier of the bin edge, per chunk fragment | random, attribute |

`multiresolution.coarsen.register_coarsener(key, fn)` adds a coarsener. Each
level records its build parameters for `rebuild_pyramid_from_level`.

## `build_pyramid` options not on the CLI

| Option | Effect |
|---|---|
| `method` | coarsener key from the table above |
| `sparsity_seed` | seed for random selection, so a rebuild keeps the same objects |
| `executor` | run chunks in parallel in the skeleton and streamline coarseners; see [Large data](large_data.md) |
| `refinement` | `"replace"` (default) or `"add"` ([additive pyramids](pyramids.md#storing-each-object-once-additive-pyramids)); `"add"` raises `ValueError` unless `multiresolution.additive.supports_additive()` is `True` |

The return value is a summary with `levels_created`, `level_specs` (one per
level, naming the coarsener that ran), `refinement`, and the
`cross_level_storage` and `cross_level_depth` that were written. With
`"add"` it also has an `additive` report per level.

## Builders outside `build_pyramid`

| Function | Use it for |
|---|---|
| `multiresolution.strategies.points.build_point_subset_pyramid(store, levels=5, divisor=8.0, seed=0)` | point clouds without object ids: each level a random 1/divisor of the one above; `refinement="add"` supported |
| `multiresolution.additive.make_pyramid_additive(store)` | converting a finished pyramid to an additive one in place |
| `multiresolution.strategies.skeletons.build_skeleton_pyramid(store, strides=[8, 8], chunk_scale_factors=[2, 2])` | skeletons, with skeleton-only options `attr_agg` and `drop_interior_below` |
| `multiresolution.strategies.mesh_decimate_level.build_mesh_pyramid_to_floor(store, 4.0)` | meshes: adds 4x-smaller levels until the surface cannot shrink or `min_faces` (default 256) is reached |
| `multiresolution.refresh.rebuild_pyramid_from_level(root, 0)` | rebuilding every level above one that changed, from an open root group (`zarr_vectors.building.open_store(path, mode="r+")`). Refuses levels no registered coarsener wrote, such as `build_point_subset_pyramid`'s |

## Helpers that `build_pyramid` does not use

In-memory functions for experiments or your own coarsener:
`coarsen_points`, `coarsen_points_store` (`strategies.points`),
`coarsen_graph` (`strategies.graphs`), `simplify_skeleton`, `decimate_skeleton`
(`strategies.skeletons`), `coarsen_mesh_cluster`, `coarsen_mesh_quadric`
(`strategies.meshes`, `[mesh]` extra), `decimate_mesh`
(`strategies.mesh_decimate`) and the selection functions in
`multiresolution.object_selection`.
