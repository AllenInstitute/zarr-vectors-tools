# Pyramids

A pyramid adds coarser copies of level 0, so a viewer can draw a whole dataset
from a small level and load detail only where you zoom in. This page shows how
to build one; the [Pyramid reference](pyramid_reference.md) lists every
coarsener and option.

Build the pyramid while converting, or later on an existing store:

```bash
zvtools convert cells.csv cells.zv --chunk-shape 100,100,100 --bin-shape 10,10,10 \
    --coarsen 2,2,2 --sparsity 2,2,2 --cross-level-storage none

zvtools pyramid cells.zv --coarsen 2,2,2 --sparsity 2,2,2 --cross-level-storage none
```

`zvtools pyramid` refuses a store that already has coarse levels unless you pass
`--replace`, which removes them first.

## How the flags combine

Each flag takes one value per coarser level. Each value is relative to the
level below, so the values compound:

| Flag | Each value means | `2,2,2` gives, relative to level 0 |
|---|---|---|
| `--coarsen` | simplify each object's geometry by this factor | 2×, 4×, 8× coarser |
| `--sparsity` | keep 1/value of the objects of the level below | 1/2, 1/4, 1/8 of the objects |
| `--chunk-scale` | multiply the chunk edge by this whole number | chunks 2×, 4×, 8× wider |

Give `--coarsen`, `--sparsity` or both; a missing one is `1` at every level.
Both must have the same number of entries. Every value is at least 1, and `1`
means no change. Prefer `2,2,2` to `2,4,8`, which compounds to 1/64 of the
objects at the last level.

## Recommended recipes

| Geometry | Command options |
|---|---|
| Points | `--bin-shape B,B,B --coarsen 2,2,2 --sparsity 2,2,2 --cross-level-storage none`, with B near the point spacing |
| Points without object ids | Python `build_point_subset_pyramid` ([below](#point-clouds-without-object-ids)) |
| Lines | `--bin-shape B,B,B --coarsen 2,2,2 --sparsity 2,2,2 --cross-level-storage none` |
| Graphs | `--bin-shape B,B,B --coarsen 2,2,2 --sparsity 1,1,1 --chunk-scale 2,2,2 --cross-level-storage none` |
| Streamlines | `--coarsen 1,1,1 --sparsity 2,2,2 --rdp-tolerance 0.5,1,2 --chunk-scale 2,2,2` (tolerances in mm) |
| Skeletons | `--coarsen 4,4 --sparsity 2,2 --chunk-scale 2,2 --sparsity-strategy length` |
| Meshes | `--coarsen 4,4 --sparsity 1,1 --chunk-scale 2,2 --method mesh_decimate` |

Graph and mesh pyramids usually keep every object, so give them growing
chunks. Otherwise the levels look alike to the viewer, and it draws the
coarsest one at every zoom. Add levels until the coarsest fits the viewer's
[GPU budget](visualise.md#sizing-the-pyramid-for-the-gpu-budget).

## What `--coarsen` does for each geometry

| Geometry | Coarsening |
|---|---|
| Points, graphs, lines | Vertices of one object that share a bin merge into one, at their mean position. The bin edge grows by the coarsen factor at each level, so set [`--bin-shape`](store_layout.md#bin-shape). Float attributes take the mean; other attributes take one merged vertex's value. |
| Lines | A segment whose two ends fall in one bin is dropped from that level and every coarser one. |
| Graphs | Vertices merge only within one connected component, so a graph cannot shrink below one vertex per component. Edges are carried onto the merged vertices. A graph converted from an edge list is one object, so `--sparsity` has nothing to drop. |
| Streamlines | Douglas–Peucker simplification, one chunk at a time. `--rdp-tolerance` sets the tolerance per level, in store units. Without it, the tolerance is half the smallest bin edge. A run inside one chunk keeps at least its two end points, so use `--chunk-scale` to keep simplifying. With `--rdp-tolerance`, `--coarsen` can be `1`. |
| Skeletons | Keep every *n*-th vertex along each branch, plus roots, branch points, end points and chunk crossings, so each tree stays one tree. A [precomputed ingest](convert/skeletons.md#neuroglancer-precomputed-skeleton-layers) defaults `--chunk-scale` to 2; elsewhere it is 1. |
| Meshes | Use `--method mesh_decimate`: each value is the size reduction for that level (more than 1), and the surface stays closed. `--sparsity` must be 1. Per-vertex attributes need scipy (`[mesh]` extra). The default, vertex clustering, can leave holes. Use `--coarsen 2` steps for thin structures such as neurites. |

`--method` picks the coarsener for any store; the default, `auto`, chooses by
geometry. In Python, pass `method=` to `build_pyramid`.

### Face groups for per-object reads

Each mesh level stores each object's faces together in every chunk, so a viewer
can read one object's faces alone. With a zarr-vectors core that supports it,
the level is stamped `fragment_link_groups`; without that support the faces are
still grouped but not stamped, and `zvtools index-faces` refuses to run. To
group and stamp a store written some other way, run:

```bash
zvtools index-faces brain.zv            # all levels; --levels 0 for one
```

Any later write to a level's faces (an edit, a merge, a rechunk) removes the
stamp, so run it again afterwards. After `zvtools shard`, run it with
`--verify`. `--dry-run` reports what would change.

## Choosing which objects survive

`--sparsity-strategy` decides which objects a level keeps:

| Strategy | Keeps |
|---|---|
| `random` (default) | a uniform random subset |
| `length` | the longest objects |
| `spatial_coverage` | at least one object per occupied bin, the rest by density |
| `point_thinning` | at most one object per bin (the bin size sets the count, not `--sparsity`) |
| `group` | the same fraction of every named group, at least one per group |
| `attribute` | the highest values of the object attribute named by `--sparsity-attribute NAME` |

The [Pyramid reference](pyramid_reference.md#coarseners) lists the strategies
each geometry supports; any other is refused before anything is written, as is
`group` on a store without groups. For `length` on streamlines in a later
`zvtools pyramid`, convert with `--compute-length`. The CLI has no seed, so two
runs keep different objects; in Python, pass `sparsity_seed` to `build_pyramid`.

## Point clouds without object ids

A point cloud converted without object ids is one object, so `--sparsity` has
nothing to drop. Either give the points object ids
([Single-cell data](convert/single_cell.md)), or build a subset pyramid. It
keeps a random 1/`divisor` of the points per level, with every attribute:

```python
from zarr_vectors_tools.multiresolution.strategies.points import build_point_subset_pyramid

build_point_subset_pyramid("cells.zv", levels=3, divisor=8.0, seed=0)
```

## Storing each object once (additive pyramids)

`--refinement add` stores each object only at the coarsest level that keeps
it. A level's full content is then that level plus every coarser one. The
pyramid costs about as much as level 0 alone.

```bash
zvtools pyramid tracts.zv --sparsity 2,2,2 --refinement add
```

Additive pyramids need a zarr-vectors core that supports them;
`zarr_vectors_tools.multiresolution.additive.supports_additive()` reports
whether this one does. Without it, `--refinement add` is refused before
anything is written or removed.

:::{warning}
Level 0 is rewritten in place: afterwards it holds only the objects no coarser
level kept. A reader that does not understand additive levels sees level 0
incomplete.
:::

The result is exact only when levels differ only in which objects they keep:
streamline sparsity with `--coarsen 1`, and `build_point_subset_pyramid`.
Otherwise it warns and builds anyway. An additive store cannot be rebuilt:
`zvtools pyramid --replace`, `rebuild_pyramid_from_level` and merges with
`--pyramid rebuild` or `drop` refuse it.

## Links between levels

`--cross-level-storage` links each vertex to the coarse vertex it was merged
into. `implicit` writes the links at the finer level; `explicit`, the default,
also writes the reverse. `--cross-level-depth N` links levels up to N apart
(default 1). Only point, graph and line pyramids write links, and the viewer
does not read them. Pass `--cross-level-storage none` unless your own code
follows them.

## Building in Python

`zvtools pyramid` and `convert` call `build_pyramid`. Its `factors` argument is
`--coarsen` and `--sparsity` zipped into pairs:

```python
from zarr_vectors_tools.multiresolution.coarsen import build_pyramid

build_pyramid(
    "tracts.zv",
    factors=[(1, 2), (1, 2), (1, 2)],      # (coarsen, sparsity) per level
    chunk_scale_factors=[2, 2, 2],
    rdp_tolerances=[0.5, 1.0, 2.0],
)
```
