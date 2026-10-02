# Pyramids

A pyramid adds coarser copies of level 0, so a viewer can draw a whole dataset
from a small level and load detail only where you zoom in. This page is the one
place that recommends pyramid values.

Build the pyramid while converting, or later on an existing store:

```bash
zvtools convert cells.csv cells.zv --chunk-shape 100,100,100 --bin-shape 10,10,10 \
    --coarsen 2,2,2 --sparsity 2,2,2 --cross-level-storage none

zvtools pyramid cells.zv --coarsen 2,2,2 --sparsity 2,2,2 --cross-level-storage none
```

`zvtools pyramid` refuses a store that already has coarse levels unless you pass
`--replace`, which removes them first.

## How the flags combine

Each flag takes one value per coarser level, and every value is relative to the
level below, so the values compound:

| Flag | Each value means | `2,2,2` gives, relative to level 0 |
|---|---|---|
| `--coarsen` | simplify each object's geometry by this factor (meaning varies by geometry, below) | 2×, 4×, 8× coarser |
| `--sparsity` | keep 1/value of the objects kept at the level below | 1/2, 1/4, 1/8 of the objects |
| `--chunk-scale` | multiply the chunk edge by this | chunks 2×, 4×, 8× wider |

`--coarsen` and `--sparsity` must have the same number of entries, and every
value must be at least 1. Use `1` for "no change at this level". Prefer `2,2,2`
over `2,4,8`, which compounds to 1/64 of the objects and 64× wider chunks at
the last level.

## Recommended recipes

Measured on synthetic data with the command shown; the counts are each level's
`vertex_count`, as `zvtools info` and the viewer see it. Your counts will
differ, but the shape of each pyramid should not.

| Geometry | Command options | `vertex_count` per level (measured) |
|---|---|---|
| Points | `--bin-shape B,B,B --coarsen 2,2,2 --sparsity 2,2,2 --cross-level-storage none`, with B near the point spacing | 50,000 → 18,912 → 4,365 → 751 |
| Points without object ids | Python `build_point_subset_pyramid` (below) | 50,000 → 6,250 → 781 → 97 |
| Lines | `--bin-shape B,B,B --coarsen 2,2,2 --sparsity 2,2,2 --cross-level-storage none` | 10,000 → 4,304 → 1,592 → 424 (5,000 segments, median 23 units long, B = 10) |
| Graphs | `--bin-shape B,B,B --coarsen 2,2,2 --sparsity 1,1,1 --chunk-scale 2,2,2 --cross-level-storage none` | 4,000 → 3,138 → 1,000 → 143 (a 4,000-node graph, 3,974 of them in one component, B = 10) |
| Streamlines | `--coarsen 1,1,1 --sparsity 2,2,2 --rdp-tolerance 0.5,1,2 --chunk-scale 2,2,2` (tolerances in mm) | 240,000 → 24,602 → 7,638 → 2,268 |
| Skeletons | `--coarsen 4,4 --sparsity 2,2 --chunk-scale 2,2 --sparsity-strategy length` | 1,866,688 → 408,176 → 130,985 (one FlyWire chunk); 120,000 → 16,140 → 2,552 (40 SWC trees) |
| Meshes | `--coarsen 4,4 --sparsity 1,1 --chunk-scale 2,2 --method mesh_decimate` | 61,452 → 15,372 → 3,852 |

Graph and mesh pyramids usually keep every object, so give them growing
chunks. Levels with the same objects and the same chunk size look alike to the
viewer, which then draws the coarsest level at every zoom.

For the viewer, keep adding levels until the coarsest one fits comfortably in
its GPU budget. See [Visualise](visualise.md#sizing-the-pyramid-for-the-gpu-budget).

## What `--coarsen` does for each geometry

**Points, graphs and lines** snap each object's vertices to a grid of bins and
merge the vertices that share a bin into one at their mean position; each
object keeps its own vertex in a bin it shares with others. The bin edge at
level *n* is `--bin-shape` multiplied by the coarsen factors up to *n*. The base
bin defaults to the chunk shape, so without `--bin-shape` the first coarse level
already merges each chunk into a few vertices. On 50,000 points in 500 objects
and 100-unit chunks, `--coarsen 8,8` gave 50,000 → 909 → 615 vertices. With
`--bin-shape 10,10,10 --coarsen 2,2,2`, the same points gave
50,000 → 38,287 → 17,712 → 6,373. Per-vertex attributes come along: a float
column takes the bin's mean, any other column (codes, labels) the value of one
of the merged vertices. Each merged vertex is stored in the chunk that contains
it, even where a bin does not divide the chunk (an 80-unit bin in a 100-unit
chunk).

**Lines** keep each segment's direction. A segment whose two ends fall in the
same bin is dropped from that level, and so from every coarser one, rather than
kept as a single vertex. The level's coarsening record counts them:
`read_coarsening_record(root, level)["collapsed_objects"]` (from
`zarr_vectors_tools.multiresolution.coarsen`). On 5,000 segments of median length
23 units with `--bin-shape 10,10,10 --coarsen 2,2,2 --sparsity 1,1,1`, levels
1 to 3 dropped 652, 1,208 and 1,435 segments (bins of 20, 40 and 80 units).

**Graphs** merge vertices only within one connected component, so no level joins
two components that the level below keeps apart. A coarse level's edges are
exactly the edges of the level below, carried onto the merged vertices: an edge
inside one merged vertex disappears, and parallel edges become one. Edges that
cross a chunk face stay linked across it. Edge attributes come along like vertex
attributes: a float column takes the mean of the merged edges, any other column
the value of one of them. A graph made of many small components cannot shrink
below one vertex per component: a 4,000-node graph in 1,714 components gave
4,000 → 3,251 → 2,494 → 2,052 (still 1,714 components at level 3), against
4,000 → 3,138 → 1,000 → 143 for a 4,000-node graph in 17 components. A graph
converted from an edge list is one object, so `--sparsity` has nothing to drop.

**Streamlines** are simplified with Douglas–Peucker, one chunk at a time. Set the
tolerance per level with `--rdp-tolerance`, in store units (mm for
tractography): the furthest a simplified streamline may stray from the level
below. Without it, the tolerance is half the smallest bin edge times the coarsen
factor, which is usually far too large. `--coarsen 8,8` on 2,000 streamlines
gave 240,000 → 43,542 → 43,542 vertices: level 2 could not simplify further,
because every chunk's run was already down to its two end points. Growing the
chunks with `--chunk-scale 2,2,2` is what lets each level keep simplifying. With
`--rdp-tolerance` given, `--coarsen` no longer decides how much is simplified,
so `1`s are fine. Each level's scale in the store's OME `multiscales` metadata
then grows by `--coarsen` or `--chunk-scale`, whichever is larger: the
streamline recipe above gives scales 1, 2, 4 and 8. `--coarsen-mode decimate`
keeps every *n*-th vertex instead.

**Skeletons** keep every *n*-th vertex along each branch (n = the coarsen value,
rounded), and always keep roots, branch points, end points and the vertices
where a branch crosses a chunk boundary, so every tree stays one tree at every
level. On 40 SWC trees of 3,000 nodes in 200-unit chunks, `--coarsen 4,4
--sparsity 1,1 --chunk-scale 2,2` gave 120,000 → 32,010 → 9,525 vertices, still
40 trees. A [precomputed ingest](convert/skeletons.md#neuroglancer-precomputed-skeleton-layers)
builds its pyramid inside the ingest; pass `--chunk-scale` explicitly there, as
it defaults to 2 (elsewhere, 1).

**Meshes** are coarsened best by edge collapse, `--method mesh_decimate`: each
`--coarsen` value is the on-disk size reduction for that level (more than 1),
the surface stays a closed manifold, and per-vertex attributes are carried. It
keeps every object, so `--sparsity` must be 1, and needs scipy (in the `[mesh]`
extra):

```bash
zvtools convert brain.obj brain.zv --chunk-shape 100,100,100 --split-objects \
    --coarsen 4,4 --sparsity 1,1 --chunk-scale 2,2 --method mesh_decimate
```

On 61,452 vertices that gave 61,452 → 15,372 → 3,852 vertices in chunks of 100,
200 and 400 units. Without `--method`, meshes use vertex clustering (vertices
closer than the median edge length times the factor merge): 61,452 → 19,376 →
4,814 with `--coarsen 2,2`, but clustered levels have holes and non-manifold
edges. In Python, pass `method="mesh_decimate"` to `build_pyramid`.

`--method` picks the coarsener for any store (`auto`, the default, chooses by
geometry); the methods are listed in the [Pyramid reference](pyramid_reference.md).

## Choosing which objects survive

`--sparsity-strategy` decides which objects a level keeps:

| Strategy | Keeps | Works for |
|---|---|---|
| `random` (default) | a uniform random subset | points, graphs, lines, streamlines, meshes |
| `length` | the longest objects (vertex count for points) | points, graphs, lines, streamlines, skeletons |
| `spatial_coverage` | at least one object per occupied bin, the rest in proportion to density | points, graphs, lines |
| `point_thinning` | at most one object per bin (the bin size sets the count, not `--sparsity`) | points, graphs, lines |
| `group` | the same fraction of every named group, never fewer than one per group | points, graphs, lines, streamlines |
| `attribute` | the highest values of the object attribute `--sparsity-attribute NAME` names (read from level 0) | points, graphs, lines, streamlines, skeletons, meshes |

Skeletons have no random selection: `random` is replaced by `length`.
Edge-collapse mesh decimation (`mesh_decimate`) keeps every object.

`length` on streamlines ranks by a stored per-streamline length. `zvtools convert`
computes it automatically when you choose `length`. If you will run
`zvtools pyramid --sparsity-strategy length` later, convert with
`--compute-length`.

```bash
# Keep the streamlines with the highest per-streamline weight (a TRK property,
# such as SIFT2 weights) at each level.
zvtools pyramid tracts.zv --coarsen 1,1 --sparsity 2,2 --rdp-tolerance 0.5,1 \
    --sparsity-strategy attribute --sparsity-attribute weight
```

Random selection has no seed flag, so two runs keep different objects.

## Point clouds without object ids

A point cloud converted without object ids is one object, so `--sparsity` has
nothing to drop. Either give the points object ids (`--format table --object-id-column NAME`, see
[Single-cell data](convert/single_cell.md)), or build a subset pyramid, which keeps
a random 1/`divisor` of the points per level (exact points, not bin means)
and carries every per-vertex attribute:

```python
from zarr_vectors_tools.multiresolution.strategies.points import build_point_subset_pyramid

build_point_subset_pyramid("cells.zv", levels=3, divisor=8.0, seed=0)
```

## Links between levels

`--cross-level-storage` links every vertex of a surviving object to the coarse
vertex of the same object that it was merged into. `implicit` writes these
links at the finer level (`links/+1`); `explicit`, the default, also writes the
reverse at the coarser level (`links/-1`), and the two are exact mirrors.
`--cross-level-depth N` links levels up to N apart (default 1).

Only point, graph and line pyramids write them. The store's root metadata
records what was written, so a streamline, skeleton or mesh pyramid reads
`cross_level_storage: none` whatever you passed.

The viewer does not read them, and they nearly double the time to build a
pyramid: `zvtools pyramid --coarsen 2,2,2 --sparsity 2,2,2` on 50,000 points
took 2.3 s with them and 1.3 s without. Pass `--cross-level-storage none`
unless your own code follows those links.

## Building in Python

`zvtools pyramid` and `convert` call
`zarr_vectors_tools.multiresolution.coarsen.build_pyramid`. Its `factors`
argument is the CLI's `--coarsen` and `--sparsity` zipped into pairs:

```python
from zarr_vectors_tools.multiresolution.coarsen import build_pyramid

build_pyramid(
    "tracts.zv",
    factors=[(1, 2), (1, 2), (1, 2)],      # (coarsen, sparsity) per level
    chunk_scale_factors=[2, 2, 2],
    rdp_tolerances=[0.5, 1.0, 2.0],
)
```

See [Pyramid reference](pyramid_reference.md) for every builder and option.
