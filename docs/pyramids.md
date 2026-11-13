# Pyramids

A pyramid adds coarser copies of level 0, so a viewer can draw a whole dataset
from a small level and load detail only where you zoom in. This page shows how
to choose pyramid values; the [Pyramid reference](pyramid_reference.md) lists
the coarseners and the Python-only builders and options.

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
| `--chunk-scale` | multiply the chunk edge by this (whole numbers) | chunks 2×, 4×, 8× wider |

Give `--coarsen`, `--sparsity` or both. When you give only one, the other is
`1` at every level; when you give both, they must have the same number of
entries. Every value must be at least 1, and `1` means "no change at this
level". Prefer `2,2,2` over `2,4,8`, which compounds to 1/64 of the objects and
64× wider chunks at the last level.

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
chunks. Levels with the same objects and the same chunk size look alike to the
viewer, which then draws the coarsest level at every zoom.

For the viewer, keep adding levels until the coarsest one fits comfortably in
its GPU budget. See [Visualise](visualise.md#sizing-the-pyramid-for-the-gpu-budget).

## What `--coarsen` does for each geometry

**Points, graphs and lines** snap each object's vertices to a grid of bins and
merge the vertices that share a bin into one at their mean position; each
object keeps its own vertex in a bin it shares with others. The bin edge at
level *n* is the level-0 bin shape multiplied by the coarsen factors up to *n*,
so set `--bin-shape` as [Store layout](store_layout.md#bin-shape) describes.
Per-vertex attributes come along: a float column takes the bin's mean, any
other column (codes, labels) the value of one of the merged vertices. Each
merged vertex is stored in the chunk that contains it, even where a bin does
not divide the chunk.

**Lines** keep each segment's direction. A segment whose two ends fall in the
same bin is dropped from that level, and so from every coarser one, rather
than kept as a single vertex. Each level's coarsening record counts the
segments it dropped:
`read_coarsening_record(root, level)["collapsed_objects"]` (from
`zarr_vectors_tools.multiresolution.coarsen`).

**Graphs** merge vertices only within one connected component, so no level
joins two components that the level below keeps apart. A coarse level's edges
are the edges of the level below, carried onto the merged vertices: an edge
inside one merged vertex disappears, and parallel edges become one. Edges that
cross a chunk face stay linked across it. Edge attributes come along like
vertex attributes. A graph made of many small components cannot shrink below
one vertex per component. A graph converted from an edge list is one object,
so `--sparsity` has nothing to drop.

**Streamlines** are simplified with Douglas–Peucker, one chunk at a time. Set
the tolerance per level with `--rdp-tolerance`, in store units (mm for
tractography): the furthest a simplified streamline may stray from the level
below. Without it, the tolerance is half the smallest edge of the level's bin
(the level below's bin times the coarsen factor); a TRK store has no
`--bin-shape`, so its bin is the whole chunk. A run of a streamline inside one
chunk never simplifies below its two end points, so growing the chunks with
`--chunk-scale` is what lets each level keep simplifying. With
`--rdp-tolerance` given, `--coarsen` no longer decides how much is simplified,
so `1`s are fine; each level's scale in the store's OME `multiscales` metadata
then grows by `--coarsen` or `--chunk-scale`, whichever is larger (the
streamline recipe above gives scales 1, 2, 4 and 8). `--coarsen-mode decimate`
keeps every *n*-th vertex instead, and takes no `--rdp-tolerance`.

**Skeletons** keep every *n*-th vertex along each branch (n = the coarsen
value, rounded), and always keep roots, branch points, end points and the
vertices where a branch crosses a chunk boundary, so every tree stays one tree
at every level. A
[precomputed ingest](convert/skeletons.md#neuroglancer-precomputed-skeleton-layers)
builds its pyramid inside the ingest; pass `--chunk-scale` explicitly there, as
it defaults to 2 (elsewhere, 1).

**Meshes** are coarsened best by edge collapse, `--method mesh_decimate`: each
`--coarsen` value is the on-disk size reduction for that level (more than 1),
and the surface stays a closed manifold. It keeps every object, so
`--sparsity` must be 1. Per-vertex attributes are carried when scipy (in the
`[mesh]` extra) is installed; without it the coarse levels have none, with a
warning.

```bash
zvtools convert brain.obj brain.zv --chunk-shape 100,100,100 --split-objects \
    --coarsen 4,4 --sparsity 1,1 --chunk-scale 2,2 --method mesh_decimate
```

Without `--method`, meshes use vertex clustering (vertices closer than the
median edge length times the factor merge), which can leave holes and
non-manifold edges. In Python, pass `method="mesh_decimate"` to
`build_pyramid`. Thin structures such as neurites collapse into long sliver
triangles when a level is cut too hard; prefer `--coarsen 2` steps for them.

`--method` picks the coarsener for any store (`auto`, the default, chooses by
geometry); the methods are listed in the [Pyramid reference](pyramid_reference.md).

### Face groups for per-object reads

Each mesh level is written with one face group per object in every chunk, so
a viewer drawing one object can read just its faces. zarr-vectors-py checks
the grouping and stamps the level `fragment_link_groups`. To group and stamp a
store written some other way, run:

```bash
zvtools index-faces brain.zv            # all levels; --levels 0 for one
```

Usually only the small `link_fragments` index is rewritten; the faces are
reordered only where a chunk did not keep each object's faces together, and
the face data is unchanged. A level is left unstamped where a face joins two
objects, where its objects' vertex fragments overlap, or where its face groups
do not hold every face of a chunk exactly once. `--dry-run` reports what would
change without writing.

Any later write to a level's faces or fragments (an edit, a `zvtools merge`,
a rechunk) withdraws the stamp; run `zvtools index-faces` again afterwards.
`--verify` re-checks levels that are already stamped, repairs them where it
can, and withdraws the stamp where it cannot; run it after `zvtools shard`.
Stamping needs a zarr-vectors-py that provides `fragment_link_groups`. With one
that does not, pyramid levels are grouped but not stamped, and
`zvtools index-faces` stops with an error.

## Choosing which objects survive

`--sparsity-strategy` decides which objects a level keeps:

| Strategy | Keeps |
|---|---|
| `random` (default) | a uniform random subset |
| `length` | the longest objects (vertex count for points) |
| `spatial_coverage` | at least one object per occupied bin, the rest in proportion to density |
| `point_thinning` | at most one object per bin (the bin size sets the count, not `--sparsity`) |
| `group` | the same fraction of every named group, never fewer than one per group; needs groups on the level |
| `attribute` | the highest values of the object attribute `--sparsity-attribute NAME` names (read from level 0) |

Which strategies each geometry supports is in the
[Pyramid reference](pyramid_reference.md#coarseners). Skeletons have no random
selection: `random` is replaced by `length`. An unsupported strategy stops the
build with an error; with `--replace`, the old coarse levels are already
removed by then.

`length` on streamlines ranks by a stored per-streamline length. `zvtools convert`
computes it when you choose `length`. If you will run
`zvtools pyramid --sparsity-strategy length` later, convert with
`--compute-length`.

```bash
# Keep the streamlines with the highest per-streamline weight (a TRK property,
# such as SIFT2 weights) at each level.
zvtools pyramid tracts.zv --coarsen 1,1 --sparsity 2,2 --rdp-tolerance 0.5,1 \
    --sparsity-strategy attribute --sparsity-attribute weight
```

The CLI has no seed flag for random selection, so two runs keep different
objects; in Python, pass `sparsity_seed` to `build_pyramid`.

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

## Storing each object once (additive pyramids)

A sparsity pyramid stores the objects a coarse level keeps again at every
finer level. `--refinement add` stores each object (or point) only at the
coarsest level that keeps it: every level but the coarsest then holds only
what the next coarser level lacks, and a level's complete content is the union
of it and every coarser level. The pyramid then costs about what level 0 alone
did, and a viewer zooming in fetches only what it has not drawn yet.

```bash
zvtools pyramid tracts.zv --sparsity 2,2,2 --refinement add
```

This needs a zarr-vectors-py that writes additive levels
(`zarr_vectors_tools.multiresolution.additive.supports_additive()` returns
`True`). With one that does not:

| Command | Result |
|---|---|
| `zvtools pyramid --refinement add` | stops with `error: refinement='add' needs a zarr-vectors-py that writes additive levels; the installed one does not`, before writing anything |
| `zvtools convert --refinement add` | writes level 0, then stops with the same error, leaving a store with no pyramid |
| `zvtools convert --refinement add` for TRK and precomputed skeleton input | writes level 0 and an ordinary pyramid, then stops with `RuntimeError: the installed zarr-vectors-py cannot write additive levels` |
| `build_pyramid(..., refinement="add")` | raises `ValueError` before writing anything |
| `build_point_subset_pyramid(..., refinement="add")` | raises `CoarseningError` before writing anything |

The pyramid is built as usual and then converted, finest level first.
**Level 0 is rewritten in place**: afterwards it holds only the objects no
coarser level kept, so level 0 alone is no longer the whole dataset. The
command says so on stderr. The store lists `additive_levels` in its root's
`required_capabilities`; a zarr-vectors-py reader that implements it returns
each level's complete content (`own_level_only=True` reads one level's stored
data), and a reader that checks the list but lacks the capability refuses the
store. A reader that does not check the list shows level 0 incomplete.

It is exact when the levels differ only in which objects they keep:
streamline and polyline sparsity with `--coarsen 1` (each kept streamline is
byte-identical at every level, also with `--chunk-scale`), and
`build_point_subset_pyramid(..., refinement="add")`. It warns, with a banner
on stderr, and builds anyway when:

- the store is a mesh: every mesh coarsener re-tessellates, so coarse and
  fine surfaces mix in the finest view;
- a level coarsens geometry (`--coarsen` above 1, an explicit
  `--rdp-tolerance`, or a coarsener that bins vertices even at factor 1:
  points, graphs, lines, skeletons): the objects kept there exist only in
  coarse form, so the finest view shows them coarse;
- a level keeps every object of the level below (`--sparsity 1`), which
  leaves the additive level empty;
- a point subset pyramid does not carry every attribute: a level's complete
  content keeps only the attributes all its levels have.

Cross-level links are removed: they join one object at two levels, and an
object of an additive pyramid is at one. An additive store cannot be
rebuilt, refreshed or have levels removed: `zvtools pyramid --replace`,
`rebuild_pyramid_from_level` and merges with `--pyramid rebuild` or `drop`
refuse it (a merge with `--pyramid keep` adds the new objects to level 0,
which is consistent). Merging *from* an additive store reads all of it.
Sharding (`zvtools shard`, `--shard`) keeps it as it is.

## Links between levels

`--cross-level-storage` links every vertex of a surviving object to the coarse
vertex of the same object that it was merged into. `implicit` writes these
links at the finer level (`links/+1`); `explicit`, the default, also writes the
reverse at the coarser level (`links/-1`). `--cross-level-depth N` links levels
up to N apart (default 1; `-1` links every pair of levels).

Only point, graph and line pyramids write them. The store's root attributes
record what was written, so a streamline, skeleton or mesh pyramid reads
`cross_level_storage: none` whatever you passed.

The viewer does not read them, and they add to the time to build a pyramid.
Pass `--cross-level-storage none` unless your own code follows those links.

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

See the [Pyramid reference](pyramid_reference.md) for every builder and option.
