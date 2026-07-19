# Coarsening strategies

`coarsen_level` never takes a strategy argument. It reads the store's
root metadata, calls `select_coarsener_key` to get a dispatch key, and
looks that key up in the coarsener registry. The geometry decides.

```python
from zarr_vectors.core.store import open_store, read_root_metadata
from zarr_vectors_tools.multiresolution.coarsen import (
    get_coarsener,
    register_coarsener,
    select_coarsener_key,
)

root_meta = read_root_metadata(open_store("tracts.zv", mode="r"))
key = select_coarsener_key(root_meta)   # "skeleton" | "polyline" | "per_object"
fn = get_coarsener(key)                 # raises CoarseningError if unregistered

# Override or add a key without editing coarsen_level.  The callable takes
# (store_path, source_level, target_level) plus the uniform keyword set
# coarsen_level forwards, and returns a summary dict.
register_coarsener("polyline", my_polyline_coarsener)
```

The three built-ins are registered at the bottom of
`zarr_vectors_tools.multiresolution.coarsen`, so importing that module
(or `zarr_vectors_tools` itself) is what populates the registry.

## Dispatch

| Store condition | Coarsener | Notes |
| --- | --- | --- |
| `links_convention == "implicit_sequential_with_branches"` | `"skeleton"` | `coarsen_factor` is reinterpreted as a decimation **stride**; `chunk_scale_factor` defaults to **2**; `sparsity_strategy="random"` silently degrades to deterministic `"length"`; `coarsen_mode` is ignored |
| `links_convention == "implicit_sequential"` **and** `"streamline" in geometry_types` | `"polyline"` | `coarsen_mode` selects `"rdp"` (Douglas-Peucker) versus `"decimate"` (uniform stride, where `coarsen_factor` is the stride) |
| everything else | `"per_object"` | `coarsen_mode` and `executor` are accepted for signature parity but no-op |

:::{warning}
The skeleton row's three reinterpretations are silent. Passing
`coarsen_factor=8.0` to a skeleton store does not mean "8× bigger bins",
it means "keep every 8th non-anchor vertex", and asking for
`sparsity_strategy="random"` gets you `"length"` without a warning. Check
`select_coarsener_key` before you reason about what a factor will do.
:::

## Skeletons

`zarr_vectors_tools.multiresolution.strategies.skeletons` — the
*topology-preserving* downsample. Roots, leaves, and branch points (any
vertex with ≥ 2 children) always survive; only the degree-2 vertices
along smooth unbranched runs are thinned, so the tree stays recognisable
at every level.

```python
from zarr_vectors_tools.multiresolution.strategies.skeletons import (
    coarsen_skeleton_level,
    decimate_skeleton,
    simplify_skeleton,
)

# Pure function, RDP along each unbranched chain between two anchors.
out = simplify_skeleton(
    positions, edges,             # (N, D) and (M, 2) [child, parent] edges
    tolerance=500.0,              # RDP perpendicular distance, nm.  <= 0 = identity
    attributes={"radius": radius},
    attr_agg="max",               # "max" | "mean" | "min" | "first"
)
out["positions"]              # (K, D) survivors
out["edges"]                  # [child, parent] over the NEW indices
out["kept_source_indices"]    # (K,) new index -> original index

# Pure function, uniform stride.  Same return shape.  More predictable
# than RDP, which bottoms out once a skeleton is near-minimal.
decimate_skeleton(positions, edges, stride=8, forced_keep=boundary_vertices)

# The level coarsener, routed to from coarsen_level.  Uses decimate_skeleton.
coarsen_skeleton_level(
    "skel.zv", 0, 1,
    stride=8,                     # ~8x reduction per level
    chunk_scale_factor=2,         # 2x grid = 8x volume in 3D, cancelling the stride
    sparsity_strategy="length",   # drops the shortest skeletons first
    attr_agg="max",
    drop_interior_below=0,
    boundary_offset_nm=None,
    executor=None,
)
```

`attr_agg` aggregates each survivor over the run of source vertices that
collapsed onto it — `"max"` is the default because radius-like attributes
should not be diluted by the thin vertices between two thick ones.

`drop_interior_below` is a level-of-detail drop, not a decimation knob:
after a target chunk's pieces are written, any object whose total
surviving vertex count is at or below the threshold **and** which has no
vertex on the chunk's outer faces is discarded. The outer-face test is
what makes it safe — an interior-only object cannot extend into a
neighbouring chunk, so dropping it cannot orphan anything.

`boundary_offset_nm` is the phase of the chunk grid in position units.
The "is this vertex on a chunk face" test is `(coord - offset) % chunk_shape == 0`;
leave it at `None` (all zeros) unless the store's chunk grid origin is
offset from zero, in which case every boundary vertex would be
misclassified as interior.

:::{note}
Cross-chunk links are deliberately not reconstructed by the skeleton
coarsener. The ingest convention for this geometry is "ignore cross-chunk
edges if missing", and each simplified piece maps to exactly one fragment
in exactly one target chunk because the coarse grids are nested.
:::

## Polylines and streamlines

`zarr_vectors_tools.multiresolution.strategies.polylines` — Douglas-Peucker
simplification, chunk-local.

```python
from zarr_vectors_tools.multiresolution.strategies.polylines import (
    coarsen_polyline_level,
    decimate_polyline,
    simplify_polyline,
)

# Pure functions.  Both always preserve the first and last vertex.
simplify_polyline(vertices, epsilon=250.0)   # Douglas-Peucker
decimate_polyline(vertices, stride=8)        # every stride-th interior vertex

coarsen_polyline_level(
    "tracts.zv", 0, 1,
    coarsen_factor=8.0,     # the stride when coarsen_mode="decimate"
    sparsity_factor=2.0,
    coarsen_mode="rdp",     # "rdp" | "decimate"
    simplify_epsilon=None,  # explicit RDP epsilon; see below when None
    executor=None,          # (func, items, shared) -> list[result]
)
```

In `"rdp"` mode a `simplify_epsilon` of `None` is derived as
`min(source_chunk_shape) * 0.5 * coarsen_factor`, so the tolerance scales
with both the chunk grid and the requested factor. Pass it explicitly
when you want a fixed tolerance in position units instead.

The important property is memory, not geometry. Peak memory is
**O(one target chunk's source fan-in)**, independent of dataset size —
this is the fix for the OOM that whole-level coarsening hit at 5M
streamlines, where the old implementation loaded every fragment,
manifest, and per-object vertex sequence of the source level at once
(fine at 100K, fatal at 5M). Per-target-chunk work is dispatched through
`executor`, and the default serial executor produces output identical to
the parallel path by construction.

The coarsener also emits **directed** cross-chunk links, endpoint 0 being
the predecessor and endpoint 1 the successor. Walk order is data for a
streamline — a viewer computing tangents from an undirected edge list
would get sign flips at every chunk boundary — so `directed=True` is
stamped as a family-wide policy on `links/0` and cannot be flipped later.

## Points

`zarr_vectors_tools.multiresolution.strategies.points` — bin, then take
one representative per bin.

```python
from zarr_vectors_tools.multiresolution.strategies.points import (
    coarsen_points,
    coarsen_points_store,
)

out = coarsen_points(
    positions, bin_size=100.0,
    attributes={"intensity": intensity},
    object_ids=oids,          # each metanode inherits the MAJORITY object ID
    agg_mode="mean",
    use_medoid=False,         # True = medoid mode, see below
)
out["positions"]        # (M, D)
out["reduction_ratio"]  # N / M

# Read a level, coarsen, write a new level.
coarsen_points_store("pts.zv", target_level=1, bin_size=200.0, use_medoid=True)
```

`use_medoid=False` (the default) places each metanode at its bin's
**centroid** — a synthetic coordinate that was never in the input.
`use_medoid=True` instead picks the input vertex closest to that
centroid, which **preserves the original coordinate precision**: useful
when downstream code round-trips positions back to source IDs, or when
the coordinates are quantised and an averaged value would fall off the
lattice.

Object IDs are inherited by majority vote over each bin's members, so a
metanode straddling two objects is attributed to whichever contributed
more points.

## Meshes

`zarr_vectors_tools.multiresolution.strategies.meshes` — two approaches,
with a fallback between them.

```python
from zarr_vectors_tools.multiresolution.strategies.meshes import (
    coarsen_mesh_cluster,
    coarsen_mesh_quadric,
)

# 1. Vertex clustering: merge vertices per spatial bin, remap face
#    indices, drop faces that collapsed to an edge or point, dedupe.
out = coarsen_mesh_cluster(vertices, faces, bin_size=50.0, agg_mode="mean")
out["degenerate_faces_removed"]   # collapsed + duplicate faces
out["face_reduction"]             # F / E

# 2. Quadric error edge collapse, via pyfqmr.  Give ONE of the targets.
out = coarsen_mesh_quadric(vertices, faces, target_ratio=0.25)
out["method"]   # "quadric_pyfqmr" or "vertex_clustering_fallback"
```

`coarsen_mesh_quadric` needs the optional `pyfqmr` dependency — install
the `mesh` extra. Without it the `ImportError` is caught and the call
falls back to `coarsen_mesh_cluster` with a bin size estimated from the
requested reduction, reporting `method="vertex_clustering_fallback"`.
Check that key rather than assuming you got quadric decimation.

## Graphs

`zarr_vectors_tools.multiresolution.strategies.graphs` — grid contraction
and terminal-branch pruning.

```python
from zarr_vectors_tools.multiresolution.strategies.graphs import (
    coarsen_graph,
    prune_skeleton,
)

# Grid contraction: nodes in one bin become one metanode.  Intra-bin
# edges (self-loops) vanish; parallel inter-bin edges merge by SUMMING
# their weights, so a contracted edge records the total traffic.
out = coarsen_graph(positions, edges, bin_size=100.0, edge_weights=w)

# Terminal-branch pruning for trees: iteratively remove leaf-to-branch-point
# paths below a threshold.  Branch points themselves are never pruned.
out = prune_skeleton(
    positions, edges,
    min_branch_length=1000.0,   # Euclidean path length
    min_branch_vertices=3,      # or a node count; either triggers a prune
)
out["branches_removed"]
out["kept_indices"]   # original indices of the survivors
```

Pruning iterates to a fixed point, so removing one terminal branch can
expose a new leaf that is itself short enough to go.

## Metanodes

Every binning strategy above routes through the same primitive.

```python
from zarr_vectors_tools.multiresolution.metanodes import generate_metanodes

out = generate_metanodes(
    positions, bin_size=100.0,
    attributes={"radius": radius},
    agg_mode="mean",   # "mean" | "sum" | "first" | "count"
)
out["metanode_positions"]    # (M, D) bin centroids
out["metanode_counts"]       # (M,) vertices per bin
out["children"]              # list of M arrays of global source indices
out["metanode_attributes"]   # {name: (M, ...) aggregated}
out["bin_coords"]            # (M, D) int grid coordinates
```

`agg_mode="count"` ignores the input values entirely and returns the
per-bin member count, which is the density channel a viewer needs when a
coarse level has thrown away the thing it would otherwise have counted.

## See also

- [Multiresolution index](index.md)
- [Concepts](concepts.md) — coarsening versus sparsity.
- [Building pyramids](building_pyramids.md) — the factors that feed these coarseners.
- [Object selection](object_selection.md) — the `sparsity_strategy` half.
- [Parallelism](../how_to/parallelism.md) — `executor=` for the chunk-local coarseners.
