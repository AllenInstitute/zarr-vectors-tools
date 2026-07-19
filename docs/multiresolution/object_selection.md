# Object selection

Sparsity drops whole objects. `zarr_vectors_tools.multiresolution.object_selection`
holds the five selectors that decide *which* objects survive, and
`apply_sparsity`, the wrapper the coarseners actually call. Every
selector returns `kept_indices` — a sorted `int64` array of indices into
the original object list.

```python
from zarr_vectors_tools.multiresolution.object_selection import (
    apply_sparsity,
    compute_polyline_lengths,
    compute_representative_points,
    select_by_attribute,
    select_by_length,
    select_by_spatial_coverage,
    select_point_thinning,
    select_random,
)

lengths = compute_polyline_lengths(polylines)          # (N,) float64 path lengths
midpoints = compute_representative_points(polylines)   # (N, D) per-object midpoints

select_by_length(lengths, target_count=1000)
select_by_attribute(fa_values, target_count=1000, mode="max")   # or mode="min"
select_by_spatial_coverage(midpoints, bin_shape=(500.0,) * 3, target_count=1000)
select_point_thinning(midpoints, bin_shape=(500.0,) * 3, seed=0)
select_random(n_objects=50_000, target_count=1000, seed=0)
```

## Choosing a selector

| Selector | `strategy` name | Keeps | Use when |
| --- | --- | --- | --- |
| `select_random` | `"random"` | A uniform random subset | You want an unbiased thumbnail of the population; the default |
| `select_by_length` | `"length"` | The longest objects | Streamlines or skeletons, where short objects are noise and long ones carry the structure |
| `select_by_attribute` | `"attribute"` | Highest (or lowest) scoring | You have a per-object score — FA, volume, importance — worth ranking on |
| `select_by_spatial_coverage` | `"spatial_coverage"` | A density-proportional spread, ≥ 1 per occupied bin | Empty regions in the coarse view would be misread as "no data" |
| `select_point_thinning` | `"point_thinning"` | ≤ 1 per occupied bin | You want *uniform density*, and the resulting count is whatever the bin layout gives |

`select_by_spatial_coverage` and `select_point_thinning` both bin, but
they differ in what is fixed. Coverage distributes a **fixed
`target_count`** budget across bins proportional to each bin's
occupancy — one per bin first, then the remainder weighted — so you get
exactly the count you asked for. Thinning **enforces a maximum of one
survivor per bin** and the count falls out of the geometry (≤ the number
of occupied bins); `sparsity` is ignored entirely for that strategy.
Thinning is deterministic without a `seed` (lowest-index candidate wins
each bin) and randomised within each bin with one.

## `apply_sparsity`

```python
from zarr_vectors_tools.multiresolution.object_selection import apply_sparsity

kept = apply_sparsity(
    n_objects,
    sparsity=0.5,           # FRACTION KEPT, in (0, 1] — not the CLI's divisor
    strategy="length",
    seed=0,
    lengths=lengths,        # required by "length"
    # attribute_values=..., attribute_mode="max",   # required by "attribute"
    # representative_points=..., bin_shape=...,     # required by the two spatial ones
    alive_mask=alive,       # see below
    relative_to="alive",    # see below
)
```

`sparsity >= 1.0` short-circuits to "keep everything". A missing
strategy-specific array raises `ValueError` rather than silently falling
back, and an unknown `strategy` raises with the list of valid names.

:::{note}
`apply_sparsity` takes the fraction *kept*; `coarsen_level` and the CLI
take a *divisor*. `coarsen_level` converts with
`keep_frac = 1.0 / sparsity_factor` before calling here, so
`--sparsity 4` arrives as `sparsity=0.25`.
:::

### `alive_mask`

A `(n_objects,)` boolean, `True` for object indices that still have data
at the source level. Object IDs are preserved across pyramid levels, so
an object dropped at level 1 still occupies its OID slot at level 2 — as
an empty manifest.

```python
import numpy as np

# What coarsen_level builds before every sparsity call:
alive_mask = np.array(
    [len(src_manifests[oid]) > 0 for oid in range(n_src_objects)],
    dtype=bool,
)
```

Without it, every index is a candidate — including the empty ones. For
`"random"` that is a real bug, not a stylistic point: `select_random`
draws uniformly from the full range, so once earlier levels have emptied
most of the OID space, most of the budget is spent re-"keeping" objects
that have no vertices, and the level comes out far sparser than
requested. Nothing errors.

:::{warning}
The ranking strategies only dodge this **incidentally**. A dead object's
length or attribute value is degenerate (zero, or whatever the padded
attribute row holds), so it never wins a top-N rank — but that is a
property of the values, not a guarantee of the API, and it does not hold
for every factor sequence or attribute. Pass `alive_mask` regardless of
strategy.
:::

### `relative_to`

What `sparsity` is a fraction *of*.

| Value | Target count | Behaviour across levels |
| --- | --- | --- |
| `"original"` (default) | `round(n_objects * sparsity)` | Absolute and saturating. A constant `sparsity` reselects the same absolute count every level, and once the alive pool has shrunk to that count, keeps everything |
| `"alive"` | `round(len(candidates) * sparsity)` | Compounding. Each level keeps a fixed fraction of the *previous* survivors |

```python
# 1000 objects, sparsity=0.5 applied three times.
# relative_to="original": 500, 500, 500  (saturates — level 3 is level 2)
# relative_to="alive":    500, 250, 125  (what a pyramid wants)
```

**Pyramids want `"alive"`**, and `coarsen_level` passes it explicitly.
The default is `"original"` for backward compatibility with direct
callers that mean an absolute-of-original target. The result is clamped
to `len(candidates)` and floored at 1 either way, so you never get an
empty level from rounding.

## Helpers

```python
# Euclidean path length per polyline; a polyline of < 2 vertices is 0.0.
lengths = compute_polyline_lengths(polylines)

# The midpoint VERTEX of each polyline — poly[len(poly) // 2], an actual
# input coordinate, not an interpolated or arc-length midpoint.
points = compute_representative_points(polylines)
```

`compute_representative_points` is what feeds `representative_points` for
the two spatial strategies. Because it indexes rather than interpolates,
the returned point is always a real vertex of the object — which is what
makes the bin assignment meaningful for a polyline whose ends sit in
different chunks.

## See also

- [Multiresolution index](index.md)
- [Concepts](concepts.md) — why sparsity is not coarsening.
- [Building pyramids](building_pyramids.md) — `--sparsity` and `--sparsity-strategy`.
- [Strategies](strategies.md) — the coarsening half, including the skeleton coarsener's silent `random` → `length` degradation.
