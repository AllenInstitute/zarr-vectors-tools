# Refreshing a pyramid

Editing level 0 leaves every coarser level stale.
`rebuild_pyramid_from_level` re-coarsens every level *above* a given one,
reusing each level's own stored settings so the result is byte-for-byte
equivalent to a from-scratch `build_pyramid`.

```python
from zarr_vectors.core.store import open_store
from zarr_vectors_tools.multiresolution.refresh import rebuild_pyramid_from_level

root = open_store("tracts.zv", mode="r+")

# Re-coarsen levels 1..N from the (just-edited) level 0.  Returns one
# summary dict per rebuilt level, in ascending level order.
summaries = rebuild_pyramid_from_level(root, source_level=0)
```

Levels at or below `source_level` are untouched. A `source_level` that
is not present in the store raises `EditError`, and a store with no
levels above it returns `[]` without doing anything.

## What is reused

Nothing is inferred from arguments — the plan comes off disk.

```python
# For each target level, BEFORE deleting it, refresh snapshots:
#
#   coarsen_factor    <- bin_ratio[0], or compute_bin_ratio(base_bin, bin_shape)
#   sparsity_factor   <- 1.0 / object_sparsity
#   chunk_scale_factor<- round(chunk_shape / root chunk_shape) per axis
#   parent_level      <- parent_level, or level - 1
#
# then removes the level and re-runs coarsen_level with exactly those.
```

The snapshot has to happen before the delete, because
`remove_resolution_level` wipes the group's attrs along with its arrays.
On a transactional backend each delete is committed before the
re-coarsen, since `coarsen_level` re-opens the store from its URL and
would otherwise read the pre-delete state.

:::{warning}
`coarsen_level` re-opens the store from a path or URL and spawns its own
backend session, so it does **not** see uncommitted writes. Inside an
`EditSession`, pending edits must already be committed or the refresh
will faithfully coarsen the *pre-edit* geometry. `rebuild_pyramid_from_level`
issues a pre-refresh commit itself when `session_for(root)` is not
`None`, but only for writes made through that same `root` handle.
:::

## This is not wired into `EditSession`

The rich re-coarsening is deliberately **not** injected into core.
`zarr_vectors_tools.__init__` registers the coarsening and selection
*strategies* into `zarr_vectors.multiresolution.registry`, but it does
not register a refresher.

```python
from zarr_vectors.ops.edit import EditSession

root = open_store("tracts.zv", mode="r+")

# refresh_pyramid=True calls zarr_vectors.ops.refresh.rebuild_pyramid_from_level
# — CORE's same-named basic refresher, NOT the tools one imported above.
with EditSession(root, refresh_pyramid=True) as session:
    ...

# For the rich re-coarsening: let the session record what went stale,
# flush (which commits), then drive the refresh yourself.
with EditSession(root, refresh_pyramid=False) as session:
    ...
    report = session.flush()
    report.dirty_pyramid_levels   # every level above the lowest one touched

rebuild_pyramid_from_level(root, source_level=0)
```

The two functions share a name and differ in package, which is the whole
trap: `refresh_pyramid=True` on a skeleton or streamline store rebuilds
its coarser levels with core's `per_object` binning pyramid —
geometrically valid, but not the topology-preserving skeleton decimation
or the chunk-local polyline coarsening the store was built with.
`refresh_pyramid=False` is the honest setting: it leaves the coarse
levels stale and tells you so via `report.dirty_pyramid_levels`.

:::{note}
`rebuild_pyramid_from_level` calls `coarsen_level` with only the four
snapshotted parameters. It does not thread `sparsity_strategy`,
`sparsity_seed`, `compressor`, `executor`, or `cross_level_storage` — so
a refreshed level is re-coarsened with the defaults for those, and a
pyramid that needs cross-level links rebuilt wants a full `build_pyramid`
instead.
:::

## See also

- [Multiresolution index](index.md)
- [Building pyramids](building_pyramids.md) — the from-scratch path.
- [Strategies](strategies.md) — which coarsener a refresh will dispatch to.
- [Cross-level links](cross_level_links.md) — not rebuilt by a refresh.
