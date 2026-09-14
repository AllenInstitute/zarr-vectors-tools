# Refreshing a pyramid

Editing level 0 leaves every coarser level stale.
`rebuild_pyramid_from_level` re-coarsens every level *above* a given one,
reusing each level's own stored settings — including the strategy and, for
polylines, the reduction mode that produced it — so the result matches a
from-scratch `build_pyramid` rather than approximating it.

Where the store does not record a parameter, the refresh **refuses** rather
than guessing. Today that is the skeleton stride; pass it explicitly with
`coarsen_factors={level: stride}`.

```python
from zarr_vectors.building import open_store
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
#   chunk_scale_factor<- round(chunk_shape / parent chunk_shape) per axis
#   parent_level      <- parent_level, or level - 1
#   method            <- coarsening_method  (per_object, per_fragment, mesh,
#                        mesh_decimate, skeleton, polyline)
#   coarsen_mode      <- "decimate" when coarsening_method is
#                        "polyline_decimate", else "rdp"
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

## This is not wired into the parent package's editing path

The rich re-coarsening is deliberately **not** injected into the parent
package. `zarr_vectors_tools.__init__` registers the coarsening and
selection *strategies* into its registry, but it does not register a
refresher.

So editing with `refresh_pyramid=True` rebuilds the coarser levels with
the parent package's own basic `per_object` binning pyramid — not the
topology-preserving skeleton decimation or chunk-local polyline coarsening
the store was built with. The editing surface itself is documented at
{zvpy}`the data API <api/api.html>`.

```python
import zarr_vectors as zv
from zarr_vectors.building import open_store
from zarr_vectors_tools.multiresolution.refresh import rebuild_pyramid_from_level

ds = zv.open("tracts.zv")

# For the rich re-coarsening: let the edit record what went stale, flush
# (which commits), then drive the refresh yourself.
with ds.editing(refresh_pyramid=False) as edit:
    ...
    report = edit.flush()
    report.dirty_pyramid_levels   # every level above the lowest one touched

rebuild_pyramid_from_level(open_store("tracts.zv", mode="r+"), source_level=0)
```

`refresh_pyramid=False` is the honest setting: it leaves the coarse levels
stale and tells you so via `report.dirty_pyramid_levels`, which is what
this package's refresher is then pointed at.

:::{warning}
This package's `rebuild_pyramid_from_level` and the parent package's
internal refresher share a name. They are different functions in different
packages, and only the one imported from `zarr_vectors_tools` uses the
rich strategies.
:::

:::{note}
The store records *what* each level is, not *how* it was asked for. So
`sparsity_strategy`, `sparsity_seed`, `compressor` and `executor` are
parameters of `rebuild_pyramid_from_level` — pass the ones the pyramid was
built with, or the refresh uses the defaults (`random` selection, no
compression, serial):

```python
rebuild_pyramid_from_level(
    root, source_level=0,
    sparsity_strategy="length",   # what build_pyramid was given
    compressor="zstd",            # level codecs are fixed at creation
    coarsen_factors={1: 8},       # required for skeleton levels
)
```

Cross-level links are still not rebuilt; a pyramid that needs them wants a
full `build_pyramid`.
:::

## See also

- [Multiresolution index](index.md)
- [Building pyramids](building_pyramids.md) — the from-scratch path.
- [Strategies](strategies.md) — which coarsener a refresh will dispatch to.
- [Cross-level links](cross_level_links.md) — not rebuilt by a refresh.
