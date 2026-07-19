# Building pyramids

`build_pyramid` writes every coarser level in sequence;
`coarsen_level` writes exactly one, for callers who want manual control.
Both live in `zarr_vectors_tools.multiresolution.coarsen` and both pick
the coarsening strategy from the store's root metadata rather than from
an argument.

## The CLI form and the Python form

```bash
# Two coarser levels: 8x vertex reduction each, keeping 1/2 then 1/4
# of the objects.  --coarsen and --sparsity are comma lists, one entry
# per COARSER level (level 0 already exists).
zvtools pyramid tracts.zv --coarsen 8,8 --sparsity 2,4

# With the optional knobs:
zvtools pyramid tracts.zv \
    --coarsen 8,8 --sparsity 2,4 \
    --chunk-scale 2,2 \
    --sparsity-strategy length \
    --coarsen-mode rdp \
    --cross-level-storage explicit --cross-level-depth 1 \
    --compressor zstd \
    --workers 8
```

```python
from zarr_vectors_tools.multiresolution.coarsen import build_pyramid

summary = build_pyramid(
    "tracts.zv",
    # One (coarsen_factor, sparsity_factor) tuple per coarser level.
    # factors[i] turns level i into level i+1.  Either entry at 1.0
    # opts out of that axis for that level.
    factors=[(8.0, 2.0), (8.0, 4.0)],
    # Per-level multiplier on the source level's chunk_shape.  Aligned
    # with `factors` (same length) or None for all-ones.  Scalar =
    # uniform per axis; a tuple sets per-axis multipliers.
    chunk_scale_factors=[2, 2],
    sparsity_strategy="length",
    sparsity_seed=0,
    cross_level_storage="explicit",   # default
    cross_level_depth=1,              # default, from core's DEFAULT_CROSS_LEVEL_DEPTH
    coarsen_mode="rdp",               # polyline stores only
    compressor="zstd",
    executor=None,                    # serial; see below
)

summary["levels_created"]   # 2
summary["level_specs"]      # per-level summary dicts
```

The two forms are the same call. `zvtools pyramid` parses `--coarsen`
and `--sparsity` into two float lists and hands them to `build_factors`,
which zips them into `[(coarsen, sparsity), ...]`.

## Paired lists

`--coarsen` and `--sparsity` are **paired, equal-length** lists, one
entry per coarser level. `build_factors` in
`zarr_vectors_tools.cli._args` zips them; unequal lengths raise
`SystemExit` with the two counts, and passing neither means "no pyramid
requested" (`build_factors` returns `None`, and `zvtools pyramid` then
exits, since a pyramid is the whole point of that subcommand).

```python
from zarr_vectors_tools.cli._args import build_factors

build_factors([8.0, 8.0], [2.0, 4.0])   # [(8.0, 2.0), (8.0, 4.0)]
build_factors([8.0, 8.0], [2.0])        # SystemExit: 2 entries vs 1
build_factors(None, None)               # None — no pyramid requested
```

`sparsity` is a **divisor**, not a fraction kept: `1` keeps every object,
`2` keeps half, `8` keeps an eighth. Internally `coarsen_level` converts
it with `keep_frac = 1.0 / sparsity_factor`.

:::{warning}
The fractions compound. `--sparsity 2,4` does not keep 1/2 then 1/4 of
the *original* population — each level's fraction is taken against the
pool that survived the previous one, because `coarsen_level` calls
`apply_sparsity` with `relative_to="alive"`. See
[object_selection.md](object_selection.md).
:::

## Chunk scaling

`--chunk-scale` (Python: `chunk_scale_factors`) sets a per-level
multiplier on the source level's `chunk_shape`. It is a separate axis
from `--coarsen`: coarsening thins vertices, chunk scaling widens the
grid those vertices are bucketed into.

```python
# 8x fewer vertices AND a 2x-wider chunk grid per level.  A 2x grid in
# 3D is 8x the volume per chunk, which cancels the 8x vertex reduction
# and holds per-chunk payload roughly constant across levels.
build_pyramid("skel.zv", factors=[(8.0, 1.0)], chunk_scale_factors=[2])
```

Multipliers must be positive integers so the grids stay nested — a
non-nested coarse grid would break the "a simplified piece never
straddles a target chunk boundary" invariant the chunk-local coarseners
rely on. A scalar applies to every axis; a tuple must have `sid_ndim`
entries or `CoarseningError` is raised. When the resulting chunk shape
equals the root's, the per-level override is omitted from disk and the
level inherits from root.

## The seed advances per level

`sparsity_seed` is not passed through unchanged. `build_pyramid` sends
`sparsity_seed + i` to level `i`:

```python
# level 1 gets seed=0, level 2 gets seed=1, level 3 gets seed=2
build_pyramid("pts.zv", factors=[(4.0, 2.0)] * 3, sparsity_seed=0)
```

A constant seed would apply the *same* selection to a nested candidate
pool, so `random` would keep re-picking the same prefix of survivors and
the levels would differ only in vertex count. Advancing the seed makes
each level draw a genuinely different subset. `sparsity_seed=None`
leaves every level unseeded (non-reproducible). The ranking strategies
(`length`, `attribute`, `spatial_coverage`) ignore the seed entirely.

## Compression and parallelism

```python
# A chunk array's codec pipeline is fixed when the array is created, so
# `compressor` is forwarded to EVERY level — level 0's codec does not
# propagate to the coarser ones.  Pass what level 0 was ingested with.
build_pyramid("mesh.zv", factors=[(4.0, 1.0)], compressor="zstd")

# `executor` is a map-like (func, items, shared) -> list[result] callable
# used by the chunk-local skeleton and polyline coarseners to parallelise
# per-target-chunk work.  The per_object coarsener ignores it.
from zarr_vectors_tools.ingest._parallel import process_pool_executor

with process_pool_executor(8) as ex:
    build_pyramid("skel.zv", factors=[(8.0, 2.0)], executor=ex)
```

See [compressors.md](../how_to/compressors.md) for codec choice and
[parallelism.md](../how_to/parallelism.md) for executor backends. The CLI
wraps the same thing behind `--compressor` and `--workers` /
`--workers-backend`.

## One level at a time

```python
from zarr_vectors_tools.multiresolution.coarsen import coarsen_level

# Same dispatch, same keywords, but you own the level numbering and the
# cross-level wiring.  Leave cross_level_storage at its "none" default
# unless you are reimplementing build_pyramid's finalise pass.
result = coarsen_level(
    "tracts.zv",
    source_level=0,
    target_level=1,
    coarsen_factor=8.0,
    sparsity_factor=2.0,
    chunk_scale_factor=2,
    sparsity_strategy="length",
    sparsity_seed=0,
)

result["method"]                # e.g. "per_object"
result["preserves_object_ids"]  # bool
result["vertex_count"]          # int
```

`target_level` must not already exist. `coarsen_level` does *not* run
`_finalize_cross_level_for_store`, so a hand-rolled loop over
`coarsen_level` produces no root cross-level metadata and no `±N` arrays
for `N ≥ 2` — see [cross_level_links.md](cross_level_links.md).

## See also

- [Multiresolution index](index.md)
- [Concepts](concepts.md) — why coarsening and sparsity are separate axes.
- [Strategies](strategies.md) — which coarsener your store will get.
- [Object selection](object_selection.md) — what `sparsity_strategy` chooses between.
- [Cross-level links](cross_level_links.md) — `cross_level_storage` / `cross_level_depth`.
- [Compressors](../how_to/compressors.md) — the `compressor=` argument.
- [Parallelism](../how_to/parallelism.md) — the `executor=` argument.
