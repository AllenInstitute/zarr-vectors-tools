# Cross-level links

A cross-level link connects a fine vertex to the coarse metanode it was
aggregated into. Materialising them lets a viewer jump between levels —
click a blob at level 3, get the level-0 vertices under it — without
recomputing the binning. They are stored in the same
`links/<delta>/<offsets>/` families as ordinary connectivity, with a
non-zero `delta` naming the level offset instead of `0`.

```bash
zvtools pyramid tracts.zv --coarsen 8,8 --sparsity 2,4 \
    --cross-level-storage explicit \
    --cross-level-depth 1
```

```python
from zarr_vectors_tools.multiresolution.coarsen import build_pyramid

build_pyramid(
    "tracts.zv",
    factors=[(8.0, 2.0), (8.0, 4.0)],
    # "none" | "implicit" | "explicit".  Default "explicit"
    # (core's DEFAULT_CROSS_LEVEL_STORAGE).
    cross_level_storage="explicit",
    # 0 = none, N = up to +/-N per pair, -1 = walk all level pairs.
    # Default 1 (core's DEFAULT_CROSS_LEVEL_DEPTH).
    cross_level_depth=1,
)
```

Both CLI flags default to `None` and are simply omitted from the
`build_pyramid` call when unset, so the Python defaults above are what
you get.

## Storage modes

| `cross_level_storage` | Writes | Where |
| --- | --- | --- |
| `"none"` | nothing | – |
| `"implicit"` | `+N` only | under the **finer** level |
| `"explicit"` | `+N` **and** `-N` | `+N` under the finer level, `-N` under the coarser |

`"explicit"` is the default because the reverse direction is the one a
viewer needs first — coarse-to-fine drill-down — and deriving it from
`+N` at read time means scanning the whole fine-level family. The `-N`
arrays are not a mirror of the `+N` ones: `_write_cross_level_edges`
re-partitions from the coarse side, because whether an edge is
chunk-aligned is evaluated against the coarse chunk grid, and that split
differs from the fine-side view whenever the two grids do not line up.

## Depth

`cross_level_depth` is the maximum absolute level delta materialised.

| Value | Meaning |
| --- | --- |
| `0` | No cross-level arrays at all (root metadata is still stamped) |
| `N` | Up to `±N` for every level pair |
| `-1` | Walk all available level pairs — `max(levels) - min(levels)` |

Values below `-1` raise `ValueError`, as does a `cross_level_storage`
outside `VALID_XLEVEL_STORAGE`.

## How the deltas get built

The `±1` arrays are emitted **inline**, during each coarsening step, by
`_emit_inline_cross_level_links` — that is the only point at which the
fine→coarse parent map is still in memory for free. Deeper deltas are
composed afterwards.

```python
# What build_pyramid does after the last coarsen_level call:
#
#   _finalize_cross_level_for_store(store_path, cross_level_depth=..., ...)
#
# 1. Stamp root cross_level_depth + cross_level_storage.
# 2. Return early if storage == "none" or depth == 0.
# 3. For each adjacent (fine, fine+1) pair, decode a flat fine->coarse
#    `parent` array back OUT of the on-disk +1 links via
#    _decode_parent_from_plus_one.
# 4. Compose step by step — parent_{+2} = parent_{+1} of parent_{+1} —
#    and write each +N / -N family, for N from 2 up to the depth.
```

Composition, not recomputation: `+2` is derived by chaining two decoded
`+1` maps rather than by re-binning level 0 against level 2's grid. That
keeps the deeper deltas exactly consistent with the shallow ones, and it
is why `max_delta < 2` returns early — `±1` already exists on disk.

:::{warning}
`coarsen_level` on its own does **not** run the finalise pass. A
hand-rolled loop over `coarsen_level` gets whatever inline `±1` arrays
the coarsener emitted and nothing else: no `±N` for `N ≥ 2`, no root
`cross_level_depth` / `cross_level_storage`, no capability token. Use
`build_pyramid` unless you intend to call
`_finalize_cross_level_for_store` yourself.
:::

## The capability token

`CAP_MULTISCALE_LINKS` is stamped on root `format_capabilities` by
`_write_cross_level_edges` — the single choke point that actually writes
a `delta != 0` family — and deliberately *not* by the finalise pass.
Stamping optimistically up front claimed cross-level links on stores that
had none: `max_delta < 2` returns without emitting, and a coarsener that
never emits inline `±1` links (the polyline strategy does not) leaves the
store with no cross-level arrays at all. The token now means what it
says.

```python
from zarr_vectors.core.store import open_store, read_root_metadata

md = read_root_metadata(open_store("tracts.zv", mode="r"))
md.cross_level_depth      # 1
md.cross_level_storage    # "explicit"
md.format_capabilities    # includes "multiscale_links" only if arrays were written
```

`zvtools info STORE` prints the same three fields.

## See also

- [Multiresolution index](index.md)
- [Building pyramids](building_pyramids.md) — where `cross_level_*` is passed.
- [Strategies](strategies.md) — which coarseners emit inline `±1` links.
- [Algorithms](../algorithms/index.md) — the `links/<delta>/<offsets>/` layout and its readers.
