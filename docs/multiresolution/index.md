# Multiresolution

A resolution pyramid stores the same geometry several times over at
decreasing detail, so a viewer can stream the level that matches its
current zoom instead of the whole dataset. Neuroglancer and similar
clients pull the coarsest level first and refine downwards; a query that
only needs a global answer reads level 3 and never touches level 0, which
is what keeps peak memory bounded on a store larger than RAM.

```python
from zarr_vectors_tools.multiresolution.coarsen import build_pyramid

# Two coarser levels on an existing store.  Each tuple is
# (coarsen_factor, sparsity_factor) and produces one level:
#   level 1 = level 0 coarsened 8x, keeping 1/2 of the objects
#   level 2 = level 1 coarsened 8x, keeping 1/4 of what survived
summary = build_pyramid(
    "tracts.zv",
    factors=[(8.0, 2.0), (8.0, 4.0)],
    sparsity_strategy="length",   # keep the longest objects at each level
    sparsity_seed=0,              # advanced per level internally: seed + i
)

summary["levels_created"]   # 2
summary["method"]           # the coarsener the store's metadata dispatched to
```

`build_pyramid` picks the coarsening strategy from the store's own root
metadata — you do not name it. See [strategies.md](strategies.md) for the
dispatch table.

## The two axes

A pyramid level is produced by two independent reductions:
`coarsen_factor` reduces *vertices within each object*, `sparsity_factor`
drops *whole objects*. They compose but do not substitute for one
another, and conflating them is the usual first mistake.
[concepts.md](concepts.md) works through the distinction in depth; the
rest of this section assumes it.

## Pages

| Page | Covers |
| --- | --- |
| [concepts.md](concepts.md) | Coarsening versus sparsity — the two axes, in depth |
| [building_pyramids.md](building_pyramids.md) | `build_pyramid`, `coarsen_level`, and the `zvtools pyramid` CLI |
| [strategies.md](strategies.md) | How a coarsener is selected, and what each one does |
| [object_selection.md](object_selection.md) | The five selectors, `apply_sparsity`, and its two footgun parameters |
| [cross_level_links.md](cross_level_links.md) | `cross_level_storage`, `cross_level_depth`, and the `±N` link families |
| [refresh.md](refresh.md) | Re-coarsening a pyramid after a level has been edited |

## Who owns what

Core (`zarr_vectors`) deliberately ships only a *basic*, dependency-free
multiresolution layer: the `per_object` binning pyramid and `random`
object selection, plus a plug-in registry at
`zarr_vectors.multiresolution.registry`. Everything richer — skeleton and
polyline coarsening, spatial-coverage / length / attribute /
point-thinning selection, and the third-party dependencies they need —
lives in this package.

```python
# Importing zarr_vectors_tools runs _register_multiscale_strategies(),
# which does two things:
import zarr_vectors_tools  # noqa: F401

#  1. populates the tools-local coarsener registry, so
#     zarr_vectors_tools.multiresolution.coarsen.coarsen_level dispatches;
#  2. calls register_coarsen_strategy("skeleton"|"polyline", ...) and
#     register_selection_strategy("spatial_coverage"|"length"|...) into
#     zarr_vectors.multiresolution.registry.

# Core can now dispatch by name without importing tools:
from zarr_vectors.multiresolution.coarsen import coarsen_level

coarsen_level("skel.zv", 0, 1, method="skeleton")
```

The registration is one-directional. Tools depends on core; core never
depends on tools, and degrades quietly (the registry import is wrapped in
a `try`/`except ImportError`) against a core that predates the registry.

:::{note}
Pyramid *refresh* is the one piece not injected this way.
`EditSession(refresh_pyramid=...)` uses core's own basic refresher — see
[refresh.md](refresh.md) for the workaround.
:::

## See also

- [Concepts](concepts.md) — coarsening versus sparsity.
- [Building pyramids](building_pyramids.md) — the practical how-to.
- [Algorithms](../algorithms/index.md) — what reads the levels you build.
- [Parallelism](../how_to/parallelism.md) — `executor=` for the chunk-local coarseners.
