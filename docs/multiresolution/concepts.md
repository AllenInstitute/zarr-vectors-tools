# Concepts: coarsening versus sparsity

A Zarr Vectors pyramid reduces data along **two orthogonal axes**. Almost
every confusing pyramid result traces back to conflating them, so it is
worth being precise:

| | **Coarsening** | **Sparsity** |
| --- | --- | --- |
| Operates on | vertices *within* an object | whole objects |
| Question it answers | "how detailed is each object?" | "how many objects are there?" |
| Object count | unchanged | reduced |
| Vertex count per surviving object | reduced | unchanged |
| Parameter | `coarsen_factor` | `sparsity_factor` |
| CLI flag | `--coarsen` | `--sparsity` |

Coarsen 100 neurons by 8 and you still have 100 neurons, each drawn with
an eighth of the vertices. Sparsify 100 neurons by 8 and you have about
12 neurons, each drawn at full detail. Most useful pyramids do both.

```text
             coarsening ──▶
           ┌─────────────────────────────────────────┐
           │  ▓▓▓▓▓▓▓▓▓▓▓▓    ▓▓▓▓▓▓        ▓▓▓      │
  sparsity │  ▓▓▓▓▓▓▓▓▓▓▓▓    ▓▓▓▓▓▓        ▓▓▓      │
      │    │  ▓▓▓▓▓▓▓▓▓▓▓▓    ▓▓▓▓▓▓        ▓▓▓      │
      ▼    │  ▓▓▓▓▓▓▓▓▓▓▓▓                           │
           │  ▓▓▓▓▓▓▓▓▓▓▓▓    ▓▓▓▓▓▓                 │
           │  ▓▓▓▓▓▓▓▓▓▓▓▓                  ▓▓▓      │
           └─────────────────────────────────────────┘
             level 0          level 1        level 2
             all objects,     fewer objects, fewer still,
             full detail      less detail    less detail

  each ▓▓▓ row is one object; its width is its vertex count
```

## Coarsening: fewer vertices per object

`coarsen_factor` is a per-object vertex-reduction factor, always **≥ 1**,
where `1.0` is the identity. What it *means* depends on which coarsener
is running — this is the part that surprises people:

| Coarsener | Interpretation of `coarsen_factor` |
| --- | --- |
| `per_object` (default) | metavertex binning — aggregate roughly this many source vertices into each output vertex |
| `polyline`, `coarsen_mode="rdp"` | Douglas-Peucker tolerance scale |
| `polyline`, `coarsen_mode="decimate"` | keep every *n*-th vertex |
| `skeleton` | decimation **stride** — keep every *n*-th node along a chain |

The skeleton reinterpretation is the sharp edge. On a store whose
`links_convention` is `implicit_sequential_with_branches`, `--coarsen 8`
does not mean "bin 8 vertices", it means "stride 8". The visual result is
similar; the arithmetic is not.

Coarsening is **topology-preserving** for skeletons. The skeleton
coarsener always keeps branch points, endpoints, roots, and any
chunk-boundary vertices; it thins only the unbranched chains *between*
those anchors. So an eight-fold coarsened neuron still has exactly the
same branching structure — it is just drawn with fewer points along each
branch. You never lose a branch to coarsening.

:::{note}
`strategies/skeletons.py` exposes two reducers, and the pyramid uses the
second: `simplify_skeleton` (RDP, tolerance-driven) and
`decimate_skeleton` (uniform stride). `coarsen_skeleton_level` calls
**`decimate_skeleton`**, because RDP bottoms out once a skeleton is
near-minimal — it stops reducing at deeper levels — whereas a stride
yields a predictable ~*n*× reduction per level regardless of geometry.
That is why `coarsen_factor` means "stride" on skeleton stores.
:::

## Sparsity: fewer objects

`sparsity_factor` is a **divisor**, also always ≥ 1:

| `--sparsity` | Objects surviving |
| --- | --- |
| `1` | all of them (identity) |
| `2` | about half |
| `8` | about an eighth |

:::{warning}
Two different conventions exist in the codebase and they are inverses of
each other. `coarsen_level(sparsity_factor=...)` and the CLI's
`--sparsity` take a **divisor** (`8` → keep an eighth). The lower-level
`apply_sparsity(sparsity=...)` takes a **fraction to keep** in `(0, 1]`
(`0.125` → keep an eighth). `coarsen_level` converts between them with
`keep_frac = 1.0 / sparsity_factor`. If you call `apply_sparsity`
directly, you are on the fraction side of that boundary.
:::

### Survivors keep their identity

This is the property that makes pyramids useful for interactive viewing:
**object IDs are stable across levels**. An object that survives to level
2 has the same OID it had at level 0. An object that was dropped leaves
an *empty manifest slot* rather than causing everything after it to
renumber.

So a viewer can select object 4 021 at a coarse level, zoom in, and still
be looking at object 4 021 — and a query for an object that was dropped
returns empty rather than returning the wrong neuron.

### Which objects survive

Sparsity needs a policy, set by `--sparsity-strategy` /
`sparsity_strategy`:

| Strategy | Keeps | Good for |
| --- | --- | --- |
| `random` | a uniform random subset | unbiased visual density reduction |
| `length` | the longest objects | tractography, where short streamlines are usually noise |
| `spatial_coverage` | a spatially even spread | avoiding bald patches in dense regions |
| `attribute` | top or bottom ranked by a stored attribute | "keep the highest-confidence objects" |
| `point_thinning` | derived from `bin_shape`, not from `sparsity` | point clouds |

[Object selection](object_selection.md) covers each in detail.

:::{note}
The seed advances per level (`seed + i`), so `random` picks a *different*
subset at each level rather than nesting the same choice. And on
skeleton stores, `sparsity_strategy="random"` silently degrades to the
deterministic `"length"` strategy — see [Strategies](strategies.md).
:::

## The two axes are applied in order

Within a single level, sparsity runs **first**, then coarsening runs on
the survivors:

1. Select which source objects survive (`apply_sparsity`).
2. Coarsen the vertices of each survivor.
3. Write the target level; dropped objects get empty manifest slots.

Which means coarsening never wastes work on an object that is about to be
discarded — and it means the two factors compose multiplicatively per
level, not additively.

## Cumulative or absolute? The `relative_to` question

Levels compound, and there are two defensible ways to read a constant
`--sparsity 2` across three levels:

- **cumulative** — half, then half of that, then half of that: 1/2, 1/4, 1/8.
- **absolute** — half of the *original* count, every time: 1/2, 1/2, 1/2
  (i.e. levels 2 and 3 do nothing once the pool has shrunk to target).

`apply_sparsity` exposes this as `relative_to`:

| Value | Target count | Behaviour across levels |
| --- | --- | --- |
| `"alive"` | `round(len(candidates) * sparsity)` | **cumulative** |
| `"original"` (default) | `round(n_objects * sparsity)` | absolute; saturates |

:::{note}
**Pyramids want `"alive"`, and `coarsen_level` already passes it.** You do
not need to do anything. The default is `"original"` only for backward
compatibility with direct callers who mean an absolute-of-original
target. This matters solely if you call `apply_sparsity` yourself — in
which case the default is probably not what you want for a pyramid.
:::

## The `alive_mask` footgun

Once a level has dropped objects, their OIDs still exist — they just have
empty manifests. A selector that does not know this will happily "keep"
an already-empty object, spending budget on nothing.

`alive_mask` restricts the candidate pool to objects that still have data.
`coarsen_level` builds it from the source manifests and passes it in
automatically:

```python
alive_mask = np.array(
    [len(src_manifests[oid]) > 0 for oid in range(n_src_objects)],
    dtype=bool,
)
```

:::{warning}
Without `alive_mask`, the `random` strategy degrades badly and silently.
By level 3 of an aggressive pyramid, most OIDs are dead, so most of a
uniform random draw lands on empty objects and the level ends up far
sparser than requested.

The ranking strategies (`length`, `attribute`) appear immune, but only
*incidentally* — a dead object's length is degenerate and never wins a
top-N rank. That is not a guarantee, and it does not hold for every
factor sequence. Pass `alive_mask` regardless.
:::

## Choosing factors

A reasonable starting point for most stores is to reduce both axes by a
similar amount per level, so the level's total size falls by roughly the
product:

```bash
# 3 coarser levels; each drops half the objects and 8× the vertex detail.
zvtools pyramid store.zarrvectors --coarsen 8,8,8 --sparsity 2,2,2
```

The lists must be the same length — one entry per coarser level — or
`build_factors` raises `SystemExit`. Level 0 is never in the list; it is
the source.

Two rules of thumb:

- **Favour sparsity for object-dense data** (tractography with millions of
  streamlines): the viewer's bottleneck is object count, not vertices
  per object.
- **Favour coarsening for detail-dense data** (a few large meshes or long
  skeletons): dropping whole objects loses structures the user is looking
  for, while decimating them does not.

## See also

- [Building pyramids](building_pyramids.md) — the practical how-to.
- [Object selection](object_selection.md) — the five sparsity strategies.
- [Strategies](strategies.md) — how a coarsener is chosen, and what each does.
- [Cross-level links](cross_level_links.md) — connecting a level to its neighbours.
- [Choosing chunk and bin shape](../how_to/choose_chunk_and_bin.md)
