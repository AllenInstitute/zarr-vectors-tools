# Merging and splitting stores

`ingest` makes a store from a file. `export` makes a file from a store.
Neither answers *"put this into the store I already have"* — which is
what a second tractogram, a second imaging run, or a bundle atlas needs.
`zarr_vectors_tools.compose` fills that gap, in both directions:

```text
file  --merge_stores-->  existing store
store --merge_stores-->  existing store
store --split_store -->  many stores
```

```python
from zarr_vectors_tools.compose import merge_stores, split_store

merge_stores("tractogram.zarrvectors", ["bundles.trk"])
split_store("tractogram.zarrvectors", "parts/", by="groups")
```

## Why this is not `add_polylines`

`Dataset.add_polylines` reads like an append and is a replace. Called
twice on one store, the second call leaves the first call's objects gone
and its object-attribute columns overwritten, with no error raised
anywhere:

```python
ds.add_polylines([a, b])   # 2 objects
ds.add_polylines([c])      # 1 object — a and b are gone
```

The append primitive is `EditSession.add_object`, which allocates an id
past the existing ones, splits the vertices across the chunks they land
in, and appends a fragment to each. `merge_stores` is that primitive plus
the five things around it that are otherwise lost.

| Carried | Lost without it |
| --- | --- |
| Object ids | Ids are reallocated; anything referring to the old ones is wrong |
| Groups | A group copied verbatim names whatever now occupies those slots |
| Object attributes | A dense column left short reads back as its fill value, not as an error |
| Vertex attributes | A fragment appended without extending the attribute blob pairs values with the wrong vertices |
| Headers | A second TRK's header overwrites the first's |
| `segment_id` | The merge succeeds and the *pyramid rebuild* fails |
| Pyramid | Coarse levels still describe the store as it was |

## Sources

A source is "geometry from somewhere", and merge never learns which kind
it was handed.

| Class | Reads |
| --- | --- |
| `StoreSource` | Another store, at any level, optionally a subset of ids |
| `FileSource` | A file — natively for TRK/TCK/TRX, staged through a scratch store for everything else |
| `GeometrySource` | Geometry already in memory |

Every source takes a `transform=`: a 4×4 affine or a callable, applied
before anything is binned, so the target's grid and bounds check see
final coordinates.

```python
from zarr_vectors_tools.compose import StoreSource, merge_stores

merge_stores(
    "target.zarrvectors",
    [StoreSource("other.zarrvectors", transform=orig_to_target)],
)
```

## The grid is the target's

A merge writes onto the target's existing cells — that is what keeps the
cost proportional to what is added rather than to what is already there.
Geometry outside those cells is checked for **before** anything is
written, against the shape the arrays were *allocated* with:

```text
StoreError: bundles.trk does not fit the target's grid.
  source extent [3.22, 28.31, 9.68] -> [61.33, 166.1, 94.46]
  grid covers   [0.0, 0.0, 0.0] -> [120.0, 105.0, 171.0]
  overhang      below min by [0, 0, 0], above max by [0, 61.1, 0]
```

The allocated shape and the declared bounds can disagree: a store created
over `0..10` with 3-unit cells declares a 4×4×4 grid, while the array on
disk may be 4×3×4 because the writer sized it from the data it was given.
The allocation is what a write is checked against, so it is what compose
checks against.

`on_out_of_bounds=` chooses what happens:

| Value | Behaviour |
| --- | --- |
| `"raise"` (default) | Refuse, naming the axis and the overhang |
| `"skip"` | Drop the offending objects whole, and count them |
| `"expand"` | Grow the grid upwards to cover them, and refuse whatever growing upwards cannot reach |

Expansion is free and safe: a chunk key is `floor(p / cell)`, an absolute
cell index, so adding cells at the top leaves every existing key meaning
exactly what it meant. Growing *downwards* is always refused — that moves
the origin and renumbers every chunk already written.

## Bundle atlases

The common shape for a labelled atlas is one tractogram, one integer code
per streamline, and a JSON lookup table — three artefacts that only mean
something together. Groups are the store's way of holding that as one
thing, and they are readable without the writing application's source
next to them, which a bare `label_id` column is not.

```python
from zarr_vectors_tools.compose import (
    GeometrySource, derive_groups, load_lut, merge_stores, read_trk,
)

bundles = read_trk("atlas.trk")                     # space as stored
derive_groups(bundles, "label_id", names=load_lut("atlas.json"))
merge_stores(
    "tractogram.zarrvectors",
    [GeometrySource(bundles, transform=to_target_frame)],
)
```

or from the terminal:

```bash
zvtools merge tractogram.zarrvectors atlas.trk \
    --group-by label_id --lut atlas.json \
    --transform orig_to_xpct.npy
```

`load_lut` inverts the `{name: code}` direction these files ship in, and
canonicalises the keys — a LUT read from JSON has integer keys while the
column read from a TRK is `float32`, and without that the join silently
finds nothing and every bundle comes back named `label_id_53`.

## Reading TRK

`read_trk` parses the container directly rather than going through
nibabel, which makes positions, per-point scalars and per-streamline
properties available without materialising a whole `Tractogram` — 22
million points in about a second.

It also reads `n_properties` from byte **238**, where the TrackVis spec
puts it. `ingest/trk_parallel.py` reads it from byte 236, which is inside
the `scalar_name` table: for a file with no properties both give zero and
nothing is wrong, and for a file *with* them every record's stride is
four bytes short and the parse walks off the data.

`space=` chooses the coordinate frame:

| Value | Meaning |
| --- | --- |
| `"voxmm"` (default) | Coordinates exactly as stored — what `zvtools convert` produces without `--apply-affine` |
| `"ras"` | The header affine applied, reaching RAS millimetres (needs nibabel) |

The default is the stored space because matching it is what lets a second
file land on an existing store's grid.

## Splitting

A split is a merge run N times, each output fed by the source restricted
to one set of object ids — so it inherits the same id allocation,
attribute, group, header and provenance handling.

| `by=` | Cuts on |
| --- | --- |
| `"groups"` (default) | The store's named object groups |
| `"attribute"` | Distinct values of a per-object attribute |
| `"objects"` | Explicit `{name: [ids]}` |
| `"provenance"` | The id offsets a previous merge recorded |

`by="provenance"` is what makes a merge reversible: each merge records
the offset it wrote at, and consecutive offsets bracket exactly the
objects one source contributed.

Each part keeps the parent's bounds and cell size by default, so the
pieces stay cell-aligned with the parent and with each other.
`bounds="fit"` tightens each output to its own contents instead, which is
smaller and no longer comparable.

```bash
zvtools split atlas.zarrvectors parts/ --by groups --min-objects 100
zvtools split merged.zarrvectors parts/ --by provenance
```

## Planning first

Both operations have a plan that reads only metadata, so it is cheap on a
store of any size and answers the question that actually blocks a merge:

```bash
zvtools merge target.zarrvectors source.trk --dry-run
zvtools split store.zarrvectors out/ --by groups --dry-run
```

```python
from zarr_vectors_tools.compose import plan_merge, plan_split
```

## Pyramids

Every merge invalidates the coarse levels, because they were built from a
level 0 that has just changed. `pyramid=` says what to do about it:

| Value | Behaviour |
| --- | --- |
| `"rebuild"` (default for merge) | Drop the coarse levels and build them again, reusing the factors inferred from what was there |
| `"drop"` (default for split) | Remove them; a reader gets no pyramid rather than a wrong one |
| `"keep"` | Leave them, stamped `compose_stale`, for when the rebuild is deferred to the end of several merges |

Factors are recovered by dividing consecutive levels' `bin_ratio` and
`object_sparsity`, which are cumulative against level 0. When a level
records too little to divide, the rebuild refuses rather than guessing —
pass `pyramid_factors=` explicitly.

### RDP does not compound

For polylines the default coarsener derives its Douglas-Peucker tolerance
as `min(chunk_shape) * 0.5 * coarsen_factor`, and `chunk_shape` is the
same at every level unless `--chunk-scale` grows it. So a second RDP pass
runs at the *same* tolerance as the first and removes nothing: a pyramid
asked for `--pyramid-coarsen 4,4,4` comes out with three identical coarse
levels.

Three ways to get a pyramid that actually ladders:

| Approach | |
| --- | --- |
| `--coarsen-mode decimate` | Strides are applied to each source level in turn, so `4,4,4` compounds to 64× |
| Sparsity | `--pyramid-coarsen 1,1,1 --pyramid-sparsity 10,10,10` drops whole objects, 10× per level |
| `--chunk-scale` | Grows the cells per level, which grows the RDP tolerance with them |

### Sparsity that keeps every bundle

Plain sparsity draws from one global pool, which quietly destroys a
taxonomy: on the Maffei BG-pathways atlas, `1/4` per level over four
levels leaves 336 of 85,989 streamlines and **sixteen of forty-three
bundles gone entirely** — `opt` (one streamline) by L2, `ac` (four) by
L3, ten more by L4. A coarse level that has lost a third of the atlas
is not a coarse view of the data.

`--sparsity-strategy group` thins each named group by the same factor
with a floor of one, so the proportions coarsen and the set of groups
does not:

```bash
zvtools merge atlas.zarrvectors bundles.trk --create \
    --group-by label_id --lut bundles.json \
    --pyramid-coarsen 1,1,1,1 --pyramid-sparsity 4,4,4,4 \
    --sparsity-strategy group
```

```text
                 global random          within-group
L0  85,989       43/43 bundles          43/43
L1  21,497       43/43                  43/43
L2   5,374       41/43                  43/43
L3   1,344       37/43                  43/43
L4     336       27/43                  43/43
```

Large bundles still coarsen normally — `putvmpfc` goes 51,921 → 12,980 →
3,245 → 811 → 203 — while `opt` holds its single streamline all the way
up. Objects in no group form their own stratum and are thinned like the
rest, so a store holding a whole-brain tractogram *and* a labelled atlas
coarsens the unlabelled bulk while keeping every named bundle.

The cost is bounded and worth naming: the floor means the survivor count
exceeds a strict `1/N` budget by at most one object per group per level.

## Not supported yet

Meshes, graphs and skeletons carry their meaning in `faces` and `edges`,
indexed into a vertex array, and re-indexing those across two stores' id
spaces is not implemented. `merge_stores` refuses them by name rather
than dropping the topology and producing a store that reads as a valid
point cloud and is silently no longer a surface.
