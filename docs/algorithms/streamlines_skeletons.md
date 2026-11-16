# Streamlines, skeletons and surfaces

Region selection, filter trees and bundle summaries on streamline stores;
metrics on skeleton stores; segment lookup across stores; and parcel
summaries on cortical surfaces.

## Streamlines

| Function | What it does | Returns |
| --- | --- | --- |
| `select_streamlines(store, *, mask=None, mask_affine=None, box=None, mode="path", level=0, sample_spacing=None, reach=None)` | Streamlines through a region, reading only the chunks near it. | sorted object ids |
| `bundle_summary(store, *, level=None, write=True, orient_endpoints=True)` | Per group: streamline count, length statistics, mean tortuosity, start and end centroids. | DataFrame, one row per group |
| `read_bundle_summary(store, *, level=0)` | The rows `bundle_summary` stored. | DataFrame |

`select_streamlines` takes exactly one region: `mask` (a NIfTI path, a nibabel
image, or an array with `mask_affine`; needs the `trk` extra) or
`box=(min_corner, max_corner)` in RAS mm. `mode` is `"path"` (any part inside),
`"endpoints"` (either end) or `"both_endpoints"`. `reach` is how far past the
region chunks are read (default: one chunk). A segment longer than `reach` can
jump over the region, so select at a fine level.

`bundle_summary` needs groups: TRX groups, LINC TRK labels or
`zvtools merge --group-by`. By default it computes from level 0 and writes
`group_attributes/bundle_*` to every level; `level=N` summarises and writes
level N only. Each bundle is flipped to agree along its main axis before
endpoints are averaged; `orient_endpoints=False` keeps them as stored. The
command `zvtools bundles STORE` does the same, with `--level`, `--no-write`,
`--as-stored` and `--csv FILE`.

```python
from zarr_vectors_tools.algorithms import bundle_summary, select_streamlines

ids = select_streamlines("tracts.zv", box=([15, 25, 40], [30, 40, 50]))
bundle_summary("tracts.zv", write=False)[["streamline_count", "length_mean"]]
```

## Filter trees

`zarr_vectors_tools.filtering` evaluates TrackVis-style filters: regions
(`Sphere`, `Box`, `Slab`, `Disk`, `Mask`), column comparisons (`Column`) and
group membership (`InGroup`), combined with `&` (and), `|` (or), `~` (not) and `-` (and not). A tree has a
JSON form (`to_json`, `from_json`). `filtering.evaluate` tests every segment
and defines the answer; `IndexedEngine` tests only segments near each region,
gives the same answer, and caches every leaf. The subpackage needs only numpy.

```python
from zarr_vectors_tools.filtering import IndexedEngine, Passes, Sphere, load_tractogram

tracts = load_tractogram("tracts.zv")   # level 0, RAS mm
tree = Passes(Sphere((20, 30, 45), 5)) - Passes(Sphere((45, 32, 32), 5))
accepted = IndexedEngine(tracts).evaluate(tree)   # bool per streamline
```

## Skeleton metrics

`compute_skeleton_metrics(store, *, level=0, metrics=None, write=True,
object_ids=None, executor=None)` measures every object and returns a DataFrame
indexed by `object_id`. With `write=True` it stores each metric as
`object_attributes/<metric>`; `object_ids` needs `write=False`.

| Metric | Meaning |
| --- | --- |
| `cable_length` | sum of edge lengths, in store units |
| `node_count`, `component_count` | distinct vertices; connected pieces |
| `leaf_count`, `branch_count` | vertices with at most one neighbour; with three or more |
| `max_strahler` | Strahler order at the root (an SWC soma), else at the tip with the smallest (x, y, z) |
| `extent` | bounding-box size per axis; `extent_x/y/z` in the table |

`SegmentLink(stores, *, level=0)` matches objects across stores by
`object_attributes/segment_id`, which skeleton and `zvtools synapses` stores
have. `resolve(segment_id)` gives the object id in each store,
`link(store_name, object_id)` does the same from one store's object,
`attributes(segment_id, {store_name: [names]})` reads attributes, and
`resolve_many(ids)` returns a DataFrame.

```python
from zarr_vectors_tools.algorithms import SegmentLink, compute_skeleton_metrics

metrics = compute_skeleton_metrics("neurons.zv")
link = SegmentLink({"skeletons": "neurons.zv", "synapses": "synapses.zv"})
link.resolve(1)   # {'skeletons': 0, 'synapses': 0}
```

## Cortical surfaces

| Function | What it does | Returns |
| --- | --- | --- |
| `read_hemisphere(store, hemisphere, *, coords=None, attributes=(), level=0)` | One hemisphere in the source file's vertex order, with positions from any kept surface (`coords`; default the geometry). | `vertices`, `faces`, `source_vertex`, `attributes`, `surface` |
| `parcel_summary(store, parcellation, *, metrics=None, surface="white", ...)` | Per parcel, as FreeSurfer's `?h.aparc.stats`: area on `surface`, each vertex taking a third of its triangles. | DataFrame indexed by (hemisphere, parcel): `code`, `vertex_count`, `surface_area`, `<map>_mean`, `<map>_std`, `<map>_area_weighted_mean` |
| `parcel_at(store, points, parcellation, *, surface="white", ...)` | The parcel of the vertex nearest each point. | DataFrame: `hemisphere`, `parcel`, `code`, `vertex`, `distance` |

```python
from zarr_vectors_tools.algorithms import parcel_summary

parcel_summary("surf.zv", "aparc", metrics=["thickness"])
```

Converting surfaces: [Surfaces](../convert/surfaces.md).
