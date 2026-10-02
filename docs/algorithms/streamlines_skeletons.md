# Streamlines, skeletons and surfaces

`tracts.trx` holds two groups, `AF_L` (40 streamlines along x) and `CST_L`
(20 along z); `roi.nii.gz` is a 10 mm cube only `CST_L` crosses. The second
line adds a coarser level (see [Pyramids](../pyramids.md)):

```bash
zvtools convert tracts.trx tracts.zv --chunk-shape 20,20,20 --compute-length --compute-endpoints \
    --coarsen 1 --sparsity 2 --rdp-tolerance 0.5 --chunk-scale 2 --cross-level-storage none
```

## Select streamlines by region

```python
from zarr_vectors_tools.algorithms import select_streamlines
from zarr_vectors_tools.convert.export.trk import export_trk

through = select_streamlines("tracts.zv", mask="roi.nii.gz")        # 20 ids: CST_L
ending = select_streamlines("tracts.zv", box=([0, 20, 20], [15, 40, 40]), mode="endpoints")  # 40 ids: AF_L
export_trk("tracts.zv", "roi_tracts.trk", object_ids=through.tolist())
```

| Argument | Meaning |
| --- | --- |
| `mask` / `box` | the region (pass exactly one): a NIfTI path, a nibabel image, or a 3-D array with `mask_affine`; or `(min_corner, max_corner)` in RAS mm |
| `mode` | `"path"` (default): any part inside; `"endpoints"`: either end; `"both_endpoints"`: both ends |
| `sample_spacing` | mm between points tested along each segment; default: half the smallest voxel edge or box side |

It returns sorted object ids, reading only the chunks the region touches. A
TRK store kept in TrackVis voxel mm is mapped to RAS first. A segment longer
than a chunk can jump over a region: at a heavily simplified `level`, select
at a finer one. A mask needs nibabel (`[trk]` extra).

## Bundle summaries

One row per object group: streamline count; length mean, standard deviation,
minimum, median and maximum; mean tortuosity (path length over endpoint
distance); and start and end centroids. The store needs groups: TRX groups,
LINC TRK labels, or `zvtools merge --group-by`.

```bash
zvtools bundles tracts.zv --csv bundles.csv
# prints the table, then: summarised 2 group(s) from level 0; written to level(s) [0, 1]
```

```python
from zarr_vectors_tools.algorithms import bundle_summary, read_bundle_summary

table = bundle_summary("tracts.zv")                    # write=True, level=None
table.loc["CST_L", ["streamline_count", "length_mean", "start_z", "end_z"]].tolist()  # [20.0, 80.0, 5.0, 85.0]
read_bundle_summary("tracts.zv", level=1)["streamline_count"].tolist()   # [40, 20]
bundle_summary("tracts.zv", level=1, write=False)["streamline_count"].tolist()  # e.g. [21, 9]
```

By default (`level=None`) the rows come from level 0 and go to every level as
`group_attributes/bundle_*`: a coarse level reports the whole bundle, not what
sparsity kept. `--level N` (`level=N`) summarises level N's own members and
writes there only; `--no-write` (`write=False`) stores nothing; `--csv FILE`
adds a CSV. Each bundle is flipped to agree along its main axis before
endpoints are averaged (`start_*` is the lower end); `--as-stored`
(`orient_endpoints=False`) keeps directed streamlines as they are. TRX export
carries the rows as `dpg`. After changing the groups, run it again.

## Skeleton metrics

`compute_skeleton_metrics` measures every object at one level and, by default
(`write=True`), stores each metric at that level as `object_attributes/<metric>`.

```bash
zvtools convert neurons.swc neurons.zv --chunk-shape 32,32,32
```

```python
from zarr_vectors_tools.algorithms import compute_skeleton_metrics

metrics = compute_skeleton_metrics("neurons.zv")       # 2 trees; DataFrame indexed by object_id
metrics.loc[0, ["cable_length", "leaf_count", "branch_count", "max_strahler"]].tolist()
# [124.8528137423857, 3.0, 1.0, 2.0]
```

| Metric | Meaning |
| --- | --- |
| `cable_length` | sum of edge lengths, including edges between chunks, in store units |
| `node_count`, `component_count` | distinct vertices; connected pieces (1 for a single tree) |
| `leaf_count`, `branch_count` | vertices with at most one neighbour (an unbranched root counts); with three or more |
| `max_strahler` | Horton–Strahler order at the stored root (an SWC soma), else at the tip with the smallest (x, y, z) |
| `extent` | bounding-box size per axis: one `(objects, 3)` attribute; `extent_x/y/z` in the table |

Options: `metrics=[...]`; `write=False`; `object_ids=[...]` (needs
`write=False`); `executor=`, a `map`-like callable over chunks, for parallel runs.

## One segment across stores

`SegmentLink` matches objects across stores by `object_attributes/segment_id`,
which [skeleton stores](../convert/skeletons.md) (precomputed and SWC) and
`zvtools synapses` stores have. Here `em.zv` holds three EM skeletons:

```bash
zvtools synapses synapses.csv em.zv synapses.zv
```

```python
from zarr_vectors_tools.algorithms import SegmentLink

link = SegmentLink({"skeletons": "em.zv", "synapses": "synapses.zv"})
link.resolve(720575940611111111)   # {'skeletons': 0, 'synapses': 0}: object id in each store
link.link("synapses", 0)           # {'skeletons': 0, 'synapses': 0}: an object picked in one store
link.attributes(720575940611111111, {"skeletons": ["synapse_post_count"]})
# {'skeletons': {'synapse_post_count': 2}}
link.resolve_many([720575940611111111, 720575940633333333, 99])
# DataFrame by segment_id; <NA>: not in that store at that level
```

## Cortical surfaces

`anat/` holds GIFTI midthickness and white surfaces, `thickness` and an `aparc`
parcellation for both hemispheres (see [Surfaces](../convert/surfaces.md)).

```bash
zvtools convert anat surf.zv --chunk-shape 20,20,20
```

```python
from zarr_vectors_tools.algorithms import parcel_at, parcel_summary, read_hemisphere

lh = read_hemisphere("surf.zv", "lh", coords="white", attributes=["thickness", "aparc"])
lh["vertices"].shape, lh["faces"].shape    # (642, 3) (1280, 3); vertex i = source vertex i

table = parcel_summary("surf.zv", "aparc", metrics=["thickness"])   # index (hemisphere, parcel)
table.loc[("left", "superior"), ["vertex_count", "surface_area", "thickness_mean"]].round(2).tolist()
# [305.0, 481.88, 2.53]
parcel_at("surf.zv", [[-30.0, 0.0, 12.0]], "aparc")   # one row: left, superior, code 1, vertex 25, distance 3.0
```

`read_hemisphere` returns `vertices`, `faces`, `source_vertex`, `attributes`
and `surface` in the source file's vertex order, with positions from any
surface the store kept (`coords`; default: its geometry). `parcel_summary`
follows FreeSurfer's `?h.aparc.stats`: `vertex_count`, `surface_area` on the
`white` surface (`surface=None`: the geometry), each vertex taking a third of
its triangles (`vertex_areas` in `zarr_vectors_tools.algorithms.parcels`),
and `<map>_mean`, `<map>_std` (population) and `<map>_area_weighted_mean`.
`parcel_at` gives the parcel of the `white` vertex nearest each point.
