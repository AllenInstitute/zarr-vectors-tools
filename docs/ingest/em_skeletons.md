# Skeletons in EM

Electron-microscopy segmentations publish skeletons as CloudVolume /
Neuroglancer **precomputed** layers, keyed by uint64 segment ID. Two
ingest paths cover them, and which one you need is decided by the layer's
`info` file:

| Layer has | Function | Reader | Read pattern |
| --- | --- | --- | --- |
| a `spatial_index` block | `zarr_vectors_tools.ingest.precomputed_skeletons.run_ingest` | `PrecomputedFragsReader` | one `<bbox>.frags` MapBuffer per spatial chunk |
| no spatial index | `zarr_vectors_tools.ingest.precomputed_plain_skeletons.run_ingest_plain` | `PlainPrecomputedReader` | one cloud read per segment ID |

Both need the `precomputed` extra:

```bash
pip install "zarr-vectors-tools[precomputed]"   # cloud-volume, mapbuffer, cloud-files
```

Both produce the same thing — a multiscale skeleton store whose object
index preserves the original uint64 segment IDs, so a segment picked in
Neuroglancer resolves to the same object in the ZVF store.

## Spatially indexed sources — `run_ingest`

Each `<x0>-<x1>_<y0>-<y1>_<z0>-<z1>.frags` file is a seung-lab
**MapBuffer** keyed by uint64 segment ID; each value is one precomputed
skeleton *piece* — vertices in nanometres, edges, and whatever per-vertex
attributes the layer declares (`radius`, `cross_sectional_area`). The
pipeline is per-chunk throughout, so RAM stays bounded no matter how
large the block is.

Worked example — a FlyWire cutout, mirroring
`scripts/ingest_cutout_parralel.py`:

```python
import numpy as np

from zarr_vectors_tools.ingest.precomputed_skeletons import (
    PrecomputedFragsReader, enumerate_frag_keys, run_ingest,
)

URL = "gs://flywire_v141_m783/skeletons_mip_1"
COUNTS = np.array([8, 8, 4])            # .frags chunks per axis

reader = PrecomputedFragsReader(URL)     # frags_dir="" → layer root
info = reader.info
res = np.asarray(info.resolution_nm)          # (32, 32, 40) nm at mip_1
csv_ = np.asarray(info.chunk_size_voxels)     # (512, 512, 512) voxels

# The anchor must be a REAL .frags chunk corner, in spatial-index voxels.
anchor = np.array([17398, 10448, 3088])

keys = enumerate_frag_keys(info, tuple(anchor), tuple(COUNTS))
bounds = (
    [float(anchor[a] * res[a]) for a in range(3)],
    [float((anchor[a] + COUNTS[a] * csv_[a]) * res[a]) for a in range(3)],
)

summary = run_ingest(
    reader, "flywire_cutout.zv", keys,
    bounds_nm=bounds,
    strides=[8, 8, 8],              # per-level decimation: keep every 8th vertex
    chunk_scale_factors=[2, 2, 2],  # per-level chunk-grid multiplier
    sparsity_factors=[1, 1, 4],     # per-level object dropping
    drop_interior_below=3,          # LOD: drop tiny fully-interior objects
    workers=8,                      # Dask local cluster; needs the parallel extra
)

# Dask process workers re-import this module, so a __main__ guard is
# mandatory when this lives in a script — without it every worker
# re-runs the ingest and spawns another cluster.
```

### `--anchor` and why keys are enumerated, not listed

`enumerate_frag_keys` walks outward from one known-good chunk corner in
steps of `chunk_size_voxels`, reconstructing filenames from the igneous
`SpatialIndex` naming scheme rather than listing the bucket. A
bucket-wide listing on a production EM layer is expensive and slow, so
the anchor is not a convenience — it is the mechanism. It is a required
argument on the module's own CLI, and passing a corner that is not on the
`.frags` grid yields keys for files that do not exist.

### Chunk-grid alignment

`align=True` (the default) shifts stored coordinates so the block's
minimum corner — itself a `.frags` boundary — maps to 0, and records the
shift as the NGFF world translation. Each `.frags` piece then falls
wholly inside one zarr chunk at level 0, which is what makes the parallel
extract-and-write safe: workers write disjoint files. `align=False`
(`--no-align`) keeps absolute coordinates on the origin-0 grid, so pieces
phase-split across chunk boundaries and the resulting cross-chunk edges
are merged upward by the coarsener.

### Module CLI

`precomputed_skeletons.py` carries its own argparse entry point, separate
from `zvtools`:

```bash
python -m zarr_vectors_tools.ingest.precomputed_skeletons \
    gs://flywire_v141_m783/skeletons_mip_1 \
    flywire_cutout.zv \
    --anchor "17398 10448 3088" \
    --counts "8 8 4" \
    --strides 8,8,8 --chunk-scales 2,2,2 --sparsity 1,1,4 \
    --drop-interior-below 3 \
    --workers 8
```

`--frags-dir` names a subdirectory when the `.frags` files do not sit at
the layer root. `--workers 0` (the default) runs serially.

## Plain sources — `run_ingest_plain`

No spatial index means no way to fetch a spatial region: each segment's
skeleton is its own object in the bucket, so the ingest reads segments
individually and works out the bounding box afterwards. The segment ID
list comes from the layer's `segment_properties/info` inline `ids` array.

Worked example — Allen Mouselight, mirroring `scripts/ingest_mouselight.py`:

```python
from zarr_vectors_tools.ingest.precomputed_plain_skeletons import (
    PlainPrecomputedReader, run_ingest_plain,
)

reader = PlainPrecomputedReader("precomputed://gs://allen_neuroglancer_ccf/Mouselight")
print(len(reader.segment_ids))       # discovered from segment_properties/info

summary = run_ingest_plain(
    reader,
    "mouselight.zv",
    chunk_shape_nm=(1_000_000.0, 1_000_000.0, 1_000_000.0),  # 1 mm³ chunks
    seg_ids=None,                    # None = every ID in the layer
    bounds_nm=None,                  # None = computed from the vertex data
    strides=[8, 8, 8],
    chunk_scale_factors=[2, 2, 2],
    sparsity_factors=[1.0, 1.0, 0.25],
    drop_interior_below=3,
    read_workers=16,                 # THREADS — network I/O-bound
    pyramid_workers=8,               # PROCESSES — CPU-bound coarsening
)
print(summary["objects"], summary["level0_fragments"],
      summary["level0_cross_chunk_edges"])
```

:::{note}
Pass a short `seg_ids` list first — five or ten IDs — to validate the
chunk shape and pyramid settings. A full plain-layer ingest issues one
cloud read per segment, so a wrong `chunk_shape_nm` is expensive to
discover late.
:::

### Two worker pools, two reasons

`read_workers` (default 8) is a `ThreadPoolExecutor`: skeleton reads are
network-bound, spend their time waiting on the GIL released, and threads
share one process's connection pool. `pyramid_workers` (default `None`,
i.e. serial) starts a Dask local cluster of **processes**, because
coarsening is CPU-bound and would serialise behind the GIL in threads. It
needs the `parallel` extra, and it adapts downward per level — a level
with four target chunks uses four workers regardless of what you asked
for.

Raising `read_workers` costs connections, not memory. Raising
`pyramid_workers` costs memory, because each process holds its own
working set.

### Segment properties become object attributes

The plain path converts the layer's inline property table to per-object
arrays via `write_object_attributes`, mapped by property type:

| Precomputed property type | ZVF object attribute dtype |
| --- | --- |
| `number` | the declared `data_type` (`uint8` … `float32`) |
| `label`, `string`, `description` | `S256` fixed-width bytes — decode as UTF-8 |
| `tags` | `uint64` bitmask, bit *N* set when tag *N* applies (tags past 63 are dropped) |

Segments listed in the property table but carrying no geometry are
skipped silently, so the attribute arrays stay aligned with the dense
object IDs rather than with the original table order.

## `drop_interior_below`

Available on both paths, and applied at each **coarse** level only. It
drops an object when its entire decimated skeleton is at or below `N`
vertices *and* every one of its vertices is chunk-interior — no vertex
touches a boundary, so the object cannot extend into a neighbouring
chunk. That second condition is what makes it safe: a fragment that might
continue elsewhere is never removed, only genuinely self-contained
specks. `0` (the default) keeps everything.

## See also

- [Skeletons](skeletons.md) — SWC morphology ingest
- [Parallelism](../how_to/parallelism.md)
- [Building pyramids](../multiresolution/building_pyramids.md)
- [Object selection](../multiresolution/object_selection.md)
- [Ingest workflows](index.md)
