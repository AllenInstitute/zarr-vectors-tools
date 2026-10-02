# Skeletons

```bash
zvtools convert neuron.swc neuron.zv --chunk-shape 50,50,50
```

A skeleton store holds trees: vertices joined by parent edges, one object per
tree. SWC needs no extra; precomputed layers need the `[precomputed]` extra
(cloud-volume, cloud-files, mapbuffer); see [Install](../install.md).

## SWC

One file per call; `--chunk-shape` is in the file's units, usually µm.

- Rows are `ID type X Y Z radius parent`. IDs need not be contiguous; a node
  whose parent is not in the file becomes a root.
- A file with several roots is one object per tree, numbered from 0 in the
  order the roots appear. `object_attributes/segment_id` numbers the same
  trees from 1, because Neuroglancer treats segment 0 as background; the
  precomputed export and `zvtools synapses` use it.
- `radius` and `compartment` (the SWC type: 1 soma, 2 axon, 3 basal
  dendrite, 4 apical dendrite) become vertex attributes.
- `#` comment lines are kept in the store's SWC header; see
  [Headers and attributes](../reference/headers_attributes.md).

From Python, `ingest_swc` adds tree metrics as vertex attributes, each tree
measured from its own root:

```python
from zarr_vectors_tools.convert.ingest.swc import ingest_swc

ingest_swc(
    "neuron.swc", "neuron_metrics.zv",
    (50.0, 50.0, 50.0),               # chunk_shape, in the file's units
    compute_topological_depth=True,   # topological_depth: edges from the root
    compute_strahler=True,            # strahler: Strahler order, tips are 1
    compute_node_kind=True,           # node_kind: 0 soma, 1 branch, 2 continuation, 3 terminal
)
```

### Pyramids

```bash
zvtools convert forest.swc forest.zv --chunk-shape 200,200,200 \
    --coarsen 4,4 --sparsity 1,1 --chunk-scale 2,2
```

Each level keeps every 4th vertex along each branch, and always keeps roots,
branch points, tips and both ends of every edge that still crosses a chunk
boundary. Each tree stays one tree with the same root at every level. A kept
vertex keeps its own `compartment`, and takes the largest `radius` among the
vertices it stands for. `zvtools pyramid` builds the same levels on an
existing SWC store. Skeleton sparsity has no `random`; it becomes `length`.
See [Pyramids](../pyramids.md).

## Neuroglancer precomputed skeleton layers

The input is the skeleton layer itself: a URL (`gs://`, `s3://`, `https://`,
`file://`, with or without `precomputed://`) or a local directory holding its
`info` file. A segmentation layer is refused, with the skeleton directory its
`info` names. `gs://` reads with your Google credentials; a public bucket
also reads without them as `https://storage.googleapis.com/<bucket>/<path>`.

```bash
# Spatially indexed (FlyWire): one .frags chunk, with a pyramid.
zvtools convert https://storage.googleapis.com/flywire_v141_m783/skeletons_mip_1 flywire.zv \
    --anchor 17398,10448,3088 --counts 1,1,1 \
    --coarsen 4,4 --sparsity 2,2 --chunk-scale 2,2 --sparsity-strategy length

# No spatial index (Mouselight): choose a grid in nm, here 1 mm chunks.
zvtools convert https://storage.googleapis.com/allen_neuroglancer_ccf/Mouselight/skeleton \
    mouselight.zv --chunk-shape 1000000,1000000,1000000 \
    --segment-id 18806 --segment-id 18807
```

| Layer | Level-0 chunks | Selecting what to ingest |
| --- | --- | --- |
| with a `spatial_index` | the index's own, one per `.frags` file; `--chunk-shape` is refused | `--anchor X,Y,Z`, the voxel corner of one `.frags` chunk, and `--counts NX,NY,NZ` (default `1,1,1`) take that block. Without `--anchor`, every `.frags` file in the layer is listed and read. `--frags-dir DIR` names their subdirectory. |
| without one | `--chunk-shape X,Y,Z` in nm, required | `--segment-id ID` (repeatable). Default: every id listed in the layer's `segment_properties`; a layer without them needs `--segment-id`. |

- Each segment is one object; `object_attributes/segment_id` keeps its
  uint64 id. The vertex attributes the `info` declares (such as `radius`)
  are kept, and a plain layer's segment properties become object
  attributes.
- The pyramid is built inside the ingest. `--coarsen` is a vertex stride per
  level, and each neuron stays one tree at every level. Skeleton sparsity has
  no `random`; it becomes `length`.
  `--drop-interior-below N` drops, at each coarser level, objects of at most
  N vertices that touch no chunk boundary.
- `--compressor`, `--dtype`, `--bin-shape` and `--cross-level-*` are
  refused. `--workers N` runs the pyramid, and a spatial index's `.frags`
  reads, in worker processes; see [Large data](../large_data.md).

From Python, `zarr_vectors_tools.convert.ingest.precomputed.ingest_precomputed(source,
out_store, chunk_shape=None, ...)` takes the same options as keywords:
`anchor`, `counts`, `frags_dir`, `segment_ids`, `strides` (`--coarsen`),
`chunk_scale_factors`, `sparsity_factors`, `sparsity_strategy`,
`drop_interior_below`, `workers` or `executor`.

## Synapses

`zvtools synapses` joins a CAVE-style synapse table to a skeleton store by
segment id. It writes a point store whose object ids are the skeleton
store's, and adds `synapse_pre_count` and `synapse_post_count` to every level
of the skeleton store.

```bash
zvtools synapses synapses.csv flywire.zv synapses.zv --resolution 4,4,40 --column size
```

| Option | Default | Effect |
| --- | --- | --- |
| `--side post\|pre` | `post` | Whose segment owns each synapse (`post_pt_root_id` or `pre_pt_root_id`); the other is kept as `pre_segment_id` or `post_segment_id`. |
| `--position COLUMN` | `ctr_pt_position` | One column of `[x y z]` strings, or `COLUMN_x`, `_y`, `_z`. |
| `--resolution X,Y,Z` | `1,1,1` | Nanometres per voxel of the positions. The default is rarely right: FlyWire and MICrONS tables are `4,4,40`. |
| `--chunk-shape X,Y,Z` | the skeleton store's | Chunk edge in nm. |
| `--column NAME` | none | A further numeric column to keep (repeatable). |
| `--unmatched keep\|drop\|error` | `keep` | Synapses whose segment the skeleton store lacks: one extra object, dropped, or an error. |
| `--no-counts` | off | Leave the skeleton store unchanged. |

The table is `.csv`, `.tsv` or `.parquet` (with pyarrow), and the skeleton
store needs `object_attributes/segment_id` (the SWC and precomputed ingests
write it).
The synapse `id` column is kept as the join key, so further tables attach
with `zvtools attach`. Skeleton metrics and looking one segment up across
stores: [Streamlines and skeletons](../algorithms/streamlines_skeletons.md).

## Export to SWC and precomputed

```bash
zvtools convert neuron.zv neuron_out.swc
zvtools convert flywire.zv neurons --format swc --object-id 0 --object-id 1
zvtools convert flywire.zv flywire_layer --format precomputed \
    --segment-id 720575940501251441 --segment-id 720575940501247345
```

- **SWC.** Without `--object-id` the whole level is one file, one tree per
  object. With it, OUTPUT is a directory of one file per object, named by
  segment id (`720575940439472850.swc`; `1.swc` for an SWC store's object 0),
  else by object id. `--level` exports a pyramid level.
- Node ids are renumbered from 1. `radius` and `compartment` come from the
  vertex attributes of those names; without them every node gets radius 1.0
  and type 3, and the first root type 1.
- A store ingested from SWC writes back the original `#` comment lines,
  then `# SWC exported by zarr-vectors`. Other stores write only that line.
- **Precomputed.** A URL OUTPUT (`gs://`, `s3://`, `file://`) picks the
  format; a local directory needs `--format precomputed`. The result is a
  segmentation layer with a `skeletons/` directory and its
  `segment_properties`, which Neuroglancer and CloudVolume open;
  `zvtools convert LAYER/skeletons` reads it back.
- Segment ids come from `object_attributes/segment_id`, else object id + 1
  (Neuroglancer never draws segment 0). `--segment-id` or `--object-id`
  select.
- Positions are written in nm. A store that records no unit is taken as nm;
  for anything else pass `--unit micrometer` (or `nanometer`, `millimeter`).
- Float vertex attributes are written as float32 and integers of up to 32
  bits as they are; wider integers are skipped (refused if named with
  `--attribute`). Object attributes become segment properties.
