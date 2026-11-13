# Skeletons

Convert SWC files and Neuroglancer precomputed skeleton layers to skeleton
stores, join synapse tables to them, and export back. A skeleton store holds
trees: vertices joined by parent edges. Chunk shape, `--overwrite` and the
export options are in [Converting files](index.md).

| Format | In / out | Extra | Units |
|---|---|---|---|
| `swc` | in, out | — | the file's, usually µm |
| `precomputed` | in, out | `precomputed` | nm |

```bash
zvtools convert neuron.swc neuron.zv --chunk-shape 50,50,50
```

```python
from zarr_vectors_tools.convert.ingest.swc import ingest_swc

ingest_swc("neuron.swc", "neuron.zv", (50.0, 50.0, 50.0),
           compute_strahler=True)   # also compute_topological_depth, compute_node_kind
```

## What the store holds

- One object per tree, numbered from 0 in root order. A node whose parent is not
  in the file is a root.
- `object_attributes/segment_id`: the precomputed segment id, or 1 to N for SWC
  trees (Neuroglancer treats segment 0 as background).
- Vertex attributes: SWC `radius` and `compartment` (the SWC type); a layer's
  declared vertex attributes; Python metrics such as `strahler`.
- A plain layer's segment properties as object attributes.
- SWC `#` comment lines, in the store's SWC header.

Every name: [Headers and attributes](../reference/headers_attributes.md).
Pyramids keep each tree one tree at every level; see [Pyramids](../pyramids.md).

## Neuroglancer precomputed skeleton layers

INPUT is the skeleton layer: a URL (`gs://`, `s3://`, `https://`, `file://`) or
a local directory holding its `info`. A segmentation layer is refused.

```bash
zvtools convert https://storage.googleapis.com/flywire_v141_m783/skeletons_mip_1 flywire.zv \
    --anchor 17398,10448,3088 --counts 1,1,1 --coarsen 4,4 --sparsity 2,2 --chunk-scale 2,2
```

| Layer | Level-0 chunks | Options |
| --- | --- | --- |
| with a `spatial_index` | one per `.frags` file; `--chunk-shape` is refused | `--anchor X,Y,Z` (voxel corner of one `.frags` chunk) and `--counts NX,NY,NZ` take a block. Default: every `.frags` file. |
| without one | `--chunk-shape X,Y,Z` in nm, required | `--segment-id ID` (repeatable). Default: every id in `segment_properties`. |

The pyramid is built inside the ingest. `--drop-interior-below N` drops, at
coarser levels, objects of at most N vertices that touch no chunk boundary.
In Python, `ingest_precomputed(source, out_store, chunk_shape=None, ...)` takes
the same options as keywords (`anchor`, `counts`, `segment_ids`, `strides` for
`--coarsen`, …).

## Synapses

`zvtools synapses TABLE SKELETONS OUTPUT` joins a CAVE-style synapse table
(`.csv`, `.tsv` or `.parquet`) to a skeleton store by `segment_id`. It writes a
point store with the skeleton store's object ids, and adds
`synapse_pre_count` and `synapse_post_count` to the skeleton store.

```bash
zvtools synapses synapses.csv flywire.zv synapses.zv --resolution 4,4,40
```

| Option | Default | Effect |
| --- | --- | --- |
| `--side post\|pre` | `post` | Whose segment owns each synapse. |
| `--position COLUMN` | `ctr_pt_position` | One column of `[x y z]` strings, or `COLUMN_x`, `_y`, `_z`. |
| `--resolution X,Y,Z` | `1,1,1` | nm per voxel. FlyWire and MICrONS tables are `4,4,40`. |
| `--column NAME` | none | A further numeric column to keep (repeatable). |
| `--unmatched keep\|drop\|error` | `keep` | Synapses on segments the skeleton store lacks. |
| `--no-counts` | off | Leave the skeleton store unchanged. |

## Export to SWC and precomputed

```bash
zvtools convert neuron.zv neuron_out.swc
```

- **SWC.** One file, one tree per object. With `--object-id`, OUTPUT is a
  directory of one file per object, named by segment id. Node ids start at 1;
  without `radius` and `compartment` every node gets radius 1.0 and type 3.
- **Precomputed.** A segmentation layer with `skeletons/` and its
  `segment_properties`, in nm. Segment ids come from `segment_id`, else object
  id + 1. Pass `--unit micrometer` if the store records no unit and is not in
  nm. Integer vertex attributes wider than 32 bits are skipped.
