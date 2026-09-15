# Skeletons

The Zarr Vectors skeleton geometry writes out to the seven-column SWC format
via `export_swc`. No third-party dependency.

```python
from zarr_vectors_tools.convert.export.swc import export_swc

summary = export_swc(
    "neuron.zv",                # store_path
    "neuron.swc",               # output_path
    level=0,                    # resolution level to read
    chunks=None,                # optional chunk whitelist
)
summary["node_count"]           # → int
summary["root_count"]           # → int; 1 for a single intact tree
summary["attributes_carried"]   # e.g. ["radius", "compartment"]
```

Raises `ExportError` if the store (or the chunk whitelist) yields no nodes.

## One file per neuron

An EM store holds thousands of segments, and what is usually wanted from it
is one SWC per neuron. Pass `object_ids`, and `output_path` becomes a
directory with one single-tree file per object:

```python
summary = export_swc("flywire_cutout.zv", "neurons/", object_ids=[0, 1, 2])
summary["files"]    # ["neurons/720575940611111111.swc", ...]
```

Files are named by segment id when the store has one (the precomputed
ingesters keep a `segment_id` per object), and by object id otherwise. A
store with segment ids reads just the requested skeletons, touching only
the chunks they live in. Any other skeleton store is read once and split
by its object manifests.

`object_ids` and `chunks` cannot be combined, and an id the level does not
hold raises `ExportError`.

```bash
zvtools convert flywire_cutout.zarrvectors neurons --format swc --object-id 0 --object-id 1
```

The output has no `.swc` extension to infer the format from, so pass
`--format swc`.

## Neuroglancer precomputed

`export_precomputed` writes a segmentation layer that Neuroglancer,
CloudVolume and igneous open: skeleton and graph stores become
`neuroglancer_skeletons`, mesh stores `neuroglancer_legacy_mesh`.

```python
from zarr_vectors_tools.convert.export.precomputed import export_precomputed

summary = export_precomputed("flywire_cutout.zv", "gs://bucket/cutout", level=2)
```

```bash
zvtools convert flywire_cutout.zarrvectors gs://bucket/cutout --level 2
```

A URL output picks this format; for a local directory pass
`--format precomputed`. Segment ids are the store's `segment_id` when it has
one, otherwise object id + 1, because Neuroglancer never draws segment 0.
Coordinates are written in nanometres. For a store that does not record its
unit, such as SWC in micrometres, pass `unit="micrometer"` (`--unit`).
Object attributes become segment properties. `CloudVolume(out)` opens the
layer, and `ingest_precomputed` reads a skeleton layer back to the same
points, radius and labels.

Skeletons that cross a chunk boundary are exported whole: core's per-segment
reader does not return the links between chunks, and the exporter reads them
back itself.

## The parent and type columns

Edges are stored as `[child, parent]`, and that orientation becomes the
parent column. Node ids are renumbered to 1-based positions, so they will
not match the ids in an original SWC file.

Radius and compartment come from the store's `radius` and `compartment`
vertex attributes when it has them; `ingest_swc` writes both, and the
precomputed ingesters carry `radius`. Without them every node gets radius
`1.0` and type `3` (dendrite), and the first root becomes the soma, type
`1`. `attributes_carried` lists which columns came from the store.

## Filtering with `chunks`

```python
# Only nodes stored in these chunks survive.
summary = export_swc("neuron.zv", "slab.swc", chunks=[(0, 0, 0), (0, 0, 1)])
```

An edge spanning a listed and an unlisted chunk is dropped, which orphans
the child node on the listed side. Each surviving connected piece is
still written as a valid SWC tree with its own `-1` root, so one input
neuron can come out as several disjoint trees in a single file.

## See also

- [Export overview](index.md) — shared call shape and the `level=` parameter.
- [Ingest → skeletons](../ingest/skeletons.md) — the symmetric direction.
- [Skeletons in EM](../ingest/em_skeletons.md) — the stores `object_ids` export is for.
- [Headers](../headers.md) — `SWCHeader` holds the comment lines, `coordinate_space`, and `scaling` that ingest preserved.
