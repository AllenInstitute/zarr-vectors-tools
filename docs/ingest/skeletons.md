# Skeletons

A **skeleton** in ZVF is a tree-structured graph — each node has at
most one parent. Currently one ingest covers this: the SWC format, the
standard for neuronal morphology.

## SWC — `ingest_swc`

```python
from zarr_vectors_tools.ingest import ingest_swc

ingest_swc(
    input_path="neuron.swc",
    output_path="neuron.zv",
    chunk_shape=(50.0, 50.0, 50.0),
    compute_topological_depth=True,
    compute_strahler=True,
    compute_node_kind=True,
)
```

SWC is a 7-column text format: `ID type X Y Z radius parent_ID`. The
ingest reads every non-comment row, remaps non-contiguous IDs, drops
orphaned nodes (parent missing from the file), and writes a ZVF
skeleton (graph with tree convention).

**Per-node attributes always written**

`radius` *(float)*
: Column 5 of the SWC row.

`compartment` *(int)*
: Column 2 — the SWC type code (1 = soma, 2 = axon, 3 = basal dendrite,
  4 = apical dendrite, …).

**Optional enrichment options**

`compute_topological_depth`
: Writes `node_attributes["topological_depth"]` (uint16) — the edge
  count from each node back to the soma.

`compute_strahler`
: Writes `node_attributes["strahler"]` (uint8) — the Strahler stream
  order, a standard measure of branching hierarchy.

`compute_node_kind`
: Writes `node_attributes["node_kind"]` (uint8) — categorical:
  `0 = SOMA`, `1 = BRANCH`, `2 = CONTINUATION`, `3 = TERMINAL`.

`preserve_header`
: Defaults `True`. Stores SWC comment lines (the `#`-prefixed lines that
  many tools use for `coordinate_space` / `scaling` annotations) in a
  [`SWCHeader`](../headers.md) for round-trip export.

## See also

- [Enrichments → skeletons](../enrichments.md#skeletons-swc)
- [Export → SWC](../export/skeletons.md)
- Parent: [Skeleton spec](https://zarr-vectors.readthedocs.io/en/latest/spec/geometry_types/skeleton.html)
