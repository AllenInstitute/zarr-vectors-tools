# Skeletons

A **skeleton** in ZVF is a tree-structured graph: every node has at most
one parent. `ingest_swc` covers the light-microscopy morphology case —
one reconstructed neuron per file, in the standard SWC text format. For
segment skeletons pulled out of an EM volume, see
[Skeletons in EM](em_skeletons.md).

## SWC — `ingest_swc`

```python
from zarr_vectors_tools.ingest.swc import ingest_swc

summary = ingest_swc(
    "neuron.swc",
    "neuron.zv",
    (50.0, 50.0, 50.0),               # chunk_shape in the file's own units
    compute_topological_depth=True,   # edges from the soma
    compute_strahler=True,            # branching hierarchy
    compute_node_kind=True,           # soma / branch / continuation / terminal
    preserve_header=True,             # default — keep the # comment lines
)
```

Pure-Python parser, no extra to install. SWC is seven whitespace-separated
columns — `ID type X Y Z radius parent_ID` — and the ingest reads every
non-comment row with at least seven fields, remaps non-contiguous IDs to
dense indices, and drops edges whose parent is missing from the file
(orphans stay as isolated nodes). `parent_ID == -1` marks a root.

Two per-node attributes are always written: `radius` from column 5, and
`compartment` from column 2 — the SWC type code, where 1 is soma, 2 axon,
3 basal dendrite, 4 apical dendrite.

## What the three tree metrics mean

`compute_topological_depth`
: Number of edges from each node back to the root. Path distance in
  *topology*, not in micrometres — it is unaffected by how densely the
  reconstruction was sampled, which makes it the right axis for
  comparing branch positions across neurons traced by different people.

`compute_strahler`
: Strahler order. Terminal tips are order 1; when two branches of equal
  order `n` meet, the parent becomes `n + 1`; when unequal orders meet,
  the parent takes the larger. So the order at the soma summarises how
  deeply nested the whole arbor is, and low-order nodes are the fine
  distal twigs. Filtering to `strahler >= 3` is a quick way to strip
  terminal detail and see the trunk.

`compute_node_kind`
: A categorical label — `0 = soma`, `1 = branch` (more than one child),
  `2 = continuation` (exactly one child), `3 = terminal` (no children).
  Most nodes in a dense reconstruction are continuations; selecting on
  kind gives you the branch and tip point sets directly, without
  recomputing the degree.

All three are computed in one pass over the parent-index array, so
enabling all three costs the same as enabling one.

:::{note}
The metrics are computed from a single root — the first node whose
parent is `-1`. An SWC holding several disconnected trees will produce
depths measured from only one of them.
:::

`preserve_header=True` stores the `#`-prefixed comment lines in an
`SWCHeader`. Many tools stash `coordinate_space` and scaling annotations
there, and export needs them to write a file the original toolchain will
still read.

## See also

- [Skeletons in EM](em_skeletons.md) — precomputed / CloudVolume sources
- [Enrichments → skeletons](../enrichments.md#skeletons-swc)
- [Export → SWC](../export/skeletons.md)
- [Headers](../headers.md)
- [Ingest workflows](index.md)
