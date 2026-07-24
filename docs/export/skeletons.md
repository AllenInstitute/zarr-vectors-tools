# Skeletons

The ZVF `graphs` geometry writes out to the seven-column SWC format via
`export_swc`. No third-party dependency.

```python
from zarr_vectors_tools.export.swc import export_swc

summary = export_swc(
    "neuron.zv",                # store_path
    "neuron.swc",               # output_path
    level=0,                    # resolution level to read
    chunks=None,                # the only filter this exporter accepts
)
summary["node_count"]           # → int
```

`export_swc` takes no `bbox`, `object_ids`, or `group_ids` — `chunks` is
the whole filtering surface. Raises `ExportError` if the store (or the
chunk whitelist) yields no nodes.

## Reconstructing the parent column

The store holds an undirected edge list; SWC needs a parent per node. The
exporter derives one using the DFS-ordering convention — **for each edge,
the endpoint with the smaller index is the parent** — and assigns each
node the first parent it encounters. Nodes left without one keep `-1`
and become roots.

Two fields are synthesised rather than read from the store: every node
gets radius `1.0` and compartment type `3` (dendrite), except the first
root, which is typed `1` (soma). Node IDs are renumbered to 1-based
positional indices, so they will not match IDs in the original SWC file.

:::{warning}
Radius is not preserved. If the source skeleton carried per-node radii as
a store attribute, `export_swc` discards them and writes a constant
`1.0000` in column six.
:::

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
- [Headers](../headers.md) — `SWCHeader` holds the comment lines, `coordinate_space`, and `scaling` that ingest preserved.
