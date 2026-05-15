# Skeletons

The ZVF skeleton (tree) geometry round-trips to SWC via `export_swc`.

## SWC — `export_swc`

```python
from zarr_vectors_tools.export import export_swc

export_swc(
    store_path="neuron.zv",
    output_path="neuron.swc",
)
```

Returns `{"node_count": int}`.

The exporter reconstructs SWC's parent column from the store's edge
list using the DFS-ordering convention: for each edge, the node with
the smaller index is the parent. Orphaned subtrees (created when a
`chunks` filter drops a parent-child edge) are still written as valid
SWC trees, each with its own root.

`level`
: Resolution level to export.

`chunks`
: Optional whitelist of chunk-coordinate tuples. Nodes outside listed
  chunks are excluded; edges spanning a listed and an unlisted chunk
  are dropped, which can split the tree into multiple smaller trees in
  the output.

If the source store was ingested with `ingest_swc(..., preserve_header=True)`,
the [`SWCHeader`](../headers.md) (comment lines, `coordinate_space`,
`scaling`) is restored as `#`-prefixed lines at the top of the output
file.

## See also

- [Ingest → SWC](../ingest/skeletons.md)
