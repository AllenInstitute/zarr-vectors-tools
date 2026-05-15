# Quickstart

A round trip through the three workflows this package provides: ingest a
file format → run an algorithm → export back to a file format. All three
hand off through a Zarr Vectors store on disk; no intermediate
materialisation in memory.

## Ingest a point cloud from CSV

```python
from zarr_vectors_tools.ingest import ingest_csv

ingest_csv(
    input_path="points.csv",
    output_path="points.zv",
    chunk_shape=(100.0, 100.0, 100.0),
    auto_detect_columns=True,
    knn_distance_k=8,            # adds attributes/knn_distance
)
```

The store at `points.zv` is now a v3 Zarr group with chunked vertex
arrays, a spatial index, and (because `knn_distance_k` was supplied) a
per-vertex `knn_distance` attribute. See
[Ingest workflows](../ingest/index.md) for the full file-format matrix.

## Run an algorithm

For a graph (or skeleton) store, every algorithm in
`zarr_vectors_tools.algorithms` takes the store path and streams chunks
as it goes.

```python
from zarr_vectors_tools.algorithms import compute_connected_components

result = compute_connected_components("graph.zv", write_back=True)
print(result["n_components"], result["largest_component_size"])
```

`write_back=True` persists the per-vertex component label to
`attributes/component_label/` on the store so subsequent reads (or other
tools downstream) can use it without recomputing. See
[Algorithms](../algorithms/index.md) for the full menu.

For a mesh store:

```python
from zarr_vectors_tools.algorithms import compute_mesh_summary

result = compute_mesh_summary("mesh.zv", per_object=True)
for obj in result["per_object"]:
    print(obj["object_id"], obj["surface_area"], obj["face_count"])
```

## Export to a portable format

```python
from zarr_vectors_tools.export import export_ply

export_ply(
    store_path="points.zv",
    output_path="points.ply",
    bbox=([-100, -100, -100], [100, 100, 100]),   # optional spatial filter
    attribute_names=["knn_distance"],             # write the enrichment we computed
)
```

Every export function accepts `level`, `bbox`, `object_ids`, and `chunks`
filters where they apply. See [Export workflows](../export/index.md).

## End-to-end round trip

```python
from zarr_vectors_tools.ingest import ingest_swc
from zarr_vectors_tools.algorithms import compute_connected_components
from zarr_vectors_tools.export import export_swc

# 1. Ingest a SWC skeleton, auto-deriving Strahler order + node kinds.
ingest_swc(
    input_path="neuron.swc",
    output_path="neuron.zv",
    chunk_shape=(50.0, 50.0, 50.0),
    compute_strahler=True,
    compute_node_kind=True,
)

# 2. Label connected components and persist them on the store.
compute_connected_components("neuron.zv", write_back=True)

# 3. Round-trip back to SWC. The original header is preserved.
export_swc("neuron.zv", "neuron_labelled.swc")
```

The exported SWC carries the same `coordinate_space` and `scaling`
metadata that the source file had — preserved by the
[header registry](../headers.md).

## See also

- [Concepts](concepts.md) — the three workflows in detail.
- [Ingest formats matrix](../ingest/index.md)
- [Algorithm matrix](../algorithms/index.md)
- [Export formats matrix](../export/index.md)
