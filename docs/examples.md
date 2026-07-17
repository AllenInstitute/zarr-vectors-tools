# Examples

The [`examples/`](https://github.com/Andrew-Keenlyside/zarr-vectors-tools/tree/main/examples)
directory of the repository ships eleven runnable Jupyter notebooks
covering every geometry type and every algorithm in this package.

## Notebooks

| Notebook | Topic |
| --- | --- |
| [`01_point_clouds.ipynb`](https://github.com/Andrew-Keenlyside/zarr-vectors-tools/blob/main/examples/01_point_clouds.ipynb) | CSV / LAS / PLY point-cloud ingest and export |
| [`02_lines.ipynb`](https://github.com/Andrew-Keenlyside/zarr-vectors-tools/blob/main/examples/02_lines.ipynb) | Line-segment ingest from CSV |
| [`03_polylines_streamlines.ipynb`](https://github.com/Andrew-Keenlyside/zarr-vectors-tools/blob/main/examples/03_polylines_streamlines.ipynb) | TCK / TRK / TRX streamline round trip |
| [`04_graphs.ipynb`](https://github.com/Andrew-Keenlyside/zarr-vectors-tools/blob/main/examples/04_graphs.ipynb) | GraphML and edge-list ingest |
| [`05_skeletons.ipynb`](https://github.com/Andrew-Keenlyside/zarr-vectors-tools/blob/main/examples/05_skeletons.ipynb) | SWC ingest with Strahler / depth / node-kind enrichments |
| [`06_meshes.ipynb`](https://github.com/Andrew-Keenlyside/zarr-vectors-tools/blob/main/examples/06_meshes.ipynb) | OBJ, PLY, and STL mesh ingest and export |
| [`07_network_metrics.ipynb`](https://github.com/Andrew-Keenlyside/zarr-vectors-tools/blob/main/examples/07_network_metrics.ipynb) | k-core, label propagation, Louvain, and connected components |
| [`08_network_search.ipynb`](https://github.com/Andrew-Keenlyside/zarr-vectors-tools/blob/main/examples/08_network_search.ipynb) | BFS, Dijkstra, and A* path queries |
| [`09_mesh_algorithms.ipynb`](https://github.com/Andrew-Keenlyside/zarr-vectors-tools/blob/main/examples/09_mesh_algorithms.ipynb) | Mesh summary, normals, mean curvature, closest-point, ray casting |
| [`10_graph_clustering.ipynb`](https://github.com/Andrew-Keenlyside/zarr-vectors-tools/blob/main/examples/10_graph_clustering.ipynb) | Community detection comparison on the same graph |
| [`tractogram_to_zarrvectors.ipynb`](https://github.com/Andrew-Keenlyside/zarr-vectors-tools/blob/main/examples/tractogram_to_zarrvectors.ipynb) | Larger end-to-end streamline ingest workflow |

## Running the notebooks

Clone the repo and install the `[all,dev]` extras:

```bash
git clone https://github.com/Andrew-Keenlyside/zarr-vectors-tools
cd zarr-vectors-tools
pip install -e ".[all,dev]"
jupyter notebook examples/
```

The notebooks build the stores they need as they run, so none of them
depend on a committed sample store or on downloading external data.
`tractogram_to_zarrvectors.ipynb` writes `tracts.zarrvectors` from
scratch; run it first if you want that store on disk.
