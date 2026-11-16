# Spatial read benchmark: point clouds, meshes, networks

Moved here from zarr-vectors-py, where it lived as `benchmarks/octree/` on
the `octree` and `perf/speedups-on-octree` branches.  It needs a
zarr-vectors-py with octree chunks: an optional octree sharding codec
(zarr-octree's `zarr_octree_sharding`) and a choice of shard reader.  The
codec is opt-in and the default reader follows it, one zarr array at a time:
an array in the octree codec is read by the optimised reader, any other
exactly as before.  Without the codec, stores are byte-identical and read
with identical I/O.

| Setting | Values |
|---|---|
| `Layout(octree=...)` at construction | `False` (default: `sharding_indexed`); `True` or the codec's configuration, e.g. `{"target": 100000}` (its default), marks the store octree (root `octree_sharding` + capability `octree_sharding`); needs zarr-octree and shards |
| `zarr_vectors.sharding.set_octree(store, octree)` | converts an existing sharded store either way; back to `False` gives the store as written unmarked |
| `ZARR_VECTORS_SHARD_READER` | `auto` (default: per array, `optimised` for a `zarr_octree_sharding` array and `standard` for any other), `standard` (every array as always: a local store's shards read off the files one range per cell, any other store through zarr), `optimised` (every array: each shard index once, coalesced ranges, any store) or `zarr` (zarr's API only) |

## Running

zarr-octree is not on PyPI:
`pip install git+https://github.com/Andrew-Keenlyside/zarr-octree`.

```
python bench_kinds.py [--quick] [--layouts indexed octree] [--kinds points mesh network] [--vertices 4000000]
python figures_kinds.py            # needs matplotlib
```

`--layouts indexed` runs without zarr-octree; `octree` / `runs` are stores
marked octree at construction.  Stores go to `_stores/`,
results to `results/bench_kinds.json`, figures to `results/figures/`.

Data (`kinds_data.py`, synthetic): point clouds, meshes (closed surfaces)
and networks (branching trees) with the same vertex count and the same
uneven density.  Queries: the whole level, one vertex (a 0.5-unit box), one
or more objects, boxes; local, 20 ms and 80 ms per request
(`latency_store.py`); every store and reader must return the same answer.

## The structure being read

Each level of a zarr-vectors store is a group of zarr arrays, all chunked
on the same spatial grid of cells (sharded `--shard_cells` per side):

| Array | Holds | Point cloud | Mesh | Graph |
|---|---|---|---|---|
| `vertices` | positions, one ragged row block per cell | yes | yes | yes |
| `vertex_fragments` | per cell: which rows belong to which fragment (object part) | yes | yes | yes |
| `links/0/<offsets>` | faces / edges, one array per neighbour offset between their vertices' cells | – | 61 arrays here | 15 arrays here |
| `link_fragments` | fragment index over the intra-cell links | – | yes | yes |
| `object_index/manifests` | per object: its (cell, fragment) pairs | yes | yes | yes |

Reads go: object manifest or box → cells → fragment indexes → vertex rows →
links in those cells, kept when every endpoint was read.

## Current limitations (as found)

* **Graphs could not be read by object** (`read_graph(object_ids=)` refused)
  until this branch: they now take the mesh object path
  (`read_object_links`), with a skeleton's implicit edges rebuilt per
  fragment.  Meshes, graphs and skeletons still cannot be read by group.
* **Each array's metadata lists every non-empty cell** (`nonempty_chunks`):
  120 kB per array at 20k points, 685 kB at 4 M.  A cold read fetches it
  for each array it touches, so it dominates small reads.
* **Links are split into one array per neighbour offset** (61 for these
  meshes): each with its own metadata and shards, so a read touches many
  arrays, and links are most of a mesh's or graph's bytes.
* **Through zarr's API, a whole-level read re-reads a shard's index for every
  cell**: GBs read from a store of MBs, set by the cell count, not the data.

## Results (4 M vertices, 64³ cells, 16³ per shard)

`results/bench_kinds.json`, figures in `results/figures/`.  Whole-level
reads through zarr's API were not run (minutes to hours each).

| | Point cloud | Mesh | Network |
|---|---|---|---|
| store size, indexed / octree | 87 / 82 MB | 342 / 103 MB | 156 / 96 MB |
| all, local: standard / optimised | 641 / 10.9 s | 653 / 48 s | 962 / 44 s |
| objects, 20 ms: zarr / optimised | 6.5 / 1.5 s | 4.5 / 2.4 s | not run (before graph object reads) |
| box, 20 ms: zarr / optimised | 2.0 / 1.0 s | 3.6 / 2.1 s | 3.9 / 1.5 s |
| one vertex, 20 ms (any) | 0.71 s | 0.52 s | 0.72 s |

* **The reader, not the layout, sets the time.**  Octree and
  `sharding_indexed` are within ~5 % at 20 and 80 ms; locally octree is up
  to 15 % slower (40 % for objects with the standard reader).
* **Octree shrinks link-heavy stores.**  A `sharding_indexed` shard index is
  16 B for every cell of the shard whether it holds data or not (65 kB at
  16³); with 61 link arrays (mesh) that is ~230 MB of the 342 MB store.
  Octree indexes only occupied cells.  Through zarr it also halves the bytes
  a query reads (it re-reads the index per cell), but at these latencies
  time follows request count, not bytes.
* **One vertex costs ~0.5-0.7 s at 20 ms whatever the store or reader**:
  10-30 MB read, nearly all array metadata (`nonempty_chunks`) and object
  index; the vertex data is bytes.
