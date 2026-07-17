"""Chunked connected components for graph stores.

Per-chunk union-find on intra-chunk edges + a single global pass over
the cross-chunk array. Scales beyond memory because per-chunk edge
arrays are loaded one at a time.

``write_back=True`` persists the component labels via
:meth:`ZVWriter.add_node_attribute_sync` under
``attributes/component_label/``.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from zarr_vectors.core.arrays import list_chunk_keys, read_links
from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors.lazy import open_zv
from zarr_vectors.spatial.boundary import chunk_local_to_global_offsets

from zarr_vectors_tools.algorithms._links import link_prefetch_plan


class _DSU:
    """Simple disjoint-set with path compression + union by rank."""

    __slots__ = ("parent", "rank")

    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int32)

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = int(self.parent[root])
        # Path compression.
        cur = x
        while self.parent[cur] != root:
            nxt = int(self.parent[cur])
            self.parent[cur] = root
            cur = nxt
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def compute_connected_components(
    store_path: str | Path,
    *,
    level: int = 0,
    write_back: bool = False,
) -> dict[str, Any]:
    """Compute 0-indexed connected-component labels for a graph store.

    Args:
        store_path: Path to a zarr-vectors graph (or skeleton) store.
        level: Resolution level to operate on.
        write_back: When True, persist the labels under
            ``attributes/component_label/`` via
            :meth:`ZVWriter.add_node_attribute_sync`.

    Returns:
        Dict with:
          - ``labels``: ``(N,) uint32``, component label per node in
            global ordering (matches ``read_graph``'s position order).
          - ``n_components`` (int)
          - ``largest_component_size`` (int)
          - ``component_sizes`` (np.ndarray): count of nodes per label,
            indexed by component id.
    """
    root = open_store(str(store_path))
    level_group = get_resolution_level(root, level)

    offsets, chunk_keys, n_vertices = chunk_local_to_global_offsets(level_group)

    dsu = _DSU(n_vertices)

    # Connectivity is one family: every record is a tuple of
    # (chunk_coords, local_index) endpoints, and an intra-chunk edge is
    # just one whose endpoints share a chunk.  So a single whole-family
    # read replaces the old per-chunk intra loop PLUS the separate cross
    # read — doing both against read_links would union every intra edge
    # twice and silently double its degree.
    with level_group.batched_reads(link_prefetch_plan(level_group, chunk_keys)):
        try:
            records = read_links(level_group, delta=0)
        except Exception:
            records = []
    for (chunk_a, vi_a), (chunk_b, vi_b) in records:
        dsu.union(offsets[chunk_a] + int(vi_a), offsets[chunk_b] + int(vi_b))

    # Compact roots into 0-indexed labels.
    roots = np.array([dsu.find(i) for i in range(n_vertices)], dtype=np.int64)
    unique_roots, labels = np.unique(roots, return_inverse=True)
    labels = labels.astype(np.uint32)

    sizes = np.zeros(len(unique_roots), dtype=np.int64)
    if n_vertices:
        counts = Counter(int(l) for l in labels)
        for k, v in counts.items():
            sizes[k] = v

    if write_back and n_vertices:
        zv = open_zv(str(store_path))
        with zv[level].writer() as w:
            w.add_node_attribute_sync("component_label", labels, dtype=np.uint32)

    return {
        "labels": labels,
        "n_components": int(len(unique_roots)),
        "largest_component_size": int(sizes.max()) if len(sizes) else 0,
        "component_sizes": sizes,
    }
