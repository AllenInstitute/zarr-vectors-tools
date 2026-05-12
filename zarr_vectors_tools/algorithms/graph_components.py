"""Chunked connected components for graph stores.

Per-chunk union-find on intra-chunk edges + a single global pass over
the cross-chunk array. Scales beyond memory because per-chunk edge
arrays are loaded one at a time.

``write_back=True`` requires Add 5 (post-hoc per-node attribute writes)
to land in core. Today it raises ``NotImplementedError`` — callers can
write the labels themselves by re-running ``write_graph`` with the
returned labels in ``node_attributes``.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from zarr_vectors.core.arrays import list_chunk_keys, read_chunk_links
from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors_tools.algorithms._cross_chunk_index import build_cross_chunk_index
from zarr_vectors_tools.algorithms._indexing import (
    build_chunk_to_global_offset,
    resolve_endpoints,
)


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
        write_back: Reserved. Persisting labels in place requires core
            Add 5; this flag raises ``NotImplementedError`` for now.

    Returns:
        Dict with:
          - ``labels``: ``(N,) uint32``, component label per node in
            global ordering (matches ``read_graph``'s position order).
          - ``n_components`` (int)
          - ``largest_component_size`` (int)
          - ``component_sizes`` (np.ndarray): count of nodes per label,
            indexed by component id.
    """
    if write_back:
        raise NotImplementedError(
            "write_back requires core Add 5 (post-hoc per-node attribute "
            "writes). Re-run write_graph with the returned labels in "
            "node_attributes to persist them today."
        )

    root = open_store(str(store_path))
    level_group = get_resolution_level(root, level)

    offsets, chunk_keys, n_vertices = build_chunk_to_global_offset(level_group)

    dsu = _DSU(n_vertices)

    # Intra-chunk edges: local indices within the chunk become global
    # indices by adding the chunk's start offset.
    for chunk_key in chunk_keys:
        try:
            link_groups = read_chunk_links(level_group, chunk_key, link_width=2)
        except Exception:
            continue
        base = offsets[chunk_key]
        for edges in link_groups:
            if len(edges) == 0:
                continue
            arr = np.asarray(edges, dtype=np.int64)
            for u_local, v_local in arr:
                dsu.union(int(u_local) + base, int(v_local) + base)

    # Cross-chunk edges: each endpoint is (chunk_key, local_index).
    cross_links, _ = build_cross_chunk_index(level_group)
    if cross_links:
        endpoints_a = [(c, vi) for (c, vi), _ in cross_links]
        endpoints_b = [(c, vi) for _, (c, vi) in cross_links]
        a_global = resolve_endpoints(endpoints_a, offsets)
        b_global = resolve_endpoints(endpoints_b, offsets)
        for ai, bi in zip(a_global, b_global):
            dsu.union(int(ai), int(bi))

    # Compact roots into 0-indexed labels.
    roots = np.array([dsu.find(i) for i in range(n_vertices)], dtype=np.int64)
    unique_roots, labels = np.unique(roots, return_inverse=True)
    labels = labels.astype(np.uint32)

    sizes = np.zeros(len(unique_roots), dtype=np.int64)
    if n_vertices:
        counts = Counter(int(l) for l in labels)
        for k, v in counts.items():
            sizes[k] = v

    return {
        "labels": labels,
        "n_components": int(len(unique_roots)),
        "largest_component_size": int(sizes.max()) if len(sizes) else 0,
        "component_sizes": sizes,
    }
