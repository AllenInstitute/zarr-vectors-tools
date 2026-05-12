"""Frontier graph search over a chunked store: BFS + Dijkstra / A*.

Builds a tools-side adjacency map from the chunked storage (intra-chunk
edges resolved via global-offset table, cross-chunk edges merged in
from the global cross array), then runs a standard frontier loop in
memory.

This is not yet a true on-demand frontier — adjacency for every node
is built up front rather than chunk-by-chunk — because the missing
core Add 2 (per-chunk cross index) means each frontier expansion would
otherwise scan the full cross array. Once Add 2 lands the implementation
switches to true lazy expansion. Memory cost today: O(N + E).
"""

from __future__ import annotations

import heapq
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np

from zarr_vectors.core.arrays import (
    list_chunk_keys,
    read_chunk_links,
)
from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors_tools.algorithms._cross_chunk_index import build_cross_chunk_index
from zarr_vectors_tools.algorithms._indexing import (
    build_chunk_to_global_offset,
    resolve_endpoints,
)
from zarr_vectors_tools.algorithms._link_attributes import read_chunk_link_attributes


def _build_adjacency(
    level_group,
    *,
    weight_attr: str | None = None,
) -> tuple[list[list[tuple[int, float]]], int]:
    """Materialise an adjacency list keyed by global vertex index.

    Args:
        level_group: Resolution level group.
        weight_attr: Optional name of an edge attribute to use as the
            weight. When ``None`` every edge has weight 1.0.

    Returns:
        ``(adj, n)`` where ``adj[v]`` is a list of ``(neighbour, weight)``
        pairs and ``n`` is the total vertex count.
    """
    offsets, chunk_keys, n_vertices = build_chunk_to_global_offset(level_group)
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n_vertices)]

    for chunk_key in chunk_keys:
        try:
            link_groups = read_chunk_links(level_group, chunk_key, link_width=2)
        except Exception:
            continue
        base = offsets[chunk_key]

        attrs_per_chunk = None
        if weight_attr is not None:
            attrs_per_chunk = read_chunk_link_attributes(
                level_group,
                weight_attr,
                chunk_key,
                [len(g) for g in link_groups],
            )

        for gi, edges in enumerate(link_groups):
            if len(edges) == 0:
                continue
            arr = np.asarray(edges, dtype=np.int64)
            if (
                attrs_per_chunk is not None
                and gi < len(attrs_per_chunk)
                and len(attrs_per_chunk[gi]) == len(arr)
            ):
                weights = np.asarray(attrs_per_chunk[gi], dtype=np.float64)
            else:
                weights = np.ones(len(arr), dtype=np.float64)

            for (u_local, v_local), w in zip(arr, weights):
                u_g = int(u_local) + base
                v_g = int(v_local) + base
                adj[u_g].append((v_g, float(w)))
                adj[v_g].append((u_g, float(w)))

    cross_links, _ = build_cross_chunk_index(level_group)
    if cross_links:
        endpoints_a = [(c, vi) for (c, vi), _ in cross_links]
        endpoints_b = [(c, vi) for _, (c, vi) in cross_links]
        a_global = resolve_endpoints(endpoints_a, offsets)
        b_global = resolve_endpoints(endpoints_b, offsets)
        for ai, bi in zip(a_global, b_global):
            # Cross-chunk edges currently lack a weight slot in core;
            # treat as unit weight (or ``weight_attr=None``).
            adj[int(ai)].append((int(bi), 1.0))
            adj[int(bi)].append((int(ai), 1.0))

    return adj, n_vertices


def bfs_distances(
    store_path: str | Path,
    source: int,
    *,
    level: int = 0,
    max_distance: int | None = None,
) -> dict[str, Any]:
    """Unweighted shortest-path distances from a seed node.

    Args:
        store_path: Path to a graph (or skeleton) store.
        source: Global vertex index to start from.
        level: Resolution level.
        max_distance: Optional cutoff; nodes beyond this distance are
            left as ``-1`` in the output. ``None`` runs until the
            connected component is exhausted.

    Returns:
        Dict with:
          - ``distances`` (np.ndarray int32, shape (N,)): -1 for unreached.
          - ``predecessors`` (np.ndarray int64, shape (N,)): parent node
            on the shortest path; -1 for the source and for unreached nodes.
    """
    root = open_store(str(store_path))
    level_group = get_resolution_level(root, level)
    adj, n = _build_adjacency(level_group)
    if not (0 <= source < n):
        raise IndexError(f"source {source} out of range [0, {n})")

    distances = np.full(n, -1, dtype=np.int32)
    predecessors = np.full(n, -1, dtype=np.int64)
    distances[source] = 0

    q: deque[int] = deque([source])
    while q:
        u = q.popleft()
        if max_distance is not None and distances[u] >= max_distance:
            continue
        for v, _w in adj[u]:
            if distances[v] == -1:
                distances[v] = distances[u] + 1
                predecessors[v] = u
                q.append(v)

    return {"distances": distances, "predecessors": predecessors}


def shortest_path(
    store_path: str | Path,
    source: int,
    target: int,
    *,
    level: int = 0,
    weight: str | None = None,
    heuristic: Callable[[int], float] | None = None,
) -> dict[str, Any]:
    """Dijkstra (or A*, if ``heuristic`` given) shortest path.

    Args:
        store_path: Path to a graph store.
        source: Global vertex index of the start node.
        target: Global vertex index of the goal node.
        level: Resolution level.
        weight: Edge-attribute name to use as edge weight. Default ``None``
            means unit weights (equivalent to BFS).
        heuristic: Optional admissible heuristic mapping node index to
            a lower bound on remaining distance. When supplied the
            search becomes A*.

    Returns:
        Dict with:
          - ``path`` (list[int]): node sequence from source to target.
          - ``cost`` (float): total path cost; ``inf`` if unreachable.
          - ``visited`` (int): how many nodes were popped from the queue.
    """
    root = open_store(str(store_path))
    level_group = get_resolution_level(root, level)
    adj, n = _build_adjacency(level_group, weight_attr=weight)
    if not (0 <= source < n):
        raise IndexError(f"source {source} out of range [0, {n})")
    if not (0 <= target < n):
        raise IndexError(f"target {target} out of range [0, {n})")

    dist = np.full(n, np.inf, dtype=np.float64)
    prev = np.full(n, -1, dtype=np.int64)
    dist[source] = 0.0
    visited = 0

    h = heuristic if heuristic is not None else (lambda _i: 0.0)
    heap: list[tuple[float, int]] = [(float(h(source)), source)]

    while heap:
        _f, u = heapq.heappop(heap)
        visited += 1
        if u == target:
            break
        u_dist = dist[u]
        for v, w in adj[u]:
            alt = u_dist + w
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(heap, (alt + float(h(v)), v))

    if not np.isfinite(dist[target]):
        return {"path": [], "cost": float("inf"), "visited": visited}

    path: list[int] = [target]
    while path[-1] != source:
        path.append(int(prev[path[-1]]))
    path.reverse()
    return {"path": path, "cost": float(dist[target]), "visited": visited}
