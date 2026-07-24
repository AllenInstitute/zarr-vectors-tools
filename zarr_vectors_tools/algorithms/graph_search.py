"""Frontier graph search over a chunked store: BFS + Dijkstra / A*.

Builds a tools-side adjacency map from the chunked storage (intra-chunk
edges resolved via the public global-offset helper, cross-chunk edges
merged in from the global cross array), then runs a standard frontier
loop in memory. Memory cost: O(N + E).

Cross-chunk edges are not a special case: connectivity is one family, so
they take per-edge weights from the same ``link_attributes/<weight>/0/``
family as intra-chunk edges.  A store with no such family falls back to
unit weights silently.
"""

from __future__ import annotations

import heapq
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np

from zarr_vectors.core.arrays import (
    list_chunk_keys,
    read_link_attributes,
    read_links,
)
from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors.spatial.boundary import chunk_local_to_global_offsets

from zarr_vectors_tools.algorithms._links import link_prefetch_plan


def build_adjacency(
    level_group,
    *,
    weight_attr: str | None = None,
) -> tuple[list[list[tuple[int, float]]], int]:
    """Materialise an adjacency list keyed by global vertex index.

    Public helper shared with ``graph_clustering``. The clustering
    algorithms (LPA, Louvain) need the same in-memory adjacency that
    ``shortest_path`` builds.

    Cross-chunk edges are not a special case: connectivity is one family,
    so they carry weights from the same ``link_attributes`` family as
    intra-chunk edges and are read by the same call.

    Args:
        level_group: Resolution level group.
        weight_attr: Optional name of an edge attribute to use as the
            weight. When ``None`` every edge has weight 1.0.

    Returns:
        ``(adj, n)`` where ``adj[v]`` is a list of ``(neighbour, weight)``
        pairs and ``n`` is the total vertex count.
    """
    offsets, chunk_keys, n_vertices = chunk_local_to_global_offsets(level_group)
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n_vertices)]

    attrs = (weight_attr,) if weight_attr is not None else ()
    with level_group.batched_reads(
        link_prefetch_plan(level_group, chunk_keys, attrs=attrs)
    ):
        # One whole-family read: every record is already a tuple of
        # (chunk_coords, local_index) endpoints, intra and cross alike.
        # Adding a per-chunk read_chunk_links loop on top of this would
        # union every intra edge twice and silently double its degree.
        try:
            records = read_links(level_group, delta=0)
        except Exception:
            records = []

        weights: np.ndarray | None = None
        if weight_attr is not None and records:
            try:
                weights = read_link_attributes(
                    level_group, weight_attr, delta=0,
                ).astype(np.float64, copy=False)
            except Exception:
                weights = None
            # read_links and read_link_attributes enumerate in the same
            # (segment, cell) order — that shared order is the only thing
            # aligning row i to record i — so a length mismatch means a
            # partial/stale write and the rows cannot be trusted.
            if weights is not None and len(weights) != len(records):
                weights = None

    for i, ((chunk_a, vi_a), (chunk_b, vi_b)) in enumerate(records):
        ai = offsets[chunk_a] + int(vi_a)
        bi = offsets[chunk_b] + int(vi_b)
        w = float(weights[i]) if weights is not None else 1.0
        adj[ai].append((bi, w))
        adj[bi].append((ai, w))

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
    adj, n = build_adjacency(level_group)
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
    adj, n = build_adjacency(level_group, weight_attr=weight)
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
