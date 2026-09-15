"""Graph search over a chunked store: BFS, Dijkstra and A*.

The level's links are read once as arrays into a compressed sparse row
adjacency (:mod:`~zarr_vectors_tools.algorithms._graph_edges`).  BFS then
expands a whole frontier per step with numpy; Dijkstra and A* keep a heap,
but walk the adjacency's arrays, and stop at the target.  Memory: O(N + E).

Cross-chunk edges are not a special case: connectivity is one family, so
they take per-edge weights from the same ``link_attributes/<weight>/0/``
family as intra-chunk edges.  A store with no such family falls back to
unit weights silently.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable
from pathlib import Path
from typing import Any

from zarr_vectors.building import get_resolution_level, open_store

from zarr_vectors_tools.algorithms._graph_edges import bfs_levels, read_adjacency


def build_adjacency(
    level_group,
    *,
    weight_attr: str | None = None,
) -> tuple[list[list[tuple[int, float]]], int]:
    """Materialise an adjacency list keyed by global vertex index.

    Kept for callers that want Python lists; the algorithms in this package
    use the array form, :func:`~zarr_vectors_tools.algorithms._graph_edges.read_adjacency`.

    Args:
        level_group: Resolution level group.
        weight_attr: Optional name of an edge attribute to use as the
            weight. When ``None`` every edge has weight 1.0.

    Returns:
        ``(adj, n)`` where ``adj[v]`` is a list of ``(neighbour, weight)``
        pairs and ``n`` is the total vertex count.
    """
    adjacency = read_adjacency(level_group, weight_attr=weight_attr)
    return adjacency.to_lists(), adjacency.n


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
            on a shortest path -- the lowest-numbered neighbour one hop
            nearer the source; -1 for the source and for unreached nodes.
    """
    level_group = get_resolution_level(open_store(str(store_path)), level)
    adjacency = read_adjacency(level_group)
    if not (0 <= source < adjacency.n):
        raise IndexError(f"source {source} out of range [0, {adjacency.n})")
    distances, predecessors = bfs_levels(adjacency, int(source), max_distance)
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
    level_group = get_resolution_level(open_store(str(store_path)), level)
    adjacency = read_adjacency(level_group, weight_attr=weight)
    n = adjacency.n
    if not (0 <= source < n):
        raise IndexError(f"source {source} out of range [0, {n})")
    if not (0 <= target < n):
        raise IndexError(f"target {target} out of range [0, {n})")

    # Plain lists: the heap loop reads and writes them one element at a
    # time, which numpy scalars make several times slower.
    indptr = adjacency.indptr.tolist()
    indices = adjacency.indices.tolist()
    weights = adjacency.weights.tolist()

    inf = float("inf")
    dist = [inf] * n
    prev = [-1] * n
    dist[source] = 0.0
    visited = 0

    h = heuristic if heuristic is not None else (lambda _i: 0.0)
    heap: list[tuple[float, int]] = [(float(h(source)), source)]

    while heap:
        f, u = heapq.heappop(heap)
        if f > dist[u] + float(h(u)):
            # Stale: pushed before a shorter route to ``u`` was found.  A
            # node is not closed once popped, so an admissible heuristic
            # that is not consistent can still reopen it.
            continue
        visited += 1
        if u == target:
            break
        u_dist = dist[u]
        for i in range(indptr[u], indptr[u + 1]):
            v = indices[i]
            alt = u_dist + weights[i]
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(heap, (alt + float(h(v)), v))

    if dist[target] == inf:
        return {"path": [], "cost": inf, "visited": visited}

    path: list[int] = [target]
    while path[-1] != source:
        path.append(prev[path[-1]])
    path.reverse()
    return {"path": path, "cost": float(dist[target]), "visited": visited}
