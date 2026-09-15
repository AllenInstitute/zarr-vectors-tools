"""The array kernels behind the graph algorithms, against plain references.

Each kernel replaced a Python loop over an adjacency list.  The references
here are those loops, written the obvious way, and the graphs are random,
with parallel edges, self-loops, isolated vertices and shuffled ids, so an
answer that depended on edge order or on a vertex's number would show.
"""

from __future__ import annotations

import heapq
from collections import deque

import numpy as np
import pytest

from zarr_vectors_tools.algorithms._graph_edges import (
    bfs_levels,
    build_csr,
    compact_labels,
    component_roots,
    contract,
    k_core,
    label_propagation_round,
    modularity,
)


def _graph(seed: int, n: int = 400, m: int = 520):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, n, m)
    b = rng.integers(0, n, m)
    a = np.concatenate([a, a[:15], [3, 3]])     # parallel edges, then self-loops
    b = np.concatenate([b, b[:15], [3, 3]])
    return n, a.astype(np.int64), b.astype(np.int64)


def _lists(n, a, b, w=None):
    adj = [[] for _ in range(n)]
    for i, (x, y) in enumerate(zip(a.tolist(), b.tolist())):
        weight = 1.0 if w is None else float(w[i])
        adj[x].append((y, weight))
        adj[y].append((x, weight))
    return adj


SEEDS = [0, 1, 2, 3]


@pytest.mark.parametrize("seed", SEEDS)
def test_components_match_a_disjoint_set(seed: int) -> None:
    n, a, b = _graph(seed)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for x, y in zip(a.tolist(), b.tolist()):
        parent[find(x)] = find(y)
    reference = [find(v) for v in range(n)]

    roots = component_roots(n, a, b)
    pairs = set(zip(reference, roots.tolist()))
    assert len(pairs) == len(set(reference)) == len(set(roots.tolist()))
    # Each root is its component's smallest vertex.
    assert all(roots[v] <= v and roots[roots[v]] == roots[v] for v in range(n))


def test_components_of_a_long_shuffled_chain() -> None:
    n = 20_000
    order = np.random.default_rng(7).permutation(n)
    roots = component_roots(n, order[:-1].astype(np.int64), order[1:].astype(np.int64))
    assert (roots == 0).all()


def test_compact_labels_number_by_lowest_vertex() -> None:
    labels, sizes = compact_labels(np.array([9, 4, 9, 7, 4, 4]))
    assert labels.tolist() == [0, 1, 0, 2, 1, 1]
    assert sizes.tolist() == [2, 3, 1]


@pytest.mark.parametrize("seed", SEEDS)
def test_bfs_matches_a_queue(seed: int) -> None:
    n, a, b = _graph(seed)
    adj = _lists(n, a, b)
    adjacency = build_csr(n, a, b)
    source = int(a[0])
    reference = [-1] * n
    reference[source] = 0
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v, _w in adj[u]:
            if reference[v] == -1:
                reference[v] = reference[u] + 1
                queue.append(v)

    distances, predecessors = bfs_levels(adjacency, source)
    assert distances.tolist() == reference
    for v, p in enumerate(predecessors.tolist()):
        if v == source or distances[v] == -1:
            assert p == -1
        else:
            assert distances[p] == distances[v] - 1
            assert p == min(u for u, _w in adj[v] if distances[u] == distances[v] - 1)

    capped, _ = bfs_levels(adjacency, source, max_distance=2)
    assert capped.tolist() == [d if 0 <= d <= 2 else -1 for d in reference]


@pytest.mark.parametrize("seed", SEEDS)
def test_k_core_matches_one_vertex_at_a_time_peeling(seed: int) -> None:
    n, a, b = _graph(seed, m=900)
    adj = _lists(n, a, b)
    degree = [len(nbrs) for nbrs in adj]
    alive = [True] * n
    reference = [0] * n
    heap = [(d, v) for v, d in enumerate(degree)]
    heapq.heapify(heap)
    core = 0
    while heap:
        d, v = heapq.heappop(heap)
        if not alive[v] or d != degree[v]:
            continue
        core = max(core, d)
        reference[v] = core
        alive[v] = False
        for u, _w in adj[v]:
            if alive[u]:
                degree[u] -= 1
                heapq.heappush(heap, (degree[u], u))

    assert k_core(build_csr(n, a, b)).tolist() == reference


def test_label_propagation_takes_the_commonest_neighbour_label() -> None:
    # Vertex 0's neighbours carry labels 5, 5, 6; vertex 4 has no neighbours.
    a = np.array([0, 0, 0, 1], dtype=np.int64)
    b = np.array([1, 2, 3, 2], dtype=np.int64)
    adjacency = build_csr(5, a, b)
    labels = np.array([0, 5, 5, 6, 8], dtype=np.int64)
    out = label_propagation_round(adjacency, labels, adjacency.sources(),
                                  np.random.default_rng(0))
    assert out[0] == 5
    assert out[3] == 0 and out[4] == 8
    assert out[1] in (0, 5) and out[2] in (0, 5)


def test_label_propagation_breaks_ties_among_the_tied_only() -> None:
    # A star: the centre sees four labels once each, every tie.
    a = np.zeros(4, dtype=np.int64)
    b = np.arange(1, 5, dtype=np.int64)
    adjacency = build_csr(5, a, b)
    labels = np.array([0, 11, 12, 13, 14], dtype=np.int64)
    seen = {
        int(label_propagation_round(adjacency, labels, adjacency.sources(),
                                    np.random.default_rng(s))[0])
        for s in range(40)
    }
    assert seen == {11, 12, 13, 14}


@pytest.mark.parametrize("seed", SEEDS)
def test_modularity_and_contraction_match_the_lists(seed: int) -> None:
    n, a, b = _graph(seed)
    rng = np.random.default_rng(seed)
    w = rng.random(len(a)) + 0.5
    labels = rng.integers(0, 12, n).astype(np.int64)
    adjacency = build_csr(n, a, b, w)
    adj = _lists(n, a, b, w)

    k = np.array([sum(x for _u, x in nbrs) for nbrs in adj])
    two_m = k.sum()
    intra = sum(x for v in range(n) for u, x in adj[v] if labels[u] == labels[v])
    community = np.bincount(labels, weights=k)
    reference = intra / two_m - (community ** 2).sum() / two_m ** 2
    assert modularity(adjacency, labels) == pytest.approx(reference)

    compact = np.unique(labels, return_inverse=True)[1].reshape(-1)
    new_n = int(compact.max()) + 1
    super_graph = contract(adjacency, compact, new_n)
    # A partition's modularity is its communities' modularity on the super-graph.
    assert modularity(super_graph, np.arange(new_n)) == pytest.approx(reference)
    assert super_graph.weights.sum() == pytest.approx(adjacency.weights.sum())
