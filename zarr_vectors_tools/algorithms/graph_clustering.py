"""Graph clustering / community detection for chunked stores.

Three algorithms covering the common "what are the modules in this
network?" question:

- :func:`compute_k_core` — Batagelj-Zaversnik degree-peeling. Returns
  per-vertex coreness.
- :func:`compute_label_propagation` — synchronous LPA. Returns
  community labels.
- :func:`compute_louvain` — greedy modularity optimisation with the
  classic two-phase Blondel et al. algorithm. Returns community labels
  + final modularity.

All three materialise the in-memory adjacency once via
``graph_search.build_adjacency``. For LPA and Louvain that's
unavoidable (they touch every edge per iteration); for k-core it could
be done in true streaming form once a per-chunk cross index lands in
core (catalog Add 2), but for now uniformity wins.

Cross-chunk edges currently lack a per-edge weight slot in core, so
``compute_louvain(weight=...)`` treats boundary edges as unit weight.
This biases community boundaries slightly away from chunk borders; see
the function's docstring for the workaround.
"""

from __future__ import annotations

import heapq
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from zarr_vectors.core.store import get_resolution_level, open_store

from zarr_vectors_tools.algorithms.graph_search import build_adjacency


# =====================================================================
# k-core decomposition
# =====================================================================

def compute_k_core(
    store_path: str | Path,
    *,
    level: int = 0,
) -> dict[str, Any]:
    """Per-vertex k-coreness via Batagelj-Zaversnik degree-peeling.

    The k-coreness of vertex ``v`` is the largest ``k`` such that ``v``
    belongs to a subgraph where every vertex has degree ≥ k.
    Returns a vector aligned with the global vertex ordering used by
    :func:`zarr_vectors.types.graphs.read_graph`.

    Args:
        store_path: Path to a graph (or skeleton) store.
        level: Resolution level.

    Returns:
        Dict with:
          - ``coreness`` (``(N,) uint32``): per-vertex coreness.
          - ``max_core`` (int): the largest coreness present in the graph.
          - ``core_sizes`` (``(max_core+1,) int64``): how many vertices
            have each coreness value.
    """
    root = open_store(str(store_path))
    level_group = get_resolution_level(root, level)
    adj, n = build_adjacency(level_group)

    if n == 0:
        return {
            "coreness": np.zeros(0, dtype=np.uint32),
            "max_core": 0,
            "core_sizes": np.zeros(1, dtype=np.int64),
        }

    # Working copy of degrees; mutated during peeling.
    degree = np.array([len(nbrs) for nbrs in adj], dtype=np.int64)
    # Track which neighbours are still alive (haven't been peeled yet).
    alive = np.ones(n, dtype=bool)
    coreness = np.zeros(n, dtype=np.int64)

    # Min-heap keyed by (current_degree, vertex). Stale entries (whose
    # degree has been decremented since the push) are filtered on pop.
    heap: list[tuple[int, int]] = [(int(degree[v]), v) for v in range(n)]
    heapq.heapify(heap)

    current_core = 0
    while heap:
        d, v = heapq.heappop(heap)
        if not alive[v] or d != degree[v]:
            continue  # stale entry
        if d > current_core:
            current_core = d
        coreness[v] = current_core
        alive[v] = False
        for u, _w in adj[v]:
            if alive[u]:
                degree[u] -= 1
                heapq.heappush(heap, (int(degree[u]), u))

    max_core = int(coreness.max()) if n else 0
    sizes = np.zeros(max_core + 1, dtype=np.int64)
    for c, count in Counter(coreness.tolist()).items():
        sizes[int(c)] = count

    return {
        "coreness": coreness.astype(np.uint32),
        "max_core": max_core,
        "core_sizes": sizes,
    }


# =====================================================================
# Label propagation
# =====================================================================

def compute_label_propagation(
    store_path: str | Path,
    *,
    level: int = 0,
    max_iter: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    """Synchronous label propagation (Raghavan-Albert-Kumara 2007).

    Each vertex starts in its own community. Per round, every vertex's
    new label is the most frequent label among its neighbours (ties
    broken using a seeded RNG). Convergence: usually 5-20 rounds.

    Args:
        store_path: Path to a graph (or skeleton) store.
        level: Resolution level.
        max_iter: Maximum number of synchronous rounds.
        seed: RNG seed for breaking neighbour-label ties.

    Returns:
        Dict with:
          - ``labels`` (``(N,) uint32``): 0-indexed community labels.
          - ``n_communities`` (int).
          - ``iterations`` (int): number of rounds actually executed.
          - ``converged`` (bool): True if labels stabilised before
            ``max_iter``.
          - ``community_sizes`` (``(n_communities,) int64``).
    """
    root = open_store(str(store_path))
    level_group = get_resolution_level(root, level)
    adj, n = build_adjacency(level_group)

    if n == 0:
        return {
            "labels": np.zeros(0, dtype=np.uint32),
            "n_communities": 0,
            "iterations": 0,
            "converged": True,
            "community_sizes": np.zeros(0, dtype=np.int64),
        }

    rng = np.random.default_rng(seed)
    labels = np.arange(n, dtype=np.int64)

    iterations = 0
    converged = False
    for it in range(max_iter):
        iterations = it + 1
        new_labels = labels.copy()
        for v in range(n):
            if not adj[v]:
                continue
            counts: Counter[int] = Counter(labels[u] for u, _ in adj[v])
            top = counts.most_common()
            best = top[0][1]
            tied = [lbl for lbl, c in top if c == best]
            new_labels[v] = (
                tied[0] if len(tied) == 1 else int(rng.choice(tied))
            )
        if np.array_equal(new_labels, labels):
            converged = True
            labels = new_labels
            break
        labels = new_labels

    # Compact labels into 0..n_communities-1.
    unique, compact = np.unique(labels, return_inverse=True)
    sizes = np.bincount(compact).astype(np.int64)

    return {
        "labels": compact.astype(np.uint32),
        "n_communities": int(len(unique)),
        "iterations": iterations,
        "converged": bool(converged),
        "community_sizes": sizes,
    }


# =====================================================================
# Louvain modularity optimisation
# =====================================================================

def compute_louvain(
    store_path: str | Path,
    *,
    level: int = 0,
    weight: str | None = None,
    max_iter: int = 10,
    seed: int = 0,
) -> dict[str, Any]:
    """Greedy modularity optimisation (Blondel et al. 2008).

    Two-phase loop: local moves to maximise modularity gain, then
    contract each community into a super-node and recurse. Stops when
    a full Phase-2 round yields modularity gain < 1e-6 or ``max_iter``
    outer rounds elapsed.

    Args:
        store_path: Path to a graph (or skeleton) store.
        level: Resolution level.
        weight: Optional edge-attribute name. ``None`` means unit
            weights. **Cross-chunk edges currently have no per-edge
            weight slot in core**, so they always contribute unit weight
            regardless of ``weight=``; this biases community boundaries
            slightly away from chunk borders (see module docstring).
        max_iter: Maximum number of Phase-1+Phase-2 outer rounds.
        seed: RNG seed for tie-breaking in the local-move order.

    Returns:
        Dict with:
          - ``labels`` (``(N,) uint32``): 0-indexed level-0 community
            labels (the dendrogram from Phase-2 levels is collapsed
            back to the original vertices).
          - ``modularity`` (float): final modularity Q.
          - ``n_communities`` (int).
          - ``iterations`` (int): outer rounds executed.
          - ``community_sizes`` (``(n_communities,) int64``).
    """
    root = open_store(str(store_path))
    level_group = get_resolution_level(root, level)
    adj, n = build_adjacency(level_group, weight_attr=weight)

    if n == 0:
        return {
            "labels": np.zeros(0, dtype=np.uint32),
            "modularity": 0.0,
            "n_communities": 0,
            "iterations": 0,
            "community_sizes": np.zeros(0, dtype=np.int64),
        }

    rng = np.random.default_rng(seed)

    # Current per-original-vertex community label. We update this after
    # each Phase-1+2 round by remapping through the super-graph
    # communities.
    base_labels = np.arange(n, dtype=np.int64)

    # The graph the current outer round operates on. We start with the
    # original adjacency, then replace it with the contracted super-graph
    # each round.
    cur_adj = adj
    cur_n = n

    final_q = _modularity_from_adj(cur_adj, base_labels[:cur_n])
    iterations = 0

    for outer in range(max_iter):
        iterations = outer + 1
        labels = _louvain_phase1(cur_adj, rng)
        # Re-label communities to 0..k-1.
        _, compact = np.unique(labels, return_inverse=True)

        # Propagate this round's community assignment back to the
        # original vertices.
        if outer == 0:
            base_labels = compact.astype(np.int64)
        else:
            base_labels = compact[base_labels].astype(np.int64)

        # Contract into super-graph for next round.
        new_n = int(compact.max()) + 1 if cur_n else 0
        if new_n == cur_n:
            break  # no contraction → converged
        cur_adj = _contract(cur_adj, compact, new_n)
        cur_n = new_n
        new_q = _modularity_from_adj(cur_adj, np.arange(cur_n, dtype=np.int64))
        if new_q - final_q < 1e-6:
            final_q = new_q
            break
        final_q = new_q

    # Compact final labels into 0..k-1.
    unique, compact = np.unique(base_labels, return_inverse=True)
    sizes = np.bincount(compact).astype(np.int64)

    return {
        "labels": compact.astype(np.uint32),
        "modularity": float(final_q),
        "n_communities": int(len(unique)),
        "iterations": iterations,
        "community_sizes": sizes,
    }


# =====================================================================
# Louvain helpers
# =====================================================================

def _modularity_from_adj(
    adj: list[list[tuple[int, float]]],
    labels: np.ndarray,
) -> float:
    """Modularity Q = (1/2m) Σ_ij [A_ij - k_i·k_j/(2m)] δ(c_i, c_j)."""
    n = len(adj)
    if n == 0:
        return 0.0
    k = np.array([sum(w for _, w in nbrs) for nbrs in adj], dtype=np.float64)
    two_m = float(k.sum())
    if two_m == 0:
        return 0.0

    intra = 0.0
    for v in range(n):
        cv = labels[v]
        for u, w in adj[v]:
            if labels[u] == cv:
                intra += w
    # intra counts each undirected edge twice (once for u→v, once for v→u).
    # k_C^2 sum:
    comm_k: dict[int, float] = defaultdict(float)
    for v, c in enumerate(labels):
        comm_k[int(c)] += k[v]
    sum_kc_sq = sum(v * v for v in comm_k.values())

    return intra / two_m - sum_kc_sq / (two_m * two_m)


def _louvain_phase1(
    adj: list[list[tuple[int, float]]],
    rng: np.random.Generator,
) -> np.ndarray:
    """Greedy local-move phase. Returns per-vertex community labels."""
    n = len(adj)
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    # Per-vertex degree (sum of incident weights including self-loop ×2).
    k = np.array([sum(w for _, w in nbrs) for nbrs in adj], dtype=np.float64)
    two_m = float(k.sum())
    if two_m == 0:
        return np.arange(n, dtype=np.int64)

    labels = np.arange(n, dtype=np.int64)
    # Σ_tot[C] = sum of degrees of vertices in C.
    sigma_tot = k.copy()

    improved = True
    inner_iter = 0
    max_inner = 20  # standard cap for the inner pass
    while improved and inner_iter < max_inner:
        improved = False
        inner_iter += 1
        order = rng.permutation(n)
        for v in order:
            cv = labels[v]
            # Edge weights from v into each community (excluding self).
            k_iC: dict[int, float] = defaultdict(float)
            for u, w in adj[v]:
                if u == v:
                    continue
                k_iC[int(labels[u])] += w

            # Remove v from its current community for evaluation.
            sigma_tot[cv] -= k[v]
            k_iC_cur = k_iC.get(int(cv), 0.0)

            # Compute best community (including staying put).
            best_c = cv
            best_gain = 0.0
            for c, k_iC_c in k_iC.items():
                # ΔQ for moving v into c (relative to isolated v):
                # ΔQ(v → c) = (k_iC_c / m) - (Σ_tot[c] * k[v]) / (2 m^2)
                gain = k_iC_c - sigma_tot[c] * k[v] / two_m
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            # Compare against staying (which itself was removed from cv).
            stay_gain = k_iC_cur - sigma_tot[cv] * k[v] / two_m
            if stay_gain > best_gain:
                best_gain = stay_gain
                best_c = cv

            sigma_tot[best_c] += k[v]
            if best_c != cv:
                labels[v] = best_c
                improved = True

    return labels


def _contract(
    adj: list[list[tuple[int, float]]],
    labels: np.ndarray,
    new_n: int,
) -> list[list[tuple[int, float]]]:
    """Contract communities into super-nodes; sum inter-community edges."""
    # Use a dict to sum weights per super-edge.
    super_w: list[dict[int, float]] = [defaultdict(float) for _ in range(new_n)]
    for v, nbrs in enumerate(adj):
        cv = int(labels[v])
        for u, w in nbrs:
            cu = int(labels[u])
            super_w[cv][cu] += w
    return [list(d.items()) for d in super_w]
