"""Graph clustering / community detection for chunked stores.

Three algorithms covering the common "what are the modules in this
network?" question:

- :func:`compute_k_core` — degree peeling. Returns per-vertex coreness.
- :func:`compute_label_propagation` — synchronous LPA. Returns
  community labels.
- :func:`compute_louvain` — greedy modularity optimisation with the
  classic two-phase Blondel et al. algorithm. Returns community labels
  + final modularity.

All three read the level's links once as arrays into a compressed sparse
row adjacency (:mod:`~zarr_vectors_tools.algorithms._graph_edges`).
k-core peels every vertex at the current core number in one numpy step,
and a label-propagation round is one sort over the edge ends.  Louvain's
local moves are sequential by definition, so its inner loop still visits
one vertex at a time, over plain lists; its modularity and contraction
are array operations.

Cross-chunk edges are not a special case: connectivity is one family, so
they take per-edge weights from the same ``link_attributes/<weight>/0/``
family as intra-chunk edges.  A store with no such family falls back to
unit weights silently.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from zarr_vectors.building import get_resolution_level, open_store

from zarr_vectors_tools.algorithms._graph_edges import (
    Adjacency,
    compact_labels,
    contract,
    k_core,
    label_propagation_round,
    modularity,
    read_adjacency,
)

# =====================================================================
# k-core decomposition
# =====================================================================

def compute_k_core(
    store_path: str | Path,
    *,
    level: int = 0,
) -> dict[str, Any]:
    """Per-vertex k-coreness by degree peeling.

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
    level_group = get_resolution_level(open_store(str(store_path)), level)
    adjacency = read_adjacency(level_group)

    if adjacency.n == 0:
        return {
            "coreness": np.zeros(0, dtype=np.uint32),
            "max_core": 0,
            "core_sizes": np.zeros(1, dtype=np.int64),
        }

    coreness = k_core(adjacency)
    max_core = int(coreness.max())
    return {
        "coreness": coreness.astype(np.uint32),
        "max_core": max_core,
        "core_sizes": np.bincount(coreness, minlength=max_core + 1).astype(np.int64),
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
          - ``labels`` (``(N,) uint32``): 0-indexed community labels,
            numbered in order of each community's lowest vertex.
          - ``n_communities`` (int).
          - ``iterations`` (int): number of rounds actually executed.
          - ``converged`` (bool): True if labels stabilised before
            ``max_iter``.
          - ``community_sizes`` (``(n_communities,) int64``).
    """
    level_group = get_resolution_level(open_store(str(store_path)), level)
    adjacency = read_adjacency(level_group)
    n = adjacency.n

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
    sources = adjacency.sources()

    iterations = 0
    converged = False
    for it in range(max_iter):
        iterations = it + 1
        new_labels = label_propagation_round(adjacency, labels, sources, rng)
        if np.array_equal(new_labels, labels):
            converged = True
            break
        labels = new_labels

    compact, sizes = compact_labels(labels)
    return {
        "labels": compact,
        "n_communities": int(len(sizes)),
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
            weights.  Every edge, intra- or cross-chunk, takes its weight
            from ``link_attributes/<weight>/0/``; a store without that
            family uses unit weight (silent fallback).
        max_iter: Maximum number of Phase-1+Phase-2 outer rounds.
        seed: RNG seed for tie-breaking in the local-move order.

    Returns:
        Dict with:
          - ``labels`` (``(N,) uint32``): 0-indexed level-0 community
            labels (the dendrogram from Phase-2 levels is collapsed
            back to the original vertices), numbered in order of each
            community's lowest vertex.
          - ``modularity`` (float): final modularity Q.
          - ``n_communities`` (int).
          - ``iterations`` (int): outer rounds executed.
          - ``community_sizes`` (``(n_communities,) int64``).
    """
    level_group = get_resolution_level(open_store(str(store_path)), level)
    adjacency = read_adjacency(level_group, weight_attr=weight)
    n = adjacency.n

    if n == 0:
        return {
            "labels": np.zeros(0, dtype=np.uint32),
            "modularity": 0.0,
            "n_communities": 0,
            "iterations": 0,
            "community_sizes": np.zeros(0, dtype=np.int64),
        }

    rng = np.random.default_rng(seed)

    # Each original vertex's community, remapped through every round's
    # super-graph communities.
    base_labels = np.arange(n, dtype=np.int64)
    current = adjacency
    final_q = modularity(current, base_labels)
    iterations = 0

    for outer in range(max_iter):
        iterations = outer + 1
        labels = _louvain_phase1(current, rng)
        _, compact = np.unique(labels, return_inverse=True)
        compact = compact.reshape(-1).astype(np.int64)
        base_labels = compact if outer == 0 else compact[base_labels]

        new_n = int(compact.max()) + 1
        if new_n == current.n:
            break  # no contraction: converged
        current = contract(current, compact, new_n)
        new_q = modularity(current, np.arange(new_n, dtype=np.int64))
        if new_q - final_q < 1e-6:
            final_q = new_q
            break
        final_q = new_q

    labels_out, sizes = compact_labels(base_labels)
    return {
        "labels": labels_out,
        "modularity": float(final_q),
        "n_communities": int(len(sizes)),
        "iterations": iterations,
        "community_sizes": sizes,
    }


# =====================================================================
# Louvain local moves
# =====================================================================

def _louvain_phase1(adjacency: Adjacency, rng: np.random.Generator) -> np.ndarray:
    """Greedy local-move phase. Returns per-vertex community labels."""
    n = adjacency.n
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    # Per-vertex strength: the sum of incident weights, a self-loop's twice.
    strength = np.bincount(adjacency.sources(), weights=adjacency.weights, minlength=n)
    two_m = float(strength.sum())
    if two_m == 0:
        return np.arange(n, dtype=np.int64)

    # Plain lists: the moves below read and write one vertex at a time.
    indptr = adjacency.indptr.tolist()
    indices = adjacency.indices.tolist()
    weights = adjacency.weights.tolist()
    k = strength.tolist()
    labels = list(range(n))
    # Σ_tot[C] = sum of strengths of the vertices in C.
    sigma_tot = list(k)

    improved = True
    inner_iter = 0
    max_inner = 20  # standard cap for the inner pass
    while improved and inner_iter < max_inner:
        improved = False
        inner_iter += 1
        for v in rng.permutation(n).tolist():
            cv = labels[v]
            k_v = k[v]
            # Edge weights from v into each community (excluding self).
            k_iC: dict[int, float] = defaultdict(float)
            for i in range(indptr[v], indptr[v + 1]):
                u = indices[i]
                if u != v:
                    k_iC[labels[u]] += weights[i]

            # Remove v from its current community for evaluation.
            sigma_tot[cv] -= k_v
            best_c = cv
            best_gain = 0.0
            for c, k_iC_c in k_iC.items():
                # ΔQ for moving v into c, up to a positive factor.
                gain = k_iC_c - sigma_tot[c] * k_v / two_m
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            # Compare against staying (which itself was removed from cv).
            stay_gain = k_iC.get(cv, 0.0) - sigma_tot[cv] * k_v / two_m
            if stay_gain > best_gain:
                best_c = cv

            sigma_tot[best_c] += k_v
            if best_c != cv:
                labels[v] = best_c
                improved = True

    return np.asarray(labels, dtype=np.int64)
