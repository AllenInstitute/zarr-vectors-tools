"""Skeleton graph-preprocessing algorithms for the downsampling layer.

These are *algorithmic* helpers (not on-disk format access): they massage a
vertex set + undirected edges into the rooted-tree pieces that the format
writer (:func:`zarr_vectors.types.skeletons.write_skeleton_chunk`, via
:func:`zarr_vectors.types.skeletons.decompose_tree_to_paths`) then encodes.

``split_components`` is consumed by skeleton coarsening
(:mod:`zarr_vectors_tools.multiresolution.strategies.skeletons`) and by foreign-format
ingest pipelines (e.g. precomputed ``.frags`` ingest in ``zarr-vectors-tools``).
It deliberately lives in the multiresolution (downsampling) layer rather than
in the format-access ``types`` modules, which stay free of graph algorithms.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np
import numpy.typing as npt


def split_components(
    positions: npt.NDArray[np.floating],
    edges: npt.NDArray[np.integer],
    attributes: dict[str, npt.NDArray] | None = None,
    vertex_ids: npt.NDArray[np.integer] | None = None,
) -> list[dict[str, Any]]:
    """Split a vertex set + undirected edges into rooted-tree pieces.

    Each connected component becomes one piece, re-indexed to a compact
    ``0..k-1`` local range and oriented as ``[child, parent]`` edges
    (BFS from the component's lowest-index node as root; the root has no
    edge row).  Components are returned in ascending order of their
    lowest-index member.

    Args:
        positions: ``(N, D)`` vertex positions.
        edges: ``(M, 2)`` undirected edges (any orientation).
        attributes: Optional ``{name: (N,) or (N, C)}`` per-vertex data.
        vertex_ids: Optional ``(N,)`` opaque ids carried through the
            re-indexing; each piece gets ``"vertex_ids"`` aligned to its
            local vertex order, so callers can locate a specific input
            vertex (e.g. a cross-chunk endpoint) within its component.

    Returns:
        List of ``{"positions", "edges", "attributes"[, "vertex_ids"]}``
        dicts, one per connected component.

    Note: a pure-Python adjacency-list BFS is *intentional* here.  Each call
    operates on one object's (small) merged skeleton, and this runs ~1M times
    per coarsen level; a per-call ``scipy.sparse.csgraph`` formulation is
    dominated by its constant setup cost on tiny graphs and benchmarks ~1.75×
    slower end-to-end.  Vectorization for this layer must instead **batch** a
    whole chunk's components into one graph (see the batched coarsener).
    """
    positions = np.asarray(positions)
    n = len(positions)
    attributes = attributes or {}
    if n == 0:
        return []

    adj: dict[int, list[int]] = defaultdict(list)
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    for a, b in e:
        a = int(a); b = int(b)
        if a == b:
            continue
        adj[a].append(b)
        adj[b].append(a)

    visited = np.zeros(n, dtype=bool)
    pieces: list[dict[str, Any]] = []
    for seed in range(n):
        if visited[seed]:
            continue
        order: list[int] = []
        parent_of: dict[int, int] = {seed: -1}
        dq = deque([seed])
        visited[seed] = True
        while dq:
            u = dq.popleft()
            order.append(u)
            for w in adj.get(u, ()):
                if not visited[w]:
                    visited[w] = True
                    parent_of[w] = u
                    dq.append(w)
        local_of = {g: i for i, g in enumerate(order)}
        comp_edges = [
            (local_of[g], local_of[parent_of[g]])
            for g in order
            if parent_of[g] >= 0
        ]
        gidx = np.asarray(order, dtype=np.int64)
        piece = {
            "positions": positions[gidx],
            "edges": np.asarray(comp_edges, dtype=np.int64).reshape(-1, 2),
            "attributes": {k: np.asarray(v)[gidx] for k, v in attributes.items()},
        }
        if vertex_ids is not None:
            piece["vertex_ids"] = np.asarray(vertex_ids)[gidx]
        pieces.append(piece)
    return pieces
