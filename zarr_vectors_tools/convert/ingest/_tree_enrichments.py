"""Shared tree (skeleton) enrichment helpers.

Internal to zarr_vectors_tools; not part of the public API.
"""

from __future__ import annotations

import numpy as np

# node_kind codes
SOMA = 0
BRANCH = 1
CONTINUATION = 2
TERMINAL = 3


def compute_tree_metrics(
    parents: np.ndarray,
    root_idx: int | list[int] = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One traversal returning topological depth, Strahler number, and node kind.

    Args:
        parents: ``(N,)`` integer array. ``parents[i]`` is the parent index
            of node i, or -1 for a root. Must encode a valid tree, or a
            forest when ``root_idx`` lists every root.
        root_idx: Index of the root node (used to set node_kind=SOMA), or a
            list of them for a forest -- each tree is measured from its own.

    Returns:
        ``(topological_depth, strahler, node_kind)`` — each ``(N,)``:
          - ``topological_depth`` (uint16): edge count from root to node.
          - ``strahler`` (uint8): Strahler stream order.
          - ``node_kind`` (uint8): SOMA / BRANCH / CONTINUATION / TERMINAL.
    """
    n = len(parents)
    if n == 0:
        return (
            np.zeros((0,), dtype=np.uint16),
            np.zeros((0,), dtype=np.uint8),
            np.zeros((0,), dtype=np.uint8),
        )

    # Build child lists for O(N) traversal.
    children: list[list[int]] = [[] for _ in range(n)]
    for i, p in enumerate(parents):
        if p >= 0 and p != i:
            children[int(p)].append(i)

    roots = [int(root_idx)] if np.isscalar(root_idx) else [int(r) for r in root_idx]
    is_root = set(roots)

    # Topological depth via BFS from the root(s).
    depth = np.zeros((n,), dtype=np.uint16)
    stack = list(roots)
    while stack:
        node = stack.pop()
        for c in children[node]:
            depth[c] = depth[node] + 1
            stack.append(c)

    # Node kind from child count + whether it's the root.
    node_kind = np.full((n,), CONTINUATION, dtype=np.uint8)
    for i in range(n):
        nc = len(children[i])
        if i in is_root:
            node_kind[i] = SOMA
        elif nc == 0:
            node_kind[i] = TERMINAL
        elif nc >= 2:
            node_kind[i] = BRANCH
        # else: 1 child → CONTINUATION (default)

    # Strahler via post-order. Iterative to avoid recursion limits.
    strahler = np.zeros((n,), dtype=np.uint8)
    order: list[int] = []
    visit_stack = list(roots)
    while visit_stack:
        node = visit_stack.pop()
        order.append(node)
        for c in children[node]:
            visit_stack.append(c)
    for node in reversed(order):
        kids = children[node]
        if not kids:
            strahler[node] = 1
            continue
        kid_strahlers = [strahler[c] for c in kids]
        m = max(kid_strahlers)
        if sum(1 for s in kid_strahlers if s == m) >= 2:
            strahler[node] = m + 1
        else:
            strahler[node] = m

    return depth, strahler, node_kind
