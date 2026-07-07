"""Graph and skeleton coarsening strategies.

Two approaches:

1. **Grid contraction**: merge nodes in the same spatial bin into a
   metanode.  Intra-bin edges collapse; inter-bin edges become
   metanode edges.  Edge weights are summed or averaged.

2. **Skeleton pruning**: for tree-structured graphs, remove short
   terminal branches below a length threshold while preserving
   branch topology.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors_tools.multiresolution.metanodes import generate_metanodes


# ===================================================================
# Grid contraction (general graphs)
# ===================================================================

def coarsen_graph(
    positions: npt.NDArray[np.floating],
    edges: npt.NDArray[np.integer],
    bin_size: float | tuple[float, ...],
    *,
    node_attributes: dict[str, npt.NDArray] | None = None,
    edge_weights: npt.NDArray[np.floating] | None = None,
    agg_mode: str = "mean",
) -> dict[str, Any]:
    """Coarsen a graph by contracting nodes within spatial bins.

    Nodes in the same bin merge into a metanode (centroid position).
    Edges between different bins become metanode edges.  Self-loops
    (intra-bin edges) are removed.  Parallel edges are merged by
    summing weights.

    Args:
        positions: ``(N, D)`` node positions.
        edges: ``(M, 2)`` edge list.
        bin_size: Spatial bin edge length.
        node_attributes: Per-node attributes to aggregate.
        edge_weights: ``(M,)`` edge weights.  Default: all 1.0.
        agg_mode: Node attribute aggregation mode.

    Returns:
        Dict with:
        - ``positions``: ``(K, D)`` metanode positions
        - ``edges``: ``(E, 2)`` metanode edge list
        - ``edge_weights``: ``(E,)`` aggregated weights
        - ``node_attributes``: aggregated node attributes
        - ``children``: list of K arrays of original node indices
        - ``node_count``, ``edge_count``
        - ``reduction_ratio``: original nodes / metanodes
    """
    n_nodes = len(positions)
    n_edges = len(edges)
    ndim = positions.shape[1]

    if n_nodes == 0:
        return _empty_graph_coarsen(ndim)

    # Generate metanodes
    meta_result = generate_metanodes(
        positions, bin_size,
        attributes=node_attributes,
        agg_mode=agg_mode,
    )

    meta_pos = meta_result["metanode_positions"]
    children = meta_result["children"]
    meta_attrs = meta_result["metanode_attributes"]
    n_meta = len(meta_pos)

    # Build node → metanode mapping
    node_to_meta = np.empty(n_nodes, dtype=np.int64)
    for m_idx in range(n_meta):
        for c in children[m_idx]:
            node_to_meta[c] = m_idx

    # Map edges to metanode edges, removing self-loops
    if edge_weights is None:
        edge_weights = np.ones(n_edges, dtype=np.float32)
    else:
        edge_weights = np.asarray(edge_weights, dtype=np.float64)

    # Remap and aggregate
    meta_edge_dict: dict[tuple[int, int], float] = {}
    for i in range(n_edges):
        ma = int(node_to_meta[edges[i, 0]])
        mb = int(node_to_meta[edges[i, 1]])
        if ma == mb:
            continue  # self-loop
        key = (min(ma, mb), max(ma, mb))
        meta_edge_dict[key] = meta_edge_dict.get(key, 0.0) + float(edge_weights[i])

    if meta_edge_dict:
        meta_edges = np.array(list(meta_edge_dict.keys()), dtype=np.int64)
        meta_weights = np.array(list(meta_edge_dict.values()), dtype=np.float32)
    else:
        meta_edges = np.zeros((0, 2), dtype=np.int64)
        meta_weights = np.zeros(0, dtype=np.float32)

    return {
        "positions": meta_pos,
        "edges": meta_edges,
        "edge_weights": meta_weights,
        "node_attributes": meta_attrs,
        "children": children,
        "node_count": n_meta,
        "edge_count": len(meta_edges),
        "reduction_ratio": n_nodes / max(n_meta, 1),
    }


# ===================================================================
# Skeleton pruning (trees)
# ===================================================================

def prune_skeleton(
    positions: npt.NDArray[np.floating],
    edges: npt.NDArray[np.integer],
    *,
    min_branch_length: float = 0.0,
    min_branch_vertices: int = 0,
    node_attributes: dict[str, npt.NDArray] | None = None,
) -> dict[str, Any]:
    """Remove short terminal branches from a skeleton tree.

    A terminal branch is a path from a leaf to the nearest branch
    point (node with degree ≥ 3).  Branches shorter than
    ``min_branch_length`` (Euclidean path length) or with fewer than
    ``min_branch_vertices`` nodes are removed.

    Args:
        positions: ``(N, D)`` node positions.
        edges: ``(M, 2)`` edge list.
        min_branch_length: Minimum Euclidean path length to keep.
        min_branch_vertices: Minimum node count to keep.
        node_attributes: Per-node attributes to filter.

    Returns:
        Dict with:
        - ``positions``: pruned positions
        - ``edges``: pruned edges (remapped indices)
        - ``node_attributes``: filtered attributes
        - ``node_count``, ``edge_count``
        - ``branches_removed``: count of pruned branches
        - ``kept_indices``: original indices of surviving nodes
    """
    n_nodes = len(positions)
    n_edges = len(edges)
    ndim = positions.shape[1]

    if n_nodes == 0 or n_edges == 0:
        return {
            "positions": positions.copy(),
            "edges": edges.copy(),
            "node_attributes": {k: v.copy() for k, v in (node_attributes or {}).items()},
            "node_count": n_nodes,
            "edge_count": n_edges,
            "branches_removed": 0,
            "kept_indices": np.arange(n_nodes, dtype=np.int64),
        }

    # Build adjacency
    adj: dict[int, list[int]] = {i: [] for i in range(n_nodes)}
    for e in edges:
        a, b = int(e[0]), int(e[1])
        adj[a].append(b)
        adj[b].append(a)

    # Find terminal branches
    remove_set: set[int] = set()
    branches_removed = 0

    # Iteratively prune leaves until stable
    changed = True
    while changed:
        changed = False
        degree = {i: len([nb for nb in adj[i] if nb not in remove_set])
                  for i in range(n_nodes) if i not in remove_set}

        leaves = [i for i, d in degree.items() if d == 1]

        for leaf in leaves:
            # Trace back to branch point
            branch_nodes: list[int] = [leaf]
            branch_length = 0.0
            current = leaf

            while True:
                neighbors = [nb for nb in adj[current] if nb not in remove_set and nb not in branch_nodes]
                if not neighbors:
                    break
                next_node = neighbors[0]
                seg_len = float(np.linalg.norm(
                    positions[next_node] - positions[current]
                ))
                branch_length += seg_len
                branch_nodes.append(next_node)

                # Check if next_node is a branch point (degree ≥ 3)
                next_degree = len([nb for nb in adj[next_node]
                                   if nb not in remove_set])
                if next_degree >= 3:
                    break  # reached branch point — don't remove it
                if next_degree == 1 and next_node != leaf:
                    break  # reached another leaf or root
                current = next_node

            # Decide whether to prune (exclude the branch point itself)
            prune_nodes = branch_nodes[:-1] if len(branch_nodes) > 1 else branch_nodes

            should_prune = False
            if min_branch_length > 0 and branch_length < min_branch_length:
                should_prune = True
            if min_branch_vertices > 0 and len(prune_nodes) < min_branch_vertices:
                should_prune = True

            if should_prune and prune_nodes:
                remove_set.update(prune_nodes)
                branches_removed += 1
                changed = True

    # Build kept indices
    kept = sorted(set(range(n_nodes)) - remove_set)
    kept_arr = np.array(kept, dtype=np.int64)
    old_to_new = {old: new for new, old in enumerate(kept)}

    # Filter positions
    new_positions = positions[kept_arr]

    # Filter edges
    new_edges_list: list[list[int]] = []
    for e in edges:
        a, b = int(e[0]), int(e[1])
        if a in old_to_new and b in old_to_new:
            new_edges_list.append([old_to_new[a], old_to_new[b]])

    new_edges = (
        np.array(new_edges_list, dtype=np.int64)
        if new_edges_list
        else np.zeros((0, 2), dtype=np.int64)
    )

    # Filter attributes
    new_attrs: dict[str, npt.NDArray] = {}
    if node_attributes:
        for name, data in node_attributes.items():
            new_attrs[name] = data[kept_arr]

    return {
        "positions": new_positions,
        "edges": new_edges,
        "node_attributes": new_attrs,
        "node_count": len(new_positions),
        "edge_count": len(new_edges),
        "branches_removed": branches_removed,
        "kept_indices": kept_arr,
    }


def _empty_graph_coarsen(ndim: int) -> dict[str, Any]:
    return {
        "positions": np.zeros((0, ndim), dtype=np.float32),
        "edges": np.zeros((0, 2), dtype=np.int64),
        "edge_weights": np.zeros(0, dtype=np.float32),
        "node_attributes": {},
        "children": [],
        "node_count": 0,
        "edge_count": 0,
        "reduction_ratio": 0,
    }
