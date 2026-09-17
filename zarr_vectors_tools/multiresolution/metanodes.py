"""Metanode generation for multi-resolution pyramids.

A metanode represents a group of vertices within a coarser spatial bin.
Its position is the centroid of the vertices it contains, and it
carries aggregated attributes (mean, sum, or first value).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def generate_metanodes(
    positions: npt.NDArray[np.floating],
    bin_size: float | tuple[float, ...],
    *,
    attributes: dict[str, npt.NDArray] | None = None,
    agg_mode: str = "mean",
) -> dict[str, Any]:
    """Group vertices into spatial bins and compute metanodes.

    Args:
        positions: ``(N, D)`` vertex positions.
        bin_size: Spatial bin edge length (scalar for isotropic, or per-dim).
        attributes: Optional per-vertex attributes to aggregate.
        agg_mode: Aggregation for attributes: ``"mean"``, ``"sum"``,
            ``"first"``, ``"count"``.

    Returns:
        Dict with:
        - ``metanode_positions``: ``(M, D)`` centroid positions
        - ``metanode_counts``: ``(M,)`` int — vertices per metanode
        - ``children``: list of M arrays, each containing global vertex
          indices belonging to that metanode
        - ``metanode_attributes``: ``{name: (M, ...) array}`` aggregated
        - ``bin_coords``: ``(M, D)`` int — grid coordinates of each bin
    """
    n, ndim = positions.shape

    if isinstance(bin_size, (int, float)):
        bin_sizes = np.array([float(bin_size)] * ndim)
    else:
        bin_sizes = np.array(bin_size, dtype=np.float64)

    # Assign each vertex to a grid bin
    bin_indices = np.floor(positions / bin_sizes).astype(np.int64)

    # Find unique bins
    unique_bins, inverse = np.unique(bin_indices, axis=0, return_inverse=True)
    n_metanodes = len(unique_bins)

    # Centroids, counts and members in three passes over the vertices, not
    # one pass per bin.  ``inverse == m`` inside a loop over the bins is
    # O(bins x vertices): measured 0.34 s at 20k points and 2.09 s at 80k,
    # which is hours at the millions this feeds (coarsen_points_store,
    # coarsen_graph and coarsen_mesh_cluster all come through here).
    inverse = np.asarray(inverse).ravel()
    metanode_counts = np.bincount(inverse, minlength=n_metanodes).astype(np.int64)
    metanode_positions = np.zeros((n_metanodes, ndim), dtype=np.float64)
    for d in range(ndim):
        metanode_positions[:, d] = np.bincount(
            inverse, weights=positions[:, d].astype(np.float64),
            minlength=n_metanodes,
        )
    metanode_positions /= np.maximum(metanode_counts, 1)[:, None]

    # Members per bin: one stable sort, then split at the group boundaries.
    # Ascending within each group, which is what the per-bin flatnonzero
    # produced.
    order = np.argsort(inverse, kind="stable")
    boundaries = np.cumsum(metanode_counts)[:-1]
    children: list[npt.NDArray[np.int64]] = [
        part.astype(np.int64) for part in np.split(order, boundaries)
    ]

    # Aggregate attributes
    metanode_attributes: dict[str, npt.NDArray] = {}
    if attributes:
        for name, data in attributes.items():
            agg = _aggregate(data, inverse, n_metanodes, agg_mode)
            metanode_attributes[name] = agg

    return {
        "metanode_positions": metanode_positions.astype(positions.dtype),
        "metanode_counts": metanode_counts,
        "children": children,
        "metanode_attributes": metanode_attributes,
        "bin_coords": unique_bins,
    }


def _aggregate(
    data: npt.NDArray,
    inverse: npt.NDArray[np.int64],
    n_groups: int,
    mode: str,
) -> npt.NDArray:
    """Aggregate per-vertex data by group."""
    if data.ndim == 1:
        data = data.reshape(-1, 1)
        squeeze = True
    else:
        squeeze = False

    n, c = data.shape
    result = np.zeros((n_groups, c), dtype=np.float64)

    if mode == "mean":
        counts = np.zeros(n_groups, dtype=np.float64)
        for i in range(n):
            g = inverse[i]
            result[g] += data[i]
            counts[g] += 1
        result /= np.maximum(counts.reshape(-1, 1), 1)
    elif mode == "sum":
        for i in range(n):
            result[inverse[i]] += data[i]
    elif mode == "first":
        seen = set()
        for i in range(n):
            g = inverse[i]
            if g not in seen:
                result[g] = data[i]
                seen.add(g)
    elif mode == "count":
        for i in range(n):
            result[inverse[i], 0] += 1
    else:
        raise ValueError(f"Unknown agg_mode: '{mode}'")

    result = result.astype(data.dtype)
    if squeeze:
        result = result.squeeze(axis=1)
    return result
