"""Shared point-cloud enrichment helpers.

Internal to zarr_vectors_tools; not part of the public API.
"""

from __future__ import annotations

import numpy as np

from zarr_vectors.exceptions import IngestError


def compute_knn_distance(positions: np.ndarray, k: int) -> np.ndarray:
    """Mean Euclidean distance from each point to its k nearest neighbours.

    Args:
        positions: ``(N, D)`` array of point positions.
        k: Number of neighbours to consider (excludes the point itself).

    Returns:
        ``(N,)`` float32 array of mean kNN distances.

    Raises:
        IngestError: If scipy is not installed.
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError as e:
        raise IngestError(
            "scipy is required for kNN distance enrichment. "
            "Install with: pip install zarr-vectors-tools[points-enrichment]"
        ) from e

    n = positions.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.float32)

    # Cap k at n-1 so a small store doesn't crash.
    eff_k = min(k, max(n - 1, 1))
    tree = cKDTree(positions)
    # Query k+1 because the point itself is included at distance 0.
    dists, _ = tree.query(positions, k=eff_k + 1)
    # Drop the self-distance column.
    if dists.ndim == 1:
        # k+1 == 1 case (n == 1): all zero
        return np.zeros((n,), dtype=np.float32)
    neighbour_dists = dists[:, 1:]
    return neighbour_dists.mean(axis=1).astype(np.float32)


def compute_per_object_vertex_count(object_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Count vertices per object.

    Args:
        object_ids: ``(N,)`` integer array of per-vertex object IDs.

    Returns:
        ``(unique_ids, counts)`` where both arrays are sorted by id, shape ``(M,)``.
    """
    unique_ids, counts = np.unique(object_ids, return_counts=True)
    return unique_ids.astype(np.int64), counts.astype(np.int64)
