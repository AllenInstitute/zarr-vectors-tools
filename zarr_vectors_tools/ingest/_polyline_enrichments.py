"""Shared polyline / streamline enrichment helpers.

Internal to zarr_vectors_tools; not part of the public API.
"""

from __future__ import annotations

import numpy as np


def compute_lengths(polylines: list[np.ndarray]) -> np.ndarray:
    """Total path length of each polyline (sum of segment Euclidean distances).

    Args:
        polylines: List of ``(n_k, D)`` arrays — one entry per polyline.

    Returns:
        ``(O,)`` float32 array of path lengths.
    """
    out = np.zeros((len(polylines),), dtype=np.float32)
    for i, p in enumerate(polylines):
        if len(p) < 2:
            continue
        diffs = np.diff(p, axis=0)
        out[i] = float(np.linalg.norm(diffs, axis=1).sum())
    return out


def compute_endpoints(polylines: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """First and last vertex of each polyline.

    Args:
        polylines: List of ``(n_k, D)`` arrays.

    Returns:
        ``(start, end)`` — each is ``(O, D)`` float32.
    """
    if not polylines:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)
    d = polylines[0].shape[1]
    start = np.zeros((len(polylines), d), dtype=np.float32)
    end = np.zeros((len(polylines), d), dtype=np.float32)
    for i, p in enumerate(polylines):
        if len(p) == 0:
            continue
        start[i] = p[0]
        end[i] = p[-1]
    return start, end


def filter_by_length(
    polylines: list[np.ndarray],
    length_range: tuple[float, float],
    *,
    lengths: np.ndarray | None = None,
) -> tuple[list[np.ndarray], np.ndarray, int]:
    """Drop polylines whose total length falls outside ``length_range``.

    Args:
        polylines: List of ``(n_k, D)`` arrays.
        length_range: ``(min, max)`` inclusive bounds, in the same units
            as the input positions.
        lengths: Optional pre-computed lengths array (saves recomputation).

    Returns:
        ``(kept_polylines, kept_indices, dropped_count)``.
    """
    lo, hi = length_range
    if lengths is None:
        lengths = compute_lengths(polylines)
    mask = (lengths >= lo) & (lengths <= hi)
    kept_indices = np.where(mask)[0]
    kept = [polylines[i] for i in kept_indices]
    return kept, kept_indices, int((~mask).sum())
