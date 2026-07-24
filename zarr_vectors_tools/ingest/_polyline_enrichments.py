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


# ---------------------------------------------------------------------------
# Synthetic / derived attributes for coloring test data
# ---------------------------------------------------------------------------
#
# These generate colorable per-object and per-vertex attributes from geometry
# alone (TRK files ingested here carry no native scalars).  The array-based
# ``*_from_endpoints`` / per-vertex helpers are what the parallel ingester
# (``trk_parallel.py``) calls on data it already has; the list-based wrappers
# are the whole-file / unit-test entry points.


def orientation_from_endpoints(
    starts: np.ndarray, ends: np.ndarray
) -> np.ndarray:
    """Per-object start→end unit direction (DEC-style RGB), ``(O, D)`` float32.

    Degenerate (start == end) rows are left as the zero vector.
    """
    v = (ends - starts).astype(np.float64)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    out = np.zeros_like(v)
    nz = norms[:, 0] > 1e-12
    out[nz] = v[nz] / norms[nz]
    return out.astype(np.float32)


def tortuosity_from_endpoints(
    starts: np.ndarray, ends: np.ndarray, lengths: np.ndarray
) -> np.ndarray:
    """Per-object tortuosity = path length ÷ straight-line endpoint distance.

    ``(O,)`` float32, ≥ 1 for a real path.  Rows whose endpoints coincide
    (a closed loop or a single point) have no meaningful ratio and are set
    to 1.0.
    """
    d = np.linalg.norm((ends - starts).astype(np.float64), axis=1)
    out = np.ones((len(d),), dtype=np.float32)
    nz = d > 1e-12
    out[nz] = (np.asarray(lengths, dtype=np.float64)[nz] / d[nz]).astype(np.float32)
    return out


def compute_orientation(polylines: list[np.ndarray]) -> np.ndarray:
    """Per-object start→end unit direction, ``(O, D)`` float32."""
    starts, ends = compute_endpoints(polylines)
    return orientation_from_endpoints(starts, ends)


def compute_tortuosity(
    polylines: list[np.ndarray], *, lengths: np.ndarray | None = None
) -> np.ndarray:
    """Per-object tortuosity, ``(O,)`` float32 (see tortuosity_from_endpoints)."""
    if lengths is None:
        lengths = compute_lengths(polylines)
    starts, ends = compute_endpoints(polylines)
    return tortuosity_from_endpoints(starts, ends, lengths)


def compute_vertex_counts(polylines: list[np.ndarray]) -> np.ndarray:
    """Per-object vertex count, ``(O,)`` uint32."""
    return np.array([len(p) for p in polylines], dtype=np.uint32)


def arc_length_normalized(vertices: np.ndarray) -> np.ndarray:
    """Per-vertex cumulative arc length along one polyline, normalized 0→1.

    ``(N,)`` float32: 0 at the first vertex, 1 at the last.  A zero-length
    (single-point or coincident) polyline returns all-zeros.
    """
    n = len(vertices)
    out = np.zeros((n,), dtype=np.float32)
    if n < 2:
        return out
    seg = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total > 0:
        out[:] = (cum / total).astype(np.float32)
    return out


def index_normalized(n: int) -> np.ndarray:
    """Per-vertex ordinal within one polyline, normalized 0→1. ``(N,)`` float32."""
    if n <= 1:
        return np.zeros((max(n, 0),), dtype=np.float32)
    return (np.arange(n, dtype=np.float32) / np.float32(n - 1))


def compute_tangents(vertices: np.ndarray) -> np.ndarray:
    """Per-vertex local unit tangent (central differences), ``(N, D)`` float32.

    Endpoints use a one-sided difference.  Degenerate (coincident-neighbor)
    rows are left as the zero vector.
    """
    n = len(vertices)
    d = vertices.shape[1] if vertices.ndim == 2 else 3
    out = np.zeros((n, d), dtype=np.float32)
    if n < 2:
        return out
    diff = np.zeros((n, d), dtype=np.float64)
    diff[1:-1] = vertices[2:].astype(np.float64) - vertices[:-2].astype(np.float64)
    diff[0] = vertices[1].astype(np.float64) - vertices[0].astype(np.float64)
    diff[-1] = vertices[-1].astype(np.float64) - vertices[-2].astype(np.float64)
    norms = np.linalg.norm(diff, axis=1, keepdims=True)
    nz = norms[:, 0] > 1e-12
    out[nz] = (diff[nz] / norms[nz]).astype(np.float32)
    return out


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
