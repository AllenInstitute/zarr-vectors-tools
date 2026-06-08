"""Polyline and streamline coarsening strategies.

Two complementary approaches:

1. **Douglas-Peucker simplification**: reduces vertex count within each
   polyline while preserving shape.  Good for reducing per-streamline
   resolution while keeping all streamlines.

2. **Spatial subsampling**: selects a representative subset of
   streamlines per spatial bin.  Good for reducing total streamline
   count while keeping full vertex resolution on survivors.

Both can be composed: simplify first, then subsample.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def simplify_polyline(
    vertices: npt.NDArray[np.floating],
    epsilon: float,
) -> npt.NDArray[np.floating]:
    """Simplify a polyline using Douglas-Peucker algorithm.

    Args:
        vertices: ``(N, D)`` ordered vertex positions.
        epsilon: Maximum perpendicular distance threshold.  Larger
            values produce more aggressive simplification.

    Returns:
        ``(M, D)`` simplified polyline with M ≤ N vertices.
        First and last vertices are always preserved.
    """
    n = len(vertices)
    if n <= 2:
        return vertices.copy()

    keep = np.zeros(n, dtype=bool)
    keep[0] = True
    keep[-1] = True

    _dp_recurse(vertices, 0, n - 1, epsilon, keep)

    return vertices[keep].copy()


def _dp_recurse(
    vertices: npt.NDArray,
    start: int,
    end: int,
    epsilon: float,
    keep: npt.NDArray[np.bool_],
) -> None:
    """Recursive Douglas-Peucker step."""
    if end - start <= 1:
        return

    # Line segment from vertices[start] to vertices[end]
    line_start = vertices[start]
    line_end = vertices[end]
    line_vec = line_end - line_start
    line_len_sq = np.dot(line_vec, line_vec)

    if line_len_sq < 1e-30:
        # Degenerate segment — keep the farthest point
        dists = np.sum((vertices[start + 1 : end] - line_start) ** 2, axis=1)
        max_idx = start + 1 + np.argmax(dists)
        if np.sqrt(dists[max_idx - start - 1]) > epsilon:
            keep[max_idx] = True
            _dp_recurse(vertices, start, max_idx, epsilon, keep)
            _dp_recurse(vertices, max_idx, end, epsilon, keep)
        return

    # Perpendicular distances from each interior point to the line
    points = vertices[start + 1 : end]
    t = np.clip(
        np.dot(points - line_start, line_vec) / line_len_sq,
        0.0, 1.0,
    )
    projections = line_start + t[:, np.newaxis] * line_vec
    dists = np.sqrt(np.sum((points - projections) ** 2, axis=1))

    max_local = np.argmax(dists)
    max_dist = dists[max_local]
    max_idx = start + 1 + max_local

    if max_dist > epsilon:
        keep[max_idx] = True
        _dp_recurse(vertices, start, max_idx, epsilon, keep)
        _dp_recurse(vertices, max_idx, end, epsilon, keep)


def simplify_polylines(
    polylines: list[npt.NDArray[np.floating]],
    epsilon: float,
    *,
    min_vertices: int = 2,
) -> list[npt.NDArray[np.floating]]:
    """Simplify a list of polylines using Douglas-Peucker.

    Args:
        polylines: List of ``(N_k, D)`` arrays.
        epsilon: Distance threshold.
        min_vertices: Minimum vertices to keep per polyline.

    Returns:
        List of simplified polylines.
    """
    result: list[npt.NDArray] = []
    for poly in polylines:
        simplified = simplify_polyline(poly, epsilon)
        if len(simplified) < min_vertices:
            # Keep first and last at minimum
            if len(poly) >= min_vertices:
                indices = np.linspace(0, len(poly) - 1, min_vertices, dtype=int)
                simplified = poly[indices]
            else:
                simplified = poly.copy()
        result.append(simplified)
    return result


def subsample_polylines(
    polylines: list[npt.NDArray[np.floating]],
    bin_size: float | tuple[float, ...],
    *,
    max_per_bin: int = 1,
    selection: str = "longest",
) -> dict[str, Any]:
    """Spatially subsample polylines, keeping representatives per bin.

    Assigns each polyline to the spatial bin containing its midpoint,
    then selects up to ``max_per_bin`` representatives per bin.

    Args:
        polylines: List of ``(N_k, D)`` arrays.
        bin_size: Spatial bin edge length.
        max_per_bin: How many polylines to keep per bin.
        selection: How to pick representatives:
            ``"longest"``: keep the longest polyline(s).
            ``"random"``: keep random polyline(s).
            ``"first"``: keep the first polyline(s) by index.

    Returns:
        Dict with:
        - ``polylines``: subsampled list of arrays
        - ``indices``: original indices of kept polylines
        - ``polyline_count``: number kept
        - ``reduction_ratio``: original / kept
    """
    n_total = len(polylines)
    if n_total == 0:
        return {
            "polylines": [],
            "indices": np.array([], dtype=np.int64),
            "polyline_count": 0,
            "reduction_ratio": 0,
        }

    ndim = polylines[0].shape[1]

    if isinstance(bin_size, (int, float)):
        bin_sizes = np.array([float(bin_size)] * ndim)
    else:
        bin_sizes = np.array(bin_size, dtype=np.float64)

    # Compute midpoint for each polyline
    midpoints = np.array([
        poly[len(poly) // 2] for poly in polylines
    ], dtype=np.float64)

    # Assign to bins
    bin_indices = np.floor(midpoints / bin_sizes).astype(np.int64)

    # Group by bin
    bin_to_polys: dict[tuple, list[int]] = {}
    for i in range(n_total):
        key = tuple(bin_indices[i].tolist())
        if key not in bin_to_polys:
            bin_to_polys[key] = []
        bin_to_polys[key].append(i)

    # Select representatives
    kept_indices: list[int] = []
    for bin_key, members in bin_to_polys.items():
        if len(members) <= max_per_bin:
            kept_indices.extend(members)
            continue

        if selection == "longest":
            lengths = [len(polylines[m]) for m in members]
            sorted_members = [m for _, m in sorted(
                zip(lengths, members), reverse=True
            )]
            kept_indices.extend(sorted_members[:max_per_bin])
        elif selection == "random":
            rng = np.random.default_rng()
            chosen = rng.choice(members, size=max_per_bin, replace=False)
            kept_indices.extend(chosen.tolist())
        elif selection == "first":
            kept_indices.extend(members[:max_per_bin])
        else:
            kept_indices.extend(members[:max_per_bin])

    kept_indices.sort()
    kept_polys = [polylines[i] for i in kept_indices]

    return {
        "polylines": kept_polys,
        "indices": np.array(kept_indices, dtype=np.int64),
        "polyline_count": len(kept_polys),
        "reduction_ratio": n_total / max(len(kept_polys), 1),
    }


def coarsen_polylines(
    polylines: list[npt.NDArray[np.floating]],
    *,
    simplify_epsilon: float | None = None,
    subsample_bin_size: float | None = None,
    max_per_bin: int = 1,
    min_vertices: int = 2,
    selection: str = "longest",
) -> dict[str, Any]:
    """Combined polyline coarsening: simplify then subsample.

    Args:
        polylines: Input polylines.
        simplify_epsilon: Douglas-Peucker epsilon.  None to skip.
        subsample_bin_size: Spatial subsampling bin size.  None to skip.
        max_per_bin: Polylines to keep per bin (for subsampling).
        min_vertices: Minimum vertices per polyline (for simplification).
        selection: Subsampling selection mode.

    Returns:
        Dict with:
        - ``polylines``: coarsened polylines
        - ``vertex_count``: total vertices
        - ``polyline_count``: number of polylines
        - ``simplification_ratio``: vertex reduction from DP
        - ``subsampling_ratio``: polyline reduction from subsampling
    """
    n_input = len(polylines)
    v_input = sum(len(p) for p in polylines)
    current = polylines

    simplification_ratio = 1.0
    subsampling_ratio = 1.0

    # Step 1: simplify
    if simplify_epsilon is not None:
        current = simplify_polylines(
            current, simplify_epsilon, min_vertices=min_vertices,
        )
        v_after = sum(len(p) for p in current)
        simplification_ratio = v_input / max(v_after, 1)

    # Step 2: subsample
    kept_indices: npt.NDArray | None = None
    if subsample_bin_size is not None:
        sub_result = subsample_polylines(
            current, subsample_bin_size,
            max_per_bin=max_per_bin,
            selection=selection,
        )
        current = sub_result["polylines"]
        kept_indices = sub_result["indices"]
        subsampling_ratio = sub_result["reduction_ratio"]

    return {
        "polylines": current,
        "vertex_count": sum(len(p) for p in current),
        "polyline_count": len(current),
        "simplification_ratio": simplification_ratio,
        "subsampling_ratio": subsampling_ratio,
        "kept_indices": kept_indices,
    }
