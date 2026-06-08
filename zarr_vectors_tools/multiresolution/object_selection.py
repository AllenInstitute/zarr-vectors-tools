"""Object selection strategies for multi-resolution sparsity.

Each strategy selects a subset of objects to retain at a coarser
resolution level.  All functions return ``kept_indices`` — an array
of integer indices into the original object list.

Four strategies:

- **Spatial coverage**: greedy selection maximising spatial spread
- **Length**: keep the longest objects (streamlines, skeletons)
- **Attribute**: keep objects with highest/lowest attribute values
- **Random**: uniform random selection (reproducible with seed)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def select_by_spatial_coverage(
    representative_points: npt.NDArray[np.floating],
    bin_shape: tuple[float, ...] | float,
    target_count: int,
) -> npt.NDArray[np.int64]:
    """Select objects that maximise spatial coverage.

    Assigns each object to a spatial bin by its representative point
    (e.g. midpoint for polylines, centroid for meshes).  Distributes
    the ``target_count`` budget across bins proportional to each bin's
    object density, ensuring every occupied bin keeps at least one
    representative.

    Args:
        representative_points: ``(N, D)`` — one point per object.
        bin_shape: Spatial bin edge lengths for coverage assessment.
        target_count: Number of objects to keep.

    Returns:
        ``(target_count,)`` sorted array of kept object indices.

    Raises:
        ValueError: If target_count is invalid.
    """
    n_objects = len(representative_points)
    _validate_target(n_objects, target_count)

    if target_count >= n_objects:
        return np.arange(n_objects, dtype=np.int64)

    ndim = representative_points.shape[1]
    if isinstance(bin_shape, (int, float)):
        bs = np.array([float(bin_shape)] * ndim)
    else:
        bs = np.array(bin_shape, dtype=np.float64)

    # Assign objects to bins
    bin_indices = np.floor(representative_points / bs).astype(np.int64)
    bin_keys = [tuple(row) for row in bin_indices]

    # Group by bin
    bin_to_objects: dict[tuple, list[int]] = {}
    for i, key in enumerate(bin_keys):
        if key not in bin_to_objects:
            bin_to_objects[key] = []
        bin_to_objects[key].append(i)

    n_bins = len(bin_to_objects)

    if target_count <= n_bins:
        # Fewer targets than bins: pick one per bin until budget exhausted
        kept: list[int] = []
        for bin_key, members in bin_to_objects.items():
            if len(kept) >= target_count:
                break
            kept.append(members[0])
        return np.array(sorted(kept[:target_count]), dtype=np.int64)

    # More targets than bins: distribute budget proportionally
    kept = []
    remaining = target_count

    # First pass: at least one per bin
    for bin_key, members in bin_to_objects.items():
        kept.append(members[0])
        remaining -= 1

    # Second pass: distribute remaining proportionally
    if remaining > 0:
        bin_list = list(bin_to_objects.items())
        weights = np.array([len(members) - 1 for _, members in bin_list], dtype=np.float64)
        total_weight = weights.sum()

        if total_weight > 0:
            allocations = np.floor(weights / total_weight * remaining).astype(int)
            # Distribute remainder
            leftover = remaining - allocations.sum()
            top_bins = np.argsort(-weights)[:leftover]
            allocations[top_bins] += 1

            for idx, (bin_key, members) in enumerate(bin_list):
                extra = int(allocations[idx])
                if extra > 0:
                    # Already took members[0], take more from this bin
                    available = members[1 : 1 + extra]
                    kept.extend(available)

    kept = sorted(set(kept))[:target_count]
    return np.array(kept, dtype=np.int64)


def select_by_length(
    lengths: npt.NDArray[np.floating],
    target_count: int,
) -> npt.NDArray[np.int64]:
    """Select objects with the greatest lengths.

    For streamlines, ``length`` is the sum of segment Euclidean
    distances.  For graphs, total edge length.  For any geometry,
    a scalar measure of object size.

    Args:
        lengths: ``(N,)`` array of per-object length/size values.
        target_count: Number of objects to keep.

    Returns:
        ``(target_count,)`` sorted array of kept object indices.
    """
    n_objects = len(lengths)
    _validate_target(n_objects, target_count)

    if target_count >= n_objects:
        return np.arange(n_objects, dtype=np.int64)

    # Argsort descending, take top target_count
    order = np.argsort(-np.asarray(lengths, dtype=np.float64))
    kept = np.sort(order[:target_count])
    return kept.astype(np.int64)


def select_by_attribute(
    attribute_values: npt.NDArray[np.floating],
    target_count: int,
    *,
    mode: str = "max",
) -> npt.NDArray[np.int64]:
    """Select objects with the highest or lowest attribute values.

    Generic selection by any scalar per-object attribute (FA, volume,
    importance score, etc.).

    Args:
        attribute_values: ``(N,)`` array of per-object values.
        target_count: Number of objects to keep.
        mode: ``"max"`` keeps highest values, ``"min"`` keeps lowest.

    Returns:
        ``(target_count,)`` sorted array of kept object indices.

    Raises:
        ValueError: If mode is not ``"max"`` or ``"min"``.
    """
    n_objects = len(attribute_values)
    _validate_target(n_objects, target_count)

    if target_count >= n_objects:
        return np.arange(n_objects, dtype=np.int64)

    vals = np.asarray(attribute_values, dtype=np.float64)

    if mode == "max":
        order = np.argsort(-vals)
    elif mode == "min":
        order = np.argsort(vals)
    else:
        raise ValueError(f"mode must be 'max' or 'min', got '{mode}'")

    kept = np.sort(order[:target_count])
    return kept.astype(np.int64)


def select_point_thinning(
    positions: npt.NDArray[np.floating],
    bin_shape: tuple[float, ...] | float,
    *,
    seed: int | None = None,
) -> npt.NDArray[np.int64]:
    """Spatially-aware point thinning: ≤1 point retained per spatial bin.

    Distinct from :func:`select_by_spatial_coverage`: that function
    distributes a *fixed* ``target_count`` budget proportionally to
    each bin's density; this one *enforces* a max of one survivor per
    bin and the resulting count is determined by the bin layout
    (≤ number of occupied bins).  Useful for point-cloud levels where
    the goal is uniform density, not a target object count.

    Args:
        positions: ``(N, D)`` point positions.
        bin_shape: Spatial bin edge lengths.  Larger bins → fewer
            survivors.
        seed: Optional seed for selecting the survivor when a bin
            contains multiple candidates.  When ``None``, the
            lowest-index candidate is kept (deterministic).

    Returns:
        Sorted array of kept point indices, one per occupied bin.
    """
    n = len(positions)
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    ndim = positions.shape[1]
    if isinstance(bin_shape, (int, float)):
        bs = np.array([float(bin_shape)] * ndim)
    else:
        bs = np.array(bin_shape, dtype=np.float64)
    if np.any(bs <= 0):
        raise ValueError(f"bin_shape components must be > 0, got {bin_shape}")

    # Bin each point.  np.unique over the row-of-int64 bin keys gives
    # one representative per unique bin via ``return_index``.
    bin_keys = np.floor(positions / bs).astype(np.int64)

    if seed is not None:
        # Permute then unique → the permuted order's first hit per bin
        # is the survivor (randomised within bin).
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        permuted = bin_keys[perm]
        _, perm_first = np.unique(permuted, axis=0, return_index=True)
        kept = perm[perm_first]
    else:
        _, first = np.unique(bin_keys, axis=0, return_index=True)
        kept = first

    return np.sort(kept).astype(np.int64)


def select_random(
    n_objects: int,
    target_count: int,
    *,
    seed: int | None = None,
) -> npt.NDArray[np.int64]:
    """Select objects uniformly at random.

    Reproducible with a fixed seed.

    Args:
        n_objects: Total number of objects.
        target_count: Number to keep.
        seed: Random seed for reproducibility.

    Returns:
        ``(target_count,)`` sorted array of kept object indices.
    """
    _validate_target(n_objects, target_count)

    if target_count >= n_objects:
        return np.arange(n_objects, dtype=np.int64)

    rng = np.random.default_rng(seed)
    chosen = rng.choice(n_objects, size=target_count, replace=False)
    return np.sort(chosen).astype(np.int64)


def apply_sparsity(
    n_objects: int,
    sparsity: float,
    strategy: str = "random",
    *,
    seed: int | None = None,
    lengths: npt.NDArray | None = None,
    attribute_values: npt.NDArray | None = None,
    attribute_mode: str = "max",
    representative_points: npt.NDArray | None = None,
    bin_shape: tuple[float, ...] | float | None = None,
) -> npt.NDArray[np.int64]:
    """Convenience wrapper: compute target count from sparsity and dispatch.

    Args:
        n_objects: Total number of objects.
        sparsity: Fraction to keep, in (0, 1].  Ignored by the
            ``"point_thinning"`` strategy, which derives the survivor
            count from the ``bin_shape`` instead.
        strategy: One of ``"random"``, ``"length"``, ``"attribute"``,
            ``"spatial_coverage"``, ``"point_thinning"``.
        seed: Random seed (for ``"random"`` / ``"point_thinning"``).
        lengths: Per-object lengths (for ``"length"`` strategy).
        attribute_values: Per-object values (for ``"attribute"`` strategy).
        attribute_mode: ``"max"`` or ``"min"`` (for ``"attribute"``).
        representative_points: ``(N, D)`` points (for ``"spatial_coverage"``
            and ``"point_thinning"``).
        bin_shape: Bin shape for spatial coverage / point thinning.

    Returns:
        Sorted array of kept object indices.

    Raises:
        ValueError: If required data for the strategy is missing.
    """
    if sparsity >= 1.0:
        return np.arange(n_objects, dtype=np.int64)

    target_count = max(1, round(n_objects * sparsity))

    if strategy == "random":
        return select_random(n_objects, target_count, seed=seed)

    elif strategy == "length":
        if lengths is None:
            raise ValueError("'length' strategy requires 'lengths' array")
        return select_by_length(lengths, target_count)

    elif strategy == "attribute":
        if attribute_values is None:
            raise ValueError("'attribute' strategy requires 'attribute_values' array")
        return select_by_attribute(
            attribute_values, target_count, mode=attribute_mode,
        )

    elif strategy == "spatial_coverage":
        if representative_points is None:
            raise ValueError(
                "'spatial_coverage' strategy requires 'representative_points'"
            )
        if bin_shape is None:
            raise ValueError(
                "'spatial_coverage' strategy requires 'bin_shape'"
            )
        return select_by_spatial_coverage(
            representative_points, bin_shape, target_count,
        )

    elif strategy == "point_thinning":
        if representative_points is None:
            raise ValueError(
                "'point_thinning' strategy requires 'representative_points'"
            )
        if bin_shape is None:
            raise ValueError(
                "'point_thinning' strategy requires 'bin_shape'"
            )
        return select_point_thinning(
            representative_points, bin_shape, seed=seed,
        )

    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Must be one of: "
            f"'random', 'length', 'attribute', 'spatial_coverage', "
            f"'point_thinning'"
        )


# ===================================================================
# Polyline length computation helper
# ===================================================================

def compute_polyline_lengths(
    polylines: list[npt.NDArray[np.floating]],
) -> npt.NDArray[np.float64]:
    """Compute Euclidean path length for each polyline.

    Args:
        polylines: List of ``(N_k, D)`` arrays.

    Returns:
        ``(N,)`` float64 array of path lengths.
    """
    lengths = np.empty(len(polylines), dtype=np.float64)
    for i, poly in enumerate(polylines):
        if len(poly) < 2:
            lengths[i] = 0.0
        else:
            diffs = np.diff(poly, axis=0)
            lengths[i] = float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))))
    return lengths


def compute_representative_points(
    polylines: list[npt.NDArray[np.floating]],
) -> npt.NDArray[np.floating]:
    """Compute the midpoint of each polyline (representative point).

    Args:
        polylines: List of ``(N_k, D)`` arrays.

    Returns:
        ``(N, D)`` array of midpoints.
    """
    ndim = polylines[0].shape[1] if polylines else 3
    points = np.empty((len(polylines), ndim), dtype=np.float64)
    for i, poly in enumerate(polylines):
        points[i] = poly[len(poly) // 2]
    return points


def _validate_target(n_objects: int, target_count: int) -> None:
    """Validate target count."""
    if target_count < 1:
        raise ValueError(f"target_count must be >= 1, got {target_count}")
    if n_objects < 1:
        raise ValueError(f"n_objects must be >= 1, got {n_objects}")
