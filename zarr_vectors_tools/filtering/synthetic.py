"""Synthetic tractograms for tests and benchmarks.

:func:`whole_brain_like` makes smooth, randomly curving streamlines with a
whole-brain's spread of lengths, filling an ellipsoid of brain size.  It is
not anatomy -- there are no bundles to speak of -- but it has what filtering
cost depends on: the number of streamlines, the vertex spacing, how long
they run and how densely they pack space.  Generation is vectorised and
seeded, and uses numpy alone, so the same call builds the same set under
CPython and under Pyodide.
"""

from __future__ import annotations

import numpy as np

from .tractogram import Tractogram

__all__ = ["whole_brain_like"]

#: Streamlines generated per batch, to bound the temporaries.
_BATCH = 50_000


def whole_brain_like(
    n: int,
    *,
    seed: int = 0,
    step: float = 1.0,
    mean_length: float = 60.0,
    length_range: tuple[float, float] = (20.0, 200.0),
    semi_axes: tuple[float, float, float] = (70.0, 85.0, 60.0),
    curvature: float = 0.08,
    n_groups: int = 8,
    absent_fraction: float = 0.01,
) -> Tractogram:
    """``n`` streamlines in float32, with columns and groups to filter on.

    Every vertex lies inside the ellipsoid ``semi_axes``: a streamline that
    reaches its wall is reflected back in.

    Args:
        n: Number of streamlines.
        seed: Seed for numpy's PCG64 generator.
        step: Distance between consecutive vertices, in millimetres.
        mean_length: Mean streamline length before clipping to
            ``length_range`` (lengths are log-normal).
        length_range: Shortest and longest streamline.
        semi_axes: The ellipsoid the streamlines fill, centred on the origin.
        curvature: How quickly the direction wanders, per step.
        n_groups: Groups to assign rows to, round-robin by start octant and
            row; 0 for none.
        absent_fraction: Share of rows whose ``fa_mean`` is absent (NaN).

    Columns:
        ``length`` and ``n_vertices`` (computed), ``fa_mean`` (float32 in
        0.15..0.75, some absent) and ``label`` (int32, the group).
    """
    rng = np.random.default_rng(seed)
    sigma = np.log(1.0 + 0.6 ** 2) ** 0.5
    lengths = rng.lognormal(np.log(mean_length) - sigma ** 2 / 2, sigma, n)
    lengths = np.clip(lengths, *length_range)
    counts = np.maximum(np.rint(lengths / step).astype(np.int64), 1) + 1

    axes = np.asarray(semi_axes, dtype=np.float64)
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    # Uniform in the ellipsoid shrunk to 80 %, so most of each streamline
    # stays inside the brain-sized volume.
    starts = rng.normal(size=(n, 3))
    starts /= np.linalg.norm(starts, axis=1, keepdims=True)
    starts *= rng.random((n, 1)) ** (1 / 3) * 0.8 * axes

    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    vertices = np.empty((int(offsets[-1]), 3), dtype=np.float32)
    for lo in range(0, n, _BATCH):
        hi = min(lo + _BATCH, n)
        c = counts[lo:hi].astype(np.intp)  # intp: 32-bit under Pyodide
        rows = np.repeat(np.arange(hi - lo), c)
        first = np.cumsum(c) - c
        # The heading drifts by a running sum of small kicks, restarted at
        # each streamline, then the positions are a running sum of steps.
        kick = rng.normal(scale=curvature, size=(int(c.sum()), 3))
        drift = _restart_cumsum(kick, first, c)
        heading = direction[lo:hi][rows] + drift
        heading /= np.linalg.norm(heading, axis=1, keepdims=True)
        heading[first] = 0.0  # the first vertex is the start itself
        path = _restart_cumsum(heading * step, first, c)
        vertices[offsets[lo]:offsets[hi]] = _fold(starts[lo:hi][rows] + path, axes)

    octant = (starts > 0).astype(np.int64) @ np.array([1, 2, 4])
    if n_groups:
        label = ((octant + np.arange(n)) % n_groups).astype(np.int32)
    else:
        label = np.zeros(n, np.int32)
    fa = rng.uniform(0.15, 0.75, n).astype(np.float32)
    fa[rng.random(n) < absent_fraction] = np.nan
    groups = [np.flatnonzero(label == g) for g in range(n_groups)]
    return Tractogram(
        vertices, offsets, columns={"fa_mean": fa, "label": label}, groups=groups,
    )


def _fold(points: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """Reflect points back into the ellipsoid, as a ray bounces off its wall.

    In coordinates scaled to the unit ball, a radius r becomes a triangle
    wave of r with period 2, so a streamline leaving the ellipsoid turns back
    in, stays continuous, and the set keeps a brain's density.
    """
    unit = points / axes
    r = np.linalg.norm(unit, axis=1, keepdims=True)
    folded = np.abs((r + 1.0) % 2.0 - 1.0)
    scale = np.divide(folded, r, out=np.ones_like(r), where=r > 0)
    return unit * scale * axes


def _restart_cumsum(values: np.ndarray, first: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Cumulative sum along axis 0 that restarts at each ``first`` row."""
    total = np.cumsum(values, axis=0)
    before = np.zeros((len(first), values.shape[1]), dtype=total.dtype)
    before[1:] = total[first[1:] - 1]
    return total - np.repeat(before, counts, axis=0)
