"""Exact segment and point tests against each region type.

Every function here is vectorised over a batch of segments ``a[i] -> b[i]``
(each ``(E, 3)`` float64) or points, and implements the meaning
:mod:`~zarr_vectors_tools.filtering.predicates` defines for the region.  The
oracle and the indexed engine both call these, so they differ only in which
segments they hand over: all of them, or the ones a spatial index says can
matter.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from .predicates import Box, Disk, Mask, Region, Slab, Sphere

__all__ = ["contains", "region_bounds", "segment_hits"]

Array = npt.NDArray[np.float64]


def segment_hits(a: Array, b: Array, region: Region) -> npt.NDArray[np.bool_]:
    """Whether each segment ``a[i] -> b[i]`` meets ``region``."""
    if isinstance(region, Sphere):
        return _sphere(a, b, np.asarray(region.center), region.radius)
    if isinstance(region, Box):
        return _box(a, b, np.asarray(region.min), np.asarray(region.max))
    if isinstance(region, Slab):
        return _slab(a, b, region.axis, region.min, region.max)
    if isinstance(region, Disk):
        return _disk(a, b, region)
    if isinstance(region, Mask):
        return _mask(a, b, region)
    raise TypeError(f"no segment test for {type(region).__name__}")


def contains(points: Array, region: Region) -> npt.NDArray[np.bool_]:
    """Whether each point lies in ``region`` (for the endpoint modes)."""
    if isinstance(region, Sphere):
        offset = points - np.asarray(region.center)
        return np.einsum("ij,ij->i", offset, offset) <= region.radius ** 2
    if isinstance(region, Box):
        lo, hi = np.asarray(region.min), np.asarray(region.max)
        return np.all((points >= lo) & (points <= hi), axis=1)
    if isinstance(region, Slab):
        coord = points[:, region.axis]
        return (coord >= region.min) & (coord <= region.max)
    if isinstance(region, Mask):
        return region.contains(points)
    raise TypeError(f"{type(region).__name__} has no inside to hold a point")


def region_bounds(region: Region) -> tuple[Array, Array]:
    """World box that every segment meeting ``region`` must reach into.

    Unbounded axes (a slab's other two) are infinite.  An empty mask gives
    an empty box (min above max).
    """
    if isinstance(region, Sphere):
        c = np.asarray(region.center)
        return c - region.radius, c + region.radius
    if isinstance(region, Box):
        return np.asarray(region.min), np.asarray(region.max)
    if isinstance(region, Slab):
        lo, hi = np.full(3, -np.inf), np.full(3, np.inf)
        lo[region.axis], hi[region.axis] = region.min, region.max
        return lo, hi
    if isinstance(region, Disk):
        # A disk of radius r with unit normal n spans r * sqrt(1 - n_k^2)
        # along axis k.
        c, n = np.asarray(region.center), np.asarray(region.normal)
        half = region.radius * np.sqrt(np.clip(1.0 - n * n, 0.0, 1.0))
        return c - half, c + half
    if isinstance(region, Mask):
        return region.world_bounds()
    raise TypeError(f"no bounds for {type(region).__name__}")


def _sphere(a: Array, b: Array, center: Array, radius: float) -> npt.NDArray[np.bool_]:
    d = b - a
    dd = np.einsum("ij,ij->i", d, d)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.einsum("ij,ij->i", center - a, d) / dd
    t = np.where(dd > 0, np.clip(t, 0.0, 1.0), 0.0)
    nearest = a + t[:, None] * d - center
    return np.einsum("ij,ij->i", nearest, nearest) <= radius * radius


def _box(a: Array, b: Array, lo: Array, hi: Array) -> npt.NDArray[np.bool_]:
    # Slab method over the parameter t in [0, 1].  An axis the segment does
    # not move along admits every t or none, by whether it is inside.
    d = b - a
    moving = d != 0
    with np.errstate(invalid="ignore", divide="ignore"):
        t1 = (lo - a) / d
        t2 = (hi - a) / d
    inside = (a >= lo) & (a <= hi)
    enter = np.where(moving, np.minimum(t1, t2), np.where(inside, -np.inf, np.inf))
    leave = np.where(moving, np.maximum(t1, t2), np.where(inside, np.inf, -np.inf))
    t_in = np.maximum(enter.max(axis=1), 0.0)
    t_out = np.minimum(leave.min(axis=1), 1.0)
    return t_in <= t_out


def _slab(a: Array, b: Array, axis: int, lo: float, hi: float) -> npt.NDArray[np.bool_]:
    ca, cb = a[:, axis], b[:, axis]
    return (np.minimum(ca, cb) <= hi) & (np.maximum(ca, cb) >= lo)


def _disk(a: Array, b: Array, disk: Disk) -> npt.NDArray[np.bool_]:
    c, n = np.asarray(disk.center), np.asarray(disk.normal)
    d0 = (a - c) @ n
    d1 = (b - c) @ n
    crosses = (((d0 <= 0) & (d1 >= 0)) | ((d0 >= 0) & (d1 <= 0))) & ~((d0 == 0) & (d1 == 0))
    out = np.zeros(len(a), dtype=bool)
    if not crosses.any():
        return out
    a, b, d0, d1 = a[crosses], b[crosses], d0[crosses], d1[crosses]
    d = b - a
    point = a + (d0 / (d0 - d1))[:, None] * d - c
    within = np.einsum("ij,ij->i", point, point) <= disk.radius ** 2
    # |cos| of the angle between the segment and the normal axis.  A
    # crossing segment is never zero-length: both ends would be on the plane.
    cos = np.clip(np.abs(d @ n) / np.linalg.norm(d, axis=1), 0.0, 1.0)
    lo, hi = disk.angle
    aligned = (cos <= _cos_deg(lo)) & (cos >= _cos_deg(hi))
    out[np.flatnonzero(crosses)] = within & aligned
    return out


def _cos_deg(angle: float) -> float:
    # Exact at the ends, so (0, 90) accepts every crossing.
    if angle <= 0:
        return 1.0
    if angle >= 90:
        return 0.0
    return math.cos(math.radians(angle))


#: Points sampled per batch in the mask test, to bound the temporaries.
_MASK_BATCH = 1 << 21


def _mask(a: Array, b: Array, mask: Mask) -> npt.NDArray[np.bool_]:
    spacing = mask.sample_spacing
    d = b - a
    length = np.linalg.norm(d, axis=1)
    # Interior points per segment: ceil(length / spacing) - 1, as
    # select_streamlines samples.  The two vertices are tested as they are,
    # not recomputed as a + 1 * d.
    interior = np.maximum(np.ceil(length / spacing).astype(np.int64) - 1, 0)
    hit = mask.contains(a) | mask.contains(b)
    todo = np.flatnonzero(~hit & (interior > 0))
    if len(todo) == 0:
        return hit
    counts = interior[todo].astype(np.intp)
    ends = np.cumsum(counts)
    start = 0
    while start < len(todo):
        # A batch of whole segments holding about _MASK_BATCH points.
        stop = max(int(np.searchsorted(ends, (ends[start - 1] if start else 0) + _MASK_BATCH)),
                   start + 1)
        stop = min(stop, len(todo))
        rows, n = todo[start:stop], counts[start:stop]
        which = np.repeat(np.arange(len(rows)), n)
        offsets = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
        fraction = (offsets + 1) / (n[which] + 1)
        points = a[rows][which] + d[rows][which] * fraction[:, None]
        inside = mask.contains(points)
        hit[rows[np.unique(which[inside])]] = True
        start = stop
    return hit
