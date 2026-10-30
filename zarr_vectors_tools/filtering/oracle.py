"""The brute-force reference: every predicate, tested on every segment.

This is the ground truth the faster engines are measured against -- the
indexed numpy engine here, and the browser engines built later.  It keeps
nothing between calls and prunes nothing: a region leaf runs the exact
segment test of :mod:`~zarr_vectors_tools.filtering.geometry` over all of
the tractogram's segments, a batch at a time.  Slow on purpose, and simple
enough to check by reading.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from .geometry import contains, segment_hits
from .predicates import All, And, Column, InGroup, Node, Not, Or, Passes
from .tractogram import Tractogram

__all__ = ["column_mask", "evaluate", "histogram", "passes", "summarize"]

#: Segments tested per batch.
_BLOCK = 1 << 20


def evaluate(tree: Node, tractogram: Tractogram) -> npt.NDArray[np.bool_]:
    """Which streamlines the filter accepts: an ``(N,)`` bool array by row."""
    if isinstance(tree, All):
        return np.ones(tractogram.n_objects, dtype=bool)
    if isinstance(tree, Passes):
        return passes(tree, tractogram)
    if isinstance(tree, Column):
        return column_mask(tree, tractogram)
    if isinstance(tree, InGroup):
        return tractogram.group_mask(tree.ids)
    if isinstance(tree, And):
        return np.logical_and.reduce([evaluate(c, tractogram) for c in tree.children])
    if isinstance(tree, Or):
        return np.logical_or.reduce([evaluate(c, tractogram) for c in tree.children])
    if isinstance(tree, Not):
        return ~evaluate(tree.child, tractogram)
    raise TypeError(f"cannot evaluate {type(tree).__name__}")


def passes(leaf: Passes, tractogram: Tractogram) -> npt.NDArray[np.bool_]:
    """Rows that meet the leaf's region, by path or by their ends."""
    out = np.zeros(tractogram.n_objects, dtype=bool)
    if leaf.mode == "path":
        start, end, owner = tractogram.edges()
        vertices = tractogram.vertices
        for lo in range(0, len(start), _BLOCK):
            a = vertices[start[lo:lo + _BLOCK]].astype(np.float64)
            b = vertices[end[lo:lo + _BLOCK]].astype(np.float64)
            hit = segment_hits(a, b, leaf.region)
            out[owner[lo:lo + _BLOCK][hit]] = True
        return out
    first, last, has = tractogram.endpoints()
    at_first = contains(first[has], leaf.region)
    at_last = contains(last[has], leaf.region)
    out[has] = at_first | at_last if leaf.mode == "endpoints" else at_first & at_last
    return out


def column_mask(leaf: Column, tractogram: Tractogram) -> npt.NDArray[np.bool_]:
    """Rows whose column value passes the comparison (absent values never do)."""
    values, present = tractogram.column(leaf.name, leaf.channel)
    if leaf.cmp == "absent":
        return ~present
    with np.errstate(invalid="ignore"):
        if leaf.cmp == "<":
            hit = values < leaf.value
        elif leaf.cmp == "<=":
            hit = values <= leaf.value
        elif leaf.cmp == ">":
            hit = values > leaf.value
        elif leaf.cmp == ">=":
            hit = values >= leaf.value
        elif leaf.cmp == "==":
            hit = values == leaf.value
        elif leaf.cmp == "!=":
            hit = values != leaf.value
        elif leaf.cmp == "between":
            hit = (values >= leaf.value[0]) & (values <= leaf.value[1])
        else:  # "in"
            hit = np.isin(values, np.asarray(leaf.value))
    return np.asarray(hit, dtype=bool) & present


def summarize(
    accepted: npt.NDArray[np.bool_],
    tractogram: Tractogram,
    columns: Sequence[str] = ("length",),
) -> dict[str, Any]:
    """Count of accepted rows and, per column, statistics over them.

    Each column reports ``n`` (accepted rows with a value), ``mean``,
    ``std`` (population, ddof 0), ``min`` and ``max``; NaN where ``n`` is 0.
    Multi-channel columns are summarised per channel as ``name[c]``.
    """
    accepted = np.asarray(accepted, dtype=bool)
    out: dict[str, Any] = {"count": int(accepted.sum())}
    for name in columns:
        width = tractogram.columns[name].shape[1] if tractogram.columns[name].ndim == 2 else None
        for channel in [None] if width is None else range(width):
            values, present = tractogram.column(name, channel)
            chosen = values[accepted & present].astype(np.float64)
            label = name if channel is None else f"{name}[{channel}]"
            if len(chosen) == 0:
                out[label] = {"n": 0, "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
            else:
                out[label] = {
                    "n": int(len(chosen)), "mean": float(chosen.mean()),
                    "std": float(chosen.std()), "min": float(chosen.min()),
                    "max": float(chosen.max()),
                }
    return out


def histogram(
    accepted: npt.NDArray[np.bool_],
    tractogram: Tractogram,
    column: str,
    bins: npt.ArrayLike,
    channel: int | None = None,
) -> npt.NDArray[np.int64]:
    """Counts of accepted rows' values in ``bins`` (edges, as numpy.histogram)."""
    values, present = tractogram.column(column, channel)
    counts, _ = np.histogram(values[np.asarray(accepted, bool) & present], bins=np.asarray(bins))
    return counts
