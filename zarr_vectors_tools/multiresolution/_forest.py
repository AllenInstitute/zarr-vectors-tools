"""Vectorized forest operations for batched skeleton decimation.

The per-fragment skeleton coarsening (:func:`...strategies.skeletons.decimate_skeleton`
+ ``_collapse_to_kept``) runs ~1M times per pyramid level with per-vertex Python
loops — the dominant cost.  A per-call ``scipy`` formulation loses to its own
constant overhead on tiny skeletons (benchmarked ~1.75× slower).  The win is to
**batch a whole chunk's fragments into one forest** (a single global ``parent``
array, ``-1`` for roots) and run these O(N) numpy passes once.

Each function here is a vectorized equivalent of the corresponding per-tree step
and is pinned to the scalar reference by ``tests/test_forest_vectorized.py``.
Index ordering matches the reference (`np.flatnonzero(keep)` ascending), so a
single-tree forest reproduces the scalar output exactly.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt


def child_counts(parent: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Number of children per node, from a ``parent`` array (``-1`` = root)."""
    n = len(parent)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    p = parent[parent >= 0]
    return np.bincount(p, minlength=n).astype(np.int64)


def _nearest_special_ancestor(
    parent: npt.NDArray[np.int64], is_special: npt.NDArray[np.bool_]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """For each node, the nearest at-or-above ancestor that ``is_special``.

    Returns ``(anc, dist)``: ``anc[v]`` is that ancestor's index (``v`` itself
    if special; ``-1`` if no special node up to the root), and ``dist[v]`` the
    number of parent-hops to it.  Uses **path-doubling** (Hillis–Steele): each
    pass at least doubles every node's climb, so it converges in O(log height)
    vectorized passes — vs O(height) for naive relaxation, which is fatal on the
    long unbranched chains of real skeletons (benchmarked ~10× slower).
    """
    n = len(parent)
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    idx = np.arange(n, dtype=np.int64)
    # nxt[v]: current ancestor pointer (self if special, else parent).
    nxt = np.where(is_special, idx, parent)
    dd = np.where(is_special, 0, (parent >= 0).astype(np.int64))
    while True:
        valid = nxt >= 0
        sp = np.zeros(n, dtype=bool)
        sp[valid] = is_special[nxt[valid]]
        active = valid & ~sp
        if not active.any():
            break
        nn = nxt[active]               # current (non-special) pointer
        dd[active] = dd[active] + dd[nn]   # accumulate distance (old values)
        nxt[active] = nxt[nn]              # jump to ancestor's ancestor
    anc = np.where(valid & np.where(valid, is_special[np.clip(nxt, 0, None)], False),
                   nxt, -1)
    return anc, dd


def dist_to_anchor(
    parent: npt.NDArray[np.int64], anchor: npt.NDArray[np.bool_]
) -> npt.NDArray[np.int64]:
    """For each node, hops up to the nearest anchor at-or-above it (anchors → 0).

    Roots are always anchors in the decimator (a root has no parent edge), so
    every node resolves.
    """
    _, dist = _nearest_special_ancestor(parent, anchor)
    return dist


def nearest_kept_at_or_above(
    parent: npt.NDArray[np.int64], keep: npt.NDArray[np.bool_]
) -> npt.NDArray[np.int64]:
    """For each node, the index of the nearest kept node at-or-above it.

    A kept node maps to itself; an unkept node to its nearest kept *strict*
    ancestor (``-1`` if none up to an unkept root — does not occur in the
    decimator, where roots are always kept).
    """
    anc, _ = _nearest_special_ancestor(parent, keep)
    return anc


def decimate_keep(
    parent: npt.NDArray[np.int64],
    *,
    stride: int,
    forced: npt.NDArray[np.bool_] | None = None,
) -> npt.NDArray[np.bool_]:
    """Vectorized equivalent of ``decimate_skeleton``'s keep-mask computation.

    Keeps roots, leaves, branch points, ``forced`` nodes, and — along each
    unbranched chain between anchors — every ``stride``-th node counted from the
    upper anchor.  Mirrors the scalar code exactly, including its
    ``if stride > 1`` guard (``stride <= 1`` keeps anchors only).
    """
    n = len(parent)
    if n == 0:
        return np.zeros(0, dtype=bool)
    cc = child_counts(parent)
    anchor = (parent < 0) | (cc != 1)
    if forced is not None:
        anchor = anchor | forced
    keep = anchor.copy()
    if stride > 1:
        D = dist_to_anchor(parent, anchor)
        keep |= (D % stride == 0)
    return keep


def collapse(
    parent: npt.NDArray[np.int64],
    keep: npt.NDArray[np.bool_],
    attributes: dict[str, npt.NDArray] | None,
    attr_agg: str,
) -> dict:
    """Vectorized equivalent of ``_collapse_to_kept`` over a whole forest.

    Reconnects each survivor to its nearest kept ancestor and aggregates
    per-vertex attributes onto survivors.  ``kept`` is ascending global index
    order (matching the scalar reference), so per-tree output is identical.

    Returns ``{"kept", "edges", "attributes", "new_of_old", "owner_new"}`` where
    ``kept`` are the survivors' global indices, ``edges`` are ``[child, parent]``
    in compact ``0..K-1`` survivor indices, and ``owner_new`` maps every input
    node to its survivor's compact index (``-1`` if dropped to nothing).
    """
    n = len(parent)
    kept = np.flatnonzero(keep)
    K = len(kept)
    new_of_old = np.full(n, -1, dtype=np.int64)
    new_of_old[kept] = np.arange(K, dtype=np.int64)

    ka = nearest_kept_at_or_above(parent, keep)  # kept node at-or-above each v

    # New [child, parent] edges: each kept node with a parent connects to its
    # nearest kept strict ancestor = ka[parent[k]].
    pk = parent[kept]
    has_par = pk >= 0
    par_kept = np.full(K, -1, dtype=np.int64)
    par_kept[has_par] = ka[pk[has_par]]
    valid = has_par & (par_kept >= 0)
    edges = np.stack(
        [new_of_old[kept[valid]], new_of_old[par_kept[valid]]], axis=1
    ) if valid.any() else np.zeros((0, 2), dtype=np.int64)

    # owner_new[v] = compact survivor index v collapses onto (-1 if none).
    owner_new = np.where(ka >= 0, new_of_old[np.clip(ka, 0, None)], -1)

    out_attrs: dict[str, npt.NDArray] = {}
    for name, data in (attributes or {}).items():
        data = np.asarray(data)
        tail = data.shape[1:]
        good = owner_new >= 0
        ow = owner_new[good]
        vals = data[good]
        if attr_agg == "max":
            agg = np.zeros((K, *tail), dtype=data.dtype)
            np.fmax.at(agg, ow, vals)
        elif attr_agg == "min":
            fill = (np.iinfo(data.dtype).max
                    if np.issubdtype(data.dtype, np.integer) else np.inf)
            agg = np.full((K, *tail), fill, dtype=data.dtype)
            np.fmin.at(agg, ow, vals)
        elif attr_agg == "first":
            agg = data[kept]
        else:  # mean
            counts = np.zeros(K, dtype=np.int64)
            acc = np.zeros((K, *tail), dtype=np.float64)
            np.add.at(acc, ow, vals.astype(np.float64))
            np.add.at(counts, ow, 1)
            counts = np.maximum(counts, 1)
            agg = (acc / counts.reshape((K,) + (1,) * len(tail))).astype(data.dtype)
        out_attrs[name] = agg

    return {
        "kept": kept,
        "edges": edges,
        "attributes": out_attrs,
        "new_of_old": new_of_old,
        "owner_new": owner_new,
    }
