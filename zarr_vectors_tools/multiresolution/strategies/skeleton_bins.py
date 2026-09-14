"""Skeleton coarsening by connected-bin metavertex contraction.

Where :mod:`~zarr_vectors_tools.multiresolution.strategies.skeletons` thins a
skeleton by *index* (keep every k-th vertex along a chain), this strategy thins
it by *space*: vertices sharing a spatial bin collapse onto one metavertex at
their centroid.  Two properties follow, and they are the whole point:

**Connected components are preserved exactly.**  Edges are contracted only when
both endpoints share a bin AND are joined in the graph — the union-find runs
over edges, never over bin membership alone.  Every cluster is therefore a
*connected* subgraph, clusters are vertex-disjoint, and contracting
vertex-disjoint connected subgraphs cannot change the number of connected
components.  This holds independently of geometry, bin size, chunk size, or how
the work is distributed.  Binning by proximity alone would instead weld two
branches that merely pass close to each other, inventing a cycle.

**Geometric error is bounded by the bin diagonal.**  No vertex can be further
from its metavertex than the diagonal of the bin containing it, because the
centroid of a point set lies inside that set's bounding box.  Index-stride
decimation offers no such bound at any ratio: it never consults geometry, so a
hairpin short in vertex count is cut straight across.

Coarsening is *density-adaptive* as an emergent property of a fixed lattice: a
dense arbor puts many vertices in one bin and collapses hard, while a sparse
axon puts one or two in each and barely changes.

Edge orientation is inherited rather than recomputed.  Callers pass a rooted
``[child, parent]`` tree (as :func:`split_components` produces); a cluster is a
connected subtree, so exactly one of its members has a parent outside it, and
mapping the edges through ``owner_new`` yields a correctly-oriented forest with
``K - n_roots`` rows.  No re-rooting pass is needed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "contract_skeleton_bins",
    "default_attr_agg",
    "heavy_path_relabel",
]

#: Aggregations that make sense per attribute kind.  ``radius`` averages;
#: a categorical code must be taken from a real member, never averaged or
#: maximised (the incumbent's ``attr_agg="max"`` default would bias calibre
#: systematically thick and make a compartment label meaningless).
_CONTINUOUS_DEFAULT = "mean"
_CATEGORICAL_DEFAULT = "nearest"
_KNOWN_CONTINUOUS = frozenset({"radius", "cross_sectional_area"})


def default_attr_agg(
    attr_dtypes: Mapping[str, Any],
) -> dict[str, str]:
    """Per-attribute aggregation modes, inferred from name then dtype.

    Named continuous quantities and floating dtypes average; everything else
    (integer codes such as ``compartment``) takes the value of the member
    nearest the metavertex, so the result is always a value that genuinely
    occurred in the data.
    """
    modes: dict[str, str] = {}
    for name, dt in attr_dtypes.items():
        if name in _KNOWN_CONTINUOUS:
            modes[name] = _CONTINUOUS_DEFAULT
            continue
        modes[name] = (
            _CONTINUOUS_DEFAULT
            if np.issubdtype(np.dtype(dt), np.floating)
            else _CATEGORICAL_DEFAULT
        )
    return modes


def _resolve_modes(
    attributes: Mapping[str, npt.NDArray] | None,
    attr_agg: str | Mapping[str, str] | None,
) -> dict[str, str]:
    if not attributes:
        return {}
    if attr_agg is None:
        return default_attr_agg({k: v.dtype for k, v in attributes.items()})
    if isinstance(attr_agg, str):
        return {k: attr_agg for k in attributes}
    inferred = default_attr_agg({k: v.dtype for k, v in attributes.items()})
    return {k: str(attr_agg.get(k, inferred[k])) for k in attributes}


def _bin_keys(
    positions: npt.NDArray[np.floating], bin_shape: npt.NDArray[np.float64]
) -> npt.NDArray[np.int64]:
    """Lattice cell index per vertex, anchored at coordinate 0.

    Anchored at 0 rather than at the data's own minimum so that the lattice is
    a property of the coordinate frame, not of which subset of vertices this
    call happens to see.  Two workers coarsening adjacent chunks therefore
    agree on where the bin edges fall.
    """
    return np.floor(np.asarray(positions, dtype=np.float64) / bin_shape).astype(
        np.int64
    )


def _cluster_owners(
    n: int,
    edges: npt.NDArray[np.integer],
    keys: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """Map each vertex to its cluster, contracting only intra-bin edges.

    Union-find with path halving.  Deliberately iterates edges rather than
    grouping by bin: an edge is the only evidence that two co-binned vertices
    are actually connected, and clustering without it would merge branches that
    merely pass nearby.
    """
    parent = np.arange(n, dtype=np.int64)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    if len(edges):
        same_bin = np.all(keys[edges[:, 0]] == keys[edges[:, 1]], axis=1)
        for a, b in edges[same_bin]:
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[ra] = rb

    roots = np.fromiter((find(i) for i in range(n)), dtype=np.int64, count=n)
    _, owner = np.unique(roots, return_inverse=True)
    return owner.astype(np.int64)


def _aggregate(
    data: npt.NDArray,
    owner: npt.NDArray[np.int64],
    k: int,
    mode: str,
    *,
    order_within_cluster: npt.NDArray[np.int64],
) -> npt.NDArray:
    """Collapse per-vertex values onto their clusters.

    ``order_within_cluster`` ranks each vertex within its cluster (0 = the
    representative), so "nearest" and "first" are a single gather.
    """
    flat = data.reshape(len(data), -1)
    out = np.empty((k, flat.shape[1]), dtype=data.dtype)

    if mode == "mean":
        acc = np.zeros((k, flat.shape[1]), dtype=np.float64)
        np.add.at(acc, owner, flat.astype(np.float64))
        counts = np.bincount(owner, minlength=k).astype(np.float64)[:, None]
        avg = acc / counts
        # Integer attributes round rather than truncate; a truncating mean of
        # e.g. radii would be biased low at every level.
        out[:] = (
            np.rint(avg).astype(data.dtype)
            if np.issubdtype(data.dtype, np.integer)
            else avg.astype(data.dtype)
        )
    elif mode in ("nearest", "first"):
        rep = order_within_cluster == 0
        out[owner[rep]] = flat[rep]
    elif mode in ("max", "min"):
        # Seed from a real member rather than from zeros, so negative-valued
        # attributes are not silently clamped at 0.
        rep = order_within_cluster == 0
        out[owner[rep]] = flat[rep]
        ufunc = np.fmax if mode == "max" else np.fmin
        ufunc.at(out, owner, flat)
    else:
        raise ValueError(
            f"unknown attribute aggregation {mode!r}; expected one of "
            "'mean', 'nearest', 'first', 'max', 'min'"
        )
    return out.reshape((k,) + data.shape[1:])


def contract_skeleton_bins(
    positions: npt.NDArray[np.floating],
    edges: npt.NDArray[np.integer],
    *,
    bin_shape: float | Sequence[float],
    attributes: Mapping[str, npt.NDArray] | None = None,
    attr_agg: str | Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Contract a rooted skeleton onto a spatial lattice.

    Args:
        positions: ``(N, D)`` vertex positions.
        edges: ``(M, 2)`` ``[child, parent]`` rows, as stored for the skeleton
            convention.  Orientation is preserved through the contraction.
        bin_shape: Lattice cell size, scalar or per-axis, in position units.
            ``<= 0`` is the identity (no contraction).
        attributes: Optional ``{name: (N,) or (N, C)}`` per-vertex values.
        attr_agg: Aggregation mode, one for all attributes or per name.  When
            omitted, :func:`default_attr_agg` picks per name and dtype.

    Returns:
        The :func:`decimate_skeleton` result shape — ``positions`` ``(K, D)``,
        ``edges`` ``(<=K-1, 2)`` in new indices, ``attributes``,
        ``kept_source_indices`` ``(K,)`` — plus ``owner_new`` ``(N,)`` mapping
        *every* source vertex to its metavertex.  ``owner_new`` is total, which
        is what lets a cross-chunk link endpoint always resolve after
        coarsening; index decimation has no equivalent, which is why it needs
        to force-keep boundary vertices instead.

        Note ``positions[i]`` is the centroid of cluster ``i``, not the
        position of ``kept_source_indices[i]``; the latter names a
        representative member (lowest source index) for provenance only.
    """
    positions = np.asarray(positions)
    n = len(positions)
    ndim = positions.shape[1] if positions.ndim == 2 else 0
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    attributes = dict(attributes or {})
    modes = _resolve_modes(attributes, attr_agg)

    if n == 0:
        return {
            "positions": positions.reshape(0, ndim),
            "edges": np.zeros((0, 2), dtype=np.int64),
            "attributes": {k: v[:0] for k, v in attributes.items()},
            "kept_source_indices": np.zeros(0, dtype=np.int64),
            "owner_new": np.zeros(0, dtype=np.int64),
        }

    bin_arr = np.asarray(
        np.broadcast_to(np.asarray(bin_shape, dtype=np.float64), (ndim,))
    )
    if n <= 1 or not np.all(bin_arr > 0):
        owner = np.arange(n, dtype=np.int64)
    else:
        owner = _cluster_owners(n, edges, _bin_keys(positions, bin_arr))

    k = int(owner.max()) + 1

    # Centroids.  bincount per axis rather than np.add.at: same result, and it
    # is the hot path for a 12.6M-vertex level.
    counts = np.bincount(owner, minlength=k).astype(np.float64)
    centroids = np.empty((k, ndim), dtype=np.float64)
    for d in range(ndim):
        centroids[:, d] = (
            np.bincount(owner, weights=positions[:, d].astype(np.float64), minlength=k)
            / counts
        )

    # Rank members within each cluster by distance to the centroid, breaking
    # ties on source index so the choice is deterministic across workers.
    d2 = np.sum((positions.astype(np.float64) - centroids[owner]) ** 2, axis=1)
    order = np.lexsort((np.arange(n, dtype=np.int64), d2, owner))
    rank = np.empty(n, dtype=np.int64)
    starts = np.zeros(k + 1, dtype=np.int64)
    np.cumsum(counts.astype(np.int64), out=starts[1:])
    rank[order] = np.arange(n, dtype=np.int64) - starts[owner[order]]

    new_edges = owner[edges] if len(edges) else np.zeros((0, 2), dtype=np.int64)
    if len(new_edges):
        new_edges = new_edges[new_edges[:, 0] != new_edges[:, 1]]
        # Dedupe whole rows, never sorted rows: sorting would destroy the
        # [child, parent] orientation the format relies on.  A true forest
        # cannot produce a duplicate here, but ingest cycles can.
        if len(new_edges):
            new_edges = np.unique(new_edges, axis=0)

    out_attrs = {
        name: _aggregate(
            np.asarray(values), owner, k, modes[name], order_within_cluster=rank
        )
        for name, values in attributes.items()
    }

    representative = np.empty(k, dtype=np.int64)
    representative[owner[rank == 0]] = np.flatnonzero(rank == 0)

    return {
        "positions": centroids.astype(positions.dtype, copy=False),
        "edges": new_edges.reshape(-1, 2).astype(np.int64),
        "attributes": out_attrs,
        "kept_source_indices": representative,
        "owner_new": owner,
    }


def heavy_path_relabel(
    positions: npt.NDArray[np.floating],
    edges: npt.NDArray[np.integer],
    attributes: Mapping[str, npt.NDArray] | None = None,
) -> tuple[npt.NDArray, npt.NDArray, dict[str, npt.NDArray], npt.NDArray[np.int64]]:
    """Relabel a rooted forest in heavy-first DFS preorder.

    Lengthens the implicit runs the writer produces, without touching the
    writer.  :func:`zarr_vectors.types.skeletons.decompose_tree_to_paths` builds
    each node's child list in ascending vertex index and continues the current
    path through ``kids[0]`` — the lowest-indexed child.  Visiting the heaviest
    subtree first in a preorder walk gives that child the lowest index among its
    siblings, so the path decomposition follows the heavy path instead of an
    arbitrary one.  The longest root-to-leaf run then comes back as a single
    fragment.

    This cannot change how *many* fragments there are: ``#fragments ==
    #leaves`` and ``#branch_links == #fragments - #roots`` regardless of
    ordering.  What it changes is the run-length distribution — which is what a
    reader pays for, since each fragment costs a separate per-fragment link
    read.

    "Heavy" is subtree cable length, not vertex count, so a long thin axon
    outranks a bushy but compact tuft.

    Returns:
        ``(positions, edges, attributes, new_of_old)`` in the new labelling.
    """
    positions = np.asarray(positions)
    n = len(positions)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    attributes = dict(attributes or {})
    if n == 0:
        return positions, edges, attributes, np.zeros(0, dtype=np.int64)

    parent = np.full(n, -1, dtype=np.int64)
    if len(edges):
        parent[edges[:, 0]] = edges[:, 1]
    children: list[list[int]] = [[] for _ in range(n)]
    for child in np.flatnonzero(parent >= 0):
        children[int(parent[child])].append(int(child))

    seg_len = np.zeros(n, dtype=np.float64)
    if len(edges):
        seg_len[edges[:, 0]] = np.linalg.norm(
            positions[edges[:, 0]].astype(np.float64)
            - positions[edges[:, 1]].astype(np.float64),
            axis=1,
        )

    roots = [int(v) for v in np.flatnonzero(parent < 0)]

    # Post-order accumulation of subtree cable length.  Iterative: these trees
    # run thousands of vertices deep and recursion would blow the stack.
    visit: list[int] = []
    stack = list(roots)
    while stack:
        v = stack.pop()
        visit.append(v)
        stack.extend(children[v])
    height = np.zeros(n, dtype=np.float64)
    for v in reversed(visit):
        for c in children[v]:
            height[v] = max(height[v], height[c] + seg_len[c])

    # Preorder, heaviest child first.  Pushed in ascending weight so the
    # heaviest is popped first and therefore numbered first.
    new_of_old = np.full(n, -1, dtype=np.int64)
    order: list[int] = []
    stack = list(reversed(roots))
    while stack:
        v = stack.pop()
        new_of_old[v] = len(order)
        order.append(v)
        kids = children[v]
        if len(kids) > 1:
            kids = sorted(kids, key=lambda c: (height[c] + seg_len[c], -c))
        stack.extend(kids)

    perm = np.asarray(order, dtype=np.int64)
    new_edges = (
        np.column_stack((new_of_old[edges[:, 0]], new_of_old[edges[:, 1]]))
        if len(edges)
        else np.zeros((0, 2), dtype=np.int64)
    )
    return (
        positions[perm],
        new_edges.astype(np.int64),
        {name: np.asarray(v)[perm] for name, v in attributes.items()},
        new_of_old,
    )
