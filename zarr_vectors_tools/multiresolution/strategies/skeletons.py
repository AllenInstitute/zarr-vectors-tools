"""Skeleton-aware coarsening (path simplification).

Unlike the generic per-object metavertex binning in
:mod:`zarr_vectors_tools.multiresolution.coarsen`, skeletons need a
*topology-preserving* downsample: branch points and endpoints must
survive every level so the tree's shape is recognizable, while the
many degree-2 vertices along smooth runs are decimated.

This module provides:

- :func:`simplify_skeleton` — pure function: simplify one rooted tree
  (positions + ``[child, parent]`` edges) with Ramer–Douglas–Peucker
  along each unbranched chain, always keeping endpoints + branch
  points, aggregating per-vertex attributes (e.g. ``radius``,
  ``cross_sectional_area``) onto the survivors.

- :func:`coarsen_skeleton_level` — read one resolution level of a
  ``links_convention="implicit_sequential_with_branches"`` store,
  simplify every (surviving) object's per-chunk skeleton pieces, and
  write the coarser level with stable object IDs.  Routed to from
  :func:`zarr_vectors_tools.multiresolution.coarsen.coarsen_level` when the
  store's links convention is the skeleton convention.

The coarsener treats each stored *fragment* as one connected skeleton
piece living entirely within a single chunk.  Because coarser chunk
grids are nested (a target chunk is the union of ``chunk_scale`` source
chunks per axis), a simplified piece never straddles a target chunk
boundary, so each piece maps to exactly one fragment in one target
chunk.  Cross-chunk links are intentionally not reconstructed (the
ingest convention is "ignore cross-chunk edges if missing").
"""

from __future__ import annotations

import pickle
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import (
    VERTEX_ATTRIBUTES,
    VERTICES,
)
from zarr_vectors_tools.multiresolution._ccl_compat import COARSEN_SKELETON


# ===================================================================
# Pure tree simplification
# ===================================================================

def _build_rooted_tree(
    n: int, edges: npt.NDArray[np.integer]
) -> tuple[npt.NDArray[np.int64], dict[int, list[int]], list[int]]:
    """Build (parent, children, roots) from ``[child, parent]`` edges.

    A node with no incoming parent edge is a root (skeleton pieces are
    single-rooted, but disconnected input is tolerated → multiple
    roots).
    """
    parent = np.full(n, -1, dtype=np.int64)
    if len(edges) > 0:
        e = np.asarray(edges, dtype=np.int64)
        parent[e[:, 0]] = e[:, 1]
    children: dict[int, list[int]] = defaultdict(list)
    for c in range(n):
        p = int(parent[c])
        if p >= 0:
            children[p].append(c)
    roots = [i for i in range(n) if parent[i] < 0]
    return parent, children, roots


def _rdp_keep_indices(
    points: npt.NDArray[np.floating], tolerance: float
) -> list[int]:
    """Ramer–Douglas–Peucker: indices (into ``points``) to keep.

    Always keeps the first and last point.  Iterative stack-based
    implementation (no recursion-depth limit).  Distance is point-to-
    segment in D dimensions.
    """
    m = len(points)
    if m <= 2:
        return list(range(m))
    keep = np.zeros(m, dtype=bool)
    keep[0] = keep[-1] = True
    stack: list[tuple[int, int]] = [(0, m - 1)]
    tol2 = float(tolerance) * float(tolerance)
    while stack:
        lo, hi = stack.pop()
        if hi - lo < 2:
            continue
        a = points[lo]
        b = points[hi]
        ab = b - a
        ab_len2 = float(ab @ ab)
        seg = points[lo + 1 : hi]
        ap = seg - a
        if ab_len2 <= 1e-30:
            # Degenerate segment: distance to the shared endpoint.
            d2 = np.einsum("ij,ij->i", ap, ap)
        else:
            t = (ap @ ab) / ab_len2
            t = np.clip(t, 0.0, 1.0)
            proj = a + t[:, None] * ab
            diff = seg - proj
            d2 = np.einsum("ij,ij->i", diff, diff)
        k = int(np.argmax(d2))
        if d2[k] > tol2:
            idx = lo + 1 + k
            keep[idx] = True
            stack.append((lo, idx))
            stack.append((idx, hi))
    return np.flatnonzero(keep).tolist()


def simplify_skeleton(
    positions: npt.NDArray[np.floating],
    edges: npt.NDArray[np.integer],
    *,
    tolerance: float,
    attributes: dict[str, npt.NDArray] | None = None,
    attr_agg: str = "max",
) -> dict[str, Any]:
    """Simplify one rooted tree, preserving endpoints and branch points.

    Args:
        positions: ``(N, D)`` vertex positions.
        edges: ``(M, 2)`` ``[child, parent]`` tree edges (as stored for
            the skeleton convention).
        tolerance: RDP perpendicular-distance threshold in position
            units (nm).  ``<= 0`` is the identity (no decimation).
        attributes: Optional per-vertex attributes ``{name: (N,) or
            (N, C)}``.  Each survivor aggregates the values of the
            source vertices that collapse onto it.
        attr_agg: Aggregation for collapsed runs — ``"max"`` (default),
            ``"mean"``, ``"min"``, or ``"first"``.

    Returns:
        Dict with ``positions`` ``(K, D)``, ``edges`` ``(K-1 or fewer,
        2)`` ``[child, parent]`` over the *new* indices,
        ``attributes`` (aggregated, same keys as input), and
        ``kept_source_indices`` ``(K,)`` mapping new index → original
        index.  The result is NOT re-rooted/re-ordered here — callers
        that need DFS order + branch-link extraction run the existing
        ``_reorder_tree`` / ``_extract_branch_links`` helpers on it.
    """
    positions = np.asarray(positions)
    n = len(positions)
    if n == 0:
        return {
            "positions": positions.reshape(0, positions.shape[1] if positions.ndim == 2 else 0),
            "edges": np.zeros((0, 2), dtype=np.int64),
            "attributes": {k: np.asarray(v)[:0] for k, v in (attributes or {}).items()},
            "kept_source_indices": np.zeros(0, dtype=np.int64),
        }

    parent, children, roots = _build_rooted_tree(n, edges)

    # Anchors: roots, leaves (no children), and branch points (>1 child)
    # always survive.
    keep = np.zeros(n, dtype=bool)
    for r in roots:
        keep[r] = True
    for v in range(n):
        nc = len(children.get(v, ()))
        if nc == 0 or nc >= 2:
            keep[v] = True

    # RDP along each unbranched chain between two anchors.  Walk down
    # from every anchor through degree-2 nodes until the next anchor.
    if tolerance > 0 and n > 2:
        anchors = np.flatnonzero(keep).tolist()
        for a in anchors:
            for first_child in children.get(a, ()):
                chain = [a]
                cur = first_child
                # Follow while strictly degree-2 (one child) and not an
                # anchor itself.
                while True:
                    chain.append(cur)
                    if keep[cur]:
                        break
                    kids = children.get(cur, ())
                    if len(kids) != 1:
                        break  # leaf or branch → already an anchor
                    cur = kids[0]
                if len(chain) <= 2:
                    continue
                pts = positions[np.asarray(chain, dtype=np.int64)]
                local_keep = _rdp_keep_indices(pts, tolerance)
                for li in local_keep:
                    keep[chain[li]] = True

    return _collapse_to_kept(positions, parent, keep, attributes, attr_agg)


def _collapse_to_kept(
    positions: npt.NDArray,
    parent: npt.NDArray[np.int64],
    keep: npt.NDArray[np.bool_],
    attributes: dict[str, npt.NDArray] | None,
    attr_agg: str,
) -> dict[str, Any]:
    """Collapse a rooted tree to its kept vertices: reconnect each survivor
    to its nearest kept ancestor and aggregate per-vertex attributes onto
    survivors.  Shared by :func:`simplify_skeleton` and
    :func:`decimate_skeleton` — they differ only in which vertices ``keep``.
    """
    n = len(positions)
    kept = np.flatnonzero(keep)
    new_of_old = np.full(n, -1, dtype=np.int64)
    new_of_old[kept] = np.arange(len(kept), dtype=np.int64)

    nearest_keep_anc: dict[int, int] = {}

    def _nearest_kept_ancestor(node: int) -> int:
        path: list[int] = []
        cur = int(parent[node])
        while cur >= 0 and not keep[cur]:
            if cur in nearest_keep_anc:
                cur = nearest_keep_anc[cur]
                break
            path.append(cur)
            cur = int(parent[cur])
        for p in path:
            nearest_keep_anc[p] = cur
        return cur

    new_edges: list[tuple[int, int]] = []
    for old in kept.tolist():
        if parent[old] < 0:
            continue
        anc = _nearest_kept_ancestor(old)
        if anc < 0:
            continue
        new_edges.append((int(new_of_old[old]), int(new_of_old[anc])))

    owner_new = np.empty(n, dtype=np.int64)
    for v in range(n):
        if keep[v]:
            owner_new[v] = new_of_old[v]
        else:
            anc = _nearest_kept_ancestor(v)
            owner_new[v] = new_of_old[anc] if anc >= 0 else -1

    out_attrs: dict[str, npt.NDArray] = {}
    if attributes:
        K = len(kept)
        for name, data in attributes.items():
            data = np.asarray(data)
            tail = data.shape[1:]
            agg = np.zeros((K, *tail), dtype=data.dtype)
            valid = owner_new >= 0
            ow = owner_new[valid]
            vals = data[valid]
            if attr_agg == "max":
                np.fmax.at(agg, ow, vals)
            elif attr_agg == "min":
                agg[:] = np.iinfo(data.dtype).max if np.issubdtype(data.dtype, np.integer) else np.inf
                np.fmin.at(agg, ow, vals)
            elif attr_agg == "first":
                agg[new_of_old[kept]] = data[kept]
            else:  # mean
                counts = np.zeros(K, dtype=np.int64)
                acc = np.zeros((K, *tail), dtype=np.float64)
                np.add.at(acc, ow, vals.astype(np.float64))
                np.add.at(counts, ow, 1)
                counts = np.maximum(counts, 1)
                agg = (acc / counts.reshape((K,) + (1,) * len(tail))).astype(data.dtype)
            out_attrs[name] = agg

    return {
        "positions": positions[kept],
        "edges": np.asarray(new_edges, dtype=np.int64).reshape(-1, 2),
        "attributes": out_attrs,
        "kept_source_indices": kept.astype(np.int64),
    }


def decimate_skeleton(
    positions: npt.NDArray[np.floating],
    edges: npt.NDArray[np.integer],
    *,
    stride: int = 8,
    forced_keep: npt.NDArray[np.integer] | list[int] | None = None,
    attributes: dict[str, npt.NDArray] | None = None,
    attr_agg: str = "max",
) -> dict[str, Any]:
    """Uniformly decimate a rooted tree, keeping topology + anchors.

    Simpler and more predictable than RDP: ALWAYS keep branch points,
    endpoints (leaves), roots, and any ``forced_keep`` vertices (e.g.
    chunk-boundary / cross-chunk vertices); along each unbranched chain
    between two kept anchors, keep every ``stride``-th vertex.  This yields
    a ~``stride``× reduction per call regardless of geometry — and keeps
    reducing at deeper pyramid levels (RDP bottoms out once a skeleton is
    near-minimal).

    Same return shape as :func:`simplify_skeleton`.
    """
    positions = np.asarray(positions)
    n = len(positions)
    if n == 0:
        return {
            "positions": positions.reshape(0, positions.shape[1] if positions.ndim == 2 else 0),
            "edges": np.zeros((0, 2), dtype=np.int64),
            "attributes": {k: np.asarray(v)[:0] for k, v in (attributes or {}).items()},
            "kept_source_indices": np.zeros(0, dtype=np.int64),
        }

    parent, children, roots = _build_rooted_tree(n, edges)
    keep = np.zeros(n, dtype=bool)
    for r in roots:
        keep[r] = True
    for v in range(n):
        nc = len(children.get(v, ()))
        if nc == 0 or nc >= 2:
            keep[v] = True
    if forced_keep is not None:
        for v in np.asarray(forced_keep, dtype=np.int64).ravel():
            iv = int(v)
            if 0 <= iv < n:
                keep[iv] = True

    if stride > 1:
        anchors = np.flatnonzero(keep).tolist()
        for a in anchors:
            for first_child in children.get(a, ()):
                chain = [a]
                cur = first_child
                while True:
                    chain.append(cur)
                    if keep[cur]:
                        break
                    kids = children.get(cur, ())
                    if len(kids) != 1:
                        break
                    cur = kids[0]
                # chain[0] and chain[-1] are anchors; keep every stride-th
                # interior vertex (counted from the upper anchor).
                for j in range(stride, len(chain) - 1, stride):
                    keep[chain[j]] = True

    return _collapse_to_kept(positions, parent, keep, attributes, attr_agg)


# ===================================================================
# Level coarsening
# ===================================================================

def _merge_parts(parts, attr_names):
    """Concatenate an object's per-chunk pieces into one vertex/edge set.

    Edges are rebased by each part's running vertex offset; treated as
    undirected by the downstream re-rooting.  Missing attributes are
    zero-filled so all parts align.
    """
    vlist = []
    elist = []
    acc = {n: [] for n in attr_names}
    off = 0
    for verts, edges, attrs in parts:
        vlist.append(np.asarray(verts))
        e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        if len(e) > 0:
            elist.append(e + off)
        for name in attr_names:
            a = attrs.get(name)
            if a is None:
                a = np.zeros((len(verts),), dtype=np.float32)
            acc[name].append(np.asarray(a))
        off += len(verts)
    V = np.concatenate(vlist, axis=0)
    E = np.concatenate(elist, axis=0) if elist else np.zeros((0, 2), np.int64)
    A = {
        n: (np.concatenate(acc[n], axis=0) if acc[n] else np.zeros((0,), np.float32))
        for n in attr_names
    }
    return V, E, A


def _frag_edges(n: int) -> npt.NDArray:
    """Within-path implicit edges ``(i, i-1)`` for one linear fragment.

    Each stored fragment is a linear path (the skeleton convention), so its
    internal edges are sequential; branch links between an object's paths are
    re-added separately at the (object, target-chunk) level.
    """
    if n <= 1:
        return np.zeros((0, 2), dtype=np.int64)
    i = np.arange(1, n, dtype=np.int64)
    return np.stack([i, i - 1], axis=1)


def _build_local_plan(
    src,
    tcc: tuple[int, ...],
    *,
    scale: tuple[int, ...],
    ndim: int,
    attr_names: list[str],
    attr_dtypes: dict[str, np.dtype],
    keep_mask: npt.NDArray[np.uint8] | None,
    boundary_off: npt.NDArray,
    target_cs: npt.NDArray,
    source_cs: npt.NDArray,
    ccl_cells: list | None = None,
) -> tuple[list[dict], dict, dict]:
    """Build ONE target chunk's coarsen plan by reading only its source children.

    This replaces the old coordinator-built, whole-level, per-fragment plan (the
    ~40 GB of `dict[(chunk,fidx)]` structures that OOM'd at scale).  Everything
    here is local to the target chunk:

    - read the source children ``scc ∈ [tcc·scale, (tcc+1)·scale)``: per-fragment
            vertices/attributes, per-fragment ``segment_id`` + ``object_id`` (grouping), the
      branch links, and the fragment ranges (chunk-local → fragment),
        - group fragments by per-fragment ``object_id`` (skip non-kept via
            ``keep_mask[oid]`` under sparsity),
    - per object, compute merge edges in merged-index space from (a) branch links
      (remapped) and (b) **coincident boundary vertices on faces *interior* to the
      target chunk** (source-face AND NOT target-face) — the same connectivity the
      stored cross-chunk links encode, recovered geometrically,
    - force-keep vertices on the target chunk's **outer** faces (so Phase B can
      stitch cross-target links by coincidence).

    Returns ``(groups, vcache, acache)`` where ``groups`` matches the dict shape the
    decimate/write loop already consumes (``oid``/``segment_id``/``members``/
    ``intra_extra``/``forced``), and ``vcache``/``acache`` are the per-fragment
    vertex/attribute reads (reused by that loop, no double read).
    """
    from itertools import product

    from zarr_vectors.core.arrays import (
        read_chunk_attributes,
        read_chunk_fragment_attributes,
        read_chunk_links,
        read_chunk_vertices,
    )
    from zarr_vectors.exceptions import ArrayError

    tcc = tuple(int(x) for x in tcc)
    child_ccs = [
        tuple(tcc[a] * scale[a] + d[a] for a in range(ndim))
        for d in product(*[range(scale[a]) for a in range(ndim)])
    ]

    vcache: dict = {}
    acache: dict = {name: {} for name in attr_names}
    fragseg: dict = {}
    fragoid: dict = {}
    fraglinks: dict = {}
    child_ranges: dict = {}  # scc -> (starts, ends, fidxs) for chunk-local→fragment
    for scc in child_ccs:
        try:
            vgroups = read_chunk_vertices(src, scc, dtype=np.float32, ndim=ndim)
        except ArrayError:
            continue
        if not vgroups:
            continue
        try:
            segs = read_chunk_fragment_attributes(src, "segment_id", scc, dtype=np.uint64)
        except ArrayError:
            segs = None
        try:
            oids = read_chunk_fragment_attributes(src, "object_id", scc, dtype=np.uint64)
        except ArrayError:
            oids = None
        try:
            lgroups = read_chunk_links(src, scc, link_width=2, delta=0)
        except ArrayError:
            lgroups = None

        keep_fidx: set[int] = set()
        starts = []
        st = 0
        for fidx in range(len(vgroups)):
            cnt = len(vgroups[fidx])
            starts.append((st, st + cnt, fidx))
            st += cnt
            seg = int(segs[fidx]) if segs is not None and fidx < len(segs) else -1
            oid = int(oids[fidx]) if oids is not None and fidx < len(oids) else -1
            if seg < 0 or oid < 0:
                continue
            if keep_mask is not None:
                if oid < 0 or oid >= len(keep_mask) or int(keep_mask[oid]) == 0:
                    continue
            keep_fidx.add(fidx)
            vcache[(scc, fidx)] = vgroups[fidx]
            fragseg[(scc, fidx)] = seg
            fragoid[(scc, fidx)] = oid
            fraglinks[(scc, fidx)] = (
                lgroups[fidx]
                if lgroups is not None and fidx < len(lgroups)
                else np.zeros((0, 2), np.int64)
            )

        if keep_fidx:
            for name in attr_names:
                try:
                    ag = read_chunk_attributes(src, name, scc, dtype=attr_dtypes[name])
                except ArrayError:
                    ag = None
                if ag is not None:
                    for fidx in keep_fidx:
                        if fidx < len(ag):
                            acache[name][(scc, fidx)] = ag[fidx]
        child_ranges[scc] = (
            np.asarray([r[0] for r in starts], np.int64),
            np.asarray([r[1] for r in starts], np.int64),
            np.asarray([r[2] for r in starts], np.int64),
        )

    # group fragments → objects by segment_id
    members_by_oid: dict[int, list] = defaultdict(list)
    seg_by_oid: dict[int, int] = {}
    for (scc, fidx), seg in fragseg.items():
        oid = int(fragoid.get((scc, fidx), -1))
        if oid < 0:
            continue
        members_by_oid[oid].append((scc, fidx))
        seg_by_oid[oid] = seg

    cs_src = np.rint(np.asarray(source_cs, dtype=np.float64)).astype(np.int64)
    cs_tgt = np.rint(np.asarray(target_cs, dtype=np.float64)).astype(np.int64)
    off = np.rint(np.asarray(boundary_off, dtype=np.float64)).astype(np.int64)

    def _resolve(scc, vi):
        s, e, f = child_ranges[scc]
        i = int(np.searchsorted(s, vi, side="right")) - 1
        if i < 0 or vi >= int(e[i]):
            return None
        return int(f[i]), vi - int(s[i])

    # Stored source cross-chunk links touching our children → intra-target
    # merges.  Reading the v0.8 partitioned ccl per chunk-pair stays local and,
    # unlike geometric coincidence, handles BOTH coincident boundary vertices
    # AND phase-split (distinct-vertex) cross edges.
    from zarr_vectors_tools.multiresolution._ccl_compat import (
        read_cross_chunk_link_leaf,
    )

    child_set = set(child_ranges)
    oid_of_frag: dict = {}
    for _oid, _mems in members_by_oid.items():
        for _m in _mems:
            oid_of_frag[_m] = _oid
    conns_by_oid: dict = defaultdict(list)
    # ``ccl_cells`` are the source cross-chunk-link cells touching this target's
    # children (enumerated once by the coordinator and bucketed per target).
    for cell in (ccl_cells or ()):
        try:
            recs = read_cross_chunk_link_leaf(src, cell, delta=0)
        except Exception:
            continue
        for rec in recs:
            if len(rec) != 2:
                continue
            (ccA, viA), (ccB, viB) = rec
            ccA = tuple(int(x) for x in ccA)
            ccB = tuple(int(x) for x in ccB)
            if ccA not in child_set or ccB not in child_set:
                continue  # one endpoint outside this target chunk → cross-target
            rA = _resolve(ccA, int(viA))
            rB = _resolve(ccB, int(viB))
            if rA is None or rB is None:
                continue
            oa = oid_of_frag.get((ccA, rA[0]))
            ob = oid_of_frag.get((ccB, rB[0]))
            if oa is None or oa != ob:
                continue
            conns_by_oid[oa].append((ccA, rA[0], rA[1], ccB, rB[0], rB[1]))

    groups: list[dict] = []
    for oid in sorted(members_by_oid):
        members = sorted(members_by_oid[oid])
        goff: dict = {}
        o = 0
        for m in members:
            goff[m] = o
            o += len(vcache[m])

        intra_extra: list[tuple[int, int]] = []
        # (a) branch links (chunk-local endpoints) → merged-index edges
        for (scc, fidx) in members:
            links = fraglinks.get((scc, fidx))
            if links is None or len(links) == 0:
                continue
            for ch_cl, par_cl in np.asarray(links, np.int64).reshape(-1, 2):
                ra = _resolve(scc, int(ch_cl))
                rb = _resolve(scc, int(par_cl))
                if ra is None or rb is None:
                    continue
                if (scc, ra[0]) not in goff or (scc, rb[0]) not in goff:
                    continue
                intra_extra.append(
                    (goff[(scc, ra[0])] + ra[1], goff[(scc, rb[0])] + rb[1])
                )

        # (c) stored intra-target cross-chunk links → merge edges
        for ccA, fA, lA, ccB, fB, lB in conns_by_oid.get(oid, ()):
            if (ccA, fA) in goff and (ccB, fB) in goff:
                intra_extra.append(
                    (goff[(ccA, fA)] + lA, goff[(ccB, fB)] + lB)
                )

        # (b) coincident boundary vertices on INTERIOR faces → merge edges;
        #     vertices on OUTER target faces → force-keep for Phase B.
        forced: set = set()
        coord_cols = []
        midx_cols = []
        for (scc, fidx) in members:
            verts = vcache[(scc, fidx)]
            if len(verts) == 0:
                continue
            coord = np.rint(verts).astype(np.int64)
            on_src = np.any(np.mod(coord - off, cs_src) == 0, axis=1)
            on_tgt = np.any(np.mod(coord - off, cs_tgt) == 0, axis=1)
            base = goff[(scc, fidx)]
            for li in np.flatnonzero(on_tgt):
                forced.add(base + int(li))
            interior = np.flatnonzero(on_src & ~on_tgt)
            if len(interior):
                coord_cols.append(coord[interior])
                midx_cols.append(base + interior)
        if coord_cols:
            allc = np.concatenate(coord_cols, axis=0)
            allm = np.concatenate(midx_cols)
            order = np.lexsort([allc[:, i] for i in range(ndim - 1, -1, -1)])
            sc = allc[order]
            sm = allm[order]
            change = np.any(sc[1:] != sc[:-1], axis=1)
            starts2 = np.concatenate([[0], np.flatnonzero(change) + 1])
            ends2 = np.concatenate([starts2[1:], [len(order)]])
            for s2, e2 in zip(starts2, ends2):
                if e2 - s2 < 2:
                    continue
                reps = sm[s2:e2]
                for k in range(1, len(reps)):
                    intra_extra.append((int(reps[0]), int(reps[k])))

        groups.append({
            "oid": oid,
            "segment_id": int(seg_by_oid[oid]),
            "members": [(list(cc), int(fidx)) for cc, fidx in members],
            "intra_extra": [[int(a), int(b)] for a, b in intra_extra],
            "forced": sorted(forced),
        })
    return groups, vcache, acache


def _coarsen_target_chunk(payload: dict, shared: dict | None = None) -> dict:
    """Coarsen ONE target chunk — a picklable worker for parallel pyramiding.

    Reads only this target chunk's source children (the source chunks that nest
    into it) — vertices + per-vertex attributes — then for each object group
    merges its source fragments, re-splits into connected components, decimates
    (force-keeping chunk-boundary / cross-target anchors), and writes the target
    chunk.  Returns the chunk's object-index ``records``, cross-target
    ``anchor_locs``, and total output vertex count.

    The coordinator (:func:`coarsen_skeleton_level`) precomputes every plan
    field from *metadata only* (manifests, per-fragment vertex counts, branch /
    cross-chunk links), so this worker never needs the whole source level in RAM
    — bounding memory and letting an executor run target chunks in parallel.
    The plan that is common to the whole level (per-target-chunk object groups,
    attribute spec, level params) arrives via ``shared`` — scattered to the
    workers **once** rather than re-pickled into every per-chunk payload (which,
    on dense/few-chunk levels where one chunk holds most objects, otherwise
    dominates the runtime).  Workers write disjoint chunk files; the level +
    arrays are created by the coordinator before dispatch.
    """
    from zarr_vectors.core.store import get_resolution_level, open_store
    from zarr_vectors_tools.multiresolution.skeleton_graph import split_components
    from zarr_vectors.types.skeletons import write_skeleton_chunk

    shared = shared or {}
    ndim = shared["ndim"]
    attr_names = shared["attr_names"]
    attr_dtypes = {n: np.dtype(d) for n, d in shared["attr_dtypes"].items()}
    stride = shared["stride"]
    attr_agg = shared["attr_agg"]
    tcc = tuple(int(x) for x in payload["tcc"])
    scale = tuple(int(s) for s in shared["scale"])
    keep_mask = shared.get("keep_mask")  # None ⇒ keep all
    boundary_off = np.asarray(shared["boundary_offset"], dtype=np.float64)
    target_cs = np.asarray(shared["target_cs"], dtype=np.float64)
    source_cs = np.asarray(shared["source_cs"], dtype=np.float64)
    drop_below = int(shared.get("drop_interior_below", 0) or 0)

    root = open_store(shared["store_path"], mode="r+")
    src = get_resolution_level(root, shared["source_level"])
    level_group = get_resolution_level(root, shared["target_level"])

    # Self-plan + read children locally — no coordinator plan (kills the per-fragment
    # central state). ``forced`` here = vertices on the target chunk's OUTER faces.
    groups, vcache, acache = _build_local_plan(
        src, tcc, scale=scale, ndim=ndim, attr_names=attr_names,
        attr_dtypes=attr_dtypes, keep_mask=keep_mask,
        boundary_off=boundary_off, target_cs=target_cs, source_cs=source_cs,
        ccl_cells=payload.get("ccl_cells"),
    )
    input_fragments = int(len(vcache))
    input_vertices = int(sum(len(v) for v in vcache.values()))
    input_objects = int(len(groups))

    pieces: list = []
    total_out_vertices = 0
    anchor_meta: dict = {}   # tag -> (segment_id, coord-tuple) for outer-face verts
    tagc = 0
    for g in groups:
        parts = []
        for cc, fidx in g["members"]:
            key = (tuple(cc), fidx)
            verts = vcache.get(key)
            if verts is None or len(verts) == 0:
                continue
            attrs = {
                name: acache[name][key]
                for name in attr_names
                if key in acache[name]
            }
            parts.append((verts, _frag_edges(len(verts)), attrs))
        if not parts:
            continue
        mverts, medges, mattrs = _merge_parts(parts, attr_names)
        extra = g["intra_extra"]
        if extra:
            ex = np.asarray(extra, dtype=np.int64).reshape(-1, 2)
            medges = np.concatenate([medges, ex], axis=0) if len(medges) else ex
        forced_set = set(g["forced"])
        comps = split_components(
            mverts, medges, mattrs, vertex_ids=np.arange(len(mverts)),
        )
        for comp in comps:
            comp_vids = comp["vertex_ids"].tolist()
            pos_in_comp = {int(mv): ci for ci, mv in enumerate(comp_vids)}
            forced_local = None
            if forced_set:
                forced_local = [pos_in_comp[mv] for mv in forced_set
                                if mv in pos_in_comp]
            simp = decimate_skeleton(
                comp["positions"], comp["edges"],
                stride=stride, forced_keep=forced_local,
                attributes=comp["attributes"], attr_agg=attr_agg,
            )
            rpos = simp["positions"]
            if len(rpos) == 0:
                continue
            piece: dict = {
                "object_id": g["oid"],
                "segment_id": g["segment_id"],
                "positions": rpos,
                "edges": simp["edges"],
                "attributes": simp["attributes"],
            }
            # Tag surviving OUTER-face vertices as anchors → resolve their stored
            # chunk-local index after write → cross-target sidecar for Phase B.
            if forced_set:
                kept_pos = {
                    int(c): i
                    for i, c in enumerate(simp["kept_source_indices"].tolist())
                }
                anchors = {}
                for mv in forced_set:
                    ci = pos_in_comp.get(mv)
                    if ci is None:
                        continue
                    sl = kept_pos.get(ci)
                    if sl is None:
                        continue
                    anchors[tagc] = sl
                    anchor_meta[tagc] = (
                        int(g["segment_id"]),
                        tuple(int(x) for x in np.rint(comp["positions"][ci]).astype(np.int64)),
                    )
                    tagc += 1
                if anchors:
                    piece["anchors"] = anchors
            pieces.append(piece)
            total_out_vertices += len(rpos)

    # LOD drop of small, fully-interior objects (no outer-face vertex → cannot
    # extend into a neighbour chunk).  Interior objects have empty ``forced``,
    # so re-test geometrically on the written pieces.
    dropped_oids: list = []
    if drop_below > 0:
        off = np.rint(boundary_off).astype(np.int64)
        cs = np.rint(target_cs).astype(np.int64)
        by_oid: dict = defaultdict(list)
        for i, p in enumerate(pieces):
            by_oid[int(p["object_id"])].append(i)
        keep_mask = [True] * len(pieces)
        for oid, idxs in by_oid.items():
            if sum(len(pieces[i]["positions"]) for i in idxs) > drop_below:
                continue
            interior = True
            for i in idxs:
                coord = np.rint(pieces[i]["positions"]).astype(np.int64)
                if len(coord) and np.any(np.mod(coord - off, cs) == 0):
                    interior = False
                    break
            if interior:
                for i in idxs:
                    keep_mask[i] = False
                dropped_oids.append(oid)
        if dropped_oids:
            pieces = [p for i, p in enumerate(pieces) if keep_mask[i]]
            total_out_vertices = sum(len(p["positions"]) for p in pieces)

    recs, alocs = write_skeleton_chunk(
        level_group, tcc, pieces, attr_dtypes=attr_dtypes,
    )
    # Spill object-index refs partitioned by OID shard to avoid a central
    # gather of whole-level ``{oid: manifest}`` structures.
    oid_shards = int(shared.get("oid_reduce_shards", 1) or 1)
    oid_tmp_dir = str(shared.get("oid_reduce_tmp_dir", ""))
    sidecar_tmp_dir = str(shared.get("sidecar_tmp_dir", ""))
    shard_rows: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r in recs:
        oid = int(r[0])
        fidx = int(r[2])
        shard_rows[oid % oid_shards].append((oid, fidx))
    shard_files: list[tuple[int, str]] = []
    if oid_tmp_dir and shard_rows:
        tcc_tag = ".".join(str(int(x)) for x in tcc)
        for shard, rows in shard_rows.items():
            arr = np.asarray(rows, dtype=np.int64).reshape(-1, 2)
            path = Path(oid_tmp_dir) / f"{tcc_tag}.s{int(shard)}.npy"
            np.save(path, arr, allow_pickle=False)
            shard_files.append((int(shard), str(path)))
    sc_rows = []
    for tag, (seg, coord) in anchor_meta.items():
        loc = alocs.get(tag)
        if loc is not None:
            sc_rows.append((seg, *coord, int(loc[1])))  # [segment_id, *coord, vi]
    sidecar = (np.asarray(sc_rows, dtype=np.int64) if sc_rows
               else np.zeros((0, 2 + ndim), dtype=np.int64))
    sidecar_path = None
    if sidecar_tmp_dir and len(sidecar):
        tcc_tag = ".".join(str(int(x)) for x in tcc)
        p = Path(sidecar_tmp_dir) / f"{tcc_tag}.anchors.npy"
        np.save(p, sidecar, allow_pickle=False)
        sidecar_path = str(p)
    return {
        "tcc": tcc,
        "input_fragments": input_fragments,
        "input_vertices": input_vertices,
        "input_objects": input_objects,
        "fragment_count": int(len(recs)),
        "oid_shards": shard_files,
        "sidecar_path": sidecar_path,
        "vertex_count": int(total_out_vertices),
        "dropped_oids": np.asarray(dropped_oids, dtype=np.int64),
    }


def _cross_edge_shard(payload: dict, shared: dict | None = None) -> dict:
    """Phase B worker: write the cross-target-chunk links for ONE ccl shard.

    Each task owns the adjacent target-chunk pairs whose ``k2`` cells fall in a
    single outer shard (the coordinator partitions pairs by shard, so writers
    never collide).  For each pair it matches coincident ``(segment_id, coord)``
    OUTER-face vertices — the same igneous-style boundary coincidence used at L0 —
    from the two chunks' sidecars, and writes the link records into the sharded
    store via :func:`write_cross_chunk_link_cells`.
    """
    from zarr_vectors_tools.multiresolution._ccl_compat import write_cross_chunk_link_cells
    from zarr_vectors.core.store import get_resolution_level, open_store

    shared = shared or {}
    ndim = shared["ndim"]
    cgs = tuple(int(x) for x in shared["chunk_grid_shape"])
    corigin = tuple(int(x) for x in shared["chunk_origin"])
    sidecar_paths = {
        tuple(int(x) for x in e["tcc"]): str(e["path"])
        for e in payload.get("sidecar_paths", [])
    }
    pairs = payload["pairs"]
    sidecar_cache: dict[str, npt.NDArray[np.int64]] = {}
    keymap_cache: dict[tuple[int, ...], dict[tuple[int, tuple[int, ...]], int]] = {}

    def _read_sidecar(cc: tuple[int, ...]) -> npt.NDArray[np.int64] | None:
        path = sidecar_paths.get(cc)
        if path is None:
            return None
        arr = sidecar_cache.get(path)
        if arr is None:
            arr = np.asarray(np.load(path, allow_pickle=False), dtype=np.int64)
            sidecar_cache[path] = arr
        return arr

    def _chunk_keymap(cc: tuple[int, ...]) -> dict[tuple[int, tuple[int, ...]], int]:
        km = keymap_cache.get(cc)
        if km is not None:
            return km
        arr = _read_sidecar(cc)
        km = {}
        if arr is not None and len(arr):
            for row in arr:
                key = (int(row[0]), tuple(int(x) for x in row[1:1 + ndim]))
                if key not in km:
                    km[key] = int(row[1 + ndim])
        keymap_cache[cc] = km
        return km

    links: list = []
    for A_, B_ in pairs:
        A = tuple(int(x) for x in A_)
        B = tuple(int(x) for x in B_)
        amap = _chunk_keymap(A)
        bmap = _chunk_keymap(B)
        if not amap or not bmap:
            continue
        # Intersect keymaps directly (seg_id + coord), iterating the smaller map.
        if len(amap) <= len(bmap):
            for k, viA in amap.items():
                viB = bmap.get(k)
                if viB is not None:
                    links.append([(A, int(viA)), (B, int(viB))])
        else:
            for k, viB in bmap.items():
                viA = amap.get(k)
                if viA is not None:
                    links.append([(A, int(viA)), (B, int(viB))])
    if links:
        root = open_store(shared["store_path"], mode="r+")
        level_group = get_resolution_level(root, shared["target_level"])
        write_cross_chunk_link_cells(
            level_group, links, sid_ndim=ndim,
            chunk_grid_shape=cgs, chunk_origin=corigin, delta=0, link_width=2,
        )
    return {"n_links": len(links)}


def _reduce_object_index_shard(payload: dict, shared: dict | None = None) -> dict:
    """Reduce one OID shard's chunk spills into encoded manifest blobs.

    Phase A workers spill per-chunk ``(oid, fidx)`` rows partitioned by
    ``oid % num_shards``. This worker ingests one shard's spills, groups rows by
    OID, and returns encoded v0.6 manifest blobs for just those OIDs.
    """
    from zarr_vectors.encoding.fragments import encode_object_manifest_blocks

    shared = shared or {}
    sid_ndim = int(shared["sid_ndim"])

    manifests: dict[int, list[tuple[tuple[int, ...], int]]] = defaultdict(list)
    for e in payload.get("entries", []):
        tcc = tuple(int(x) for x in e["tcc"])
        path = str(e["path"])
        arr = np.load(path, allow_pickle=False)
        if arr.size == 0:
            continue
        rows = np.asarray(arr, dtype=np.int64).reshape(-1, 2)
        for oid, fidx in rows.tolist():
            manifests[int(oid)].append((tcc, int(fidx)))

    oids = np.asarray(sorted(manifests), dtype=np.int64)
    blobs: list[bytes] = []
    for oid in oids.tolist():
        blocks = [
            (tuple(int(c) for c in chunk_coords), int(fragment_index))
            for chunk_coords, fragment_index in sorted(
                manifests[int(oid)], key=lambda x: (tuple(x[0]), int(x[1]))
            )
        ]
        blobs.append(encode_object_manifest_blocks(blocks, sid_ndim=sid_ndim))

    return {
        "oid": oids,
        "blobs": pickle.dumps(blobs, protocol=pickle.HIGHEST_PROTOCOL),
    }


def coarsen_skeleton_level(
    store_path: str | Path,
    source_level: int,
    target_level: int,
    *,
    stride: int = 8,
    sparsity_factor: float = 1.0,
    chunk_scale_factor: int | tuple[int, ...] = 2,
    sparsity_strategy: str = "length",
    sparsity_seed: int | None = None,
    attr_agg: str = "max",
    drop_interior_below: int = 0,
    boundary_offset_nm: Sequence[float] | None = None,
    executor: Any = None,
) -> dict[str, Any]:
    """Coarsen one skeleton level by uniform per-path decimation.

    Reads ``source_level`` (a ``implicit_sequential_with_branches``
    store written by :func:`zarr_vectors.types.skeletons.write_skeleton_chunk`),
    decimates every surviving object with :func:`decimate_skeleton`
    (keep all branch points, endpoints, and chunk-boundary / cross-chunk
    vertices; otherwise keep every ``stride``-th vertex), and writes
    ``target_level`` with stable object IDs (dropped objects leave empty
    manifest slots).

    Args:
        store_path: Store path.
        source_level / target_level: Source and (new) target levels.
        stride: Keep every ``stride``-th non-anchor vertex (~``stride``×
            reduction per level).  ``8`` pairs with ``chunk_scale_factor=2``
            (8× volume) to hold per-chunk size roughly constant.
        sparsity_factor: Object-drop factor (≥1).  ``1.0`` keeps all
            objects.  Survivors keep their OIDs.
        chunk_scale_factor: Per-axis chunk-grid multiplier (target
            chunk_shape = source × factor).  Default 2 (nested 2× grid).
        sparsity_strategy: Object-selection strategy (see
            :mod:`zarr_vectors_tools.multiresolution.object_selection`).
            ``"length"`` drops shortest skeletons first.
        sparsity_seed: RNG seed.
        attr_agg: Per-vertex attribute aggregation over collapsed runs.

    Returns:
        Summary dict.
    """
    # Imports are local to avoid a module-load cycle
    # (coarsen.py imports this module).
    from zarr_vectors.core.arrays import (
        OBJECT_INDEX,
        OBJECT_INDEX_LAYOUT_V1,
        _write_object_index_manifests,
        create_attribute_array,
        create_fragment_attribute_array,
        create_links_array,
        create_object_attributes_array,
        create_object_index_array,
        create_vertices_array,
        read_chunk_fragment_attributes,
        list_chunk_keys,
        read_all_object_manifests,
        read_object_attributes,
        write_object_attributes,
    )
    from zarr_vectors_tools.multiresolution._ccl_compat import (
        CROSS_CHUNK_LINK_SHARD_AXIS,
        create_cross_chunk_link_kN_arrays,
        finalize_cross_chunk_links,
    )
    from zarr_vectors.core.metadata import (
        LevelMetadata,
        get_level_chunk_shape,
    )
    from zarr_vectors.core.store import (
        create_resolution_level,
        get_resolution_level,
        open_store,
        read_level_metadata,
        read_root_metadata,
    )
    from zarr_vectors.exceptions import ArrayError
    from zarr_vectors_tools.multiresolution.object_selection import apply_sparsity
    from zarr_vectors.core.multiscale import upsert_level_transform
    from zarr_vectors.types.skeletons import get_coordinate_offset

    # Per-target-chunk work is dispatched through ``executor`` (a
    # ``map``-like callable); the default runs serially in-process, so serial
    # output is identical to the parallel path by construction.
    if executor is None:
        def executor(func, items, shared=None):
            return [func(it, shared=shared) for it in items]

    import time as _time
    _t0 = _time.perf_counter()
    _timings: dict[str, float] = {}

    def _progress(msg: str) -> None:
        print(
            f"[coarsen L{int(source_level)}->L{int(target_level)}] {msg}",
            flush=True,
        )

    _progress("start")

    root = open_store(str(store_path), mode="r+")
    root_meta = read_root_metadata(root)
    ndim = root_meta.sid_ndim
    src = get_resolution_level(root, source_level)

    try:
        src_level_meta = read_level_metadata(root, source_level)
    except Exception:
        src_level_meta = None
    src_chunk_shape = get_level_chunk_shape(root_meta, src_level_meta)

    if isinstance(chunk_scale_factor, (tuple, list)):
        scale = tuple(int(s) for s in chunk_scale_factor)
    else:
        scale = tuple(int(chunk_scale_factor) for _ in range(ndim))
    target_chunk_shape = tuple(
        float(s) * int(r) for s, r in zip(src_chunk_shape, scale)
    )
    same_as_root = all(
        abs(t - r) < 1e-9 for t, r in zip(target_chunk_shape, root_meta.chunk_shape)
    )
    chunk_shape_override = None if same_as_root else target_chunk_shape

    # --- attribute names + dtypes ---------------------------------------
    attr_names: list[str] = []
    attr_dtypes: dict[str, np.dtype] = {}
    if VERTEX_ATTRIBUTES in src:
        for name in src[VERTEX_ATTRIBUTES]:
            try:
                meta = src.read_array_meta(f"{VERTEX_ATTRIBUTES}/{name}")
                attr_dtypes[name] = np.dtype(meta.get("dtype", "float32"))
                attr_names.append(name)
            except ArrayError:
                continue

    # --- object space (O(objects), not O(fragments)) --------------------
    # The dense OID space comes from object_attributes/segment_id, while worker
    # grouping requires per-fragment object_id on the source level.
    try:
        seg_array = np.asarray(read_object_attributes(src, "segment_id"))
    except ArrayError:
        seg_array = None
    if seg_array is None or len(seg_array) == 0:
        return {"vertex_count": 0, "object_count": 0, "method": COARSEN_SKELETON}
    seg_array = seg_array.astype(np.uint64)
    n_src = int(len(seg_array))

    # Hard requirement: source fragments must carry object_id. Probe a real
    # source chunk to avoid metadata-only false positives.
    src_vertex_chunks = list_chunk_keys(src, VERTICES)
    if not src_vertex_chunks:
        return {"vertex_count": 0, "object_count": 0, "method": COARSEN_SKELETON}
    src_has_fragment_oid = False
    probe_oid = read_chunk_fragment_attributes(
        src,
        "object_id",
        tuple(int(x) for x in src_vertex_chunks[0]),
        dtype=np.uint64,
        default=None,
    )
    src_has_fragment_oid = probe_oid is not None
    if not src_has_fragment_oid:
        raise ValueError(
            "coarsen_skeleton_level now requires fragment_attributes/object_id "
            "on the source level; re-ingest or migrate the source level first"
        )

    # Sparsity keep-set (O(objects)).  The "length" strategy needs per-object
    # sizes — the only path that still reads all manifests (O(fragments)), and
    # only when sparsity is active.
    if sparsity_factor > 1.0 and n_src > 1:
        manifest_lens = np.array(
            [len(m) for m in read_all_object_manifests(src)], dtype=np.float64
        )
        lengths = manifest_lens if sparsity_strategy == "length" else None
        # Objects already emptied by an earlier pyramid level's sparsity
        # drop must not be re-"kept" here — see `apply_sparsity`'s
        # `alive_mask` docstring.
        alive_mask = manifest_lens > 0
        kept = apply_sparsity(
            n_src, 1.0 / sparsity_factor, sparsity_strategy,
            seed=sparsity_seed, lengths=lengths, alive_mask=alive_mask,
        )
        keep_mask = np.zeros(n_src, dtype=np.uint8)
        keep_mask[np.asarray(kept, dtype=np.int64)] = 1
    else:
        keep_mask = None  # keep all

    # --- target chunks from the source chunk grid (O(chunks)) -----------
    target_chunks = sorted({
        tuple(int(cc[a] // scale[a]) for a in range(ndim))
        for cc in src_vertex_chunks
    })
    if not target_chunks:
        return {"vertex_count": 0, "object_count": 0, "method": COARSEN_SKELETON}

    # --- create target level + arrays (vertex_count patched after workers) -
    level_meta = LevelMetadata(
        level=target_level,
        vertex_count=0,
        arrays_present=[VERTICES, "links", "object_index"],
        bin_shape=tuple(root_meta.effective_bin_shape),
        bin_ratio=tuple(1 for _ in range(ndim)),
        chunk_shape=chunk_shape_override,
        object_sparsity=max(1e-9, min(1.0, 1.0 / sparsity_factor)),
        coarsening_method=COARSEN_SKELETON,
        parent_level=source_level,
        preserves_object_ids=True,
        inherited_num_objects=n_src,
    )
    level_group = create_resolution_level(root, target_level, level_meta)
    # Vertices stay in the stored (shifted) world frame at every level; mirror
    # the coordinate offset onto this level's NGFF transform.
    _offset = get_coordinate_offset(root, ndim)
    if np.any(_offset != 0):
        upsert_level_transform(
            root, target_level, scale=[1.0] * ndim,
            translation=[float(x) for x in _offset],
        )
    create_vertices_array(level_group, dtype="float32")
    create_links_array(level_group, link_width=2, delta=0)
    create_object_index_array(level_group)
    create_fragment_attribute_array(level_group, "segment_id", dtype="uint64")
    create_fragment_attribute_array(level_group, "object_id", dtype="uint64")
    for name in attr_names:
        create_attribute_array(level_group, name, dtype=str(attr_dtypes[name]))

    # Pre-create the kN cross-chunk-link arrays with LEVEL-WIDE dims so Phase-B
    # workers only WRITE cells (no create-race) and every writer agrees on the
    # array shape + origin.  Grid bounds come from the target-chunk extent.
    tc_arr = np.asarray(target_chunks, dtype=np.int64)
    cmin = tc_arr.min(axis=0)
    cmax = tc_arr.max(axis=0)
    chunk_origin = tuple(int(min(0, int(cmin[a]))) for a in range(ndim))
    chunk_grid_shape = tuple(
        int(max(1, int(cmax[a]) - chunk_origin[a] + 1)) for a in range(ndim)
    )
    create_cross_chunk_link_kN_arrays(
        level_group, sid_ndim=ndim, link_width=2,
        chunk_grid_shape=chunk_grid_shape, chunk_origin=chunk_origin, max_K=2,
    )

    # --- Phase A: decimate each target chunk (workers self-plan locally) ---
    boundary_off = (list(boundary_offset_nm)
                    if boundary_offset_nm is not None else [0.0] * ndim)
    oid_reduce_shards = 64
    oid_reduce_tmp_dir = tempfile.mkdtemp(prefix=f"oid_reduce_l{target_level}_")
    sidecar_tmp_dir = tempfile.mkdtemp(prefix=f"sidecar_l{target_level}_")

    sharedA = {
        "store_path": str(store_path),
        "source_level": int(source_level),
        "target_level": int(target_level),
        "scale": list(scale),
        "ndim": int(ndim),
        "attr_names": list(attr_names),
        "attr_dtypes": {n: str(attr_dtypes[n]) for n in attr_names},
        "stride": int(stride),
        "attr_agg": attr_agg,
        "keep_mask": keep_mask,
        "boundary_offset": boundary_off,
        "target_cs": list(target_chunk_shape),
        "source_cs": list(src_chunk_shape),
        "drop_interior_below": int(drop_interior_below or 0),
        "oid_reduce_shards": int(oid_reduce_shards),
        "oid_reduce_tmp_dir": str(oid_reduce_tmp_dir),
        "sidecar_tmp_dir": str(sidecar_tmp_dir),
    }
    # Enumerate the source cross-chunk-link cells ONCE and bucket each cell to
    # the target chunk(s) that own its endpoint chunks (chunk // scale). Replaces
    # a per-target-per-child list_cross_chunk_link_leaves() scan
    # (O(target_chunks × children × cells)) with a single O(cells) pass; workers
    # then read only their bucket's cells.
    from zarr_vectors_tools.multiresolution._ccl_compat import (
        list_cross_chunk_link_leaves,
    )
    cells_by_target: dict[tuple[int, ...], list] = defaultdict(list)
    for cell in list_cross_chunk_link_leaves(src, delta=0):
        seen_t: set = set()
        for c in cell:
            t = tuple(int(c[a]) // scale[a] for a in range(ndim))
            if t in seen_t:
                continue
            seen_t.add(t)
            cells_by_target[t].append(cell)
    payloadsA = [
        {"tcc": list(tcc), "ccl_cells": cells_by_target.get(tuple(tcc), [])}
        for tcc in target_chunks
    ]
    _timings["setup"] = _time.perf_counter() - _t0
    _progress(f"phase A start: target_chunks={len(payloadsA)}")
    _tmap = _time.perf_counter()
    resultsA = list(executor(_coarsen_target_chunk, payloadsA, sharedA))
    _timings["map_phase_a"] = _time.perf_counter() - _tmap
    _progress(f"phase A done: dt={_timings['map_phase_a']:.2f}s")
    _tfin = _time.perf_counter()

    # --- gather (compact NumPy) + build object_index --------------------
    total_out_vertices = 0
    total_fragments = 0
    total_in_fragments = 0
    total_in_vertices = 0
    total_in_objects = 0
    max_in_fragments = 0
    max_in_vertices = 0
    max_in_objects = 0
    sidecar_paths: dict[tuple[int, ...], str] = {}
    shard_entries: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for res in resultsA:
        in_f = int(res.get("input_fragments", 0))
        in_v = int(res.get("input_vertices", 0))
        in_o = int(res.get("input_objects", 0))
        total_in_fragments += in_f
        total_in_vertices += in_v
        total_in_objects += in_o
        max_in_fragments = max(max_in_fragments, in_f)
        max_in_vertices = max(max_in_vertices, in_v)
        max_in_objects = max(max_in_objects, in_o)
        total_fragments += int(res["fragment_count"])
        total_out_vertices += int(res["vertex_count"])
        tcc = tuple(int(x) for x in res["tcc"])
        sp = res.get("sidecar_path")
        if sp:
            sidecar_paths[tcc] = str(sp)
        for shard, path in res.get("oid_shards", []):
            shard_entries[int(shard)].append(
                {"tcc": list(tcc), "path": str(path)}
            )
    # Release phase-A result payloads as soon as we've compacted what we need.
    resultsA = []

    level_meta.vertex_count = int(total_out_vertices)
    create_resolution_level(root, target_level, level_meta)
    _progress(
        "phase A stats: "
        f"in_objs={total_in_objects} in_frags={total_in_fragments} "
        f"in_verts={total_in_vertices} "
        f"max_task_objs={max_in_objects} max_task_frags={max_in_fragments} "
        f"max_task_verts={max_in_vertices}"
    )

    # --- Phase C: shard-parallel object-index reduce + one-shot commit -----
    empty_blob = b"\x00\x00\x00\x00"  # encode_object_manifest_blocks([], sid_ndim)
    manifest_blobs: list[bytes] = [empty_blob] * int(n_src)
    try:
        _tc = _time.perf_counter()
        payloadsC = [
            {"shard": int(s), "entries": es}
            for s, es in sorted(shard_entries.items())
            if es
        ]
        _progress(f"phase C start: shard_tasks={len(payloadsC)}")
        sharedC = {"sid_ndim": int(ndim)}
        if payloadsC:
            for rc in executor(_reduce_object_index_shard, payloadsC, sharedC):
                oids = np.asarray(rc.get("oid", np.zeros((0,), np.int64)), np.int64)
                blobs = pickle.loads(rc.get("blobs", b"")) if len(oids) else []
                for i, oid in enumerate(oids.tolist()):
                    if 0 <= int(oid) < int(n_src):
                        manifest_blobs[int(oid)] = blobs[i]
        _write_object_index_manifests(level_group, manifest_blobs)
        level_group.write_array_meta(OBJECT_INDEX, {
            "zv_array": "object_index",
            "num_objects": int(n_src),
            "sid_ndim": int(ndim),
            "layout": OBJECT_INDEX_LAYOUT_V1,
        })
        _timings["reduce_phase_c"] = _time.perf_counter() - _tc
        _progress(f"phase C done: dt={_timings['reduce_phase_c']:.2f}s")
    finally:
        shutil.rmtree(oid_reduce_tmp_dir, ignore_errors=True)

    # carry object attributes forward; "present" = objects with geometry here.
    present_oids = np.flatnonzero(
        np.fromiter((1 if b != empty_blob else 0 for b in manifest_blobs), np.uint8)
    ).astype(np.int64)
    # object attributes are flat arrays under object_attributes/; enumerate via
    # children() (group_keys + array_keys) — iterating the group yields only
    # sub-group names, so a plain comprehension would miss every attribute.
    src_attr_names = (
        list(src["object_attributes"].children()) if "object_attributes" in src else []
    )
    mask = np.zeros(n_src, dtype=np.uint8)
    if len(present_oids):
        mask[present_oids] = 1
    _progress(f"object attrs start: names={len(src_attr_names)}")
    _tattrs = _time.perf_counter()
    for aname in src_attr_names:
        try:
            src_data = read_object_attributes(src, aname)
        except ArrayError:
            continue
        out = np.zeros_like(src_data)
        if len(present_oids):
            out[present_oids] = src_data[present_oids]
        create_object_attributes_array(level_group, aname, dtype=str(src_data.dtype))
        write_object_attributes(level_group, aname, out, present_mask=mask)
    _timings["object_attrs"] = _time.perf_counter() - _tattrs
    _progress(f"object attrs done: dt={_timings['object_attrs']:.2f}s")

    # Phase A wrote the vertices / links / attribute cells from separate
    # processes, whose per-array ``nonempty_chunks`` manifest RMWs race and can
    # under-report.  Re-derive them single-process from the on-disk cells so the
    # next level's coarsening source scan and the algorithms readers see every
    # chunk.  (Phase B below only touches the per-cell cross-chunk-link group,
    # which carries no such manifest.)
    level_group.rebuild_nonempty_manifests()

    # --- Phase B: cross-target links, decentralized per ccl shard -------
    # Each adjacent target-chunk pair's k2 cell falls in one outer shard; group
    # pairs by shard so every shard is written by exactly one task (the only
    # concurrency-safe partition for the sharded ccl store).  Workers match
    # coincident OUTER-face vertices (sidecars) and write their shard's cells.
    tcc_set = set(target_chunks)
    shard_shape = tuple(
        min(CROSS_CHUNK_LINK_SHARD_AXIS, chunk_grid_shape[i % ndim])
        for i in range(2 * ndim)
    )
    shard_pairs: dict = defaultdict(list)
    for tcc in target_chunks:
        for a in range(ndim):
            nb = tuple(tcc[i] + (1 if i == a else 0) for i in range(ndim))
            if nb not in tcc_set:
                continue
            su = tuple(sorted((tcc, nb)))
            cell = [su[k][ax] - chunk_origin[ax] for k in range(2) for ax in range(ndim)]
            shard = tuple(
                (cell[i] // shard_shape[i]) * shard_shape[i] for i in range(2 * ndim)
            )
            shard_pairs[shard].append((tcc, nb))
    n_cross = 0
    try:
        _tb = _time.perf_counter()
        _progress(f"phase B start: shard_groups={len(shard_pairs)}")
        if shard_pairs:
            payloadsB = []
            for _shard, sps in shard_pairs.items():
                need: set = set()
                for A, B in sps:
                    need.add(A)
                    need.add(B)
                sc_sub = [
                    {"tcc": list(c), "path": sidecar_paths[c]}
                    for c in need
                    if c in sidecar_paths
                ]
                payloadsB.append({
                    "pairs": [(list(A), list(B)) for A, B in sps],
                    "sidecar_paths": sc_sub,
                })
            sharedB = {
                "store_path": str(store_path),
                "target_level": int(target_level),
                "ndim": int(ndim),
                "chunk_grid_shape": list(chunk_grid_shape),
                "chunk_origin": list(chunk_origin),
            }
            for rb in executor(_cross_edge_shard, payloadsB, sharedB):
                n_cross += int(rb.get("n_links", 0))
        _timings["cross_links_phase_b"] = _time.perf_counter() - _tb
        _progress(
            f"phase B done: links={n_cross} dt={_timings['cross_links_phase_b']:.2f}s"
        )
    finally:
        shutil.rmtree(sidecar_tmp_dir, ignore_errors=True)

    # Reconcile the cross-chunk-link family counts after the decentralized
    # per-cell writes above (the core cells layout defers count metadata to a
    # finalize pass; the old kN-array layout self-described so needed none).
    if n_cross:
        finalize_cross_chunk_links(level_group, delta=0)

    _timings["finalize"] = _time.perf_counter() - _tfin
    _timings["total"] = _time.perf_counter() - _t0
    _timings["n_target_chunks"] = len(target_chunks)
    _progress(f"done: total={_timings['total']:.2f}s")

    return {
        "vertex_count": int(total_out_vertices),
        "fragment_count": int(total_fragments),
        "object_count": int(len(present_oids)),
        "objects_kept": int(len(present_oids)),
        "source_objects": n_src,
        "cross_chunk_edges": int(n_cross),
        "method": COARSEN_SKELETON,
        "preserves_object_ids": True,
        "target_chunk_shape": target_chunk_shape,
        "timings": _timings,
    }


# Re-export the ChunkCoords name used in annotations above.
from zarr_vectors.typing import ChunkCoords  # noqa: E402


def build_skeleton_pyramid(
    store_path: str | Path,
    *,
    strides: list[int],
    chunk_scale_factors: list[int | tuple[int, ...]] | None = None,
    sparsity_factors: list[float] | None = None,
    sparsity_strategy: str = "length",
    sparsity_seed: int | None = None,
    attr_agg: str = "max",
    drop_interior_below: int = 0,
    boundary_offset_nm: Sequence[float] | None = None,
    executor: Any = None,
) -> dict[str, Any]:
    """Build a skeleton pyramid by repeated :func:`coarsen_skeleton_level`.

    ``strides[i]`` (keep-every-kth) produces level ``i+1`` from level
    ``i``.  Optional per-level ``chunk_scale_factors`` (default 2 per axis)
    and ``sparsity_factors`` (default 1.0 = keep all) are aligned with
    ``strides``.  ``executor`` (a ``map``-like callable) is threaded into each
    level to coarsen target chunks in parallel; levels stay sequential
    (level ``i+1`` reads ``i``).  Default ``None`` → serial.

    Returns a summary with one entry per produced level.
    """
    n = len(strides)
    if chunk_scale_factors is not None and len(chunk_scale_factors) != n:
        raise ValueError("chunk_scale_factors length must match strides")
    if sparsity_factors is not None and len(sparsity_factors) != n:
        raise ValueError("sparsity_factors length must match strides")
    summaries = []
    for i in range(n):
        csf = chunk_scale_factors[i] if chunk_scale_factors is not None else 2
        spf = sparsity_factors[i] if sparsity_factors is not None else 1.0
        summaries.append(coarsen_skeleton_level(
            store_path, source_level=i, target_level=i + 1,
            stride=int(strides[i]),
            sparsity_factor=float(spf),
            chunk_scale_factor=csf,
            sparsity_strategy=sparsity_strategy,
            sparsity_seed=sparsity_seed,
            attr_agg=attr_agg,
            drop_interior_below=int(drop_interior_below or 0),
            boundary_offset_nm=boundary_offset_nm,
            executor=executor,
        ))
    return {"levels": summaries, "num_levels": n + 1}
