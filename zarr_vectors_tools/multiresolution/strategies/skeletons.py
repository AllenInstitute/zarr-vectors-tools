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

The coarsener merges each object's fragments across all of a target
chunk's source children before re-splitting by actual connectivity, so
a piece can and does span what were several source chunks.  A level-0
cross-chunk link whose two endpoint chunks nest into the *same* target
chunk is read and promoted into an ordinary intra-target edge.  A link
whose endpoints land in *different* target chunks cannot be resolved
locally by one chunk's worker, so it is instead carried forward as an
identity-keyed anchor (source chunk, source vertex) -> (target chunk,
target vertex) and rejoined once every target chunk has been written
(:func:`_cross_edge_shard`, "Phase B"); a geometric coincidence join on
rounded ``(segment_id, position)`` runs afterward as a supplementary
recovery for records the identity join could not place (e.g. an
endpoint whose owning chunk failed to read).  See ``open-items.md`` in
the zarr-vectors-hackathon project for the investigation that found the
geometric join alone silently drops any cross-chunk link whose two
sides are not bit-identical duplicate vertices (the igneous ``.frags``
convention) -- e.g. any precomputed-skeleton source where a boundary is
a phase-split pair of distinct nearby vertices.
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from zarr_vectors.building import rebuild_presence
from zarr_vectors.constants import (
    VERTEX_ATTRIBUTES,
    VERTICES,
)

from zarr_vectors_tools.multiresolution.constants import COARSEN_SKELETON

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
                agg[:] = (
                    np.iinfo(data.dtype).max
                    if np.issubdtype(data.dtype, np.integer) else np.inf
                )
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

    ``stride <= 1`` is the IDENTITY: every vertex is kept.  It used to mean
    "keep anchors only", the most aggressive setting this function has,
    which is the opposite of what a factor of 1 means everywhere else in the
    package (``coarsen_level`` documents 1.0 as "no aggregation") and turned
    a pyramid refresh that could not recover the stride into a level with
    the interior of every chain deleted.

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

    if stride <= 1:
        # The identity, matching every other coarsener's factor of 1.
        keep[:] = True
    else:
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


def _read_skeleton_children(
    src,
    child_ccs: list[tuple[int, ...]],
    *,
    ndim: int,
    attr_names: list[str],
    attr_dtypes: dict[str, np.dtype],
    ccl_cells: list,
    ccl_segments: list | None = None,
    link_spec: dict | None = None,
) -> tuple[list[tuple], list[list]]:
    """One prefetch for everything a target chunk's plan reads from the source.

    Returns ``(children, cell_records)``: ``children`` holds ``(scc, vgroups,
    segs, oids, lgroups, attr_cells)`` for every child chunk with vertices --
    the per-fragment ``segment_id`` / ``object_id`` columns, the branch link
    groups and each per-vertex attribute's groups, any of them None where the
    array or cell is absent -- and ``cell_records`` holds the decoded records
    of each entry of ``ccl_cells``, in order, ``[]`` where a cell could not be
    read.

    Same reason as the polyline coarsener's ``_read_polyline_children``: each
    read resolved its array node afresh, so a target chunk's plan was mostly
    ``zarr.json`` reads (measured on a 1500-skeleton store: 115 of Phase A's
    142 seconds were the reads, 64 of them in ``open``).  ``cached_nodes``
    resolves each node once per task and ``batched_reads`` fetches every cell
    the plan names in one pass.  With ``ccl_segments`` / ``link_spec`` (see
    :func:`zarr_vectors_tools.algorithms._links.link_cell_reads`) the link
    cells are prefetched and decoded exactly, with no per-task segment
    enumeration and no per-cell metadata lookup, as the polyline reader does.
    """
    from zarr_vectors.building import (
        read_chunk_attributes,
        read_chunk_fragment_attributes,
        read_chunk_links,
        read_chunk_vertices,
        read_links_for_tuple,
    )
    from zarr_vectors.constants import FRAGMENT_ATTRIBUTES, LINK_FRAGMENTS, VERTEX_FRAGMENTS
    from zarr_vectors.exceptions import ArrayError

    from zarr_vectors_tools.algorithms._links import (
        chunk_key_str,
        link_cell_prefetch_plan,
        link_prefetch_plan,
        prune_prefetch_plan,
        read_link_cell_records,
    )

    child_keys = [chunk_key_str(c) for c in child_ccs]
    plan: list[tuple[str, list[str]]] = [
        (VERTICES, child_keys),
        (VERTEX_FRAGMENTS, child_keys),
        (f"{FRAGMENT_ATTRIBUTES}/segment_id", child_keys),
        (f"{FRAGMENT_ATTRIBUTES}/object_id", child_keys),
        *((f"{VERTEX_ATTRIBUTES}/{name}", child_keys) for name in attr_names),
    ]
    use_spec = link_spec is not None and ccl_segments is not None
    children: list[tuple] = []
    cell_records: list[list] = []
    with src.cached_nodes():
        if use_spec:
            # The children's intra (branch) link cells and their
            # link_fragments sidecars; the cross-chunk cells are added below,
            # after pruning, since the coordinator listed them.
            plan.append((LINK_FRAGMENTS, child_keys))
            if link_spec["intra_path"] is not None:
                plan.append((link_spec["intra_path"], child_keys))
        else:
            link_chunks = sorted(
                {tuple(int(c) for c in cell[0]) for cell in ccl_cells} | set(child_ccs)
            )
            plan.extend(link_prefetch_plan(src, link_chunks, delta=0))
        # Only cells that exist: on a local filesystem each absent cell would
        # cost a failed open().
        plan = prune_prefetch_plan(src, plan)
        if use_spec:
            plan.extend(link_cell_prefetch_plan(ccl_cells, ccl_segments, link_spec))
        with src.batched_reads(plan):
            for scc in child_ccs:
                try:
                    vgroups = read_chunk_vertices(src, scc, dtype=np.float32, ndim=ndim)
                except ArrayError:
                    continue
                if not vgroups:
                    continue
                try:
                    segs = read_chunk_fragment_attributes(
                        src, "segment_id", scc, dtype=np.uint64,
                    )
                except ArrayError:
                    segs = None
                try:
                    oids = read_chunk_fragment_attributes(
                        src, "object_id", scc, dtype=np.uint64,
                    )
                except ArrayError:
                    oids = None
                try:
                    lgroups = read_chunk_links(src, scc, link_width=2, delta=0)
                except ArrayError:
                    lgroups = None
                attr_cells: dict[str, Any] = {}
                for name in attr_names:
                    try:
                        attr_cells[name] = read_chunk_attributes(
                            src, name, scc, dtype=attr_dtypes[name],
                        )
                    except ArrayError:
                        attr_cells[name] = None
                children.append((scc, vgroups, segs, oids, lgroups, attr_cells))
            for i, cell in enumerate(ccl_cells):
                try:
                    if use_spec:
                        recs = read_link_cell_records(src, cell, ccl_segments[i], link_spec)
                    else:
                        recs = read_links_for_tuple(src, cell, delta=0)
                except Exception:
                    recs = []
                cell_records.append(recs)
    return children, cell_records


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
    ccl_segments: list | None = None,
    link_spec: dict | None = None,
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

    tcc = tuple(int(x) for x in tcc)
    child_ccs = [
        tuple(tcc[a] * scale[a] + d[a] for a in range(ndim))
        for d in product(*[range(scale[a]) for a in range(ndim)])
    ]

    children, cell_records = _read_skeleton_children(
        src, child_ccs, ndim=ndim, attr_names=attr_names, attr_dtypes=attr_dtypes,
        ccl_cells=list(ccl_cells or ()),
        ccl_segments=ccl_segments, link_spec=link_spec,
    )

    vcache: dict = {}
    acache: dict = {name: {} for name in attr_names}
    fragseg: dict = {}
    fragoid: dict = {}
    fraglinks: dict = {}
    child_ranges: dict = {}  # scc -> (starts, ends, fidxs) for chunk-local→fragment
    for scc, vgroups, segs, oids, lgroups, attr_cells in children:
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
                ag = attr_cells.get(name)
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
    child_set = set(child_ranges)
    oid_of_frag: dict = {}
    for _oid, _mems in members_by_oid.items():
        for _m in _mems:
            oid_of_frag[_m] = _oid
    conns_by_oid: dict = defaultdict(list)
    # A link whose OTHER endpoint is outside this target chunk cannot be
    # resolved locally (the neighbouring target chunk is a separate worker's
    # plan, decimated independently and maybe concurrently).  Its own-side
    # endpoint is still captured here -- by SOURCE IDENTITY, i.e. exactly
    # which (source chunk, source vertex) it was -- so Phase B can rejoin the
    # two sides without depending on their post-decimation positions
    # coinciding.  See the module docstring.
    straddle_by_oid: dict = defaultdict(list)
    # ``ccl_cells`` are the source cross-chunk-link cells touching this target's
    # children (enumerated once by the coordinator and bucketed per target);
    # their records were read above, one list per cell.
    for recs in cell_records:
        for rec in recs:
            if len(rec) != 2:
                continue
            (ccA, viA), (ccB, viB) = rec
            ccA = tuple(int(x) for x in ccA)
            ccB = tuple(int(x) for x in ccB)
            a_mine = ccA in child_set
            b_mine = ccB in child_set
            if a_mine and b_mine:
                rA = _resolve(ccA, int(viA))
                rB = _resolve(ccB, int(viB))
                if rA is None or rB is None:
                    continue
                oa = oid_of_frag.get((ccA, rA[0]))
                ob = oid_of_frag.get((ccB, rB[0]))
                if oa is None or oa != ob:
                    continue
                conns_by_oid[oa].append((ccA, rA[0], rA[1], ccB, rB[0], rB[1]))
                continue
            if not (a_mine or b_mine):
                continue  # neither endpoint is ours -- not our record
            # Exactly one endpoint is ours: a cross-TARGET link.  Resolve our
            # own side and remember its source identity for Phase B; the
            # other side is the neighbouring target chunk's job to capture
            # from its own copy of this same cell.
            scc, svi = (ccA, int(viA)) if a_mine else (ccB, int(viB))
            r = _resolve(scc, svi)
            if r is None:
                continue
            oid = oid_of_frag.get((scc, r[0]))
            if oid is None:
                continue
            straddle_by_oid[oid].append((scc, r[0], r[1], svi))

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

        # (d) cross-TARGET link endpoints → force-keep (explicit, not merely
        #     incidental to the geometric on_tgt test below, which depends on
        #     boundary_offset_nm being passed correctly) + identity anchor so
        #     Phase B can rejoin by source identity rather than by geometric
        #     coincidence of the surviving position.
        straddle_ident: dict[int, tuple] = {}  # merged index -> (scc, source_vi)
        forced: set = set()
        for scc, fidx, lidx, svi in straddle_by_oid.get(oid, ()):
            if (scc, fidx) not in goff:
                continue
            mv = goff[(scc, fidx)] + lidx
            forced.add(mv)
            straddle_ident[mv] = (scc, svi)

        # (b) coincident boundary vertices on INTERIOR faces → merge edges;
        #     vertices on OUTER target faces → force-keep for Phase B.
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
            "straddle_ident": straddle_ident,
        })
    return groups, vcache, acache


def _decimate_components(
    comps: list[dict],
    forced_pairs: list[list[tuple[int, int]] | None],
    *,
    stride: int,
    attr_agg: str,
) -> list[dict[str, Any]]:
    """:func:`decimate_skeleton` over many components in one batched pass.

    ``comps`` are :func:`split_components` pieces; ``forced_pairs[i]`` is
    ``None`` or component ``i``'s forced vertices as ``(any, local index)``
    pairs.  Entry ``i`` of the result equals ``decimate_skeleton(
    comps[i]["positions"], comps[i]["edges"], stride=stride, forced_keep=
    [local indices], attributes=comps[i]["attributes"], attr_agg=attr_agg)``:
    positions, edge rows and their order, aggregated attributes and
    ``kept_source_indices``.

    The components are concatenated into one forest -- a global ``parent``
    array filled the way :func:`_build_rooted_tree` fills each piece's -- and
    run through the vectorised ``_forest.decimate_keep`` / ``_forest.collapse``,
    which are pinned to the scalar reference.  Splitting back is exact: kept
    vertices come out in ascending global order, so each component's
    survivors and the edges they own are one contiguous run, and each
    aggregate is a ``ufunc.at`` over slots no other component shares, applied
    in the same row order as the per-component call (so ``mean`` sums are
    bit-identical).  Components whose positions or attributes differ in dtype
    or trailing shape go in separate batches, since concatenating them would
    promote.
    """
    from zarr_vectors_tools.multiresolution import _forest

    out: list[dict[str, Any]] = [{} for _ in comps]
    batches: dict[tuple, list[int]] = defaultdict(list)
    for i, comp in enumerate(comps):
        pos = np.asarray(comp["positions"])
        if len(pos) == 0:
            pairs = forced_pairs[i]
            out[i] = decimate_skeleton(
                pos, comp["edges"], stride=stride,
                forced_keep=None if pairs is None else [ci for _mv, ci in pairs],
                attributes=comp["attributes"], attr_agg=attr_agg,
            )
            continue
        sig = (
            pos.dtype.str,
            pos.shape[1:],
            tuple(
                (name, a.dtype.str, a.shape[1:])
                for name, a in (
                    (name, np.asarray(v)) for name, v in comp["attributes"].items()
                )
            ),
        )
        batches[sig].append(i)

    for (_pdt, _ptail, attr_sig), idxs in batches.items():
        sizes_list = [len(comps[i]["positions"]) for i in idxs]
        offs = np.zeros(len(idxs) + 1, dtype=np.int64)
        np.cumsum(sizes_list, out=offs[1:])
        offs_list = offs.tolist()
        n = offs_list[-1]
        positions = np.concatenate(
            [np.asarray(comps[i]["positions"]) for i in idxs], axis=0,
        )
        # One fancy assignment over every component's [child, parent] rows,
        # each shifted to its component's offset: the rows of one component
        # keep their relative order, so a repeated child resolves as in
        # _build_rooted_tree.
        edge_blocks = [
            np.asarray(comps[i]["edges"], dtype=np.int64).reshape(-1, 2) for i in idxs
        ]
        edge_counts = [len(e) for e in edge_blocks]
        parent = np.full(n, -1, dtype=np.int64)
        if sum(edge_counts):
            all_edges = np.concatenate(edge_blocks, axis=0)
            shift = np.repeat(offs[:-1], edge_counts)
            parent[all_edges[:, 0] + shift] = all_edges[:, 1] + shift
        forced = np.zeros(n, dtype=bool)
        forced_idx: list[int] = []
        for j, i in enumerate(idxs):
            pairs = forced_pairs[i]
            if pairs is None:
                continue
            o, size = offs_list[j], sizes_list[j]
            forced_idx.extend(o + ci for _mv, ci in pairs if 0 <= ci < size)
        if forced_idx:
            forced[np.asarray(forced_idx, dtype=np.int64)] = True
        attributes = {
            name: np.concatenate(
                [np.asarray(comps[i]["attributes"][name]) for i in idxs], axis=0,
            )
            for name, _dt, _tail in attr_sig
        }
        keep = _forest.decimate_keep(parent, stride=stride, forced=forced)
        res = _forest.collapse(parent, keep, attributes, attr_agg)
        kept = res["kept"]
        edges = res["edges"]
        kept_bounds = np.searchsorted(kept, offs)
        edge_bounds = np.searchsorted(edges[:, 0], kept_bounds)
        # Rebase to component-local indices once, then hand out slices.
        comp_ids = np.arange(len(idxs), dtype=np.int64)
        kept_comp = np.repeat(comp_ids, np.diff(kept_bounds))
        local_kept = kept - offs[kept_comp]
        local_edges = edges - kept_bounds[np.repeat(comp_ids, np.diff(edge_bounds))][:, None]
        kept_positions = positions[kept]
        aggs = list(res["attributes"].items())
        kb = kept_bounds.tolist()
        eb = edge_bounds.tolist()
        for j, i in enumerate(idxs):
            k0, k1 = kb[j], kb[j + 1]
            out[i] = {
                "positions": kept_positions[k0:k1],
                "edges": local_edges[eb[j]:eb[j + 1]],
                "attributes": {name: agg[k0:k1] for name, agg in aggs},
                "kept_source_indices": local_kept[k0:k1],
            }
    return out


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
    from zarr_vectors.building import get_resolution_level, open_store
    from zarr_vectors.types.skeletons import write_skeleton_chunk

    from zarr_vectors_tools.multiresolution.skeleton_graph import split_components

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
        ccl_segments=payload.get("ccl_segments"),
        link_spec=shared.get("link_spec"),
    )
    input_fragments = int(len(vcache))
    input_vertices = int(sum(len(v) for v in vcache.values()))
    input_objects = int(len(groups))

    # Pass 1: merge each object group's fragments and split it into rooted
    # components.  A group with forced (outer-face) vertices records, per
    # component, its forced vertices as ``(merged index, local index)`` in
    # ``forced_set`` iteration order -- the order anchor tags are handed out
    # in below.
    plans: list[tuple[dict, list, list]] = []
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
        forced_pairs: list[list[tuple[int, int]] | None] = [None] * len(comps)
        if forced_set:
            n_merged = len(mverts)
            comp_of = np.empty(n_merged, dtype=np.int64)
            local_of = np.empty(n_merged, dtype=np.int64)
            for ci, comp in enumerate(comps):
                vids = np.asarray(comp["vertex_ids"], dtype=np.int64)
                comp_of[vids] = ci
                local_of[vids] = np.arange(len(vids), dtype=np.int64)
            buckets: list[list[tuple[int, int]]] = [[] for _ in comps]
            for mv in forced_set:
                if 0 <= mv < n_merged:
                    buckets[int(comp_of[mv])].append((mv, int(local_of[mv])))
            forced_pairs = list(buckets)
        plans.append((g, comps, forced_pairs))

    # Pass 2: decimate every component of this target chunk as one forest.
    simps = _decimate_components(
        [comp for _g, comps, _fp in plans for comp in comps],
        [pairs for _g, _c, fps in plans for pairs in fps],
        stride=stride, attr_agg=attr_agg,
    )

    pieces: list = []
    total_out_vertices = 0
    anchor_meta: dict = {}   # tag -> (segment_id, coord-tuple) for outer-face verts
    ident_meta: dict = {}    # tag -> (source_chunk, source_vertex_index), cross-target only
    tagc = 0
    k = 0
    n_straddle_lost = 0      # a forced straddle vertex that did not survive decimation
    for g, comps, forced_pairs in plans:
        straddle_ident = g.get("straddle_ident") or {}
        for comp, pairs in zip(comps, forced_pairs):
            simp = simps[k]
            k += 1
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
            if pairs is not None:
                kept_pos = {
                    int(c): i
                    for i, c in enumerate(simp["kept_source_indices"].tolist())
                }
                anchors = {}
                for mv, ci in pairs:
                    sl = kept_pos.get(ci)
                    if sl is None:
                        if mv in straddle_ident:
                            # A cross-target link endpoint that was force-kept
                            # (see _build_local_plan "(d)") must survive by
                            # construction; landing here means force-keep
                            # itself failed to reach the decimator for this
                            # vertex -- a regression, not an expected miss.
                            n_straddle_lost += 1
                        continue
                    anchors[tagc] = sl
                    anchor_meta[tagc] = (
                        int(g["segment_id"]),
                        tuple(int(x) for x in np.rint(comp["positions"][ci]).astype(np.int64)),
                    )
                    ident = straddle_ident.get(mv)
                    if ident is not None:
                        ident_meta[tagc] = ident
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

    # record_presence=False: this is a per-target-chunk worker that may run in
    # a separate PROCESS.  ``nonempty_chunks`` is array-wide state, so stamping
    # it per chunk is a read-modify-write that concurrent workers collide on (a
    # Windows atomic-rename hard-fail).  The coordinator's
    # rebuild_nonempty_manifests pass re-derives every manifest after Phase A.
    recs, alocs = write_skeleton_chunk(
        level_group, tcc, pieces, attr_dtypes=attr_dtypes,
        record_presence=False,
    )
    # Object-index rows ``(oid, fragment_idx)`` and cross-target anchor rows
    # ``(segment_id, *coord, vi)`` travel back inline as compact int64
    # arrays.  Both used to go through temp files -- the rows as one .npy
    # per (target chunk, oid shard), the anchors as one .npy per chunk --
    # read back one file at a time by Phases C and B, which on a level of
    # thousands of target chunks was hundreds of thousands of tiny files.
    # Both are O(fragments) at 16-40 bytes a row, the same order the
    # polyline coarsener already carries inline.
    oid_rows = (
        np.asarray([(int(r[0]), int(r[2])) for r in recs], dtype=np.int64).reshape(-1, 2)
        if recs else None
    )
    sc_rows = []
    for tag, (seg, coord) in anchor_meta.items():
        loc = alocs.get(tag)
        if loc is not None:
            sc_rows.append((seg, *coord, int(loc[1])))  # [segment_id, *coord, vi]
    anchors = (
        np.asarray(sc_rows, dtype=np.int64).reshape(-1, 2 + ndim) if sc_rows else None
    )
    # Identity sidecar for Phase B's cross-target rejoin: one row per
    # cross-target link endpoint that survived decimation, keyed by the exact
    # SOURCE identity the level-0 link record named (not by position) --
    # ``[*source_chunk_coords, source_vertex_index, new_target_vertex_index]``.
    id_rows = []
    for tag, (src_scc, src_vi) in ident_meta.items():
        loc = alocs.get(tag)
        if loc is not None:
            id_rows.append((*src_scc, int(src_vi), int(loc[1])))
    ident_anchors = (
        np.asarray(id_rows, dtype=np.int64).reshape(-1, ndim + 2) if id_rows else None
    )
    return {
        "tcc": tcc,
        "input_fragments": input_fragments,
        "input_vertices": input_vertices,
        "input_objects": input_objects,
        "fragment_count": int(len(recs)),
        "oid_rows": oid_rows,
        "anchors": anchors,
        "ident_anchors": ident_anchors,
        "n_straddle_lost": n_straddle_lost,
        "vertex_count": int(total_out_vertices),
        "dropped_oids": np.asarray(dropped_oids, dtype=np.int64),
    }


def _anchor_join_table(
    arr: npt.NDArray[np.int64] | None, ndim: int,
) -> tuple | None:
    """Sorted join table for one target chunk's anchor rows, or None if empty.

    Rows are ``(segment_id, *coord, vi)``; the key is ``(segment_id, *coord)``
    and the FIRST row carrying a key supplies its ``vi``.  Returns
    ``(sorted_keys, sorted_vi, first_keys, first_vi)``: the distinct keys in
    sorted order (for ``searchsorted``) and again in first-occurrence order
    (for probing).  Keys are compared as opaque fixed-width byte blobs, which
    is all an equi-join needs.
    """
    if arr is None:
        return None
    rows = np.ascontiguousarray(np.asarray(arr, dtype=np.int64).reshape(-1, 2 + ndim))
    if len(rows) == 0:
        return None
    width = 1 + ndim
    keys = np.ascontiguousarray(rows[:, :width])
    blobs = keys.view(np.dtype((np.void, keys.dtype.itemsize * width))).reshape(-1)
    order = np.argsort(blobs, kind="stable")
    sorted_blobs = blobs[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = sorted_blobs[1:] != sorted_blobs[:-1]
    # A stable sort keeps a key's rows in input order, so the head of each
    # run is that key's first occurrence.
    first_rows = order[first]
    by_occurrence = np.sort(first_rows)
    return (
        sorted_blobs[first],
        rows[first_rows, 1 + ndim],
        blobs[by_occurrence],
        rows[by_occurrence, 1 + ndim],
    )


def _cross_edge_shard(payload: dict, shared: dict | None = None) -> dict:
    """Phase B worker: write the cross-target-chunk links for ONE task shard.

    Each task owns a disjoint set of adjacent target-chunk pairs (the
    coordinator partitions pairs by shard, so writers never collide).  For
    each pair it runs two recovery passes and writes their union via
    :func:`write_link_cells`:

    1. **Identity join** (tried first, exact).  ``payload["straddle"]``
       carries, per pair, the level-0 cross-chunk-link *cells* whose two
       endpoint chunks nest into this pair's two target chunks (the
       coordinator identified these once from the same enumeration Phase A
       used).  Each cell is re-decoded from the SOURCE level via
       :func:`~zarr_vectors_tools.algorithms._links.read_link_cell_records`
       — the exact same primitive Phase A used — and its two endpoints are
       looked up in the two target chunks' identity sidecars (``(source
       chunk, source vertex) -> new target vertex``, built in
       :func:`_coarsen_target_chunk`).  A hit needs no assumption about
       where the survivors ended up geometrically: it is the original
       level-0 record, rejoined by construction.
    2. **Geometric coincidence** (unchanged, kept as a supplement).  Matches
       coincident ``(segment_id, coord)`` OUTER-face vertices — the same
       igneous-style boundary coincidence used at L0 — from the two chunks'
       position sidecars.  This is the only recovery for a record the
       identity pass could not place (e.g. one endpoint's owning chunk
       failed to read), but on its own it silently drops any cross-chunk
       link whose two sides are not bit-identical duplicate vertices (a
       phase-split boundary, as precomputed-skeleton sources use) —
       see the module docstring.

    Per pair, an identity hit is recorded before the coincidence pass runs
    over the same pair, and the coincidence pass skips any ``(a, b)`` vertex
    pair identity already produced — the union, not a duplicate.

    A record's cell is ``(offsets_segment, source_chunk)``, which maps 1:1
    to its endpoint pair, so disjoint pairs are disjoint cells and the
    partition is race-safe for any grouping.
    """
    from zarr_vectors.building import get_resolution_level, open_store, write_link_cells

    from zarr_vectors_tools.algorithms._links import read_link_cell_records

    shared = shared or {}
    ndim = shared["ndim"]
    # Anchor rows arrive inline (see _coarsen_target_chunk), keyed by target
    # chunk; only the chunks this task's pairs touch are carried.
    sidecar_arrays: dict[tuple[int, ...], npt.NDArray[np.int64]] = {
        tuple(int(x) for x in e["tcc"]): np.asarray(e["arr"], dtype=np.int64)
        for e in payload.get("anchors", [])
    }
    ident_sidecar: dict[tuple[int, ...], npt.NDArray[np.int64]] = {
        tuple(int(x) for x in e["tcc"]): np.asarray(e["arr"], dtype=np.int64)
        for e in payload.get("ident_anchors", [])
    }
    pairs = payload["pairs"]
    straddle = payload.get("straddle") or [None] * len(pairs)
    table_cache: dict[tuple[int, ...], tuple | None] = {}
    ident_cache: dict[tuple[int, ...], dict] = {}

    def _table(cc: tuple[int, ...]) -> tuple | None:
        if cc not in table_cache:
            table_cache[cc] = _anchor_join_table(sidecar_arrays.get(cc), ndim)
        return table_cache[cc]

    def _ident_table(cc: tuple[int, ...]) -> dict:
        # {(source_chunk, source_vertex_index): new_target_vertex_index}
        if cc not in ident_cache:
            d: dict = {}
            arr = ident_sidecar.get(cc)
            if arr is not None and len(arr):
                for row in np.asarray(arr, dtype=np.int64).tolist():
                    d[(tuple(row[:ndim]), int(row[ndim]))] = int(row[ndim + 1])
            ident_cache[cc] = d
        return ident_cache[cc]

    root = None  # opened lazily -- only shards with straddle or write work need it
    src = None   # the store's source level, a view onto the same `root`
    link_spec = shared.get("link_spec")
    n_identity = 0
    links: list = []
    for (A_, B_), sinfo in zip(pairs, straddle):
        A = tuple(int(x) for x in A_)
        B = tuple(int(x) for x in B_)
        seen: set[tuple[int, int]] = set()  # (a, b) vertex pairs already emitted

        # --- pass 1: identity join ---
        if sinfo and sinfo.get("cells"):
            if src is None:
                root = open_store(shared["store_path"], mode="r+")
                src = get_resolution_level(root, shared["source_level"])
            id_a = _ident_table(A)
            id_b = _ident_table(B)
            for cell_raw, seg in zip(sinfo["cells"], sinfo["segments"]):
                cell = tuple(tuple(int(x) for x in c) for c in cell_raw)
                try:
                    recs = read_link_cell_records(src, cell, int(seg), link_spec)
                except Exception:
                    continue
                for (ccA, viA), (ccB, viB) in recs:
                    keyA = (tuple(int(x) for x in ccA), int(viA))
                    keyB = (tuple(int(x) for x in ccB), int(viB))
                    vi_a = id_a.get(keyA)
                    vi_b = id_b.get(keyB)
                    if vi_a is None or vi_b is None:
                        # try the other assignment -- the record's endpoint
                        # order need not match which side is A vs B here.
                        vi_a = id_a.get(keyB)
                        vi_b = id_b.get(keyA)
                    if vi_a is None or vi_b is None:
                        continue
                    if (vi_a, vi_b) in seen:
                        continue
                    seen.add((vi_a, vi_b))
                    links.append([(A, vi_a), (B, vi_b)])
                    n_identity += 1

        # --- pass 2: geometric coincidence, supplementary ---
        ta = _table(A)
        tb = _table(B)
        if ta is None or tb is None:
            continue
        # Probe with the map holding fewer distinct keys, in the order those
        # keys first occur in its anchor rows (ties probe with A) -- the
        # iteration order of the per-chunk dict this join replaced, so the
        # records come out in the same order.
        a_probes = len(ta[0]) <= len(tb[0])
        probe, target = (ta, tb) if a_probes else (tb, ta)
        t_keys, t_vi, p_keys, p_vi = target[0], target[1], probe[2], probe[3]
        pos = np.searchsorted(t_keys, p_keys)
        in_range = pos < len(t_keys)
        pos = np.where(in_range, pos, 0)
        hit = in_range & (t_keys[pos] == p_keys)
        if not hit.any():
            continue
        own = p_vi[hit].tolist()
        other = t_vi[pos[hit]].tolist()
        via, vib = (own, other) if a_probes else (other, own)
        for a, b in zip(via, vib):
            if (a, b) in seen:
                continue
            seen.add((a, b))
            links.append([(A, a), (B, b)])
    if links:
        if root is None:
            root = open_store(shared["store_path"], mode="r+")
        level_group = get_resolution_level(root, shared["target_level"])
        # Writes only the cells these records touch and does NOT maintain
        # the family-wide counts; the coordinator's finalize_links pass
        # reconciles them once every worker is done.
        #
        # directed=True matches the level-0 skeleton family and drops the
        # perm_idx column every row would otherwise carry.  It is safe
        # because the pairs are built as (tcc, tcc + e_a) — always
        # (lower, higher) — so canonical sorting is a no-op on them and
        # placement is identical either way.  It means "endpoint order is
        # (lower, higher) and stable", NOT "endpoint order is data": these
        # matches have no parent→child direction.
        write_link_cells(
            level_group, links, ndim, delta=0, link_width=2, directed=True,
        )
    return {
        "n_links": len(links),
        "n_identity": n_identity,
        "n_coincident": len(links) - n_identity,
    }


def _reduce_object_index_shard(payload: dict, shared: dict | None = None) -> dict:
    """Reduce one OID shard's rows into encoded manifest blobs.

    ``payload["rows"]`` is one int64 array ``(oid, fragment_idx, *target
    chunk)`` holding every fragment of this shard's objects, cut from the
    Phase A results by the coordinator (see
    :func:`zarr_vectors_tools.multiresolution.object_index.shard_rows_by_object`).
    This worker groups the rows by OID and returns encoded v0.6 manifest
    blobs for just those OIDs.
    """
    from zarr_vectors.building import encode_object_manifest_blocks

    shared = shared or {}
    sid_ndim = int(shared["sid_ndim"])
    width = 2 + sid_ndim

    rows = np.asarray(
        payload.get("rows", np.zeros((0, width), np.int64)), dtype=np.int64,
    ).reshape(-1, width)
    manifests: dict[int, list[tuple[tuple[int, ...], int]]] = defaultdict(list)
    for row in rows.tolist():
        manifests[row[0]].append((tuple(row[2:2 + sid_ndim]), row[1]))

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
    compressor: Any = None,
    executor: Any = None,
    progress: bool = False,
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
        progress: Print ``[coarsen Lx->Ly]`` phase lines as the level is
            built.  Off by default; the per-phase durations are returned
            under ``timings`` either way.

    Returns:
        Summary dict.
    """
    # Imports are local to avoid a module-load cycle
    # (coarsen.py imports this module).
    from zarr_vectors.building import (
        OBJECT_INDEX,
        OBJECT_INDEX_LAYOUT_V1,
        LevelMetadata,
        create_attribute_array,
        create_fragment_attribute_array,
        create_links_array,
        create_links_family,
        create_object_attributes_array,
        create_object_index_array,
        create_resolution_level,
        create_vertices_array,
        finalize_links,
        get_level_chunk_shape,
        get_resolution_level,
        list_chunk_keys,
        open_store,
        read_all_object_manifests,
        read_chunk_fragment_attributes,
        read_level_metadata,
        read_object_attributes,
        read_root_metadata,
        upsert_level_transform,
        write_object_attributes,
        write_object_manifests,
    )
    from zarr_vectors.exceptions import ArrayError
    from zarr_vectors.types.skeletons import get_coordinate_offset

    from zarr_vectors_tools.multiresolution.constants import (
        CROSS_LINK_TASK_SHARD_AXIS,
    )
    from zarr_vectors_tools.multiresolution.object_index import shard_rows_by_object
    from zarr_vectors_tools.multiresolution.object_selection import apply_sparsity

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
        if not progress:
            return
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
        # `.children()`, NOT bare iteration. `Group.__iter__` yields sub-GROUPS
        # only (it is `sorted(self._zarr.group_keys())`), while each per-vertex
        # attribute is an ARRAY — so `for name in src[VERTEX_ATTRIBUTES]`
        # silently yielded nothing, `attr_names` stayed empty, and every coarse
        # level was written with no `vertex_attributes/` at all. Nothing
        # errored: the store just lost `radius`/`compartment` above level 0,
        # and the reader zero-fills the missing arrays, so `prop_radius()`
        # evaluated to 0.0 at every level the camera actually uses.
        #
        # `Group.children()` exists precisely for this and says so in its
        # docstring; `strategies/polylines.py` already uses it here. This is
        # the only remaining site that did not.
        for name in src[VERTEX_ATTRIBUTES].children():
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
            # Cumulative per level: fraction of the surviving pool, not of
            # the original count.  See apply_sparsity's `relative_to`.
            relative_to="alive",
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
    # A chunk array's codec pipeline is fixed when it is created, so the
    # compressor only has to be active around the create_* calls — every later
    # per-cell write (Phase A/B workers included) encodes to match.  The block
    # must CLOSE before any worker dispatch: batched_writes defers array metas
    # to its flush, and a worker reading an unflushed meta would see nothing.
    _codec_ctx = (
        level_group.batched_writes(compressor=compressor)
        if compressor else nullcontext()
    )
    with _codec_ctx:
        create_vertices_array(level_group, dtype="float32")
        # directed=True is family-wide at (level, delta=0) and cannot be
        # flipped later, so the intra array created here and Phase B's cross
        # cells must agree.  True matches the level-0 skeleton family
        # (types.skeletons stamps it so parent->child order survives) and
        # keeps a whole pyramid on one policy.
        create_links_array(
            level_group, link_width=2, delta=0, sid_ndim=ndim, directed=True,
        )
        create_object_index_array(level_group)
        create_fragment_attribute_array(
            level_group, "segment_id", dtype="uint64",
        )
        create_fragment_attribute_array(
            level_group, "object_id", dtype="uint64",
        )
        for name in attr_names:
            create_attribute_array(
                level_group, name, dtype=str(attr_dtypes[name]),
            )

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
    # Fix the family policy up front so the decentralized Phase B workers
    # agree on it rather than racing to establish it.  No offsets array is
    # materialised here: which ones exist depends on where the records land.
    create_links_family(
        level_group, delta=0, link_width=2, sid_ndim=ndim, directed=True,
    )

    # --- Phase A: decimate each target chunk (workers self-plan locally) ---
    boundary_off = (list(boundary_offset_nm)
                    if boundary_offset_nm is not None else [0.0] * ndim)
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
    }
    # Enumerate the source cross-chunk-link cells ONCE and bucket each cell to
    # the target chunk(s) that own its endpoint chunks (chunk // scale). Replaces
    # a per-target-per-child cell scan (O(target_chunks × children × cells))
    # with a single O(cells) pass; workers then read only their bucket's cells.
    # The same pass records which offsets array each cell lives in; the
    # arrays' decode parameters travel once in ``sharedA``.
    from zarr_vectors_tools.algorithms._links import link_cell_reads

    link_cells, link_segments, link_spec = link_cell_reads(src, delta=0)
    sharedA["link_spec"] = link_spec
    cells_by_target: dict[tuple[int, ...], list] = defaultdict(list)
    segs_by_target: dict[tuple[int, ...], list] = defaultdict(list)
    # A cell whose two endpoint chunks nest into two DIFFERENT target chunks
    # cannot be resolved by either chunk's Phase A task alone (see
    # _build_local_plan); recorded here, keyed by the sorted target-chunk
    # pair, so Phase B can re-read it against the source level and rejoin by
    # identity rather than by geometric coincidence.  Link width is always 2
    # (delta=0 skeleton family), so ``cell`` names exactly two chunks and
    # ``seen_t`` has at most two distinct targets.
    straddle_cells_by_pair: dict[tuple, list] = defaultdict(list)
    for cell, seg in zip(link_cells, link_segments):
        seen_t: set = set()
        for c in cell:
            t = tuple(int(c[a]) // scale[a] for a in range(ndim))
            if t in seen_t:
                continue
            seen_t.add(t)
            cells_by_target[t].append(cell)
            segs_by_target[t].append(seg)
        if len(seen_t) == 2:
            straddle_cells_by_pair[tuple(sorted(seen_t))].append((cell, seg))
    payloadsA = [
        {
            "tcc": list(tcc),
            "ccl_cells": cells_by_target.get(tuple(tcc), []),
            "ccl_segments": segs_by_target.get(tuple(tcc), []),
        }
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
    sidecar_arrays: dict[tuple[int, ...], np.ndarray] = {}
    ident_sidecar_arrays: dict[tuple[int, ...], np.ndarray] = {}
    total_straddle_lost = 0
    # Object-index rows from every target chunk, suffixed with the chunk they
    # came from so the reducer can name the fragment; see _coarsen_target_chunk.
    row_blocks: list[np.ndarray] = []
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
        anchors = res.get("anchors")
        if anchors is not None and len(anchors):
            sidecar_arrays[tcc] = anchors
        ident_anchors = res.get("ident_anchors")
        if ident_anchors is not None and len(ident_anchors):
            ident_sidecar_arrays[tcc] = ident_anchors
        total_straddle_lost += int(res.get("n_straddle_lost", 0) or 0)
        rows = res.get("oid_rows")
        if rows is not None and len(rows):
            tcc_cols = np.broadcast_to(np.asarray(tcc, dtype=np.int64), (len(rows), ndim))
            row_blocks.append(np.concatenate([rows, tcc_cols], axis=1))
    # Release phase-A result payloads as soon as we've compacted what we need.
    resultsA = []
    if total_straddle_lost:
        # A cross-target link endpoint was force-kept ("(d)" in
        # _build_local_plan) yet did not survive decimation -- should be
        # unreachable; force-keep is unconditional on such vertices. Not
        # raised: a level with this defect is still a usable, if imperfect,
        # pyramid, and the caller can inspect ``cross_chunk_edges_lost_force_keep``.
        _progress(
            f"WARNING: {total_straddle_lost} cross-target link endpoint(s) "
            "were force-kept but absent after decimation"
        )

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
    # Shards are contiguous object-id ranges cut from the gathered rows, so
    # each task owns whole objects and there is no spill directory to
    # populate, scan and remove.
    empty_blob = b"\x00\x00\x00\x00"  # encode_object_manifest_blocks([], sid_ndim)
    manifest_blobs: list[bytes] = [empty_blob] * int(n_src)
    _tc = _time.perf_counter()
    all_rows = (
        np.concatenate(row_blocks, axis=0) if row_blocks
        else np.zeros((0, 2 + ndim), dtype=np.int64)
    )
    row_blocks = []
    payloadsC = [{"rows": shard} for shard in shard_rows_by_object(all_rows)]
    _progress(f"phase C start: shard_tasks={len(payloadsC)}")
    sharedC = {"sid_ndim": int(ndim)}
    if payloadsC:
        for rc in executor(_reduce_object_index_shard, payloadsC, sharedC):
            oids = np.asarray(rc.get("oid", np.zeros((0,), np.int64)), np.int64)
            blobs = pickle.loads(rc.get("blobs", b"")) if len(oids) else []
            for i, oid in enumerate(oids.tolist()):
                if 0 <= int(oid) < int(n_src):
                    manifest_blobs[int(oid)] = blobs[i]
    write_object_manifests(level_group, manifest_blobs)
    level_group.write_array_meta(OBJECT_INDEX, {
        "zv_array": "object_index",
        "num_objects": int(n_src),
        "sid_ndim": int(ndim),
        "layout": OBJECT_INDEX_LAYOUT_V1,
    })
    _timings["reduce_phase_c"] = _time.perf_counter() - _tc
    _progress(f"phase C done: dt={_timings['reduce_phase_c']:.2f}s")

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
    # chunk.  Phase B's cross-link cells are NOT covered here — under the merged
    # layout they are ordinary chunk-grid arrays that carry (and race on) the
    # same manifest, so their rebuild belongs to the ``finalize_links`` call
    # after Phase B, which re-derives it per offsets segment.

    rebuild_presence(level_group)

    # --- Phase B: cross-target links, decentralized per ccl shard -------
    # Each adjacent target-chunk pair's k2 cell falls in one outer shard; group
    # pairs by shard so every shard is written by exactly one task (the only
    # concurrency-safe partition for the sharded ccl store).  Workers match
    # coincident OUTER-face vertices (sidecars) and write their shard's cells.
    tcc_set = set(target_chunks)
    shard_shape = tuple(
        min(CROSS_LINK_TASK_SHARD_AXIS, chunk_grid_shape[i % ndim])
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
    n_cross_identity = 0
    n_cross_coincident = 0
    _tb = _time.perf_counter()
    _progress(f"phase B start: shard_groups={len(shard_pairs)}")
    if shard_pairs:
        payloadsB = []
        for _shard, sps in shard_pairs.items():
            need: set = set()
            for A, B in sps:
                need.add(A)
                need.add(B)
            # Forward each needed chunk's anchor array inline; only the
            # chunks this shard's pairs touch are carried, so a worker never
            # receives the whole level's anchors.
            sc_sub = [
                {"tcc": list(c), "arr": sidecar_arrays[c]}
                for c in need
                if c in sidecar_arrays
            ]
            id_sub = [
                {"tcc": list(c), "arr": ident_sidecar_arrays[c]}
                for c in need
                if c in ident_sidecar_arrays
            ]
            # Straddling level-0 cells for each pair, aligned 1:1 with
            # "pairs" below (same iteration over ``sps``) so the worker can
            # zip them without any dict lookup of its own.
            straddle_sub = [
                {
                    "cells": [
                        [list(c) for c in cell]
                        for cell, _seg in straddle_cells_by_pair.get(
                            tuple(sorted((A, B))), (),
                        )
                    ],
                    "segments": [
                        seg for _cell, seg in straddle_cells_by_pair.get(
                            tuple(sorted((A, B))), (),
                        )
                    ],
                }
                for A, B in sps
            ]
            payloadsB.append({
                "pairs": [(list(A), list(B)) for A, B in sps],
                "anchors": sc_sub,
                "ident_anchors": id_sub,
                "straddle": straddle_sub,
            })
        sharedB = {
            "store_path": str(store_path),
            "source_level": int(source_level),
            "target_level": int(target_level),
            "ndim": int(ndim),
            "chunk_grid_shape": list(chunk_grid_shape),
            "chunk_origin": list(chunk_origin),
            "link_spec": link_spec,
        }
        for rb in executor(_cross_edge_shard, payloadsB, sharedB):
            n_cross += int(rb.get("n_links", 0))
            n_cross_identity += int(rb.get("n_identity", 0))
            n_cross_coincident += int(rb.get("n_coincident", 0))
    _timings["cross_links_phase_b"] = _time.perf_counter() - _tb
    _progress(
        f"phase B done: links={n_cross} "
        f"(identity={n_cross_identity} coincident={n_cross_coincident}) "
        f"dt={_timings['cross_links_phase_b']:.2f}s"
    )

    # Reconcile the whole links/0 family after the decentralized per-cell
    # writes above: recompute num_links and re-derive each offsets array's
    # nonempty_chunks from the store listing.  Unconditional, not gated on
    # n_cross: the family also holds Phase A's intra links, which carry no
    # counts of their own, so a level with zero cross links would otherwise
    # get no num_links stamp at all.  Must be the last writer of family
    # meta — a later create_links_* would drop the counts it merges in.
    finalize_links(level_group, delta=0)

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
        # Split by recovery path -- see _cross_edge_shard.  Before this fix
        # every cross-target link went through the coincidence path alone
        # (identity is always 0), which is what silently dropped phase-split
        # boundaries; identity/coincident here is how to tell whether a given
        # store's crossings were actually recovered by construction versus by
        # accident.
        "cross_chunk_edges_identity": int(n_cross_identity),
        "cross_chunk_edges_coincident": int(n_cross_coincident),
        # A force-kept cross-target endpoint that vanished before the anchor
        # tag could resolve it -- should be 0; see the gather-loop warning.
        "cross_chunk_edges_lost_force_keep": int(total_straddle_lost),
        "method": COARSEN_SKELETON,
        "preserves_object_ids": True,
        "target_chunk_shape": target_chunk_shape,
        "timings": _timings,
    }


# Re-export the ChunkCoords name used in annotations above.
from zarr_vectors.typing import ChunkCoords  # noqa: E402,F401


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
    compressor: Any = None,
    executor: Any = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Build a skeleton pyramid by repeated :func:`coarsen_skeleton_level`.

    ``strides[i]`` (keep-every-kth) produces level ``i+1`` from level
    ``i``.  Optional per-level ``chunk_scale_factors`` (default 2 per axis)
    and ``sparsity_factors`` (default 1.0 = keep all) are aligned with
    ``strides``.  ``executor`` (a ``map``-like callable) is threaded into each
    level to coarsen target chunks in parallel; levels stay sequential
    (level ``i+1`` reads ``i``).  Default ``None`` → serial.  ``progress``
    is forwarded to every level.

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
            # Advance the seed per level so "random" sparsity differs each
            # level (see build_pyramid).
            sparsity_seed=(
                None if sparsity_seed is None else int(sparsity_seed) + i
            ),
            attr_agg=attr_agg,
            drop_interior_below=int(drop_interior_below or 0),
            boundary_offset_nm=boundary_offset_nm,
            # Each level's codec is fixed at its own array creation, so the
            # compressor is forwarded per level.
            compressor=compressor,
            executor=executor,
            progress=progress,
        ))
    return {"levels": summaries, "num_levels": n + 1}
