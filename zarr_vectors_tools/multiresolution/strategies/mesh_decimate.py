"""Quadric-error edge collapse for meshes, in batched rounds.

WHY NOT CLUSTERING
------------------
``strategies.meshes.coarsen_mesh_level`` bins vertices and DELETES any face
with two corners in one bin. Faces vanish faster than vertices, so the level
fills with dust: measured on the 1720-neuron MICrONS store, 6.4 / 34.0 / 66.2 %
of vertices at levels 1-3 are referenced by no face at all, and the
vertex/face ratio climbs from 0.508 to 3.30 where a closed triangle mesh sits
at ~0.5. Neuron axons are ~450 nm tubes against a ~93 nm edge length, so once
the bin reaches the tube diameter both walls fall in one bin and the tube is
erased rather than simplified.

An edge COLLAPSE cannot do that. It merges two vertices into one and removes
only the faces that degenerate as a result; every surviving vertex keeps the
faces around it. Orphans are impossible by construction rather than something
to clean up afterwards, and the tube thins instead of vanishing because the
link condition below refuses the collapse that would pinch it shut.

WHY ROUNDS AND NOT A PRIORITY QUEUE
-----------------------------------
Textbook QEM pops one edge at a time. The test cell is 25.2M faces, so a
target of 100x is ~12.5M sequential collapses; in Python that is hours. Each
round here instead scores every edge at once, takes a MUTUAL-BEST MATCHING
(edge (u,v) is collapsed only if it is the cheapest incident edge of both u
and v), and applies the whole matching in one vectorised step. A matching
touches each vertex at most once, so the collapses within a round are
independent and cannot interact. Rounds are O(F) numpy and the face count
falls geometrically, so any reduction is reached in tens of rounds.

The cost is that a matching is not the greedy optimum -- some cheap edges wait
for a later round. Quality is measured, not assumed: see mesh_eval.py.

SAFETY
------
Two guards decide whether a matched edge may actually collapse.

*Link condition* (Dey et al. 1999): for edge (u,v) the collapse preserves
topology iff the vertices adjacent to both u and v are exactly the vertices
opposite (u,v) in the faces that contain it -- 2 for a manifold interior edge,
1 on a boundary. Violate it and the surface pinches: two sheets that were
apart get welded, or a tube closes into a disc. This is what protects the thin
axon tubes, and it is why the tube diameter never appears as a parameter.

*Normal flip*: the collapse is rejected if it would turn any surviving
incident face through more than ``max_normal_turn``. This catches the
geometric fold-overs the link condition permits.

Both are computed for all matched edges at once.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "decimate_mesh",
    "cull_small_components",
    "weld_vertices",
    "canonical_faces",
    "mesh_floor_report",
]


# ===================================================================
# primitives
# ===================================================================

def weld_vertices(
    vertices: npt.NDArray, faces: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Merge bit-identical vertex positions.

    The precomputed multi-LOD source arrives as one fragment per octree chunk
    with the boundary vertices duplicated in every fragment that touches them:
    on the test cell 218,441 duplicates across 39,515 index-disjoint pieces
    that are really 1,725 components. Decimating before welding treats every
    chunk seam as a free boundary and shreds it, so this runs first and always.

    Exact float equality, deliberately: the duplicates are bitwise copies, so
    no tolerance is needed and none is offered -- a tolerance would be a second
    parameter able to weld genuinely distinct surfaces.
    """
    v = np.ascontiguousarray(vertices)
    view = v.view([("x", v.dtype), ("y", v.dtype), ("z", v.dtype)]).ravel()
    _, first, inverse = np.unique(view, return_index=True, return_inverse=True)
    return v[first], inverse[faces].astype(faces.dtype), inverse


def canonical_faces(faces: npt.NDArray) -> npt.NDArray:
    """Rotate each face so its smallest index leads, preserving cyclic order.

    Winding-aware, unlike sorting the triple. On the test cell 11,084 of 27,738
    duplicated triples are genuine opposite-wound double walls -- two sheets of
    membrane back to back -- and a winding-blind dedup deletes one of each,
    turning 252 boundary edges into 31,317.
    """
    f = np.asarray(faces)
    amin = f.argmin(axis=1)
    idx = (amin[:, None] + np.arange(3)[None, :]) % 3
    return np.take_along_axis(f, idx, axis=1)


def _dedup_faces(faces: npt.NDArray) -> npt.NDArray:
    canon = canonical_faces(faces)
    _, keep = np.unique(canon, axis=0, return_index=True)
    return faces[np.sort(keep)]


def _drop_degenerate(faces: npt.NDArray) -> npt.NDArray:
    ok = ((faces[:, 0] != faces[:, 1])
          & (faces[:, 1] != faces[:, 2])
          & (faces[:, 0] != faces[:, 2]))
    return faces[ok]


def _compact(vertices: npt.NDArray, faces: npt.NDArray):
    """Drop vertices no face references, and reindex.

    This is the only place a vertex can leave the array, and it runs at the end
    of every round, which is what makes the no-orphan invariant hold by
    construction at every intermediate state and not just at the end.
    """
    if len(faces) == 0:
        return vertices[:0], faces.reshape(0, 3), np.zeros(0, np.int64)
    used = np.unique(faces)
    remap = np.full(len(vertices), -1, np.int64)
    remap[used] = np.arange(len(used))
    return vertices[used], remap[faces].astype(faces.dtype), used


def cull_small_components(
    vertices: npt.NDArray, faces: npt.NDArray, min_faces: int,
) -> tuple[npt.NDArray, npt.NDArray, dict]:
    """Drop connected components smaller than ``min_faces``.

    Segmentation leaves dust: the welded source of the test cell is 1,725
    components of which 1,488 hold under 100 faces, together only 0.20% of the
    surface. Carrying them costs nothing at full resolution and everything at
    the coarse end, because a component cannot go below ~4 faces however tight
    the budget -- so at a 223k-face target 941 fragments were still absorbing
    2.58% of the budget and, more importantly, holding 18.1% of it outside the
    main arbour. MICrONS's own decimator consolidates 1,725 -> 166 components
    and puts 95.8% of its faces in the largest, against 81.9% here; that gap is
    most of why this decimator collapsed three times as many tubes at matched
    face counts.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if min_faces <= 0 or len(faces) == 0:
        return vertices, faces, {"components_removed": 0, "faces_removed": 0}
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
                                faces[:, [2, 0]]], axis=0), axis=1)
    g = coo_matrix((np.ones(len(e), np.int8), (e[:, 0], e[:, 1])),
                   shape=(len(vertices), len(vertices)))
    n, label = connected_components(g, directed=False)
    fl = label[faces[:, 0]]
    per = np.bincount(fl, minlength=n)
    keep_comp = per >= min_faces
    if keep_comp.all():
        return vertices, faces, {"components_removed": 0, "faces_removed": 0}
    keep = keep_comp[fl]
    stats = {"components_removed": int((~keep_comp & (per > 0)).sum()),
             "faces_removed": int((~keep).sum())}
    v2, f2, _ = _compact(vertices, faces[keep])
    return v2, f2, stats


def _face_planes(v: npt.NDArray, f: npt.NDArray):
    """Unit normals and plane offsets, plus twice the triangle area."""
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    area2 = np.linalg.norm(n, axis=1)
    good = area2 > 0
    nn = np.zeros_like(n)
    nn[good] = n[good] / area2[good, None]
    d = -np.einsum("ij,ij->i", nn, p0)
    return nn, d, area2


def _vertex_quadrics(v: npt.NDArray, f: npt.NDArray) -> npt.NDArray:
    """Area-weighted fundamental error quadrics, packed as 10 uppertriangular
    coefficients per vertex: (a2, ab, ac, ad, b2, bc, bd, c2, cd, d2)."""
    n, d, area2 = _face_planes(v, f)
    a, b, c = n[:, 0], n[:, 1], n[:, 2]
    w = area2                      # 2*area; weighting by area is standard and
                                   # keeps big flat regions from being eaten
    comp = np.stack([a * a, a * b, a * c, a * d,
                     b * b, b * c, b * d,
                     c * c, c * d,
                     d * d], axis=1) * w[:, None]
    # np.add.at is an unbuffered ufunc loop and costs minutes at 25M faces;
    # bincount does the same scatter-add in C, once per component.
    # f.ravel() is corner-major -- face i's three corners are consecutive --
    # so the weights must be REPEATed (each face's value three times), not
    # tiled (which would pair face i's corner with face i+1's quadric).
    flat = f.ravel()
    Q = np.empty((len(v), 10), np.float64)
    for k in range(10):
        Q[:, k] = np.bincount(flat, weights=np.repeat(comp[:, k], 3),
                              minlength=len(v))
    return Q


def _quadric_cost(Q: npt.NDArray, p: npt.NDArray) -> npt.NDArray:
    """v^T Q v for packed quadrics and candidate points."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    return (Q[:, 0] * x * x + 2 * Q[:, 1] * x * y + 2 * Q[:, 2] * x * z
            + 2 * Q[:, 3] * x + Q[:, 4] * y * y + 2 * Q[:, 5] * y * z
            + 2 * Q[:, 6] * y + Q[:, 7] * z * z + 2 * Q[:, 8] * z + Q[:, 9])


def _quadric_optimum(Q: npt.NDArray, mid: npt.NDArray,
                     pu: npt.NDArray, pv: npt.NDArray):
    """The point minimising v^T Q v, and whether it is trustworthy.

    Solves the 3x3 system A x = -b with A the quadratic block of Q. Explicit
    cofactors rather than np.linalg.solve: the determinant is needed anyway to
    judge conditioning, and a batched solve raises on the singular rows that
    are common here (flat regions give a rank-1 A).

    Rejected when the system is ill conditioned, or when the solution lands
    further than one edge length outside the edge -- on a nearly flat pair of
    triangles the optimum is only weakly determined along the surface and can
    slide arbitrarily far without the cost rising.
    """
    a11, a12, a13 = Q[:, 0], Q[:, 1], Q[:, 2]
    a22, a23, a33 = Q[:, 4], Q[:, 5], Q[:, 7]
    b1, b2, b3 = Q[:, 3], Q[:, 6], Q[:, 8]

    c11 = a22 * a33 - a23 * a23
    c12 = a13 * a23 - a12 * a33
    c13 = a12 * a23 - a13 * a22
    det = a11 * c11 + a12 * c12 + a13 * c13

    scale = np.abs(a11) + np.abs(a22) + np.abs(a33) + 1e-300
    ok = np.abs(det) > 1e-10 * scale ** 3
    safe = np.where(ok, det, 1.0)

    c22 = a11 * a33 - a13 * a13
    c23 = a12 * a13 - a11 * a23
    c33 = a11 * a22 - a12 * a12
    x = -(c11 * b1 + c12 * b2 + c13 * b3) / safe
    y = -(c12 * b1 + c22 * b2 + c23 * b3) / safe
    z = -(c13 * b1 + c23 * b2 + c33 * b3) / safe
    p = np.stack([x, y, z], axis=1)

    elen = np.linalg.norm(pv - pu, axis=1)
    stray = np.linalg.norm(p - mid, axis=1) > np.maximum(elen, 1e-12)
    ok &= ~stray & np.isfinite(p).all(axis=1)
    return np.where(ok[:, None], p, mid), ok


_STRIDE = np.int64(1) << np.int64(32)


def _unique_edges(f: npt.NDArray):
    """Undirected unique edges, and the directed half-edge -> edge id map."""
    he = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    lo = np.minimum(he[:, 0], he[:, 1])
    hi = np.maximum(he[:, 0], he[:, 1])
    # one int64 key beats np.unique(axis=0), which pays a structured-void
    # comparison per row; vertex counts here stay far below 2**31.
    key = lo.astype(np.int64) * _STRIDE + hi
    ukey, inv = np.unique(key, return_inverse=True)
    uniq = np.stack([ukey // _STRIDE, ukey % _STRIDE], axis=1)
    return uniq, inv, he


# ===================================================================
# guards
# ===================================================================

def _build_link_tables(f: npt.NDArray, n_verts: int):
    """Adjacency CSR and the sorted edge-key table the link test needs.

    Hoisted out of :func:`_link_condition_ok` because the topology is fixed for
    a whole round while the link test runs once per selection pass; rebuilding
    it per pass made a 1.1M-face mesh miss a two-minute budget.
    """
    a = np.concatenate([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
    b = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
    akey = np.unique(a.astype(np.int64) * _STRIDE + b)
    src = (akey // _STRIDE).astype(np.int64)
    nbr = (akey % _STRIDE).astype(np.int64)
    deg = np.bincount(src, minlength=n_verts)
    start = np.concatenate([[0], np.cumsum(deg)])
    fa = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    fb = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    fk = np.sort(np.minimum(fa, fb).astype(np.int64) * _STRIDE
                 + np.maximum(fa, fb))
    return deg, start, nbr, fk


def _link_condition_ok(
    edges: npt.NDArray, cand: npt.NDArray, f: npt.NDArray, n_verts: int,
    tables=None,
) -> npt.NDArray:
    """Vectorised link condition for the candidate edges.

    For a triangle mesh, collapsing (u,v) is topology-safe iff
    ``|adj(u) & adj(v)| == number of faces containing both u and v``. On a
    closed manifold that is 2; on a boundary edge 1. Anything larger means
    another vertex is adjacent to both without sharing a face with the edge --
    collapsing would identify two parts of the surface that are not local
    neighbours, which is exactly the pinch that closes a thin tube.
    """
    if len(cand) == 0:
        return np.zeros(0, bool)

    # --- adjacency in CSR, built once per round -------------------------
    deg, start, nbr, fk_sorted = (tables if tables is not None
                                  else _build_link_tables(f, n_verts))

    u = edges[cand, 0]
    v = edges[cand, 1]

    # --- gather both neighbour lists, tagged by candidate id -------------
    du, dv = deg[u], deg[v]
    total = int(du.sum() + dv.sum())
    if total == 0:
        return np.zeros(len(cand), bool)
    cid = np.concatenate([np.repeat(np.arange(len(cand)), du),
                          np.repeat(np.arange(len(cand)), dv)])
    off_u = np.repeat(start[u], du) + _ranges(du)
    off_v = np.repeat(start[v], dv) + _ranges(dv)
    vals = nbr[np.concatenate([off_u, off_v])]

    # a shared neighbour appears exactly twice under the same candidate id
    key = cid.astype(np.int64) * _STRIDE + vals
    key.sort()
    shared = np.zeros(len(cand), np.int64)
    dupes = key[1:][key[1:] == key[:-1]]
    if len(dupes):
        shared += np.bincount((dupes // _STRIDE).astype(np.int64),
                              minlength=len(cand))

    # --- how many faces contain both endpoints --------------------------
    ek = (np.minimum(edges[cand, 0], edges[cand, 1]).astype(np.int64) * _STRIDE
          + np.maximum(edges[cand, 0], edges[cand, 1]))
    nfaces = (np.searchsorted(fk_sorted, ek, "right")
              - np.searchsorted(fk_sorted, ek, "left"))

    return shared == nfaces


def _ranges(counts: npt.NDArray) -> npt.NDArray:
    """[0,1,..,c0-1, 0,1,..,c1-1, ...] without a Python loop."""
    counts = np.asarray(counts, np.int64)
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, np.int64)
    out = np.ones(total, np.int64)
    idx = np.cumsum(counts)[:-1]
    out[0] = 0
    out[idx] = 1 - counts[:-1]
    return np.cumsum(out)


def _normal_flip_ok(
    v: npt.NDArray, f: npt.NDArray, edges: npt.NDArray, cand: npt.NDArray,
    newpos: npt.NDArray, max_turn_cos: float,
) -> npt.NDArray:
    """Reject collapses that fold a surviving incident face over.

    Only faces incident to exactly one endpoint matter; the faces containing
    both are the ones that legitimately disappear.
    """
    if len(cand) == 0:
        return np.zeros(0, bool)
    u, vv = edges[cand, 0], edges[cand, 1]
    moved = np.full(len(v), -1, np.int64)      # vertex -> candidate id
    moved[u] = np.arange(len(cand))
    moved[vv] = np.arange(len(cand))
    target = np.zeros((len(v), 3), np.float64)
    target[u] = newpos
    target[vv] = newpos

    ok = np.ones(len(cand), bool)
    fm = moved[f]                                 # (F,3) candidate id or -1
    touched = (fm >= 0).any(axis=1)
    ft = f[touched]
    if len(ft) == 0:
        return ok
    fmt = fm[touched]
    # a face where both endpoints of the SAME candidate appear is destroyed
    cidf = fmt.max(axis=1)
    both = ((fmt == cidf[:, None]).sum(axis=1) >= 2)
    ft, cidf = ft[~both], cidf[~both]
    if len(ft) == 0:
        return ok
    old = v[ft]
    new = old.copy()
    fmt2 = moved[ft]
    sel = fmt2 >= 0
    new[sel] = target[ft[sel]]
    n_old = np.cross(old[:, 1] - old[:, 0], old[:, 2] - old[:, 0])
    n_new = np.cross(new[:, 1] - new[:, 0], new[:, 2] - new[:, 0])
    lo = np.linalg.norm(n_old, axis=1)
    ln = np.linalg.norm(n_new, axis=1)
    good = (lo > 0) & (ln > 0)
    cosang = np.ones(len(ft))
    cosang[good] = np.einsum("ij,ij->i", n_old[good], n_new[good]) / (lo[good] * ln[good])
    bad = cosang < max_turn_cos
    if bad.any():
        ok[np.unique(cidf[bad])] = False
    return ok


# ===================================================================
# the decimator
# ===================================================================

def decimate_mesh(
    vertices: npt.NDArray,
    faces: npt.NDArray,
    *,
    target_faces: int | None = None,
    target_ratio: float | None = None,
    max_normal_turn: float = 100.0,
    max_rounds: int = 200,
    max_passes: int = 200,
    min_progress: float = 0.005,
    do_weld: bool = True,
    min_component_faces: int = 0,
    verbose: bool = False,
) -> dict[str, Any]:
    """Decimate to ``target_faces`` by batched quadric-error edge collapse.

    Args:
        vertices: ``(V, 3)`` positions.
        faces: ``(F, 3)`` triangle indices.
        target_faces: stop once the face count is at or below this.
        target_ratio: alternative to target_faces; keep this fraction.
        max_normal_turn: degrees a surviving face may rotate. The default is
            deliberately permissive -- these surfaces are noisy segmentation
            boundaries, and a tight bound stalls the collapse long before the
            target.
        max_rounds: safety stop.
        max_passes: safety cap on selection retries within one round. Each
            pass retires the edges the guards refused and re-selects from what
            is left, which is what stops a refused edge from permanently
            blocking its two endpoints (see the round body). This MUST NOT be
            the binding constraint: at the old default of 12 a round still
            accepted ~1000 collapses on its last allowed pass, so the run
            stopped with work left and reported ``hit_floor`` -- a pyramid
            level came out at 2.71x instead of the requested 4x and the
            shortfall was read as a topological floor. Raising it to 100
            reached 4.03x on the same input in FEWER rounds and less wall
            clock. ``pass_budget_bound`` in the result says whether this cap
            was ever reached, so a floor can be distinguished from a cap.
        min_progress: if a round removes a smaller fraction of faces than
            this, the mesh is at its topological floor and the run stops. This
            is what "as small as it can get and still be a mesh" means in
            practice -- every remaining collapse is refused by the guards.
        do_weld: weld coincident positions first. Required for chunk-fragmented
            input; harmless otherwise.
        min_component_faces: drop connected components below this many faces
            before decimating, so the budget is spent on the structure that
            matters rather than on segmentation dust. See
            :func:`cull_small_components`. 0 disables.

    Returns:
        dict with ``vertices``, ``faces``, and statistics including
        ``hit_floor`` (True if it stopped for lack of safe collapses rather
        than because the target was met).
    """
    v = np.asarray(vertices, np.float64)
    f = np.asarray(faces, np.int64)
    n_v0, n_f0 = len(v), len(f)

    # Work about the centroid. A quadric's constant term is d^2 where d is the
    # plane's distance from the ORIGIN, so at MICrONS coordinates (~1e6 nm)
    # that term is ~1e12 while the geometric differences being ranked are
    # ~1e2 -- ten of float64's sixteen digits gone to cancellation. Recentring
    # made two identical spheres placed at x=2000 and x=9000 decimate
    # identically; before it they lost 20% and 32% of their volume.
    origin = (v.min(axis=0) + v.max(axis=0)) * 0.5 if len(v) else np.zeros(3)
    v = v - origin

    if do_weld:
        v, f, _ = weld_vertices(v, f)
    f = _dedup_faces(_drop_degenerate(f))
    v, f, _ = _compact(v, f)
    cull = {"components_removed": 0, "faces_removed": 0}
    if min_component_faces > 0:
        v, f, cull = cull_small_components(v, f, min_component_faces)

    if target_faces is None:
        target_faces = (max(4, int(round(len(f) * target_ratio)))
                        if target_ratio is not None else max(4, len(f) // 4))
    target_faces = max(4, int(target_faces))

    cos_lim = float(np.cos(np.radians(max_normal_turn)))
    rounds = 0
    hit_floor = False
    refused = 0
    pass_budget_bound = False

    while len(f) > target_faces and rounds < max_rounds:
        rounds += 1
        before = len(f)

        Q = _vertex_quadrics(v, f)
        edges, _, _ = _unique_edges(f)
        if len(edges) == 0:
            hit_floor = True
            break

        # --- cost of every edge, best of {u, v, midpoint, QEM optimum} ----
        eu, ev = edges[:, 0], edges[:, 1]
        Qe = Q[eu] + Q[ev]
        mid = 0.5 * (v[eu] + v[ev])
        opt, opt_ok = _quadric_optimum(Qe, mid, v[eu], v[ev])
        cands = np.stack([v[eu], v[ev], mid, opt], axis=1)
        costs = np.stack([_quadric_cost(Qe, cands[:, k]) for k in range(4)],
                         axis=1)
        # The optimum is the placement that actually preserves volume: on a
        # convex patch it bulges outward, where {u, v, midpoint} can only cut
        # the corner off and shrink the surface inward every round. It is
        # accepted only where the 3x3 solve is well conditioned and the result
        # stays near the edge (_quadric_optimum), so a near-degenerate quadric
        # cannot fling a vertex across the mesh, and coarse vertices stay
        # inside the source bounding box.
        costs[~opt_ok, 3] = np.inf
        pick = costs.argmin(axis=1)
        cost = costs[np.arange(len(edges)), pick]
        newpos = cands[np.arange(len(edges)), pick]
        cost = np.maximum(cost, 0.0)

        # --- independent set: edges that are the cheapest at BOTH ends ----
        # A total order over edges (cost, then a deterministic hash of the
        # endpoints) makes every rank unique, and an edge is taken iff it is
        # the minimum rank at both of its endpoints. That always selects at
        # least the globally cheapest edge, so a round can never come back
        # empty while a legal collapse exists.
        #
        # The hash tiebreak is not cosmetic. Ranking ties by edge INDEX makes
        # the selection degenerate on a uniform mesh -- on a path 0-1-2-3 with
        # equal costs every edge loses at one endpoint and nothing is selected,
        # which stalled a clean cylinder at round 1. Hashing scatters the ties
        # across the surface and restores a constant selection fraction.
        n_v = len(v)
        eid = np.arange(len(edges), dtype=np.int64)
        h = (eu.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
             ^ ev.astype(np.uint64) * np.uint64(0xBF58476D1CE4E5B9))
        h ^= h >> np.uint64(31)
        order = np.lexsort((h.astype(np.int64), cost))
        rank = np.empty(len(edges), np.int64)
        rank[order] = eid

        # --- select, guard, and RETRY within the round --------------------
        # A single selection pass deadlocks. An edge that a guard refuses is
        # still the cheapest edge at both its endpoints next round, so it wins
        # the selection again, is refused again, and meanwhile blocks every
        # other edge at those two vertices from ever being tried. Measured on
        # the LOD 2 test mesh, link-condition acceptance decayed 95.5% -> 4.3%
        # over 13 rounds and the face count flatlined at 371k.
        #
        # Retiring a refused edge for the remainder of the round lets its
        # neighbours be selected instead, so each round converts most of its
        # available edges into collapses rather than re-proposing the same
        # rejects. Refused edges return in the next round, when the changed
        # topology around them may have made the collapse legal.
        max_collapse = max(1, (len(f) - target_faces + 1) // 2)
        tables = _build_link_tables(f, n_v)
        alive = np.ones(len(edges), bool)
        used = np.zeros(n_v, bool)
        accepted: list[npt.NDArray] = []
        n_acc = 0

        for _pass in range(max_passes):
            cand = alive & ~used[eu] & ~used[ev]
            if not cand.any() or n_acc >= max_collapse:
                break
            if _pass == max_passes - 1:
                # exiting on the cap rather than on exhaustion: whatever
                # shortfall follows is a budget artefact, not a floor
                pass_budget_bound = True
            cid = eid[cand]
            crank = rank[cid]
            o = cid[np.argsort(crank, kind="stable")][::-1]   # descending rank
            pair = np.empty(2 * len(o), np.int64)
            pair[0::2] = eu[o]
            pair[1::2] = ev[o]
            best = np.full(n_v, np.iinfo(np.int64).max, np.int64)
            best[pair] = np.repeat(rank[o], 2)
            sel = cid[(best[eu[cid]] == rank[cid]) & (best[ev[cid]] == rank[cid])]
            if len(sel) == 0:
                break
            sel = sel[np.argsort(rank[sel], kind="stable")][:max_collapse - n_acc]

            ok = _link_condition_ok(edges, sel, f, n_v, tables)
            good = sel[ok]
            refused += int((~ok).sum())
            if len(good):
                ok2 = _normal_flip_ok(v, f, edges, good, newpos[good], cos_lim)
                refused += int((~ok2).sum())
                good = good[ok2]
            alive[sel] = False               # retire everything tried
            if len(good):
                used[eu[good]] = True
                used[ev[good]] = True
                accepted.append(good)
                n_acc += len(good)

        if n_acc == 0:
            hit_floor = True
            break
        matched2 = np.concatenate(accepted)

        # --- apply the whole matching at once -----------------------------
        u2, v2 = edges[matched2, 0], edges[matched2, 1]
        keepv = np.minimum(u2, v2)
        dropv = np.maximum(u2, v2)
        v[keepv] = newpos[matched2]
        remap = np.arange(len(v), dtype=np.int64)
        remap[dropv] = keepv
        f = remap[f]
        f = _dedup_faces(_drop_degenerate(f))
        v, f, _ = _compact(v, f)

        removed = before - len(f)
        if verbose:
            print(f"  round {rounds:3d}: {before:>10,} -> {len(f):>10,} faces "
                  f"({len(matched2):,} collapses, {removed / max(before,1):.3%})")
        if removed / max(before, 1) < min_progress:
            hit_floor = True
            break

    # the invariant, asserted rather than trusted
    if len(f):
        assert np.bincount(f.ravel(), minlength=len(v)).min() > 0, \
            "orphan vertex produced"

    return {
        "vertices": (v + origin).astype(np.float32),
        "faces": f.astype(np.int64),
        "vertex_count": len(v),
        "face_count": len(f),
        "input_vertices": n_v0,
        "input_faces": n_f0,
        "face_reduction": n_f0 / max(len(f), 1),
        "vertex_reduction": n_v0 / max(len(v), 1),
        "rounds": rounds,
        "hit_floor": bool(hit_floor),
        "target_faces": int(target_faces),
        "target_met": bool(len(f) <= target_faces),
        "collapses_refused": int(refused),
        "components_culled": int(cull["components_removed"]),
        "faces_culled": int(cull["faces_removed"]),
        "pass_budget_bound": bool(pass_budget_bound),
        "method": "quadric_edge_collapse",
    }


# ===================================================================
# the floor
# ===================================================================

def mesh_floor_report(vertices: npt.NDArray, faces: npt.NDArray) -> dict:
    """Is this still a mesh? The stop test for a pyramid.

    "Still a mesh" is taken to mean: every vertex is used, every component
    still encloses something (a closed triangulated surface needs at least the
    4 faces of a tetrahedron), and the surface has not shattered into isolated
    triangles. Reported rather than judged, so the caller sets the policy.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    nv, nf = len(vertices), len(faces)
    if nf == 0:
        return {"n_vertices": nv, "n_faces": 0, "is_mesh": False,
                "reason": "no faces"}
    used = len(np.unique(faces))
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
                                faces[:, [2, 0]]], axis=0), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    g = coo_matrix((np.ones(len(e), np.int8), (e[:, 0], e[:, 1])),
                   shape=(nv, nv))
    ncomp, labels = connected_components(g, directed=False)
    fcomp = labels[faces[:, 0]]
    per_comp = np.bincount(fcomp, minlength=ncomp)
    live = per_comp[per_comp > 0]
    return {
        "n_vertices": nv,
        "n_faces": nf,
        "v_over_f": round(nv / nf, 3),
        "orphans": int(nv - used),
        "components": int((per_comp > 0).sum()),
        "components_under_4_faces": int((live < 4).sum()),
        "faces_in_tiny_components": int(live[live < 4].sum()),
        "boundary_edges": int((counts == 1).sum()),
        "nonmanifold_edges": int((counts > 2).sum()),
        "is_mesh": bool(nv - used == 0 and nf >= 4),
    }
