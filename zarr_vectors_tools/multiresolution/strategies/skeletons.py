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

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import (
    COARSEN_SKELETON,
    VERTEX_ATTRIBUTES,
    VERTICES,
)


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
    from zarr_vectors.core.arrays import read_chunk_attributes, read_chunk_vertices
    from zarr_vectors.core.store import get_resolution_level, open_store
    from zarr_vectors.exceptions import ArrayError
    from zarr_vectors_tools.multiresolution.skeleton_graph import split_components
    from zarr_vectors.types.skeletons import write_skeleton_chunk

    shared = shared or {}
    ndim = shared["ndim"]
    attr_names = shared["attr_names"]
    attr_dtypes = {n: np.dtype(d) for n, d in shared["attr_dtypes"].items()}
    stride = shared["stride"]
    attr_agg = shared["attr_agg"]
    tcc = tuple(payload["tcc"])
    groups = shared["groups_by_tcc"][tcc]

    root = open_store(shared["store_path"], mode="r+")
    src = get_resolution_level(root, shared["source_level"])
    level_group = get_resolution_level(root, shared["target_level"])

    # Read this target chunk's source children once (vertices + attrs only).
    child_ccs = sorted({tuple(cc) for g in groups for cc, _ in g["members"]})
    vcache: dict = {}
    acache: dict = {name: {} for name in attr_names}
    for cc in child_ccs:
        try:
            vgroups = read_chunk_vertices(src, cc, dtype=np.float32, ndim=ndim)
        except ArrayError:
            continue
        for fidx, vg in enumerate(vgroups):
            vcache[(cc, fidx)] = vg
        for name in attr_names:
            try:
                ag = read_chunk_attributes(src, name, cc, dtype=attr_dtypes[name])
            except ArrayError:
                ag = None
            if ag is not None:
                for fidx in range(len(ag)):
                    acache[name][(cc, fidx)] = ag[fidx]

    pieces: list = []
    total_out_vertices = 0
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
        needed = g["anchor_needed"]  # {merged_idx: [p_id, endpoint]}
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
            if needed:
                kept_pos = {
                    int(c): i
                    for i, c in enumerate(simp["kept_source_indices"].tolist())
                }
                anchors = {}
                for ci, mv in enumerate(comp_vids):
                    tag = needed.get(int(mv))
                    if tag is None:
                        continue
                    sl = kept_pos.get(ci)
                    if sl is not None:
                        anchors[tuple(tag)] = sl
                if anchors:
                    piece["anchors"] = anchors
            pieces.append(piece)
            total_out_vertices += len(rpos)

    recs, alocs = write_skeleton_chunk(
        level_group, tcc, pieces, attr_dtypes=attr_dtypes,
    )
    return {
        "records": recs,
        "anchor_locs": alocs,
        "vertex_count": total_out_vertices,
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
        create_attribute_array,
        create_cross_chunk_links_array,
        create_fragment_attribute_array,
        create_links_array,
        create_object_attributes_array,
        create_object_index_array,
        create_vertices_array,
        list_chunk_keys,
        read_all_object_manifests,
        read_chunk_links,
        read_cross_chunk_links,
        read_object_attributes,
        read_vertex_fragment_index,
        write_cross_chunk_links,
        write_object_attributes,
        write_object_index,
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

    # --- source manifests + sparsity ------------------------------------
    src_manifests = read_all_object_manifests(src)
    n_src = len(src_manifests)
    if n_src == 0:
        return {"vertex_count": 0, "object_count": 0, "method": COARSEN_SKELETON}

    if sparsity_factor > 1.0 and n_src > 1:
        keep_frac = 1.0 / sparsity_factor
        # length = source vertex count per object (cheap, from manifests +
        # fragment sizes); used by the "length" strategy.
        lengths = None
        if sparsity_strategy == "length":
            lengths = np.array(
                [len(m) for m in src_manifests], dtype=np.float64
            )  # fragment count as a proxy for skeleton size
        kept = apply_sparsity(
            n_src, keep_frac, sparsity_strategy, seed=sparsity_seed,
            lengths=lengths,
        )
        keep_oids = sorted(int(o) for o in kept)
    else:
        keep_oids = list(range(n_src))

    # --- metadata-only source read (bounded memory, no vertex positions) ---
    # The coordinator plans purely from per-fragment vertex COUNTS (from the
    # ``vertex_fragments`` index headers) plus the small branch-link arrays;
    # the bulk vertex/attribute data is read per target chunk by the workers
    # (:func:`_coarsen_target_chunk`).  This keeps the coordinator's footprint
    # ~O(#fragments) rather than O(level vertices), so the level is never
    # loaded whole in one place.
    src_frag_count: dict[tuple[ChunkCoords, int], int] = {}
    src_frag_start: dict[tuple[ChunkCoords, int], int] = {}
    src_links: dict[tuple[ChunkCoords, int], npt.NDArray] = {}
    for cc in list_chunk_keys(src, VERTICES):
        try:
            fi = read_vertex_fragment_index(src, cc)
        except ArrayError:
            continue
        try:
            lgroups = read_chunk_links(src, cc, link_width=2, delta=0)
        except ArrayError:
            lgroups = None
        start = 0
        for fidx in range(fi.num_fragments):
            cnt = int(fi.range(fidx)[1]) if fi.is_range(fidx) else int(len(fi.indices(fidx)))
            src_frag_count[(cc, fidx)] = cnt
            src_frag_start[(cc, fidx)] = start
            src_links[(cc, fidx)] = (
                lgroups[fidx]
                if lgroups is not None and fidx < len(lgroups)
                else np.zeros((0, 2), np.int64)
            )
            start += cnt

    keep_set = set(keep_oids)

    # --- resolve source cross-chunk links into per-object connections ----
    # As the grid coarsens (chunk_scale per axis), the source chunks an
    # object touched collapse into one target chunk.  A source cross-chunk
    # link whose two endpoints now fall in the SAME target chunk becomes an
    # intra-chunk edge that re-merges the two fragments; one that still
    # spans target chunks is re-emitted as a target-level cross-chunk link.
    # Connectivity comes ONLY from these stored links — never from
    # geometric proximity.
    frag_oid: dict[tuple[ChunkCoords, int], int] = {}
    for oid in keep_oids:
        for cc, fidx in src_manifests[oid]:
            frag_oid[(cc, fidx)] = oid
    # Per chunk: sorted fragment starts + parallel (end, fidx) arrays, so a
    # chunk-local index maps to its fragment via binary search.  A linear
    # scan here is O(fragments) per lookup → O(fragments²) per chunk, which
    # explodes on dense coarse-level chunks (tens of thousands of paths).
    cc_starts: dict[ChunkCoords, npt.NDArray] = {}
    cc_ends: dict[ChunkCoords, npt.NDArray] = {}
    cc_fidx: dict[ChunkCoords, npt.NDArray] = {}
    _tmp: dict[ChunkCoords, list[tuple[int, int, int]]] = defaultdict(list)
    for (cc, fidx), start in src_frag_start.items():
        ln = src_frag_count[(cc, fidx)]
        _tmp[cc].append((start, start + ln, fidx))
    for cc, rows in _tmp.items():
        rows.sort()
        cc_starts[cc] = np.asarray([r[0] for r in rows], dtype=np.int64)
        cc_ends[cc] = np.asarray([r[1] for r in rows], dtype=np.int64)
        cc_fidx[cc] = np.asarray([r[2] for r in rows], dtype=np.int64)

    def _resolve_ep(cc, vi):
        starts = cc_starts.get(cc)
        if starts is None:
            return None
        i = int(np.searchsorted(starts, vi, side="right")) - 1
        if i < 0 or vi >= int(cc_ends[cc][i]):
            return None
        return int(cc_fidx[cc][i]), vi - int(starts[i])

    try:
        src_ccl = read_cross_chunk_links(src, delta=0)
    except Exception:
        src_ccl = []
    conns_by_oid: dict[int, list[tuple]] = defaultdict(list)
    for rec in src_ccl:
        if len(rec) != 2:
            continue
        (ccA, viA), (ccB, viB) = rec
        ccA = tuple(int(x) for x in ccA)
        ccB = tuple(int(x) for x in ccB)
        rA = _resolve_ep(ccA, int(viA))
        rB = _resolve_ep(ccB, int(viB))
        if rA is None or rB is None:
            continue
        oidA = frag_oid.get((ccA, rA[0]))
        oidB = frag_oid.get((ccB, rB[0]))
        if oidA is None or oidA != oidB or oidA not in keep_set:
            continue
        conns_by_oid[oidA].append((ccA, rA[0], rA[1], ccB, rB[0], rB[1]))

    # --- group an object's fragments by target chunk + merged offsets ---
    # A fragment lives entirely in source chunk ``cc``, which nests into target
    # chunk ``floor(cc / scale)`` — pure grid arithmetic, so the target chunk
    # is known WITHOUT reading any vertex positions.  ``group_members`` keeps
    # the source ``(cc, fidx)`` refs in manifest order; the workers read the
    # vertex data themselves.  ``group_offset`` (merged-vertex offsets, used to
    # express intra/cross-target links in merged-index space) comes from the
    # per-fragment counts.
    group_members: dict[tuple[int, ChunkCoords], list[tuple[ChunkCoords, int]]] = defaultdict(list)
    frag_tcc: dict[tuple[ChunkCoords, int], ChunkCoords] = {}
    for oid in keep_oids:
        for cc, fidx in src_manifests[oid]:
            cnt = src_frag_count.get((cc, fidx))
            if not cnt:  # absent chunk or empty fragment
                continue
            tcc = tuple(int(cc[a] // scale[a]) for a in range(ndim))
            frag_tcc[(cc, fidx)] = tcc
            group_members[(oid, tcc)].append((cc, fidx))
    group_offset: dict[tuple[int, ChunkCoords], dict[tuple[ChunkCoords, int], int]] = {}
    for key, members in group_members.items():
        off = 0
        m = {}
        for cc, fidx in members:
            m[(cc, fidx)] = off
            off += src_frag_count[(cc, fidx)]
        group_offset[key] = m

    # Connections → intra-merge edges (same target chunk) or pending
    # cross-target links (different target chunks).
    intra_extra: dict[tuple[int, ChunkCoords], list[tuple[int, int]]] = defaultdict(list)
    pending: list[tuple[int, tuple[ChunkCoords, int], tuple[ChunkCoords, int]]] = []

    # Intra-chunk branch links: each source path fragment stores its start
    # branch link (chunk-local) connecting it to another path of the same
    # object in the same chunk.  Both endpoints land in the same target
    # chunk, so they re-merge as intra edges.
    for oid in keep_oids:
        for cc, fidx in src_manifests[oid]:
            tcc = frag_tcc.get((cc, fidx))
            if tcc is None:
                continue
            links = src_links.get((cc, fidx))
            if links is None or len(links) == 0:
                continue
            goff = group_offset.get((oid, tcc))
            if goff is None:
                continue
            for ch_cl, par_cl in np.asarray(links, dtype=np.int64).reshape(-1, 2):
                ra = _resolve_ep(cc, int(ch_cl))
                rb = _resolve_ep(cc, int(par_cl))
                if ra is None or rb is None:
                    continue
                if (cc, ra[0]) not in goff or (cc, rb[0]) not in goff:
                    continue
                mA = goff[(cc, ra[0])] + ra[1]
                mB = goff[(cc, rb[0])] + rb[1]
                intra_extra[(oid, tcc)].append((mA, mB))

    # Cross-chunk (coincident chunk-boundary) vertices are force-kept by the
    # decimator so the stitch points survive — "fix all points at chunk
    # boundaries".
    forced_keep_by_group: dict[tuple[int, ChunkCoords], set[int]] = defaultdict(set)
    for oid, conns in conns_by_oid.items():
        for ccA, fidxA, localA, ccB, fidxB, localB in conns:
            tA = frag_tcc.get((ccA, fidxA))
            tB = frag_tcc.get((ccB, fidxB))
            if tA is None or tB is None:
                continue
            mA = group_offset[(oid, tA)][(ccA, fidxA)] + localA
            mB = group_offset[(oid, tB)][(ccB, fidxB)] + localB
            if tA == tB:
                intra_extra[(oid, tA)].append((mA, mB))
                forced_keep_by_group[(oid, tA)].add(mA)
                forced_keep_by_group[(oid, tA)].add(mB)
            else:
                pending.append((oid, (tA, mA), (tB, mB)))
                forced_keep_by_group[(oid, tA)].add(mA)
                forced_keep_by_group[(oid, tB)].add(mB)

    anchor_needed: dict[tuple[int, ChunkCoords], dict[int, tuple]] = defaultdict(dict)
    for p_id, (oid, (tA, mA), (tB, mB)) in enumerate(pending):
        anchor_needed[(oid, tA)][mA] = (p_id, 0)
        anchor_needed[(oid, tB)][mB] = (p_id, 1)

    # Map dense OID → original (e.g. flywire) segment id, so coarse-level
    # fragments carry the SAME per-fragment ``segment_id`` as level 0 (which
    # drives both colour-matching with the flat segmentation and the picked
    # global id).  Without this, coarse fragments would be tagged with the
    # dense OID instead of the real segment id.  Falls back to the OID when
    # the source level has no ``object_attributes/segment_id``.
    try:
        segid_of_oid = read_object_attributes(src, "segment_id")
    except ArrayError:
        segid_of_oid = None

    # --- create target level + arrays (vertex_count patched after workers) -
    # Skeleton vertices stay in absolute world coordinates at every level
    # (we decimate, we do NOT rescale), so the level transform is the
    # identity: bin_ratio=1 (scale=1), bin_shape=base_bin.  ``vertex_count``
    # is only known after the workers decimate, so create with a placeholder
    # and patch the level attrs afterwards.
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
    # Mirror the store's world coordinate offset onto this level's NGFF
    # transform (vertices stay in the shifted/stored frame at every level).
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
    for name in attr_names:
        create_attribute_array(level_group, name, dtype=str(attr_dtypes[name]))

    # --- per-target-chunk plans (metadata only) → dispatch via executor ----
    # Each target chunk's groups are ordered by ascending OID — the same order
    # the serial path appended fragments — so the gathered object-index records
    # / manifests are independent of worker completion order (serial output ==
    # parallel output).  Each worker reads only its own children and writes
    # disjoint chunk files.
    target_chunks = sorted({tcc for (_oid, tcc) in group_members})
    groups_by_tcc: dict[ChunkCoords, list[dict[str, Any]]] = {}
    for tcc in target_chunks:
        groups: list[dict[str, Any]] = []
        for oid in keep_oids:
            key = (oid, tcc)
            members = group_members.get(key)
            if not members:
                continue
            groups.append({
                "oid": int(oid),
                "segment_id": (
                    int(segid_of_oid[oid]) if segid_of_oid is not None else int(oid)
                ),
                "members": [(list(cc), int(fidx)) for cc, fidx in members],
                "intra_extra": [
                    [int(a), int(b)] for a, b in intra_extra.get(key, [])
                ],
                "forced": sorted(int(x) for x in forced_keep_by_group.get(key, set())),
                "anchor_needed": {
                    int(mv): [int(tag[0]), int(tag[1])]
                    for mv, tag in anchor_needed.get(key, {}).items()
                },
            })
        groups_by_tcc[tcc] = groups

    # Level-wide plan is shared (scattered once); each per-chunk payload is just
    # its target-chunk coordinate, so few/dense-chunk levels are no longer
    # dominated by re-pickling the whole plan into every task.
    shared = {
        "store_path": str(store_path),
        "source_level": int(source_level),
        "target_level": int(target_level),
        "groups_by_tcc": groups_by_tcc,
        "attr_names": list(attr_names),
        "attr_dtypes": {n: str(attr_dtypes[n]) for n in attr_names},
        "stride": int(stride),
        "attr_agg": attr_agg,
        "ndim": int(ndim),
    }
    payloads = [{"tcc": list(tcc)} for tcc in target_chunks]

    _timings["setup"] = _time.perf_counter() - _t0
    _tmap = _time.perf_counter()
    results = list(executor(_coarsen_target_chunk, payloads, shared))
    _timings["map"] = _time.perf_counter() - _tmap
    _tfin = _time.perf_counter()

    # --- gather worker outputs ------------------------------------------
    anchor_locs: dict[Any, tuple[ChunkCoords, int]] = {}
    new_manifests = defaultdict(list)
    total_out_vertices = 0
    total_fragments = 0
    for res in results:
        for tag, loc in res["anchor_locs"].items():
            anchor_locs[tag] = loc
        for seg, cc, fidx in res["records"]:
            new_manifests[seg].append((cc, fidx))
            total_fragments += 1
        total_out_vertices += int(res["vertex_count"])

    # Patch the level's vertex_count now that the workers have produced it
    # (re-writes the level group attrs only; arrays + chunk data untouched).
    level_meta.vertex_count = int(total_out_vertices)
    create_resolution_level(root, target_level, level_meta)

    # --- re-emit cross-chunk links still spanning target chunks ---------
    # Stamp ``sid_ndim`` at creation so coarse levels with zero surviving
    # cross-chunk links still carry it (``write_cross_chunk_links`` only
    # stamps it when it actually writes records).
    create_cross_chunk_links_array(level_group, delta=0, link_width=2, sid_ndim=ndim)
    target_links = []
    for p_id, (oid, (tA, mA), (tB, mB)) in enumerate(pending):
        la = anchor_locs.get((p_id, 0))
        lb = anchor_locs.get((p_id, 1))
        if la is None or lb is None or tuple(la[0]) == tuple(lb[0]):
            continue
        target_links.append((la, lb))
    if target_links:
        write_cross_chunk_links(level_group, target_links, sid_ndim=ndim, delta=0)

    # --- object index (gap-filled to inherited OID space) ---------------
    write_object_index(
        level_group, dict(new_manifests), sid_ndim=ndim, total_objects=n_src,
    )

    # --- carry object attributes forward with present_mask --------------
    src_attr_names = []
    if "object_attributes" in src:
        src_attr_names = [n for n in src["object_attributes"]]
    mask = np.zeros(n_src, dtype=np.uint8)
    for o in keep_oids:
        mask[o] = 1
    for aname in src_attr_names:
        try:
            src_data = read_object_attributes(src, aname)
        except ArrayError:
            continue
        out = np.zeros_like(src_data)
        out[keep_oids] = src_data[keep_oids]
        create_object_attributes_array(level_group, aname, dtype=str(src_data.dtype))
        write_object_attributes(level_group, aname, out, present_mask=mask)

    _timings["finalize"] = _time.perf_counter() - _tfin
    _timings["total"] = _time.perf_counter() - _t0
    _timings["n_target_chunks"] = len(target_chunks)

    return {
        "vertex_count": int(total_out_vertices),
        "fragment_count": int(total_fragments),
        "object_count": len(keep_oids),
        "objects_kept": len(keep_oids),
        "source_objects": n_src,
        "cross_chunk_edges": len(target_links),
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
            executor=executor,
        ))
    return {"levels": summaries, "num_levels": n + 1}
