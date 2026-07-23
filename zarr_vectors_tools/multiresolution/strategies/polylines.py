"""Polyline and streamline coarsening strategies.

Two complementary approaches:

1. **Douglas-Peucker simplification**: reduces vertex count within each
   polyline while preserving shape.  Good for reducing per-streamline
   resolution while keeping all streamlines.

2. **Spatial subsampling**: selects a representative subset of
   streamlines per spatial bin.  Good for reducing total streamline
   count while keeping full vertex resolution on survivors.

Both can be composed: simplify first, then subsample.
"""

from __future__ import annotations

import pickle
import shutil
import tempfile
from contextlib import nullcontext
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


def simplify_polyline(
    vertices: npt.NDArray[np.floating],
    epsilon: float,
) -> npt.NDArray[np.floating]:
    """Simplify a polyline using Douglas-Peucker algorithm.

    Args:
        vertices: ``(N, D)`` ordered vertex positions.
        epsilon: Maximum perpendicular distance threshold.  Larger
            values produce more aggressive simplification.

    Returns:
        ``(M, D)`` simplified polyline with M ≤ N vertices.
        First and last vertices are always preserved.
    """
    n = len(vertices)
    if n <= 2:
        return vertices.copy()

    keep = np.zeros(n, dtype=bool)
    keep[0] = True
    keep[-1] = True

    _dp_recurse(vertices, 0, n - 1, epsilon, keep)

    return vertices[keep].copy()


def decimate_polyline(
    vertices: npt.NDArray[np.floating],
    stride: int,
) -> npt.NDArray[np.floating]:
    """Uniformly decimate a polyline, keeping every ``stride``-th vertex.

    The linear-path equivalent of skeleton decimation
    (:func:`zarr_vectors_tools.multiresolution.strategies.skeletons.decimate_skeleton`):
    since a polyline has no branch points, its only anchors are the two
    endpoints, which are always kept.

    Args:
        vertices: ``(N, D)`` ordered vertex positions.
        stride: Keep every ``stride``-th interior vertex, counted from the
            first vertex.  ``stride <= 1`` keeps every vertex.

    Returns:
        ``(M, D)`` decimated polyline with M ≤ N vertices.  First and last
        vertices are always preserved.
    """
    n = len(vertices)
    if n <= 2 or stride <= 1:
        return vertices.copy()

    keep = np.zeros(n, dtype=bool)
    keep[0] = True
    keep[-1] = True
    keep[0:n:stride] = True

    return vertices[keep].copy()


def _dp_recurse(
    vertices: npt.NDArray,
    start: int,
    end: int,
    epsilon: float,
    keep: npt.NDArray[np.bool_],
) -> None:
    """Recursive Douglas-Peucker step."""
    if end - start <= 1:
        return

    # Line segment from vertices[start] to vertices[end]
    line_start = vertices[start]
    line_end = vertices[end]
    line_vec = line_end - line_start
    line_len_sq = np.dot(line_vec, line_vec)

    if line_len_sq < 1e-30:
        # Degenerate segment — keep the farthest point
        dists = np.sum((vertices[start + 1 : end] - line_start) ** 2, axis=1)
        max_idx = start + 1 + np.argmax(dists)
        if np.sqrt(dists[max_idx - start - 1]) > epsilon:
            keep[max_idx] = True
            _dp_recurse(vertices, start, max_idx, epsilon, keep)
            _dp_recurse(vertices, max_idx, end, epsilon, keep)
        return

    # Perpendicular distances from each interior point to the line
    points = vertices[start + 1 : end]
    t = np.clip(
        np.dot(points - line_start, line_vec) / line_len_sq,
        0.0, 1.0,
    )
    projections = line_start + t[:, np.newaxis] * line_vec
    dists = np.sqrt(np.sum((points - projections) ** 2, axis=1))

    max_local = np.argmax(dists)
    max_dist = dists[max_local]
    max_idx = start + 1 + max_local

    if max_dist > epsilon:
        keep[max_idx] = True
        _dp_recurse(vertices, start, max_idx, epsilon, keep)
        _dp_recurse(vertices, max_idx, end, epsilon, keep)


def simplify_polylines(
    polylines: list[npt.NDArray[np.floating]],
    epsilon: float,
    *,
    min_vertices: int = 2,
) -> list[npt.NDArray[np.floating]]:
    """Simplify a list of polylines using Douglas-Peucker.

    Args:
        polylines: List of ``(N_k, D)`` arrays.
        epsilon: Distance threshold.
        min_vertices: Minimum vertices to keep per polyline.

    Returns:
        List of simplified polylines.
    """
    result: list[npt.NDArray] = []
    for poly in polylines:
        simplified = simplify_polyline(poly, epsilon)
        if len(simplified) < min_vertices:
            # Keep first and last at minimum
            if len(poly) >= min_vertices:
                indices = np.linspace(0, len(poly) - 1, min_vertices, dtype=int)
                simplified = poly[indices]
            else:
                simplified = poly.copy()
        result.append(simplified)
    return result


def subsample_polylines(
    polylines: list[npt.NDArray[np.floating]],
    bin_size: float | tuple[float, ...],
    *,
    max_per_bin: int = 1,
    selection: str = "longest",
) -> dict[str, Any]:
    """Spatially subsample polylines, keeping representatives per bin.

    Assigns each polyline to the spatial bin containing its midpoint,
    then selects up to ``max_per_bin`` representatives per bin.

    Args:
        polylines: List of ``(N_k, D)`` arrays.
        bin_size: Spatial bin edge length.
        max_per_bin: How many polylines to keep per bin.
        selection: How to pick representatives:
            ``"longest"``: keep the longest polyline(s).
            ``"random"``: keep random polyline(s).
            ``"first"``: keep the first polyline(s) by index.

    Returns:
        Dict with:
        - ``polylines``: subsampled list of arrays
        - ``indices``: original indices of kept polylines
        - ``polyline_count``: number kept
        - ``reduction_ratio``: original / kept
    """
    n_total = len(polylines)
    if n_total == 0:
        return {
            "polylines": [],
            "indices": np.array([], dtype=np.int64),
            "polyline_count": 0,
            "reduction_ratio": 0,
        }

    ndim = polylines[0].shape[1]

    if isinstance(bin_size, (int, float)):
        bin_sizes = np.array([float(bin_size)] * ndim)
    else:
        bin_sizes = np.array(bin_size, dtype=np.float64)

    # Compute midpoint for each polyline
    midpoints = np.array([
        poly[len(poly) // 2] for poly in polylines
    ], dtype=np.float64)

    # Assign to bins
    bin_indices = np.floor(midpoints / bin_sizes).astype(np.int64)

    # Group by bin
    bin_to_polys: dict[tuple, list[int]] = {}
    for i in range(n_total):
        key = tuple(bin_indices[i].tolist())
        if key not in bin_to_polys:
            bin_to_polys[key] = []
        bin_to_polys[key].append(i)

    # Select representatives
    kept_indices: list[int] = []
    for bin_key, members in bin_to_polys.items():
        if len(members) <= max_per_bin:
            kept_indices.extend(members)
            continue

        if selection == "longest":
            lengths = [len(polylines[m]) for m in members]
            sorted_members = [m for _, m in sorted(
                zip(lengths, members), reverse=True
            )]
            kept_indices.extend(sorted_members[:max_per_bin])
        elif selection == "random":
            rng = np.random.default_rng()
            chosen = rng.choice(members, size=max_per_bin, replace=False)
            kept_indices.extend(chosen.tolist())
        elif selection == "first":
            kept_indices.extend(members[:max_per_bin])
        else:
            kept_indices.extend(members[:max_per_bin])

    kept_indices.sort()
    kept_polys = [polylines[i] for i in kept_indices]

    return {
        "polylines": kept_polys,
        "indices": np.array(kept_indices, dtype=np.int64),
        "polyline_count": len(kept_polys),
        "reduction_ratio": n_total / max(len(kept_polys), 1),
    }


def coarsen_polylines(
    polylines: list[npt.NDArray[np.floating]],
    *,
    simplify_epsilon: float | None = None,
    subsample_bin_size: float | None = None,
    max_per_bin: int = 1,
    min_vertices: int = 2,
    selection: str = "longest",
) -> dict[str, Any]:
    """Combined polyline coarsening: simplify then subsample.

    Args:
        polylines: Input polylines.
        simplify_epsilon: Douglas-Peucker epsilon.  None to skip.
        subsample_bin_size: Spatial subsampling bin size.  None to skip.
        max_per_bin: Polylines to keep per bin (for subsampling).
        min_vertices: Minimum vertices per polyline (for simplification).
        selection: Subsampling selection mode.

    Returns:
        Dict with:
        - ``polylines``: coarsened polylines
        - ``vertex_count``: total vertices
        - ``polyline_count``: number of polylines
        - ``simplification_ratio``: vertex reduction from DP
        - ``subsampling_ratio``: polyline reduction from subsampling
    """
    n_input = len(polylines)
    v_input = sum(len(p) for p in polylines)
    current = polylines

    simplification_ratio = 1.0
    subsampling_ratio = 1.0

    # Step 1: simplify
    if simplify_epsilon is not None:
        current = simplify_polylines(
            current, simplify_epsilon, min_vertices=min_vertices,
        )
        v_after = sum(len(p) for p in current)
        simplification_ratio = v_input / max(v_after, 1)

    # Step 2: subsample
    kept_indices: npt.NDArray | None = None
    if subsample_bin_size is not None:
        sub_result = subsample_polylines(
            current, subsample_bin_size,
            max_per_bin=max_per_bin,
            selection=selection,
        )
        current = sub_result["polylines"]
        kept_indices = sub_result["indices"]
        subsampling_ratio = sub_result["reduction_ratio"]

    return {
        "polylines": current,
        "vertex_count": sum(len(p) for p in current),
        "polyline_count": len(current),
        "simplification_ratio": simplification_ratio,
        "subsampling_ratio": subsampling_ratio,
        "kept_indices": kept_indices,
    }


# ===================================================================
# Chunk-local, executor-parallel pyramid coarsening
# ===================================================================
#
# Unlike the two helpers above (pure functions operating on in-memory
# polyline lists), everything below reads/writes a zarr-vectors store
# directly, one TARGET chunk at a time, so peak memory is O(one target
# chunk's fan-in) instead of O(the whole source level) — the fix for the
# OOM the whole-level ``_polyline_coarsen`` in ``coarsen.py`` hit on a
# 5M-streamline TRK file.  Structure mirrors
# :func:`zarr_vectors_tools.multiresolution.strategies.skeletons.coarsen_skeleton_level`
# (Phase A per-target-chunk decimate+write+sidecar, Phase B cross-target
# link stitching by shard, Phase C sharded object-index reduce), but
# simplified for the streamline convention:
#
# - One streamline == one object; its per-chunk ``segment_id`` fragment
#   attribute *is* the (stable, dense) object id — no separate
#   ``object_id`` attribute needed.
# - Streamlines carry no branch structure, so intra-object connectivity
#   is a simple chain, not a general graph.
# - Every fragment boundary (source-level ``_build_manifests_and_cross_links``)
#   is stored as an EXPLICIT, DIRECTED links/0 record with a non-zero
#   offsets segment (predecessor endpoint 0 -> successor endpoint 1).  This lets a
#   target-chunk worker recover object order (and walk direction) purely
#   from local reads, without ever touching a global object manifest —
#   geometric boundary-vertex coincidence (as skeletons.py falls back to)
#   is not needed at all, and direction (which geometric coincidence
#   alone cannot supply) is read straight off the source link.


def _build_local_polyline_plan(
    src,
    tcc: tuple[int, ...],
    *,
    scale: tuple[int, ...],
    ndim: int,
    keep_mask: npt.NDArray[np.uint8] | None,
    ccl_cells: list | None = None,
) -> tuple[list[dict], dict]:
    """Build ONE target chunk's coarsen plan by reading only its source children.

    Reads the source children ``scc ∈ [tcc·scale, (tcc+1)·scale)``: per-fragment
    vertices + ``segment_id`` (the object id), and the source-level directed
    cross-chunk links (delta=0) incident on those children.  A link whose both
    endpoints resolve inside this target chunk chains two fragments of the same
    object into one run (in predecessor -> successor order); a link with only
    one endpoint inside is a genuine cross-target transition, and the inside
    fragment's role (predecessor = about to exit, successor = just entered)
    comes straight from which side of the record it was on.

    Returns ``(groups, vcache)``: ``groups`` is one dict per surviving *run*
    (an object may contribute more than one run to a target chunk if it visits
    and leaves more than once) with keys ``oid``, ``members`` (ordered list of
    ``(chunk_coords, fragment_idx)``), ``head_cross``/``tail_cross`` (whether
    the run's first/last vertex is a cross-target anchor Phase B must stitch).
    ``vcache`` is the per-fragment vertex read cache (reused by the caller, no
    double read).
    """
    from itertools import product

    from zarr_vectors.core.arrays import (
        read_chunk_fragment_attributes,
        read_chunk_vertices,
    )
    from zarr_vectors.core.arrays import read_links_for_tuple
    from zarr_vectors.exceptions import ArrayError

    tcc = tuple(int(x) for x in tcc)
    child_ccs = [
        tuple(tcc[a] * scale[a] + d[a] for a in range(ndim))
        for d in product(*[range(scale[a]) for a in range(ndim)])
    ]

    vcache: dict = {}
    fragoid: dict = {}
    child_ranges: dict = {}  # scc -> (starts, ends, fidxs) for local->fragment resolve
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

        starts = []
        st = 0
        for fidx in range(len(vgroups)):
            cnt = len(vgroups[fidx])
            starts.append((st, st + cnt, fidx))
            st += cnt
            oid = int(segs[fidx]) if segs is not None and fidx < len(segs) else -1
            if oid < 0:
                continue
            if keep_mask is not None:
                if oid >= len(keep_mask) or int(keep_mask[oid]) == 0:
                    continue
            vcache[(scc, fidx)] = vgroups[fidx]
            fragoid[(scc, fidx)] = oid
        child_ranges[scc] = (
            np.asarray([r[0] for r in starts], np.int64),
            np.asarray([r[1] for r in starts], np.int64),
            np.asarray([r[2] for r in starts], np.int64),
        )

    def _resolve(scc, vi):
        ranges = child_ranges.get(scc)
        if ranges is None:
            return None
        s, e, f = ranges
        if len(s) == 0:
            return None
        i = int(np.searchsorted(s, vi, side="right")) - 1
        if i < 0 or vi >= int(e[i]):
            return None
        return int(f[i]), int(vi - s[i])

    # Directed intra/cross classification from stored source cross-chunk
    # links: both endpoints resolve inside this target chunk -> intra-target
    # chain edge; exactly one resolves inside -> that fragment's matching end
    # (last vertex if it was the predecessor slot, first vertex if it was the
    # successor slot) is a cross-target anchor for Phase B, tagged with the
    # OTHER target chunk it connects to AND the source link's own identity
    # (``link_key``).  ``split_polyline_at_boundaries`` does not duplicate a
    # coincident vertex at a chunk crossing — a fragment boundary is simply
    # wherever an ORIGINAL vertex happens to fall — so, unlike skeletons.py's
    # Phase B, there is no shared coordinate for two independent workers to
    # match on.  What they CAN both compute identically is the exact source
    # record ``((ccA, viA), (ccB, viB))``: cross-chunk-link storage is keyed
    # by the sorted-unique chunk set, so a query from either side's own
    # children reads the identical physical record — that raw tuple is the
    # join key Phase B stitches on.  It also means a transition can land in
    # ANY other chunk, not just a face neighbor, so Phase B is told the exact
    # partner chunk rather than discovering it via a grid-adjacency scan.
    child_set = set(child_ranges)
    intra_next: dict = {}
    cross_out: dict = {}  # member -> (partner tcc, link_key) its LAST vertex exits via
    cross_in: dict = {}   # member -> (partner tcc, link_key) its FIRST vertex entered via
    # ``ccl_cells`` are the source cross-chunk-link cells touching this target's
    # children (enumerated once by the coordinator and bucketed per target).
    for cell in (ccl_cells or ()):
        try:
            recs = read_links_for_tuple(src, cell, delta=0)
        except Exception:
            continue
        for rec in recs:
            if len(rec) != 2:
                continue
            (ccA, viA), (ccB, viB) = rec
            ccA = tuple(int(x) for x in ccA)
            ccB = tuple(int(x) for x in ccB)
            viA = int(viA)
            viB = int(viB)
            inA = ccA in child_set
            inB = ccB in child_set
            if not inA and not inB:
                continue
            rA = _resolve(ccA, viA) if inA else None
            rB = _resolve(ccB, viB) if inB else None
            memberA = (ccA, rA[0]) if rA is not None else None
            memberB = (ccB, rB[0]) if rB is not None else None
            if memberA is not None and memberA not in vcache:
                memberA = None
            if memberB is not None and memberB not in vcache:
                memberB = None
            if memberA is None and memberB is None:
                continue
            if memberA is not None and memberB is not None:
                if fragoid.get(memberA) != fragoid.get(memberB):
                    continue
                intra_next[memberA] = memberB
            elif memberA is not None:
                partner_tcc = tuple(int(ccB[a]) // scale[a] for a in range(ndim))
                link_key = (ccA, viA, ccB, viB)
                cross_out[memberA] = (partner_tcc, link_key)
            else:
                partner_tcc = tuple(int(ccA[a]) // scale[a] for a in range(ndim))
                link_key = (ccA, viA, ccB, viB)
                cross_in[memberB] = (partner_tcc, link_key)

    members_by_oid: dict[int, list] = defaultdict(list)
    for m, oid in fragoid.items():
        if m in vcache:
            members_by_oid[oid].append(m)

    intra_targets = set(intra_next.values())
    groups: list[dict] = []
    for oid in sorted(members_by_oid):
        members = set(members_by_oid[oid])
        visited: set = set()
        heads = [m for m in sorted(members) if m not in intra_targets]
        for head in heads:
            if head in visited:
                continue
            chain = [head]
            visited.add(head)
            cur = head
            while (
                cur in intra_next
                and intra_next[cur] in members
                and intra_next[cur] not in visited
            ):
                cur = intra_next[cur]
                chain.append(cur)
                visited.add(cur)
            head_in = cross_in.get(chain[0])
            tail_out = cross_out.get(chain[-1])
            groups.append({
                "oid": int(oid),
                "members": chain,
                "head_partner": head_in[0] if head_in else None,
                "head_link": head_in[1] if head_in else None,
                "tail_partner": tail_out[0] if tail_out else None,
                "tail_link": tail_out[1] if tail_out else None,
            })
        # Defensive: a cycle in (corrupt) source data would leave members
        # unreached by the head walk above; emit them as singleton runs
        # rather than silently dropping vertices.
        for m in sorted(members - visited):
            head_in = cross_in.get(m)
            tail_out = cross_out.get(m)
            groups.append({
                "oid": int(oid),
                "members": [m],
                "head_partner": head_in[0] if head_in else None,
                "head_link": head_in[1] if head_in else None,
                "tail_partner": tail_out[0] if tail_out else None,
                "tail_link": tail_out[1] if tail_out else None,
            })

    return groups, vcache


def _coarsen_polyline_target_chunk(payload: dict, shared: dict | None = None) -> dict:
    """Coarsen ONE target chunk — a picklable worker for parallel pyramiding.

    Reads only this target chunk's source children (via
    :func:`_build_local_polyline_plan`), decimates/simplifies each surviving
    run, writes the target chunk's vertices + ``segment_id``, and spills (a)
    object-index rows sharded by OID for Phase C and (b) cross-target anchor
    rows (with role + resolved local vertex index) for Phase B — all without
    ever holding more than one target chunk's fan-in in memory.
    """
    from zarr_vectors.core.arrays import write_chunk_fragment_attributes, write_chunk_vertices
    from zarr_vectors.core.store import get_resolution_level, open_store

    shared = shared or {}
    ndim = int(shared["ndim"])
    scale = tuple(int(s) for s in shared["scale"])
    tcc = tuple(int(x) for x in payload["tcc"])
    keep_mask = shared.get("keep_mask")
    coarsen_mode = shared["coarsen_mode"]
    stride = int(shared.get("stride", 1))
    simplify_epsilon = shared.get("simplify_epsilon")

    root = open_store(shared["store_path"], mode="r+")
    src = get_resolution_level(root, shared["source_level"])
    level_group = get_resolution_level(root, shared["target_level"])

    groups, vcache = _build_local_polyline_plan(
        src, tcc, scale=scale, ndim=ndim, keep_mask=keep_mask,
        ccl_cells=payload.get("ccl_cells"),
    )
    input_fragments = int(len(vcache))
    input_vertices = int(sum(len(v) for v in vcache.values()))
    input_objects = int(len({g["oid"] for g in groups}))

    pieces: list[tuple[int, npt.NDArray]] = []
    # sidecar row = (*ccA, viA, *ccB, viB, role, local_vi); role 0=pred(exit)
    # 1=succ(entry).  (ccA, viA, ccB, viB) is the SOURCE link's own identity
    # (see _build_local_polyline_plan) — the join key Phase B stitches two
    # target chunks' sidecars on, since polyline fragments have no shared
    # coincident coordinate at a boundary to match on geometrically.
    sc_rows: list[tuple[int, ...]] = []
    partners: set[tuple[int, ...]] = set()
    total_out_vertices = 0
    chunk_offset = 0
    for g in groups:
        parts = [vcache[m] for m in g["members"]]
        merged = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
        if len(merged) == 0:
            continue
        if coarsen_mode == "decimate":
            rpos = decimate_polyline(merged, stride)
        elif simplify_epsilon is None or simplify_epsilon <= 0:
            # coarsen_factor <= 1: rdp is a no-op, keep full vertex resolution
            # (matches decimate's stride-1 identity — the level is then a pure
            # sparser subset of full-resolution streamlines, not a decimated
            # one).  Guard also stops a None epsilon reaching the DP recursion.
            rpos = merged
        else:
            simp = simplify_polylines([merged], simplify_epsilon, min_vertices=2)
            rpos = simp[0] if simp else merged[:2]
        if len(rpos) == 0:
            continue

        local_first = chunk_offset
        local_last = chunk_offset + len(rpos) - 1
        if g["head_link"] is not None:
            ccA, viA, ccB, viB = g["head_link"]
            sc_rows.append((*ccA, viA, *ccB, viB, 1, local_first))
            partners.add(g["head_partner"])
        if g["tail_link"] is not None:
            ccA, viA, ccB, viB = g["tail_link"]
            sc_rows.append((*ccA, viA, *ccB, viB, 0, local_last))
            partners.add(g["tail_partner"])

        pieces.append((g["oid"], rpos))
        chunk_offset += len(rpos)
        total_out_vertices += len(rpos)

    recs: list[tuple[int, int]] = []
    if pieces:
        # record_presence=False: these run in parallel worker PROCESSES, and
        # nonempty_chunks is an array-wide attribute, so stamping it here would
        # race across workers (a Windows hard-fail on the zarr.json rename).
        # The coordinator's rebuild_nonempty_manifests pass re-derives the
        # manifests once, after Phase A, from the on-disk cells.
        write_chunk_vertices(
            level_group, tcc, [p[1] for p in pieces], dtype=np.float32,
            record_presence=False,
        )
        seg_ids = np.array([p[0] for p in pieces], dtype=np.uint64)
        write_chunk_fragment_attributes(
            level_group, "segment_id", tcc, seg_ids, dtype=np.uint64,
            record_presence=False,
        )
        recs = [(int(p[0]), fidx) for fidx, p in enumerate(pieces)]

    # Spill object-index refs partitioned by OID shard (Phase C reduces these
    # without ever gathering a whole-level {oid: manifest} structure).
    oid_shards = int(shared.get("oid_reduce_shards", 1) or 1)
    oid_tmp_dir = str(shared.get("oid_reduce_tmp_dir", ""))
    shard_rows: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for oid, fidx in recs:
        shard_rows[oid % oid_shards].append((oid, fidx))
    shard_files: list[tuple[int, str]] = []
    if oid_tmp_dir and shard_rows:
        tcc_tag = ".".join(str(int(x)) for x in tcc)
        for shard, rows in shard_rows.items():
            arr = np.asarray(rows, dtype=np.int64).reshape(-1, 2)
            path = Path(oid_tmp_dir) / f"{tcc_tag}.s{int(shard)}.npy"
            np.save(path, arr, allow_pickle=False)
            shard_files.append((int(shard), str(path)))

    # Cross-target anchor rows travel back inline in the result dict rather
    # than through a per-chunk .npy sidecar: the coordinator forwards them
    # straight into the Phase B payloads.  The total is O(cross-chunk
    # crossings) — far smaller than vertex volume — so it fits in RAM, and
    # this removes thousands of tiny-file writes/reads/deletes (a real
    # serial cost, and a Windows handle-churn hazard).
    anchors = (
        np.asarray(sc_rows, dtype=np.int64).reshape(-1, 2 * ndim + 4)
        if sc_rows else None
    )

    return {
        "tcc": tcc,
        "input_fragments": input_fragments,
        "input_vertices": input_vertices,
        "input_objects": input_objects,
        "fragment_count": int(len(recs)),
        "oid_shards": shard_files,
        "anchors": anchors,
        "partners": [list(p) for p in partners],
        "vertex_count": int(total_out_vertices),
    }


def _polyline_cross_edge_shard(payload: dict, shared: dict | None = None) -> dict:
    """Phase B worker: write directed cross-target-chunk links for ONE ccl shard.

    Each task owns the adjacent target-chunk pairs whose ``k2`` cells fall in a
    single outer shard (the coordinator partitions pairs by shard, so writers
    never collide).  For each pair it matches coincident ``(segment_id, coord)``
    anchors from the two chunks' sidecars, using the ``role`` each anchor was
    tagged with in Phase A (0=predecessor/exit, 1=successor/entry) to orient
    the written link — this is what a plain undirected coincidence match
    (as skeletons.py Phase B does) cannot supply on its own.
    """
    from zarr_vectors.core.arrays import write_link_cells
    from zarr_vectors.core.store import get_resolution_level, open_store

    shared = shared or {}
    ndim = int(shared["ndim"])
    # Anchor rows arrive inline (see _coarsen_polyline_target_chunk), keyed by
    # source target-chunk; no per-chunk .npy sidecar to load.
    sidecar_arrays = {
        tuple(int(x) for x in e["tcc"]): e["arr"]
        for e in payload.get("anchors", [])
    }
    pairs = payload["pairs"]
    link_w = 2 * ndim + 2

    # Sorted-key equi-join, entirely in NumPy — no per-row Python objects.
    # A single chunk's sidecar can hold MILLIONS of rows at whole-brain
    # scale; a `dict[tuple -> list]` keymap there costs ~500-600 bytes of
    # pure Python object overhead per row (measured ~1.2GB for 2M rows),
    # enough alone to blow a worker's memory budget.  Each row's key (the
    # source link's own identity, see _build_local_polyline_plan) is
    # unique within one chunk's sidecar by construction — a chunk only
    # emits ONE endpoint per source cross-chunk-link record — so this is a
    # simple sorted-array equi-join: sort both chunks' keys (viewed as
    # opaque fixed-width blobs, ~30 bytes/row including the sort's working
    # arrays), binary-search one into the other, keep exact hits.
    sorted_cache: dict[tuple[int, ...], tuple | None] = {}

    def _load_sorted(cc: tuple[int, ...]):
        if cc in sorted_cache:
            return sorted_cache[cc]
        raw = sidecar_arrays.get(cc)
        if raw is None:
            sorted_cache[cc] = None
            return None
        arr = np.asarray(raw, dtype=np.int64)
        if arr.size == 0:
            sorted_cache[cc] = None
            return None
        keys = np.ascontiguousarray(arr[:, :link_w])
        void_keys = keys.view(np.dtype((np.void, keys.dtype.itemsize * link_w))).reshape(-1)
        order = np.argsort(void_keys, kind="stable")
        result = (void_keys[order], arr[order, link_w], arr[order, link_w + 1])
        sorted_cache[cc] = result
        return result

    links: list = []
    for A_, B_ in pairs:
        A = tuple(int(x) for x in A_)
        B = tuple(int(x) for x in B_)
        da = _load_sorted(A)
        db = _load_sorted(B)
        if da is None or db is None:
            continue
        keysA, rolesA, visA = da
        keysB, rolesB, visB = db
        pos = np.searchsorted(keysB, keysA)
        in_range = pos < len(keysB)
        pos_clamped = np.where(in_range, pos, 0)
        hit = in_range & (keysB[pos_clamped] == keysA)
        if not np.any(hit):
            continue
        idxA = np.flatnonzero(hit)
        idxB = pos[idxA]
        ra, va = rolesA[idxA], visA[idxA]
        rb, vb = rolesB[idxB], visB[idxB]
        stitch = ra != rb  # same-role anchors can't stitch (data anomaly)
        idxA, idxB = idxA[stitch], idxB[stitch]
        ra, va = ra[stitch], va[stitch]
        vb = vb[stitch]
        a_is_pred = ra == 0
        for i in range(len(idxA)):
            if a_is_pred[i]:
                links.append([(A, int(va[i])), (B, int(vb[i]))])
            else:
                links.append([(B, int(vb[i])), (A, int(va[i]))])
    if links:
        root = open_store(shared["store_path"], mode="r+")
        level_group = get_resolution_level(root, shared["target_level"])
        # Writes only the cells these records touch and does NOT maintain
        # the family-wide counts; the coordinator's finalize_links pass
        # reconciles them once every worker is done.
        #
        # directed=True is load-bearing here: endpoint order IS the data
        # (predecessor -> successor, chosen by role above), so a canonical
        # sort would destroy it.
        write_link_cells(
            level_group, links, ndim, delta=0, link_width=2, directed=True,
        )
    return {"n_links": len(links)}


def coarsen_polyline_level(
    store_path: str,
    source_level: int,
    target_level: int,
    *,
    coarsen_factor: float = 1.0,
    sparsity_factor: float = 1.0,
    chunk_scale_factor: int | tuple[int, ...] = 1,
    sparsity_strategy: str = "random",
    sparsity_seed: int | None = None,
    coarsen_mode: str = "rdp",
    simplify_epsilon: float | None = None,
    compressor: Any = None,
    executor: Any = None,
) -> dict[str, Any]:
    """Coarsen one streamline/polyline level, chunk-local and executor-parallel.

    Memory-bounded replacement for the whole-level ``_polyline_coarsen`` in
    ``coarsen.py`` (which loaded every fragment, manifest, and per-object
    vertex sequence of the source level into memory at once — fine at 100K
    streamlines, an OOM at 5M).  Peak memory here is O(one target chunk's
    source fan-in), independent of dataset size; per-target-chunk work is
    dispatched through ``executor`` (a ``map``-like ``(func, items, shared)``
    callable) so it can run on a dask worker pool.  The default executor runs
    serially in-process, so serial output is identical to the parallel path
    by construction.

    Preserves every existing streamline-specific behavior: dense, stable
    object ids across levels (survivors keep their OID; dropped objects leave
    empty manifest slots), ``rdp``/``decimate`` mode selection, and directed
    (predecessor -> successor) cross-chunk links so walk-order tangents stay
    correct after coarsening.

    Args: see :func:`zarr_vectors_tools.multiresolution.coarsen.coarsen_level`
    for the shared parameter semantics.
    """
    from zarr_vectors.constants import CAP_PRESERVED_OBJECT_IDS
    from zarr_vectors.core.arrays import (
        OBJECT_INDEX,
        OBJECT_INDEX_LAYOUT_V1,
        VERTICES,
        _write_object_index_manifests,
        create_fragment_attribute_array,
        create_object_attributes_array,
        create_object_index_array,
        create_vertices_array,
        list_chunk_keys,
        read_all_object_manifests,
        read_chunk_fragment_attributes,
        read_object_attributes,
        write_object_attributes,
    )
    from zarr_vectors.core.metadata import LevelMetadata, get_level_chunk_shape
    from zarr_vectors.core.store import (
        create_resolution_level,
        get_resolution_level,
        open_store,
        read_level_metadata,
        read_root_metadata,
    )
    from zarr_vectors.core.arrays import (
        create_links_array,
        create_links_family,
        finalize_links,
    )
    from zarr_vectors_tools.multiresolution.constants import (
        CROSS_LINK_TASK_SHARD_AXIS,
    )
    from zarr_vectors_tools.multiresolution.coarsen import _stamp_root_capability
    from zarr_vectors_tools.multiresolution.object_selection import apply_sparsity
    from zarr_vectors_tools.multiresolution.strategies.skeletons import (
        _reduce_object_index_shard,
    )

    # Per-target-chunk work is dispatched through ``executor``; the default
    # runs serially in-process (mirrors coarsen_skeleton_level's contract).
    if executor is None:
        def executor(func, items, shared=None):
            return [func(it, shared=shared) for it in items]

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
    target_chunk_shape = tuple(float(s) * int(r) for s, r in zip(src_chunk_shape, scale))
    same_as_root = all(
        abs(t - r) < 1e-9 for t, r in zip(target_chunk_shape, root_meta.chunk_shape)
    )
    chunk_shape_override = None if same_as_root else target_chunk_shape

    # ``coarsen_factor <= 1`` is the documented identity (no vertex reduction)
    # for BOTH modes — see coarsen_level's "1.0 is the identity" contract.
    # Decimate already collapses to stride 1 (a straight copy); rdp must match
    # rather than simplify at a chunk-derived epsilon, so leave the epsilon
    # unset (the worker treats an unset/non-positive epsilon as a no-op) and
    # only derive one when the factor genuinely asks for reduction.
    if coarsen_mode == "rdp" and simplify_epsilon is None and coarsen_factor > 1.0:
        simplify_epsilon = min(src_chunk_shape) * 0.5 * float(coarsen_factor)
    stride = max(1, int(round(coarsen_factor)))

    coarsening_method = "polyline_decimate" if coarsen_mode == "decimate" else "polyline_rdp"

    src_index_meta = src.read_array_meta(OBJECT_INDEX)
    n_src = int(src_index_meta.get("num_objects", 0))
    if n_src == 0:
        return {"vertex_count": 0, "object_count": 0, "method": coarsening_method}

    src_vertex_chunks = list_chunk_keys(src, VERTICES)
    if not src_vertex_chunks:
        return {"vertex_count": 0, "object_count": 0, "method": coarsening_method}

    # Hard requirement: every fragment boundary must be reconstructable
    # locally, which needs fragment_attributes/segment_id on the source.
    probe_seg = read_chunk_fragment_attributes(
        src, "segment_id", tuple(int(x) for x in src_vertex_chunks[0]),
        dtype=np.uint64, default=None,
    )
    if probe_seg is None:
        raise ValueError(
            "coarsen_polyline_level requires fragment_attributes/segment_id "
            "on the source level; re-ingest or rebuild the pyramid from level 0"
        )

    # --- Phase 0: sparsity keep-set (O(objects), never O(fragments) unless
    # no cheap per-object signal exists at a coarser source level) ---------
    if sparsity_factor > 1.0 and n_src > 1:
        # A missing object attribute raises StoreError on current core (older
        # cores raised KeyError); treat both as "length not present".
        from zarr_vectors.exceptions import StoreError
        length_arr = None
        try:
            length_arr = np.asarray(read_object_attributes(src, "length"), dtype=np.float64)
        except (KeyError, StoreError):
            length_arr = None

        if length_arr is not None:
            alive_mask = length_arr > 0
            lengths = length_arr if sparsity_strategy == "length" else None
        elif source_level == 0:
            alive_mask = np.ones(n_src, dtype=bool)
            lengths = None
            if sparsity_strategy == "length":
                raise ValueError(
                    "'length' sparsity strategy requires object_attributes/"
                    "length (enable compute_length at ingest)"
                )
        else:
            # No cheap per-object signal at this coarser source level — fall
            # back to an O(fragments) manifest read, exactly like
            # coarsen_skeleton_level's "length" fallback.  Only paid when
            # sparsity is active, not on the unconditional hot path.
            manifest_lens = np.array(
                [len(m) for m in read_all_object_manifests(src)], dtype=np.float64,
            )
            alive_mask = manifest_lens > 0
            lengths = manifest_lens if sparsity_strategy == "length" else None

        kept = apply_sparsity(
            n_src, 1.0 / sparsity_factor, sparsity_strategy,
            seed=sparsity_seed, lengths=lengths, alive_mask=alive_mask,
            # Cumulative: keep 1/sparsity_factor of the SURVIVING objects, so
            # a repeated factor sparsifies each level relative to the previous
            # one (503k -> 50k -> 5k -> ...), not relative to the original.
            relative_to="alive",
        )
        keep_mask = np.zeros(n_src, dtype=np.uint8)
        keep_mask[np.asarray(kept, dtype=np.int64)] = 1
    else:
        keep_mask = None

    target_chunks = sorted({
        tuple(int(cc[a] // scale[a]) for a in range(ndim))
        for cc in src_vertex_chunks
    })
    if not target_chunks:
        return {"vertex_count": 0, "object_count": 0, "method": coarsening_method}

    level_meta = LevelMetadata(
        level=target_level,
        vertex_count=0,
        # "fragment_attributes" must be listed: this coarsener always writes
        # fragment_attributes/segment_id (needed to reconstruct object
        # connectivity locally — see the module docstring).  Omitting it here
        # previously made neuroglancer's reader treat coarsened levels as
        # lacking per-fragment segment ids (gated on this exact list, see
        # datasource/zarr-vectors/frontend.ts's hasFragmentSegmentIds), so it
        # silently fell back to a meaningless chunk-local fragment index for
        # picking/selection at every level except the finest.
        arrays_present=["vertices", "object_index", "fragment_attributes"],
        bin_shape=tuple(
            float(b) * float(coarsen_factor) for b in root_meta.effective_bin_shape
        ),
        bin_ratio=tuple(max(1, int(round(coarsen_factor))) for _ in range(ndim)),
        chunk_shape=chunk_shape_override,
        object_sparsity=(1.0 / sparsity_factor),
        coarsening_method=coarsening_method,
        parent_level=source_level,
        preserves_object_ids=True,
        inherited_num_objects=n_src,
        shared_fragments=False,
    )
    level_group = create_resolution_level(root, target_level, level_meta)
    # A chunk array's codec pipeline is fixed when it is created, so the
    # compressor only has to be active around the create_* calls — every later
    # per-cell write (Phase A/B workers included, in their own processes)
    # encodes to match via zarr's array API.  The `with` block must CLOSE
    # before any worker dispatch: batched_writes defers array metas to its
    # flush, and a worker that reads an unflushed meta would see nothing.
    _codec_ctx = (
        level_group.batched_writes(compressor=compressor)
        if compressor else nullcontext()
    )
    with _codec_ctx:
        create_vertices_array(level_group, dtype="float32")
        create_object_index_array(level_group)
        create_fragment_attribute_array(
            level_group, "segment_id", dtype="uint64",
        )

    # Pre-create the kN cross-chunk-link arrays with LEVEL-WIDE dims so
    # Phase-B workers only WRITE cells (no create-race).
    tc_arr = np.asarray(target_chunks, dtype=np.int64)
    cmin = tc_arr.min(axis=0)
    cmax = tc_arr.max(axis=0)
    chunk_origin = tuple(int(min(0, int(cmin[a]))) for a in range(ndim))
    chunk_grid_shape = tuple(
        int(max(1, int(cmax[a]) - chunk_origin[a] + 1)) for a in range(ndim)
    )
    # Fix the family policy up front so the decentralized Phase-B workers
    # agree on it rather than racing to establish it.  This stamps the
    # links/0 GROUP only and materialises no offsets array, which is what
    # this family wants: polyline connectivity is implicit-sequential
    # within a fragment, so there are no intra records and the only arrays
    # that should ever appear here are the ones Phase B's records land in.
    # (An empty-but-existing array — zarr.json present, zero cells — makes
    # neuroglancer's reader 404 and fail the whole chunk download instead
    # of treating it as "no records", which cascades into the LOD picker
    # falling back to level 0.)
    #
    # directed=True because endpoint order is the data here: a record is
    # predecessor -> successor, and a canonical sort would destroy it.
    create_links_family(
        level_group, delta=0, link_width=2, sid_ndim=ndim, directed=True,
    )

    # Scale shard count with the dense OID space instead of a fixed 64: a
    # level with millions of surviving objects (e.g. a low-sparsity level
    # right after a high-sparsity one keeps most of its huge parent
    # population) needs more, smaller shards to keep each
    # _reduce_object_index_shard call's working set bounded — ~20k
    # objects/shard, capped so trivially small levels don't pay excess
    # file-handle overhead for no benefit.
    oid_reduce_shards = max(64, min(4096, -(-n_src // 20_000)))
    oid_reduce_tmp_dir = tempfile.mkdtemp(prefix=f"polyline_oid_reduce_l{target_level}_")

    sharedA = {
        "store_path": str(store_path),
        "source_level": int(source_level),
        "target_level": int(target_level),
        "scale": list(scale),
        "ndim": int(ndim),
        "keep_mask": keep_mask,
        "coarsen_mode": coarsen_mode,
        "stride": int(stride),
        "simplify_epsilon": simplify_epsilon,
        "oid_reduce_shards": int(oid_reduce_shards),
        "oid_reduce_tmp_dir": str(oid_reduce_tmp_dir),
    }
    # Enumerate the source cross-chunk-link cells ONCE and bucket each cell to
    # the target chunk(s) that own its endpoint chunks (chunk // scale). Replaces
    # a per-target-per-child cell scan (O(target_chunks × children × cells))
    # with a single O(cells) pass; workers then read only their bucket's cells.
    from zarr_vectors_tools.algorithms._links import list_link_cells

    cells_by_target: dict[tuple[int, ...], list] = defaultdict(list)
    for cell in list_link_cells(src, delta=0):
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

    manifest_blobs: list[bytes] = []
    empty_blob = b"\x00\x00\x00\x00"
    try:
        resultsA = list(executor(_coarsen_polyline_target_chunk, payloadsA, sharedA))

        total_out_vertices = 0
        total_fragments = 0
        sidecar_arrays: dict[tuple[int, ...], np.ndarray] = {}
        shard_entries: dict[int, list[dict[str, Any]]] = defaultdict(list)
        cross_target_pairs: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
        for res in resultsA:
            total_fragments += int(res["fragment_count"])
            total_out_vertices += int(res["vertex_count"])
            tcc = tuple(int(x) for x in res["tcc"])
            anchors = res.get("anchors")
            if anchors is not None and len(anchors):
                sidecar_arrays[tcc] = anchors
            for shard, path in res.get("oid_shards", []):
                shard_entries[int(shard)].append({"tcc": list(tcc), "path": str(path)})
            # Cross-target anchors report their OTHER target chunk directly —
            # a source-level transition can land in ANY other chunk, not just
            # a face neighbor (split_polyline_at_boundaries does not
            # decompose multi-axis crossings into single-axis hops) — so
            # Phase B's shard pairs come from what Phase A actually observed,
            # not a grid-adjacency scan.
            for p in res.get("partners", []):
                partner = tuple(int(x) for x in p)
                cross_target_pairs.add(tuple(sorted((tcc, partner))))
        resultsA = []

        level_meta.vertex_count = int(total_out_vertices)
        create_resolution_level(root, target_level, level_meta)

        # --- Phase C: shard-parallel object-index reduce + one-shot commit ---
        manifest_blobs = [empty_blob] * int(n_src)
        payloadsC = [
            {"shard": int(s), "entries": es}
            for s, es in sorted(shard_entries.items()) if es
        ]
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
    finally:
        shutil.rmtree(oid_reduce_tmp_dir, ignore_errors=True)

    # Carry object attributes forward; "present" = objects with geometry here.
    present_oids = np.flatnonzero(
        np.fromiter((1 if b != empty_blob else 0 for b in manifest_blobs), np.uint8)
    ).astype(np.int64)
    mask = np.zeros(n_src, dtype=np.uint8)
    if len(present_oids):
        mask[present_oids] = 1
    # object attributes are flat arrays; enumerate via children() (iterating the
    # group yields only sub-group names, missing every flat-array attribute).
    src_attr_names = (
        list(src["object_attributes"].children()) if "object_attributes" in src else []
    )
    for aname in src_attr_names:
        try:
            src_data = read_object_attributes(src, aname)
        except KeyError:
            continue
        out = np.zeros_like(src_data)
        if len(present_oids):
            out[present_oids] = src_data[present_oids]
        create_object_attributes_array(level_group, aname, dtype=str(src_data.dtype))
        write_object_attributes(level_group, aname, out, present_mask=mask)

    # Phase A wrote the vertices / fragment-attribute cells from separate
    # processes, whose per-array ``nonempty_chunks`` manifest RMWs race and can
    # under-report.  Re-derive them single-process from the on-disk cells so the
    # next level's coarsening source scan and the algorithms readers see every
    # chunk.  Phase B's link cells are NOT covered here — under the merged
    # layout they are ordinary chunk-grid arrays that carry (and race on) the
    # same manifest, so their rebuild belongs to the ``finalize_links`` call
    # after Phase B, which re-derives it per offsets segment.
    from zarr_vectors_tools._manifests import rebuild_nonempty_manifests

    rebuild_nonempty_manifests(level_group)

    # --- Phase B: cross-target links, decentralized per ccl shard ---------
    # ``cross_target_pairs`` (gathered above from Phase A's actual reports)
    # already IS the set of chunk pairs needing a stitch — no grid-adjacency
    # scan needed (and none would suffice; see the comment above).
    #
    # Pre-create every offsets ARRAY the Phase B workers will write into,
    # serially, up front.  The shard partition gives each worker disjoint
    # CELLS, but two workers in different shards can still land records in the
    # SAME offsets array (same displacement, different source chunk) — and
    # ``create_links_array`` is not concurrency-safe on the array's zarr.json
    # (a Windows atomic-rename hard-fail).  A cross-target pair (A, B) yields
    # a record sourced at A with offset B-A or sourced at B with offset A-B
    # (the direction is the streamline's), so both displacement arrays are
    # possible; creating both here (empty ones are harmless — finalize_links
    # counts them as zero) means workers only ever WRITE cells, never create.
    cross_offsets: set[tuple[int, ...]] = set()
    for A, B in cross_target_pairs:
        off = tuple(int(B[i]) - int(A[i]) for i in range(ndim))
        cross_offsets.add(off)
        cross_offsets.add(tuple(-x for x in off))
    # Same codec session as the level's other arrays: these are created here,
    # after Phase A, so they need their own block — and it must close before
    # the Phase B dispatch below.
    _link_codec_ctx = (
        level_group.batched_writes(compressor=compressor)
        if compressor else nullcontext()
    )
    with _link_codec_ctx:
        for off in sorted(cross_offsets):
            create_links_array(
                level_group, link_width=2, delta=0, sid_ndim=ndim,
                offsets=(off,), directed=True,
            )

    shard_shape = tuple(
        min(CROSS_LINK_TASK_SHARD_AXIS, chunk_grid_shape[i % ndim])
        for i in range(2 * ndim)
    )
    shard_pairs: dict[tuple[int, ...], list] = defaultdict(list)
    for tcc, nb in cross_target_pairs:
        su = tuple(sorted((tcc, nb)))
        cell = [su[k][ax] - chunk_origin[ax] for k in range(2) for ax in range(ndim)]
        shard = tuple((cell[i] // shard_shape[i]) * shard_shape[i] for i in range(2 * ndim))
        shard_pairs[shard].append((tcc, nb))

    n_cross = 0
    if shard_pairs:
        payloadsB = []
        for _shard, sps in shard_pairs.items():
            need: set = set()
            for A, B in sps:
                need.add(A)
                need.add(B)
            # Forward each needed chunk's anchor array inline; only the chunks
            # this shard's pairs touch are carried, so a worker never receives
            # the whole level's anchors.
            sc_sub = [
                {"tcc": list(c), "arr": sidecar_arrays[c]}
                for c in need if c in sidecar_arrays
            ]
            payloadsB.append({
                "pairs": [(list(A), list(B)) for A, B in sps],
                "anchors": sc_sub,
            })
        sharedB = {
            "store_path": str(store_path),
            "target_level": int(target_level),
            "ndim": int(ndim),
            "chunk_grid_shape": list(chunk_grid_shape),
            "chunk_origin": list(chunk_origin),
        }
        for rb in executor(_polyline_cross_edge_shard, payloadsB, sharedB):
            n_cross += int(rb.get("n_links", 0))

    # Reconcile the links/0 family after the decentralized per-cell writes:
    # recompute num_links and re-derive each offsets array's nonempty_chunks
    # from the store listing.  Runs after every Phase-B worker has finished,
    # once, and must be the last writer of family meta.
    #
    # Unconditional, not gated on n_cross: a level with zero cross-target
    # transitions (a coarse level can collapse to a single chunk) still needs
    # its counts stamped, and write_link_cells creates an offsets array only
    # for records that exist — so zero links leaves zero arrays and there is
    # nothing empty for a reader to 404 on.  create_links_family above already
    # left the group meta readers expect to find on an empty level.
    finalize_links(level_group, delta=0)

    _stamp_root_capability(root, CAP_PRESERVED_OBJECT_IDS)

    summary = {
        "vertex_count": int(total_out_vertices),
        "object_count": int(len(present_oids)),
        "objects_kept": int(len(present_oids)),
        "source_objects": n_src,
        "cross_chunk_edges": int(n_cross),
        "method": coarsening_method,
        "preserves_object_ids": True,
    }
    if coarsen_mode == "decimate":
        summary["stride"] = stride
    else:
        summary["simplify_epsilon"] = simplify_epsilon
    return summary
