#!/usr/bin/env python
"""Locate, and cost, the per-record Python work in the read/write path.

    python profile_hotspots.py            # all five hotspots
    python profile_hotspots.py --only 3

The sweep in `run_sweep.py` says *that* bulk I/O is slower than a single
binary file. This says *why*, and how much of it is recoverable. Each
check pairs the code as it stands today against a vectorised equivalent
on the same input and asserts the two produce the same answer, so the
speedup column is a measured floor on what the change would buy — not an
estimate.

The recurring shape is the same everywhere: the work is proportional to
the number of records touched in Python — objects, fragments, faces,
polylines — rather than to the bytes moved. Compression and disk are not
involved: Zstd versus no codec on a 10⁶-point write is 14.67 s versus
14.54 s, and the 125 actual zarr chunk writes are ~2.5 s of a ~20 s run.

Numbers quoted in `README.md` come from this script at N = 10⁶ on the
machine recorded in `results/environment.json`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _harness as H  # noqa: E402

N = 1_000_000
_RESULTS: list[tuple[str, str, float, float, bool]] = []


def record(num, name, t_now, t_new, same):
    _RESULTS.append((num, name, t_now, t_new, same))
    verdict = "identical" if same else "DIFFERS"
    print(f"    current {t_now:7.2f}s   vectorised {t_new:6.2f}s   "
          f"{t_now / t_new:5.1f}x   output {verdict}")


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return time.perf_counter() - t0, out


# ====================================================================
# 1. types/points.py:437 -- per-object grouping inside a chunk
# ====================================================================

def hotspot_1():
    """`for obj_id in unique_objs: mask = chunk_obj_ids == obj_id`.

    One full boolean scan of the chunk per object, so the cost is
    (vertices per chunk) x (distinct objects per chunk) -- quadratic once
    objects are as fine-grained as vertices. A stable argsort plus a
    split at the group boundaries does the same partition in O(n log n).
    """
    print("1. types/points.py:437  per-object grouping in a chunk")
    rng = np.random.default_rng(0)
    per_chunk, n_chunks = N // 125, 125
    pos = rng.uniform(0, 200, (per_chunk, 3)).astype(np.float32)
    oid = rng.integers(0, per_chunk, per_chunk)          # one object per vertex

    def now(pos, oid):
        groups, manifest = [], []
        for i, obj_id in enumerate(np.unique(oid)):
            groups.append(pos[oid == obj_id])
            manifest.append((int(obj_id), i))
        return groups, manifest

    def new(pos, oid):
        order = np.argsort(oid, kind="stable")
        ids = oid[order]
        starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
        groups = np.split(pos[order], starts[1:])
        return groups, list(zip(ids[starts].tolist(), range(len(starts))))

    t_a, a = timed(now, pos, oid)
    t_b, b = timed(new, pos, oid)
    same = (len(a[0]) == len(b[0]) and a[1] == b[1]
            and all(np.array_equal(np.sort(x, axis=0), np.sort(y, axis=0))
                    for x, y in zip(a[0], b[0])))
    record("1", "point grouping", t_a * n_chunks, t_b * n_chunks, same)


# ====================================================================
# 2. core/arrays.py:1676 + encoding/fragments.py:593 -- manifest encode
# ====================================================================

def hotspot_2():
    """One `encode_object_manifest_blocks` call per object, each doing a
    coordinate genexpr, three `struct.pack`s and a join.

    In the shape the bulk writers always produce -- one block per object,
    one fragment per block -- a manifest is a fixed-width record, so the
    whole index is one structured array and the encoding is a memory
    layout rather than a loop.
    """
    print("2. core/arrays.py:1676  object-manifest encoding")
    from zarr_vectors.encoding.fragments import encode_object_manifest_blocks

    ndim = 3
    rng = np.random.default_rng(0)
    coords = rng.integers(0, 5, (N, ndim)).astype(np.int64)
    frags = rng.integers(0, 8000, N).astype(np.int64)
    manifests = {i: [(tuple(coords[i]), int(frags[i]))] for i in range(N)}

    def now():
        out = []
        for oid in range(N):
            blocks = [(tuple(int(c) for c in cc), int(fi))
                      for cc, fi in manifests.get(oid, [])]
            out.append(encode_object_manifest_blocks(blocks, sid_ndim=ndim))
        return out

    dt = np.dtype([("n", "<u4"), ("coords", "<i8", ndim),
                   ("mode", "u1"), ("idx", "<i8")], align=False)

    def new():
        arr = np.zeros(N, dtype=dt)
        arr["n"] = 1
        arr["coords"] = coords
        arr["mode"] = 0                       # MANIFEST_MODE_SINGLE
        arr["idx"] = frags
        flat = np.frombuffer(arr.tobytes(),
                             dtype=np.uint8).reshape(N, dt.itemsize)
        return [row.tobytes() for row in flat]

    t_a, a = timed(now)
    t_b, b = timed(new)
    record("2", "manifest encode", t_a, t_b, a == b)


# ====================================================================
# 3. types/meshes.py:277 + spatial/boundary.py:563 -- face partitioning
# ====================================================================

def _mesh_topology(data):
    from zarr_vectors.spatial.chunking import assign_chunks
    verts, faces = data["vertices"], data["faces"]
    assign = assign_chunks(verts, H.CHUNK)
    chunk_list = sorted(assign)
    v_chunk = np.empty(len(verts), dtype=np.int64)
    v_local = np.empty(len(verts), dtype=np.int64)
    for ci, cc in enumerate(chunk_list):
        gi = np.asarray(assign[cc], dtype=np.int64)
        v_chunk[gi] = ci
        v_local[gi] = np.arange(len(gi))
    return chunk_list, v_chunk[faces], v_local[faces]


def hotspot_3(data):
    """Every face is turned into a Python list of `(chunk, index)` tuples
    and pushed through `partition_records_by_offset`, which per record
    does a cell-placement search, a Lehmer permutation encode and an
    offsets path-string format.

    ~96 % of faces have all three vertices in one chunk. For those the
    permutation is the identity, the Lehmer code is 0 and the offsets
    string is a constant, so none of that work is needed -- they group by
    source chunk with one stable argsort. Only the boundary-crossing
    remainder needs the general path.
    """
    print("3. spatial/boundary.py:563  mesh face partitioning")
    from zarr_vectors.core.paths import format_offsets
    from zarr_vectors.spatial.boundary import partition_records_by_offset

    chunk_list, f_chunk, f_local = _mesh_topology(data)
    is_intra = np.all(f_chunk == f_chunk[:, :1], axis=1)
    print(f"    {len(f_chunk):,} faces, {is_intra.mean():.1%} intra-chunk")
    kw = dict(link_width=3, sid_ndim=3, scale_src=(1, 1, 1), scale_trg=(1, 1, 1))

    def now():
        records = [[(chunk_list[ci], li) for ci, li in zip(rc, rl)]
                   for rc, rl in zip(f_chunk.tolist(), f_local.tolist())]
        return partition_records_by_offset(records, **kw)

    def new():
        buckets = {}
        rows = np.flatnonzero(is_intra)
        src = f_chunk[rows, 0]
        order = np.argsort(src, kind="stable")      # keeps input order
        rows_s, src_s = rows[order], src[order]
        starts = np.flatnonzero(np.r_[True, src_s[1:] != src_s[:-1]])
        vi_all, rows_l = f_local[rows_s].tolist(), rows_s.tolist()
        intra_seg = format_offsets(((0, 0, 0), (0, 0, 0)))
        for b, s in enumerate(starts):
            e = starts[b + 1] if b + 1 < len(starts) else len(rows_s)
            buckets[(intra_seg, chunk_list[int(src_s[s])])] = [
                (vi_all[i], 0, rows_l[i]) for i in range(s, e)
            ]
        cross = np.flatnonzero(~is_intra)
        if cross.size:
            sub = [[(chunk_list[ci], li) for ci, li in zip(rc, rl)]
                   for rc, rl in zip(f_chunk[cross].tolist(),
                                     f_local[cross].tolist())]
            for key, vals in partition_records_by_offset(sub, **kw).items():
                buckets.setdefault(key, []).extend(
                    (vi, p, int(cross[ii])) for vi, p, ii in vals)
        return buckets

    t_a, a = timed(now)
    t_b, b = timed(new)
    record("3", "face partitioning", t_a, t_b, a == b)


# ====================================================================
# 4. core/arrays.py:4476 + types/meshes.py:541 -- face reassembly on read
# ====================================================================

def hotspot_4(data):
    """`read_links` returns one Python tuple per face; `read_mesh` then
    walks them doing three dict lookups and three adds each.

    Inside `read_links` the same rows already exist as an `(n, L)` integer
    array per cell, and the cell's endpoint chunks are known once. Local
    to global is then one broadcast add per cell -- ~1 100 of them instead
    of 1.68 M per-face iterations.
    """
    print("4. types/meshes.py:541  face reassembly on read")
    from run_sweep import ZV_WRITERS
    from zarr_vectors.building import open_store
    from zarr_vectors.core.arrays import (
        _link_cell_rows,
        _link_scales,
        _parse_chunk_key,
        cell_endpoint_chunks,
        links_group_path,
        links_has_perm,
        list_link_offsets,
        read_chunk_vertices,
        read_links,
        resolve_chunk_keys,
    )
    from zarr_vectors.core.paths import is_intra, parse_offsets

    ws = H.Workspace()
    try:
        store = ws.store("mesh")
        ZV_WRITERS["meshes"](store, data)
        lg = open_store(str(store), mode="r")["0"]

        offsets_of_chunk, run = {}, 0
        for cc in resolve_chunk_keys(lg, None, bbox=None, chunks=None):
            offsets_of_chunk[cc] = run
            run += sum(len(g) for g in read_chunk_vertices(lg, cc, ndim=3))

        def now():
            rows = []
            for face in read_links(lg, delta=0):
                ids = []
                for cc, li in face:
                    off = offsets_of_chunk.get(cc)
                    if off is None:
                        ids = []
                        break
                    ids.append(off + int(li))
                if len(ids) == 3:
                    rows.append(ids)
            return np.asarray(rows, dtype=np.int64)

        def new():
            fam = links_group_path(0)
            meta = lg.read_array_meta(fam)
            width, ndim = int(meta["link_width"]), int(meta["sid_ndim"])
            scale_src, scale_trg = _link_scales(lg, 0, ndim)
            blocks = []
            for seg in list_link_offsets(lg, 0):
                name = f"{fam}/{seg}"
                offsets = parse_offsets(seg, sid_ndim=ndim, link_width=width)
                am = lg.read_array_meta(name) or {}
                has_perm = bool(am.get("has_perm", links_has_perm(
                    offsets, delta=0, directed=bool(meta.get("directed", False)),
                    store=str(meta.get("store", "canonical")))))
                ncols = (1 + width) if has_perm else width
                cdt = np.dtype(am.get("dtype", "int64"))
                for key in sorted(lg.list_chunks(name)):
                    blob = lg.read_bytes(name, key)
                    if not blob:
                        continue
                    r = _link_cell_rows(blob, ncols=ncols,
                                        flat=is_intra(offsets), dtype=cdt)
                    if r.size == 0:
                        continue
                    chunks = cell_endpoint_chunks(_parse_chunk_key(key), offsets,
                                                  scale_src, scale_trg)
                    base = [offsets_of_chunk.get(c) for c in chunks]
                    if any(b is None for b in base):
                        continue
                    vi = r[:, 1:1 + width] if has_perm else r[:, :width]
                    blocks.append(vi.astype(np.int64)
                                  + np.asarray(base, dtype=np.int64))
            return np.concatenate(blocks) if blocks else np.zeros((0, 3), np.int64)

        t_a, a = timed(now)
        t_b, b = timed(new)
        canon = lambda x: np.unique(np.sort(x, axis=1), axis=0)  # noqa: E731
        record("4", "face reassembly", t_a, t_b,
               np.array_equal(canon(a), canon(b)))
    finally:
        ws.close()


# ====================================================================
# 5. spatial/boundary.py:27 -- per-polyline boundary split
# ====================================================================

def hotspot_5(data):
    """`split_polyline_at_boundaries` is vectorised inside, but it is
    called once per polyline, so 83 333 streamlines each pay numpy call
    overhead on a ~12-row array.

    Batching over the concatenated point buffer does the same split in a
    fixed number of numpy calls: break where the chunk changes, and
    always at a polyline start.
    """
    print("5. spatial/boundary.py:27  polyline boundary split")
    from zarr_vectors.spatial.boundary import split_polyline_at_boundaries
    polys = data["polylines"]

    def now():
        return [split_polyline_at_boundaries(p, H.CHUNK) for p in polys]

    def new():
        lengths = np.fromiter((len(p) for p in polys), np.int64, len(polys))
        flat = np.concatenate(polys)
        ci = np.floor(flat / np.asarray(H.CHUNK, dtype=np.float64)).astype(np.int64)
        starts = np.zeros(len(lengths) + 1, dtype=np.int64)
        np.cumsum(lengths, out=starts[1:])
        brk = np.zeros(len(flat), dtype=bool)
        brk[1:] = np.any(ci[1:] != ci[:-1], axis=1)
        brk[starts[:-1]] = True
        seg_start = np.flatnonzero(brk)
        seg_end = np.r_[seg_start[1:], len(flat)]
        seg_poly = np.searchsorted(starts, seg_start, side="right") - 1
        out = [[] for _ in polys]
        for s, e, c, p in zip(seg_start.tolist(), seg_end.tolist(),
                              ci[seg_start].tolist(), seg_poly.tolist()):
            out[p].append((tuple(c), flat[s:e]))
        return out

    t_a, a = timed(now)
    t_b, b = timed(new)
    same = (len(a) == len(b) and all(
        len(x) == len(y) and all(cx == cy and np.array_equal(vx, vy)
                                 for (cx, vx), (cy, vy) in zip(x, y))
        for x, y in zip(a, b)))
    record("5", "polyline split", t_a, t_b, same)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None,
                    help="comma-separated hotspot numbers, e.g. '3,4'")
    args = ap.parse_args()
    want = ({s.strip() for s in args.only.split(",")} if args.only
            else {"1", "2", "3", "4", "5"})

    print(f"N = {N:,} vertices, chunk_shape = {H.CHUNK}\n")
    if "1" in want:
        hotspot_1()
    if "2" in want:
        hotspot_2()
    if want & {"3", "4"}:
        mesh = H.gen_mesh(N)
        if "3" in want:
            hotspot_3(mesh)
        if "4" in want:
            hotspot_4(mesh)
    if "5" in want:
        hotspot_5(H.gen_streamlines(N))

    print(f"\n{'#':<3} {'hotspot':<20} {'current':>10} {'vectorised':>12} "
          f"{'speedup':>9}  output")
    for num, name, ta, tb, same in _RESULTS:
        print(f"{num:<3} {name:<20} {ta:9.2f}s {tb:11.2f}s {ta / tb:8.1f}x  "
              f"{'identical' if same else 'DIFFERS'}")


if __name__ == "__main__":
    main()
