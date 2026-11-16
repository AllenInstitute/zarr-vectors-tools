"""Spatial read times in zarr-vectors for point clouds, meshes and networks.

Run from this directory, in an environment with this branch of zarr-vectors
(and zarr-octree, only for the ``octree`` / ``runs`` stores):

    python bench_kinds.py [--quick] [--layouts indexed]

Data (``kinds_data.py``, synthetic): about ``--vertices`` vertices of each kind
over one uneven density in a 1000-unit cube, written through zarr-vectors
(64³ cells, 16³-cell shards, zstd): a point cloud (``add_points``), a mesh of
closed surfaces (``add_mesh``) and a network of branching trees
(``add_graph``).

Stores (``--layouts``): ``indexed`` (an unmarked store: zarr's
``sharding_indexed``, zarr-vectors' default) and, optionally, stores marked
octree at construction (``Layout(octree=...)``): ``octree`` (zarr-octree's
``zarr_octree_sharding`` with its defaults: an octree aimed at 100 kB) or
``runs`` (its runs layout, 160-640 kB).  zarr-octree is only needed for those.

Readers (zarr-vectors' ``ZARR_VECTORS_SHARD_READER``): ``auto`` (the default:
``optimised`` for ``zarr_octree_sharding`` arrays, ``standard`` for the rest),
``standard`` (as zarr-vectors always read), ``optimised`` (each shard index
once, coalesced ranges, any store) and ``zarr`` (zarr's array API only).  The
benchmark always names its reader.

Queries, each through ``select(...).read()``, with the read's faces / edges:

* ``all``: the whole level;
* ``vertex``: a box 0.5 units wide (inside one cell) on an existing vertex,
  or placed uniformly (mostly empty space);
* ``object``: one small / median / large object (the 10th / 50th / 90th
  percentile of object size), and 10 or 100 objects at random.  zarr-vectors
  cannot read a graph by object (``select(objects=)`` and
  ``read_graph(object_ids=)`` refuse), so networks have no object queries;
* ``bbox``: boxes 31.25-250 units wide, centred on a vertex or placed uniformly.

Latency: the local store as it is, or behind ``LatencyStore`` with 20 / 80
ms and 100 MB/s per request (zarr-vectors' object-store path).  Every query
is read with a cold cache (a new dataset per store and reader, no caching
store), and every store and reader must return the same vertices and
faces / edges.  Recorded: time, requests, bytes read (by array: vertices,
vertex fragments, links, object index, metadata).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
from zarr.storage import LocalStore

import kinds_data as zv_kinds
from latency_store import LatencyStore, array_class

HERE = Path(__file__).parent
CELLS, SHARD = 64, 16
ZV_KIND = {"points": "point_cloud", "mesh": "mesh", "network": "graph"}
LAYOUTS = {  # name: (codec, Layout(octree=...)); True is the codec's defaults
    "indexed": ("sharding_indexed", False),
    "octree": ("zarr_octree_sharding", True),
    "runs": ("zarr_octree_sharding", {"layout": "runs", "min_weight": 160_000, "max_weight": 640_000}),
    # sharding_indexed, then level 0 regrouped into octree chunks (to_octree):
    # leaves of up to OCTREE_TARGET vertices, fitted to the shards.  Point
    # clouds and networks only; to_octree does not take meshes.
    "octree_chunks": ("sharding_indexed", False),
}
OCTREE_TARGET = 2_000
NO_OCTREE_CHUNKS = {"mesh"}
READERS = ("zarr", "standard", "optimised")
VERTEX_BOX = 0.5
NO_OBJECT_READS: set[str] = set()  # every kind reads by object (graphs since read_graph(object_ids=))


def make_data(kind: str, n: int) -> dict:
    return zv_kinds.GENERATORS[kind](n)


def build(root: Path, kind: str, layout: str, data: dict) -> dict:
    import zarr_vectors as zv

    codec, octree = LAYOUTS[layout]
    path = root / f"{kind}-{layout}.zv"
    shutil.rmtree(path, ignore_errors=True)
    schema = zv.Schema(bounds=([0, 0, 0], [zv_kinds.BOX] * 3), kind=ZV_KIND[kind],
                       layout=zv.Layout(cells=CELLS, shard_cells=SHARD, compression="zstd", octree=octree))
    t0 = time.perf_counter()
    ds = zv.create(str(path), schema=schema)
    if kind == "points":
        ds.add_points(data["positions"], object_ids=data["object_ids"])
    elif kind == "mesh":
        ds.add_mesh(data["positions"], data["faces"], object_ids=data["object_ids"])
    else:
        ds.add_graph(data["positions"], data["edges"], object_ids=data["object_ids"])
    if layout == "octree_chunks":
        from zarr_vectors.octree_chunks import to_octree

        to_octree(str(path), target=OCTREE_TARGET)
    seconds = time.perf_counter() - t0
    meta = json.loads((path / "0" / "vertices" / "zarr.json").read_text())
    assert meta["codecs"][0]["name"] == codec, meta["codecs"][0]["name"]
    by_array: dict = {}
    for p in path.rglob("*"):
        if p.is_file():
            cls = array_class(str(p.relative_to(path)))
            by_array[cls] = by_array.get(cls, 0) + p.stat().st_size
    runs = []
    if codec == "zarr_octree_sharding":
        from zarr_octree.codec import index_from_shard

        shard = meta["chunk_grid"]["configuration"]["chunk_shape"]
        inner = meta["codecs"][0]["configuration"]["chunk_shape"]
        per_shard = tuple(s // i for s, i in zip(shard, inner))
        for f in (path / "0" / "vertices").rglob("*"):
            if f.is_file() and f.name != "zarr.json":
                runs += [b - a for a, b, _ in index_from_shard(f.read_bytes(), per_shard).runs()]
    return {"write_seconds": seconds, "disk_bytes": sum(by_array.values()), "disk_by_array": by_array,
            "files": sum(1 for p in path.rglob("*") if p.is_file()), "vertex_runs": len(runs),
            "vertex_run_bytes_median": float(np.median(runs)) if runs else None}


def make_queries(kind: str, data: dict, quick: bool, rng: np.random.Generator) -> list[tuple[str, str, float, dict]]:
    """(query kind, placement / size label, size, select kwargs) for every query."""
    pts = data["positions"]
    n_q = 3 if quick else 5
    box = zv_kinds.BOX
    out = [("all", "level", 0.0, {}) for _ in range(1 if quick else 2)]

    def boxes(size, placement):
        for _ in range(n_q):
            c = pts[rng.integers(len(pts))] if placement == "on vertex" else rng.uniform(0, box, 3)
            lo = np.clip(c - size / 2, 0, box - size)
            yield {"bbox": (lo.tolist(), (lo + size).tolist())}

    for placement in ("on vertex", "uniform"):
        out += [("vertex", placement, VERTEX_BOX, sel) for sel in boxes(VERTEX_BOX, placement)]
    if kind not in NO_OBJECT_READS:
        ids = data["object_ids"]
        sizes = np.bincount(ids)
        present = np.flatnonzero(sizes)
        for label, pct in (("small", 10), ("median", 50), ("large", 90)):
            want = np.percentile(sizes[present], pct)
            near = present[np.argsort(np.abs(sizes[present] - want))[: max(n_q, 8)]]
            out += [("object", label, 1, {"objects": [int(o)]}) for o in rng.choice(near, n_q, replace=False)]
        for k in (k for k in (10, 100) if k <= len(present)):
            out += [("object", f"{k} objects", k, {"objects": [int(o) for o in rng.choice(present, k, replace=False)]})
                    for _ in range(n_q)]
    for size in ([31.25, 125.0] if quick else [31.25, 62.5, 125.0, 250.0]):
        for placement in ("on vertex", "uniform"):
            out += [("bbox", placement, size, sel) for sel in boxes(size, placement)]
    return out


def checksum(r) -> dict:
    p = np.asarray(r.positions)
    out = {"vertices": int(len(p)), "checksum": float(np.sort(p, axis=0).astype(np.float64).sum()) if len(p) else 0.0}
    for name in ("faces", "edges"):
        a = getattr(r, name, None)
        if a is not None:
            out[name] = int(len(a))
    return out


def reads_through_zarr(reader: str, latency: float) -> bool:
    """Whether ``reader`` reads this store's shards through zarr's API: the
    ``zarr`` reader always, the ``standard`` one for a store that is not local."""
    return reader == "zarr" or (reader == "standard" and bool(latency))


def run_queries(root: Path, kind: str, layout: str, reader: str, latency: float, bandwidth: float,
                max_inflight: int, queries: list, skip_zarr_all: bool = False) -> list[dict]:
    import zarr_vectors as zv

    os.environ["ZARR_VECTORS_SHARD_READER"] = reader
    os.environ.pop("ZARR_VECTORS_COALESCE", None)
    path = root / f"{kind}-{layout}.zv"
    rows = []
    for qi, (qkind, label, size, sel) in enumerate(queries):
        if skip_zarr_all and qkind == "all" and reads_through_zarr(reader, latency):
            continue
        if latency:
            store = LatencyStore(LocalStore(str(path), read_only=True), latency, bandwidth, max_inflight)
            ds = zv.open(store)
        else:
            store, ds = None, zv.open(str(path))
        if store is not None:  # opening reads metadata; count only the query
            store.reset()
        t0 = time.perf_counter()
        r = ds.select(**sel).read()
        dt = time.perf_counter() - t0
        rows.append({"kind": kind, "layout": layout, "reader": reader, "latency": latency, "query": qkind,
                     "label": label, "size": size, "qi": qi, "seconds": dt,
                     "requests": store.gets if store else None,
                     "rounds": store.rounds if store else None,
                     "bytes_read": store.bytes if store else None,
                     "bytes_by_array": dict(store.by_array) if store else None, **checksum(r)})
    return rows


def check_agree(rows: list[dict]) -> None:
    """Every store and reader returns the same answer to every query."""
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["kind"], r["latency"], r["qi"]), []).append(r)
    for key, rs in groups.items():
        ref = rs[0]
        for r in rs[1:]:
            for f in ("vertices", "faces", "edges"):
                assert r.get(f) == ref.get(f), (key, f, ref["layout"], ref["reader"], r["layout"], r["reader"], ref.get(f), r.get(f))
            assert abs(r["checksum"] - ref["checksum"]) <= 1e-6 * max(1.0, abs(ref["checksum"])), (key, ref, r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=list(zv_kinds.KINDS), choices=zv_kinds.KINDS)
    ap.add_argument("--layouts", nargs="+", default=["indexed", "octree"], choices=list(LAYOUTS))
    ap.add_argument("--vertices", type=int, default=4_000_000, help="vertices of each kind")
    ap.add_argument("--stores", type=Path, default=HERE / "_stores")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "bench_kinds.json")
    ap.add_argument("--quick", action="store_true", help="fewer queries and sizes; local and 20 ms only")
    ap.add_argument("--latencies", type=float, nargs="+", help="seconds per request (default 0 0.02 0.08)")
    ap.add_argument("--skip-build", action="store_true", help="reuse the stores and build record already there")
    ap.add_argument("--skip-zarr-all", action="store_true",
                    help="no whole-level reads through zarr's API (the zarr reader, and the standard one "
                         "off a store that is not local): at 4 M vertices each takes minutes")
    ap.add_argument("--max-inflight", type=int, default=128)
    ap.add_argument("--bandwidth", type=float, default=100e6)
    ap.add_argument("--readers", nargs="+", default=list(READERS), choices=READERS)
    ap.add_argument("--env", nargs="*", default=[], metavar="NAME=VALUE",
                    help="environment set for every read (e.g. ZARR_VECTORS_SHARD_READER=standard); recorded in the results")
    args = ap.parse_args()
    for kv in args.env:
        name, _, value = kv.partition("=")
        os.environ[name] = value
    import zarr

    zarr.config.set({"async.concurrency": 4096})
    if any(LAYOUTS[l][0] == "zarr_octree_sharding" for l in args.layouts):
        try:
            import zarr_octree.codec  # noqa: F401
        except ImportError as exc:
            raise SystemExit(f"--layouts {' '.join(args.layouts)} needs zarr-octree ({exc}); "
                             "--layouts indexed runs without it")
    latencies = args.latencies or ([0.0, 0.02] if args.quick else [0.0, 0.02, 0.08])
    args.stores.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = {"setup": {"vertices": args.vertices, "cells": CELLS, "shard_cells": SHARD, "layouts":
                         {l: LAYOUTS[l] for l in args.layouts}, "readers": READERS, "latencies": latencies,
                         "zarr": zarr.__version__, "bandwidth": args.bandwidth, "max_inflight": args.max_inflight,
                         "quick": args.quick, "no_object_reads": sorted(NO_OBJECT_READS),
                         "skip_zarr_all": args.skip_zarr_all, "env": args.env,
                         "readers_run": args.readers},
               "data": {}, "build": {}, "queries": []}
    if args.out.exists():  # keep the kinds a previous run finished
        old = json.loads(args.out.read_text())
        if old.get("setup", {}).get("vertices") == args.vertices:
            for k in ("data", "build"):
                results[k] = {kind: v for kind, v in old[k].items() if kind not in args.kinds}
            results["queries"] = [q for q in old["queries"] if q["kind"] not in args.kinds]
    for kind in args.kinds:
        t0 = time.perf_counter()
        data = make_data(kind, args.vertices)
        sizes = np.bincount(data["object_ids"])
        sizes = sizes[sizes > 0]
        results["data"][kind] = {"vertices": int(len(data["positions"])), "objects": int(len(sizes)),
                                 "object_size_p10_50_90": np.percentile(sizes, [10, 50, 90]).tolist(),
                                 **{f: int(len(data[f])) for f in ("faces", "edges") if f in data}}
        print(kind, results["data"][kind], f"{time.perf_counter() - t0:.0f} s", flush=True)
        built = args.stores / f"{kind}_build.json"
        if args.skip_build and built.exists():
            results["build"][kind] = json.loads(built.read_text())
        else:
            results["build"][kind] = {layout: build(args.stores, kind, layout, data) for layout in args.layouts
                                      if not (layout == "octree_chunks" and kind in NO_OCTREE_CHUNKS)}
            built.write_text(json.dumps(results["build"][kind], indent=1))
        for layout, b in results["build"][kind].items():
            print(f"  {layout:8s} write {b['write_seconds']:6.1f} s  {b['disk_bytes'] / 1e6:8.1f} MB  {b['files']} files", flush=True)
        queries = make_queries(kind, data, args.quick, np.random.default_rng(5))
        rows = []
        for latency in latencies:
            for layout in results["build"][kind]:
                for reader in args.readers:
                    t0 = time.perf_counter()
                    mine = run_queries(args.stores, kind, layout, reader, latency, args.bandwidth,
                                       args.max_inflight, queries, args.skip_zarr_all)
                    rows += mine
                    med = {q: np.median([r["seconds"] for r in mine if r["query"] == q]) * 1e3
                           for q in ("all", "vertex", "object", "bbox") if any(r["query"] == q for r in mine)}
                    print(f"  lat {latency * 1e3:3.0f}  {layout:8s} {reader:9s} "
                          + "  ".join(f"{q} {v:8.1f}" for q, v in med.items())
                          + f"  ms (median)  [{time.perf_counter() - t0:.0f} s]", flush=True)
        check_agree(rows)
        results["queries"] += rows
        args.out.write_text(json.dumps(results, indent=1))
        print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
