"""Read times under each shard reader, on the same stores.

    python compare_options.py STORES_DIR OUT.json [--reps-local 5] [--reps-remote 3]

``STORES_DIR`` holds ``{points,mesh,network}-{indexed,octree,octree_chunks}.zv``
(from ``bench_kinds.build``).  Every configuration reads the same queries
per store, local (median of ``--reps-local`` cold reads) and behind a
20 ms ``LatencyStore`` (median of ``--reps-remote``), and must return the
same answer as the first.  Whole-level reads with the ``standard`` reader
on a ``sharding_indexed`` store are skipped: they go through zarr's
sharded reader one cell at a time (minutes at 200k vertices).

The read options this script once compared (``ZARR_VECTORS_EXPERIMENTS``)
are now how every read works; to measure one against its absence, check
out the commit before it became the default.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
from zarr.storage import LocalStore

from latency_store import LatencyStore

CONFIGS = {"default (auto)": "auto", "standard reader": "standard"}  # name: shard reader
LATENCY = 0.02


def queries(path: Path) -> list[tuple[str, dict]]:
    import zarr_vectors as zv

    rng = np.random.default_rng(3)
    ds = zv.open(str(path))
    pos = np.asarray(ds.select().read().positions)
    present = np.sort(np.asarray(ds.select().object_ids()))
    out: list[tuple[str, dict]] = []
    for size in (125.0, 31.25):
        c = pos[rng.integers(len(pos))]
        lo = np.clip(c - size / 2, 0, 1000 - size)
        out.append((f"box {size:g}", {"bbox": (lo.tolist(), (lo + size).tolist())}))
    if len(present):
        out.append(("1 object", {"objects": [int(present[len(present) // 2])]}))
        out.append(("10 objects", {"objects": [int(o) for o in rng.choice(present, min(10, len(present)), replace=False)]}))
    out.append(("whole level", {}))
    return out


def answer(r) -> tuple:
    p = np.asarray(r.positions)
    extra = tuple(len(getattr(r, n)) for n in ("faces", "edges") if getattr(r, n, None) is not None)
    return (len(p), round(float(np.sort(p, axis=0).astype(np.float64).sum()), 3), *extra)


def read_once(path: Path, sel: dict, latency: float):
    import zarr_vectors as zv

    if latency:
        store = LatencyStore(LocalStore(str(path), read_only=True), latency, 100e6, 128)
        ds = zv.open(store)
        store.reset()
    else:
        store, ds = None, zv.open(str(path))
    t0 = time.perf_counter()
    r = ds.select(**sel).read()
    dt = time.perf_counter() - t0
    stats = (store.gets, store.bytes, store.rounds) if store else (None, None, None)
    return dt, stats, answer(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stores", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--reps-local", type=int, default=5)
    ap.add_argument("--reps-remote", type=int, default=3)
    ap.add_argument("--only", nargs="*", help="store names (e.g. points-indexed)")
    ap.add_argument("--configs", nargs="*", help="configuration names (default: all)")
    args = ap.parse_args()
    import zarr

    zarr.config.set({"async.concurrency": 4096})
    rows = []
    names = sorted(p.name[:-3] for p in args.stores.glob("*.zv") if "-presence" not in p.name)
    for name in names:
        if args.only and name not in args.only:
            continue
        base = args.stores / f"{name}.zv"
        layout = name.split("-", 1)[1]
        os.environ["ZARR_VECTORS_SHARD_READER"] = "auto"
        qs = queries(base)
        ref: dict[str, tuple] = {}
        for cname, reader in CONFIGS.items():
            if args.configs and cname not in args.configs:
                continue
            path = base
            os.environ["ZARR_VECTORS_SHARD_READER"] = reader
            for qname, sel in qs:
                if reader == "standard" and qname == "whole level" and layout != "octree":
                    rows.append({"store": name, "config": cname, "query": qname, "skipped": True})
                    continue
                local = [read_once(path, sel, 0.0) for _ in range(args.reps_local)]
                remote = [read_once(path, sel, LATENCY) for _ in range(args.reps_remote)]
                ans = {a for _, _, a in local + remote}
                assert len(ans) == 1, (name, cname, qname, ans)
                a = ans.pop()
                ref.setdefault(qname, a)
                assert a == ref[qname], (name, cname, qname, a, ref[qname])
                row = {
                    "store": name, "config": cname, "query": qname, "vertices": a[0],
                    "local_s": statistics.median(t for t, _, _ in local),
                    "remote_s": statistics.median(t for t, _, _ in remote),
                    "requests": remote[0][1][0], "bytes": remote[0][1][1], "rounds": remote[0][1][2],
                }
                rows.append(row)
                print(f"{name:22s} {cname:22s} {qname:12s} {a[0]:7d} v  local {row['local_s'] * 1e3:8.1f} ms  "
                      f"20ms {row['remote_s'] * 1e3:8.1f} ms  {row['rounds']:3d} rounds {row['requests']:5d} req "
                      f"{row['bytes'] / 1e6:7.2f} MB", flush=True)
        args.out.write_text(json.dumps(rows, indent=1))
    args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
