"""Which arrays each sequential request round of one query fetches.

    python trace_rounds.py STORE [--latency 0.02] [--reader optimised]
        [--bbox x0 y0 z0 x1 y1 z1 | --objects ID ...]

Reads ``STORE`` behind a :class:`LatencyStore` with a cold cache and prints,
per round (requests started together), the request count and bytes by
array class, and the time.  Diagnostic only.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time

from zarr.storage import LocalStore

from latency_store import LatencyStore, array_class


class TracingStore(LatencyStore):
    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        self.log: list[tuple[float, str, int]] = []

    def _with_store(self, store):  # type: ignore[no-untyped-def]
        other = TracingStore(store, self.latency, self.bandwidth, self.max_inflight)
        other._slots = self._slots
        other._shared = self._root()
        return other

    async def get(self, key, prototype, byte_range=None):  # type: ignore[no-untyped-def]
        t = asyncio.get_running_loop().time()
        buf = await super().get(key, prototype, byte_range)
        self._root().log.append((t, key, 0 if buf is None else len(buf)))
        return buf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("store")
    ap.add_argument("--latency", type=float, default=0.02)
    ap.add_argument("--reader", default="optimised")
    ap.add_argument("--bbox", type=float, nargs=6)
    ap.add_argument("--objects", type=int, nargs="+")
    ap.add_argument("--env", nargs="*", default=[])
    args = ap.parse_args()
    os.environ["ZARR_VECTORS_SHARD_READER"] = args.reader
    for kv in args.env:
        k, _, v = kv.partition("=")
        os.environ[k] = v
    import zarr

    import zarr_vectors as zv

    zarr.config.set({"async.concurrency": 4096})
    store = TracingStore(LocalStore(args.store, read_only=True), args.latency, 100e6, 128)
    ds = zv.open(store)
    store.reset()
    store.log.clear()
    sel = {}
    if args.bbox:
        sel["bbox"] = (args.bbox[:3], args.bbox[3:])
    if args.objects:
        sel["objects"] = args.objects
    t0 = time.perf_counter()
    r = ds.select(**sel).read()
    dt = time.perf_counter() - t0
    log = sorted(store.log)
    rounds: list[list[tuple[float, str, int]]] = []
    for e in log:
        if rounds and e[0] - rounds[-1][-1][0] <= args.latency / 2:
            rounds[-1].append(e)
        else:
            rounds.append([e])
    t_first = log[0][0] if log else 0.0
    print(f"{len(r.positions)} vertices, {dt:.3f} s, {len(log)} requests, {len(rounds)} rounds")
    for i, rd in enumerate(rounds):
        by: dict[str, list[int]] = {}
        for _, key, n in rd:
            c = array_class(key)
            by.setdefault(c, [0, 0])
            by[c][0] += 1
            by[c][1] += n
        parts = "  ".join(f"{c}: {k} req {b / 1e3:.0f} kB" for c, (k, b) in sorted(by.items()))
        ex = rd[0][1]
        print(f"  round {i:2d} +{(rd[0][0] - t_first) * 1e3:6.0f} ms  {parts}   e.g. {ex}")


if __name__ == "__main__":
    main()
