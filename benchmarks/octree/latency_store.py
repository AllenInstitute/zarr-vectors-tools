"""A store wrapper that emulates object storage and counts what a read fetches.

Every request sleeps ``latency`` seconds plus its size over ``bandwidth``;
at most ``max_inflight`` run at once (a client's connection pool).  Requests
and bytes are counted, the bytes also by class of array (vertices, vertex
fragments, links, object index, metadata).
"""

from __future__ import annotations

import asyncio

from zarr.abc.store import ByteRequest, Store
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.storage import WrapperStore


def array_class(key: str) -> str:
    if key.endswith("zarr.json"):
        return "metadata"
    parts = key.split("/")
    for name, cls in (("vertices", "vertices"), ("vertex_fragments", "vertex fragments"), ("links", "links"),
                      ("link_fragments", "links"), ("object_index", "object index")):
        if name in parts:
            return cls
    return "other"


class LatencyStore(WrapperStore[Store]):
    def __init__(self, store: Store, latency: float = 0.0, bandwidth: float | None = None,
                 max_inflight: int | None = None) -> None:
        super().__init__(store)
        self.latency, self.bandwidth, self.max_inflight = latency, bandwidth, max_inflight
        self._slots = asyncio.Semaphore(max_inflight) if max_inflight else None
        self.gets = 0
        self.by_array: dict[str, int] = {}
        self.starts: list[float] = []

    @property
    def rounds(self) -> int:
        """Sequential request rounds: requests started together (within half
        a latency of the previous start) count as one round."""
        if not self.starts:
            return 0
        s = sorted(self.starts)
        gap = (self.latency or 0.0) / 2
        return 1 + sum(1 for a, b in zip(s, s[1:]) if b - a > gap)

    @property
    def bytes(self) -> int:
        return sum(self.by_array.values())

    def reset(self) -> None:
        self.gets = 0
        self.by_array.clear()
        self.starts.clear()

    def _with_store(self, store: Store) -> LatencyStore:
        other = LatencyStore(store, self.latency, self.bandwidth, self.max_inflight)
        other._slots = self._slots  # views (e.g. read-only) share the pool
        other._shared = self._root()
        return other

    def _root(self) -> LatencyStore:
        return getattr(self, "_shared", self)

    # Async only: the sync fast path would bypass the latency and the counts.
    _supports_sync_io = False

    async def get(self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None) -> Buffer | None:
        self._root().starts.append(asyncio.get_running_loop().time())
        if self._slots is None:
            buf = await self._get(key, prototype, byte_range)
        else:
            async with self._slots:
                buf = await self._get(key, prototype, byte_range)
        root = self._root()
        root.gets += 1
        if buf is not None:
            cls = array_class(key)
            root.by_array[cls] = root.by_array.get(cls, 0) + len(buf)
        return buf

    async def _get(self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None) -> Buffer | None:
        if self.latency:
            await asyncio.sleep(self.latency)
        buf = await self._store.get(key, prototype, byte_range)
        if self.bandwidth and buf is not None:
            await asyncio.sleep(len(buf) / self.bandwidth)
        return buf

    async def get_partial_values(self, prototype, key_ranges):  # type: ignore[no-untyped-def]
        return list(await asyncio.gather(*(self.get(k, prototype, r) for k, r in key_ranges)))

    async def _get_many(self, requests):  # type: ignore[no-untyped-def]
        requests = list(requests)
        results = await asyncio.gather(*(self.get(*r) for r in requests))
        for req, res in zip(requests, results):
            yield req[0], res

    if hasattr(Store, "get_ranges"):  # zarr >= 3.4; the base version reads through self.get
        get_ranges = Store.get_ranges
