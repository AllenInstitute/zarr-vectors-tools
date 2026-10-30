"""An interactive filter engine in numpy: a spatial index plus cached leaves.

This is the design the platform plan proposes for the browser
(``zarr_vectors_filtering_wasm_plan.md``, part B), built from numpy alone so
that it runs unchanged under Pyodide and can be timed there (gate G1 of the
build guide).  It answers exactly what the oracle answers -- the tests hold
it to that -- but does far less work per question:

- A region leaf tests only the segments filed under the grid cells the
  region's bounding box touches.  Each segment is filed under every cell its
  own bounding box overlaps, so a segment that crosses a cell face, or skips
  over a whole cell, is still found from any cell it passes through.  (A
  rare long segment spanning many cells goes on a short list tested every
  time instead.)
- A segment that lies in one cell, where that cell lies wholly inside the
  region, meets the region without being tested; and a segment is not
  tested once its streamline has been accepted.
- Endpoint modes test only the two ends of each streamline.
- Every leaf's answer is cached by the leaf's content, so changing an
  operator, or toggling one leaf, re-runs only the boolean algebra.

Unlike the browser design, it holds level 0 whole in memory: the question it
answers is how fast numpy computes the filters, not how they are fetched.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from . import oracle
from .geometry import contains, region_bounds, segment_hits
from .predicates import All, And, Box, Column, InGroup, Node, Not, Or, Passes, Slab, Sphere
from .tractogram import Tractogram

__all__ = ["IndexedEngine"]

#: Segments filed per batch while building the index.
_BUILD_BLOCK = 1 << 21
#: Candidate segments tested per batch in a query.
_QUERY_BLOCK = 1 << 17


class IndexedEngine:
    """Filter one tractogram, interactively.

    Args:
        tractogram: The streamlines; held, not copied.
        cell_size: Edge of the index's cubic cells, in millimetres.  The
            default aims for about ``segments_per_cell`` segments per cell
            and never goes below twice the median segment length.
        segments_per_cell: Target occupancy for the default cell size.
        cache_size: Leaf answers kept (least recently used are dropped).
        long_cells: A segment whose bounding box spans more cells than this
            is not filed under each of them but kept on a short list that
            every region query tests.
    """

    def __init__(
        self,
        tractogram: Tractogram,
        *,
        cell_size: float | None = None,
        segments_per_cell: int = 64,
        cache_size: int = 256,
        long_cells: int = 64,
    ) -> None:
        self.tractogram = tractogram
        self._cache: OrderedDict[str, npt.NDArray[np.bool_]] = OrderedDict()
        self._cache_size = int(cache_size)
        self._start, self._end, self._owner = tractogram.edges()
        self._first, self._last, self._has = tractogram.endpoints()
        #: Counts from the last region query: cells visited, cells wholly
        #: inside the region, segments accepted untested, candidates, tested.
        self.last_query: dict[str, int] = {}
        self._build(cell_size, segments_per_cell, long_cells)

    # -- index -----------------------------------------------------------------

    def _build(self, cell_size: float | None, per_cell: int, long_cells: int) -> None:
        vertices = self.tractogram.vertices
        n_edges = len(self._start)
        self._long = np.zeros(0, dtype=self._start.dtype)
        if n_edges == 0:
            self.origin = np.zeros(3)
            self.cell_size = 1.0
            self.grid = np.ones(3, dtype=np.int64)
            self._cell_start = np.zeros(2, dtype=np.int64)
            self._filed = np.zeros(0, dtype=self._start.dtype)
            self._single = np.zeros(0, dtype=bool)
            return
        lo = vertices.min(axis=0).astype(np.float64)
        hi = vertices.max(axis=0).astype(np.float64)
        extent = np.maximum(hi - lo, 1e-9)
        if cell_size is None:
            sample = np.arange(0, n_edges, max(1, n_edges // 100_000))
            steps = np.linalg.norm(
                vertices[self._end[sample]].astype(np.float64) - vertices[self._start[sample]],
                axis=1,
            )
            volume_cell = float(np.prod(extent)) * per_cell / n_edges
            cell_size = max(volume_cell ** (1 / 3), 2.0 * float(np.median(steps)), 1e-6)
        self.cell_size = float(cell_size)
        self.origin = lo
        self.grid = np.floor(extent / self.cell_size).astype(np.int64) + 1
        n_cells = int(np.prod(self.grid))
        if n_cells >= 2**31 or n_edges >= 2**31:
            raise ValueError(
                f"{n_edges} segments in {n_cells} cells is too large for this engine; "
                "pass a larger cell_size"
            )

        # Each filing is one int64: cell << 32 | single-cell flag << 31 | segment.
        # Sorting them in place orders the index by cell without the int64
        # index array an argsort would allocate.
        parts: list[npt.NDArray[np.int64]] = []
        long: list[npt.NDArray] = []
        for first in range(0, n_edges, _BUILD_BLOCK):
            s = self._start[first:first + _BUILD_BLOCK]
            e = self._end[first:first + _BUILD_BLOCK]
            a = vertices[s].astype(np.float64)
            b = vertices[e].astype(np.float64)
            c_lo = self._cell(np.minimum(a, b))
            c_hi = self._cell(np.maximum(a, b))
            del a, b
            span = c_hi - c_lo + 1
            copies = span.prod(axis=1)
            ids = np.arange(first, first + len(s), dtype=np.int64)
            one = copies == 1
            parts.append(self._linear(c_lo[one]) << 32 | np.int64(1) << 31 | ids[one])
            many = (copies > 1) & (copies <= long_cells)
            long.append(ids[copies > long_cells].astype(self._start.dtype))
            if many.any():
                # Repeat counts are intp: numpy refuses int64 ones where intp
                # is 32-bit, as under Pyodide.
                n = copies[many].astype(np.intp)
                which = np.repeat(np.arange(len(n)), n)
                k = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
                sy, sz = span[many][which, 1], span[many][which, 2]
                cell = c_lo[many][which] + np.column_stack([k // (sy * sz), (k // sz) % sy, k % sz])
                parts.append(self._linear(cell) << 32 | ids[many][which])
        packed = np.concatenate(parts)
        del parts
        packed.sort()
        self._long = np.concatenate(long)
        self._cell_start = np.searchsorted(
            packed, np.arange(n_cells + 1, dtype=np.int64) << 32,
        ).astype(np.int64)
        self._filed = np.empty(len(packed), dtype=self._start.dtype)
        self._single = np.empty(len(packed), dtype=bool)
        for first in range(0, len(packed), _BUILD_BLOCK):
            block = packed[first:first + _BUILD_BLOCK]
            self._filed[first:first + _BUILD_BLOCK] = block & 0x7FFFFFFF
            self._single[first:first + _BUILD_BLOCK] = (block >> 31) & 1
        del packed

    def _cell(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        cell = np.floor((points - self.origin) / self.cell_size).astype(np.int64)
        return np.clip(cell, 0, self.grid - 1)

    def _linear(self, cell: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        return (cell[:, 0] * self.grid[1] + cell[:, 1]) * self.grid[2] + cell[:, 2]

    @property
    def index_bytes(self) -> int:
        """Bytes the index adds on top of the tractogram."""
        return int(
            self._filed.nbytes + self._single.nbytes + self._cell_start.nbytes
            + self._long.nbytes + self._start.nbytes + self._end.nbytes + self._owner.nbytes
            + self._first.nbytes + self._last.nbytes
        )

    def _cells(self, region) -> npt.NDArray[np.int64]:
        """``(i, j, k)`` of every cell the region's bounding box touches."""
        lo, hi = region_bounds(region)
        # Widened by a hair, so a point the exact test accepts by rounding at
        # the region's rim is never in a cell left out.
        with np.errstate(invalid="ignore"):  # a slab's infinite sides stay infinite
            lo = np.where(np.isfinite(lo), lo - 1e-9 * (1.0 + np.abs(lo)), lo)
            hi = np.where(np.isfinite(hi), hi + 1e-9 * (1.0 + np.abs(hi)), hi)
        world_hi = self.origin + self.grid * self.cell_size
        if np.any(hi < self.origin) or np.any(lo > world_hi) or np.any(hi < lo):
            return np.zeros((0, 3), dtype=np.int64)
        c_lo = self._cell(np.maximum(lo, self.origin)[None])[0]
        c_hi = self._cell(np.minimum(hi, world_hi)[None])[0]
        axes = [np.arange(c_lo[i], c_hi[i] + 1) for i in range(3)]
        return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)

    def _inside(self, region, cells: npt.NDArray[np.int64]) -> npt.NDArray[np.bool_]:
        """Which cells lie wholly inside the region (always False for a disk or mask).

        A segment filed under one cell only lies in that cell, so in such a
        cell it meets the region without being tested.  The cells are
        widened by a hair, so rounding can only make a cell count as not
        inside.
        """
        if len(cells) == 0 or not isinstance(region, (Sphere, Box, Slab)):
            return np.zeros(len(cells), dtype=bool)
        lo = self.origin + cells * self.cell_size
        hi = lo + self.cell_size
        lo = lo - 1e-9 * (1.0 + np.abs(lo) + self.cell_size)
        hi = hi + 1e-9 * (1.0 + np.abs(hi) + self.cell_size)
        if isinstance(region, Sphere):
            c = np.asarray(region.center)
            far = np.maximum(np.abs(lo - c), np.abs(hi - c))
            return np.einsum("ij,ij->i", far, far) <= region.radius ** 2
        if isinstance(region, Box):
            return np.all(lo >= np.asarray(region.min), axis=1) & np.all(
                hi <= np.asarray(region.max), axis=1)
        return (lo[:, region.axis] >= region.min) & (hi[:, region.axis] <= region.max)

    def _positions(self, cells: npt.NDArray[np.int64]) -> npt.NDArray[np.intp]:
        """Positions in the filing of every segment filed under ``cells``."""
        linear = self._linear(cells)
        begin = self._cell_start[linear]
        counts = (self._cell_start[linear + 1] - begin).astype(np.intp)
        total = int(counts.sum())
        k = np.arange(total, dtype=np.intp) - np.repeat(np.cumsum(counts) - counts, counts)
        return np.repeat(begin, counts).astype(np.intp) + k

    def _passes_path(self, region) -> npt.NDArray[np.bool_]:
        out = np.zeros(self.tractogram.n_objects, dtype=bool)
        cells = self._cells(region)
        inside = self._inside(region, cells)
        pos = self._positions(cells[inside])
        single = self._single[pos]
        sure = self._filed[pos[single]]
        out[self._owner[sure]] = True
        pos = np.concatenate([pos[~single], self._positions(cells[~inside])])
        test = np.concatenate([self._filed[pos], self._long])
        tested = 0
        vertices = self.tractogram.vertices
        for first in range(0, len(test), _QUERY_BLOCK):
            # Skip segments of streamlines already accepted.
            chunk = test[first:first + _QUERY_BLOCK]
            chunk = chunk[~out[self._owner[chunk]]]
            tested += len(chunk)
            a = vertices[self._start[chunk]].astype(np.float64)
            b = vertices[self._end[chunk]].astype(np.float64)
            out[self._owner[chunk[segment_hits(a, b, region)]]] = True
        self.last_query = {
            "cells": int(len(cells)), "inside_cells": int(inside.sum()),
            "accepted_untested": int(len(sure)), "candidates": int(len(test)),
            "tested": int(tested),
        }
        return out

    # -- leaves ----------------------------------------------------------------

    def leaf(self, node: Node) -> npt.NDArray[np.bool_]:
        """A leaf's answer, from the cache when the same leaf was asked before."""
        key = node.key()
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        result = self._compute(node)
        result.setflags(write=False)
        self._cache[key] = result
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return result

    def _compute(self, node: Node) -> npt.NDArray[np.bool_]:
        n = self.tractogram.n_objects
        if isinstance(node, Passes):
            if node.mode != "path":
                has = self._has
                at_first = contains(self._first[has], node.region)
                at_last = contains(self._last[has], node.region)
                out = np.zeros(n, dtype=bool)
                out[has] = at_first | at_last if node.mode == "endpoints" else at_first & at_last
                return out
            return self._passes_path(node.region)
        if isinstance(node, Column):
            return oracle.column_mask(node, self.tractogram)
        if isinstance(node, InGroup):
            return self.tractogram.group_mask(node.ids)
        raise TypeError(f"{type(node).__name__} is not a leaf")

    # -- trees -----------------------------------------------------------------

    def evaluate(self, tree: Node) -> npt.NDArray[np.bool_]:
        """Which streamlines the filter accepts: an ``(N,)`` bool array by row."""
        if isinstance(tree, All):
            return np.ones(self.tractogram.n_objects, dtype=bool)
        if isinstance(tree, And):
            out = self.evaluate(tree.children[0]).copy()
            for child in tree.children[1:]:
                out &= self.evaluate(child)
            return out
        if isinstance(tree, Or):
            out = self.evaluate(tree.children[0]).copy()
            for child in tree.children[1:]:
                out |= self.evaluate(child)
            return out
        if isinstance(tree, Not):
            return ~self.evaluate(tree.child)
        return self.leaf(tree)

    def clear(self) -> None:
        """Forget every cached leaf (after the tractogram changes)."""
        self._cache.clear()

    @staticmethod
    def publish(groups: Sequence[npt.NDArray[np.bool_]]) -> npt.NDArray[np.uint8]:
        """One byte per row for the viewer: 0 hidden, else 1 + the first group showing it.

        Groups are in draw order; at most 255 fit a byte.
        """
        if len(groups) > 255:
            raise ValueError(f"at most 255 groups fit a byte mask, got {len(groups)}")
        if not groups:
            return np.zeros(0, dtype=np.uint8)
        out = np.zeros(len(groups[0]), dtype=np.uint8)
        for index in range(len(groups) - 1, -1, -1):
            out[groups[index]] = index + 1
        return out
