"""A mesh level's faces, for the chunk-by-chunk mesh algorithms.

Core stores every face of a level in one links family (see
``zarr_vectors.types.meshes``).  A face whose corners share a chunk sits in
that chunk's intra cell and is read with the chunk; a face whose corners lie
in different chunks sits in the cell naming the offsets between them.  The
mesh algorithms stream chunks, so they read the first kind chunk by chunk
(:func:`chunk_faces`) and hold the second kind, which is a thin band along
the chunk boundaries, in a :class:`SpanningFaces` that collects each corner's
position as its chunk streams past.

Faces may have any number of corners, three or more: a store made from an
all-quad OBJ keeps its quads.  :func:`fan` splits a face ``(a, b, c, d)``
into ``(a, b, c)`` and ``(a, c, d)``, so every algorithm sees triangles.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import numpy.typing as npt
from zarr_vectors.building import (
    link_endpoints_to_rows,
    link_family_policy,
    read_chunk_links,
    read_chunk_vertices,
    read_link_arrays,
)
from zarr_vectors.typing import ChunkCoords

__all__ = ["SpanningFaces", "chunk_faces", "chunk_positions", "face_width", "fan"]


def face_width(level_group: Any, *, what: str) -> int:
    """Corners per face at this level: 3 for triangles, 4 for quads.

    Raises ``NotImplementedError`` for a level with no links family, or one
    whose links have fewer than three endpoints (a graph or skeleton).
    """
    policy = link_family_policy(level_group, 0)
    if policy is None:
        raise NotImplementedError(
            f"{what} needs a mesh level; this level has no links/0 family"
        )
    width = int(policy[0])
    if width < 3:
        raise NotImplementedError(
            f"{what} needs faces of three or more corners; this level's links "
            f"have {width} endpoints, so it is not a mesh"
        )
    return width


def fan(faces: npt.NDArray[np.integer]) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """``(triangles, source)``: each face split into a fan of triangles.

    ``(a, b, c, d)`` becomes ``(a, b, c)`` and ``(a, c, d)``, keeping the
    face's winding; a triangle is returned as it is.  ``source[i]`` is the
    row of ``faces`` triangle ``i`` came from.
    """
    faces = np.asarray(faces, dtype=np.int64)
    n, width = faces.shape
    if width == 3:
        return faces, np.arange(n, dtype=np.int64)
    k = width - 2
    triangles = np.stack([
        np.repeat(faces[:, :1], k, axis=1), faces[:, 1:width - 1], faces[:, 2:width],
    ], axis=2).reshape(-1, 3)
    return triangles, np.repeat(np.arange(n, dtype=np.int64), k)


def polygon_edges(faces: npt.NDArray[np.integer]) -> npt.NDArray[np.int64]:
    """``(F * L, 2)`` corner pairs round each face's boundary, unsorted.

    The boundary only: a quad has four edges, not the diagonal :func:`fan`
    adds.
    """
    faces = np.asarray(faces, dtype=np.int64)
    return np.stack([faces, np.roll(faces, -1, axis=1)], axis=2).reshape(-1, 2)


def chunk_positions(
    level_group: Any, chunk: ChunkCoords, *, dtype: Any = None, ndim: int | None = None,
) -> npt.NDArray[np.float64] | None:
    """The chunk's vertex rows as ``float64``, or ``None`` when it holds none."""
    kwargs: dict[str, Any] = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if ndim is not None:
        kwargs["ndim"] = ndim
    try:
        groups = read_chunk_vertices(level_group, chunk, **kwargs)
    except Exception:  # noqa: BLE001 - an absent or unreadable chunk holds nothing
        return None
    if not groups:
        return None
    return np.concatenate(groups, axis=0).astype(np.float64, copy=False)


def chunk_faces(level_group: Any, chunk: ChunkCoords, width: int) -> list[npt.NDArray[np.int64]]:
    """The faces whose corners all lie in ``chunk``, as ``(F_k, width)`` groups.

    Corner indices are rows of the chunk's vertex buffer.  Empty groups are
    dropped, and so is a group of another width, which a mesh level does not
    hold: reading it at the wrong width would mis-read every face.
    """
    try:
        groups = read_chunk_links(level_group, chunk)
    except Exception:  # noqa: BLE001 - a chunk with no faces of its own
        return []
    out = []
    for group in groups:
        arr = np.asarray(group, dtype=np.int64)
        if arr.ndim == 2 and arr.shape[1] == width and len(arr):
            out.append(arr)
    return out


class SpanningFaces:
    """The faces whose corners lie in more than one chunk.

    Read once, whole, from the link family's non-intra cells, with each
    face's stored winding restored.  Corner positions start unknown; pass
    each chunk's vertex rows to :meth:`collect` as the chunk is read, and the
    faces become usable once every corner's chunk has been seen.
    """

    def __init__(self, level_group: Any) -> None:
        chunks, vi = read_link_arrays(level_group, delta=0, include_intra=False)
        #: ``(M, L, D)`` chunk of each corner, ``(M, L)`` its row in that chunk.
        self.chunks = np.asarray(chunks, dtype=np.int64)
        self.vi = np.asarray(vi, dtype=np.int64)
        self.n = int(self.vi.shape[0])
        self.width = int(self.vi.shape[1]) if self.n else 0
        self._xyz: npt.NDArray[np.float64] | None = None
        self._found = np.zeros(self.n * self.width, dtype=bool)
        self._by_chunk: dict[ChunkCoords, npt.NDArray[np.int64]] = {}
        #: ``(M, L)`` each corner's chunk, as an index into :attr:`chunk_list`.
        self.chunk_id = np.zeros((self.n, self.width), dtype=np.int64)
        self.chunk_list: list[ChunkCoords] = []
        if not self.n:
            return
        flat = self.chunks.reshape(self.n * self.width, -1)
        unique, inverse = np.unique(flat, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)
        self.chunk_id = inverse.reshape(self.n, self.width)
        self.chunk_list = [tuple(chunk) for chunk in unique.tolist()]
        order = np.argsort(inverse, kind="stable")
        bounds = np.cumsum(np.bincount(inverse, minlength=len(unique)))[:-1]
        for chunk, slots in zip(self.chunk_list, np.split(order, bounds)):
            self._by_chunk[chunk] = slots

    def __len__(self) -> int:
        return self.n

    def corners_in(
        self, chunk: ChunkCoords,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]] | None:
        """``(slots, rows)`` of the corners in ``chunk``, or ``None``.

        ``slots`` index the flattened ``(M * L)`` corners; ``rows`` are their
        rows in the chunk's vertex buffer.
        """
        slots = self._by_chunk.get(tuple(int(c) for c in chunk))
        if slots is None:
            return None
        return slots, self.vi.reshape(-1)[slots]

    def collect(self, chunk: ChunkCoords, positions: npt.NDArray[Any] | None) -> None:
        """Record the positions of the corners in ``chunk`` from its vertex rows."""
        found = self.corners_in(chunk)
        if found is None or positions is None or not len(positions):
            return
        slots, rows = found
        ok = (rows >= 0) & (rows < len(positions))
        if self._xyz is None:
            self._xyz = np.full((self.n * self.width, positions.shape[1]), np.nan)
        self._xyz[slots[ok]] = positions[rows[ok]]
        self._found[slots[ok]] = True

    def complete(self) -> npt.NDArray[np.bool_]:
        """``(M,)``: the faces whose every corner has a position."""
        return self._found.reshape(self.n, self.width).all(axis=1)

    def require_complete(self, faces: npt.NDArray[np.int64] | None = None) -> None:
        """Raise ``ValueError`` unless every corner of ``faces`` (default all) has a position."""
        found = self._found.reshape(self.n, self.width)
        missing = ~(found if faces is None else found[np.asarray(faces, dtype=np.int64)])
        if missing.any():
            raise ValueError(
                f"{int(missing.sum())} corner(s) of faces that span chunks name a "
                f"vertex the level does not hold, so the mesh is incomplete; its "
                f"vertices presence manifest may be stale (rebuild it with "
                f"zarr_vectors.building.rebuild_presence)"
            )

    def triangles(
        self, faces: npt.NDArray[np.int64] | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64],
               npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """``(a, b, c, corner_slots, source)`` for the fanned faces named.

        ``faces`` defaults to every face.  ``a``, ``b`` and ``c`` are the
        triangles' corner positions; ``corner_slots`` ``(T, 3)`` index the
        flattened corners; ``source[i]`` is the face triangle ``i`` came from.
        """
        if faces is None:
            faces = np.arange(self.n, dtype=np.int64)
        faces = np.asarray(faces, dtype=np.int64)
        slots = faces[:, None] * self.width + np.arange(self.width, dtype=np.int64)
        tri, src = fan(slots)
        xyz = self._xyz
        if xyz is None:
            xyz = np.full((self.n * self.width, 3), np.nan)
        return xyz[tri[:, 0]], xyz[tri[:, 1]], xyz[tri[:, 2]], tri, faces[src]

    def global_rows(self, offsets: dict[ChunkCoords, int]) -> npt.NDArray[np.int64]:
        """``(M, L)`` each corner's global vertex number, given each chunk's first."""
        if not self.n:
            return np.zeros((0, self.width), dtype=np.int64)
        return np.asarray(link_endpoints_to_rows(self.chunks, self.vi, offsets), dtype=np.int64)

    def record(self, face: int) -> tuple[tuple[ChunkCoords, int], ...]:
        """Face ``face`` as ``((chunk, row), ...)``, the form core's ``read_links`` returns."""
        return tuple(
            (tuple(int(c) for c in self.chunks[face, k]), int(self.vi[face, k]))
            for k in range(self.width)
        )

    def edges(self) -> tuple[dict[ChunkCoords, npt.NDArray[np.int64]], int]:
        """``(same_chunk, n_spanning)``: the faces' boundary edges, split by chunk.

        ``same_chunk[chunk]`` holds ``(K, 2)`` row pairs for edges whose two
        corners share ``chunk`` -- an intra face can share such an edge, so
        the caller dedups them with that chunk's own.  ``n_spanning`` counts
        the distinct edges between two chunks, which only these faces have.
        """
        if not self.n:
            return {}, 0
        slots = np.arange(self.n * self.width, dtype=np.int64).reshape(self.n, self.width)
        pairs = polygon_edges(slots)
        cid = self.chunk_id.reshape(-1)
        vi = self.vi.reshape(-1)
        a, b = pairs[:, 0], pairs[:, 1]
        same = cid[a] == cid[b]

        out: dict[ChunkCoords, npt.NDArray[np.int64]] = {}
        if same.any():
            chunk_of = cid[a[same]]
            rows = np.stack([vi[a[same]], vi[b[same]]], axis=1)
            order = np.argsort(chunk_of, kind="stable")
            chunk_of, rows = chunk_of[order], rows[order]
            starts = np.flatnonzero(np.r_[True, chunk_of[1:] != chunk_of[:-1]])
            ends = np.r_[starts[1:], len(chunk_of)]
            for s, e in zip(starts.tolist(), ends.tolist()):
                out[self.chunk_list[int(chunk_of[s])]] = rows[s:e]

        cross = ~same
        n_spanning = 0
        if cross.any():
            stride = int(vi.max()) + 1 if len(vi) else 1
            key = cid * stride + vi
            ends = np.stack([key[a[cross]], key[b[cross]]], axis=1)
            ends.sort(axis=1)
            n_spanning = int(len(np.unique(ends, axis=0)))
        return out, n_spanning

    def faces_through(self) -> dict[ChunkCoords, npt.NDArray[np.int64]]:
        """For each chunk, the faces that may pass through it.

        A face lies inside the box of chunks its corners' chunks span, so a
        point on it -- a ray's hit, the point nearest a query -- lies in one
        of those chunks, which need not hold any of its corners.  Known from
        the corners' chunks alone, before any position is read.
        """
        if not self.n:
            return {}
        lo = self.chunks.min(axis=1)
        extent = self.chunks.max(axis=1) - lo
        face_parts: list[npt.NDArray[np.int64]] = []
        chunk_parts: list[npt.NDArray[np.int64]] = []
        # Neighbouring chunks, the usual case, in a few vectorised passes;
        # a face reaching further is enumerated on its own.
        small = (extent <= 1).all(axis=1)
        for offset in product((0, 1), repeat=lo.shape[1]):
            off = np.asarray(offset, dtype=np.int64)
            sel = np.flatnonzero(small & (extent >= off).all(axis=1))
            face_parts.append(sel)
            chunk_parts.append(lo[sel] + off)
        for face in np.flatnonzero(~small).tolist():
            for offset in product(*(range(int(e) + 1) for e in extent[face])):
                face_parts.append(np.array([face], dtype=np.int64))
                chunk_parts.append((lo[face] + np.asarray(offset, dtype=np.int64))[None, :])
        faces = np.concatenate(face_parts)
        chunks = np.concatenate(chunk_parts, axis=0)
        unique, inverse = np.unique(chunks, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)
        order = np.argsort(inverse, kind="stable")
        bounds = np.cumsum(np.bincount(inverse, minlength=len(unique)))[:-1]
        return {
            tuple(chunk): np.sort(group)
            for chunk, group in zip(unique.tolist(), np.split(faces[order], bounds))
        }


def triangle_area_volume(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64], c: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Per-triangle area, and signed volume of the tetrahedron to the origin."""
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    volume = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    return area, volume
