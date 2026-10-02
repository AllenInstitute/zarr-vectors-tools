"""Spatial queries on chunked mesh stores: closest-point and ray-cast.

Both localise the search to chunks -- rings of chunks around the query, or
the chunks a ray walks through -- and test the faces each one holds, plus
the faces that span chunks and may pass through it (see
:meth:`~zarr_vectors_tools.algorithms._mesh_faces.SpanningFaces.faces_through`).
Every face is tested, whichever chunks its corners lie in, so the answer
does not depend on the chunk shape.  Faces of four or more corners are split
into a fan of triangles.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from zarr_vectors.building import (
    chunks_intersecting_bbox,
    get_resolution_level,
    list_chunk_keys,
    open_store,
    read_root_metadata,
)
from zarr_vectors.constants import (
    VERTEX_FRAGMENTS,
    VERTICES,
)
from zarr_vectors.typing import ChunkCoords

from zarr_vectors_tools.algorithms._links import (
    chunk_key_str,
    link_prefetch_plan,
)
from zarr_vectors_tools.algorithms._mesh_faces import (
    SpanningFaces,
    chunk_faces,
    chunk_positions,
    face_width,
    fan,
)

# =====================================================================
# Helpers
# =====================================================================

def _closest_point_on_triangle(
    p: np.ndarray,         # (3,)
    a: np.ndarray,         # (F, 3)
    b: np.ndarray,         # (F, 3)
    c: np.ndarray,         # (F, 3)
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised closest-point-on-triangle (Eberly 2001).

    Returns:
        ``(points, dist2)`` where ``points`` is ``(F, 3)`` and
        ``dist2`` is ``(F,)`` squared distances from ``p`` to each
        triangle's closest point.
    """
    ab = b - a
    ac = c - a
    ap = p[None, :] - a

    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)

    # Region 1: vertex A.
    out = a.copy()
    region_set = (d1 <= 0) & (d2 <= 0)

    # Region 2: vertex B.
    bp = p[None, :] - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    m2 = (~region_set) & (d3 >= 0) & (d4 <= d3)
    out = np.where(m2[:, None], b, out)
    region_set = region_set | m2

    # Region 3: edge AB.
    vc = d1 * d4 - d3 * d2
    m3 = (~region_set) & (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    denom_ab = np.where(d1 - d3 != 0, d1 - d3, 1.0)
    v_ab = (d1 / denom_ab)[:, None]
    out = np.where(m3[:, None], a + ab * v_ab, out)
    region_set = region_set | m3

    # Region 4: vertex C.
    cp = p[None, :] - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    m4 = (~region_set) & (d6 >= 0) & (d5 <= d6)
    out = np.where(m4[:, None], c, out)
    region_set = region_set | m4

    # Region 5: edge AC.
    vb = d5 * d2 - d1 * d6
    m5 = (~region_set) & (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    denom_ac = np.where(d2 - d6 != 0, d2 - d6, 1.0)
    w_ac = (d2 / denom_ac)[:, None]
    out = np.where(m5[:, None], a + ac * w_ac, out)
    region_set = region_set | m5

    # Region 6: edge BC.
    va = d3 * d6 - d5 * d4
    m6 = (~region_set) & (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    denom_bc = np.where(((d4 - d3) + (d5 - d6)) != 0,
                         (d4 - d3) + (d5 - d6), 1.0)
    w_bc = ((d4 - d3) / denom_bc)[:, None]
    out = np.where(m6[:, None], b + (c - b) * w_bc, out)
    region_set = region_set | m6

    # Region 0 (interior): everything else.
    denom = va + vb + vc
    safe_denom = np.where(denom != 0, denom, 1.0)
    v_in = (vb / safe_denom)[:, None]
    w_in = (vc / safe_denom)[:, None]
    interior_point = a + ab * v_in + ac * w_in
    out = np.where(region_set[:, None], out, interior_point)

    dist2 = np.einsum("ij,ij->i", p[None, :] - out, p[None, :] - out)
    return out, dist2


def _moller_trumbore(
    origin: np.ndarray,        # (3,)
    direction: np.ndarray,     # (3,) unit-ish
    a: np.ndarray,             # (F, 3)
    b: np.ndarray,
    c: np.ndarray,
    eps: float = 1e-7,
) -> np.ndarray:
    """Vectorised Möller–Trumbore. Returns ``t`` per face; ``nan`` for miss.

    Only positive ``t`` (in front of origin) and barycentrics within
    ``[0, 1]`` are reported. Caller picks the minimum.
    """
    edge1 = b - a
    edge2 = c - a
    # h = direction x edge2; direction is (3,) and edge2 is (F, 3), so
    # np.cross broadcasts to (F, 3) — both operands of the dot are
    # per-row, hence "ij,ij->i".
    h = np.cross(direction, edge2)
    det = np.einsum("ij,ij->i", edge1, h)

    parallel = np.abs(det) < eps
    inv_det = np.where(parallel, 1.0, 1.0 / np.where(parallel, 1.0, det))

    s = origin[None, :] - a
    u = inv_det * np.einsum("ij,ij->i", s, h)
    miss_u = (u < 0) | (u > 1)

    q = np.cross(s, edge1)
    v = inv_det * np.einsum("j,ij->i", direction, q)
    miss_v = (v < 0) | (u + v > 1)

    t = inv_det * np.einsum("ij,ij->i", edge2, q)
    miss_t = t <= eps

    miss = parallel | miss_u | miss_v | miss_t
    return np.where(miss, np.nan, t)


# =====================================================================
# Public API
# =====================================================================


def _level_chunk_shape(root, root_meta, level: int) -> tuple[float, ...]:
    """The chunk edge lengths in force at ``level``.

    A coarser level may declare its own, larger ``chunk_shape``; when it does
    not, it inherits the root's.  Core owns the fallback rule, so ask it
    rather than re-deriving one here.
    """
    from zarr_vectors.building import get_level_chunk_shape, read_level_metadata

    try:
        level_meta = read_level_metadata(root, level)
    except Exception:  # noqa: BLE001 - a level with no metadata inherits root
        level_meta = None
    return get_level_chunk_shape(root_meta, level_meta)


class _Faces:
    """The faces a query tests on reaching each chunk, every face once.

    On reaching a chunk: its own faces, and the spanning faces whose chunk
    box contains it that no earlier chunk has handed out.  Chunk vertex rows
    are read once and kept, since a spanning face needs its neighbours'.
    """

    def __init__(self, level_group: Any, width: int, vertex_dtype: Any, ndim: int) -> None:
        self.level_group = level_group
        self.width = width
        self._read = {"dtype": vertex_dtype, "ndim": ndim}
        self._positions: dict[ChunkCoords, np.ndarray | None] = {}
        self.spanning = SpanningFaces(level_group)
        #: chunk -> the spanning faces that may pass through it.
        self.through = self.spanning.faces_through()
        self._handed_out = np.zeros(len(self.spanning), dtype=bool)
        self._collected: set[ChunkCoords] = set()

    def positions(self, chunk: ChunkCoords) -> np.ndarray | None:
        if chunk not in self._positions:
            self._positions[chunk] = chunk_positions(self.level_group, chunk, **self._read)
        return self._positions[chunk]

    def at(self, chunk: ChunkCoords) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, Callable]]:
        """Triangle batches ``(a, b, c, describe)`` to test on reaching ``chunk``.

        ``describe(i)`` gives ``(chunk_key, face_index, corners)`` for the face
        triangle ``i`` of the batch came from.
        """
        batches: list[tuple[np.ndarray, np.ndarray, np.ndarray, Callable]] = []
        positions = self.positions(chunk)
        groups = chunk_faces(self.level_group, chunk, self.width) if positions is not None else []
        if groups:
            own = np.concatenate(groups, axis=0)
            triangles, source = fan(own)

            def describe_own(i: int, own=own, source=source) -> tuple:
                face = int(source[i])
                return chunk, face, tuple((chunk, int(v)) for v in own[face])

            batches.append((
                positions[triangles[:, 0]], positions[triangles[:, 1]],
                positions[triangles[:, 2]], describe_own,
            ))

        passing = self.through.get(chunk)
        if passing is not None:
            passing = passing[~self._handed_out[passing]]
        if passing is not None and passing.size:
            self._handed_out[passing] = True
            for k in np.unique(self.spanning.chunk_id[passing]).tolist():
                corner_chunk = self.spanning.chunk_list[k]
                if corner_chunk not in self._collected:
                    self._collected.add(corner_chunk)
                    self.spanning.collect(corner_chunk, self.positions(corner_chunk))
            self.spanning.require_complete(passing)
            a, b, c, _slots, source = self.spanning.triangles(passing)

            def describe_spanning(i: int, source=source) -> tuple:
                corners = self.spanning.record(int(source[i]))
                return corners[0][0], None, corners

            batches.append((a, b, c, describe_spanning))
        return batches


def closest_point(
    store_path: str | Path,
    query: np.ndarray,
    *,
    level: int = 0,
    max_distance: float | None = None,
    max_expansion_rings: int = 4,
) -> dict[str, Any]:
    """Closest point on a chunked mesh to a query point.

    Args:
        store_path: Path to the mesh store.
        query: ``(3,)`` query point.
        level: Resolution level.
        max_distance: Optional upper bound on the search radius. When
            ``None``, the search expands rings of neighbouring chunks
            until a hit is found or all chunks have been checked.
        max_expansion_rings: Safety cap on the number of halo expansions
            (each ring widens the search bbox by one chunk size).

    Returns:
        Dict with:
          - ``found`` (bool): True if a candidate face was found.
          - ``position`` (np.ndarray (3,)): the closest point on the
            mesh, or ``query`` if no face was found.
          - ``distance`` (float): Euclidean distance.
          - ``chunk_key`` (tuple | None): chunk holding the winning face's
            first corner.
          - ``face_index`` (int | None): the winning face's index among
            ``chunk_key``'s own faces; ``None`` for a face whose corners lie
            in more than one chunk.
          - ``corners`` (tuple | None): the winning face's corners, each
            ``(chunk, index in chunk)``, in its stored winding.

    Raises:
        NotImplementedError: If the level holds no faces of three or more
            corners.
    """
    query = np.asarray(query, dtype=np.float64).reshape(3)

    root = open_store(str(store_path))
    root_meta = read_root_metadata(root)
    level_group = get_resolution_level(root, level)
    # A coarse level may carry its own, larger chunk_shape.  Taking the
    # root's regardless put every chunk lookup on the wrong grid above
    # level 0, so the bbox scan visited chunks that do not exist and missed
    # the ones that do.
    chunk_shape = np.asarray(
        _level_chunk_shape(root, root_meta, level), dtype=np.float64,
    )
    ndim = root_meta.sid_ndim

    vmeta = level_group.read_array_meta("vertices")
    vertex_dtype = np.dtype(vmeta.get("dtype", "float32"))

    width = face_width(level_group, what="closest_point")

    occupied = set(list_chunk_keys(level_group))
    occupied_chunk_strs = [chunk_key_str(cc) for cc in occupied]

    best_dist2 = (
        np.inf if max_distance is None else float(max_distance) ** 2
    )
    best_point = query.copy()
    best: tuple | None = None

    visited: set[ChunkCoords] = set()

    with level_group.batched_reads([
        (VERTICES, occupied_chunk_strs),
        (VERTEX_FRAGMENTS, occupied_chunk_strs),
        *link_prefetch_plan(level_group, occupied),
    ]):
        faces = _Faces(level_group, width, vertex_dtype, ndim)
        # A face spanning chunks can pass through a chunk holding no vertex.
        searchable = occupied | set(faces.through)
        for ring in range(max_expansion_rings + 1):
            radius = (ring + 0.5) * chunk_shape
            lo = query - radius
            hi = query + radius
            candidates = set(
                chunks_intersecting_bbox(lo, hi, tuple(chunk_shape))
            )
            new = [c for c in candidates if c in searchable and c not in visited]
            if not new:
                # An empty ring is not the end of the search: the mesh may
                # simply start further out.  Breaking here made every query
                # more than ~1.5 chunks from the surface return found=False
                # however many rings the caller allowed.  Stop only once
                # nothing is left to visit.
                if len(visited) >= len(searchable):
                    break
                continue

            for chunk_key in new:
                visited.add(chunk_key)
                for a, b, c, describe in faces.at(chunk_key):
                    pts, dist2 = _closest_point_on_triangle(query, a, b, c)
                    local_argmin = int(np.argmin(dist2))
                    if dist2[local_argmin] < best_dist2:
                        best_dist2 = float(dist2[local_argmin])
                        best_point = pts[local_argmin]
                        best = describe(local_argmin)

            # If we found something within the current ring radius, the
            # answer is final: a closer face would have its nearest point
            # inside the searched bbox, so in a visited chunk, where it was
            # tested -- its own chunk, or one it passes through.
            if best is not None and best_dist2 <= np.min(radius) ** 2:
                break

    found = best is not None and np.isfinite(best_dist2)
    chunk_key, face_index, corners = best if best is not None else (None, None, None)
    return {
        "found": bool(found),
        "position": best_point.astype(np.float64),
        "distance": float(np.sqrt(best_dist2)) if found else float("inf"),
        "chunk_key": chunk_key,
        "face_index": face_index,
        "corners": corners,
    }


def cast_ray(
    store_path: str | Path,
    origin: np.ndarray,
    direction: np.ndarray,
    *,
    level: int = 0,
    max_distance: float | None = None,
) -> dict[str, Any]:
    """First-hit intersection of a ray with a chunked mesh.

    Walks the chunk grid via 3D DDA along ``direction``, testing each
    visited chunk's faces, and the faces spanning chunks that pass through
    it, via Möller–Trumbore.  Stops once the nearest hit so far lies no
    further than where the ray leaves the current chunk, or at
    ``max_distance``.

    Args:
        store_path: Path to the mesh store.
        origin: ``(3,)`` ray origin.
        direction: ``(3,)`` direction; will be normalised internally.
        level: Resolution level.
        max_distance: Optional upper bound on hit distance.

    Returns:
        Dict with:
          - ``hit`` (bool)
          - ``t`` (float): ray parameter at the hit; ``inf`` for miss.
          - ``position`` (np.ndarray (3,)): hit position.
          - ``chunk_key`` (tuple | None): chunk holding the hit face's first
            corner.
          - ``face_index`` (int | None): the hit face's index among
            ``chunk_key``'s own faces; ``None`` for a face whose corners lie
            in more than one chunk.
          - ``corners`` (tuple | None): the hit face's corners, each
            ``(chunk, index in chunk)``, in its stored winding.

    Raises:
        ValueError: If ``direction`` is zero.
        NotImplementedError: If the level holds no faces of three or more
            corners.
    """
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    dnorm = float(np.linalg.norm(direction))
    if dnorm == 0:
        raise ValueError("ray direction must be non-zero")
    direction = direction / dnorm

    root = open_store(str(store_path))
    root_meta = read_root_metadata(root)
    level_group = get_resolution_level(root, level)
    # See closest_point: the grid is the LEVEL's, not the root's.
    chunk_shape = np.asarray(
        _level_chunk_shape(root, root_meta, level), dtype=np.float64,
    )
    ndim = root_meta.sid_ndim

    vmeta = level_group.read_array_meta("vertices")
    vertex_dtype = np.dtype(vmeta.get("dtype", "float32"))
    width = face_width(level_group, what="cast_ray")

    occupied = set(list_chunk_keys(level_group))
    if not occupied:
        return {
            "hit": False, "t": float("inf"), "position": origin,
            "chunk_key": None, "face_index": None, "corners": None,
        }
    occupied_chunk_strs = [chunk_key_str(cc) for cc in occupied]

    # 3D DDA on chunk coordinates.
    start_chunk = tuple(int(np.floor(origin[d] / chunk_shape[d])) for d in range(3))
    step = np.sign(direction).astype(np.int64)
    # Distance along the ray to the next chunk-grid plane on each axis.
    next_boundary = np.array([
        (np.floor(origin[d] / chunk_shape[d]) + (1 if step[d] > 0 else 0))
        * chunk_shape[d]
        for d in range(3)
    ])
    safe_dir = np.where(direction != 0, direction, 1.0)
    t_max = np.abs((next_boundary - origin) / safe_dir)
    t_max = np.where(direction != 0, t_max, np.inf)
    t_delta = np.abs(chunk_shape / safe_dir)
    t_delta = np.where(direction != 0, t_delta, np.inf)

    cur = list(start_chunk)
    travelled = 0.0
    max_t = max_distance if max_distance is not None else np.inf

    best_t = np.inf
    best_position = origin.copy()
    best: tuple | None = None

    # Halt criterion: along some axis we've permanently exited the bbox
    # of occupied chunks (i.e. step is taking us further away and we're
    # already past it). The previous "outside by more than 1" check would
    # fire while the ray was still walking *towards* the mesh.  A face
    # spanning chunks lies within the box of its corners' chunks, which
    # hold vertices, so this box bounds those faces too.
    occupied_min = np.min(np.array(list(occupied)), axis=0)
    occupied_max = np.max(np.array(list(occupied)), axis=0)

    def _ray_past_bbox(cur_pos: list[int]) -> bool:
        for d in range(3):
            s = int(step[d])
            if s > 0 and cur_pos[d] > occupied_max[d]:
                return True
            if s < 0 and cur_pos[d] < occupied_min[d]:
                return True
            if s == 0 and (
                cur_pos[d] < occupied_min[d] or cur_pos[d] > occupied_max[d]
            ):
                return True
        return False

    with level_group.batched_reads([
        (VERTICES, occupied_chunk_strs),
        (VERTEX_FRAGMENTS, occupied_chunk_strs),
        *link_prefetch_plan(level_group, occupied),
    ]):
        faces = _Faces(level_group, width, vertex_dtype, ndim)
        while travelled <= max_t:
            cur_key: ChunkCoords = tuple(cur)
            if cur_key in occupied or cur_key in faces.through:
                for a, b, c, describe in faces.at(cur_key):
                    ts = _moller_trumbore(origin, direction, a, b, c)
                    finite = np.where(np.isfinite(ts), ts, np.inf)
                    local_argmin = int(np.argmin(finite))
                    if finite[local_argmin] < best_t and finite[local_argmin] <= max_t:
                        best_t = float(finite[local_argmin])
                        best_position = origin + best_t * direction
                        best = describe(local_argmin)

            # Step to next chunk along axis with smallest t_max.
            axis = int(np.argmin(t_max))
            # A chunk's own faces are hit inside it, but a face spanning
            # chunks may be hit further on.  A hit no further than where the
            # ray leaves this chunk is final: any face not yet tested is hit,
            # if at all, in a chunk still ahead.
            if best is not None and best_t <= t_max[axis]:
                break
            travelled = float(t_max[axis])
            cur[axis] += int(step[axis])
            t_max[axis] += t_delta[axis]

            if _ray_past_bbox(cur):
                break

    hit = best is not None
    chunk_key, face_index, corners = best if hit else (None, None, None)
    return {
        "hit": bool(hit),
        "t": best_t if hit else float("inf"),
        "position": best_position,
        "chunk_key": chunk_key,
        "face_index": face_index,
        "corners": corners,
    }
