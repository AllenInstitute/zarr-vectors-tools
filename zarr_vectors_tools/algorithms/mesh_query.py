"""Spatial queries on chunked mesh stores: closest-point and ray-cast.

Both use the existing ``chunks_intersecting_bbox`` helper to localise
candidate chunks, then test against the intra-chunk triangle set of
each candidate. Cross-chunk faces lose identity in the current core
storage and are not tested — for typical meshes that's a tiny minority
of faces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from zarr_vectors.core.arrays import (
    list_chunk_keys,
    read_chunk_links,
    read_chunk_vertices,
)
from zarr_vectors.core.store import get_resolution_level, open_store, read_root_metadata
from zarr_vectors.spatial.chunking import chunks_intersecting_bbox
from zarr_vectors.typing import ChunkCoords


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
          - ``chunk_key`` (tuple | None): chunk containing the winning face.
          - ``face_index`` (int | None): local index of the winning face
            inside that chunk.

    Cross-chunk faces lose identity in current core storage and are not
    tested. For typical meshes this is a negligible minority.
    """
    query = np.asarray(query, dtype=np.float64).reshape(3)

    root = open_store(str(store_path))
    root_meta = read_root_metadata(root)
    level_group = get_resolution_level(root, level)
    chunk_shape = np.asarray(root_meta.chunk_shape, dtype=np.float64)
    ndim = root_meta.sid_ndim

    vmeta = level_group.read_array_meta("vertices")
    vertex_dtype = np.dtype(vmeta.get("dtype", "float32"))

    try:
        lmeta = level_group.read_array_meta("links/0")
        link_width = int(lmeta.get("link_width", 3))
    except Exception:
        link_width = 3
    if link_width != 3:
        raise NotImplementedError(
            f"closest_point v0 supports triangle meshes only "
            f"(link_width=3); store has link_width={link_width}."
        )

    occupied = set(list_chunk_keys(level_group))

    best_dist2 = (
        np.inf if max_distance is None else float(max_distance) ** 2
    )
    best_point = query.copy()
    best_chunk: ChunkCoords | None = None
    best_face_index: int | None = None

    visited: set[ChunkCoords] = set()

    for ring in range(max_expansion_rings + 1):
        radius = (ring + 0.5) * chunk_shape
        lo = query - radius
        hi = query + radius
        candidates = set(
            chunks_intersecting_bbox(lo, hi, tuple(chunk_shape))
        )
        new = [c for c in candidates if c in occupied and c not in visited]
        if not new:
            if ring == 0:
                continue
            break

        for chunk_key in new:
            visited.add(chunk_key)
            try:
                vgroups = read_chunk_vertices(
                    level_group, chunk_key, dtype=vertex_dtype, ndim=ndim,
                )
            except Exception:
                continue
            if not vgroups:
                continue
            positions = np.concatenate(vgroups, axis=0).astype(np.float64)

            try:
                link_groups = read_chunk_links(level_group, chunk_key)
            except Exception:
                continue

            running_local_offset = 0
            for faces in link_groups:
                if len(faces) == 0:
                    continue
                a = positions[faces[:, 0]]
                b = positions[faces[:, 1]]
                c = positions[faces[:, 2]]
                pts, dist2 = _closest_point_on_triangle(query, a, b, c)
                local_argmin = int(np.argmin(dist2))
                if dist2[local_argmin] < best_dist2:
                    best_dist2 = float(dist2[local_argmin])
                    best_point = pts[local_argmin]
                    best_chunk = chunk_key
                    best_face_index = running_local_offset + local_argmin
                running_local_offset += len(faces)

        # If we found something within the current ring radius, the
        # answer is final (any closer face would be inside the searched
        # bbox).
        if best_chunk is not None and best_dist2 <= np.min(radius) ** 2:
            break

    found = best_chunk is not None and np.isfinite(best_dist2)
    return {
        "found": bool(found),
        "position": best_point.astype(np.float64),
        "distance": float(np.sqrt(best_dist2)) if found else float("inf"),
        "chunk_key": best_chunk,
        "face_index": best_face_index,
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

    Walks the chunk grid via 3D DDA along ``direction``. Tests each
    visited chunk's intra-chunk faces via Möller–Trumbore. Stops at the
    first hit or ``max_distance``.

    Cross-chunk faces lose identity in current core storage and are not
    tested. For typical meshes this is a negligible minority.

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
          - ``chunk_key`` (tuple | None)
          - ``face_index`` (int | None)
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
    chunk_shape = np.asarray(root_meta.chunk_shape, dtype=np.float64)
    ndim = root_meta.sid_ndim

    vmeta = level_group.read_array_meta("vertices")
    vertex_dtype = np.dtype(vmeta.get("dtype", "float32"))
    try:
        lmeta = level_group.read_array_meta("links/0")
        link_width = int(lmeta.get("link_width", 3))
    except Exception:
        link_width = 3
    if link_width != 3:
        raise NotImplementedError(
            f"cast_ray v0 supports triangle meshes only "
            f"(link_width=3); store has link_width={link_width}."
        )

    occupied = set(list_chunk_keys(level_group))
    if not occupied:
        return {
            "hit": False, "t": float("inf"),
            "position": origin, "chunk_key": None, "face_index": None,
        }

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
    best_chunk: ChunkCoords | None = None
    best_face: int | None = None

    # Halt criterion: along some axis we've permanently exited the bbox
    # of occupied chunks (i.e. step is taking us further away and we're
    # already past it). The previous "outside by more than 1" check would
    # fire while the ray was still walking *towards* the mesh.
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

    while travelled <= max_t:
        cur_key: ChunkCoords = tuple(cur)
        if cur_key in occupied:
            try:
                vgroups = read_chunk_vertices(
                    level_group, cur_key, dtype=vertex_dtype, ndim=ndim,
                )
                positions = (
                    np.concatenate(vgroups, axis=0).astype(np.float64)
                    if vgroups else None
                )
            except Exception:
                positions = None

            if positions is not None and len(positions):
                try:
                    link_groups = read_chunk_links(level_group, cur_key)
                except Exception:
                    link_groups = []

                running_local_offset = 0
                for faces in link_groups:
                    if len(faces) == 0:
                        continue
                    a = positions[faces[:, 0]]
                    b = positions[faces[:, 1]]
                    c = positions[faces[:, 2]]
                    ts = _moller_trumbore(origin, direction, a, b, c)
                    finite = np.where(np.isfinite(ts), ts, np.inf)
                    local_argmin = int(np.argmin(finite))
                    if finite[local_argmin] < best_t:
                        best_t = float(finite[local_argmin])
                        best_position = origin + best_t * direction
                        best_chunk = cur_key
                        best_face = running_local_offset + local_argmin
                    running_local_offset += len(faces)

                if best_chunk is not None:
                    break  # first hit wins along the ray walk

        # Step to next chunk along axis with smallest t_max.
        axis = int(np.argmin(t_max))
        travelled = float(t_max[axis])
        cur[axis] += int(step[axis])
        t_max[axis] += t_delta[axis]

        if _ray_past_bbox(cur):
            break

    hit = best_chunk is not None
    return {
        "hit": bool(hit),
        "t": best_t if hit else float("inf"),
        "position": best_position,
        "chunk_key": best_chunk,
        "face_index": best_face,
    }


