"""Streaming surface area / volume / Euler characteristic for mesh stores.

Loads each chunk's intra-chunk faces, accumulates per-face area and
signed-tetrahedron volume, and tallies edge incidence for the Euler
characteristic. Cross-chunk *edges* are read once from the global
cross-chunk array; cross-chunk *faces* (records of arity >= 3) are
treated as boundary chains for the edge dedup set but their per-face
area / volume contributions are excluded — the returned dict reports
the excluded edge count so callers can quantify the gap.

``per_object=True`` walks the object manifests instead of the chunk
grid: for each object, every fragment it owns contributes its area /
volume / face / vertex counts. Cross-chunk faces still cannot be
attributed (no spec mapping from a cross-chunk record back to an
object), so per-object totals are intra-fragment only.

Works on triangle meshes today. Quad / polygon support requires
fan-triangulation; not implemented in v0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from zarr_vectors.constants import (
    CROSS_CHUNK_LINKS,
    LINK_FRAGMENTS,
    LINKS,
    VERTEX_FRAGMENTS,
    VERTICES,
)
from zarr_vectors.core.arrays import (
    list_chunk_keys,
    read_all_object_manifests,
    read_chunk_links,
    read_chunk_vertices,
    read_cross_chunk_links,
    read_fragment,
)
from zarr_vectors.core.store import get_resolution_level, open_store, read_root_metadata
from zarr_vectors.typing import ChunkCoords


def compute_mesh_summary(
    store_path: str | Path,
    *,
    level: int = 0,
    per_object: bool = False,
) -> dict[str, Any]:
    """Streaming surface area / volume / Euler characteristic.

    Args:
        store_path: Path to a zarr-vectors mesh store.
        level: Resolution level to summarise.
        per_object: When True, the returned dict gains a ``per_object``
            list keyed by ``object_id`` with the same area / volume /
            face / vertex stats restricted to each object's fragments.
            Cross-chunk faces are excluded from per-object totals; the
            global keys still reflect the whole-store streaming pass.

    Returns:
        Dict with keys:
          - ``surface_area`` (float): sum of per-face triangle areas.
          - ``volume`` (float): signed-tetrahedron sum; meaningful only
            for closed meshes with consistent winding.
          - ``face_count`` (int): triangles contributing to the sum.
            Cross-chunk faces are excluded; see ``excluded_cross_face_edges``.
          - ``vertex_count`` (int): from level metadata.
          - ``edge_count`` (int): deduplicated edges (intra + cross).
          - ``euler_characteristic`` (int): ``V - E + F``. Accurate only
            when the store has no cross-chunk faces.
          - ``excluded_cross_face_edges`` (int): number of cross-chunk
            face-boundary edges contributed to the dedup set but whose
            per-face area / volume could not be attributed.
          - ``per_object`` (list[dict], only when ``per_object=True``):
            one dict per object_id with ``object_id``, ``surface_area``,
            ``volume``, ``face_count``, ``vertex_count``.

    Raises:
        FileNotFoundError: If the store cannot be opened.
    """
    root = open_store(str(store_path))
    root_meta = read_root_metadata(root)
    level_group = get_resolution_level(root, level)
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
            f"compute_mesh_summary v0 supports triangle meshes only "
            f"(link_width=3); store has link_width={link_width}."
        )

    chunk_keys = list_chunk_keys(level_group)
    chunk_key_strs = [".".join(str(c) for c in cc) for cc in chunk_keys]

    surface_area = 0.0
    volume = 0.0
    face_count = 0
    vertex_count = 0

    # Edge keys are ((chunk_key, local_index), (chunk_key, local_index))
    # with the smaller endpoint first. Same scheme applies to intra and
    # cross edges, so they share the deduplication set safely.
    edge_set: set[tuple[tuple, tuple]] = set()

    def _edge_key(a: tuple, b: tuple) -> tuple[tuple, tuple]:
        return (a, b) if a <= b else (b, a)

    with level_group.batched_reads([
        (VERTICES, chunk_key_strs),
        (VERTEX_FRAGMENTS, chunk_key_strs),
        (f"{LINKS}/0", chunk_key_strs),
        (LINK_FRAGMENTS, chunk_key_strs),
        (f"{CROSS_CHUNK_LINKS}/0", ["data"]),
    ]):
        for chunk_key in chunk_keys:
            try:
                vgroups = read_chunk_vertices(
                    level_group, chunk_key, dtype=vertex_dtype, ndim=ndim,
                )
            except Exception:
                continue

            if not vgroups:
                continue

            local_positions = np.concatenate(vgroups, axis=0)
            vertex_count += len(local_positions)

            try:
                link_groups = read_chunk_links(level_group, chunk_key)
            except Exception:
                link_groups = []

            for faces in link_groups:
                if len(faces) == 0:
                    continue
                face_count += len(faces)

                v0 = local_positions[faces[:, 0]]
                v1 = local_positions[faces[:, 1]]
                v2 = local_positions[faces[:, 2]]

                cross = np.cross(v1 - v0, v2 - v0)
                surface_area += float(np.linalg.norm(cross, axis=1).sum() * 0.5)
                volume += float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)

                for col_a, col_b in ((0, 1), (1, 2), (2, 0)):
                    a_locals = faces[:, col_a].tolist()
                    b_locals = faces[:, col_b].tolist()
                    for la, lb in zip(a_locals, b_locals):
                        edge_set.add(
                            _edge_key((chunk_key, int(la)), (chunk_key, int(lb)))
                        )

        try:
            cross_links = read_cross_chunk_links(level_group, delta=0)
        except Exception:
            cross_links = []
    excluded_cross_face_edges = 0
    # Records may have 2 endpoints (cross-chunk edge) or 3+ (cross-chunk face).
    # Each record contributes (len-1) consecutive edges to the dedup set.
    for record in cross_links:
        eps = [(chunk, int(vi)) for chunk, vi in record]
        if len(eps) < 2:
            continue
        for k in range(len(eps) - 1):
            edge_set.add(_edge_key(eps[k], eps[k + 1]))
            excluded_cross_face_edges += 1

    edge_count = len(edge_set)
    euler = vertex_count - edge_count + face_count

    result: dict[str, Any] = {
        "surface_area": surface_area,
        "volume": volume,
        "face_count": face_count,
        "vertex_count": vertex_count,
        "edge_count": edge_count,
        "euler_characteristic": euler,
        "excluded_cross_face_edges": excluded_cross_face_edges,
    }

    if per_object:
        result["per_object"] = _compute_per_object(
            level_group, vertex_dtype=vertex_dtype, ndim=ndim,
        )

    return result


def _compute_per_object(
    level_group,
    *,
    vertex_dtype: np.dtype,
    ndim: int,
) -> list[dict[str, Any]]:
    """Walk object manifests to attribute area / volume / counts per object.

    Faces in ``read_chunk_links(chunk)[fragment_idx]`` align with vertices
    in fragment ``fragment_idx`` of the same chunk (per the v0.6 fragment
    layout); a per-chunk link-group cache avoids re-decoding when many
    objects share a chunk.
    """
    manifests = read_all_object_manifests(level_group)
    if not manifests:
        return []

    referenced_chunks: set[ChunkCoords] = {
        chunk for manifest in manifests for chunk, _ in manifest
    }
    referenced_chunk_strs = [
        ".".join(str(c) for c in cc) for cc in referenced_chunks
    ]

    chunk_link_cache: dict[ChunkCoords, list[np.ndarray]] = {}
    per_object: list[dict[str, Any]] = []

    with level_group.batched_reads([
        (VERTICES, referenced_chunk_strs),
        (VERTEX_FRAGMENTS, referenced_chunk_strs),
        (f"{LINKS}/0", referenced_chunk_strs),
        (LINK_FRAGMENTS, referenced_chunk_strs),
    ]):
        for oid, manifest in enumerate(manifests):
            area = 0.0
            volume = 0.0
            face_count = 0
            vertex_count = 0

            for chunk, fragment_idx in manifest:
                verts = read_fragment(
                    level_group, chunk, fragment_idx, dtype=vertex_dtype, ndim=ndim,
                ).astype(np.float64, copy=False)
                vertex_count += len(verts)

                if chunk not in chunk_link_cache:
                    try:
                        chunk_link_cache[chunk] = read_chunk_links(level_group, chunk)
                    except Exception:
                        chunk_link_cache[chunk] = []
                groups = chunk_link_cache[chunk]

                if fragment_idx >= len(groups):
                    continue
                faces = groups[fragment_idx]
                if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
                    continue

                v0 = verts[faces[:, 0]]
                v1 = verts[faces[:, 1]]
                v2 = verts[faces[:, 2]]
                cr = np.cross(v1 - v0, v2 - v0)
                area += float(np.linalg.norm(cr, axis=1).sum() * 0.5)
                volume += float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)
                face_count += len(faces)

            per_object.append({
                "object_id": oid,
                "surface_area": area,
                "volume": volume,
                "face_count": face_count,
                "vertex_count": vertex_count,
            })

    return per_object
