"""Streaming surface area / volume / Euler characteristic for mesh stores.

Loads each chunk's intra-chunk faces, accumulates per-face area and
signed-tetrahedron volume, and tallies edge incidence for the Euler
characteristic. Cross-chunk *edges* are read once from the global
cross-chunk array; cross-chunk *faces* lose identity in the current
core storage so their per-face contributions are excluded — the
returned dict reports the excluded edge count so callers can quantify
the gap.

Works on triangle meshes today. Quad / polygon support requires
fan-triangulation; not implemented in v0.
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
from zarr_vectors_tools.algorithms._cross_chunk_index import build_cross_chunk_index


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
        per_object: Reserved for a future API extension. Currently raises
            ``NotImplementedError`` — see the module docstring for why.

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
            edges that were stored as face-boundary pairs; documents
            the v0 limitation.

    Raises:
        FileNotFoundError: If the store cannot be opened.
        NotImplementedError: If ``per_object=True``.
    """
    if per_object:
        raise NotImplementedError(
            "per_object summaries require core support for face-to-object "
            "mapping and `object_attributes` on write_mesh. Tracked as "
            "catalog Add 6."
        )

    root = open_store(str(store_path))
    root_meta = read_root_metadata(root)
    level_group = get_resolution_level(root, level)
    ndim = root_meta.sid_ndim

    vmeta = level_group.read_array_meta("vertices")
    vertex_dtype = np.dtype(vmeta.get("dtype", "float32"))

    try:
        lmeta = level_group.read_array_meta("links")
        link_width = int(lmeta.get("link_width", 3))
    except Exception:
        link_width = 3

    if link_width != 3:
        raise NotImplementedError(
            f"compute_mesh_summary v0 supports triangle meshes only "
            f"(link_width=3); store has link_width={link_width}."
        )

    chunk_keys = list_chunk_keys(level_group)

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
            link_groups = read_chunk_links(
                level_group, chunk_key, link_width=link_width,
            )
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

    cross_links, _ = build_cross_chunk_index(level_group)
    excluded_cross_face_edges = 0
    for (chunk_a, vi_a), (chunk_b, vi_b) in cross_links:
        edge_set.add(
            _edge_key((chunk_a, int(vi_a)), (chunk_b, int(vi_b)))
        )
        excluded_cross_face_edges += 1

    edge_count = len(edge_set)
    euler = vertex_count - edge_count + face_count

    return {
        "surface_area": surface_area,
        "volume": volume,
        "face_count": face_count,
        "vertex_count": vertex_count,
        "edge_count": edge_count,
        "euler_characteristic": euler,
        "excluded_cross_face_edges": excluded_cross_face_edges,
    }
