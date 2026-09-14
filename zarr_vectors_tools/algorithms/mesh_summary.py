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
    VERTEX_FRAGMENTS,
    VERTICES,
)
from zarr_vectors.building import (
    get_resolution_level,
    list_chunk_keys,
    open_store,
    read_all_object_manifests,
    read_chunk_links,
    read_chunk_vertices,
    read_root_metadata,
    read_vertex_fragment_index,
)
from zarr_vectors.typing import ChunkCoords

from zarr_vectors_tools.algorithms._links import (
    chunk_key_str,
    link_prefetch_plan,
    read_cross_links,
    require_link_width,
)


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

    # Called for the check, not the value: a non-triangle store must fail
    # here rather than silently produce areas from mis-read rows.
    require_link_width(
        level_group, 3, what="compute_mesh_summary v0 (triangle meshes only)",
    )

    chunk_keys = list_chunk_keys(level_group)
    chunk_key_strs = [chunk_key_str(cc) for cc in chunk_keys]

    surface_area = 0.0
    volume = 0.0
    face_count = 0
    vertex_count = 0

    # Edge counting, without a Python object per edge.
    #
    # An edge is ``((chunk, local), (chunk, local))``.  Two edges can only
    # be the same edge if they name the same chunk pair, so the global
    # dedup the Euler characteristic needs decomposes into one dedup per
    # chunk (edges inside it) plus one over the records that span chunks.
    # That is what lets the per-chunk half stay as a NumPy array of index
    # pairs -- 16 bytes an edge against the ~600 a set of nested tuples
    # cost, which at a hundred million faces is the difference between a
    # few gigabytes and not finishing.
    #
    # The spanning records are read FIRST so that a cross-chunk face's
    # two corners that happen to share a chunk join that chunk's own dedup
    # pass rather than needing a second global set.
    edge_count = 0
    spanning_edges: set[tuple[tuple, tuple]] = set()
    same_chunk_from_records: dict[ChunkCoords, list[tuple[int, int]]] = {}
    excluded_cross_face_edges = 0

    def _edge_key(a: tuple, b: tuple) -> tuple[tuple, tuple]:
        return (a, b) if a <= b else (b, a)

    with level_group.batched_reads([
        (VERTICES, chunk_key_strs),
        (VERTEX_FRAGMENTS, chunk_key_strs),
        *link_prefetch_plan(level_group, chunk_keys),
    ]):
        # Cross-only: the per-chunk loop below consumes every intra record
        # via read_chunk_links, so the whole-family read_links would
        # double-count them here.
        try:
            cross_links = read_cross_links(level_group, delta=0)
        except Exception:
            cross_links = []
        # Records may have 2 endpoints (a cross-chunk edge) or 3+ (a
        # cross-chunk face).  Each contributes (len - 1) consecutive edges.
        for record in cross_links:
            eps = [(tuple(chunk), int(vi)) for chunk, vi in record]
            if len(eps) < 2:
                continue
            for k in range(len(eps) - 1):
                a, b = eps[k], eps[k + 1]
                excluded_cross_face_edges += 1
                if a[0] == b[0]:
                    same_chunk_from_records.setdefault(a[0], []).append(
                        (a[1], b[1]),
                    )
                else:
                    spanning_edges.add(_edge_key(a, b))

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

            chunk_pairs: list[np.ndarray] = []
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

                chunk_pairs.append(np.concatenate([
                    faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]],
                ], axis=0))

            from_records = same_chunk_from_records.pop(tuple(chunk_key), None)
            if from_records:
                chunk_pairs.append(np.asarray(from_records, dtype=np.int64))
            if chunk_pairs:
                pairs = np.concatenate(chunk_pairs, axis=0).astype(np.int64)
                # Undirected: sort each row so (a, b) and (b, a) dedup.
                pairs.sort(axis=1)
                edge_count += int(len(np.unique(pairs, axis=0)))

    # Any chunk that held only record-derived edges (no intra faces of its
    # own) never reached the loop above.
    for leftover in same_chunk_from_records.values():
        pairs = np.asarray(leftover, dtype=np.int64)
        pairs.sort(axis=1)
        edge_count += int(len(np.unique(pairs, axis=0)))

    edge_count += len(spanning_edges)
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
    """Attribute area / volume / counts to objects, from the fragment index.

    A face's stored corner indices are **chunk-local**: they address the
    chunk's whole vertex buffer, not one fragment's slice of it.  Which
    object a face belongs to is therefore a property of the ROWS it names,
    recovered by asking the chunk's vertex-fragment index which fragment owns
    each row and the object manifests which object owns each fragment.

    Two earlier readings of this were wrong in ways that cancelled on a
    single-object store and only on one:

    * indexing ``read_chunk_links(chunk)[fragment_idx]`` into fragment
      ``fragment_idx``'s own vertices (IndexError once a chunk held two
      objects), and
    * assuming that link group aligns with the vertex fragment of the same
      index at all -- it does not, so every face in a chunk was credited to
      whichever object owned fragment 0 and every other object reported zero.

    A face whose corners span two objects (which a well-formed mesh store
    does not produce, since objects partition faces) is credited to the
    object owning its first corner.
    """
    manifests = read_all_object_manifests(level_group)
    if not manifests:
        return []
    n_objects = len(manifests)

    # fragment -> owning object.  First namer wins, matching how the
    # coarseners attribute a fragment that several objects reference.
    owner: dict[tuple[ChunkCoords, int], int] = {}
    for oid, manifest in enumerate(manifests):
        for chunk, fragment_idx in manifest:
            owner.setdefault(
                (tuple(int(c) for c in chunk), int(fragment_idx)), oid,
            )

    referenced_chunks = sorted({chunk for chunk, _ in owner})
    referenced_chunk_strs = [chunk_key_str(cc) for cc in referenced_chunks]

    area = np.zeros(n_objects, dtype=np.float64)
    volume = np.zeros(n_objects, dtype=np.float64)
    faces_per_object = np.zeros(n_objects, dtype=np.int64)
    verts_per_object = np.zeros(n_objects, dtype=np.int64)

    with level_group.batched_reads([
        (VERTICES, referenced_chunk_strs),
        (VERTEX_FRAGMENTS, referenced_chunk_strs),
        *link_prefetch_plan(level_group, referenced_chunks),
    ]):
        for chunk in referenced_chunks:
            try:
                fragment_index = read_vertex_fragment_index(level_group, chunk)
                groups = read_chunk_vertices(
                    level_group, chunk, dtype=vertex_dtype, ndim=ndim,
                )
            except Exception:
                continue
            if not groups:
                continue

            rows = [
                _fragment_rows(fragment_index, f)
                for f in range(fragment_index.num_fragments)
            ]
            n_rows = max(
                (int(r.max()) + 1 for r in rows if r.size), default=0,
            )
            if n_rows == 0:
                continue

            positions = np.zeros((n_rows, ndim), dtype=np.float64)
            row_owner = np.full(n_rows, -1, dtype=np.int64)
            for f, idx in enumerate(rows):
                if idx.size == 0:
                    continue
                if f < len(groups):
                    block = np.asarray(groups[f], dtype=np.float64)
                    if block.shape[0] == idx.size:
                        positions[idx] = block
                oid = owner.get((chunk, f), -1)
                if oid >= 0:
                    row_owner[idx] = oid
                    verts_per_object[oid] += idx.size

            try:
                link_groups = read_chunk_links(level_group, chunk)
            except Exception:
                continue
            usable = [
                np.asarray(g, dtype=np.int64) for g in link_groups
                if np.asarray(g).ndim == 2 and np.asarray(g).shape[1] == 3
                and len(g)
            ]
            if not usable:
                continue
            faces = np.concatenate(usable, axis=0)
            # Cross-chunk faces name rows this chunk does not hold; they have
            # no object attribution in the format and are excluded here, as
            # they are from the chunk-level totals.
            inside = np.all((faces >= 0) & (faces < n_rows), axis=1)
            faces = faces[inside]
            if len(faces) == 0:
                continue

            oid_per_face = row_owner[faces[:, 0]]
            keep = oid_per_face >= 0
            if not keep.any():
                continue
            faces = faces[keep]
            oid_per_face = oid_per_face[keep]

            v0 = positions[faces[:, 0]]
            v1 = positions[faces[:, 1]]
            v2 = positions[faces[:, 2]]
            cross = np.cross(v1 - v0, v2 - v0)
            face_area = np.linalg.norm(cross, axis=1) * 0.5
            face_volume = np.einsum("ij,ij->i", v0, np.cross(v1, v2)) / 6.0

            area += np.bincount(
                oid_per_face, weights=face_area, minlength=n_objects,
            )[:n_objects]
            volume += np.bincount(
                oid_per_face, weights=face_volume, minlength=n_objects,
            )[:n_objects]
            faces_per_object += np.bincount(
                oid_per_face, minlength=n_objects,
            )[:n_objects].astype(np.int64)

    return [
        {
            "object_id": oid,
            "surface_area": float(area[oid]),
            "volume": float(volume[oid]),
            "face_count": int(faces_per_object[oid]),
            "vertex_count": int(verts_per_object[oid]),
        }
        for oid in range(n_objects)
    ]


def _fragment_rows(fragment_index, f: int) -> np.ndarray:
    """The chunk-buffer rows fragment ``f`` occupies, range or explicit."""
    if fragment_index.is_range(f):
        start, count = fragment_index.range(f)
        return np.arange(int(start), int(start) + int(count), dtype=np.int64)
    return np.asarray(fragment_index.indices(f), dtype=np.int64)
