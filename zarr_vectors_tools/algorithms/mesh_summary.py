"""Streaming surface area / volume / Euler characteristic for mesh stores.

Streams the level chunk by chunk.  Each chunk's own faces contribute their
area, signed-tetrahedron volume and edges as the chunk is read.  Faces whose
corners lie in more than one chunk are read once, up front, from the link
family's cross-chunk cells; their corner positions are collected as the
chunks stream past and they are added at the end (see
:class:`~zarr_vectors_tools.algorithms._mesh_faces.SpanningFaces`).  The
result does not depend on the chunk shape.

``per_object=True`` walks the object manifests instead of the chunk grid:
every face is credited to the object owning its first corner.

Faces of four or more corners are split into a fan of triangles for area and
volume, and counted once as faces, with their boundary edges, for the Euler
characteristic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from zarr_vectors.building import (
    get_resolution_level,
    list_chunk_keys,
    open_store,
    read_all_object_manifests,
    read_chunk_vertices,
    read_root_metadata,
    read_vertex_fragment_index,
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
    polygon_edges,
    triangle_area_volume,
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
            face / vertex stats restricted to each object.

    Returns:
        Dict with keys:
          - ``surface_area`` (float): sum of face areas.
          - ``volume`` (float): signed-tetrahedron sum; meaningful only
            for closed meshes with consistent winding.
          - ``face_count`` (int): stored faces; a quad counts once.
          - ``vertex_count`` (int): vertices at the level.
          - ``edge_count`` (int): distinct face-boundary edges.
          - ``euler_characteristic`` (int): ``V - E + F``.
          - ``excluded_cross_face_edges`` (int): always 0, since every face
            is counted; kept for callers that read it.
          - ``per_object`` (list[dict], only when ``per_object=True``):
            one dict per object_id with ``object_id``, ``surface_area``,
            ``volume``, ``face_count``, ``vertex_count``.

    Raises:
        FileNotFoundError: If the store cannot be opened.
        NotImplementedError: If the level holds no faces of three or more
            corners (a graph or skeleton level).
        ValueError: If a face spanning chunks names a vertex the level does
            not hold.
    """
    root = open_store(str(store_path))
    root_meta = read_root_metadata(root)
    level_group = get_resolution_level(root, level)
    ndim = root_meta.sid_ndim

    vmeta = level_group.read_array_meta("vertices")
    vertex_dtype = np.dtype(vmeta.get("dtype", "float32"))

    # A non-mesh store must fail here rather than silently produce areas
    # from mis-read rows.
    width = face_width(level_group, what="compute_mesh_summary")

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
    # chunk (edges inside it) plus one over the edges between chunks, which
    # only the spanning faces have.  That is what lets the per-chunk half
    # stay as a NumPy array of index pairs, 16 bytes an edge.
    #
    # A spanning face's two corners can share a chunk, and that edge may
    # also belong to one of the chunk's own faces, so those pairs join that
    # chunk's dedup pass.
    edge_count = 0

    with level_group.batched_reads([
        (VERTICES, chunk_key_strs),
        (VERTEX_FRAGMENTS, chunk_key_strs),
        *link_prefetch_plan(level_group, chunk_keys),
    ]):
        spanning = SpanningFaces(level_group)
        same_chunk_from_spanning, spanning_edges = spanning.edges()

        for chunk_key in chunk_keys:
            local_positions = chunk_positions(
                level_group, chunk_key, dtype=vertex_dtype, ndim=ndim,
            )
            if local_positions is None:
                continue
            vertex_count += len(local_positions)
            spanning.collect(chunk_key, local_positions)

            chunk_pairs: list[np.ndarray] = []
            for faces in chunk_faces(level_group, chunk_key, width):
                face_count += len(faces)
                triangles, _ = fan(faces)
                area, vol = triangle_area_volume(
                    local_positions[triangles[:, 0]],
                    local_positions[triangles[:, 1]],
                    local_positions[triangles[:, 2]],
                )
                surface_area += float(area.sum())
                volume += float(vol.sum())
                chunk_pairs.append(polygon_edges(faces))

            from_spanning = same_chunk_from_spanning.pop(tuple(chunk_key), None)
            if from_spanning is not None:
                chunk_pairs.append(from_spanning)
            if chunk_pairs:
                edge_count += _distinct_pairs(chunk_pairs)

    edge_count += spanning_edges

    if len(spanning):
        # Also covers a corner in a chunk the loop never read, whose edges
        # are still waiting in same_chunk_from_spanning.
        spanning.require_complete()
        a, b, c, _slots, _source = spanning.triangles()
        area, vol = triangle_area_volume(a, b, c)
        surface_area += float(area.sum())
        volume += float(vol.sum())
        face_count += len(spanning)

    euler = vertex_count - edge_count + face_count

    result: dict[str, Any] = {
        "surface_area": surface_area,
        "volume": volume,
        "face_count": face_count,
        "vertex_count": vertex_count,
        "edge_count": edge_count,
        "euler_characteristic": euler,
        "excluded_cross_face_edges": 0,
    }

    if per_object:
        result["per_object"] = _compute_per_object(
            level_group, vertex_dtype=vertex_dtype, ndim=ndim, width=width,
        )

    return result


def _distinct_pairs(parts: list[np.ndarray]) -> int:
    """Distinct undirected pairs among ``parts``' ``(K, 2)`` rows."""
    pairs = np.concatenate(parts, axis=0).astype(np.int64)
    # Undirected: sort each row so (a, b) and (b, a) dedup.
    pairs.sort(axis=1)
    return int(len(np.unique(pairs, axis=0)))


def _compute_per_object(
    level_group,
    *,
    vertex_dtype: np.dtype,
    ndim: int,
    width: int,
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

    A face is credited to the object owning its first corner -- which also
    settles a face whose corners span two objects, which a well-formed mesh
    store does not produce, since objects partition faces.  A face whose
    corners lie in different chunks is credited the same way, once every
    chunk has been read.
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

    def credit(oid_per_face, triangles_of, a, b, c) -> None:
        """Add faces' areas and volumes, given as triangles, to their objects."""
        face_area, face_volume = triangle_area_volume(a, b, c)
        oid_per_triangle = oid_per_face[triangles_of]
        area[:] += np.bincount(
            oid_per_triangle, weights=face_area, minlength=n_objects,
        )[:n_objects]
        volume[:] += np.bincount(
            oid_per_triangle, weights=face_volume, minlength=n_objects,
        )[:n_objects]
        faces_per_object[:] += np.bincount(
            oid_per_face, minlength=n_objects,
        )[:n_objects].astype(np.int64)

    with level_group.batched_reads([
        (VERTICES, referenced_chunk_strs),
        (VERTEX_FRAGMENTS, referenced_chunk_strs),
        *link_prefetch_plan(level_group, referenced_chunks),
    ]):
        spanning = SpanningFaces(level_group)
        # The object owning each spanning face's corners, filled in as their
        # chunks are read; -1 for a corner no object owns.
        corner_owner = np.full(len(spanning) * spanning.width, -1, dtype=np.int64)

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

            spanning.collect(chunk, positions)
            corners = spanning.corners_in(chunk)
            if corners is not None:
                slots, corner_rows = corners
                ok = (corner_rows >= 0) & (corner_rows < n_rows)
                corner_owner[slots[ok]] = row_owner[corner_rows[ok]]

            usable = chunk_faces(level_group, chunk, width)
            if not usable:
                continue
            faces = np.concatenate(usable, axis=0)
            inside = np.all((faces >= 0) & (faces < n_rows), axis=1)
            faces = faces[inside]
            if len(faces) == 0:
                continue

            oid_per_face = row_owner[faces[:, 0]]
            keep = oid_per_face >= 0
            if not keep.any():
                continue
            faces = faces[keep]
            triangles, triangles_of = fan(faces)
            credit(
                oid_per_face[keep], triangles_of,
                positions[triangles[:, 0]], positions[triangles[:, 1]],
                positions[triangles[:, 2]],
            )

    if len(spanning):
        oid_per_face = corner_owner.reshape(len(spanning), spanning.width)[:, 0]
        keep = np.flatnonzero((oid_per_face >= 0) & spanning.complete())
        if keep.size:
            a, b, c, _slots, source = spanning.triangles(keep)
            # credit() indexes faces by position in ``keep``.
            position_in_keep = np.searchsorted(keep, source)
            credit(oid_per_face[keep], position_in_keep, a, b, c)

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
