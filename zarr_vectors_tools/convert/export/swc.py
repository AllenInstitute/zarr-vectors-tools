"""Export zarr vectors skeletons to SWC format."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from zarr_vectors.building import (
    chunk_local_to_global_offsets,
    get_resolution_level,
    open_store,
    read_chunk_attributes,
)
from zarr_vectors.exceptions import ExportError
from zarr_vectors.types.graphs import read_graph
from zarr_vectors.typing import ChunkCoords

#: ``vertex_attributes`` names the SWC columns are read from, and the value
#: written when a store does not carry them.  ``ingest_swc`` writes both.
_RADIUS_ATTR = "radius"
_COMPARTMENT_ATTR = "compartment"
_DEFAULT_RADIUS = 1.0
_DEFAULT_COMPARTMENT = 3  # 3 = (basal) dendrite, the SWC catch-all
#: The per-object attribute an EM skeleton store keeps its segment ids in.
_SEGMENT_ID_ATTR = "segment_id"


def _level_order_attribute(
    store_path: str | Path,
    level: int,
    name: str,
    n_nodes: int,
    chunks: list[ChunkCoords] | None,
) -> np.ndarray | None:
    """Read one per-vertex attribute in the order ``read_graph`` returns nodes.

    ``read_graph`` hands back positions in level order -- chunks in
    ``chunk_local_to_global_offsets`` order, fragments concatenated within
    each -- and per-chunk attribute cells are stored in exactly that order, so
    walking the same chunk list reproduces the alignment.  Returns ``None``
    when the attribute is absent or does not line up, which is what makes it
    safe to call unconditionally.
    """
    try:
        level_group = get_resolution_level(open_store(str(store_path)), level)
        _offsets, chunk_keys, _total = chunk_local_to_global_offsets(level_group)
        wanted = None if chunks is None else {tuple(int(c) for c in cc) for cc in chunks}
        parts: list[np.ndarray] = []
        for cc in chunk_keys:
            if wanted is not None and tuple(int(c) for c in cc) not in wanted:
                continue
            for group in read_chunk_attributes(level_group, name, cc):
                parts.append(np.asarray(group).ravel())
        if not parts:
            return None
        values = np.concatenate(parts)
        return values if len(values) == n_nodes else None
    except Exception:
        return None


def _parents(edges: np.ndarray, n_nodes: int) -> np.ndarray:
    """The SWC parent of each node, ``-1`` for a root.

    Rows are ``[child, parent]`` -- the orientation the skeleton convention
    stores and ``read_graph`` preserves.  Reading it as "smaller index is the
    parent" instead inverted every edge whose child sits in an earlier chunk
    than its parent, because level order is chunk-major and a branch may
    re-enter a chunk it already left: the exported tree then had extra roots
    and reversed neurites.
    """
    parents = np.full(n_nodes, -1, dtype=np.int64)
    if len(edges):
        e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        valid = (
            (e[:, 0] >= 0) & (e[:, 0] < n_nodes)
            & (e[:, 1] >= 0) & (e[:, 1] < n_nodes)
            & (e[:, 0] != e[:, 1])
        )
        e = e[valid]
        # First writer wins, so a node that appears as a child twice (not a
        # tree, but ingestible) keeps one parent instead of raising.
        unseen = parents[e[:, 0]] < 0
        parents[e[unseen, 0]] = e[unseen, 1]
    return parents


def _columns(
    parents: np.ndarray,
    stored_radius: np.ndarray | None,
    stored_compartment: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """The radius and type columns, from the store where it has them.

    ``ingest_swc`` writes both, and defaulting to 1.0/dendrite threw away
    the morphometry that makes an SWC worth exporting.
    """
    n_nodes = len(parents)
    radius = (
        np.asarray(stored_radius, dtype=np.float32).reshape(-1)
        if stored_radius is not None
        else np.full(n_nodes, _DEFAULT_RADIUS, dtype=np.float32)
    )
    if stored_compartment is not None:
        compartment = np.rint(np.asarray(stored_compartment).reshape(-1)).astype(np.int32)
    else:
        compartment = np.full(n_nodes, _DEFAULT_COMPARTMENT, dtype=np.int32)
        # Only invent a soma when the store did not say what the nodes are.
        roots = np.flatnonzero(parents < 0)
        if len(roots):
            compartment[roots[0]] = 1
    return radius, compartment


def _write_swc(
    output_path: Path,
    positions: np.ndarray,
    parents: np.ndarray,
    radius: np.ndarray,
    compartment: np.ndarray,
) -> None:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("# SWC exported by zarr-vectors\n")
            for i in range(len(positions)):
                # SWC: ID type X Y Z radius parent_ID (1-indexed, -1 for root)
                swc_parent = int(parents[i]) + 1 if parents[i] >= 0 else -1
                x, y, z = positions[i, 0], positions[i, 1], positions[i, 2]
                f.write(
                    f"{i + 1} {compartment[i]} "
                    f"{x:.6f} {y:.6f} {z:.6f} "
                    f"{radius[i]:.4f} {swc_parent}\n"
                )
    except Exception as e:
        raise ExportError(f"Failed to write SWC '{output_path}': {e}") from e


def export_swc(
    store_path: str | Path,
    output_path: str | Path,
    *,
    level: int = 0,
    chunks: list[ChunkCoords] | None = None,
    object_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Export a zarr vectors skeleton to SWC.

    Reconstructs the parent array from the edge list and writes the
    7-column SWC format.

    Without ``object_ids`` the whole level is one file.  With them,
    ``output_path`` is a directory and each object gets its own
    single-tree file, named after its segment id when the store has one
    (an EM skeleton store does) and its object id otherwise.

    Args:
        store_path: Path to the zarr vectors store.
        output_path: The .swc file to write, or with ``object_ids`` the
            directory to write one file per object into.
        level: Resolution level to export.
        chunks: Optional whitelist of chunk coordinate tuples; only
            nodes (and intra-chunk edges) in those chunks are exported.
            Edges spanning a listed and an unlisted chunk are dropped,
            which can orphan child nodes — each surviving connected piece
            is still written as a valid SWC tree (with its own root).
            Whole-level export only.
        object_ids: Objects to write, one file each.

    Returns:
        Summary dict with ``node_count``, ``root_count`` and
        ``attributes_carried`` (the per-vertex columns that came from the
        store rather than from the defaults).  With ``object_ids`` also
        ``object_count`` and ``files``, the paths written in the order asked.
    """
    if object_ids is not None:
        if chunks is not None:
            raise ExportError(
                "chunks= crops a level spatially and object_ids= writes whole "
                "objects; pass one or the other"
            )
        return _export_objects(store_path, Path(output_path), level, list(object_ids))

    try:
        result = read_graph(str(store_path), level=level, chunks=chunks)
    except Exception as e:
        raise ExportError(f"Failed to read store: {e}") from e

    positions = result["positions"]
    n_nodes = len(positions)
    if n_nodes == 0:
        raise ExportError("No nodes to export")

    parents = _parents(result["edges"], n_nodes)
    stored_radius = _level_order_attribute(
        store_path, level, _RADIUS_ATTR, n_nodes, chunks,
    )
    stored_compartment = _level_order_attribute(
        store_path, level, _COMPARTMENT_ATTR, n_nodes, chunks,
    )
    radius, compartment = _columns(parents, stored_radius, stored_compartment)
    _write_swc(Path(output_path), positions, parents, radius, compartment)

    return {
        "node_count": n_nodes,
        "root_count": int(np.count_nonzero(parents < 0)),
        "attributes_carried": [
            name for name, value in (
                (_RADIUS_ATTR, stored_radius),
                (_COMPARTMENT_ATTR, stored_compartment),
            ) if value is not None
        ],
    }


def _export_objects(
    store_path: str | Path, directory: Path, level: int, object_ids: list[int],
) -> dict[str, Any]:
    """One SWC per object.

    An EM skeleton store (one written by the precomputed ingesters) keeps a
    sorted ``segment_id`` per object, and core reads one skeleton by that id
    touching only the chunks the object lives in -- which matters when the
    store holds a whole dataset and the request is five neurons.  A store
    without it is read once and split by the object manifests.
    """
    from zarr_vectors.building import read_object_attributes

    if directory.suffix.lower() == ".swc":
        raise ExportError(
            f"with object_ids, {directory} names a directory to write one "
            f".swc per object into, not a file"
        )
    level_group = get_resolution_level(open_store(str(store_path)), level)
    try:
        segment_ids = np.asarray(
            read_object_attributes(level_group, _SEGMENT_ID_ATTR), dtype=np.uint64,
        ).reshape(-1)
    except Exception:  # noqa: BLE001 - not an EM store
        segment_ids = None

    if segment_ids is not None:
        pieces = _objects_by_segment_id(store_path, level, object_ids, segment_ids)
    else:
        pieces = _objects_by_manifest(store_path, level_group, level, object_ids)

    files: list[str] = []
    node_count = root_count = 0
    carried: set[str] = set()
    for oid in object_ids:
        name, positions, edges, attributes = pieces[oid]
        parents = _parents(edges, len(positions))
        radius, compartment = _columns(
            parents, attributes.get(_RADIUS_ATTR), attributes.get(_COMPARTMENT_ATTR),
        )
        carried.update(a for a in (_RADIUS_ATTR, _COMPARTMENT_ATTR) if a in attributes)
        path = directory / f"{name}.swc"
        _write_swc(path, positions, parents, radius, compartment)
        files.append(str(path))
        node_count += len(positions)
        root_count += int(np.count_nonzero(parents < 0))
    return {
        "object_count": len(object_ids),
        "node_count": node_count,
        "root_count": root_count,
        "attributes_carried": sorted(carried),
        "files": files,
    }


def _objects_by_segment_id(
    store_path: str | Path,
    level: int,
    object_ids: list[int],
    segment_ids: np.ndarray,
) -> dict[int, tuple[str, np.ndarray, np.ndarray, dict[str, np.ndarray]]]:
    from zarr_vectors.types import skeletons as sk

    from zarr_vectors_tools.algorithms._links import cross_offset_segments
    from zarr_vectors_tools.convert.export.precomputed import _crossing_edges

    level_group = get_resolution_level(open_store(str(store_path)), level)
    cross_offsets = [offsets for _seg, offsets in cross_offset_segments(level_group)]

    missing = [o for o in object_ids if not 0 <= int(o) < len(segment_ids)]
    if missing:
        raise ExportError(
            f"object id(s) {missing} are not at level {level}, which holds "
            f"{len(segment_ids)} objects"
        )
    out = {}
    for oid in object_ids:
        segment = int(segment_ids[int(oid)])
        result = sk.read_skeleton_by_segment_id(str(store_path), segment, level=level)
        if result is None or len(result["positions"]) == 0:
            raise ExportError(
                f"object {oid} (segment {segment}) has no geometry at level "
                f"{level}; a coarser level's sparsity may have dropped it"
            )
        attributes = {
            name: np.asarray(values)
            for name, values in (result.get("attributes") or {}).items()
            if len(values) == len(result["positions"])
        }
        # Core's reader joins a segment's fragments within a chunk but not
        # across a chunk boundary, so a neuron spanning two chunks read back
        # as two trees and exported with two roots.  The crossing links are
        # read back and folded in as parent links.
        crossing = (
            _crossing_edges(level_group, int(oid), cross_offsets)
            if cross_offsets else np.zeros((0, 2), dtype=np.int64)
        )
        edges = _join_across_chunks(
            np.asarray(result["edges"], dtype=np.int64).reshape(-1, 2),
            crossing, len(result["positions"]),
        )
        out[oid] = (str(segment), result["positions"], edges, attributes)
    return out


def _join_across_chunks(
    edges: np.ndarray, crossing: np.ndarray, n_nodes: int,
) -> np.ndarray:
    """``[child, parent]`` edges with the undirected ``crossing`` links folded in.

    A crossing link records two vertices, not which is the parent.  Each one
    that joins two separate pieces makes one piece hang from the other: from
    whichever end is already a piece's root, and otherwise by turning the
    piece around so that end becomes its root.  The result is one tree per
    connected piece, with the first piece's root kept.
    """
    parents = _parents(edges, n_nodes)
    for u, v in np.asarray(crossing, dtype=np.int64).reshape(-1, 2).tolist():
        root_u, root_v = _root(parents, u), _root(parents, v)
        if root_u == root_v:
            continue
        # Keep the lower-numbered root -- the first piece read -- as the top.
        child, parent = (v, u) if root_u < root_v else (u, v)
        _reroot(parents, child)
        parents[child] = parent
    rows = np.flatnonzero(parents >= 0)
    return np.column_stack([rows, parents[rows]]).astype(np.int64)


def _root(parents: np.ndarray, node: int) -> int:
    seen = 0
    while parents[node] >= 0 and seen <= len(parents):
        node = int(parents[node])
        seen += 1
    return node


def _reroot(parents: np.ndarray, node: int) -> None:
    """Make ``node`` its tree's root by reversing the path above it."""
    previous, current = -1, node
    while current >= 0:
        above = int(parents[current])
        parents[current] = previous
        previous, current = current, above


def _objects_by_manifest(
    store_path: str | Path,
    level_group: Any,
    level: int,
    object_ids: list[int],
) -> dict[int, tuple[str, np.ndarray, np.ndarray, dict[str, np.ndarray]]]:
    """Split the whole level into objects.

    ``read_graph`` returns nodes in level order: chunks in
    ``chunk_local_to_global_offsets`` order, each chunk's fragments
    concatenated.  The object manifests say which object each
    ``(chunk, fragment)`` belongs to, and the chunk's fragment index gives
    the fragment's node range, which together label every node.
    """
    from zarr_vectors.building import read_all_object_manifests, read_vertex_fragment_index

    try:
        result = read_graph(str(store_path), level=level)
    except Exception as e:
        raise ExportError(f"Failed to read store: {e}") from e
    positions = result["positions"]
    edges = np.asarray(result["edges"], dtype=np.int64).reshape(-1, 2)
    n_nodes = len(positions)

    manifests = read_all_object_manifests(level_group)
    missing = [o for o in object_ids if not 0 <= int(o) < len(manifests)]
    if missing:
        raise ExportError(
            f"object id(s) {missing} are not at level {level}, which holds "
            f"{len(manifests)} objects"
        )
    offsets, _keys, _total = chunk_local_to_global_offsets(level_group)
    owner = np.full(n_nodes, -1, dtype=np.int64)
    index_cache: dict[tuple[int, ...], Any] = {}
    for oid in object_ids:
        for cc, fragment in manifests[int(oid)]:
            key = tuple(int(c) for c in cc)
            if key not in index_cache:
                index_cache[key] = read_vertex_fragment_index(level_group, key)
            start, count = index_cache[key].range(int(fragment))
            base = int(offsets[key]) + int(start)
            owner[base:base + int(count)] = int(oid)

    attributes_all = {
        name: values for name in (_RADIUS_ATTR, _COMPARTMENT_ATTR)
        if (values := _level_order_attribute(store_path, level, name, n_nodes, None))
        is not None
    }

    out = {}
    for oid in object_ids:
        nodes = np.flatnonzero(owner == int(oid))
        if len(nodes) == 0:
            raise ExportError(f"object {oid} has no geometry at level {level}")
        local = np.full(n_nodes, -1, dtype=np.int64)
        local[nodes] = np.arange(len(nodes))
        inside = (local[edges[:, 0]] >= 0) & (local[edges[:, 1]] >= 0)
        out[oid] = (
            str(int(oid)),
            positions[nodes],
            local[edges[inside]],
            {name: values[nodes] for name, values in attributes_all.items()},
        )
    return out
