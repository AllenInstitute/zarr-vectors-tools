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


def export_swc(
    store_path: str | Path,
    output_path: str | Path,
    *,
    level: int = 0,
    chunks: list[ChunkCoords] | None = None,
) -> dict[str, Any]:
    """Export a zarr vectors skeleton to an SWC file.

    Reconstructs the parent array from the edge list and writes
    the 7-column SWC format.

    Args:
        store_path: Path to the zarr vectors store.
        output_path: Path for the output .swc file.
        level: Resolution level to export.
        chunks: Optional whitelist of chunk coordinate tuples; only
            nodes (and intra-chunk edges) in those chunks are exported.
            Edges spanning a listed and an unlisted chunk are dropped,
            which can orphan child nodes — each surviving connected piece
            is still written as a valid SWC tree (with its own root).

    Returns:
        Summary dict with ``node_count``, ``root_count`` and
        ``attributes_carried`` (the per-vertex columns that came from the
        store rather than from the defaults).
    """
    try:
        result = read_graph(str(store_path), level=level, chunks=chunks)
    except Exception as e:
        raise ExportError(f"Failed to read store: {e}") from e

    positions = result["positions"]
    edges = result["edges"]
    n_nodes = len(positions)

    if n_nodes == 0:
        raise ExportError("No nodes to export")

    # Rows are ``[child, parent]`` -- the orientation the skeleton
    # convention stores and ``read_graph`` preserves.  Reading it as
    # "smaller index is the parent" instead inverted every edge whose child
    # sits in an earlier chunk than its parent, because level order is
    # chunk-major and a branch may re-enter a chunk it already left: the
    # exported tree then had extra roots and reversed neurites.
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

    # Radius and compartment come from the store when it has them --
    # ``ingest_swc`` writes both, and defaulting to 1.0/dendrite threw away
    # the morphometry that makes an SWC worth exporting.
    stored_radius = _level_order_attribute(
        store_path, level, _RADIUS_ATTR, n_nodes, chunks,
    )
    radius = (
        np.asarray(stored_radius, dtype=np.float32)
        if stored_radius is not None
        else np.full(n_nodes, _DEFAULT_RADIUS, dtype=np.float32)
    )
    stored_compartment = _level_order_attribute(
        store_path, level, _COMPARTMENT_ATTR, n_nodes, chunks,
    )
    if stored_compartment is not None:
        compartment = np.rint(stored_compartment).astype(np.int32)
    else:
        compartment = np.full(n_nodes, _DEFAULT_COMPARTMENT, dtype=np.int32)
        # Only invent a soma when the store did not say what the nodes are.
        roots = np.flatnonzero(parents < 0)
        if len(roots):
            compartment[roots[0]] = 1

    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write("# SWC exported by zarr-vectors\n")
            for i in range(n_nodes):
                # SWC: ID type X Y Z radius parent_ID (1-indexed, -1 for root)
                swc_id = i + 1
                swc_parent = int(parents[i]) + 1 if parents[i] >= 0 else -1
                x, y, z = positions[i, 0], positions[i, 1], positions[i, 2]
                f.write(
                    f"{swc_id} {compartment[i]} "
                    f"{x:.6f} {y:.6f} {z:.6f} "
                    f"{radius[i]:.4f} {swc_parent}\n"
                )
    except Exception as e:
        raise ExportError(f"Failed to write SWC '{output_path}': {e}") from e

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
