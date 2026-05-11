"""Export zarr vectors skeletons to SWC format."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from zarr_vectors.exceptions import ExportError
from zarr_vectors.types.graphs import read_graph


def export_swc(
    store_path: str | Path,
    output_path: str | Path,
    *,
    level: int = 0,
) -> dict[str, Any]:
    """Export a zarr vectors skeleton to an SWC file.

    Reconstructs the parent array from the edge list and writes
    the 7-column SWC format.

    Args:
        store_path: Path to the zarr vectors store.
        output_path: Path for the output .swc file.
        level: Resolution level to export.

    Returns:
        Summary dict with ``node_count``.
    """
    try:
        result = read_graph(str(store_path), level=level)
    except Exception as e:
        raise ExportError(f"Failed to read store: {e}") from e

    positions = result["positions"]
    edges = result["edges"]
    n_nodes = len(positions)

    if n_nodes == 0:
        raise ExportError("No nodes to export")

    # Reconstruct parent array from edges
    # Edges are [child, parent] or [src, dst] — we need to determine
    # which is the parent direction.  Convention: for each edge,
    # the node with the smaller index is the parent (DFS ordering).
    parents = np.full(n_nodes, -1, dtype=np.int64)
    for e in edges:
        a, b = int(e[0]), int(e[1])
        # In DFS-ordered trees: parent has smaller index
        child, parent = (a, b) if a > b else (b, a)
        if parents[child] == -1:
            parents[child] = parent

    # Default compartment type (3=dendrite) and radius (1.0)
    compartment = np.full(n_nodes, 3, dtype=np.int32)
    radius = np.full(n_nodes, 1.0, dtype=np.float32)

    # Root: try to find node with parent == -1
    roots = np.where(parents == -1)[0]
    if len(roots) > 0:
        compartment[roots[0]] = 1  # soma

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

    return {"node_count": n_nodes}
