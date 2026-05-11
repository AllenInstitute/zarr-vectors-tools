"""Ingest neuronal morphology from SWC files into zarr vectors.

SWC is a 7-column text format: ID type X Y Z radius parent_ID.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from zarr_vectors.exceptions import IngestError
from zarr_vectors.types.graphs import write_graph
from zarr_vectors.typing import ChunkShape


def ingest_swc(
    input_path: str | Path,
    output_path: str | Path,
    chunk_shape: ChunkShape,
    *,
    dtype: str = "float32",
    preserve_header: bool = True,
) -> dict[str, Any]:
    """Ingest an SWC file into a zarr vectors skeleton store.

    Args:
        input_path: Path to the input .swc file.
        output_path: Path for the output zarr vectors store.
        chunk_shape: Spatial chunk size per dimension (3D).
        dtype: Dtype for position data.
        preserve_header: If True, store SWC comment lines in
            ``/headers/swc/`` for round-trip export.

    Returns:
        Summary dict from :func:`write_graph`.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise IngestError(f"Input file not found: {input_path}")

    try:
        rows: list[list[float]] = []
        comment_lines: list[str] = []
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    comment_lines.append(line)
                    continue
                parts = line.split()
                if len(parts) < 7:
                    continue
                rows.append([float(p) for p in parts[:7]])

        if not rows:
            raise IngestError(f"SWC file has no data rows: {input_path}")

        data = np.array(rows, dtype=np.float64)

    except IngestError:
        raise
    except Exception as e:
        raise IngestError(f"Failed to parse SWC '{input_path}': {e}") from e

    np_dtype = np.dtype(dtype)

    # Columns: ID(0) type(1) X(2) Y(3) Z(4) radius(5) parent_ID(6)
    swc_ids = data[:, 0].astype(np.int64)
    compartment = data[:, 1].astype(np.float32)
    positions = data[:, 2:5].astype(np_dtype)
    radius = data[:, 5].astype(np.float32)
    parent_ids = data[:, 6].astype(np.int64)

    # SWC IDs may not be contiguous 0-based — build remapping
    id_to_idx = {int(sid): i for i, sid in enumerate(swc_ids)}
    n_nodes = len(swc_ids)

    # Build edge list: [child_idx, parent_idx]
    edges: list[list[int]] = []
    for i in range(n_nodes):
        pid = int(parent_ids[i])
        if pid == -1 or pid not in id_to_idx:
            continue  # root or disconnected
        edges.append([i, id_to_idx[pid]])

    edges_arr = np.array(edges, dtype=np.int64) if edges else np.zeros((0, 2), dtype=np.int64)

    node_attributes = {
        "radius": radius,
        "compartment": compartment,
    }

    result = write_graph(
        str(output_path),
        positions,
        edges_arr,
        chunk_shape=chunk_shape,
        is_tree=True,
        node_attributes=node_attributes,
        dtype=dtype,
    )

    if preserve_header:
        try:
            from zarr_vectors_tools.headers.registry import HeaderRegistry
            from zarr_vectors_tools.headers.formats import SWCHeader

            swc_header = SWCHeader(
                comment_lines=comment_lines,
            )
            reg = HeaderRegistry(str(output_path))
            reg.add("swc", swc_header)
        except Exception:
            pass

    return result
