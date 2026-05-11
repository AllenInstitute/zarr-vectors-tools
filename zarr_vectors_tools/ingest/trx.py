"""Ingest streamlines from TRX files into zarr vectors.

Requires ``trx-python``: ``pip install trx-python``.

TRX is the closest format to zarr vectors for streamlines — it uses
separate arrays for positions, offsets, per-vertex data (dpv),
per-streamline data (dps), groups, and per-group data (dpg).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from zarr_vectors.exceptions import IngestError
from zarr_vectors.types.polylines import write_polylines
from zarr_vectors.typing import ChunkShape


def ingest_trx(
    input_path: str | Path,
    output_path: str | Path,
    chunk_shape: ChunkShape,
    *,
    dtype: str = "float32",
) -> dict[str, Any]:
    """Ingest a TRX file into a zarr vectors streamline store.

    Maps TRX arrays to zarr vectors:

    - ``positions`` → ``vertices``
    - ``offsets`` → vertex group boundaries
    - ``dpv/*`` → ``attributes/*`` (per-vertex)
    - ``dps/*`` → ``object_attributes/*`` (per-streamline)
    - ``groups/*`` → ``groupings``
    - ``dpg/*`` → ``groupings_attributes``

    Args:
        input_path: Path to the input .trx file/directory.
        output_path: Path for the output zarr vectors store.
        chunk_shape: Spatial chunk size per dimension (3D).
        dtype: Dtype for position data.

    Returns:
        Summary dict from :func:`write_polylines`.

    Raises:
        IngestError: If trx-python is not installed or the file is unreadable.
    """
    try:
        from trx.trx_file_memmap import load as trx_load
    except ImportError as e:
        raise IngestError(
            "trx-python is required for TRX ingest. "
            "Install with: pip install trx-python"
        ) from e

    input_path = Path(input_path)
    if not input_path.exists():
        raise IngestError(f"Input file not found: {input_path}")

    try:
        trx = trx_load(str(input_path))
    except Exception as e:
        raise IngestError(f"Failed to read TRX '{input_path}': {e}") from e

    np_dtype = np.dtype(dtype)

    # Extract streamlines from positions + offsets
    positions = np.asarray(trx.streamlines._data, dtype=np_dtype)
    offsets = np.asarray(trx.streamlines._offsets, dtype=np.int64)

    polylines: list[np.ndarray] = []
    n_streamlines = len(offsets)
    for i in range(n_streamlines):
        start = int(offsets[i])
        end = int(offsets[i + 1]) if i + 1 < n_streamlines else len(positions)
        polylines.append(positions[start:end].copy())

    if len(polylines) == 0:
        raise IngestError(f"TRX file contains no streamlines: {input_path}")

    # Extract per-vertex data (dpv)
    vertex_attributes: dict[str, list[np.ndarray]] | None = None
    if hasattr(trx, "data_per_vertex") and trx.data_per_vertex:
        vertex_attributes = {}
        for key in trx.data_per_vertex:
            dpv_data = np.asarray(trx.data_per_vertex[key], dtype=np.float32)
            # Split by streamline offsets
            attr_list: list[np.ndarray] = []
            for i in range(n_streamlines):
                start = int(offsets[i])
                end = int(offsets[i + 1]) if i + 1 < n_streamlines else len(dpv_data)
                attr_list.append(dpv_data[start:end].copy())
            vertex_attributes[key] = attr_list

    # Extract per-streamline data (dps)
    object_attributes: dict[str, np.ndarray] | None = None
    if hasattr(trx, "data_per_streamline") and trx.data_per_streamline:
        object_attributes = {}
        for key in trx.data_per_streamline:
            object_attributes[key] = np.asarray(
                trx.data_per_streamline[key], dtype=np.float32
            )

    # Extract groups
    groups: dict[int, list[int]] | None = None
    group_names: list[str] = []
    if hasattr(trx, "groups") and trx.groups:
        groups = {}
        for gid, (group_name, group_data) in enumerate(trx.groups.items()):
            indices = np.asarray(group_data, dtype=np.int64).tolist()
            groups[gid] = indices
            group_names.append(group_name)

    # Build group attributes (tract names)
    group_attributes: dict[str, np.ndarray] | None = None
    if group_names:
        # Store group names as float IDs (string storage is more complex)
        group_attributes = {
            "group_id": np.arange(len(group_names), dtype=np.float32),
        }

    return write_polylines(
        str(output_path),
        polylines,
        chunk_shape=chunk_shape,
        vertex_attributes=vertex_attributes,
        object_attributes=object_attributes,
        groups=groups,
        group_attributes=group_attributes,
        dtype=dtype,
        geometry_type="streamline",
    )
