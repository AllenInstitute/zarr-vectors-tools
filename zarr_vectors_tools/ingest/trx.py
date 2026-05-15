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
from zarr_vectors.typing import BinShape, ChunkShape


def ingest_trx(
    input_path: str | Path,
    output_path: str | Path,
    chunk_shape: ChunkShape,
    *,
    bin_shape: BinShape | None = None,
    dtype: str = "float32",
    compute_length: bool = False,
    compute_endpoints: bool = False,
    length_range: tuple[float, float] | None = None,
    mean_scalar: str | list[str] | None = None,
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
        compute_length: If True, write per-streamline path length to
            ``object_attributes["length"]``.
        compute_endpoints: If True, write per-streamline start and end
            points to ``object_attributes["start"]`` and ``["end"]``.
        length_range: Optional ``(min, max)`` length bounds; streamlines
            outside the range are dropped before writing.
        mean_scalar: Name or list of names of per-vertex (dpv) scalars to
            aggregate. For each, write its per-streamline mean to
            ``object_attributes["mean_<name>"]``. Names not found in the
            TRX file are skipped silently.

    Returns:
        Summary dict from :func:`write_polylines`, plus any enrichment
        counters such as ``dropped_by_length``.

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

    # Optional enrichments
    from zarr_vectors_tools.ingest._polyline_enrichments import (
        compute_endpoints as _compute_endpoints,
        compute_lengths as _compute_lengths,
        filter_by_length as _filter_by_length,
    )

    enrichment_summary: dict[str, Any] = {}
    lengths: np.ndarray | None = None

    if length_range is not None:
        lengths = _compute_lengths(polylines)
        kept, kept_idx, dropped = _filter_by_length(polylines, length_range, lengths=lengths)
        if dropped:
            polylines = kept
            lengths = lengths[kept_idx]
            if vertex_attributes is not None:
                vertex_attributes = {
                    k: [v[i] for i in kept_idx] for k, v in vertex_attributes.items()
                }
            if object_attributes is not None:
                object_attributes = {
                    k: v[kept_idx] for k, v in object_attributes.items()
                }
            # Groups index into the original streamline order; rebuild against kept_idx.
            if groups is not None:
                old_to_new = {int(old): new for new, old in enumerate(kept_idx)}
                new_groups: dict[int, list[int]] = {}
                for gid, idxs in groups.items():
                    new_groups[gid] = [old_to_new[i] for i in idxs if i in old_to_new]
                groups = new_groups
        enrichment_summary["dropped_by_length"] = dropped

    if compute_length:
        if lengths is None:
            lengths = _compute_lengths(polylines)
        object_attributes = dict(object_attributes or {})
        object_attributes["length"] = lengths

    if compute_endpoints:
        start, end = _compute_endpoints(polylines)
        object_attributes = dict(object_attributes or {})
        object_attributes["start"] = start
        object_attributes["end"] = end

    if mean_scalar is not None and vertex_attributes:
        names = [mean_scalar] if isinstance(mean_scalar, str) else list(mean_scalar)
        object_attributes = dict(object_attributes or {})
        for name in names:
            if name not in vertex_attributes:
                continue
            means = np.array(
                [float(np.mean(v)) if len(v) else 0.0 for v in vertex_attributes[name]],
                dtype=np.float32,
            )
            object_attributes[f"mean_{name}"] = means

    if len(polylines) == 0:
        raise IngestError(
            f"No streamlines remain after enrichment filtering: {input_path}"
        )

    result = write_polylines(
        str(output_path),
        polylines,
        chunk_shape=chunk_shape,
        bin_shape=bin_shape,
        vertex_attributes=vertex_attributes,
        object_attributes=object_attributes,
        groups=groups,
        group_attributes=group_attributes,
        dtype=dtype,
        geometry_type="streamline",
    )
    result.update(enrichment_summary)
    return result
