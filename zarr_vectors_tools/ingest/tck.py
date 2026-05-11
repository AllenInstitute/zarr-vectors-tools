"""Ingest streamlines from MRtrix TCK files into zarr vectors.

Requires ``nibabel``: ``pip install nibabel``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from zarr_vectors.exceptions import IngestError
from zarr_vectors.types.polylines import write_polylines
from zarr_vectors.typing import ChunkShape


def ingest_tck(
    input_path: str | Path,
    output_path: str | Path,
    chunk_shape: ChunkShape,
    *,
    dtype: str = "float32",
    compute_length: bool = False,
    compute_endpoints: bool = False,
    length_range: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Ingest a TCK file into a zarr vectors streamline store.

    TCK files store streamlines in scanner (RAS) millimetre coordinates
    with no per-vertex attributes — only positions.

    Args:
        input_path: Path to the input .tck file.
        output_path: Path for the output zarr vectors store.
        chunk_shape: Spatial chunk size per dimension (3D).
        dtype: Dtype for position data.
        compute_length: If True, write per-streamline path length to
            ``object_attributes["length"]``.
        compute_endpoints: If True, write per-streamline start and end
            points to ``object_attributes["start"]`` and ``["end"]``.
        length_range: Optional ``(min, max)`` length bounds; streamlines
            outside the range are dropped before writing.

    Returns:
        Summary dict from :func:`write_polylines`, plus any enrichment
        counters such as ``dropped_by_length``.

    Raises:
        IngestError: If nibabel is not installed or the file is unreadable.
    """
    try:
        import nibabel as nib
    except ImportError as e:
        raise IngestError(
            "nibabel is required for TCK ingest. "
            "Install with: pip install nibabel"
        ) from e

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        tck = nib.streamlines.load(str(input_path))
    except Exception as e:
        raise IngestError(f"Failed to read TCK '{input_path}': {e}") from e

    streamlines = tck.streamlines
    np_dtype = np.dtype(dtype)
    polylines = [np.asarray(s, dtype=np_dtype) for s in streamlines]

    if len(polylines) == 0:
        raise IngestError(f"TCK file contains no streamlines: {input_path}")

    from zarr_vectors_tools.ingest._polyline_enrichments import (
        compute_endpoints as _compute_endpoints,
        compute_lengths as _compute_lengths,
        filter_by_length as _filter_by_length,
    )

    enrichment_summary: dict[str, Any] = {}
    lengths: np.ndarray | None = None
    object_attributes: dict[str, np.ndarray] | None = None

    if length_range is not None:
        lengths = _compute_lengths(polylines)
        kept, kept_idx, dropped = _filter_by_length(polylines, length_range, lengths=lengths)
        if dropped:
            polylines = kept
            lengths = lengths[kept_idx]
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

    if len(polylines) == 0:
        raise IngestError(
            f"No streamlines remain after enrichment filtering: {input_path}"
        )

    result = write_polylines(
        str(output_path),
        polylines,
        chunk_shape=chunk_shape,
        object_attributes=object_attributes,
        dtype=dtype,
        geometry_type="streamline",
    )
    result.update(enrichment_summary)
    return result
