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
) -> dict[str, Any]:
    """Ingest a TCK file into a zarr vectors streamline store.

    TCK files store streamlines in scanner (RAS) millimetre coordinates
    with no per-vertex attributes — only positions.

    Args:
        input_path: Path to the input .tck file.
        output_path: Path for the output zarr vectors store.
        chunk_shape: Spatial chunk size per dimension (3D).
        dtype: Dtype for position data.

    Returns:
        Summary dict from :func:`write_polylines`.

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

    return write_polylines(
        str(output_path),
        polylines,
        chunk_shape=chunk_shape,
        dtype=dtype,
        geometry_type="streamline",
    )
