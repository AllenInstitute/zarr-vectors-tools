"""Ingest point clouds from LAS/LAZ files into ZVF.

Requires the ``laspy`` package: ``pip install laspy``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from zarr_vectors.exceptions import IngestError
from zarr_vectors.types.points import write_points
from zarr_vectors.typing import ChunkShape


def ingest_las(
    input_path: str | Path,
    output_path: str | Path,
    chunk_shape: ChunkShape,
    *,
    dtype: str = "float32",
    include_attributes: bool = True,
) -> dict[str, Any]:
    """Ingest a LAS or LAZ file into a ZVF point cloud store.

    Args:
        input_path: Path to the input LAS/LAZ file.
        output_path: Path for the output ZVF store.
        chunk_shape: Spatial chunk size per dimension (3D).
        dtype: Dtype for position data.
        include_attributes: Whether to include intensity,
            classification, RGB, etc. as vertex attributes.

    Returns:
        Summary dict from :func:`~zarr_vectors.types.points.write_points`.

    Raises:
        IngestError: If laspy is not installed or the file is unreadable.
    """
    try:
        import laspy
    except ImportError as e:
        raise IngestError(
            "laspy is required for LAS/LAZ ingest. "
            "Install with: pip install laspy"
        ) from e

    input_path = Path(input_path)
    if not input_path.exists():
        raise IngestError(f"Input file not found: {input_path}")

    try:
        las = laspy.read(str(input_path))
    except Exception as e:
        raise IngestError(f"Failed to read LAS file '{input_path}': {e}") from e

    # Extract XYZ positions
    positions = np.stack([las.x, las.y, las.z], axis=1).astype(np.dtype(dtype))

    # Extract attributes
    attributes: dict[str, np.ndarray] = {}
    if include_attributes:
        if hasattr(las, "intensity") and las.intensity is not None:
            attributes["intensity"] = np.asarray(las.intensity, dtype=np.float32)

        if hasattr(las, "classification") and las.classification is not None:
            attributes["classification"] = np.asarray(
                las.classification, dtype=np.int32
            ).astype(np.float32)

        # RGB if present
        if hasattr(las, "red") and las.red is not None:
            try:
                rgb = np.stack([las.red, las.green, las.blue], axis=1)
                attributes["color"] = rgb.astype(np.float32)
            except Exception:
                pass

        if hasattr(las, "gps_time") and las.gps_time is not None:
            attributes["gps_time"] = np.asarray(las.gps_time, dtype=np.float64).astype(
                np.float32
            )

    return write_points(
        str(output_path),
        positions,
        chunk_shape=chunk_shape,
        attributes=attributes if attributes else None,
        dtype=dtype,
    )
