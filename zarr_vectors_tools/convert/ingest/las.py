"""Ingest point clouds from LAS/LAZ files into Zarr Vectors.

Requires the ``laspy`` package: ``pip install 'zarr-vectors-tools[las]'``
(which also installs ``lazrs``, the ``.laz`` decompressor).

Each dimension is stored in a dtype that holds every value it can take:
intensity, colour and classification are 8- and 16-bit integers, which
float32 holds exactly; GPS time is float64, which it does not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from zarr_vectors.exceptions import IngestError
from zarr_vectors.types.points import write_points
from zarr_vectors.typing import BinShape, ChunkShape

from zarr_vectors_tools.convert.ingest._object_columns import stamp_object_columns


def ingest_las(
    input_path: str | Path,
    output_path: str | Path,
    chunk_shape: ChunkShape,
    *,
    bin_shape: BinShape | None = None,
    dtype: str = "float32",
    include_attributes: bool = True,
    object_ids: np.ndarray | None = None,
    knn_distance_k: int | None = None,
    per_object_vertex_count: bool | None = None,
) -> dict[str, Any]:
    """Ingest a LAS or LAZ file into a Zarr Vectors point cloud store.

    Args:
        input_path: Path to the input LAS/LAZ file.
        output_path: Path for the output Zarr Vectors store.
        chunk_shape: Spatial chunk size per dimension (3D).
        dtype: Dtype for position data.
        include_attributes: Whether to include intensity,
            classification, RGB, etc. as vertex attributes.
        object_ids: Optional ``(N,)`` integer array of per-vertex object
            IDs. Required for ``per_object_vertex_count``.
        knn_distance_k: If an int, compute each point's mean Euclidean
            distance to its k nearest neighbours and store as
            ``attributes["knn_distance"]``. Requires ``scipy``.
        per_object_vertex_count: Write per-object vertex counts to
            ``object_attributes["vertex_count"]``.  ``None`` (default):
            whenever ``object_ids`` is given; ``True`` requires it.  With
            objects, each fragment's ``segment_id`` is written too (see
            :mod:`._object_columns`).

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
            "Install with: pip install 'zarr-vectors-tools[las]'"
        ) from e

    input_path = Path(input_path)
    if not input_path.exists():
        raise IngestError(f"Input file not found: {input_path}")

    try:
        las = laspy.read(str(input_path))
    except Exception as e:
        raise IngestError(f"Failed to read LAS file '{input_path}': {e}") from e

    # Extract XYZ positions
    world = np.stack([las.x, las.y, las.z], axis=1)
    positions = world.astype(np.dtype(dtype))
    precision_note = _precision_note(world, positions, las)

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
            # float64, as the file has it.  float32 steps by 1/32 s at GPS
            # week seconds (~4e5) and by 8 s at adjusted standard time
            # (~1e8), so pulses milliseconds apart collapse to one value.
            attributes["gps_time"] = np.asarray(las.gps_time, dtype=np.float64)

    if knn_distance_k is not None and len(positions):
        from zarr_vectors_tools.convert.ingest._point_enrichments import compute_knn_distance
        attributes["knn_distance"] = compute_knn_distance(positions, knn_distance_k)

    if per_object_vertex_count and object_ids is None:
        raise IngestError(
            "per_object_vertex_count requires object_ids to be supplied."
        )

    write_kwargs: dict[str, Any] = {
        "chunk_shape": chunk_shape,
        "bin_shape": bin_shape,
        "vertex_attributes": attributes if attributes else None,
        "dtype": dtype,
    }
    if object_ids is not None:
        write_kwargs["object_ids"] = object_ids

    result = write_points(str(output_path), positions, **write_kwargs)
    if object_ids is not None:
        stamp_object_columns(
            output_path, vertex_count=per_object_vertex_count is not False,
        )
    if precision_note:
        result["warnings"] = [precision_note]
    return result


def _precision_note(world: np.ndarray, stored: np.ndarray, las: Any) -> str | None:
    """Say so when the stored positions are coarser than the file's own.

    LAS stores integers times a scale (often 1 cm), so georeferenced
    coordinates are millions of units from the origin -- where float32, the
    default and the only dtype the viewer reads, steps by a quarter of a
    unit.  Not refused (the default is a choice), but not silent either.
    """
    scales = getattr(getattr(las, "header", None), "scales", None)
    if scales is None or not len(world) or stored.dtype.kind != "f":
        return None
    step = float(np.spacing(np.abs(stored).max(axis=0)).max())
    finest = float(np.min(np.asarray(scales, dtype=np.float64)))
    if step <= finest:
        return None
    return (
        f"positions are stored as {stored.dtype}, which steps by up to {step:.3g} "
        f"at these coordinates, coarser than the file's {finest:g} resolution "
        f"(max error {float(np.abs(stored - world).max()):.3g}); dtype='float64' "
        f"(--dtype float64) keeps it, but the viewer reads float32 only"
    )
