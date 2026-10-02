"""Ingest point clouds from PLY files into Zarr Vectors.

Requires the ``plyfile`` package: ``pip install 'zarr-vectors-tools[ply]'``.
Detects whether the PLY contains mesh faces or just points and
dispatches accordingly.  This module handles point clouds only;
mesh PLY ingest is in ``ingest.ply_mesh`` (future).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from zarr_vectors.exceptions import IngestError
from zarr_vectors.types.points import write_points
from zarr_vectors.typing import BinShape, ChunkShape

from zarr_vectors_tools.convert.ingest._object_columns import stamp_object_columns


def ingest_ply(
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
    """Ingest a PLY file as a Zarr Vectors point cloud.

    Args:
        input_path: Path to the input PLY file.
        output_path: Path for the output Zarr Vectors store.
        chunk_shape: Spatial chunk size per dimension.
        dtype: Dtype for position data.
        include_attributes: Whether to include non-position
            vertex properties (normals, colours, etc.).
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
        IngestError: If plyfile is not installed, the file has no vertex
            element, or reading fails.
    """
    try:
        from plyfile import PlyData
    except ImportError as e:
        raise IngestError(
            "plyfile is required for PLY ingest. "
            "Install with: pip install 'zarr-vectors-tools[ply]'"
        ) from e

    input_path = Path(input_path)
    if not input_path.exists():
        raise IngestError(f"Input file not found: {input_path}")

    try:
        ply = PlyData.read(str(input_path))
    except Exception as e:
        raise IngestError(f"Failed to read PLY '{input_path}': {e}") from e

    if "vertex" not in ply:
        raise IngestError(f"PLY file has no 'vertex' element: {input_path}")

    vertex = ply["vertex"]

    # Extract positions — look for x,y,z or X,Y,Z
    prop_names = [p.name for p in vertex.properties]
    coord_names: list[str] = []
    for candidate in [("x", "y", "z"), ("X", "Y", "Z")]:
        if all(c in prop_names for c in candidate):
            coord_names = list(candidate)
            break

    if not coord_names:
        # Try first 3 properties
        if len(prop_names) >= 3:
            coord_names = prop_names[:3]
        else:
            raise IngestError(
                f"Cannot identify position columns in PLY vertex properties: {prop_names}"
            )

    positions = np.column_stack(
        [np.asarray(vertex[c], dtype=np.float64) for c in coord_names]
    ).astype(np.dtype(dtype))

    # Extract attributes
    attributes: dict[str, np.ndarray] = {}
    if include_attributes:
        non_pos = [p for p in prop_names if p not in coord_names]
        for pname in non_pos:
            try:
                arr = np.asarray(vertex[pname], dtype=np.float32)
                attributes[pname] = arr
            except Exception:
                continue

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
    return result
