"""Export ZVF point clouds to PLY files.

Requires the ``plyfile`` package: ``pip install plyfile``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from zarr_vectors.exceptions import ExportError
from zarr_vectors.types.points import read_points
from zarr_vectors.typing import BoundingBox


def export_ply(
    store_path: str | Path,
    output_path: str | Path,
    *,
    level: int = 0,
    bbox: BoundingBox | None = None,
    object_ids: list[int] | None = None,
    attribute_names: list[str] | None = None,
    binary: bool = True,
) -> dict[str, Any]:
    """Export a ZVF point cloud to a PLY file.

    Args:
        store_path: Path to the ZVF store.
        output_path: Path for the output PLY file.
        level: Resolution level to export.
        bbox: Optional bounding box filter.
        object_ids: Optional object ID filter.
        attribute_names: Attributes to include.
        binary: Write binary PLY (default) or ASCII.

    Returns:
        Summary dict with ``vertex_count``.

    Raises:
        ExportError: If plyfile is not installed or export fails.
    """
    try:
        from plyfile import PlyData, PlyElement
    except ImportError as e:
        raise ExportError(
            "plyfile is required for PLY export. "
            "Install with: pip install plyfile"
        ) from e

    try:
        result = read_points(
            str(store_path),
            level=level,
            bbox=bbox,
            object_ids=object_ids,
            attribute_names=attribute_names,
        )
    except Exception as e:
        raise ExportError(f"Failed to read store: {e}") from e

    positions = result["positions"]
    attrs = result.get("attributes", {})
    n_pts, ndim = positions.shape

    # Build structured array for plyfile
    dim_names = ["x", "y", "z"][:ndim] if ndim <= 3 else [f"dim{i}" for i in range(ndim)]
    dtypes: list[tuple[str, str]] = [(name, "f4") for name in dim_names]

    for attr_name in (attribute_names or []):
        if attr_name in attrs:
            attr_data = attrs[attr_name]
            if attr_data.ndim == 1:
                dtypes.append((attr_name, "f4"))
            else:
                for c in range(attr_data.shape[1]):
                    dtypes.append((f"{attr_name}_{c}", "f4"))

    structured = np.empty(n_pts, dtype=dtypes)
    for i, name in enumerate(dim_names):
        structured[name] = positions[:, i]

    for attr_name in (attribute_names or []):
        if attr_name in attrs:
            attr_data = attrs[attr_name]
            if attr_data.ndim == 1:
                structured[attr_name] = attr_data.astype(np.float32)
            else:
                for c in range(attr_data.shape[1]):
                    structured[f"{attr_name}_{c}"] = attr_data[:, c].astype(np.float32)

    try:
        vertex_el = PlyElement.describe(structured, "vertex")
        ply_data = PlyData([vertex_el], text=not binary)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ply_data.write(str(output_path))
    except Exception as e:
        raise ExportError(f"Failed to write PLY '{output_path}': {e}") from e

    return {"vertex_count": n_pts}
