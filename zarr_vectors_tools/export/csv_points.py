"""Export ZVF point clouds to CSV/XYZ text files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from zarr_vectors.exceptions import ExportError
from zarr_vectors.types.points import read_points
from zarr_vectors.typing import BoundingBox


def export_csv(
    store_path: str | Path,
    output_path: str | Path,
    *,
    level: int = 0,
    bbox: BoundingBox | None = None,
    object_ids: list[int] | None = None,
    delimiter: str = ",",
    header: bool = True,
    attribute_names: list[str] | None = None,
) -> dict[str, Any]:
    """Export a ZVF point cloud to a CSV file.

    Args:
        store_path: Path to the ZVF store.
        output_path: Path for the output CSV file.
        level: Resolution level to export.
        bbox: Optional bounding box filter.
        object_ids: Optional object ID filter.
        delimiter: Column delimiter.
        header: Whether to write a header row.
        attribute_names: Attributes to include.  None = positions only.

    Returns:
        Summary dict with ``vertex_count``.

    Raises:
        ExportError: If export fails.
    """
    try:
        result = read_points(
            str(store_path),
            level=level,
            bbox=bbox,
            object_ids=object_ids,
            attribute_names=attribute_names,
        )
    except Exception as e:
        raise ExportError(f"Failed to read store '{store_path}': {e}") from e

    positions = result["positions"]
    attrs = result.get("attributes", {})
    n_pts, ndim = positions.shape

    # Build column names
    dim_names = [f"dim{i}" for i in range(ndim)]
    columns = list(dim_names)

    # Build data matrix
    parts = [positions]
    for name in (attribute_names or []):
        if name in attrs:
            attr_data = attrs[name]
            if attr_data.ndim == 1:
                attr_data = attr_data.reshape(-1, 1)
            parts.append(attr_data.astype(np.float64))
            if attr_data.shape[1] == 1:
                columns.append(name)
            else:
                columns.extend(f"{name}_{i}" for i in range(attr_data.shape[1]))

    data = np.concatenate(parts, axis=1) if len(parts) > 1 else positions

    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if header:
            header_line = delimiter.join(columns)
            np.savetxt(
                output_path, data,
                delimiter=delimiter,
                header=header_line,
                comments="",
                fmt="%.6f",
            )
        else:
            np.savetxt(output_path, data, delimiter=delimiter, fmt="%.6f")

    except Exception as e:
        raise ExportError(f"Failed to write CSV '{output_path}': {e}") from e

    return {"vertex_count": n_pts}
