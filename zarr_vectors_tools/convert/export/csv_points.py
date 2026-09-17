"""Export Zarr Vectors point clouds to CSV/XYZ text files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from zarr_vectors.exceptions import ExportError
from zarr_vectors.typing import BoundingBox, ChunkCoords

from zarr_vectors_tools.convert.export._point_batches import (
    DEFAULT_VERTEX_BUDGET,
    attribute_columns,
    iter_point_batches,
    ndim_of,
    require_attributes,
)


def export_csv(
    store_path: str | Path,
    output_path: str | Path,
    *,
    level: int = 0,
    bbox: BoundingBox | None = None,
    object_ids: list[int] | None = None,
    chunks: list[ChunkCoords] | None = None,
    delimiter: str = ",",
    header: bool = True,
    attribute_names: list[str] | None = None,
    vertex_budget: int = DEFAULT_VERTEX_BUDGET,
) -> dict[str, Any]:
    """Export a Zarr Vectors point cloud to a CSV file.

    The level is read and written a batch of chunks (or objects) at a time,
    so memory holds one batch whatever the level's size; the file is the
    same as a whole-level read would give.

    Args:
        store_path: Path to the Zarr Vectors store.
        output_path: Path for the output CSV file.
        level: Resolution level to export.
        bbox: Optional bounding box filter.
        object_ids: Optional object ID filter.
        chunks: Optional whitelist of chunk coordinate tuples; only data
            stored in those chunks is exported. AND-ed with ``bbox`` and
            ``object_ids``.
        delimiter: Column delimiter.
        header: Whether to write a header row.
        attribute_names: Attributes to include.  None = positions only.
        vertex_budget: Points to aim for per batch.

    Returns:
        Summary dict with ``vertex_count``.

    Raises:
        ExportError: If export fails.
    """
    require_attributes(store_path, level, attribute_names)
    ndim = ndim_of(store_path)
    columns = [f"dim{i}" for i in range(ndim)]
    wrote_header = False
    n_pts = 0

    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as handle:
            for result in _batches(store_path, level, bbox, object_ids, chunks, attribute_names,
                                   vertex_budget):
                positions = result["positions"]
                attrs = attribute_columns(result, attribute_names, level)
                parts = [positions]
                names = list(columns)
                for name in (attribute_names or []):
                    column = attrs.get(name)
                    if column is None:
                        continue
                    column = column.reshape(-1, 1) if column.ndim == 1 else column
                    parts.append(column.astype(np.float64))
                    names.extend(
                        [name] if column.shape[1] == 1
                        else [f"{name}_{i}" for i in range(column.shape[1])]
                    )
                if header and not wrote_header:
                    handle.write(delimiter.join(names) + "\n")
                    wrote_header = True
                if len(positions):
                    data = np.concatenate(parts, axis=1) if len(parts) > 1 else positions
                    np.savetxt(handle, data, delimiter=delimiter, fmt="%.6f")
                    n_pts += len(positions)
            if header and not wrote_header:
                handle.write(delimiter.join(columns + list(attribute_names or [])) + "\n")
    except ExportError:
        raise
    except Exception as e:
        raise ExportError(f"Failed to write CSV '{output_path}': {e}") from e

    return {"vertex_count": n_pts}


def _batches(store_path, level, bbox, object_ids, chunks, attribute_names, vertex_budget):
    try:
        yield from iter_point_batches(
            store_path, level=level, bbox=bbox, object_ids=object_ids, chunks=chunks,
            attribute_names=attribute_names, vertex_budget=vertex_budget,
        )
    except ExportError:
        raise
    except Exception as e:
        raise ExportError(f"Failed to read store '{store_path}': {e}") from e
