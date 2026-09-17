"""Export Zarr Vectors point clouds to PLY files.

Requires the ``plyfile`` package: ``pip install plyfile``.
"""

from __future__ import annotations

import shutil
import tempfile
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


def export_ply(
    store_path: str | Path,
    output_path: str | Path,
    *,
    level: int = 0,
    bbox: BoundingBox | None = None,
    object_ids: list[int] | None = None,
    chunks: list[ChunkCoords] | None = None,
    attribute_names: list[str] | None = None,
    binary: bool = True,
    vertex_budget: int = DEFAULT_VERTEX_BUDGET,
) -> dict[str, Any]:
    """Export a Zarr Vectors point cloud to a PLY file.

    PLY states its vertex count in the header, before the data, so the
    vertices are written a batch at a time to a scratch file beside the
    output, counted, and the header written in front of them.  Memory holds
    one batch whatever the level's size.

    Args:
        store_path: Path to the Zarr Vectors store.
        output_path: Path for the output PLY file.
        level: Resolution level to export.
        bbox: Optional bounding box filter.
        object_ids: Optional object ID filter.
        chunks: Optional whitelist of chunk coordinate tuples; only data
            stored in those chunks is exported. AND-ed with ``bbox`` and
            ``object_ids``.
        attribute_names: Attributes to include.
        binary: Write binary PLY (default) or ASCII.
        vertex_budget: Points to aim for per batch.

    Returns:
        Summary dict with ``vertex_count``.

    Raises:
        ExportError: If plyfile is not installed or export fails.
    """
    try:
        import plyfile  # noqa: F401 - the format's reader; kept as the documented extra
    except ImportError as e:
        raise ExportError(
            "plyfile is required for PLY export. "
            "Install with: pip install plyfile"
        ) from e

    require_attributes(store_path, level, attribute_names)
    ndim = ndim_of(store_path)
    dim_names = ["x", "y", "z"][:ndim] if ndim <= 3 else [f"dim{i}" for i in range(ndim)]
    properties: list[str] | None = None
    n_pts = 0

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent, prefix=output_path.name, suffix=".body", delete=False,
        ) as body:
            body_path = Path(body.name)
            try:
                for result in iter_point_batches(
                    store_path, level=level, bbox=bbox, object_ids=object_ids,
                    chunks=chunks, attribute_names=attribute_names,
                    vertex_budget=vertex_budget,
                ):
                    positions = result["positions"]
                    attrs = attribute_columns(result, attribute_names, level)
                    columns = [positions[:, i] for i in range(ndim)]
                    names = list(dim_names)
                    for name in (attribute_names or []):
                        column = attrs.get(name)
                        if column is None:
                            continue
                        column = column.reshape(-1, 1) if column.ndim == 1 else column
                        if column.shape[1] == 1:
                            names.append(name)
                            columns.append(column[:, 0])
                        else:
                            names.extend(f"{name}_{c}" for c in range(column.shape[1]))
                            columns.extend(column[:, c] for c in range(column.shape[1]))
                    if properties is None:
                        properties = names
                    if not len(positions):
                        continue
                    rows = np.column_stack(columns).astype("<f4")
                    if binary:
                        body.write(np.ascontiguousarray(rows).tobytes())
                    else:
                        np.savetxt(body, rows, fmt="%.9g")
                    n_pts += len(positions)
            except ExportError:
                raise
            except Exception as e:
                raise ExportError(f"Failed to read store: {e}") from e

        if properties is None:
            properties = list(dim_names) + list(attribute_names or [])
        header = [
            "ply",
            f"format {'binary_little_endian' if binary else 'ascii'} 1.0",
            f"element vertex {n_pts}",
            *[f"property float {name}" for name in properties],
            "end_header",
        ]
        with open(output_path, "wb") as out:
            out.write(("\n".join(header) + "\n").encode("ascii"))
            with open(body_path, "rb") as source:
                shutil.copyfileobj(source, out, length=1 << 24)
    except ExportError:
        raise
    except Exception as e:
        raise ExportError(f"Failed to write PLY '{output_path}': {e}") from e
    finally:
        try:
            body_path.unlink()
        except (NameError, OSError):
            pass

    return {"vertex_count": n_pts}
