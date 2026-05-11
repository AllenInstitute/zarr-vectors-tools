"""Ingest point clouds from CSV/XYZ text files into ZVF.

Supports:
- XYZ files (3 columns: x, y, z)
- CSV with header row (columns identified by name)
- CSV without header (first D columns are coordinates, rest are attributes)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from zarr_vectors.exceptions import IngestError
from zarr_vectors.types.points import write_points
from zarr_vectors.typing import ChunkShape


def ingest_csv(
    input_path: str | Path,
    output_path: str | Path,
    chunk_shape: ChunkShape,
    *,
    ndim: int = 3,
    delimiter: str = ",",
    has_header: bool = True,
    position_columns: list[str] | list[int] | None = None,
    attribute_columns: list[str] | list[int] | None = None,
    dtype: str = "float32",
    skip_rows: int = 0,
) -> dict[str, Any]:
    """Ingest a CSV or XYZ file into a ZVF point cloud store.

    Args:
        input_path: Path to input CSV/XYZ file.
        output_path: Path for output ZVF store.
        chunk_shape: Spatial chunk size per dimension.
        ndim: Number of spatial dimensions (default 3).
        delimiter: Column delimiter (default ``,``).
        has_header: Whether the first row is a header.
        position_columns: Column names or indices for positions.
            Default: first *ndim* columns.
        attribute_columns: Column names or indices for attributes.
            Default: all remaining columns.
        dtype: Dtype for position data.
        skip_rows: Number of rows to skip before data (after header).

    Returns:
        Summary dict from :func:`~zarr_vectors.types.points.write_points`.

    Raises:
        IngestError: If the file cannot be read or parsed.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise IngestError(f"Input file not found: {input_path}")

    try:
        if has_header:
            # Read header
            with open(input_path) as f:
                header_line = f.readline().strip()
            col_names = [c.strip() for c in header_line.split(delimiter)]

            data = np.loadtxt(
                input_path,
                delimiter=delimiter,
                skiprows=1 + skip_rows,
                dtype=np.float64,
            )

            if data.ndim == 1:
                data = data.reshape(1, -1)

            # Resolve position columns
            if position_columns is None:
                pos_idx = list(range(ndim))
            elif isinstance(position_columns[0], str):
                pos_idx = [col_names.index(c) for c in position_columns]
            else:
                pos_idx = list(position_columns)

            # Resolve attribute columns
            if attribute_columns is None:
                attr_idx = [i for i in range(len(col_names)) if i not in pos_idx]
            elif isinstance(attribute_columns[0], str):
                attr_idx = [col_names.index(c) for c in attribute_columns]
            else:
                attr_idx = list(attribute_columns)

            positions = data[:, pos_idx].astype(np.dtype(dtype))

            attributes: dict[str, np.ndarray] = {}
            for i in attr_idx:
                name = col_names[i] if i < len(col_names) else f"col{i}"
                attributes[name] = data[:, i].astype(np.float32)

        else:
            # No header — positional
            data = np.loadtxt(
                input_path,
                delimiter=delimiter,
                skiprows=skip_rows,
                dtype=np.float64,
            )
            if data.ndim == 1:
                data = data.reshape(1, -1)

            if position_columns is None:
                pos_idx = list(range(ndim))
            else:
                pos_idx = list(position_columns)

            positions = data[:, pos_idx].astype(np.dtype(dtype))

            if attribute_columns is None:
                attr_idx = [i for i in range(data.shape[1]) if i not in pos_idx]
            else:
                attr_idx = list(attribute_columns)

            attributes = {}
            for i in attr_idx:
                attributes[f"col{i}"] = data[:, i].astype(np.float32)

    except Exception as e:
        raise IngestError(f"Failed to read CSV '{input_path}': {e}") from e

    return write_points(
        str(output_path),
        positions,
        chunk_shape=chunk_shape,
        attributes=attributes if attributes else None,
        dtype=dtype,
    )
