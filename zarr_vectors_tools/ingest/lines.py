"""Ingest line segments from CSV text files into ZVF.

Six-column CSV format: ``x0,y0,z0,x1,y1,z1`` (one row per segment),
plus optional per-line attribute columns. Mirrors ``ingest_csv`` for
points.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from zarr_vectors.exceptions import IngestError
from zarr_vectors.types.lines import write_lines
from zarr_vectors.typing import BinShape, ChunkShape


def ingest_lines_csv(
    input_path: str | Path,
    output_path: str | Path,
    chunk_shape: ChunkShape,
    *,
    bin_shape: BinShape | None = None,
    ndim: int = 3,
    delimiter: str = ",",
    has_header: bool = True,
    dtype: str = "float32",
    attribute_columns: list[str] | None = None,
    compute_length: bool = False,
    drop_zero_length: bool = False,
    drop_na: bool = False,
    drop_duplicates: bool = False,
) -> dict[str, Any]:
    """Ingest a CSV of line segments into a ZVF line store.

    Expected layout: the first ``2 * ndim`` columns hold endpoint
    coordinates in ``(x0, y0, z0, x1, y1, z1)`` order. Remaining columns
    are treated as per-line attributes (or filtered to ``attribute_columns``
    when supplied).

    Args:
        input_path: Path to the CSV file.
        output_path: Path for the output ZVF store.
        chunk_shape: Spatial chunk size per dimension.
        ndim: Spatial dimensionality (default 3).
        delimiter: Column delimiter.
        has_header: Whether the first row is a header.
        dtype: Dtype for position data.
        attribute_columns: Names (when ``has_header=True``) of columns to
            keep as per-line attributes. Default: every non-position column.
        compute_length: If True, write ``line_attributes["length"]``.
        drop_zero_length: If True, drop segments shorter than 1e-9.
        drop_na: Drop rows with NaN in any endpoint coordinate.
        drop_duplicates: Drop rows with identical endpoint pairs.

    Returns:
        Summary dict from :func:`~zarr_vectors.types.lines.write_lines`,
        plus enrichment counters.

    Raises:
        IngestError: If the file is missing or unparseable.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise IngestError(f"Input file not found: {input_path}")

    n_pos = 2 * ndim
    col_names: list[str] = []
    try:
        if has_header:
            with open(input_path) as f:
                col_names = [c.strip() for c in f.readline().strip().split(delimiter)]
            data = np.loadtxt(input_path, delimiter=delimiter, skiprows=1, dtype=np.float64)
        else:
            data = np.loadtxt(input_path, delimiter=delimiter, dtype=np.float64)
    except Exception as e:
        raise IngestError(f"Failed to read line CSV '{input_path}': {e}") from e

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < n_pos:
        raise IngestError(
            f"Line CSV must have at least {n_pos} columns "
            f"(got {data.shape[1]}) for ndim={ndim}."
        )

    endpoints_flat = data[:, :n_pos].astype(np.dtype(dtype))
    attrs_data = data[:, n_pos:]

    line_attributes: dict[str, np.ndarray] = {}
    if has_header:
        attr_names = col_names[n_pos:]
    else:
        attr_names = [f"col{i + n_pos}" for i in range(attrs_data.shape[1])]
    keep_attr_names = (
        list(attribute_columns)
        if attribute_columns is not None
        else attr_names
    )
    for i, name in enumerate(attr_names):
        if name in keep_attr_names:
            line_attributes[name] = attrs_data[:, i].astype(np.float32)

    enrichment_summary: dict[str, Any] = {}

    if drop_na:
        mask = ~np.isnan(endpoints_flat).any(axis=1)
        dropped = int((~mask).sum())
        if dropped:
            endpoints_flat = endpoints_flat[mask]
            line_attributes = {k: v[mask] for k, v in line_attributes.items()}
        enrichment_summary["dropped_na"] = dropped

    if drop_duplicates:
        _, first = np.unique(endpoints_flat, axis=0, return_index=True)
        keep = np.zeros(len(endpoints_flat), dtype=bool)
        keep[first] = True
        dropped = int((~keep).sum())
        if dropped:
            endpoints_flat = endpoints_flat[keep]
            line_attributes = {k: v[keep] for k, v in line_attributes.items()}
        enrichment_summary["dropped_duplicates"] = dropped

    # Reshape to (N, 2, D) for write_lines.
    endpoints = endpoints_flat.reshape(-1, 2, ndim)

    # Compute lengths up front since two features may need them.
    diffs = endpoints[:, 1, :] - endpoints[:, 0, :]
    lengths = np.linalg.norm(diffs, axis=1).astype(np.float32)

    if drop_zero_length:
        mask = lengths >= 1e-9
        dropped = int((~mask).sum())
        if dropped:
            endpoints = endpoints[mask]
            lengths = lengths[mask]
            line_attributes = {k: v[mask] for k, v in line_attributes.items()}
        enrichment_summary["dropped_zero_length"] = dropped

    if compute_length:
        line_attributes["length"] = lengths

    if len(endpoints) == 0:
        raise IngestError(
            f"No line segments remain after enrichment filtering: {input_path}"
        )

    result = write_lines(
        str(output_path),
        endpoints,
        chunk_shape=chunk_shape,
        bin_shape=bin_shape,
        object_attributes=line_attributes if line_attributes else None,
        dtype=dtype,
    )
    result.update(enrichment_summary)
    return result
