"""Ingest point clouds from CSV/XYZ text files into ZVF.

Supports:
- XYZ files (3 columns: x, y, z)
- CSV with header row (columns identified by name)
- CSV without header (first D columns are coordinates, rest are attributes)

Optional enrichments (all default off):

- ``auto_detect_columns`` — heuristic detection of x/y/z, r/g/b, intensity, label.
- ``drop_na`` — drop rows with NaN positions.
- ``drop_duplicates`` — drop rows with duplicate position triples.
- ``normalise`` — centre + scale positions to ``[-1, 1]``; offset and scale
  are stored in the CSV header for round-trip export.
- ``knn_distance_k`` — write ``attributes["knn_distance"]`` (requires scipy).
- ``per_object_vertex_count`` — write ``object_attributes["vertex_count"]``
  when ``object_ids`` is supplied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from zarr_vectors.exceptions import IngestError
from zarr_vectors.types.points import write_points
from zarr_vectors.typing import BinShape, ChunkShape


# Lower-cased column name → canonical role.
_POSITION_PATTERNS = {
    "x": "x", "pos_x": "x", "posx": "x", "px": "x",
    "y": "y", "pos_y": "y", "posy": "y", "py": "y",
    "z": "z", "pos_z": "z", "posz": "z", "pz": "z",
}
_ATTRIBUTE_PATTERNS = {
    "r": "r", "red": "r",
    "g": "g", "green": "g",
    "b": "b", "blue": "b",
    "intensity": "intensity", "i": "intensity",
    "label": "label", "class": "label", "tissue": "label",
    "confidence": "confidence", "conf": "confidence",
}


def _auto_detect(col_names: list[str], ndim: int) -> tuple[list[str], list[str]]:
    """Heuristically split column names into position vs attribute roles."""
    lowered = [c.strip().lower() for c in col_names]
    pos_axes = ["x", "y", "z"][:ndim]
    pos_cols: list[str | None] = [None] * ndim
    attr_cols: list[str] = []

    for original, low in zip(col_names, lowered):
        role = _POSITION_PATTERNS.get(low)
        if role in pos_axes and pos_cols[pos_axes.index(role)] is None:
            pos_cols[pos_axes.index(role)] = original
            continue
        if low in _ATTRIBUTE_PATTERNS:
            attr_cols.append(original)
            continue

    if any(c is None for c in pos_cols):
        # Couldn't fully resolve positions; caller should fall back.
        return [], []
    return [str(c) for c in pos_cols], attr_cols


def ingest_csv(
    input_path: str | Path,
    output_path: str | Path,
    chunk_shape: ChunkShape,
    *,
    bin_shape: BinShape | None = None,
    ndim: int = 3,
    delimiter: str = ",",
    has_header: bool = True,
    position_columns: list[str] | list[int] | None = None,
    attribute_columns: list[str] | list[int] | None = None,
    dtype: str = "float32",
    skip_rows: int = 0,
    object_ids: np.ndarray | None = None,
    auto_detect_columns: bool = False,
    drop_na: bool = False,
    drop_duplicates: bool = False,
    normalise: bool = False,
    knn_distance_k: int | None = None,
    per_object_vertex_count: bool = False,
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
            Default: first *ndim* columns (or auto-detected when
            ``auto_detect_columns=True``).
        attribute_columns: Column names or indices for attributes.
            Default: all remaining columns.
        dtype: Dtype for position data.
        skip_rows: Number of rows to skip before data (after header).
        object_ids: Optional ``(N,)`` integer array of per-vertex object
            IDs. Required for ``per_object_vertex_count``.
        auto_detect_columns: If True and ``position_columns`` is None,
            detect ``x/y/z`` (case-insensitive) and common attribute
            columns (``r/g/b``, ``intensity``, ``label``, ...).
        drop_na: Drop rows with NaN in any position column.
        drop_duplicates: Drop rows whose position triple matches a
            previous row.
        normalise: Subtract centroid and divide by max-abs-coord so
            positions sit in ``[-1, 1]``. The applied offset and scale
            are stored in the CSV header (``CSVHeader.normalise_offset``,
            ``normalise_scale``) so export can invert the transform.
        knn_distance_k: If an int, compute each point's mean Euclidean
            distance to its k nearest neighbours and store as
            ``attributes["knn_distance"]``. Requires ``scipy``.
        per_object_vertex_count: If True and ``object_ids`` is provided,
            write per-object vertex counts to
            ``object_attributes["vertex_count"]``.

    Returns:
        Summary dict from :func:`~zarr_vectors.types.points.write_points`,
        plus enrichment counters (``dropped_na``, ``dropped_duplicates``).

    Raises:
        IngestError: If the file cannot be read or parsed.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise IngestError(f"Input file not found: {input_path}")

    col_names: list[str] = []
    try:
        if has_header:
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

            if position_columns is None and auto_detect_columns:
                detected_pos, detected_attr = _auto_detect(col_names, ndim)
                if detected_pos:
                    position_columns = detected_pos
                    if attribute_columns is None:
                        attribute_columns = detected_attr

            if position_columns is None:
                pos_idx = list(range(ndim))
            elif isinstance(position_columns[0], str):
                pos_idx = [col_names.index(c) for c in position_columns]
            else:
                pos_idx = list(position_columns)

            if attribute_columns is None:
                attr_idx = [i for i in range(len(col_names)) if i not in pos_idx]
            elif len(attribute_columns) > 0 and isinstance(attribute_columns[0], str):
                attr_idx = [col_names.index(c) for c in attribute_columns]
            else:
                attr_idx = list(attribute_columns)

            positions = data[:, pos_idx].astype(np.dtype(dtype))

            attributes: dict[str, np.ndarray] = {}
            for i in attr_idx:
                name = col_names[i] if i < len(col_names) else f"col{i}"
                attributes[name] = data[:, i].astype(np.float32)

        else:
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

    enrichment_summary: dict[str, Any] = {}

    if drop_na:
        mask = ~np.isnan(positions).any(axis=1)
        dropped = int((~mask).sum())
        if dropped:
            positions = positions[mask]
            attributes = {k: v[mask] for k, v in attributes.items()}
            if object_ids is not None:
                object_ids = object_ids[mask]
        enrichment_summary["dropped_na"] = dropped

    if drop_duplicates:
        _, first_idx = np.unique(positions, axis=0, return_index=True)
        keep_mask = np.zeros(len(positions), dtype=bool)
        keep_mask[first_idx] = True
        dropped = int((~keep_mask).sum())
        if dropped:
            positions = positions[keep_mask]
            attributes = {k: v[keep_mask] for k, v in attributes.items()}
            if object_ids is not None:
                object_ids = object_ids[keep_mask]
        enrichment_summary["dropped_duplicates"] = dropped

    normalise_offset: np.ndarray | None = None
    normalise_scale: float | None = None
    if normalise and len(positions):
        normalise_offset = positions.mean(axis=0)
        centred = positions - normalise_offset
        normalise_scale = float(np.abs(centred).max()) or 1.0
        positions = (centred / normalise_scale).astype(np.dtype(dtype))

    if knn_distance_k is not None and len(positions):
        from zarr_vectors_tools.ingest._point_enrichments import compute_knn_distance
        attributes["knn_distance"] = compute_knn_distance(positions, knn_distance_k)

    object_attributes: dict[str, np.ndarray] | None = None
    if per_object_vertex_count:
        if object_ids is None:
            raise IngestError(
                "per_object_vertex_count requires object_ids to be supplied."
            )
        from zarr_vectors_tools.ingest._point_enrichments import (
            compute_per_object_vertex_count,
        )
        _, counts = compute_per_object_vertex_count(object_ids)
        object_attributes = {"vertex_count": counts}

    write_kwargs: dict[str, Any] = {
        "chunk_shape": chunk_shape,
        "bin_shape": bin_shape,
        "attributes": attributes if attributes else None,
        "dtype": dtype,
    }
    if object_ids is not None:
        write_kwargs["object_ids"] = object_ids
    if object_attributes is not None:
        write_kwargs["object_attributes"] = object_attributes

    result = write_points(str(output_path), positions, **write_kwargs)
    result.update(enrichment_summary)

    if normalise_offset is not None:
        try:
            from zarr_vectors_tools.headers.formats import CSVHeader
            from zarr_vectors_tools.headers.registry import HeaderRegistry

            existing_pos_names = position_columns if (
                position_columns and isinstance(position_columns[0], str)
            ) else ["x", "y", "z"][:ndim]
            existing_attr_names = attribute_columns if (
                attribute_columns
                and len(attribute_columns)
                and isinstance(attribute_columns[0], str)
            ) else []

            csv_header = CSVHeader(
                column_names=col_names,
                delimiter=delimiter,
                position_columns=list(existing_pos_names),
                attribute_columns=list(existing_attr_names),
                has_header_row=has_header,
                normalise_offset=normalise_offset.tolist(),
                normalise_scale=normalise_scale,
            )
            HeaderRegistry(str(output_path)).add("csv", csv_header)
        except Exception:
            pass  # header preservation is best-effort

    return result
