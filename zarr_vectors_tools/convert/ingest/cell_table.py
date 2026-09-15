"""Ingest a delimited table of identified points, and stage columns from more.

This is the CSV ingester for tables whose rows carry a *stable identifier*
alongside their coordinates — the Allen Brain Cell Atlas MERFISH releases
being the motivating case, where ``ccf_coordinates.csv`` holds
``cell_label, x, y, z`` and the rest of the per-cell metadata lives in
sibling files keyed by that same ``cell_label``.

It differs from :mod:`zarr_vectors_tools.convert.ingest.csv_points` in exactly the
ways that matter for staged imports:

- **A key column.** The identifier is hashed into the ``zv_join_key``
  attribute, so later files can be joined onto these points by identity
  rather than by row position. Identifiers stay usable even when they do
  not fit a numeric column — the atlas's are 39-digit values.
- **Mixed column types.** It reads through pandas, so string and
  categorical metadata columns come through as codes plus labels, where
  ``csv_points``' ``numpy.loadtxt`` path requires everything be numeric.
- **A header.** Column labels and categories are recorded so
  :func:`~zarr_vectors_tools.convert.export.h5ad.export_h5ad` can reconstitute the
  table.

Ingest the coordinate table once with :func:`ingest_table`, then stage in
each remaining file with :func:`attach_table` or
:func:`~zarr_vectors_tools.convert.ingest.h5ad.attach_h5ad`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from zarr_vectors.exceptions import IngestError
from zarr_vectors.types.points import write_points
from zarr_vectors.typing import BinShape, ChunkShape

from zarr_vectors_tools.convert.ingest._tabular import encode_column, sanitise_name
from zarr_vectors_tools.convert.ingest.attach import (
    DEFAULT_KEY_ATTRIBUTE,
    attach_attributes,
    hash_keys,
    register_attached,
)

# Same rationale as the h5ad ingester: inlining identifiers costs store
# metadata, so past this many rows they are dropped.
DEFAULT_MAX_INDEX = 200_000


def _read_table(path: Path, delimiter: str, columns: list[str] | None):
    """Read a delimited table with pandas, keeping only what is needed."""
    import pandas as pd

    try:
        return pd.read_csv(
            path,
            sep=delimiter,
            usecols=columns,
            low_memory=False,
        )
    except ValueError as e:
        # usecols raises when a name is absent; re-read the header so the
        # error can name what was actually available.
        available = list(pd.read_csv(path, sep=delimiter, nrows=0).columns)
        raise IngestError(
            f"failed to read '{path.name}': {e}. Available columns: {available}"
        ) from e
    except Exception as e:
        raise IngestError(f"failed to read '{path}': {e}") from e


def ingest_table(
    input_path: str | Path,
    output_path: str | Path,
    chunk_shape: ChunkShape,
    *,
    position_columns: list[str],
    key_column: str | None = None,
    columns: list[str] | None = None,
    bin_shape: BinShape | None = None,
    dtype: str = "float32",
    delimiter: str = ",",
    object_id_column: str | None = None,
    drop_na: bool = True,
    preserve_index: bool = True,
    max_index: int = DEFAULT_MAX_INDEX,
) -> dict[str, Any]:
    """Ingest a delimited table of identified points into a Zarr Vectors store.

    Args:
        input_path: Path to the input table (CSV/TSV).
        output_path: Path for the output Zarr Vectors store.
        chunk_shape: Spatial chunk size, one entry per position column.
        position_columns: Column names holding the coordinates, in axis
            order (e.g. ``["x", "y", "z"]``).
        key_column: Column holding the row identifier. Hashed into the
            ``zv_join_key`` attribute so later files can be staged in
            against it. Strongly recommended — without it the store cannot
            be joined to.
        columns: Metadata columns to store as vertex attributes. Default
            (None) stores every column that is neither a position nor the
            key; pass ``[]`` for none.
        bin_shape: Optional intra-chunk sub-binning.
        dtype: Dtype for position data.
        delimiter: Column delimiter.
        object_id_column: Column grouping rows into Zarr Vectors objects.
        drop_na: Drop rows whose coordinates contain NaN. On by default —
            coordinate tables routinely carry unregistered rows, and NaN
            coordinates cannot be assigned to a chunk.
        preserve_index: Inline the key column's values into the store
            header so export can restore the original identifiers.
        max_index: Skip inlining above this many rows.

    Returns:
        Summary dict from :func:`~zarr_vectors.types.points.write_points`,
        plus ``columns_stored``, ``dropped_na`` and ``index_stored``.

    Raises:
        IngestError: If the file or the named columns cannot be read.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise IngestError(f"Input file not found: {input_path}")

    ndim = len(position_columns)
    if len(tuple(chunk_shape)) != ndim:
        raise IngestError(
            f"chunk_shape has {len(tuple(chunk_shape))} entries but "
            f"{ndim} position column(s) were named"
        )

    needed: list[str] | None = None
    if columns is not None:
        needed = list(dict.fromkeys(
            [*position_columns, *columns]
            + ([key_column] if key_column else [])
            + ([object_id_column] if object_id_column else [])
        ))

    frame = _read_table(input_path, delimiter, needed)

    missing = [c for c in position_columns if c not in frame.columns]
    if missing:
        raise IngestError(
            f"position columns not found: {missing}. Available: "
            f"{list(frame.columns)}"
        )
    if key_column and key_column not in frame.columns:
        raise IngestError(
            f"key_column {key_column!r} not found. Available: {list(frame.columns)}"
        )

    positions = frame[list(position_columns)].to_numpy(dtype=np.dtype(dtype))

    dropped = 0
    if drop_na:
        keep = ~np.isnan(positions).any(axis=1)
        dropped = int((~keep).sum())
        if dropped:
            positions = positions[keep]
            frame = frame.loc[keep].reset_index(drop=True)

    if columns is None:
        selected = [
            c for c in frame.columns
            if c not in set(position_columns) and c != key_column
        ]
    else:
        selected = [c for c in columns if c != key_column]
        unknown = [c for c in selected if c not in frame.columns]
        if unknown:
            raise IngestError(
                f"columns not found: {unknown}. Available: {list(frame.columns)}"
            )

    attributes: dict[str, np.ndarray] = {}
    used_names: set[str] = set()
    names_kept: list[str] = []
    attrs_kept: list[str] = []
    categories: dict[str, list[str]] = {}
    dtypes: dict[str, str] = {}

    for column in selected:
        encoded, levels = encode_column(frame[column])
        attr_name = sanitise_name(column, used_names)
        attributes[attr_name] = encoded
        names_kept.append(str(column))
        attrs_kept.append(attr_name)
        dtypes[str(column)] = str(frame[column].dtype)
        if levels is not None:
            categories[str(column)] = levels

    identifiers: np.ndarray | None = None
    if key_column:
        identifiers = frame[key_column].to_numpy()
        used_names.add(DEFAULT_KEY_ATTRIBUTE)
        attributes[DEFAULT_KEY_ATTRIBUTE] = hash_keys(identifiers.astype(str))

    row_attr = sanitise_name("table_row", used_names)
    attributes[row_attr] = np.arange(len(positions), dtype=np.int64)

    object_ids: np.ndarray | None = None
    object_id_categories: list[str] | None = None
    if object_id_column is not None:
        if object_id_column not in frame.columns:
            raise IngestError(
                f"object_id_column {object_id_column!r} not found. Available: "
                f"{list(frame.columns)}"
            )
        codes, object_id_categories = encode_column(frame[object_id_column])
        object_ids = np.asarray(codes, dtype=np.int64)
        if (object_ids < 0).any():
            raise IngestError(
                f"object_id_column {object_id_column!r} has missing values, "
                "which have no valid object ID. Fill or drop them first."
            )

    write_kwargs: dict[str, Any] = {
        "chunk_shape": chunk_shape,
        "bin_shape": bin_shape,
        "vertex_attributes": attributes or None,
        "dtype": dtype,
    }
    if object_ids is not None:
        write_kwargs["object_ids"] = object_ids

    result = write_points(str(output_path), positions, **write_kwargs)

    inline_index = bool(
        preserve_index and identifiers is not None and len(positions) <= max_index
    )
    try:
        from zarr_vectors_tools.headers.formats import H5ADHeader
        from zarr_vectors_tools.headers.registry import HeaderRegistry

        header = H5ADHeader(
            spatial_key="spatial",
            spatial_ndim=ndim,
            spatial_columns=list(range(ndim)),
            n_obs=len(positions),
            n_vars=0,
            obs_names=names_kept,
            obs_attrs=attrs_kept,
            categories=categories,
            dtypes=dtypes,
            row_attr=row_attr,
            obs_index=(
                [str(v) for v in identifiers] if inline_index else None
            ),
            object_id_column=object_id_column,
            object_id_categories=object_id_categories,
        )
        HeaderRegistry(str(output_path)).add("h5ad", header)
    except Exception:
        inline_index = False  # header preservation is best-effort

    result.update(
        {
            "columns_stored": len(attrs_kept),
            "dropped_na": dropped,
            "index_stored": inline_index,
            "key_column": key_column,
        }
    )
    return result


def attach_table(
    store_path: str | Path,
    input_path: str | Path,
    *,
    key_column: str,
    columns: list[str] | None = None,
    delimiter: str = ",",
    key_attribute: str = DEFAULT_KEY_ATTRIBUTE,
    level: int = 0,
    missing: str = "fill",
    fill_value: Any = np.nan,
    overwrite: bool = False,
    shard_shape: int | tuple[int, ...] | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Stage columns from a delimited table into an existing store.

    Rows are matched to points on ``key_column``, so the table needs
    neither the same order nor the same row set as the store.

    Args:
        store_path: Zarr Vectors store to add attributes to (modified in place).
        input_path: Table to read.
        key_column: Column holding the identifier to join on.
        columns: Columns to stage in. Default (None) takes every column
            except the key.
        delimiter: Column delimiter.
        key_attribute: Store attribute holding the join key.
        level: Resolution level to write into.
        missing: ``"fill"`` or ``"error"`` for points absent from the table.
        fill_value: Value written for unmatched points.
        overwrite: Allow replacing attributes that already exist.
        shard_shape: Create the new attribute arrays sharded (chunks per
            axis per storage object). See
            :func:`~zarr_vectors_tools.convert.ingest.attach.attach_attributes`.
        progress: Print progress while writing.

    Returns:
        The :func:`~zarr_vectors_tools.convert.ingest.attach.attach_attributes`
        summary, plus ``columns_attached``.

    Raises:
        IngestError: If the file or the named columns cannot be read.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise IngestError(f"Input file not found: {input_path}")

    needed = None
    if columns is not None:
        needed = list(dict.fromkeys([key_column, *columns]))
    frame = _read_table(input_path, delimiter, needed)

    if key_column not in frame.columns:
        raise IngestError(
            f"key_column {key_column!r} not found. Available: {list(frame.columns)}"
        )

    selected = (
        [c for c in frame.columns if c != key_column]
        if columns is None
        else list(columns)
    )
    unknown = [c for c in selected if c not in frame.columns]
    if unknown:
        raise IngestError(
            f"columns not found: {unknown}. Available: {list(frame.columns)}"
        )
    if not selected:
        raise IngestError("attach_table requires at least one non-key column")

    values: dict[str, np.ndarray] = {}
    used_names: set[str] = set()
    obs_map: dict[str, str] = {}
    categories: dict[str, list[str]] = {}
    dtypes: dict[str, str] = {}

    for column in selected:
        encoded, levels = encode_column(frame[column])
        attr_name = sanitise_name(column, used_names)
        values[attr_name] = encoded
        obs_map[attr_name] = str(column)
        dtypes[str(column)] = str(frame[column].dtype)
        if levels is not None:
            categories[str(column)] = levels

    result = attach_attributes(
        store_path,
        values,
        keys=frame[key_column].to_numpy().astype(str),
        level=level,
        key_attribute=key_attribute,
        missing=missing,
        fill_value=fill_value,
        overwrite=overwrite,
        shard_shape=shard_shape,
        progress=progress,
    )

    register_attached(
        store_path, obs_names=obs_map, categories=categories, dtypes=dtypes
    )

    result.update({"columns_attached": len(obs_map), "source": str(input_path)})
    return result
