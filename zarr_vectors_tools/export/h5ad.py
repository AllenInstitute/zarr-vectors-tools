"""Export Zarr Vectors point clouds to AnnData ``.h5ad`` files.

The inverse of :mod:`zarr_vectors_tools.ingest.h5ad`: positions become
``obsm[spatial_key]``, per-vertex attributes become ``obs`` columns, and
attributes that were ingested as gene expression are rebuilt into ``X``
with the matching ``var_names``.

When the store carries an
:class:`~zarr_vectors_tools.headers.formats.H5ADHeader` (any store written
by this package's h5ad ingest does), the original column labels, gene
names, categorical levels, cell barcodes and row order are all restored.
Without one, the export still works — every attribute becomes a numeric
``obs`` column under its stored name.

Requires the ``anndata`` package::

    pip install 'zarr-vectors-tools[h5ad]'
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from zarr_vectors.exceptions import ExportError
from zarr_vectors.types.points import read_points
from zarr_vectors.typing import BoundingBox, ChunkCoords

from zarr_vectors_tools.ingest.attach import DEFAULT_KEY_ATTRIBUTE


def _load_header(store_path: str):
    """Return the store's H5ADHeader, or None when it has none."""
    try:
        from zarr_vectors_tools.headers.formats import H5ADHeader
        from zarr_vectors_tools.headers.registry import HeaderRegistry

        header = HeaderRegistry(store_path).get("h5ad")
        return header if isinstance(header, H5ADHeader) else None
    except Exception:
        return None


def _list_vertex_attributes(store_path: str, level: int) -> list[str]:
    """Names of every per-vertex attribute stored at ``level``."""
    from zarr_vectors.building import (
        VERTEX_ATTRIBUTES,
        get_resolution_level,
        open_store,
    )

    try:
        root = open_store(store_path)
        group = get_resolution_level(root, level)
        return sorted(group.require_group(VERTEX_ATTRIBUTES).children())
    except Exception:
        return []


def _restore_dtype(values: np.ndarray, source_dtype: str | None) -> Any:
    """Undo the numeric encoding that ingest applied to an obs column.

    Only ``bool`` (stored as uint8) is reversed. Numeric columns kept their
    numpy dtype through the store and are returned untouched. ``datetime64``
    columns stay as int64 nanoseconds on purpose: AnnData has no h5ad
    serialisation for datetimes, so handing one back would produce a file
    that ``write_h5ad`` refuses. The original dtype stays recorded in the
    header for callers that want to convert.
    """
    if source_dtype == "bool":
        return values.astype(bool)
    return values


def export_h5ad(
    store_path: str | Path,
    output_path: str | Path,
    *,
    level: int = 0,
    bbox: BoundingBox | None = None,
    object_ids: list[int] | None = None,
    chunks: list[ChunkCoords] | None = None,
    attribute_names: list[str] | None = None,
    spatial_key: str | None = None,
    restore_order: bool = True,
    decode_categoricals: bool = True,
    compression: str | None = "gzip",
) -> dict[str, Any]:
    """Export a Zarr Vectors point cloud to an AnnData ``.h5ad`` file.

    Args:
        store_path: Path to the Zarr Vectors store.
        output_path: Path for the output ``.h5ad`` file.
        level: Resolution level to export.
        bbox: Optional bounding box filter.
        object_ids: Optional object ID filter. Note that core's
            object-filtered read path does not carry vertex attributes, so
            this may only be combined with ``attribute_names=[]``.
        chunks: Optional whitelist of chunk coordinate tuples. AND-ed with
            ``bbox`` and ``object_ids``.
        attribute_names: Attributes to export. Default (None) exports every
            attribute present at ``level``; pass ``[]`` for positions only.
        spatial_key: ``obsm`` key to write the positions under. Defaults to
            the key recorded in the store header, else ``"spatial"``.
        restore_order: Sort cells back into their original ``obs`` order
            using the row-index attribute recorded at ingest. Silently
            skipped when that attribute is absent or was filtered out.
        decode_categoricals: Rebuild ``pandas.Categorical`` obs columns
            from their integer codes using the header's category labels.
        compression: HDF5 compression for the output ("gzip", "lzf", or
            None).

    Returns:
        Summary dict with ``vertex_count``, ``n_vars``, ``obs_columns``,
        ``spatial_key`` and ``order_restored``.

    Raises:
        ExportError: If ``anndata`` is missing, the store cannot be read,
            or attributes were requested through a filter that drops them.
    """
    try:
        import anndata as ad
    except ImportError as e:
        raise ExportError(
            "anndata is required for .h5ad export. Install with: "
            "pip install 'zarr-vectors-tools[h5ad]'"
        ) from e

    import pandas as pd

    store_path = str(store_path)
    header = _load_header(store_path)

    if attribute_names is None:
        attribute_names = _list_vertex_attributes(store_path, level)
    requested = list(attribute_names)

    try:
        result = read_points(
            store_path,
            level=level,
            bbox=bbox,
            object_ids=object_ids,
            chunks=chunks,
            attribute_names=requested or None,
        )
    except Exception as e:
        raise ExportError(f"Failed to read store '{store_path}': {e}") from e

    positions = np.asarray(result["positions"])
    attrs: dict[str, np.ndarray] = dict(result.get("vertex_attributes") or {})
    n_cells = positions.shape[0]

    # Core's object-filtered read returns positions without attributes. Say
    # so rather than writing an .h5ad whose obs is silently empty.
    if requested and not attrs and object_ids is not None and n_cells:
        raise ExportError(
            "object_ids filtering drops vertex attributes in the core read "
            f"path, so the {len(requested)} requested attribute(s) came back "
            "empty. Re-run with attribute_names=[] to export positions only, "
            "or select cells with bbox=/chunks= instead."
        )

    missing = [name for name in requested if name not in attrs]
    if missing and n_cells:
        raise ExportError(
            f"attributes not found at level {level}: {missing}. "
            f"Available: {sorted(attrs) or '(none)'}"
        )

    # ---- restore the source row order ------------------------------------
    row_attr = header.row_attr if header else None
    order_restored = False
    if restore_order and row_attr and row_attr in attrs:
        order = np.argsort(np.asarray(attrs[row_attr]), kind="stable")
        positions = positions[order]
        attrs = {k: np.asarray(v)[order] for k, v in attrs.items()}
        order_restored = True

    # ---- split attributes into expression / obs / obsm -------------------
    gene_map = header.attr_to_gene() if header else {}
    obs_map = header.attr_to_obs() if header else {}
    categories = header.categories if (header and decode_categoricals) else {}
    source_dtypes = header.dtypes if header else {}

    gene_columns = [name for name in gene_map if name in attrs]
    if gene_columns:
        expression = np.column_stack(
            [np.asarray(attrs[name], dtype=np.float32) for name in gene_columns]
        )
        var_names = [gene_map[name] for name in gene_columns]
    else:
        expression = np.zeros((n_cells, 0), dtype=np.float32)
        var_names = []

    obs_data: dict[str, Any] = {}
    obsm_extra: dict[str, np.ndarray] = {}
    for name, values in attrs.items():
        # The row index and join key are Zarr Vectors bookkeeping, not data: one
        # restores ordering, the other exists so further files can be
        # staged in. Neither belongs in the exported obs table.
        if name in gene_map or name == row_attr or name == DEFAULT_KEY_ATTRIBUTE:
            continue
        values = np.asarray(values)
        label = obs_map.get(name, name)
        if values.ndim > 1:
            # Multi-channel attributes have no single obs column; keep them
            # as an obsm matrix rather than dropping them.
            obsm_extra[label] = values
            continue
        levels = categories.get(label)
        if levels is not None:
            codes = np.asarray(values, dtype=np.int32)
            obs_data[label] = pd.Categorical.from_codes(
                codes, categories=levels, ordered=False
            )
        else:
            obs_data[label] = _restore_dtype(values, source_dtypes.get(label))

    # ---- cell identities -------------------------------------------------
    obs_index: list[str] | None = None
    if header and header.obs_index and row_attr and row_attr in attrs:
        source_rows = np.asarray(attrs[row_attr], dtype=np.int64)
        if source_rows.size and 0 <= source_rows.min() and source_rows.max() < len(
            header.obs_index
        ):
            obs_index = [header.obs_index[i] for i in source_rows]
    if obs_index is None:
        obs_index = [str(i) for i in range(n_cells)]

    obs = pd.DataFrame(obs_data, index=pd.Index(obs_index, dtype=object))
    var = pd.DataFrame(index=pd.Index(var_names, dtype=object))

    key = spatial_key or (header.spatial_key if header else None) or "spatial"

    try:
        adata = ad.AnnData(X=expression, obs=obs, var=var)
        adata.obsm[key] = np.ascontiguousarray(positions)
        for name, values in obsm_extra.items():
            adata.obsm[name] = np.ascontiguousarray(values)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(str(output_path), compression=compression)
    except Exception as e:
        raise ExportError(f"Failed to write h5ad '{output_path}': {e}") from e

    return {
        "vertex_count": n_cells,
        "n_vars": len(var_names),
        "obs_columns": len(obs_data),
        "spatial_key": key,
        "order_restored": order_restored,
    }
