"""Ingest AnnData ``.h5ad`` files into ZVF point clouds.

An ``.h5ad`` file is a cell-by-gene table with side tables; the only part
of it that is *spatial* is an ``obsm`` embedding — ``obsm["spatial"]`` for
spatial transcriptomics (Visium, Xenium, MERFISH, CosMx), or a
dimensionality reduction such as ``obsm["X_umap"]`` for dissociated data.
That embedding becomes the ZVF point cloud: one point per cell.

Everything else rides along as per-vertex attributes:

- ``obs`` columns.  Numeric columns are stored as-is; categorical/string
  columns are stored as integer codes with their labels recorded in the
  :class:`~zarr_vectors_tools.headers.formats.H5ADHeader` so export can
  rebuild the ``pandas.Categorical``.
- Selected genes.  Expression is a wide matrix (often 20k+ columns) and a
  ZVF attribute is one array per name, so genes are opt-in via ``genes=``
  and pulled from ``X`` or from ``layers[layer]``.

Two details make the round-trip exact.  ZVF orders vertices by spatial
chunk rather than by source row, so the source row index is stored as its
own attribute and export sorts on it.  And attribute names must be valid
Zarr path segments, so labels are sanitised and the originals kept in the
header.

Requires the ``anndata`` package::

    pip install 'zarr-vectors-tools[h5ad]'
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from zarr_vectors.exceptions import IngestError
from zarr_vectors.types.points import write_points
from zarr_vectors.typing import BinShape, ChunkShape

from zarr_vectors_tools.ingest._tabular import encode_column, sanitise_name
from zarr_vectors_tools.ingest.attach import (
    DEFAULT_KEY_ATTRIBUTE,
    attach_attributes,
    hash_keys,
    register_attached,
)

# obsm keys tried, in order, when ``spatial_key="auto"``.  Real spatial
# coordinates first, then the conventional embeddings.
_AUTO_SPATIAL_KEYS = (
    "spatial",
    "X_spatial",
    "spatial_fov",
    "X_umap",
    "X_tsne",
    "X_pca",
)

# Inlining the obs index costs ~20 bytes/cell of JSON in the store's
# ``.zattrs``.  Past this many cells the barcodes are dropped rather than
# bloating the metadata; the row-index attribute still restores order.
DEFAULT_MAX_OBS_INDEX = 200_000


def _resolve_spatial_key(adata, spatial_key: str) -> str:
    """Pick the ``obsm`` key holding the coordinates."""
    available = list(adata.obsm.keys())
    if spatial_key != "auto":
        if spatial_key not in adata.obsm:
            raise IngestError(
                f"obsm key {spatial_key!r} not found; available keys: "
                f"{available or '(none)'}"
            )
        return spatial_key

    for candidate in _AUTO_SPATIAL_KEYS:
        if candidate in adata.obsm:
            return candidate
    raise IngestError(
        "could not auto-detect spatial coordinates: none of "
        f"{list(_AUTO_SPATIAL_KEYS)} are in obsm (available: "
        f"{available or '(none)'}). Pass spatial_key=... explicitly."
    )


def _densify(matrix) -> np.ndarray:
    """Materialise a possibly-sparse / possibly-backed expression block."""
    if hasattr(matrix, "toarray"):  # scipy.sparse
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def ingest_h5ad(
    input_path: str | Path,
    output_path: str | Path,
    chunk_shape: ChunkShape,
    *,
    bin_shape: BinShape | None = None,
    dtype: str = "float32",
    spatial_key: str = "auto",
    spatial_columns: list[int] | None = None,
    obs_columns: list[str] | None = None,
    genes: list[str] | None = None,
    layer: str | None = None,
    object_id_column: str | None = None,
    backed: bool = False,
    drop_na: bool = False,
    store_row_index: bool = True,
    store_join_key: bool = True,
    preserve_obs_index: bool = True,
    max_obs_index: int = DEFAULT_MAX_OBS_INDEX,
    knn_distance_k: int | None = None,
    per_object_vertex_count: bool = False,
) -> dict[str, Any]:
    """Ingest an AnnData ``.h5ad`` file into a ZVF point cloud store.

    Args:
        input_path: Path to the input ``.h5ad`` file.
        output_path: Path for the output ZVF store.
        chunk_shape: Spatial chunk size per dimension. Must have one entry
            per spatial dimension (2 for a 2D embedding, 3 for 3D).
        bin_shape: Optional intra-chunk sub-binning.
        dtype: Dtype for position data.
        spatial_key: ``obsm`` key holding the coordinates. ``"auto"``
            (default) tries ``spatial``, ``X_spatial``, ``spatial_fov``,
            ``X_umap``, ``X_tsne``, ``X_pca`` in that order.
        spatial_columns: Which columns of the embedding to use, e.g.
            ``[0, 1]`` to take the first two of a 3-column array. Default:
            all columns (capped at 3).
        obs_columns: ``obs`` columns to store as vertex attributes.
            Default (None) stores every column; pass ``[]`` for none.
        genes: ``var`` names whose expression to store as vertex
            attributes. Default (None) stores none — expression is wide
            and each gene costs one array.
        layer: Pull expression from ``adata.layers[layer]`` instead of
            ``adata.X``.
        object_id_column: ``obs`` column grouping cells into ZVF objects
            (e.g. ``"cell_type"`` or ``"sample"``). Categorical labels are
            encoded as integer codes; the labels go in the header.
        backed: Open the file in AnnData's backed mode, leaving ``X`` on
            disk. Worth it for large files when ``genes`` selects few
            columns; ``obs``/``obsm`` are read into memory either way.
        drop_na: Drop cells whose coordinates contain NaN. Off by default;
            with NaN coordinates present the write will fail instead.
        store_row_index: Store each cell's source ``obs`` row index as an
            attribute so export can restore the original ordering. Leave
            on unless you never intend to export back.
        store_join_key: Store the hashed ``obs`` index as the
            ``zv_join_key`` attribute, so further files covering the same
            cells can be staged in later with :func:`attach_h5ad` or
            :func:`~zarr_vectors_tools.ingest.cell_table.attach_table`.
        preserve_obs_index: Inline the ``obs`` index (cell barcodes) into
            the stored header so export can restore ``obs_names``.
        max_obs_index: Skip inlining the index above this many cells, to
            keep store metadata small. Reported as ``obs_index_stored``.
        knn_distance_k: If an int, store each cell's mean distance to its
            k nearest neighbours as ``knn_distance``. Requires ``scipy``.
        per_object_vertex_count: With ``object_id_column``, also store
            per-object cell counts as ``object_attributes["vertex_count"]``.

    Returns:
        Summary dict from :func:`~zarr_vectors.types.points.write_points`,
        plus ``spatial_key``, ``n_obs``, ``n_vars``, ``obs_columns_stored``,
        ``genes_stored``, ``obs_index_stored`` and ``dropped_na``.

    Raises:
        IngestError: If ``anndata`` is missing, the file is unreadable, or
            the requested keys/columns/genes do not exist.
    """
    try:
        import anndata as ad
    except ImportError as e:
        raise IngestError(
            "anndata is required for .h5ad ingest. Install with: "
            "pip install 'zarr-vectors-tools[h5ad]'"
        ) from e

    input_path = Path(input_path)
    if not input_path.exists():
        raise IngestError(f"Input file not found: {input_path}")

    try:
        adata = ad.read_h5ad(str(input_path), backed="r" if backed else None)
    except Exception as e:
        raise IngestError(f"Failed to read h5ad '{input_path}': {e}") from e

    n_obs, n_vars = int(adata.n_obs), int(adata.n_vars)

    # ---- positions -------------------------------------------------------
    key = _resolve_spatial_key(adata, spatial_key)
    coords = np.asarray(adata.obsm[key])
    if coords.ndim != 2:
        raise IngestError(
            f"obsm[{key!r}] must be 2-dimensional, got shape {coords.shape}"
        )

    if spatial_columns is None:
        # A UMAP/PCA embedding can be far wider than 3; take the leading
        # components, which is the conventional way these are plotted.
        used_columns = list(range(min(coords.shape[1], 3)))
    else:
        used_columns = [int(c) for c in spatial_columns]
        out_of_range = [c for c in used_columns if not 0 <= c < coords.shape[1]]
        if out_of_range:
            raise IngestError(
                f"spatial_columns {out_of_range} out of range for "
                f"obsm[{key!r}] with {coords.shape[1]} columns"
            )

    if len(used_columns) < 2:
        raise IngestError(
            f"need at least 2 spatial columns, got {len(used_columns)} from "
            f"obsm[{key!r}] with shape {coords.shape}"
        )

    positions = coords[:, used_columns].astype(np.dtype(dtype))
    ndim = positions.shape[1]

    if len(tuple(chunk_shape)) != ndim:
        raise IngestError(
            f"chunk_shape has {len(tuple(chunk_shape))} entries but the "
            f"coordinates from obsm[{key!r}] are {ndim}-dimensional"
        )

    # ---- attributes ------------------------------------------------------
    attributes: dict[str, np.ndarray] = {}
    used_names: set[str] = set()
    obs_names_kept: list[str] = []
    obs_attrs_kept: list[str] = []
    categories: dict[str, list[str]] = {}
    source_dtypes: dict[str, str] = {}

    if obs_columns is None:
        selected_obs = [str(c) for c in adata.obs.columns]
    else:
        selected_obs = [str(c) for c in obs_columns]
        missing = [c for c in selected_obs if c not in adata.obs.columns]
        if missing:
            raise IngestError(
                f"obs columns not found: {missing}. Available: "
                f"{list(adata.obs.columns)}"
            )

    for column in selected_obs:
        try:
            values, cats = encode_column(adata.obs[column])
        except Exception as e:  # a column type we cannot encode numerically
            raise IngestError(
                f"failed to encode obs column {column!r} "
                f"(dtype {adata.obs[column].dtype}): {e}"
            ) from e
        attr_name = sanitise_name(column, used_names)
        attributes[attr_name] = values
        obs_names_kept.append(column)
        obs_attrs_kept.append(attr_name)
        source_dtypes[column] = str(adata.obs[column].dtype)
        if cats is not None:
            categories[column] = cats

    gene_names_kept: list[str] = []
    gene_attrs_kept: list[str] = []
    if genes:
        var_names = list(adata.var_names)
        var_lookup = {str(v): i for i, v in enumerate(var_names)}
        missing = [g for g in genes if str(g) not in var_lookup]
        if missing:
            raise IngestError(
                f"genes not found in var_names: {missing[:10]}"
                f"{' ...' if len(missing) > 10 else ''}"
            )
        gene_idx = [var_lookup[str(g)] for g in genes]

        source = adata[:, gene_idx]
        if backed:
            source = source.to_memory()
        try:
            block = source.layers[layer] if layer is not None else source.X
        except KeyError as e:
            raise IngestError(
                f"layer {layer!r} not found; available: {list(adata.layers)}"
            ) from e
        expression = _densify(block)
        if expression.ndim == 1:
            expression = expression.reshape(-1, 1)

        for position, gene in enumerate(genes):
            attr_name = sanitise_name(f"gene_{gene}", used_names)
            attributes[attr_name] = expression[:, position].astype(np.float32)
            gene_names_kept.append(str(gene))
            gene_attrs_kept.append(attr_name)

    # ---- object grouping -------------------------------------------------
    object_ids: np.ndarray | None = None
    object_id_categories: list[str] | None = None
    if object_id_column is not None:
        if object_id_column not in adata.obs.columns:
            raise IngestError(
                f"object_id_column {object_id_column!r} not found in obs. "
                f"Available: {list(adata.obs.columns)}"
            )
        codes, object_id_categories = encode_column(adata.obs[object_id_column])
        if object_id_categories is None:
            # Already numeric -- use the values directly as object IDs.
            object_ids = np.asarray(codes).astype(np.int64)
        else:
            object_ids = np.asarray(codes, dtype=np.int64)
        if (object_ids < 0).any():
            raise IngestError(
                f"object_id_column {object_id_column!r} has missing values, "
                "which have no valid object ID. Fill or drop them first."
            )

    row_attr: str | None = None
    if store_row_index:
        row_attr = sanitise_name("h5ad_row", used_names)
        attributes[row_attr] = np.arange(n_obs, dtype=np.int64)

    if store_join_key:
        # The hashed obs index, so files that share cell identifiers but
        # not row order can be staged in later via attach_h5ad/attach_table.
        used_names.add(DEFAULT_KEY_ATTRIBUTE)
        attributes[DEFAULT_KEY_ATTRIBUTE] = hash_keys(
            np.asarray([str(name) for name in adata.obs_names])
        )

    # ---- row filtering ---------------------------------------------------
    enrichment_summary: dict[str, Any] = {}
    if drop_na:
        keep = ~np.isnan(positions).any(axis=1)
        dropped = int((~keep).sum())
        if dropped:
            positions = positions[keep]
            attributes = {k: v[keep] for k, v in attributes.items()}
            if object_ids is not None:
                object_ids = object_ids[keep]
        enrichment_summary["dropped_na"] = dropped

    if knn_distance_k is not None and len(positions):
        from zarr_vectors_tools.ingest._point_enrichments import compute_knn_distance

        attributes["knn_distance"] = compute_knn_distance(positions, knn_distance_k)

    object_attributes: dict[str, np.ndarray] | None = None
    if per_object_vertex_count:
        if object_ids is None:
            raise IngestError(
                "per_object_vertex_count requires object_id_column to be set."
            )
        from zarr_vectors_tools.ingest._point_enrichments import (
            compute_per_object_vertex_count,
        )

        _, counts = compute_per_object_vertex_count(object_ids)
        object_attributes = {"vertex_count": counts}

    # ---- write -----------------------------------------------------------
    write_kwargs: dict[str, Any] = {
        "chunk_shape": chunk_shape,
        "bin_shape": bin_shape,
        "vertex_attributes": attributes or None,
        "dtype": dtype,
    }
    if object_ids is not None:
        write_kwargs["object_ids"] = object_ids
    if object_attributes is not None:
        write_kwargs["object_attributes"] = object_attributes

    result = write_points(str(output_path), positions, **write_kwargs)

    # ---- header ----------------------------------------------------------
    inline_index = preserve_obs_index and n_obs <= max_obs_index
    try:
        from zarr_vectors_tools.headers.formats import H5ADHeader
        from zarr_vectors_tools.headers.registry import HeaderRegistry

        header = H5ADHeader(
            spatial_key=key,
            spatial_ndim=ndim,
            spatial_columns=used_columns,
            n_obs=n_obs,
            n_vars=n_vars,
            obs_names=obs_names_kept,
            obs_attrs=obs_attrs_kept,
            gene_names=gene_names_kept,
            gene_attrs=gene_attrs_kept,
            layer=layer,
            categories=categories,
            dtypes=source_dtypes,
            row_attr=row_attr,
            obs_index=([str(i) for i in adata.obs_names] if inline_index else None),
            object_id_column=object_id_column,
            object_id_categories=object_id_categories,
        )
        HeaderRegistry(str(output_path)).add("h5ad", header)
    except Exception:
        inline_index = False  # header preservation is best-effort

    if backed and adata.isbacked:
        adata.file.close()

    result.update(enrichment_summary)
    result.update(
        {
            "spatial_key": key,
            "n_obs": n_obs,
            "n_vars": n_vars,
            "obs_columns_stored": len(obs_attrs_kept),
            "genes_stored": len(gene_attrs_kept),
            "obs_index_stored": inline_index,
        }
    )
    return result


# =====================================================================
# Staged attach: add expression / obs columns to an existing store
# =====================================================================

def _read_elem(node):
    """``anndata``'s element reader, across its module reshuffles."""
    try:
        from anndata.io import read_elem
    except ImportError:  # anndata < 0.11
        from anndata.experimental import read_elem
    return read_elem(node)


def _contiguous_runs(indices: list[int]) -> list[tuple[int, int]]:
    """Collapse sorted column indices into ``(start, stop)`` slices.

    Expression matrices are chunked across genes (the atlas MERFISH files
    use ``(16281, 5)``), so a slice covering neighbouring genes costs the
    same read as a single gene. Slicing beats fancy-indexing here.
    """
    runs: list[tuple[int, int]] = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index == previous + 1:
            previous = index
        else:
            runs.append((start, previous + 1))
            start = previous = index
    runs.append((start, previous + 1))
    return runs


def _read_expression_columns(
    handle, path: str, indices: list[int], n_rows: int
) -> np.ndarray:
    """Read selected gene columns, dense or sparse, without loading all of X.

    Returns an ``(n_rows, len(indices))`` array in the caller's requested
    gene order, which need be neither sorted nor free of repeats.
    """
    import h5py

    node = handle[path]

    if isinstance(node, h5py.Group):
        # Sparse (csr/csc): let anndata decode the element, then slice.
        matrix = _read_elem(node)
        block = matrix[:, indices]
        return np.asarray(
            block.toarray() if hasattr(block, "toarray") else block, dtype=np.float32
        )

    wanted = sorted(set(indices))
    slot = {column: position for position, column in enumerate(wanted)}
    buffer = np.empty((n_rows, len(wanted)), dtype=np.float32)

    for start, stop in _contiguous_runs(wanted):
        block = node[:, start:stop]
        for column in range(start, stop):
            if column in slot:
                buffer[:, slot[column]] = block[:, column - start]

    return buffer[:, [slot[column] for column in indices]]


def attach_h5ad(
    store_path: str | Path,
    h5ad_path: str | Path,
    *,
    genes: list[str] | None = None,
    obs_columns: list[str] | None = None,
    layer: str | None = None,
    gene_by: str | None = None,
    key_obs_column: str | None = None,
    key_attribute: str = DEFAULT_KEY_ATTRIBUTE,
    level: int = 0,
    missing: str = "fill",
    fill_value: Any = np.nan,
    overwrite: bool = False,
    shard_shape: int | tuple[int, ...] | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Stage gene expression and ``obs`` columns from an ``.h5ad`` into a store.

    The store is modified in place. Cells are matched on the join key, so
    the ``.h5ad`` needs neither the same row order nor the same cell subset
    as the store — which is what lets a whole-brain expression matrix be
    staged onto a store built from a coordinate table covering only part of
    it. Neither ``X`` nor the unused ``obs`` columns are read in full.

    Args:
        store_path: ZVF store to add attributes to.
        h5ad_path: Source ``.h5ad``.
        genes: ``var`` names (or symbols, see ``gene_by``) to stage in.
        obs_columns: ``obs`` columns to stage in.
        layer: Read expression from ``layers[layer]`` instead of ``X``.
        gene_by: ``var`` column to match ``genes`` against. Default tries
            the ``var`` index first, then ``gene_symbol`` — so Ensembl IDs
            and gene symbols both work without the caller converting.
        key_obs_column: ``obs`` column holding the join key. Default is the
            ``obs`` index, which is where cell identifiers normally live.
        key_attribute: Store attribute holding the join key.
        level: Resolution level to write into.
        missing: ``"fill"`` or ``"error"`` for cells absent from this file.
        fill_value: Value written for unmatched cells.
        overwrite: Allow replacing attributes that already exist.
        shard_shape: Create the new attribute arrays sharded (chunks per
            axis per storage object). Strongly recommended when staging a
            wide gene panel -- see
            :func:`~zarr_vectors_tools.ingest.attach.attach_attributes`.
        progress: Print progress while reading and writing.

    Returns:
        The :func:`~zarr_vectors_tools.ingest.attach.attach_attributes`
        summary, plus ``genes_attached`` and ``obs_columns_attached``.

    Raises:
        IngestError: If the file, keys, genes or columns cannot be resolved.
    """
    try:
        import h5py
    except ImportError as e:
        raise IngestError(
            "h5py is required to attach from .h5ad. Install with: "
            "pip install 'zarr-vectors-tools[h5ad]'"
        ) from e

    h5ad_path = Path(h5ad_path)
    if not h5ad_path.exists():
        raise IngestError(f"Input file not found: {h5ad_path}")
    if not genes and not obs_columns:
        raise IngestError("attach_h5ad requires genes=... or obs_columns=...")

    values: dict[str, np.ndarray] = {}
    used_names: set[str] = set()
    gene_map: dict[str, str] = {}
    obs_map: dict[str, str] = {}
    categories: dict[str, list[str]] = {}
    dtypes: dict[str, str] = {}

    with h5py.File(str(h5ad_path), "r") as handle:
        obs_group = handle["obs"]
        index_name = obs_group.attrs.get("_index", "_index")
        if isinstance(index_name, bytes):
            index_name = index_name.decode()

        key_source = key_obs_column or index_name
        if key_source not in obs_group:
            raise IngestError(
                f"join key column {key_source!r} not found in obs; available: "
                f"{list(obs_group.keys())}"
            )
        keys = np.asarray(_read_elem(obs_group[key_source]))
        if keys.dtype.kind == "O":
            keys = keys.astype(str)
        n_rows = len(keys)

        if progress:
            print(f"  read {n_rows} join keys from {h5ad_path.name}", flush=True)

        for column in obs_columns or []:
            if column not in obs_group:
                raise IngestError(
                    f"obs column {column!r} not found; available: "
                    f"{list(obs_group.keys())}"
                )
            import pandas as pd

            series = pd.Series(_read_elem(obs_group[column]))
            encoded, levels = encode_column(series)
            attr_name = sanitise_name(column, used_names)
            values[attr_name] = encoded
            obs_map[attr_name] = column
            dtypes[column] = str(series.dtype)
            if levels is not None:
                categories[column] = levels

        if genes:
            var = _read_elem(handle["var"])
            var_index = [str(v) for v in var.index]
            lookup = {name: i for i, name in enumerate(var_index)}
            if gene_by is not None:
                if gene_by not in var.columns:
                    raise IngestError(
                        f"gene_by column {gene_by!r} not in var; available: "
                        f"{list(var.columns)}"
                    )
                lookup = {str(v): i for i, v in enumerate(var[gene_by])}
            elif "gene_symbol" in var.columns:
                # Fall back to symbols so callers can name either.
                for i, symbol in enumerate(var["gene_symbol"]):
                    lookup.setdefault(str(symbol), i)

            unknown = [g for g in genes if str(g) not in lookup]
            if unknown:
                raise IngestError(
                    f"genes not found: {unknown[:10]}"
                    f"{' ...' if len(unknown) > 10 else ''}"
                )
            indices = [lookup[str(g)] for g in genes]

            path = f"layers/{layer}" if layer else "X"
            if path not in handle:
                raise IngestError(f"{path!r} not present in {h5ad_path.name}")
            if progress:
                print(f"  reading {len(indices)} gene column(s) from {path}", flush=True)
            block = _read_expression_columns(handle, path, indices, n_rows)

            for position, gene in enumerate(genes):
                attr_name = sanitise_name(f"gene_{gene}", used_names)
                values[attr_name] = block[:, position].astype(np.float32)
                gene_map[attr_name] = str(gene)

    result = attach_attributes(
        store_path,
        values,
        keys=keys,
        level=level,
        key_attribute=key_attribute,
        missing=missing,
        fill_value=fill_value,
        overwrite=overwrite,
        shard_shape=shard_shape,
        progress=progress,
    )

    register_attached(
        store_path,
        obs_names=obs_map,
        gene_names=gene_map,
        categories=categories,
        dtypes=dtypes,
    )

    result.update(
        {
            "genes_attached": len(gene_map),
            "obs_columns_attached": len(obs_map),
            "source": str(h5ad_path),
        }
    )
    return result
