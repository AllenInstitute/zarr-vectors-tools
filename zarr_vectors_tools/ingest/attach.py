"""Attach attributes to an existing ZVF store, joined by key.

Ingest normally builds a store from one file. That breaks down when a
dataset is split across files that share cells but not rows — the Allen
Brain Cell Atlas MERFISH releases are the motivating case: coordinates in
one CSV, per-cell metadata in another, expression in a multi-gigabyte
``.h5ad`` with no ``obsm`` at all, and each file covering a *different*
subset of cells in a *different* order. Materialising one combined file
first is the expensive thing this module exists to avoid.

Instead the store is created once from whichever file carries the
coordinates, and every other file is *staged in* afterwards as additional
per-vertex attributes:

.. code-block:: text

    ccf_coordinates.csv  --ingest-->  store.zarr
    cell_metadata.csv    --attach-->  store.zarr   (join on cell_label)
    ...-log2.h5ad        --attach-->  store.zarr   (join on cell_label)

Two properties make this work at scale.

**The join.** ZVF orders vertices by spatial chunk, not by source row, and
the incoming file has its own order and its own cell subset. So each
vertex carries a *join key* attribute written at ingest, and attach maps
incoming rows onto vertices through it. Keys that are absent from the
incoming file get a fill value rather than failing the whole import.
:func:`hash_keys` reduces string identifiers to int64 because ZVF
attributes are numeric — and because the atlas's own ``cell_label`` is a
39-digit (128-bit) value that does not fit an integer column anyway.

**The memory bound.** Values are written one spatial chunk at a time via
core's per-chunk attribute API, so peak memory is one column of the
incoming file plus one chunk, never the whole store.

Attached columns are registered in the store's
:class:`~zarr_vectors_tools.headers.formats.H5ADHeader`, so
:func:`~zarr_vectors_tools.export.h5ad.export_h5ad` emits staged data
alongside data written at ingest, with categorical levels and original
column labels intact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from zarr_vectors.exceptions import IngestError

# Per-vertex attribute holding the 64-bit join key. Written at ingest by
# the ingesters that accept a key column; read by every attach.
DEFAULT_KEY_ATTRIBUTE = "zv_join_key"

# FNV-1a, 64-bit.
_FNV_OFFSET = np.uint64(0xCBF29CE484222325)
_FNV_PRIME = np.uint64(0x100000001B3)


def hash_keys(keys: Any) -> np.ndarray:
    """Reduce join keys to a stable ``int64`` array.

    Integer keys pass through unchanged. Everything else is hashed with
    FNV-1a over the UTF-8 bytes, vectorised across the whole array.

    The hash deliberately ignores the fixed-width zero padding that numpy
    byte-string arrays carry, so the same logical key hashes identically
    whether it arrived in an ``S39`` column or an ``S64`` one — otherwise
    two files describing the same cells would refuse to join.

    Args:
        keys: Array-like of identifiers (strings, bytes, or integers).

    Returns:
        ``(N,)`` int64 array of keys.
    """
    arr = np.asarray(keys)

    if arr.dtype.kind in "iub":
        return arr.astype(np.int64, copy=False)

    if arr.dtype.kind == "O":
        arr = arr.astype("U") if len(arr) else arr.astype("U1")
    if arr.dtype.kind == "U":
        arr = np.char.encode(arr, "utf-8")
    if arr.dtype.kind != "S":
        raise IngestError(
            f"join keys must be strings, bytes or integers, got dtype {arr.dtype}"
        )

    n = len(arr)
    width = arr.dtype.itemsize
    if n == 0 or width == 0:
        return np.zeros(n, dtype=np.int64)

    raw = np.ascontiguousarray(arr).view(np.uint8).reshape(n, width)
    # Real byte length per key: everything up to the zero padding.
    lengths = width - (raw[:, ::-1] != 0).argmax(axis=1)
    lengths = np.where((raw != 0).any(axis=1), lengths, 0)

    digest = np.full(n, _FNV_OFFSET, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for col in range(width):
            stepped = (digest ^ raw[:, col].astype(np.uint64)) * _FNV_PRIME
            digest = np.where(col < lengths, stepped, digest)

    return digest.view(np.int64)


def _build_lookup(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Sort incoming keys for lookup; report how many were duplicated."""
    order = np.argsort(keys, kind="stable")
    ordered = keys[order]
    duplicates = int(len(ordered) - len(np.unique(ordered))) if len(ordered) else 0
    return order, ordered, duplicates


def _fill_for(dtype: np.dtype, fill_value: Any) -> Any:
    """Coerce ``fill_value`` into something ``dtype`` can hold."""
    if fill_value is not None and not (
        isinstance(fill_value, float) and np.isnan(fill_value)
    ):
        return fill_value
    if dtype.kind == "f":
        return np.nan
    # NaN has no integer representation; -1 is the pandas convention for
    # "missing category code" and is out of range for any real code.
    return -1


def attach_attributes(
    store_path: str | Path,
    values: dict[str, np.ndarray],
    *,
    keys: Any | None = None,
    level: int = 0,
    key_attribute: str = DEFAULT_KEY_ATTRIBUTE,
    missing: str = "fill",
    fill_value: Any = np.nan,
    overwrite: bool = False,
    shard_shape: int | tuple[int, ...] | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Write new per-vertex attributes into an existing store.

    Args:
        store_path: Path to the ZVF store to add to (modified in place).
        values: Attribute name -> ``(M,)`` or ``(M, C)`` array of incoming
            values, all sharing the row order of ``keys``.
        keys: ``(M,)`` join keys for the rows of ``values``. Strings are
            hashed via :func:`hash_keys`. When None, the store's
            ``key_attribute`` is treated as a direct row index into
            ``values`` — the positional mode, for adding more columns from
            the file the store was built from.
        level: Resolution level to write into.
        key_attribute: Per-vertex attribute holding the join key (or the
            row index when ``keys`` is None).
        missing: ``"fill"`` (default) writes ``fill_value`` for vertices
            absent from the incoming file; ``"error"`` raises instead.
        fill_value: Value for unmatched vertices. Defaults to NaN for
            float attributes and ``-1`` for integer ones.
        overwrite: Allow replacing attributes that already exist.
        shard_shape: Create the new attribute arrays sharded, packing this
            many spatial chunks per axis into each storage object (an int
            broadcasts to every axis). Decide this *here* rather than
            resharding later: unsharded, a store costs one file per chunk
            per attribute, so staging a wide panel produces millions of
            files and repacking them means rereading every one. Match the
            value across attaches to keep the store uniform.
        progress: Print per-chunk progress — worth it on stores with many
            thousands of chunks.

    Returns:
        Summary dict with ``attributes_written``, ``vertices_matched``,
        ``vertices_unmatched``, ``chunk_count``, ``duplicate_keys`` and
        ``sharded``.

    Raises:
        IngestError: If the store, level or key attribute is missing, the
            shapes disagree, or ``missing="error"`` and a vertex is
            unmatched.
    """
    import zarr_vectors.building as B

    if not values:
        raise IngestError("attach_attributes requires at least one attribute")

    store_path = str(store_path)
    arrays = {name: np.asarray(v) for name, v in values.items()}

    lengths = {name: len(v) for name, v in arrays.items()}
    if len(set(lengths.values())) > 1:
        raise IngestError(f"attribute arrays have differing lengths: {lengths}")
    n_rows = next(iter(lengths.values()))

    if keys is not None:
        key_array = hash_keys(keys)
        if len(key_array) != n_rows:
            raise IngestError(
                f"keys has {len(key_array)} entries but the attribute arrays "
                f"have {n_rows}"
            )
        order, ordered_keys, duplicate_keys = _build_lookup(key_array)
    else:
        order = ordered_keys = None
        duplicate_keys = 0

    try:
        root = B.open_store(store_path, "a")
        level_group = B.get_resolution_level(root, level)
    except Exception as e:
        raise IngestError(f"cannot open store '{store_path}' at level {level}: {e}") from e

    existing = set(level_group.require_group(B.VERTEX_ATTRIBUTES).children())
    if key_attribute not in existing:
        raise IngestError(
            f"store has no join-key attribute {key_attribute!r} at level {level}. "
            f"Available attributes: {sorted(existing) or '(none)'}. Re-ingest the "
            f"store with a key column so later files can be joined to it."
        )
    clashes = sorted(set(arrays) & existing)
    if clashes and not overwrite:
        raise IngestError(
            f"attributes already exist: {clashes}. Pass overwrite=True to replace them."
        )

    chunk_keys = B.list_chunk_keys(level_group, "vertices")

    # Sharding has to be decided when the array is created: an unsharded
    # array is one storage object per spatial chunk, so a store staged
    # attribute-by-attribute ends up with chunks x attributes files, and
    # repacking that afterwards costs a full read-write of every one.
    # Creating the arrays inside a sharded write session avoids the repack.
    session = _write_session(B, root, level_group, level, shard_shape)

    with session:
        for name, data in arrays.items():
            channel_names = (
                [f"ch{i}" for i in range(data.shape[1])] if data.ndim == 2 else None
            )
            B.create_attribute_array(
                level_group, name, dtype=str(data.dtype),
                channel_names=channel_names, exist_ok=True,
            )

        fills = {name: _fill_for(data.dtype, fill_value) for name, data in arrays.items()}
        matched, unmatched = _write_all_chunks(
            B, level_group, chunk_keys, arrays, fills,
            key_attribute=key_attribute, order=order, ordered_keys=ordered_keys,
            n_rows=n_rows, missing=missing, progress=progress,
        )

    B.refresh_arrays_present(level_group)

    return {
        "attributes_written": sorted(arrays),
        "vertices_matched": matched,
        "vertices_unmatched": unmatched,
        "chunk_count": len(chunk_keys),
        "duplicate_keys": duplicate_keys,
        "rows_available": n_rows,
        "sharded": shard_shape is not None,
    }


def _write_session(B, root, level_group, level: int, shard_shape):
    """A sharded write session, or a no-op context when unsharded."""
    from contextlib import nullcontext

    if shard_shape is None:
        return nullcontext()

    root_meta = B.read_root_metadata(root)
    level_meta = B.read_level_metadata(root, level)
    if root_meta.bounds is None:
        raise IngestError(
            "cannot shard: the store has no recorded bounds, which are "
            "needed to size the chunk grid"
        )
    return B.open_write_session(
        level_group,
        shard_shape=shard_shape,
        bounds=(list(root_meta.bounds[0]), list(root_meta.bounds[1])),
        chunk_shape=B.get_level_chunk_shape(root_meta, level_meta),
    )


def _write_all_chunks(
    B, level_group, chunk_keys, arrays, fills, *,
    key_attribute, order, ordered_keys, n_rows, missing, progress,
) -> tuple[int, int]:
    """Join and write every chunk; returns ``(matched, unmatched)``."""
    matched = unmatched = 0

    for position, chunk_coords in enumerate(chunk_keys):
        key_groups = B.read_chunk_attributes(
            level_group, key_attribute, chunk_coords, dtype="int64"
        )
        if not key_groups:
            continue

        per_name: dict[str, list[np.ndarray]] = {name: [] for name in arrays}
        for group in key_groups:
            stored = np.asarray(group).ravel().astype(np.int64, copy=False)

            if order is None:
                rows = stored
                found = (rows >= 0) & (rows < n_rows)
                rows = np.clip(rows, 0, max(n_rows - 1, 0))
            elif len(ordered_keys):
                slot = np.searchsorted(ordered_keys, stored)
                slot = np.clip(slot, 0, len(ordered_keys) - 1)
                found = ordered_keys[slot] == stored
                rows = order[slot]
            else:
                found = np.zeros(len(stored), dtype=bool)
                rows = np.zeros(len(stored), dtype=np.int64)

            n_found = int(found.sum())
            matched += n_found
            unmatched += len(stored) - n_found

            if missing == "error" and n_found != len(stored):
                raise IngestError(
                    f"{len(stored) - n_found} vertex/vertices in chunk "
                    f"{chunk_coords} have no matching row in the incoming file. "
                    f"Pass missing='fill' to write a fill value instead."
                )

            for name, data in arrays.items():
                gathered = data[rows]
                if n_found != len(stored):
                    gathered = gathered.copy()
                    gathered[~found] = fills[name]
                per_name[name].append(gathered.astype(data.dtype, copy=False))

        for name, groups in per_name.items():
            B.write_chunk_attributes(
                level_group, name, chunk_coords, groups, dtype=arrays[name].dtype
            )

        if progress and (position + 1) % 500 == 0:
            print(f"  attach: {position + 1}/{len(chunk_keys)} chunks", flush=True)

    return matched, unmatched


def register_attached(
    store_path: str | Path,
    *,
    obs_names: dict[str, str] | None = None,
    gene_names: dict[str, str] | None = None,
    categories: dict[str, list[str]] | None = None,
    dtypes: dict[str, str] | None = None,
) -> None:
    """Record attached columns in the store's h5ad header.

    Keeps :func:`~zarr_vectors_tools.export.h5ad.export_h5ad` able to emit
    staged columns under their original labels — with categorical levels
    decoded and genes routed into ``X`` rather than ``obs`` — exactly as it
    does for columns written at ingest. Creates the header if the store has
    none (a store built from a CSV will not).

    Args:
        store_path: Path to the ZVF store.
        obs_names: Attribute name -> original ``obs`` column label.
        gene_names: Attribute name -> original ``var`` (gene) name.
        categories: Original column label -> ordered category labels.
        dtypes: Original column label -> source pandas dtype.
    """
    from zarr_vectors_tools.headers.formats import H5ADHeader
    from zarr_vectors_tools.headers.registry import HeaderRegistry

    registry = HeaderRegistry(str(store_path))
    try:
        header = registry.get("h5ad")
        if not isinstance(header, H5ADHeader):
            header = H5ADHeader()
    except Exception:
        header = H5ADHeader()

    for attr, label in (obs_names or {}).items():
        if attr not in header.obs_attrs:
            header.obs_attrs.append(attr)
            header.obs_names.append(label)
    for attr, label in (gene_names or {}).items():
        if attr not in header.gene_attrs:
            header.gene_attrs.append(attr)
            header.gene_names.append(label)

    header.categories.update(categories or {})
    header.dtypes.update(dtypes or {})

    registry.add("h5ad", header)
