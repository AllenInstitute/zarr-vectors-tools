"""Read a point level a batch at a time, in the order a whole-level read gives.

The point exporters used to read the whole level, build one array of every
row, and only then write.  At a few hundred million points that is several
times the level's size in memory before a byte reaches the file.  These
batches keep the exporters' output identical -- same rows, same order, same
filters -- while memory holds one batch:

- with no ``object_ids``, the level's chunks are walked in sorted order,
  which is the order ``read_points`` concatenates them in, a batch of chunks
  at a time;
- with ``object_ids``, the ids are taken a slice at a time in the order
  given, which is the order ``read_points`` reads objects in.

Each batch goes through ``read_points`` itself, so attribute handling is
exactly what the whole-level read did.  ``bbox`` is the exception: it is
applied here (see :func:`box_chunks`), because core's own bbox read returns
the wrong rows on some stores.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

#: Points per batch to aim for.  About 100 MB of float32 positions plus
#: attributes; an exporter holds one batch and its formatted output.
DEFAULT_VERTEX_BUDGET = 4_000_000
_OBJECTS_PER_BATCH = 50_000


def iter_point_batches(
    store_path: str | Path,
    *,
    level: int,
    bbox: Any,
    object_ids: Sequence[int] | None,
    chunks: Sequence[Sequence[int]] | None,
    attribute_names: Sequence[str] | None,
    vertex_budget: int = DEFAULT_VERTEX_BUDGET,
) -> Iterator[dict[str, Any]]:
    """Yield ``read_points`` results that together are the whole-level read."""
    from zarr_vectors.types.points import read_points

    require_object_index(store_path, level, object_ids)
    chunks = box_chunks(store_path, level, bbox, chunks)
    common = dict(level=level, attribute_names=attribute_names)
    if object_ids is not None:
        ids = [int(o) for o in object_ids]
        for start in range(0, len(ids), _OBJECTS_PER_BATCH):
            yield clip_to_box(read_points(
                str(store_path), object_ids=ids[start:start + _OBJECTS_PER_BATCH],
                chunks=None if chunks is None else list(chunks), **common,
            ), bbox)
        return

    keys, per_chunk = _level_chunks(store_path, level, chunks)
    if not keys:
        return
    step = max(1, int(vertex_budget // max(per_chunk, 1)))
    for start in range(0, len(keys), step):
        yield clip_to_box(
            read_points(str(store_path), chunks=keys[start:start + step], **common),
            bbox,
        )


def box_chunks(
    store_path: str | Path, level: int, bbox: Any,
    chunks: Sequence[Sequence[int]] | None,
) -> list[tuple[int, ...]] | None:
    """``chunks``, narrowed to the level's chunks that ``bbox`` touches.

    Workaround for core: ``read_points(bbox=...)`` returns the wrong rows
    on a store written with both object ids and a bin smaller than its
    chunk (0 of 88 points in a 250-unit chunk with 10-unit bins).  The
    exporters read the chunks the box touches instead and keep the rows
    inside it with :func:`clip_to_box` -- the same rows, in the same order,
    as a correct bbox read.  Remove once core's bbox read is fixed.
    """
    if bbox is None:
        return None if chunks is None else [tuple(int(c) for c in cc) for cc in chunks]
    from zarr_vectors.building import (
        get_level_chunk_shape,
        get_resolution_level,
        list_chunk_keys,
        open_store,
        read_level_metadata,
        read_root_metadata,
    )
    from zarr_vectors.constants import VERTICES

    root = open_store(str(store_path))
    edge = np.asarray(
        get_level_chunk_shape(read_root_metadata(root), read_level_metadata(root, level)),
        dtype=np.float64,
    )
    lo = np.floor(np.asarray(bbox[0], dtype=np.float64) / edge).astype(int)
    hi = np.floor(np.asarray(bbox[1], dtype=np.float64) / edge).astype(int)
    ndim = len(edge)
    keys = sorted(
        tuple(int(c) for c in key)
        for key in list_chunk_keys(get_resolution_level(root, level), VERTICES)
    )
    # The spatial coords are the key's last ``ndim`` entries: a store chunked
    # by an attribute puts its attribute bin first.
    inside = [
        key for key in keys
        if all(lo[d] <= key[len(key) - ndim + d] <= hi[d] for d in range(ndim))
    ]
    if chunks is not None:
        wanted = {tuple(int(c) for c in cc) for cc in chunks}
        inside = [key for key in inside if key in wanted]
    return inside


def clip_to_box(result: dict[str, Any], bbox: Any) -> dict[str, Any]:
    """Keep the rows of one ``read_points`` result that lie inside ``bbox`` (inclusive)."""
    if bbox is None:
        return result
    positions = np.asarray(result["positions"])
    keep = np.all(
        (positions >= np.asarray(bbox[0])) & (positions <= np.asarray(bbox[1])), axis=1,
    )
    if keep.all():
        return result
    out = dict(result)
    out["positions"] = positions[keep]
    for key in ("vertex_attributes", "attributes"):
        if isinstance(result.get(key), dict):
            out[key] = {n: np.asarray(v)[keep] for n, v in result[key].items()}
    ids = result.get("object_ids")
    if ids is not None and len(ids) == len(keep):
        out["object_ids"] = np.asarray(ids)[keep]
    if "vertex_count" in result:
        out["vertex_count"] = int(keep.sum())
    return out


def require_object_index(
    store_path: str | Path, level: int, object_ids: Sequence[int] | None,
) -> None:
    """Refuse an object filter on a level that records no objects.

    Core's object read fails on such a store with a bare ``KeyError``.
    """
    if object_ids is None:
        return
    from zarr_vectors.building import OBJECT_INDEX, get_resolution_level, open_store
    from zarr_vectors.exceptions import ExportError

    if OBJECT_INDEX not in get_resolution_level(open_store(str(store_path)), level):
        raise ExportError(
            f"this store has no object ids at level {level}, so there are no "
            f"objects to select (--object-id); export the whole level, or a "
            f"region of it with --bbox"
        )


def _level_chunks(
    store_path: str | Path, level: int, chunks: Sequence[Sequence[int]] | None,
) -> tuple[list[tuple[int, ...]], float]:
    """The level's chunk keys (limited to ``chunks``), sorted, and points per chunk."""
    from zarr_vectors.building import (
        get_resolution_level,
        list_chunk_keys,
        open_store,
        read_level_metadata,
    )
    from zarr_vectors.constants import VERTICES

    root = open_store(str(store_path))
    level_group = get_resolution_level(root, level)
    keys = sorted(tuple(int(c) for c in key) for key in list_chunk_keys(level_group, VERTICES))
    if chunks is not None:
        wanted = {tuple(int(c) for c in cc) for cc in chunks}
        keys = [key for key in keys if key in wanted]
    try:
        total = int(getattr(read_level_metadata(root, level), "vertex_count", 0) or 0)
    except Exception:  # noqa: BLE001 - a level without metadata
        total = 0
    all_keys = len(keys) if chunks is None else max(len(keys), 1)
    per_chunk = total / all_keys if total and all_keys else 1024.0
    return keys, float(per_chunk)


def ndim_of(store_path: str | Path) -> int:
    from zarr_vectors.building import open_store, read_root_metadata

    return int(read_root_metadata(open_store(str(store_path))).sid_ndim)


def position_names(store_path: str | Path, ndim: int) -> list[str]:
    """Column names for a level's positions.

    The source's own when the ingest recorded them (a CSV's header, a
    table's coordinate columns), else the store's axis names, else
    ``x, y, z``.
    """
    try:
        from zarr_vectors_tools.headers.registry import HeaderRegistry

        registry = HeaderRegistry(str(store_path))
        for fmt, field in (("csv", "position_columns"), ("h5ad", "position_names")):
            if registry.has(fmt):
                names = getattr(registry.get(fmt), field, None) or []
                if _usable(names, ndim):
                    return [str(n) for n in names]
    except Exception:  # noqa: BLE001 - an unreadable header: use the axes
        pass
    try:
        from zarr_vectors.building import open_store, read_root_metadata

        axes = read_root_metadata(open_store(str(store_path))).spatial_index_dims
        names = [axis.get("name") for axis in axes]
        if _usable(names, ndim):
            return [str(n) for n in names]
    except Exception:  # noqa: BLE001 - no readable axes
        pass
    return ["x", "y", "z"][:ndim] if ndim <= 3 else [f"dim{i}" for i in range(ndim)]


def _usable(names: Sequence[Any], ndim: int) -> bool:
    return (
        len(names) == ndim and all(isinstance(n, str) and n for n in names)
        and len(set(names)) == ndim
    )


def require_attributes(
    store_path: str | Path, level: int, attribute_names: Sequence[str] | None,
) -> None:
    """Refuse requested attributes the level does not have, before any row is written."""
    from zarr_vectors.building import get_resolution_level, open_store
    from zarr_vectors.constants import VERTEX_ATTRIBUTES
    from zarr_vectors.exceptions import ExportError

    if not attribute_names:
        return
    level_group = get_resolution_level(open_store(str(store_path)), level)
    try:
        present = set(level_group[VERTEX_ATTRIBUTES].children())
    except Exception:  # noqa: BLE001 - a level with no vertex attributes
        present = set()
    missing = [n for n in attribute_names if n not in present]
    if missing:
        raise ExportError(
            f"attribute(s) not present at level {level}: {missing}. "
            f"Available: {sorted(present) or '(none)'}"
        )


def attribute_columns(
    result: dict[str, Any], attribute_names: Sequence[str] | None, level: int,
) -> dict[str, np.ndarray]:
    """The requested attributes of one batch, refusing any that are absent."""
    from zarr_vectors.exceptions import ExportError

    # ``read_points`` returns the per-vertex arrays under ``vertex_attributes``;
    # the fallback keeps older cores working.
    attrs = result.get("vertex_attributes")
    if attrs is None:
        attrs = result.get("attributes", {})
    missing = [n for n in (attribute_names or []) if n not in attrs]
    if missing and len(result.get("positions", ())):
        raise ExportError(
            f"attribute(s) not present at level {level}: {missing}. "
            f"Available: {sorted(attrs) or '(none)'}"
        )
    return {n: attrs[n] for n in (attribute_names or []) if n in attrs}
