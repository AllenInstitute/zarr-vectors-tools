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

Each batch goes through ``read_points`` itself, so ``bbox`` and attribute
handling are exactly what the whole-level read did.
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

    common = dict(level=level, bbox=bbox, attribute_names=attribute_names)
    if object_ids is not None:
        ids = [int(o) for o in object_ids]
        for start in range(0, len(ids), _OBJECTS_PER_BATCH):
            yield read_points(
                str(store_path), object_ids=ids[start:start + _OBJECTS_PER_BATCH],
                chunks=None if chunks is None else list(chunks), **common,
            )
        return

    keys, per_chunk = _level_chunks(store_path, level, chunks)
    if not keys:
        return
    step = max(1, int(vertex_budget // max(per_chunk, 1)))
    for start in range(0, len(keys), step):
        yield read_points(str(store_path), chunks=keys[start:start + step], **common)


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
