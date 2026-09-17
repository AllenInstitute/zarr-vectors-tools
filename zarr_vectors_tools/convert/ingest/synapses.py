"""Synapse tables joined to an EM skeleton store by segment id.

CAVE exports a connectome's synapses as a table: one row per synapse, with
the pre- and post-synaptic segment (``pre_pt_root_id``, ``post_pt_root_id``)
and positions in voxels (``ctr_pt_position`` and friends).  A skeleton store
built by the precomputed ingesters keeps each object's segment id, so the
two join without any spatial matching:

- the synapses become a point store whose object ids are the skeleton
  store's -- synapse object 12 is skeleton object 12, the same neuron -- so
  picking either in a viewer names the same segment;
- each skeleton object gains ``synapse_pre_count`` and ``synapse_post_count``;
- the synapse store's objects carry the same ``segment_id`` attribute (0 for
  the object holding unmatched synapses), so
  :class:`~zarr_vectors_tools.algorithms.segment_link.SegmentLink` finds a
  segment in either store.

A synapse has two segments and a point store gives each point one object, so
``side`` chooses which one owns it (the post-synaptic side by default: "the
inputs of this neuron").  The other side's segment id is kept on every point
as ``pre_segment_id`` or ``post_segment_id``, and the synapse id as the join
key, so later tables can be attached to the synapses with ``zvtools attach``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from zarr_vectors.exceptions import IngestError

from zarr_vectors_tools.convert.ingest.attach import DEFAULT_KEY_ATTRIBUTE

_SIDES = ("pre", "post")
_UNMATCHED = ("keep", "drop", "error")
_POSITION_SPLIT = re.compile(r"[\s,]+")


def ingest_synapses(
    table_path: str | Path,
    skeleton_store: str | Path,
    output_path: str | Path,
    chunk_shape: Sequence[float] | None = None,
    *,
    side: str = "post",
    position: str = "ctr_pt_position",
    resolution: Sequence[float] = (1.0, 1.0, 1.0),
    pre_column: str = "pre_pt_root_id",
    post_column: str = "post_pt_root_id",
    id_column: str | None = "id",
    columns: Sequence[str] | None = None,
    unmatched: str = "keep",
    write_counts: bool = True,
    chunksize: int = 1_000_000,
) -> dict[str, Any]:
    """Write a synapse table as a point store keyed to a skeleton store's objects.

    Args:
        table_path: The synapse table: CSV/TSV (``.csv``, ``.tsv``, ``.txt``),
            or Parquet / Feather when ``pyarrow`` is installed.
        skeleton_store: A store with ``object_attributes/segment_id`` at
            level 0, as the precomputed skeleton ingesters write.
        output_path: The point store to create.
        chunk_shape: Chunk size, in nanometres.  Default: the skeleton
            store's, so the two stores' chunks line up.
        side: ``"post"`` or ``"pre"`` -- whose segment owns each synapse.
        position: The position column: either one column holding
            ``"[x y z]"`` strings, or three columns ``<position>_x/_y/_z``.
        resolution: Nanometres per voxel of the position columns
            (FlyWire and MICrONS tables are 4, 4, 40).
        pre_column, post_column: The segment id columns.
        id_column: The synapse id column, stored as the join key; ``None``
            when the table has none.
        columns: Further numeric columns to keep as point attributes, e.g.
            ``["size"]``.
        unmatched: What to do with a synapse whose owning segment is not in
            the skeleton store: ``"keep"`` it as one extra object (id one past
            the skeleton store's last), ``"drop"`` it, or raise (``"error"``).
        write_counts: Write ``synapse_pre_count`` / ``synapse_post_count`` to
            every level of the skeleton store, replacing earlier counts.
        chunksize: Table rows read at a time.

    Returns:
        Summary with ``rows_read``, ``synapses_written``, ``unmatched``,
        ``dropped``, ``objects`` (skeleton objects with at least one
        synapse), ``unassigned_object`` (the extra object id, or ``None``)
        and the point writer's own counts.

    Raises:
        IngestError: For a skeleton store without segment ids, a table
            missing a named column, an unknown ``side`` or ``unmatched``, or
            unmatched synapses with ``unmatched="error"``.
    """
    from zarr_vectors.types.points import write_points

    if side not in _SIDES:
        raise IngestError(f"side must be one of {_SIDES}, got {side!r}")
    if unmatched not in _UNMATCHED:
        raise IngestError(f"unmatched must be one of {_UNMATCHED}, got {unmatched!r}")
    scale = np.asarray(resolution, dtype=np.float64).reshape(-1)
    if scale.shape != (3,):
        raise IngestError(f"resolution needs three values, x y z, got {list(resolution)}")

    segment_ids, skeleton_chunk_shape = _skeleton_segments(skeleton_store)
    n_objects = len(segment_ids)
    owner_column = post_column if side == "post" else pre_column
    other_side = "pre" if side == "post" else "post"
    other_column = pre_column if side == "post" else post_column

    positions: list[npt.NDArray] = []
    owners: list[npt.NDArray] = []
    others: list[npt.NDArray] = []
    keys: list[npt.NDArray] = []
    extras: dict[str, list[npt.NDArray]] = {name: [] for name in (columns or [])}
    pre_counts = np.zeros(n_objects, dtype=np.int64)
    post_counts = np.zeros(n_objects, dtype=np.int64)
    rows_read = n_unmatched = n_dropped = 0

    for frame in _read_frames(Path(table_path), chunksize):
        needed = [pre_column, post_column, *(columns or [])]
        if id_column is not None:
            needed.append(id_column)
        _require_columns(frame, needed, position, table_path)
        rows_read += len(frame)

        pre_oid = _object_ids(frame[pre_column], segment_ids)
        post_oid = _object_ids(frame[post_column], segment_ids)
        pre_counts += np.bincount(pre_oid[pre_oid >= 0], minlength=n_objects)
        post_counts += np.bincount(post_oid[post_oid >= 0], minlength=n_objects)

        owner = post_oid if side == "post" else pre_oid
        missing = owner < 0
        n_unmatched += int(missing.sum())
        if missing.any() and unmatched == "error":
            sample = frame[owner_column].to_numpy()[missing][:5].tolist()
            raise IngestError(
                f"{int(missing.sum())} synapse(s) in this batch have a {side}-synaptic "
                f"segment the skeleton store does not hold, e.g. {sample}; pass "
                f"unmatched='keep' or 'drop'"
            )
        keep = ~missing if unmatched == "drop" else np.ones(len(frame), dtype=bool)
        n_dropped += int((~keep).sum())
        owner = np.where(missing, n_objects, owner)[keep]

        positions.append((_positions(frame, position) * scale)[keep].astype(np.float32))
        owners.append(owner.astype(np.int64))
        others.append(frame[other_column].to_numpy(dtype=np.int64)[keep])
        if id_column is not None:
            keys.append(frame[id_column].to_numpy(dtype=np.int64)[keep])
        for name in extras:
            extras[name].append(frame[name].to_numpy(dtype=np.float64)[keep])

    if not positions or sum(len(p) for p in positions) == 0:
        raise IngestError(f"{table_path} has no synapses to write")

    vertex_attributes: dict[str, npt.NDArray] = {
        f"{other_side}_segment_id": np.concatenate(others),
    }
    if id_column is not None:
        vertex_attributes[DEFAULT_KEY_ATTRIBUTE] = np.concatenate(keys)
    for name, parts in extras.items():
        vertex_attributes[name] = np.concatenate(parts).astype(np.float32)

    all_owners = np.concatenate(owners)
    summary = dict(write_points(
        str(output_path),
        np.concatenate(positions, axis=0),
        chunk_shape=tuple(float(c) for c in (chunk_shape or skeleton_chunk_shape)),
        object_ids=all_owners,
        vertex_attributes=vertex_attributes,
    ))

    _write_segment_ids(output_path, segment_ids, int(all_owners.max()) + 1)
    if write_counts:
        _write_counts(skeleton_store, pre_counts, post_counts)

    summary.update({
        "rows_read": rows_read,
        "synapses_written": int(len(all_owners)),
        "unmatched": n_unmatched,
        "dropped": n_dropped,
        "side": side,
        "objects": int(np.count_nonzero(np.bincount(all_owners, minlength=n_objects)[:n_objects])),
        "unassigned_object": n_objects if (unmatched == "keep" and n_unmatched) else None,
        "counts_written": bool(write_counts),
    })
    return summary


def _skeleton_segments(store: str | Path) -> tuple[npt.NDArray[np.int64], tuple[float, ...]]:
    """The skeleton store's segment id per object, and its chunk shape."""
    from zarr_vectors.building import (
        get_resolution_level,
        open_store,
        read_object_attributes,
        read_root_metadata,
    )

    root = open_store(str(store))
    try:
        segments = np.asarray(
            read_object_attributes(get_resolution_level(root, 0), "segment_id"),
        ).reshape(-1)
    except Exception as exc:  # noqa: BLE001 - reported as the store not qualifying
        raise IngestError(
            f"{store} has no object_attributes/segment_id at level 0; synapses join "
            f"to skeleton stores written by the precomputed ingesters"
        ) from exc
    segments = segments.astype(np.int64)
    if len(segments) > 1 and np.any(np.diff(segments) < 0):
        raise IngestError(f"{store}'s segment ids are not sorted, so they cannot be searched")
    return segments, tuple(float(c) for c in read_root_metadata(root).chunk_shape)


def _object_ids(column: Any, segment_ids: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Object id of each segment, ``-1`` where the store does not hold it."""
    values = np.asarray(column, dtype=np.int64)
    if len(segment_ids) == 0:
        return np.full(len(values), -1, dtype=np.int64)
    where = np.searchsorted(segment_ids, values)
    clipped = np.minimum(where, len(segment_ids) - 1)
    found = segment_ids[clipped] == values
    return np.where(found, clipped, -1).astype(np.int64)


def _positions(frame: Any, position: str) -> npt.NDArray[np.float64]:
    """``(N, 3)`` positions from a bracket-string column or three split columns."""
    split = [f"{position}_{axis}" for axis in "xyz"]
    if all(name in frame.columns for name in split):
        return frame[split].to_numpy(dtype=np.float64)
    text = frame[position].astype(str).str.strip().str.strip("[]()").str.strip()
    parts = text.str.split(_POSITION_SPLIT, expand=True)
    if parts.shape[1] != 3:
        raise IngestError(
            f"column {position!r} should hold three numbers per row, like '[x y z]'; "
            f"found {parts.shape[1]}"
        )
    return parts.astype(np.float64).to_numpy()


def _require_columns(frame: Any, needed: Sequence[str], position: str, table: Any) -> None:
    missing = [name for name in needed if name not in frame.columns]
    split = [f"{position}_{axis}" for axis in "xyz"]
    if position not in frame.columns and not all(name in frame.columns for name in split):
        missing.append(f"{position} (or {', '.join(split)})")
    if missing:
        raise IngestError(
            f"{table} is missing column(s) {missing}; it has {list(frame.columns)}"
        )


def _read_frames(path: Path, chunksize: int) -> Iterator[Any]:
    """The table a block of rows at a time."""
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise IngestError("Parquet synapse tables need pyarrow: pip install pyarrow") from exc
        for batch in pq.ParquetFile(str(path)).iter_batches(batch_size=chunksize):
            yield batch.to_pandas()
        return
    if suffix in (".feather", ".arrow"):
        yield pd.read_feather(str(path))
        return
    separator = "\t" if suffix == ".tsv" else ","
    yield from pd.read_csv(str(path), sep=separator, chunksize=chunksize)


def _write_segment_ids(
    store: str | Path, segment_ids: npt.NDArray[np.int64], n_rows: int,
) -> None:
    """Give the synapse store's objects the skeleton store's segment ids.

    An object attribute's row ``i`` is object ``i``, so the column runs to
    the highest owner, ``n_rows - 1``, and may stop short of the skeleton
    store's last object; the unassigned object, one past the skeleton
    store's, gets segment id 0.  With the ids on both stores, either can be
    searched by segment without the other.
    """
    from zarr_vectors.building import (
        create_object_attributes_array,
        get_resolution_level,
        open_store,
        write_object_attributes,
    )

    level0 = get_resolution_level(open_store(str(store), mode="r+"), 0)
    ids = np.zeros(n_rows, dtype=np.uint64)
    shared = min(len(ids), len(segment_ids))
    ids[:shared] = segment_ids[:shared]
    create_object_attributes_array(level0, "segment_id", dtype="uint64")
    write_object_attributes(level0, "segment_id", ids)


def _write_counts(
    store: str | Path, pre_counts: npt.NDArray[np.int64], post_counts: npt.NDArray[np.int64],
) -> None:
    """Replace ``synapse_pre_count`` / ``synapse_post_count`` at every level.

    Object ids are the same at every level of a skeleton store, so the level-0
    counts belong on each of them; writing them everywhere keeps a coarse
    level's picking and selection in step with level 0.  A rerun overwrites,
    so the counts always describe the last table ingested.
    """
    from zarr_vectors.building import (
        create_object_attributes_array,
        get_resolution_level,
        list_resolution_levels,
        open_store,
        write_object_attributes,
    )

    root = open_store(str(store), mode="r+")
    for level in list_resolution_levels(root):
        level_group = get_resolution_level(root, level)
        for name, counts in (("synapse_pre_count", pre_counts),
                             ("synapse_post_count", post_counts)):
            create_object_attributes_array(level_group, name)
            write_object_attributes(level_group, name, counts)
