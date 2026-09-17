"""Export zarr vectors streamlines to TRX format.

Requires ``trx-python``: ``pip install trx-python``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from zarr_vectors.exceptions import ExportError
from zarr_vectors.typing import ChunkCoords

from zarr_vectors_tools.convert.export._streamlines import (
    read_streamline_level,
    stored_space,
    trackvis_to_rasmm,
)


def export_trx(
    store_path: str | Path,
    output_path: str | Path,
    *,
    level: int = 0,
    object_ids: list[int] | None = None,
    group_ids: list[int] | None = None,
    chunks: list[ChunkCoords] | None = None,
    attribute_names: Sequence[str] | None = None,
    object_attribute_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Export zarr vectors streamlines to a TRX file.

    Writes what a TRX file can hold beside the positions: per-vertex
    attributes as ``dpv``, per-object attributes as ``dps``, the level's
    named object groups as ``groups`` with their group attributes as
    ``dpg``, and the reference image from the store's TRX (or TRK) header.
    TRX positions are RAS millimetres, so a store still in TrackVis voxel
    millimetres is converted on the way out.

    Args:
        store_path: Path to the zarr vectors store.
        output_path: Path for the output .trx file.
        level: Resolution level to export.
        object_ids: Optional object ID filter.
        group_ids: Optional group ID filter.  Also limits the groups
            written to these.
        chunks: Optional whitelist of chunk coordinate tuples. Filters at
            the *segment* level: only vertex groups stored in listed
            chunks are emitted, and each surviving contiguous run is
            written as its own streamline. The output ``streamline_count``
            can therefore exceed the source object count.
        attribute_names: Per-vertex attributes to write.  ``None``
            (default) writes every numeric one; ``[]`` writes none.
        object_attribute_names: Per-object attributes to write, with the
            same ``None`` / ``[]`` meaning.

    Returns:
        Summary dict with ``streamline_count``, ``vertex_count``,
        ``attributes_carried``, ``object_attributes_carried``,
        ``groups_carried``, ``space`` (of the stored positions), and
        ``attributes_skipped`` when non-numeric attributes were left out.

    Raises:
        ExportError: If trx-python is not installed, a requested attribute
            is missing or not numeric, or export fails.
    """
    try:
        from nibabel.streamlines.array_sequence import ArraySequence
        from trx.trx_file_memmap import TrxFile
        from trx.trx_file_memmap import save as trx_save
    except ImportError as e:
        raise ExportError(
            "trx-python is required for TRX export. "
            "Install with: pip install trx-python"
        ) from e

    data, skipped = read_streamline_level(
        store_path, level=level, object_ids=object_ids, group_ids=group_ids,
        chunks=chunks, attribute_names=attribute_names,
        object_attribute_names=object_attribute_names,
    )
    space, trk, trx_header = stored_space(store_path)

    # Build positions + offsets + lengths arrays (TRX layout).  All three are
    # required: the reader slices ``_data`` by ``_lengths``, so a file written
    # with the zero-filled default comes back as N empty streamlines.
    all_positions = np.concatenate(data.streamlines, axis=0).astype(np.float32)
    if space == "voxmm":
        if trk is None or trk.vox_to_ras is None:
            raise ExportError(
                "the store's positions are TrackVis voxel millimetres but it "
                "has no TRK header to convert them to RAS with"
            )
        to_rasmm = trackvis_to_rasmm(trk)
        all_positions = (
            all_positions @ to_rasmm[:3, :3].T + to_rasmm[:3, 3]
        ).astype(np.float32)
    lengths = data.lengths.astype(np.uint32)
    offsets = np.concatenate([[0], np.cumsum(lengths[:-1])]).astype(np.uint32)

    if trx_header is not None and trx_header.affine is not None:
        reference, dims = trx_header.affine, list(trx_header.dimensions)
    elif trk is not None and trk.vox_to_ras is not None:
        reference, dims = trk.affine, list(trk.dimensions)
    else:
        reference, dims = np.eye(4), [1, 1, 1]

    groups, group_data = _groups(data, group_ids)

    n_streamlines = len(data.streamlines)
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        trx = TrxFile(
            nb_vertices=len(all_positions),
            nb_streamlines=n_streamlines,
        )
        trx.header["VOXEL_TO_RASMM"] = np.asarray(reference, dtype=np.float64).tolist()
        trx.header["DIMENSIONS"] = [int(d) for d in dims]
        # REPLACE the pre-allocated arrays rather than assigning into them.
        # ``TrxFile`` allocates ``_data`` as **float16**, so writing into it
        # silently quantises every coordinate (measured: 7.8e-4 mm of error on
        # ordinary RASmm positions).  Rebinding hands the saver the float32
        # buffer instead, and the round-trip is then exact.
        trx.streamlines._data = np.ascontiguousarray(
            all_positions, dtype=np.float32,
        )
        trx.streamlines._offsets = offsets
        trx.streamlines._lengths = lengths

        def _per_vertex(values: npt.NDArray[Any]) -> ArraySequence:
            sequence = ArraySequence()
            sequence._data = np.ascontiguousarray(values.reshape(len(values), -1))
            sequence._offsets = offsets.astype(np.int64)
            sequence._lengths = lengths.astype(np.int64)
            return sequence

        trx.data_per_vertex = {
            name: _per_vertex(values) for name, values in data.vertex_attributes.items()
        }
        # TRX decides an array's width from its shape, so a single column
        # stays one-dimensional and anything wider is (rows, columns).
        trx.data_per_streamline = {
            name: np.ascontiguousarray(
                values if values.ndim == 1 else values.reshape(len(values), -1)
            )
            for name, values in data.object_attributes.items()
        }
        trx.groups = groups
        trx.data_per_group = group_data
        # Module-level ``save``, not ``TrxFile.save`` -- there is no such
        # method, so every export raised AttributeError, wrapped as
        # "Failed to write TRX" by the except below and never noticed
        # because the test asserted only that *something* was raised.
        trx_save(trx, str(output_path))
        trx.close()
    except Exception as e:
        raise ExportError(f"Failed to write TRX '{output_path}': {e}") from e

    summary: dict[str, Any] = {
        "streamline_count": n_streamlines,
        "vertex_count": len(all_positions),
        "attributes_carried": sorted(data.vertex_attributes),
        "object_attributes_carried": sorted(data.object_attributes),
        "groups_carried": sorted(groups),
        "space": space,
    }
    left_out = sorted(skipped["vertex"] + skipped["object"])
    if left_out:
        summary["attributes_skipped"] = left_out
    return summary


def _groups(
    data: Any, group_ids: list[int] | None,
) -> tuple[dict[str, npt.NDArray[np.uint32]], dict[str, dict[str, npt.NDArray[Any]]]]:
    """The level's groups, as TRX ``groups`` and ``dpg``.

    A group lists output streamline indices, not store object ids, so each
    member is mapped through the streamlines actually written; under a
    ``chunks`` crop one object can be several streamlines and all of them
    are listed.  A group with no exported member is left out -- TRX has no
    use for an empty one.  Group attributes become that group's ``dpg``,
    skipping values that are NaN, which is how the TRX ingest fills a key
    a group did not have.
    """
    from zarr_vectors.building import read_all_groupings, read_groupings_attributes
    from zarr_vectors.constants import GROUP_ATTRIBUTES, GROUPS

    level_group = data.level_group
    try:
        memberships = read_all_groupings(level_group)
        names = list(level_group.read_array_meta(GROUPS).get("group_names") or [])
    except Exception:  # noqa: BLE001 - a level with no groups
        return {}, {}

    rows_of: dict[int, list[int]] = {}
    for row, oid in enumerate(data.object_ids):
        rows_of.setdefault(int(oid), []).append(row)

    try:
        attribute_names = sorted(level_group[GROUP_ATTRIBUTES].children())
    except Exception:  # noqa: BLE001 - no group attributes
        attribute_names = []
    attributes = {}
    for name in attribute_names:
        try:
            attributes[name] = np.asarray(read_groupings_attributes(level_group, name))
        except Exception:  # noqa: BLE001 - unreadable column: members still export
            continue

    wanted = set(range(len(memberships))) if group_ids is None else set(group_ids)
    groups: dict[str, npt.NDArray[np.uint32]] = {}
    group_data: dict[str, dict[str, npt.NDArray[Any]]] = {}
    for gid, members in enumerate(memberships):
        if gid not in wanted:
            continue
        rows = sorted(r for oid in members for r in rows_of.get(int(oid), ()))
        if not rows:
            continue
        name = str(names[gid]) if gid < len(names) and names[gid] else f"group_{gid}"
        groups[name] = np.asarray(rows, dtype=np.uint32)
        for attr, values in attributes.items():
            if gid >= len(values) or not np.issubdtype(values.dtype, np.number):
                continue
            row = np.asarray(values[gid]).reshape(1, -1)
            if np.issubdtype(row.dtype, np.floating) and np.isnan(row).all():
                continue
            group_data.setdefault(name, {})[attr] = row
    return groups, group_data
