"""What the TRK and TRX exporters share.

Both formats hold the same four things beside the positions -- per-point
values, per-streamline values, a reference image, and (TRX only) named
bundles -- and a store holds all four too.  The exporters used to write the
positions and nothing else, so a tractogram could go into a store but never
come back out to the tools that made it.  This module reads a level with
everything attached, and works out which coordinate space the positions are
in, so each exporter only has to lay the result out in its own format.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from zarr_vectors.exceptions import ExportError
from zarr_vectors.typing import ChunkCoords


@dataclass
class StreamlineLevel:
    """One level's streamlines, with the values attached to them."""

    streamlines: list[npt.NDArray[np.float32]]
    #: The store object each output streamline came from.  Repeats when
    #: ``chunks`` cropped one object into several runs.
    object_ids: list[int]
    #: ``{name: (V,) or (V, C)}``, rows in the order of the streamlines
    #: concatenated.
    vertex_attributes: dict[str, npt.NDArray[Any]] = field(default_factory=dict)
    #: ``{name: (S,) or (S, C)}``, one row per output streamline.
    object_attributes: dict[str, npt.NDArray[Any]] = field(default_factory=dict)
    level_group: Any = None

    @property
    def lengths(self) -> npt.NDArray[np.int64]:
        return np.asarray([len(s) for s in self.streamlines], dtype=np.int64)

    def per_streamline(self, values: npt.NDArray[Any]) -> list[npt.NDArray[Any]]:
        """Split a vertex-aligned array into one ``(N_i, C)`` array per streamline."""
        rows = values.reshape(len(values), -1)
        return np.split(rows, np.cumsum(self.lengths)[:-1])


def _is_numeric(values: npt.NDArray[Any]) -> bool:
    return np.issubdtype(values.dtype, np.number) or values.dtype == np.bool_


def _select(
    available: dict[str, npt.NDArray[Any]],
    requested: Sequence[str] | None,
    *,
    kind: str,
    level: int,
) -> tuple[dict[str, npt.NDArray[Any]], list[str]]:
    """Pick the attributes to write, and report what was left out.

    ``requested=None`` means every numeric one: a streamline format can only
    hold numbers, and text columns (labels stored as fixed-width bytes) are
    skipped rather than failing the export.  An explicit name that is
    absent, or is not numeric, is refused by name -- writing a file without
    the column someone asked for is the outcome to avoid.
    """
    if requested is None:
        chosen = {n: v for n, v in available.items() if _is_numeric(v)}
        return chosen, sorted(set(available) - set(chosen))
    missing = [n for n in requested if n not in available]
    if missing:
        raise ExportError(
            f"{kind} attribute(s) not present at level {level}: {missing}. "
            f"Available: {sorted(available) or '(none)'}"
        )
    text = [n for n in requested if not _is_numeric(available[n])]
    if text:
        raise ExportError(
            f"{kind} attribute(s) {text} are not numeric and cannot be written "
            f"to a streamline file"
        )
    return {n: available[n] for n in requested}, []


def read_streamline_level(
    store_path: str | Path,
    *,
    level: int,
    object_ids: list[int] | None,
    group_ids: list[int] | None,
    chunks: list[ChunkCoords] | None,
    attribute_names: Sequence[str] | None,
    object_attribute_names: Sequence[str] | None,
) -> tuple[StreamlineLevel, dict[str, list[str]]]:
    """Read one level's streamlines and the attributes to write with them.

    Returns the level and ``{"vertex": [...], "object": [...]}``, the
    attributes skipped because they are not numeric.
    """
    from zarr_vectors.building import (
        get_resolution_level,
        open_store,
        read_object_attributes,
    )
    from zarr_vectors.constants import OBJECT_ATTRIBUTES
    from zarr_vectors.types.polylines import read_polylines

    try:
        result = read_polylines(
            str(store_path), level=level, object_ids=object_ids,
            group_ids=group_ids, chunks=chunks,
        )
    except Exception as e:
        raise ExportError(f"Failed to read store '{store_path}': {e}") from e

    streamlines = [
        np.concatenate(segments, axis=0).astype(np.float32)
        for segments in result["polylines"]
    ]
    if not streamlines:
        raise ExportError("No streamlines to export")
    out_ids = [int(i) for i in result["object_ids"]]

    vertex, skipped_vertex = _select(
        dict(result.get("vertex_attributes") or {}), attribute_names,
        kind="per-vertex", level=level,
    )

    level_group = get_resolution_level(open_store(str(store_path)), level)
    try:
        object_names = list(level_group[OBJECT_ATTRIBUTES].children())
    except Exception:  # noqa: BLE001 - a level with no object attributes
        object_names = []
    wanted = object_names if object_attribute_names is None else [
        n for n in object_attribute_names if n in object_names
    ]
    rows = np.asarray(out_ids, dtype=np.int64)
    by_object: dict[str, npt.NDArray[Any]] = {}
    for name in wanted:
        values = np.asarray(read_object_attributes(level_group, name))
        by_object[name] = values[rows]
    if object_attribute_names is not None:
        # Names absent from the level have to be reported, not dropped, so
        # hand the full request to _select with only the present ones read.
        absent = [n for n in object_attribute_names if n not in object_names]
        if absent:
            raise ExportError(
                f"per-streamline attribute(s) not present at level {level}: "
                f"{absent}. Available: {sorted(object_names) or '(none)'}"
            )
    objects, skipped_object = _select(
        by_object, object_attribute_names, kind="per-streamline", level=level,
    )

    return (
        StreamlineLevel(
            streamlines=streamlines,
            object_ids=out_ids,
            vertex_attributes=vertex,
            object_attributes=objects,
            level_group=level_group,
        ),
        {"vertex": skipped_vertex, "object": skipped_object},
    )


def stored_space(store_path: str | Path) -> tuple[str, Any, Any]:
    """``(space, trk_header, trx_header)`` for a streamline store.

    ``space`` is ``"voxmm"`` when the positions are TrackVis voxel
    millimetres, as the parallel TRK ingest keeps them by default, and
    ``"rasmm"`` otherwise.  The TRK header says which when it was written
    by a current ingester; older stores fall back to the ``crs`` the
    parallel ingest stamps (``input_space``), and a store with neither is
    taken to be in RAS millimetres -- what TRX, TCK and nibabel give.
    """
    from zarr_vectors.building import open_store, read_root_metadata

    from zarr_vectors_tools.headers.registry import HeaderRegistry

    registry = HeaderRegistry(str(store_path))
    trk = registry.get("trk") if registry.has("trk") else None
    trx = registry.get("trx") if registry.has("trx") else None

    space = getattr(trk, "space", None)
    if space is None:
        try:
            crs = read_root_metadata(open_store(str(store_path))).crs or {}
        except Exception:  # noqa: BLE001 - no root metadata to consult
            crs = {}
        space = "voxmm" if str(crs.get("input_space", "")).lower() == "voxmm" else "rasmm"
    return space, trk, trx


def trk_reference(trk: Any) -> dict[str, Any]:
    """A nibabel TRK header dict from a stored :class:`TRKHeader`."""
    from nibabel.streamlines import Field

    return {
        Field.VOXEL_TO_RASMM: np.asarray(trk.affine, dtype=np.float32),
        Field.VOXEL_SIZES: tuple(float(v) for v in trk.voxel_size),
        Field.DIMENSIONS: tuple(int(v) for v in trk.dimensions),
        Field.VOXEL_ORDER: str(trk.voxel_order or "RAS").encode("latin1"),
    }


def trackvis_to_rasmm(trk: Any) -> npt.NDArray[np.float64]:
    """The affine from a TRK file's voxel-millimetre coordinates to RAS mm.

    Delegated to nibabel, which also accounts for the voxel order and the
    half-voxel offset between voxel corners and centres.
    """
    from nibabel.streamlines.trk import get_affine_trackvis_to_rasmm

    return np.asarray(get_affine_trackvis_to_rasmm(trk_reference(trk)), dtype=np.float64)
