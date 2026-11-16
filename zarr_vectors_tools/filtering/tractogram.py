"""A streamline set held whole in memory, the input to both engines.

:class:`Tractogram` is deliberately plain: one ``(V, 3)`` float64 vertex
array, ``offsets`` cutting it into ``N`` streamlines, a dict of
per-streamline columns, and the group memberships.  Row ``k`` is object id
``k``, as in the store.  Nothing here imports the store layer, so the class
(and the engines on top of it) also runs where zarr does not, such as a
Pyodide worker handed arrays from JavaScript.

:func:`load_tractogram` fills one from a store's level 0.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["Tractogram", "load_tractogram"]

#: Segments handled per batch wherever every segment is visited, so the
#: float64 temporaries stay near 100 MB on a whole-brain tractogram.
_BLOCK = 1 << 22

Bools = npt.NDArray[np.bool_]


@dataclass
class Tractogram:
    """Streamlines as flat arrays.

    Attributes:
        vertices: ``(V, 3)`` world positions, streamline after streamline.
            Kept float32 when given float32 (as stores hold them); every
            test widens to float64 first, so the answer is the same.
        offsets: ``(N + 1,)`` int64; streamline ``k`` is
            ``vertices[offsets[k]:offsets[k + 1]]``.  A streamline may be
            empty (an object id with no geometry).
        columns: Per-streamline values, ``(N,)`` or ``(N, C)``.  ``length``
            (path length) and ``n_vertices`` are computed from the geometry
            when not supplied.
        present: For a column with absent rows, a ``(N,)`` bool mask that is
            False where the value is absent.  NaNs count as absent too.
        groups: Member rows of each group, indexed by group id.
    """

    vertices: npt.NDArray[np.floating]
    offsets: npt.NDArray[np.int64]
    columns: dict[str, npt.NDArray[Any]] = field(default_factory=dict)
    present: dict[str, npt.NDArray[np.bool_]] = field(default_factory=dict)
    groups: list[npt.NDArray[np.int64]] = field(default_factory=list)

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices)
        if vertices.dtype not in (np.float32, np.float64):
            vertices = vertices.astype(np.float64)
        self.vertices = np.ascontiguousarray(vertices).reshape(-1, 3)
        self.columns = dict(self.columns)
        self.present = dict(self.present)
        self.offsets = np.asarray(self.offsets, dtype=np.int64)
        if (
            self.offsets.ndim != 1 or len(self.offsets) == 0 or self.offsets[0] != 0
            or self.offsets[-1] != len(self.vertices) or np.any(np.diff(self.offsets) < 0)
        ):
            raise ValueError("offsets must rise from 0 to the vertex count")
        n = self.n_objects
        counts = np.diff(self.offsets)
        if "n_vertices" not in self.columns:
            self.columns["n_vertices"] = counts.copy()
        if "length" not in self.columns:
            self.columns["length"] = self._lengths()
        for name, values in self.columns.items():
            values = np.asarray(values)
            if len(values) != n:
                raise ValueError(f"column {name!r} has {len(values)} rows for {n} streamlines")
            self.columns[name] = values
        self.groups = [np.asarray(g, dtype=np.int64) for g in self.groups]

    @classmethod
    def from_polylines(
        cls,
        polylines: Sequence[npt.ArrayLike],
        *,
        columns: Mapping[str, npt.ArrayLike] | None = None,
        present: Mapping[str, npt.ArrayLike] | None = None,
        groups: Sequence[Sequence[int]] | Mapping[int, Sequence[int]] | None = None,
    ) -> Tractogram:
        """Build one from a list of ``(n_k, 3)`` arrays (row ``k`` is the ``k``-th)."""
        arrays = [np.asarray(p, dtype=np.float64).reshape(-1, 3) for p in polylines]
        offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(p) for p in arrays])
        vertices = np.concatenate(arrays) if arrays else np.zeros((0, 3))
        if isinstance(groups, Mapping):
            size = max(groups, default=-1) + 1
            groups = [list(groups.get(g, ())) for g in range(size)]
        return cls(
            vertices, offsets,
            columns={k: np.asarray(v) for k, v in (columns or {}).items()},
            present={k: np.asarray(v, dtype=bool) for k, v in (present or {}).items()},
            groups=[np.asarray(g, dtype=np.int64) for g in (groups or [])],
        )

    @property
    def n_objects(self) -> int:
        return len(self.offsets) - 1

    def polyline(self, row: int) -> npt.NDArray[np.float64]:
        return self.vertices[self.offsets[row]:self.offsets[row + 1]]

    def edges(self) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """``(start, end, owner)``: vertex indices and row of every segment.

        A one-vertex streamline contributes one zero-length segment, so it is
        tested as the point it is.  The arrays are int32 while the counts fit,
        which halves them on a whole-brain tractogram.
        """
        index = np.int32 if len(self.vertices) < 2**31 - 1 else np.int64
        counts = np.diff(self.offsets)
        # Repeat counts must be intp, which is 32-bit under Pyodide.
        per = np.where(counts >= 2, counts - 1, (counts == 1).astype(np.int64)).astype(np.intp)
        total = int(per.sum())
        owner = np.repeat(np.arange(self.n_objects, dtype=index), per)
        # start = first vertex of the row + position within the row; a
        # one-vertex row's zero-length segment ends where it starts.
        start = np.arange(total, dtype=np.int64)
        start -= np.repeat(np.cumsum(per) - per - self.offsets[:-1], per)
        start = start.astype(index)
        end = start + np.repeat(counts >= 2, per).astype(index)
        return start, end, owner

    def endpoints(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], Bools]:
        """``(first, last, has_vertices)``: each row's end vertices.

        Rows without vertices get NaN ends and ``has_vertices`` False.
        """
        counts = np.diff(self.offsets)
        has = counts > 0
        first = np.full((self.n_objects, 3), np.nan)
        last = np.full((self.n_objects, 3), np.nan)
        first[has] = self.vertices[self.offsets[:-1][has]]
        last[has] = self.vertices[self.offsets[1:][has] - 1]
        return first, last, has

    def column(self, name: str, channel: int | None = None) -> tuple[npt.NDArray[Any], Bools]:
        """``(values, present)`` for a column (one channel of a multi-channel one).

        Raises:
            KeyError: If there is no such column.
        """
        if name not in self.columns:
            raise KeyError(f"no column {name!r}; have {sorted(self.columns)}")
        values = self.columns[name]
        if values.ndim == 2:
            if channel is None:
                if values.shape[1] != 1:
                    raise ValueError(f"column {name!r} has {values.shape[1]} channels; pick one")
                channel = 0
            values = values[:, channel]
        elif channel not in (None, 0):
            raise ValueError(f"column {name!r} has one channel, not {channel + 1}")
        present = self.present.get(name)
        present = np.ones(len(values), dtype=bool) if present is None else present.copy()
        if values.dtype.kind == "f":
            present &= ~np.isnan(values)
        return values, present

    def group_mask(self, ids: Sequence[int]) -> npt.NDArray[np.bool_]:
        """Rows in any of the groups ``ids`` (a group the set lacks has no rows)."""
        out = np.zeros(self.n_objects, dtype=bool)
        for gid in ids:
            if 0 <= gid < len(self.groups):
                out[self.groups[gid]] = True
        return out

    def _lengths(self) -> npt.NDArray[np.float64]:
        start, end, owner = self.edges()
        out = np.zeros(self.n_objects)
        for lo in range(0, len(start), _BLOCK):
            s, e = start[lo:lo + _BLOCK], end[lo:lo + _BLOCK]
            step = np.linalg.norm(
                self.vertices[e].astype(np.float64) - self.vertices[s], axis=1,
            )
            out += np.bincount(owner[lo:lo + _BLOCK], weights=step, minlength=self.n_objects)
        return out


def load_tractogram(
    store_path: str | Path,
    *,
    level: int = 0,
    columns: Sequence[str] | None = None,
) -> Tractogram:
    """Read a streamline store's level into a :class:`Tractogram`, in RAS mm.

    Positions are mapped to RAS millimetres exactly as
    :func:`~zarr_vectors_tools.algorithms.select_streamlines` maps them (a
    TRK store kept in voxel millimetres goes through its header's affine),
    so regions mean the same thing to both.

    Filtering is defined at level 0.  A coarse level drops streamlines and
    moves vertices, so ``level`` is for showing how much that changes the
    answer, not for answering.

    Args:
        store_path: A streamline (or polyline) store.
        level: Resolution level to read.
        columns: Object attributes to load; default all of them.

    Raises:
        ValueError: If the store does not hold streamlines.
        NotImplementedError: If the level is part of an additive pyramid,
            whose complete content spans several levels.
    """
    from zarr_vectors.building import (
        get_resolution_level,
        object_count,
        open_store,
        read_all_groupings,
        read_object_attribute_present_mask,
        read_object_attributes,
        read_root_metadata,
    )
    from zarr_vectors.constants import OBJECT_ATTRIBUTES
    from zarr_vectors.types.polylines import read_polylines

    from zarr_vectors_tools.convert.export._streamlines import stored_space, trackvis_to_rasmm

    root = open_store(str(store_path))
    root_meta = read_root_metadata(root)
    geometry = set(root_meta.geometry_types or [])
    if not geometry & {"streamline", "polyline"}:
        raise ValueError(f"{store_path} holds {sorted(geometry) or 'no geometry'}, not streamlines")
    if "additive_levels" in (root_meta.format_capabilities or []):
        raise NotImplementedError(
            f"{store_path} is an additive pyramid: a level's streamlines span it and every "
            "coarser level, and reading that chain is not implemented here"
        )
    level_group = get_resolution_level(root, level)
    n = object_count(level_group)

    result = read_polylines(str(store_path), level=level)
    pieces: list[npt.NDArray[np.float64] | None] = [None] * n
    for oid, segments in zip(result["object_ids"], result["polylines"]):
        oid = int(oid)
        if oid >= n:
            n = oid + 1
            pieces.extend([None] * (n - len(pieces)))
        pieces[oid] = np.concatenate(segments, axis=0).astype(np.float64)
    empty = np.zeros((0, 3))
    arrays = [p if p is not None else empty for p in pieces]

    space, trk, _trx = stored_space(store_path)
    if space == "voxmm" and trk is not None:
        to_ras = trackvis_to_rasmm(trk)
        arrays = [p @ to_ras[:3, :3].T + to_ras[:3, 3] if len(p) else p for p in arrays]

    loaded: dict[str, npt.NDArray[Any]] = {}
    present: dict[str, npt.NDArray[np.bool_]] = {}
    try:
        names = sorted(level_group[OBJECT_ATTRIBUTES].children())
    except Exception:  # noqa: BLE001 - a level without object attributes
        names = []
    for name in names if columns is None else [c for c in columns if c in names]:
        values = np.asarray(read_object_attributes(level_group, name))
        if len(values) != n:
            continue
        loaded[name] = values
        mask = read_object_attribute_present_mask(level_group, name)
        if mask is not None:
            present[name] = np.asarray(mask, dtype=bool)

    try:
        groups = [np.asarray(list(g), dtype=np.int64) for g in read_all_groupings(level_group)]
    except Exception:  # noqa: BLE001 - a level without groups
        groups = []
    return Tractogram.from_polylines(arrays, columns=loaded, present=present, groups=groups)
