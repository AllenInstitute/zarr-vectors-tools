"""Find one segment in several stores: its skeleton, its mesh, its synapses.

EM stores keep the segment id of each object as the ``segment_id`` object
attribute: the precomputed skeleton and mesh ingests write it, and so does
the synapse join, whose objects are numbered like its skeleton store's.
Nothing else links the stores.  A segment id is the key they share, so a
segment picked in the mesh store can be found in the skeleton store, with
its metrics, and in the synapse store.

The lookup is by value, not by position, and no store is assumed to be
sorted: a skeleton pyramid's coarser level zeroes the segment id of each
object sparsity dropped, and a synapse store's unassigned object has
segment id 0.  A segment is *present* at a level when its object there has
geometry; one that sparsity dropped resolves to ``None``, the same as one
the store never held.

Levels without their own ``segment_id`` answer with level 0's, since the
mesh and point pyramid builders keep object ids.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

SEGMENT_ID_ATTR = "segment_id"


@dataclass(frozen=True)
class _Index:
    """One store level's segment ids, searchable."""

    segments: npt.NDArray[np.uint64]     # indexed by object id
    order: npt.NDArray[np.int64]         # argsort of segments
    ranked: npt.NDArray[np.uint64]       # segments[order]

    def object_ids(self, segment_ids: npt.NDArray[np.uint64]) -> npt.NDArray[np.int64]:
        """Object id per segment id, ``-1`` where the level does not list it."""
        pos = np.searchsorted(self.ranked, segment_ids)
        inside = pos < len(self.ranked)
        hit = np.zeros(len(segment_ids), dtype=bool)
        hit[inside] = self.ranked[pos[inside]] == segment_ids[inside]
        out = np.full(len(segment_ids), -1, dtype=np.int64)
        out[hit] = self.order[pos[hit]]
        # Segment 0 is background: the synapse join's unassigned object.
        out[segment_ids == 0] = -1
        return out


def store_segment_ids(store: str | Path, *, level: int = 0) -> npt.NDArray[np.uint64]:
    """The segment id of each object at ``level``, indexed by object id.

    Raises:
        ValueError: The store has no ``segment_id`` at ``level`` or level 0.
    """
    from zarr_vectors.building import get_resolution_level, open_store, read_object_attributes

    root = open_store(str(store))
    for source in dict.fromkeys((level, 0)):
        try:
            values = read_object_attributes(get_resolution_level(root, source), SEGMENT_ID_ATTR)
        except Exception:  # noqa: BLE001 - absent here; try level 0
            continue
        return np.asarray(values).reshape(-1).astype(np.uint64)
    raise ValueError(
        f"{store} has no object_attributes/{SEGMENT_ID_ATTR} at level {level} or 0; "
        f"segment ids come from the precomputed ingests and the synapse join"
    )


class SegmentLink:
    """Resolve segment ids across stores that share them.

    Args:
        stores: ``{name: path}``, or paths, which are then named by their
            file name without extension.
        level: The default level to look at.

    Each store's segment ids are read once per level, on first use.
    """

    def __init__(
        self,
        stores: Mapping[str, str | Path] | Sequence[str | Path],
        *,
        level: int = 0,
    ) -> None:
        if isinstance(stores, Mapping):
            named = {str(name): Path(path) for name, path in stores.items()}
        else:
            named = {}
            for path in stores:
                name = Path(path).stem
                if name in named:
                    raise ValueError(
                        f"two stores are both named {name!r}; pass {{name: path}} instead"
                    )
                named[name] = Path(path)
        if not named:
            raise ValueError("no stores to link")
        self.stores: dict[str, Path] = named
        self.level = int(level)
        self._indices: dict[tuple[str, int], _Index] = {}

    # ---------------------------------------------------------------
    # Lookups
    # ---------------------------------------------------------------

    def resolve(self, segment_id: int, *, level: int | None = None) -> dict[str, int | None]:
        """``{store name: object id, or None}`` for one segment."""
        frame = self.resolve_many([segment_id], level=level)
        row = frame.iloc[0]
        return {name: (None if row[name] is None or _is_na(row[name]) else int(row[name]))
                for name in self.stores}

    def resolve_many(self, segment_ids: Sequence[int], *, level: int | None = None) -> Any:
        """A table of object ids: one row per segment id, one column per store.

        Returns:
            A ``pandas.DataFrame`` indexed by ``segment_id``, with nullable
            ``Int64`` columns; ``<NA>`` where a store does not hold the
            segment at the level, or holds it without geometry there.
        """
        import pandas as pd

        level = self.level if level is None else int(level)
        wanted = np.asarray([int(s) for s in segment_ids], dtype=np.uint64)
        columns = {}
        for name in self.stores:
            oids = self._index(name, level).object_ids(wanted)
            oids = self._with_geometry(name, level, oids)
            columns[name] = pd.array(np.where(oids >= 0, oids, 0), dtype="Int64")
            columns[name][oids < 0] = pd.NA
        return pd.DataFrame(columns, index=pd.Index(wanted.astype(np.int64), name="segment_id"))

    def segment_of(self, store: str, object_id: int, *, level: int | None = None) -> int | None:
        """The segment id of ``object_id`` in ``store``; ``None`` for none or 0."""
        level = self.level if level is None else int(level)
        segments = self._index(self._name(store), level).segments
        if not 0 <= int(object_id) < len(segments):
            return None
        segment = int(segments[int(object_id)])
        return segment or None

    def link(
        self, store: str, object_id: int, *, level: int | None = None,
    ) -> dict[str, int | None]:
        """The object in every store with the same segment as ``object_id`` in ``store``.

        The picking case: an object id chosen in one store (a mesh clicked
        in the viewer) becomes the matching object id in each of the others.
        ``store`` maps to ``object_id`` itself when it has geometry there.
        """
        segment = self.segment_of(store, object_id, level=level)
        if segment is None:
            return {name: None for name in self.stores}
        return self.resolve(segment, level=level)

    def attributes(
        self,
        segment_id: int,
        names: Mapping[str, Sequence[str]] | None = None,
        *,
        level: int = 0,
    ) -> dict[str, dict[str, Any]]:
        """Each store's object attributes for one segment.

        Args:
            segment_id: The segment.
            names: ``{store name: attribute names}``.  Default: every object
                attribute each store has at ``level``, but ``segment_id``.
            level: Level to read from.  Metrics usually live at level 0.

        Returns:
            ``{store name: {attribute: value}}`` for the stores holding the
            segment; a multi-column attribute's value is an array.
        """
        from zarr_vectors.building import (
            OBJECT_ATTRIBUTES,
            get_resolution_level,
            open_store,
            read_object_attributes,
        )

        found = self.resolve(segment_id, level=level)
        out: dict[str, dict[str, Any]] = {}
        for name, oid in found.items():
            if oid is None:
                continue
            level_group = get_resolution_level(open_store(str(self.stores[name])), level)
            if names is not None and name in names:
                wanted = list(names[name])
            elif names is not None:
                continue
            else:
                try:
                    wanted = sorted(level_group[OBJECT_ATTRIBUTES].children())
                except Exception:  # noqa: BLE001 - a level with no object attributes
                    wanted = []
                wanted = [w for w in wanted if w != SEGMENT_ID_ATTR]
            values: dict[str, Any] = {}
            for attribute in wanted:
                try:
                    column = np.asarray(read_object_attributes(level_group, attribute))
                except Exception as e:  # noqa: BLE001 - named in the error below
                    raise KeyError(
                        f"{name} has no object attribute {attribute!r} at level {level}"
                    ) from e
                value = column[oid]
                values[attribute] = value.item() if np.ndim(value) == 0 else value
            out[name] = values
        return out

    # ---------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------

    def _name(self, store: str) -> str:
        if store not in self.stores:
            raise KeyError(f"no store named {store!r}; linked stores are {sorted(self.stores)}")
        return store

    def _index(self, name: str, level: int) -> _Index:
        key = (name, level)
        if key not in self._indices:
            segments = store_segment_ids(self.stores[name], level=level)
            order = np.argsort(segments, kind="stable")
            self._indices[key] = _Index(segments, order, segments[order])
        return self._indices[key]

    def _with_geometry(
        self, name: str, level: int, oids: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.int64]:
        """``oids`` with ``-1`` for objects that have no geometry at ``level``."""
        from zarr_vectors.building import get_resolution_level, open_store, read_object_manifests

        candidates = sorted({int(o) for o in oids if o >= 0})
        if not candidates:
            return oids
        level_group = get_resolution_level(open_store(str(self.stores[name])), level)
        # Ids the level does not declare are simply absent from the result.
        manifests = read_object_manifests(level_group, ids=candidates)
        held = np.asarray([o for o in candidates if manifests.get(o)], dtype=np.int64)
        return np.where(np.isin(oids, held), oids, -1)


def _is_na(value: Any) -> bool:
    try:
        import pandas as pd

        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
