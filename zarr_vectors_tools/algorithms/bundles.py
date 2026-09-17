"""Per-bundle summaries of a streamline store, stored as group attributes.

A tract analysis starts from a handful of numbers per bundle: how many
streamlines it holds, how long they are, how far they wander from a straight
line, and where they begin and end.  The per-object enrichments (``length``,
``start``, ``end``; see
:mod:`zarr_vectors_tools.convert.ingest._polyline_enrichments`) give those per
streamline, and nothing aggregated them per group.  :func:`bundle_summary`
does, and writes one row per group under ``group_attributes/bundle_*`` so the
numbers travel with the store: TRX export carries them as ``dpg``, a pyramid
build carries them up the levels, and :func:`read_bundle_summary` reads them
back at any level.

Which level the numbers describe
--------------------------------
By default the statistics are computed once, from level 0, and the same rows
are written to every level.  A coarse level is a sparsified, simplified view
of the same bundles, and a viewer showing level 3 still wants to say "this is
the arcuate, 2,400 streamlines, 94 mm on average" rather than quote whatever
survived thinning.  The cost of that choice is that a coarse level's
``bundle_streamline_count`` is the level-0 count, not the number of members
that level lists.  Pass ``level=k`` to summarise level ``k`` from its own
membership and its own geometry instead; those rows are written to level
``k`` only.  Which level a level's rows came from is recorded on its
``groups`` array metadata under ``bundle_summary``, and
:func:`read_bundle_summary` reports it.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from zarr_vectors.building import (
    create_groupings_attributes_array,
    get_resolution_level,
    list_resolution_levels,
    object_count,
    open_store,
    read_all_groupings,
    read_chunk_vertices,
    read_groupings_attributes,
    read_object_attribute_present_mask,
    read_object_attributes,
    read_object_manifests,
    read_root_metadata,
    write_groupings_attributes,
)
from zarr_vectors.constants import (
    GEOM_POLYLINE,
    GEOM_STREAMLINE,
    GROUP_ATTRIBUTES,
    GROUPS,
    OBJECT_ATTRIBUTES,
    VERTEX_FRAGMENTS,
    VERTICES,
)
from zarr_vectors.exceptions import ArrayError

from zarr_vectors_tools.convert.ingest._polyline_enrichments import (
    compute_endpoints,
    compute_lengths,
    tortuosity_from_endpoints,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = ["ATTRIBUTE_PREFIX", "bundle_summary", "read_bundle_summary"]

#: Prefix of every group attribute this module writes.  Ingested per-group
#: data (a TRX file's ``dpg``) lands in the same ``group_attributes/``
#: namespace under the file's own key names, so an unprefixed ``length_mean``
#: could overwrite a column the user brought with them.
ATTRIBUTE_PREFIX = "bundle_"

#: Key on each level's ``groups`` array metadata recording where that level's
#: rows came from.
PROVENANCE_KEY = "bundle_summary"

#: One value per group.
_SCALAR_COLUMNS = (
    "streamline_count",
    "length_mean",
    "length_std",
    "length_min",
    "length_median",
    "length_max",
    "tortuosity_mean",
)
#: One ``D``-vector per group.
_CENTROID_COLUMNS = ("start_centroid", "end_centroid")

_ORIENTATION_PRINCIPAL = "principal_axis"
_ORIENTATION_STORED = "as_stored"


def bundle_summary(
    store_path: str | Path,
    *,
    level: int | None = None,
    write: bool = True,
    orient_endpoints: bool = True,
    use_stored_attributes: bool = True,
    batch_size: int = 4096,
) -> pd.DataFrame:
    """Summarise every object group of a streamline store, one row per group.

    Per group: the streamline count; path length mean, standard deviation,
    minimum, median and maximum; mean tortuosity (path length over the
    straight-line distance between the endpoints, 1.0 for a streamline whose
    endpoints coincide, as in
    :func:`~zarr_vectors_tools.convert.ingest._polyline_enrichments.tortuosity_from_endpoints`);
    and the centroid of the streamlines' start points and of their end points.

    The standard deviation is the population one (``ddof=0``), so a
    one-streamline bundle reports 0 rather than NaN.  A group with no members
    reports a count of 0 and NaN for everything else.  A streamline in several
    groups counts toward each of them; one listed twice in the same group
    counts once.

    Endpoint orientation.  Tractography does not give a streamline a
    direction: half of a bundle's streamlines may be stored running from
    cortex to brainstem and half the other way, and averaging start points as
    stored puts both centroids near the middle of the bundle.  With
    ``orient_endpoints=True`` (the default) each bundle's streamlines are
    first turned to agree: the bundle's axis is the principal eigenvector of
    the scatter of its ``end - start`` vectors (a statistic that does not
    depend on how each streamline happens to be stored), its sign is chosen
    so that its largest-magnitude component is positive, and any streamline
    whose ``end - start`` points against it is reversed.  ``start_*`` is
    therefore the end of the bundle lying lower along its dominant axis --
    for a bundle running mostly along ``z``, the inferior end.  A streamline
    whose endpoints coincide, or that lies exactly perpendicular to the axis,
    is left as stored; a bundle whose endpoint vectors have no dominant
    direction gets a split that is well defined but not anatomically
    meaningful.  Pass ``orient_endpoints=False`` for directed streamlines,
    whose stored start is the real one.

    Where the per-streamline numbers come from.  At level 0, when the store
    holds the ``length``, ``start`` and ``end`` object attributes the ingest
    enrichments write (``compute_length=True, compute_endpoints=True``), those
    are used and no geometry is read; rows those attributes mark absent are
    measured from geometry.  Everywhere else lengths and endpoints are
    measured from the level's own vertices.  Stored attributes are not
    consulted above level 0 because the coarseners carry object attributes
    up the pyramid verbatim: a ``length`` at level 2 is a level-0 length, and
    using it would describe level-0 geometry while claiming level 2.

    Memory.  With stored attributes: the three object-attribute columns,
    read whole (about 28 bytes per object for float32 columns).  From
    geometry: one batch of ``batch_size`` streamlines at a time plus every
    chunk that batch touches, decoded once per batch.  A bundle is spatially
    compact, so a batch drawn from one touches few chunks; a group spanning
    the whole store (a "whole brain" group) can touch most chunks in every
    batch, which is slow rather than large, but in the worst case one batch
    decodes the whole level.  Either way, each group's per-streamline
    lengths and endpoints (``8 * (2D + 1)`` bytes a member) are held while
    that group is aggregated, because the median needs all of them; and the
    memberships of every group are held as core's
    :func:`~zarr_vectors.building.read_all_groupings` returns them.  A
    streamline in several groups is measured once per group.

    Args:
        store_path: Path to a streamline or polyline store with object groups.
        level: ``None`` (default) computes from level 0 and, with ``write``,
            writes the rows to every level of the store.  An integer computes
            from that level's own membership and geometry and writes to that
            level only.
        write: When True, store the rows as ``group_attributes/bundle_*`` (see
            Returns for the names) and record the source level on the
            ``groups`` array metadata.  Re-running overwrites them.
        orient_endpoints: Orient each bundle's streamlines consistently
            before averaging endpoints (see above).  False averages them as
            stored.
        use_stored_attributes: Use level 0's ``length`` / ``start`` / ``end``
            object attributes when all three are present.  False always
            measures geometry -- for a store whose ``length`` attribute means
            something else, such as one carried in from a file's own
            per-streamline data.
        batch_size: Streamlines read per batch when measuring geometry.

    Returns:
        A :class:`pandas.DataFrame` with one row per group, in group-id order,
        indexed by group name (``group_<id>`` for an unnamed group, as core's
        group catalogue names it) and with columns ``group_id``,
        ``streamline_count``, ``length_mean``, ``length_std``,
        ``length_min``, ``length_median``, ``length_max``,
        ``tortuosity_mean``, then ``start_<axis>`` and ``end_<axis>`` for
        each spatial axis.  ``DataFrame.attrs`` holds ``source_level``,
        ``endpoint_orientation`` (``"principal_axis"`` or ``"as_stored"``),
        ``measured_from`` (``"attributes"``, ``"geometry"`` or both) and
        ``levels_written``.  The group attributes written are
        ``bundle_streamline_count`` (int64), ``bundle_length_mean``,
        ``bundle_length_std``, ``bundle_length_min``,
        ``bundle_length_median``, ``bundle_length_max``,
        ``bundle_tortuosity_mean`` (float64, ``(G,)``) and
        ``bundle_start_centroid``, ``bundle_end_centroid`` (float64,
        ``(G, D)``).

    Raises:
        ValueError: If the store is not a streamline or polyline store, the
            level does not exist, the level has no object groups, a group
            lists an object the level does not hold, a stored ``length`` /
            ``start`` / ``end`` attribute has the wrong shape, ``batch_size``
            is below 1, or -- with ``level=None`` and ``write=True`` -- some
            level does not carry the same groups as level 0.
    """
    if int(batch_size) < 1:
        raise ValueError(f"batch_size must be at least 1 (got {batch_size})")

    root = open_store(str(store_path), mode="r")
    root_meta = read_root_metadata(root)
    _require_polylines(root_meta, store_path)
    levels = list_resolution_levels(root)
    source_level = 0 if level is None else _require_level(level, levels, store_path)

    level_group = get_resolution_level(root, source_level)
    groupings = _require_groupings(level_group, source_level, store_path)
    n_groups = len(groupings)
    targets = list(levels) if level is None else [source_level]
    if write:
        # Checked before anything is measured, so a store that cannot take
        # the rows fails in a second rather than after an hour of reading.
        _require_same_groups(root, targets, n_groups, store_path)

    names = _group_names(level_group, n_groups)
    ndim = int(root_meta.sid_ndim)
    measurer = _Measurer(
        level_group,
        level=source_level,
        ndim=ndim,
        batch_size=int(batch_size),
        stored=(
            _read_stored_measures(level_group, ndim)
            if use_stored_attributes and source_level == 0
            else None
        ),
    )

    columns = _empty_columns(n_groups, ndim)
    for gid in range(n_groups):
        members = _member_ids(groupings[gid], names[gid], measurer.n_objects, source_level)
        # Let the membership list go as soon as it has been turned into an
        # array; a whole-tractogram group can be most of the memory held.
        groupings[gid] = []
        lengths, starts, ends = measurer.measure(members, names[gid])
        _aggregate_into(columns, gid, lengths, starts, ends, orient=orient_endpoints)

    orientation = _ORIENTATION_PRINCIPAL if orient_endpoints else _ORIENTATION_STORED
    table = _table(columns, names, _axis_names(root_meta, ndim))
    table.attrs["source_level"] = source_level
    table.attrs["endpoint_orientation"] = orientation
    table.attrs["measured_from"] = sorted(measurer.sources)
    table.attrs["levels_written"] = []

    if write:
        provenance = {
            "source_level": source_level,
            "endpoint_orientation": orientation,
            "attributes": [ATTRIBUTE_PREFIX + c for c in (*_SCALAR_COLUMNS, *_CENTROID_COLUMNS)],
        }
        # A second, writable handle: the read path above opens mode="r".
        writable = open_store(str(store_path), mode="r+")
        for target in targets:
            _write_columns(get_resolution_level(writable, target), columns, provenance)
        table.attrs["levels_written"] = list(targets)
    return table


def read_bundle_summary(store_path: str | Path, *, level: int = 0) -> pd.DataFrame:
    """Read back the rows :func:`bundle_summary` wrote at one level.

    Args:
        store_path: Path to the store.
        level: Resolution level to read.

    Returns:
        The same table :func:`bundle_summary` returns.  ``DataFrame.attrs``
        holds ``source_level`` and ``endpoint_orientation`` as recorded when
        the rows were written, or ``None`` for each when the level's rows
        were carried there by a pyramid build, which copies group attributes
        but not the ``groups`` array's other metadata.

    Raises:
        ValueError: If the level does not exist, has no object groups, holds
            no bundle summary, or holds one whose row count no longer matches
            its groups.
    """
    root = open_store(str(store_path), mode="r")
    root_meta = read_root_metadata(root)
    level = _require_level(level, list_resolution_levels(root), store_path)
    level_group = get_resolution_level(root, level)
    n_groups = _group_count(level_group)
    if not n_groups:
        raise ValueError(
            f"level {level} of {store_path} has no object groups, so it holds no "
            "bundle summary"
        )

    names = [ATTRIBUTE_PREFIX + c for c in (*_SCALAR_COLUMNS, *_CENTROID_COLUMNS)]
    missing = [
        n for n in names
        if not level_group.standalone_array_exists(f"{GROUP_ATTRIBUTES}/{n}")
    ]
    if missing:
        raise ValueError(
            f"level {level} of {store_path} holds no bundle summary (missing group "
            f"attributes {missing}); run bundle_summary on the store first"
        )

    ndim = int(root_meta.sid_ndim)
    columns: dict[str, npt.NDArray[Any]] = {}
    for column in (*_SCALAR_COLUMNS, *_CENTROID_COLUMNS):
        values = np.asarray(read_groupings_attributes(level_group, ATTRIBUTE_PREFIX + column))
        expected = (n_groups,) if column in _SCALAR_COLUMNS else (n_groups, ndim)
        if values.shape != expected:
            # The groups were rewritten after the summary was: the rows no
            # longer line up with the groups they were computed for.
            raise ValueError(
                f"group attribute {ATTRIBUTE_PREFIX + column!r} at level {level} has "
                f"shape {values.shape} but the level has {n_groups} groups; the "
                "summary is stale, run bundle_summary again"
            )
        columns[column] = values

    table = _table(columns, _group_names(level_group, n_groups), _axis_names(root_meta, ndim))
    try:
        provenance = level_group.read_array_meta(GROUPS).get(PROVENANCE_KEY) or {}
    except Exception:  # noqa: BLE001 - provenance is informational
        provenance = {}
    table.attrs["source_level"] = provenance.get("source_level")
    table.attrs["endpoint_orientation"] = provenance.get("endpoint_orientation")
    return table


# ---------------------------------------------------------------------------
# Store checks
# ---------------------------------------------------------------------------


def _require_polylines(root_meta: Any, store_path: str | Path) -> None:
    types = list(getattr(root_meta, "geometry_types", None) or [])
    if not {GEOM_STREAMLINE, GEOM_POLYLINE} & set(types):
        raise ValueError(
            f"bundle_summary reads streamline and polyline stores; {store_path} "
            f"holds {types}"
        )


def _require_level(level: int, levels: Sequence[int], store_path: str | Path) -> int:
    # ``operator.index`` rather than ``int``: ``int(1.5)`` is 1, and a caller
    # passing 1.5 has made a mistake worth hearing about.
    try:
        index = operator.index(level)
    except TypeError:
        index = None
    if index is None or isinstance(level, bool):
        raise ValueError(f"level must be an integer or None (got {level!r})")
    if index not in levels:
        raise ValueError(
            f"{store_path} has no level {index}; its levels are {list(levels)}"
        )
    return int(index)


def _require_groupings(
    level_group: Any, level: int, store_path: str | Path,
) -> list[Sequence[int]]:
    try:
        groupings = list(read_all_groupings(level_group))
    except Exception:  # noqa: BLE001 - a level with no taxonomy at all
        groupings = []
    if not groupings:
        raise ValueError(
            f"level {level} of {store_path} has no object groups, so there are no "
            "bundles to summarise. Ingest a file that names its bundles (a TRX "
            "file's groups), or build the taxonomy first with "
            "zarr_vectors_tools.compose.derive_groups"
        )
    return groupings


def _group_count(level_group: Any) -> int:
    """How many groups a level declares, or 0 when it has no ``groups`` array."""
    try:
        meta = level_group.read_array_meta(GROUPS)
    except Exception:  # noqa: BLE001
        return 0
    if not meta:
        return 0
    count = meta.get("num_groups")
    if count is None:
        try:
            return len(read_all_groupings(level_group))
        except Exception:  # noqa: BLE001
            return 0
    return int(count)


def _require_same_groups(
    root: Any, levels: Sequence[int], n_groups: int, store_path: str | Path,
) -> None:
    """Refuse to write rows onto a level whose group ids mean something else.

    Group attributes are addressed by row, so rows computed against level 0's
    groups are only right on a level whose ``groups`` array has the same rows.
    Every coarsener here preserves group ids; a level with no groups, or a
    different number of them, was built some other way.
    """
    for lvl in levels:
        count = _group_count(get_resolution_level(root, lvl))
        if count != n_groups:
            have = f"{count} groups" if count else "no groups"
            raise ValueError(
                f"level {lvl} of {store_path} has {have} where the summarised level "
                f"has {n_groups}, so the rows cannot be written there. Rebuild the "
                "pyramid (build_pyramid carries groups to every level), or pass an "
                "explicit level to summarise and write that level only"
            )


def _group_names(level_group: Any, n_groups: int) -> list[str]:
    """One name per row, falling back the way core's group catalogue does.

    Using the same ``group_<id>`` fallback means a row of this table is found
    under the same name by ``zarr_vectors.open(...).level(i).groups[name]``.
    """
    try:
        declared = list(level_group.read_array_meta(GROUPS).get("group_names") or [])
    except Exception:  # noqa: BLE001 - an unnamed store is still summarisable
        declared = []
    return [
        str(declared[i]) if i < len(declared) and declared[i] else f"group_{i}"
        for i in range(n_groups)
    ]


def _axis_names(root_meta: Any, ndim: int) -> list[str]:
    dims = list(getattr(root_meta, "spatial_index_dims", None) or [])
    names = [str(d.get("name")) if isinstance(d, dict) and d.get("name") else "" for d in dims]
    if len(names) != ndim or "" in names or len(set(names)) != ndim:
        return [str(i) for i in range(ndim)]
    return names


def _member_ids(
    members: Sequence[int], name: str, n_objects: int, level: int,
) -> npt.NDArray[np.int64]:
    if isinstance(members, range):
        ids = np.arange(members.start, members.stop, dtype=np.int64)
    else:
        ids = np.unique(np.asarray(members, dtype=np.int64))
    if len(ids):
        bad = ids[(ids < 0) | (ids >= n_objects)]
        if len(bad):
            raise ValueError(
                f"group {name!r} lists object ids the level {level} object index "
                f"does not hold ({n_objects} objects): {bad[:5].tolist()}"
            )
    return ids


# ---------------------------------------------------------------------------
# Per-streamline measures
# ---------------------------------------------------------------------------


class _StoredMeasures:
    """Level 0's ``length`` / ``start`` / ``end`` columns, and which rows hold one."""

    __slots__ = ("length", "start", "end", "present")

    def __init__(
        self,
        length: npt.NDArray[Any],
        start: npt.NDArray[Any],
        end: npt.NDArray[Any],
        present: npt.NDArray[np.bool_],
    ) -> None:
        self.length = length
        self.start = start
        self.end = end
        self.present = present


def _read_stored_measures(level_group: Any, ndim: int) -> _StoredMeasures | None:
    """The stored enrichments when all three are present, else ``None``.

    Only all three together avoid reading geometry -- a length without
    endpoints still needs the vertices, which give the length for free -- so a
    partial set is not worth the extra reads.
    """
    try:
        available = set(level_group[OBJECT_ATTRIBUTES].children())
    except Exception:  # noqa: BLE001 - no object attributes at all
        return None
    if not {"length", "start", "end"} <= available:
        return None

    n_objects = object_count(level_group)
    columns: dict[str, npt.NDArray[Any]] = {}
    for name, width in (("length", None), ("start", ndim), ("end", ndim)):
        values = np.asarray(read_object_attributes(level_group, name))
        if width is None and values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        expected = (n_objects,) if width is None else (n_objects, width)
        if values.shape != expected:
            raise ValueError(
                f"object attribute {name!r} has shape {values.shape}; bundle_summary "
                f"expects {expected}, one {'value' if width is None else 'point'} per "
                "object. Pass use_stored_attributes=False to measure the geometry "
                "instead"
            )
        columns[name] = values

    present = np.ones(n_objects, dtype=bool)
    for name in columns:
        mask = read_object_attribute_present_mask(level_group, name)
        if mask is not None:
            present &= np.asarray(mask, dtype=bool)
    return _StoredMeasures(columns["length"], columns["start"], columns["end"], present)


class _Measurer:
    """Length and endpoints for a set of object ids at one level."""

    def __init__(
        self,
        level_group: Any,
        *,
        level: int,
        ndim: int,
        batch_size: int,
        stored: _StoredMeasures | None,
    ) -> None:
        self.level_group = level_group
        self.level = level
        self.ndim = ndim
        self.batch_size = batch_size
        self.stored = stored
        self.n_objects = object_count(level_group)
        self.sources: set[str] = set()
        try:
            meta = level_group.read_array_meta(VERTICES)
            self.dtype = np.dtype(meta.get("dtype", "float32"))
        except Exception:  # noqa: BLE001 - core's own readers default the same way
            self.dtype = np.dtype(np.float32)

    def measure(
        self, oids: npt.NDArray[np.int64], group_name: str,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """``(lengths (n,), starts (n, D), ends (n, D))``, float64, in ``oids`` order."""
        if self.stored is None:
            return self._from_geometry(oids, group_name)

        lengths = self.stored.length[oids].astype(np.float64)
        starts = self.stored.start[oids].astype(np.float64)
        ends = self.stored.end[oids].astype(np.float64)
        absent = ~self.stored.present[oids]
        if len(oids):
            self.sources.add("attributes")
        if absent.any():
            got = self._from_geometry(oids[absent], group_name)
            lengths[absent], starts[absent], ends[absent] = got
        return lengths, starts, ends

    def _from_geometry(
        self, oids: npt.NDArray[np.int64], group_name: str,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        n = len(oids)
        lengths = np.zeros(n, dtype=np.float64)
        starts = np.zeros((n, self.ndim), dtype=np.float64)
        ends = np.zeros((n, self.ndim), dtype=np.float64)
        if n:
            self.sources.add("geometry")
        for lo in range(0, n, self.batch_size):
            batch = oids[lo:lo + self.batch_size]
            paths = self._read_paths(batch, group_name)
            hi = lo + len(batch)
            # The ingest enrichment helpers, so a length measured here equals
            # the ``length`` attribute an ingest would have stored for it.
            lengths[lo:hi] = compute_lengths(paths)
            batch_starts, batch_ends = compute_endpoints(paths)
            starts[lo:hi] = batch_starts
            ends[lo:hi] = batch_ends
        return lengths, starts, ends

    def _read_paths(
        self, batch: npt.NDArray[np.int64], group_name: str,
    ) -> list[npt.NDArray[Any]]:
        """Each object's whole vertex path, its fragments concatenated in order.

        Read from manifests and chunk vertices directly rather than through
        ``read_polylines``, which decodes every per-vertex attribute alongside
        the positions -- on a tractogram carrying a few scalars per point, most
        of the bytes read, and none of them needed for a length.
        """
        level_group = self.level_group
        manifests = read_object_manifests(level_group, ids=[int(o) for o in batch])
        empty = [int(o) for o in batch if not manifests.get(int(o))]
        if empty:
            raise ValueError(
                f"group {group_name!r} lists objects with no geometry at level "
                f"{self.level}: {empty[:5]}"
            )
        chunks = sorted({
            tuple(int(c) for c in cc)
            for manifest in manifests.values() for cc, _ in manifest
        })
        keys = [".".join(str(c) for c in cc) for cc in chunks]
        decoded: dict[tuple[int, ...], list[npt.NDArray[Any]]] = {}
        with level_group.batched_reads([(VERTICES, keys), (VERTEX_FRAGMENTS, keys)]):
            for cc in chunks:
                try:
                    decoded[cc] = read_chunk_vertices(
                        level_group, cc, dtype=self.dtype, ndim=self.ndim,
                    )
                except ArrayError:
                    decoded[cc] = []

        paths: list[npt.NDArray[Any]] = []
        for oid in batch:
            fragments = []
            for cc, fragment_index in manifests[int(oid)]:
                block = decoded[tuple(int(c) for c in cc)]
                if not 0 <= int(fragment_index) < len(block):
                    # Skipping it would join the fragments either side with a
                    # straight segment that is not in the data, and report a
                    # length for a path that does not exist.
                    raise ValueError(
                        f"object {int(oid)} (group {group_name!r}) names fragment "
                        f"{int(fragment_index)} of chunk {cc} at level {self.level}, "
                        "which the chunk does not hold"
                    )
                fragments.append(block[int(fragment_index)])
            paths.append(np.concatenate(fragments, axis=0))
        return paths


# ---------------------------------------------------------------------------
# Aggregation and output
# ---------------------------------------------------------------------------


def _empty_columns(n_groups: int, ndim: int) -> dict[str, npt.NDArray[Any]]:
    columns: dict[str, npt.NDArray[Any]] = {
        name: np.full(n_groups, np.nan, dtype=np.float64) for name in _SCALAR_COLUMNS
    }
    columns["streamline_count"] = np.zeros(n_groups, dtype=np.int64)
    for name in _CENTROID_COLUMNS:
        columns[name] = np.full((n_groups, ndim), np.nan, dtype=np.float64)
    return columns


def _orient(
    starts: npt.NDArray[np.float64], ends: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Reverse the streamlines that run against the bundle's principal axis."""
    d = ends - starts
    scatter = d.T @ d
    if not np.any(scatter):
        # Every streamline is a loop or a point: there is no direction to agree on.
        return starts, ends
    _, vectors = np.linalg.eigh(scatter)
    axis = vectors[:, -1]
    # ``eigh`` returns an eigenvector up to sign; pin it so the answer does not
    # depend on the LAPACK build.
    axis = axis * np.sign(axis[int(np.argmax(np.abs(axis)))])
    flip = (d @ axis) < 0
    if not flip.any():
        return starts, ends
    return (
        np.where(flip[:, None], ends, starts),
        np.where(flip[:, None], starts, ends),
    )


def _aggregate_into(
    columns: dict[str, npt.NDArray[Any]],
    gid: int,
    lengths: npt.NDArray[np.float64],
    starts: npt.NDArray[np.float64],
    ends: npt.NDArray[np.float64],
    *,
    orient: bool,
) -> None:
    n = len(lengths)
    columns["streamline_count"][gid] = n
    if n == 0:
        return  # The NaNs _empty_columns filled in are the answer.
    if orient:
        starts, ends = _orient(starts, ends)
    columns["length_mean"][gid] = float(np.mean(lengths))
    columns["length_std"][gid] = float(np.std(lengths))
    columns["length_min"][gid] = float(np.min(lengths))
    columns["length_median"][gid] = float(np.median(lengths))
    columns["length_max"][gid] = float(np.max(lengths))
    tortuosity = tortuosity_from_endpoints(starts, ends, lengths)
    columns["tortuosity_mean"][gid] = float(np.mean(tortuosity, dtype=np.float64))
    columns["start_centroid"][gid] = starts.mean(axis=0)
    columns["end_centroid"][gid] = ends.mean(axis=0)


def _table(
    columns: dict[str, npt.NDArray[Any]], names: list[str], axes: list[str],
) -> pd.DataFrame:
    import pandas as pd

    data: dict[str, npt.NDArray[Any]] = {
        "group_id": np.arange(len(names), dtype=np.int64),
        "streamline_count": np.asarray(columns["streamline_count"], dtype=np.int64),
    }
    for name in _SCALAR_COLUMNS[1:]:
        data[name] = np.asarray(columns[name], dtype=np.float64)
    for prefix, column in (("start", "start_centroid"), ("end", "end_centroid")):
        values = np.asarray(columns[column], dtype=np.float64)
        for i, axis in enumerate(axes):
            data[f"{prefix}_{axis}"] = values[:, i]
    return pd.DataFrame(data, index=pd.Index(names, name="group"))


def _write_columns(
    level_group: Any, columns: dict[str, npt.NDArray[Any]], provenance: dict[str, Any],
) -> None:
    for column in (*_SCALAR_COLUMNS, *_CENTROID_COLUMNS):
        values = np.asarray(columns[column])
        name = ATTRIBUTE_PREFIX + column
        create_groupings_attributes_array(
            level_group, name,
            dtype=str(values.dtype),
            num_channels=int(values.shape[1]) if values.ndim > 1 else 1,
        )
        write_groupings_attributes(level_group, name, values)
    # Beside the group names, on the array the rows describe.  A pyramid build
    # copies the attributes up from the level below but not this key, which is
    # the honest outcome: the copied rows are level 0's, exactly what the
    # default mode writes everywhere, and nothing claims to know more.
    meta = dict(level_group.read_array_meta(GROUPS) or {})
    meta[PROVENANCE_KEY] = provenance
    level_group.write_array_meta(GROUPS, meta)
