"""One shape for "geometry coming from somewhere else".

``ingest`` reads a file and *creates a store*; the two steps are welded
together in every one of its fifteen modules, so there is no way to ask a
TRK file for its streamlines without also asking for a store to be built.
That is fine while a store has exactly one source.  It stops being fine
the moment a second dataset has to land in a store that already exists —
an atlas of labelled bundles joining a whole-brain tractogram, a second
imaging run joining the first — because the only tool on offer builds a
*separate* store and leaves the caller to reconcile two object-id spaces,
two group taxonomies and two pyramids by hand.

A :class:`Source` is the missing half: geometry, its attributes and its
group taxonomy, read in bounded batches, with no opinion about where it
is going.  Two implementations cover everything:

* :class:`StoreSource` — another zarr-vectors store, at any level.
* :class:`FileSource` — an external file, through
  :mod:`zarr_vectors_tools.compose.readers`.

Both answer the same three questions (:attr:`~Source.info`,
:meth:`~Source.iter_batches`, :meth:`~Source.groups`), so
:func:`~zarr_vectors_tools.compose.merge.merge_stores` never learns which
kind it was handed.

**Batches, not everything.**  A whole-brain level-0 store is 900 million
vertices; materialising it to merge 86 thousand more would be absurd.
Sources yield :class:`ObjectBatch` values sized by object *and* vertex
count, so peak memory is a batch rather than a dataset.  Objects are the
batching unit because an object is the smallest thing that can be written
without being torn: a polyline split across two batches would be written
as two polylines.

**Transforms belong here.**  Bringing an external file into a store's
coordinate frame is the common case, not the exception — the two files
that motivated this module differ by a rigid re-orientation that lives in
their headers.  Applying it at the source, before anything is binned,
means the target's grid, bounds check and spatial index all see final
coordinates.  Applying it afterwards would mean rewriting them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from zarr_vectors.exceptions import IngestError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zarr_vectors_tools.compose.readers import Geometry

__all__ = [
    "ObjectBatch",
    "Source",
    "SourceInfo",
    "StoreSource",
    "FileSource",
    "GeometrySource",
    "as_transform",
    "open_source",
]

# Batch ceilings.  Two of them, because either alone is wrong: a store of
# 86k two-point objects and a store of 86 objects with a million points
# each both need bounding, and no single count does both.
DEFAULT_MAX_OBJECTS = 20_000
DEFAULT_MAX_VERTICES = 4_000_000


# =====================================================================
# Transforms
# =====================================================================

Transform = Callable[["npt.NDArray[Any]"], "npt.NDArray[Any]"]


def as_transform(spec: Any) -> Transform | None:
    """Coerce ``spec`` into a positions-in, positions-out callable.

    Accepts ``None`` (identity, returned as ``None`` so callers can skip
    the call entirely), a callable, a ``(D+1, D+1)`` homogeneous affine,
    or a ``(D, D+1)`` affine without its bottom row.

    A square matrix is always read as homogeneous, so for three-
    dimensional data the transform is ``4x4`` and a bare ``3x3`` is a
    *two*-dimensional affine — not a 3-D rotation.  That reading is
    deliberate and it fails loudly: applying a 2-D affine to ``(N, 3)``
    points raises on the matmul rather than silently projecting them.
    A rotation with no translation still needs its translation column.
    """
    if spec is None:
        return None
    if callable(spec):
        return spec

    matrix = np.asarray(spec, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(
            f"transform must be a callable or a 2-D affine matrix, got shape "
            f"{matrix.shape}"
        )
    rows, cols = matrix.shape
    if cols == rows:
        if rows < 3:
            raise ValueError(f"transform matrix {matrix.shape} is too small")
        linear, offset = matrix[: rows - 1, : cols - 1], matrix[: rows - 1, cols - 1]
    elif cols == rows + 1:
        linear, offset = matrix[:, :rows], matrix[:, rows]
    else:
        raise ValueError(
            f"transform matrix has shape {matrix.shape}; expected (D+1, D+1) "
            f"homogeneous or (D, D+1). A square (D, D) matrix is ambiguous "
            f"between 'rotation only' and a mis-sliced affine, so pass the "
            f"full affine with its translation column."
        )

    def apply(points: npt.NDArray[Any]) -> npt.NDArray[Any]:
        pts = np.asarray(points)
        out = pts @ linear.T + offset
        return out.astype(pts.dtype, copy=False)

    return apply


# =====================================================================
# What a source is
# =====================================================================


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """What a source holds, known before any geometry is read.

    Everything here is cheap: metadata for a store, a header for a file.
    :func:`~zarr_vectors_tools.compose.merge.merge_stores` plans the whole
    merge — bounds, id allocation, attribute reconciliation — from these
    alone, so a merge can be rejected before a single vertex moves.
    """

    label: str
    """Human-readable origin, used in errors and provenance."""

    kind: str
    """Geometry type, one of :mod:`zarr_vectors.constants`' ``GEOM_*``."""

    ndim: int

    bounds: tuple[tuple[float, ...], tuple[float, ...]] | None
    """Extent **after** any transform, or ``None`` when not known cheaply."""

    object_count: int | None = None
    vertex_count: int | None = None
    vertex_attributes: tuple[str, ...] = ()
    object_attributes: tuple[str, ...] = ()
    group_names: tuple[str, ...] = ()
    headers: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    """Format headers to carry across, keyed by format name."""

    def describe(self) -> str:
        objects = "?" if self.object_count is None else f"{self.object_count:,}"
        vertices = "?" if self.vertex_count is None else f"{self.vertex_count:,}"
        return f"{self.label} [{self.kind}] {objects} objects, {vertices} vertices"


@dataclass(frozen=True, slots=True)
class ObjectBatch:
    """A bounded run of whole objects.

    ``parts[i]``, ``object_ids[i]`` and every column of
    ``object_attributes`` share an index.  ``vertex_attributes`` columns
    are aligned with ``concatenate(parts)`` — the same convention the
    writers use, so a batch can be handed to ``add_polylines`` unmodified.
    """

    parts: list[npt.NDArray[Any]]
    """One ``(N_k, D)`` array per object.  Never split across batches."""

    object_ids: npt.NDArray[np.int64]
    """Ids **in the source's own numbering**; the merge remaps them."""

    object_attributes: Mapping[str, npt.NDArray[Any]] = field(default_factory=dict)
    vertex_attributes: Mapping[str, npt.NDArray[Any]] = field(default_factory=dict)

    @property
    def object_count(self) -> int:
        return len(self.parts)

    @property
    def vertex_count(self) -> int:
        return int(sum(len(p) for p in self.parts))

    def bounds(self) -> tuple[npt.NDArray[Any], npt.NDArray[Any]] | None:
        """This batch's extent, or ``None`` when it holds no vertices."""
        filled = [p for p in self.parts if len(p)]
        if not filled:
            return None
        lo = np.min([p.min(axis=0) for p in filled], axis=0)
        hi = np.max([p.max(axis=0) for p in filled], axis=0)
        return lo, hi


class Source(ABC):
    """Geometry from somewhere, in a shape a merge can consume."""

    @property
    @abstractmethod
    def info(self) -> SourceInfo:
        """What this source holds.  Cheap; safe to call repeatedly."""

    @abstractmethod
    def iter_batches(
        self,
        *,
        max_objects: int = DEFAULT_MAX_OBJECTS,
        max_vertices: int = DEFAULT_MAX_VERTICES,
    ) -> Iterator[ObjectBatch]:
        """Yield whole objects in batches no larger than the given bounds.

        A single object larger than ``max_vertices`` is still emitted
        whole, in a batch of its own — truncating it would corrupt it.
        """

    def groups(self) -> Mapping[str, npt.NDArray[np.int64]]:
        """Group name -> member object ids, in the source's own numbering.

        Empty by default: most file formats have no grouping concept, and
        a source that does overrides this.
        """
        return {}

    def close(self) -> None:
        """Release anything held open.  Idempotent."""

    def __enter__(self) -> Source:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.info.describe()})"


def _batched(
    lengths: Sequence[int], max_objects: int, max_vertices: int,
) -> Iterator[tuple[int, int]]:
    """``(start, stop)`` index runs honouring both ceilings.

    An object longer than ``max_vertices`` gets a batch to itself rather
    than being dropped or split, which is why the vertex test is skipped
    when the run is still empty.
    """
    start = 0
    verts = 0
    for i, n in enumerate(lengths):
        if i > start and (i - start >= max_objects or verts + n > max_vertices):
            yield start, i
            start, verts = i, 0
        verts += int(n)
    if start < len(lengths):
        yield start, len(lengths)


# =====================================================================
# A zarr-vectors store as a source
# =====================================================================


class StoreSource(Source):
    """Another zarr-vectors store, read one object batch at a time.

    **Why not just ``level.read()``.**  The facade's polyline reader
    returns positions and object ids but never vertex attributes —
    ``ReadResult.from_polylines`` hardcodes ``attributes_read=False``, and
    ``Level._gather_attributes`` can only repair that for an *unnarrowed*
    read, because a level-ordered attribute column cannot be aligned to a
    subset without knowing which vertices survived.  Batching is exactly
    such a narrowing, so going through the facade would silently drop
    every per-vertex attribute in the store.

    This class gathers attributes the same way the reader gathers
    vertices: an object's manifest lists ``(chunk, fragment_index)`` pairs,
    and the attribute arrays are fragmented identically, so
    ``read_chunk_attributes(g, name, cc)[fi]`` is the column for the
    fragment ``read_chunk_vertices(g, cc)[fi]`` holds.  Same manifest,
    same order, so they cannot drift.
    """

    def __init__(
        self,
        store: Any,
        *,
        level: int = 0,
        objects: Sequence[int] | None = None,
        transform: Any = None,
        with_vertex_attributes: bool = True,
        with_object_attributes: bool = True,
        label: str | None = None,
    ) -> None:
        import zarr_vectors as zv

        self._dataset = store if hasattr(store, "level") else zv.open_dataset(str(store))
        self._level = self._dataset.level(int(level))
        self._level_index = int(level)
        self._transform = as_transform(transform)
        self._want_vertex_attrs = bool(with_vertex_attributes)
        self._want_object_attrs = bool(with_object_attributes)
        self._label = label or f"{self._dataset.url}[level {level}]"
        self._explicit_objects = (
            None if objects is None else np.asarray(objects, dtype=np.int64)
        )
        self._info: SourceInfo | None = None
        self._ids: npt.NDArray[np.int64] | None = None

    # ---------------- introspection ----------------

    @property
    def dataset(self) -> Any:
        return self._dataset

    def object_ids(self) -> npt.NDArray[np.int64]:
        """The ids this source will emit, in emission order."""
        if self._ids is None:
            if self._explicit_objects is not None:
                self._ids = np.sort(np.unique(self._explicit_objects))
            else:
                self._ids = np.asarray(
                    self._level.objects.ids(present=True), dtype=np.int64
                )
        return self._ids

    @property
    def info(self) -> SourceInfo:
        if self._info is None:
            self._info = self._read_info()
        return self._info

    def _read_info(self) -> SourceInfo:
        try:
            lo, hi = self._dataset.bounds
            bounds: tuple[tuple[float, ...], tuple[float, ...]] | None = (
                tuple(float(v) for v in lo), tuple(float(v) for v in hi),
            )
        except Exception:  # noqa: BLE001 - a store may declare none
            bounds = None
        if bounds is not None and self._transform is not None:
            bounds = _transform_box(bounds, self._transform)

        headers: dict[str, dict[str, Any]] = {}
        try:
            registry = self._dataset.headers
            for name in registry.available_formats:
                try:
                    headers[str(name)] = dict(registry.get(name))
                except Exception:  # noqa: BLE001 - typed headers refuse dict()
                    continue
        except Exception:  # noqa: BLE001 - no headers group at all
            pass

        ids = self.object_ids()
        return SourceInfo(
            label=self._label,
            kind=self._level.kind,
            ndim=int(self._dataset.ndim),
            bounds=bounds,
            object_count=int(len(ids)),
            vertex_count=int(self._level.vertex_count) or None,
            vertex_attributes=(
                self._level.attribute_names("vertex") if self._want_vertex_attrs else ()
            ),
            object_attributes=(
                self._level.attribute_names("object") if self._want_object_attrs else ()
            ),
            group_names=tuple(self._level.groups.names()),
            headers=headers,
        )

    def groups(self) -> Mapping[str, npt.NDArray[np.int64]]:
        """Named groups, restricted to the objects this source emits.

        A member this source is not carrying is dropped rather than
        emitted: a group naming an id the target never received would
        resolve to an empty manifest and be reported as a member, which is
        the same silent lie ``propagate_groupings`` exists to avoid.
        """
        catalog = self._level.groups
        emitted = set(int(i) for i in self.object_ids())
        out: dict[str, npt.NDArray[np.int64]] = {}
        for name in catalog.names():
            try:
                members = np.asarray(catalog[name].members, dtype=np.int64)
            except Exception:  # noqa: BLE001 - unreadable row is not fatal
                continue
            kept = members[np.isin(members, list(emitted))] if len(members) else members
            out[str(name)] = kept.astype(np.int64, copy=False)
        return out

    # ---------------- reading ----------------

    def _level_group(self) -> Any:
        return self._level.store

    def _object_attribute_columns(self) -> dict[str, npt.NDArray[Any]]:
        """Every object-attribute array, whole.

        Object attributes are indexed by object id and there is one value
        per object, so the whole column is (objects x channels) — small
        next to the geometry even for a five-million-streamline store, and
        reading it once beats re-reading it per batch.
        """
        if not self._want_object_attrs:
            return {}
        from zarr_vectors.building import read_object_attributes

        group = self._level_group()
        out: dict[str, npt.NDArray[Any]] = {}
        for name in self._level.attribute_names("object"):
            try:
                out[str(name)] = np.asarray(read_object_attributes(group, name))
            except Exception:  # noqa: BLE001 - skip what will not read
                continue
        return out

    def iter_batches(
        self,
        *,
        max_objects: int = DEFAULT_MAX_OBJECTS,
        max_vertices: int = DEFAULT_MAX_VERTICES,
    ) -> Iterator[ObjectBatch]:
        from zarr_vectors.building import read_object_manifests

        ids = self.object_ids()
        if len(ids) == 0:
            return
        obj_columns = self._object_attribute_columns()
        attr_names = tuple(self.info.vertex_attributes)
        group = self._level_group()

        # Manifest sizes are what the batcher needs, and reading manifests
        # for the whole id set up front is O(objects) small integers --
        # cheaper than a second pass and it makes the vertex ceiling exact
        # rather than a guess.
        manifests = read_object_manifests(group, ids=[int(i) for i in ids])
        order = [int(i) for i in ids if manifests.get(int(i))]
        sizes = [len(manifests[oid]) for oid in order]

        for start, stop in _batched(sizes, max_objects, max_vertices):
            window = order[start:stop]
            parts, kept = self._gather(group, window, manifests)
            if not parts:
                continue
            batch_ids = np.asarray(kept, dtype=np.int64)
            yield ObjectBatch(
                parts=parts,
                object_ids=batch_ids,
                object_attributes={
                    name: np.asarray(col)[batch_ids]
                    for name, col in obj_columns.items()
                    if len(col) > int(batch_ids.max(initial=-1))
                },
                vertex_attributes=self._gather_attributes(
                    group, kept, manifests, attr_names, parts,
                ),
            )

    def _gather(
        self, group: Any, window: Sequence[int], manifests: Mapping[int, Any],
    ) -> tuple[list[npt.NDArray[Any]], list[int]]:
        """Assemble each object's vertices from its fragments."""
        from zarr_vectors.building import read_chunk_vertices

        ndim = int(self._dataset.ndim)
        cache: dict[tuple[int, ...], list[npt.NDArray[Any]]] = {}

        def fragments(cc: tuple[int, ...]) -> list[npt.NDArray[Any]]:
            if cc not in cache:
                try:
                    cache[cc] = read_chunk_vertices(group, cc, ndim=ndim)
                except Exception:  # noqa: BLE001 - missing chunk reads as empty
                    cache[cc] = []
            return cache[cc]

        parts: list[npt.NDArray[Any]] = []
        kept: list[int] = []
        for oid in window:
            pieces = [
                frag
                for cc, fi in manifests[oid]
                for frag in (_at(fragments(tuple(cc)), fi),)
                if frag is not None and len(frag)
            ]
            if not pieces:
                continue
            joined = np.concatenate(pieces, axis=0)
            if self._transform is not None:
                joined = self._transform(joined)
            parts.append(joined)
            kept.append(int(oid))
        return parts, kept

    def _gather_attributes(
        self,
        group: Any,
        window: Sequence[int],
        manifests: Mapping[int, Any],
        names: Sequence[str],
        parts: Sequence[npt.NDArray[Any]],
    ) -> dict[str, npt.NDArray[Any]]:
        """Per-vertex attribute columns for one batch, fragment by fragment.

        Aligned by construction: the manifest walked here is the manifest
        walked in :meth:`_gather`, in the same order, so column row *i*
        belongs to ``concatenate(parts)`` row *i*.  A name whose fragments
        do not add up to the vertex count is dropped rather than written
        misaligned.
        """
        if not names:
            return {}
        from zarr_vectors.building import read_chunk_attributes

        expected = int(sum(len(p) for p in parts))
        out: dict[str, npt.NDArray[Any]] = {}
        for name in names:
            cache: dict[tuple[int, ...], list[npt.NDArray[Any]]] = {}

            def fragments(cc: tuple[int, ...], _n: str = name) -> list[npt.NDArray[Any]]:
                if cc not in cache:
                    try:
                        cache[cc] = list(read_chunk_attributes(group, _n, cc))
                    except Exception:  # noqa: BLE001
                        cache[cc] = []
                return cache[cc]

            pieces: list[npt.NDArray[Any]] = []
            for oid in window:
                for cc, fi in manifests[oid]:
                    frag = _at(fragments(tuple(cc)), fi)
                    if frag is not None and len(frag):
                        pieces.append(np.asarray(frag))
            if not pieces:
                continue
            column = np.concatenate(pieces, axis=0)
            if len(column) == expected:
                out[str(name)] = column
        return out


def _at(items: Sequence[Any], index: int) -> Any | None:
    return items[index] if 0 <= index < len(items) else None


def _transform_box(
    bounds: tuple[Sequence[float], Sequence[float]], transform: Transform,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """The transformed extent of an axis-aligned box.

    Every corner is mapped, not just the two given: a transform that
    permutes or rotates axes sends ``min`` somewhere that is no longer the
    minimum, and taking the image of two corners would report a box the
    data does not fit in.
    """
    import itertools

    lo, hi = np.asarray(bounds[0], dtype=np.float64), np.asarray(bounds[1], dtype=np.float64)
    corners = np.asarray(list(itertools.product(*zip(lo, hi))), dtype=np.float64)
    moved = transform(corners)
    return (
        tuple(float(v) for v in np.min(moved, axis=0)),
        tuple(float(v) for v in np.max(moved, axis=0)),
    )


# =====================================================================
# In-memory geometry as a source
# =====================================================================


class GeometrySource(Source):
    """Geometry already in memory, wrapped so a merge can take it.

    What :class:`FileSource` becomes once its reader has run, and the
    thing to reach for when geometry was produced in Python rather than
    read from anywhere.
    """

    def __init__(
        self,
        geometry: Geometry,
        *,
        transform: Any = None,
        label: str = "<memory>",
    ) -> None:
        self._geometry = geometry
        self._transform = as_transform(transform)
        self._label = label
        self._info: SourceInfo | None = None

    @property
    def geometry(self) -> Geometry:
        return self._geometry

    @property
    def info(self) -> SourceInfo:
        if self._info is None:
            geom = self._geometry
            bounds = geom.bounds()
            if bounds is not None and self._transform is not None:
                bounds = _transform_box(bounds, self._transform)
            self._info = SourceInfo(
                label=self._label,
                kind=geom.kind,
                ndim=geom.ndim,
                bounds=bounds,
                object_count=len(geom.parts),
                vertex_count=int(sum(len(p) for p in geom.parts)),
                vertex_attributes=tuple(sorted(geom.vertex_attributes)),
                object_attributes=tuple(sorted(geom.object_attributes)),
                group_names=tuple(sorted(geom.groups)),
                headers=dict(geom.headers),
            )
        return self._info

    def groups(self) -> Mapping[str, npt.NDArray[np.int64]]:
        return {
            str(name): np.asarray(members, dtype=np.int64)
            for name, members in self._geometry.groups.items()
        }

    def iter_batches(
        self,
        *,
        max_objects: int = DEFAULT_MAX_OBJECTS,
        max_vertices: int = DEFAULT_MAX_VERTICES,
    ) -> Iterator[ObjectBatch]:
        geom = self._geometry
        lengths = [len(p) for p in geom.parts]
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)

        for start, stop in _batched(lengths, max_objects, max_vertices):
            parts = [np.asarray(p) for p in geom.parts[start:stop]]
            if self._transform is not None:
                parts = [self._transform(p) for p in parts]
            lo, hi = int(offsets[start]), int(offsets[stop])
            yield ObjectBatch(
                parts=parts,
                object_ids=np.arange(start, stop, dtype=np.int64),
                object_attributes={
                    name: np.asarray(col)[start:stop]
                    for name, col in geom.object_attributes.items()
                },
                vertex_attributes={
                    name: np.asarray(col)[lo:hi]
                    for name, col in geom.vertex_attributes.items()
                },
            )


# =====================================================================
# A file as a source
# =====================================================================


class FileSource(Source):
    """An external file, read through :mod:`compose.readers`.

    Formats with a native reader are read straight into memory.  Anything
    else is ingested into a scratch store first and then read back as a
    :class:`StoreSource` — slower and it costs disk, but it means every
    format ``zvtools convert`` accepts can be merged today rather than
    once someone writes a reader for it.  Which route was taken is on
    :attr:`staged`, since the difference is worth reporting and not worth
    hiding.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        format: str | None = None,
        transform: Any = None,
        scratch_dir: str | Path | None = None,
        label: str | None = None,
        **reader_options: Any,
    ) -> None:
        from zarr_vectors_tools.compose.readers import (
            has_native_reader,
            read_geometry,
            resolve_reader_format,
        )

        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Input file not found: {self._path}")
        self._format = resolve_reader_format(self._path, format)
        self._label = label or f"{self._path.name} ({self._format})"
        self._delegate: Source
        self._scratch: Any = None

        if has_native_reader(self._format):
            geometry = read_geometry(
                self._path, self._format, **reader_options,
            )
            self._delegate = GeometrySource(
                geometry, transform=transform, label=self._label,
            )
            self.staged = False
        else:
            self._delegate, self._scratch = _stage_via_ingest(
                self._path,
                self._format,
                scratch_dir=scratch_dir,
                transform=transform,
                label=self._label,
                **reader_options,
            )
            self.staged = True

    @property
    def path(self) -> Path:
        return self._path

    @property
    def format(self) -> str:
        return self._format

    @property
    def info(self) -> SourceInfo:
        return self._delegate.info

    def groups(self) -> Mapping[str, npt.NDArray[np.int64]]:
        return self._delegate.groups()

    def iter_batches(
        self,
        *,
        max_objects: int = DEFAULT_MAX_OBJECTS,
        max_vertices: int = DEFAULT_MAX_VERTICES,
    ) -> Iterator[ObjectBatch]:
        return self._delegate.iter_batches(
            max_objects=max_objects, max_vertices=max_vertices,
        )

    def close(self) -> None:
        self._delegate.close()
        if self._scratch is not None:
            self._scratch.cleanup()
            self._scratch = None


def _stage_via_ingest(
    path: Path,
    fmt: str,
    *,
    scratch_dir: str | Path | None,
    transform: Any,
    label: str,
    **options: Any,
) -> tuple[Source, Any]:
    """Ingest into a throwaway store, return a source reading it back."""
    import tempfile

    from zarr_vectors_tools.compose.readers import ingest_to_store

    scratch = tempfile.TemporaryDirectory(
        prefix="zv-compose-", dir=str(scratch_dir) if scratch_dir else None,
    )
    try:
        staged = ingest_to_store(
            path, fmt, Path(scratch.name) / "staged.zarrvectors", **options,
        )
    except Exception:
        scratch.cleanup()
        raise
    return StoreSource(staged, transform=transform, label=label), scratch


# =====================================================================
# Dispatch
# =====================================================================


def open_source(spec: Any, **kw: Any) -> Source:
    """Open ``spec`` as a :class:`Source`, whatever it is.

    A store path, a store URL, an open ``Dataset``, an already-built
    :class:`Source`, or a path to a file in any format ``zvtools``
    understands.  The distinction that matters is store-versus-file, and
    it is decided by looking for the store's root metadata rather than by
    the extension — ``.zarrvectors`` is a convention, not a guarantee, and
    a store handed over under another name should still work.
    """
    if isinstance(spec, Source):
        return spec
    if hasattr(spec, "level") and hasattr(spec, "bounds"):
        return StoreSource(spec, **kw)

    text = str(spec)
    if _looks_like_store(text):
        store_kw = {
            k: v for k, v in kw.items()
            if k in {
                "level", "objects", "transform", "with_vertex_attributes",
                "with_object_attributes", "label",
            }
        }
        return StoreSource(text, **store_kw)
    return FileSource(text, **kw)


def _looks_like_store(path: str) -> bool:
    """Whether ``path`` is a zarr-vectors store, by asking it.

    A remote URL is assumed to be a store rather than probed: the probe
    would be a network round trip on the happy path and the failure mode
    of guessing wrong is a clear error from ``open_dataset`` either way.
    """
    local = Path(path)
    if not local.exists():
        return "://" in path
    if local.is_file():
        return False
    root = local / "zarr.json"
    if not root.exists():
        return False
    try:
        import json

        return "zarr_vectors" in json.loads(root.read_text()).get("attributes", {})
    except Exception:  # noqa: BLE001 - unreadable metadata is not a store
        return False


def require_geometry_match(sources: Sequence[Source], target_kind: str | None) -> str:
    """The one geometry kind these sources agree on.

    Merging a mesh into a streamline store is not a partial success to be
    discovered halfway through: the writers differ, the manifests differ,
    and the result would be a store whose declared ``geometry_types`` does
    not describe half its contents.
    """
    kinds = {s.info.kind for s in sources}
    if target_kind:
        kinds.add(target_kind)
    if len(kinds) > 1:
        detail = ", ".join(
            f"{s.info.label}={s.info.kind}" for s in sources
        )
        raise IngestError(
            f"cannot merge mixed geometry: {sorted(kinds)}. Sources are {detail}"
            + (f", target is {target_kind}" if target_kind else "")
            + ". Split the merge into one call per geometry type."
        )
    return kinds.pop() if kinds else ""
