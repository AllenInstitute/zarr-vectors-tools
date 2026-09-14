"""Readers that return geometry instead of building a store.

Every module in :mod:`zarr_vectors_tools.ingest` ends in a ``write_*``
call.  That is the right shape for "turn this file into a store" and the
wrong shape for "put this file into the store I already have": the
caller wants the streamlines, not a second store to reconcile.  These
readers are the ingest modules cut in half — the parsing, without the
writing.

**Native, or staged.**  Writing a reader per format is work, and most of
that work already exists inside the ingesters.  So the registry has two
tiers: formats with a native reader here, and everything else, which
:class:`~zarr_vectors_tools.compose.sources.FileSource` stages through a
throwaway store.  Staging costs a full write and a full read; a native
reader costs neither.  Formats get promoted as they earn it, and nothing
is unsupported in the meantime.

**On TRK.**  The native TRK reader parses the container directly rather
than going through ``nibabel``, for two reasons.  It makes positions,
per-point scalars and per-streamline properties available without
materialising a whole ``Tractogram`` — the file that motivated this is
22 million points — and it reads ``n_properties`` from byte **238**,
where the TrackVis spec puts it.  The parallel ingester reads it from
byte 236, which is inside the ``scalar_name`` table; for a file with no
properties both give zero and nothing is wrong, and for a file *with*
them the stride is short by four bytes per streamline and the parse walks
off the data.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from zarr_vectors.constants import (
    GEOM_POINT_CLOUD,
    GEOM_POLYLINE,
    GEOM_STREAMLINE,
)
from zarr_vectors.exceptions import IngestError

__all__ = [
    "Geometry",
    "derive_groups",
    "has_native_reader",
    "ingest_to_store",
    "native_formats",
    "read_geometry",
    "read_tck",
    "read_trk",
    "read_trx",
    "resolve_reader_format",
]


@dataclass(slots=True)
class Geometry:
    """Geometry in memory, in the shape a merge consumes.

    ``parts`` is one array per object for object-bearing geometry, and a
    single array for a point cloud.  ``vertex_attributes`` columns align
    with ``concatenate(parts)``; ``object_attributes`` columns align with
    ``parts`` itself.
    """

    kind: str
    parts: list[npt.NDArray[Any]]
    vertex_attributes: dict[str, npt.NDArray[Any]] = field(default_factory=dict)
    object_attributes: dict[str, npt.NDArray[Any]] = field(default_factory=dict)
    groups: dict[str, npt.NDArray[np.int64]] = field(default_factory=dict)
    headers: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: npt.NDArray[Any] | None = None
    faces: npt.NDArray[Any] | None = None

    @property
    def ndim(self) -> int:
        for part in self.parts:
            if len(part):
                return int(np.asarray(part).shape[1])
        return 3

    def bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
        filled = [np.asarray(p) for p in self.parts if len(p)]
        if not filled:
            return None
        lo = np.min([p.min(axis=0) for p in filled], axis=0)
        hi = np.max([p.max(axis=0) for p in filled], axis=0)
        return tuple(float(v) for v in lo), tuple(float(v) for v in hi)

    def __repr__(self) -> str:
        return (
            f"Geometry({self.kind}, objects={len(self.parts)}, "
            f"vertices={sum(len(p) for p in self.parts)})"
        )


# =====================================================================
# TRK
# =====================================================================

# TrackVis header v2 field offsets, from the format spec.  Spelled out
# because the one that matters is easy to get wrong: ``scalar_name`` is a
# 10x20 byte table running 38..238, so ``n_properties`` begins at 238 and
# anything reading 236 lands in the table's last two bytes.
_TRK_N_SCALARS = 36
_TRK_SCALAR_NAMES = 38
_TRK_N_PROPERTIES = 238
_TRK_PROPERTY_NAMES = 240
_TRK_VOX_TO_RAS = 440
_TRK_VOXEL_ORDER = 948
_TRK_N_COUNT = 988
_TRK_VERSION = 992
_TRK_HEADER_BYTES = 1000


def _trk_names(raw: bytes, offset: int, count: int) -> list[str]:
    """The ``count`` populated entries of a 10x20 name table."""
    out: list[str] = []
    for i in range(min(int(count), 10)):
        start = offset + i * 20
        text = raw[start:start + 20].split(b"\x00", 1)[0]
        out.append(text.decode("ascii", errors="replace") or f"scalar_{i}")
    return out


def read_trk_header(path: str | Path) -> dict[str, Any]:
    """Parse a TRK header without reading a single streamline."""
    with open(path, "rb") as handle:
        raw = handle.read(_TRK_HEADER_BYTES)
    if len(raw) < _TRK_HEADER_BYTES:
        raise IngestError(f"File too short to be a valid TRK: {path}")
    if raw[:5] != b"TRACK":
        raise IngestError(f"Not a TRK file (bad magic): {path}")

    n_scalars = struct.unpack_from("<h", raw, _TRK_N_SCALARS)[0]
    n_properties = struct.unpack_from("<h", raw, _TRK_N_PROPERTIES)[0]
    return {
        "format_name": "trk",
        "dimensions": list(struct.unpack_from("<3h", raw, 6)),
        "voxel_size": list(struct.unpack_from("<3f", raw, 12)),
        "origin": list(struct.unpack_from("<3f", raw, 24)),
        "n_scalars": int(n_scalars),
        "scalar_names": _trk_names(raw, _TRK_SCALAR_NAMES, n_scalars),
        "n_properties": int(n_properties),
        "property_names": _trk_names(raw, _TRK_PROPERTY_NAMES, n_properties),
        "vox_to_ras": [
            float(v) for v in struct.unpack_from("<16f", raw, _TRK_VOX_TO_RAS)
        ],
        "voxel_order": raw[_TRK_VOXEL_ORDER:_TRK_VOXEL_ORDER + 4]
        .rstrip(b"\x00")
        .decode("ascii", errors="replace"),
        "n_count": int(struct.unpack_from("<i", raw, _TRK_N_COUNT)[0]),
        "version": int(struct.unpack_from("<i", raw, _TRK_VERSION)[0]),
    }


def trackvis_to_rasmm(header: Mapping[str, Any]) -> npt.NDArray[np.float64]:
    """The affine taking this file's stored coordinates to RAS millimetres.

    Delegated to ``nibabel`` rather than reimplemented.  The mapping is
    not just the header's ``vox_to_ras``: it also divides out the voxel
    size, applies the voxel-order flips and shifts by half a voxel to
    reach voxel centres, and each of those is a way to place a whole
    tractogram slightly — or completely — wrong.
    """
    try:
        import nibabel as nib
        from nibabel.streamlines.trk import get_affine_trackvis_to_rasmm, header_2_dtype
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise IngestError(
            "converting TRK coordinates to RAS needs nibabel. Install with: "
            "pip install 'zarr-vectors-tools[trk]', or read the file in its "
            "stored space with space='voxmm'."
        ) from exc
    del nib

    blob = header.get("_raw")
    if blob is None:
        raise IngestError(
            "trackvis_to_rasmm needs the raw header bytes; call read_trk with "
            "space='ras' rather than converting afterwards."
        )
    parsed = np.frombuffer(blob, dtype=header_2_dtype, count=1)[0]
    return np.asarray(get_affine_trackvis_to_rasmm(parsed), dtype=np.float64)


def read_trk(
    path: str | Path,
    *,
    space: str = "voxmm",
    dtype: str = "float32",
    scalars: bool = True,
    properties: bool = True,
    limit: int | None = None,
) -> Geometry:
    """Read a TrackVis ``.trk`` file into memory.

    Args:
        path: The ``.trk`` file.
        space: ``"voxmm"`` (default) keeps the coordinates exactly as
            stored; ``"ras"`` applies the header's affine to reach RAS
            millimetres.  The default is the stored space because that is
            what a store built by ``zvtools convert`` without
            ``--apply-affine`` is already in, and matching it is what lets
            a second file land on the same grid.
        scalars: Carry per-point scalars as vertex attributes.
        properties: Carry per-streamline properties as object attributes.
        limit: Stop after this many streamlines.  For inspection.

    Returns:
        A :class:`Geometry` of streamlines, with the parsed header under
        ``headers["trk"]``.
    """
    path = Path(path)
    header = read_trk_header(path)
    n_scalars = header["n_scalars"]
    n_properties = header["n_properties"]
    width = 3 + n_scalars
    np_dtype = np.dtype(dtype)

    parts: list[npt.NDArray[Any]] = []
    scalar_columns: list[npt.NDArray[Any]] = []
    property_rows: list[npt.NDArray[Any]] = []

    with open(path, "rb") as handle:
        raw_header = handle.read(_TRK_HEADER_BYTES)
        while True:
            if limit is not None and len(parts) >= limit:
                break
            count_bytes = handle.read(4)
            if len(count_bytes) < 4:
                break
            n_points = struct.unpack("<i", count_bytes)[0]
            if n_points < 0:
                raise IngestError(
                    f"{path}: streamline {len(parts)} declares {n_points} points"
                )
            payload = handle.read(n_points * width * 4)
            if len(payload) < n_points * width * 4:
                raise IngestError(
                    f"{path}: truncated at streamline {len(parts)}; the file ends "
                    f"mid-record. Expected {n_points} points."
                )
            block = np.frombuffer(payload, dtype="<f4").reshape(n_points, width)
            parts.append(block[:, :3].astype(np_dtype, copy=True))
            if n_scalars and scalars:
                scalar_columns.append(block[:, 3:].astype(np.float32, copy=True))
            if n_properties:
                prop_bytes = handle.read(n_properties * 4)
                if len(prop_bytes) < n_properties * 4:
                    raise IngestError(
                        f"{path}: truncated in the property block of streamline "
                        f"{len(parts) - 1}"
                    )
                if properties:
                    property_rows.append(
                        np.frombuffer(prop_bytes, dtype="<f4").astype(np.float32)
                    )

    if not parts:
        raise IngestError(f"TRK file contains no streamlines: {path}")

    declared = header["n_count"]
    if limit is None and declared and declared != len(parts):
        # Worth saying out loud: a mismatch means either a truncated file
        # or a stride bug, and both produce plausible-looking geometry.
        header["n_count_mismatch"] = [int(declared), len(parts)]

    geometry = Geometry(
        kind=GEOM_STREAMLINE,
        parts=parts,
        headers={"trk": {k: v for k, v in header.items() if k != "_raw"}},
    )

    if scalar_columns:
        stacked = np.concatenate(scalar_columns, axis=0)
        for i, name in enumerate(header["scalar_names"][:n_scalars]):
            geometry.vertex_attributes[name] = stacked[:, i]
    if property_rows:
        table = np.stack(property_rows, axis=0)
        for i, name in enumerate(header["property_names"][:n_properties]):
            geometry.object_attributes[name] = table[:, i]

    if space == "ras":
        affine = trackvis_to_rasmm({**header, "_raw": raw_header})
        linear, offset = affine[:3, :3], affine[:3, 3]
        geometry.parts = [
            (p @ linear.T + offset).astype(np_dtype, copy=False) for p in geometry.parts
        ]
        geometry.headers["trk"]["space"] = "rasmm"
    elif space != "voxmm":
        raise ValueError(f"space must be 'voxmm' or 'ras', got {space!r}")
    else:
        geometry.headers["trk"]["space"] = "voxmm"

    return geometry


# =====================================================================
# TCK and TRX
# =====================================================================


def read_tck(path: str | Path, *, dtype: str = "float32") -> Geometry:
    """Read an MRtrix ``.tck`` file.  Positions only — TCK carries no
    per-point or per-streamline data."""
    try:
        import nibabel as nib
    except ImportError as exc:
        raise IngestError(
            "reading TCK needs nibabel. Install with: "
            "pip install 'zarr-vectors-tools[trk]'"
        ) from exc

    loaded = nib.streamlines.load(str(path))
    parts = [np.asarray(s, dtype=np.dtype(dtype)) for s in loaded.streamlines]
    if not parts:
        raise IngestError(f"TCK file contains no streamlines: {path}")
    return Geometry(kind=GEOM_STREAMLINE, parts=parts)


def read_trx(path: str | Path, *, dtype: str = "float32") -> Geometry:
    """Read a ``.trx`` file, with its per-point and per-streamline data."""
    try:
        from trx.trx_file_memmap import load as load_trx
    except ImportError as exc:
        raise IngestError(
            "reading TRX needs trx-python. Install with: "
            "pip install 'zarr-vectors-tools[trx]'"
        ) from exc

    trx = load_trx(str(path))
    np_dtype = np.dtype(dtype)
    positions = np.asarray(trx.streamlines._data)
    lengths = np.asarray(trx.streamlines._lengths)

    parts: list[npt.NDArray[Any]] = []
    cursor = 0
    for n in lengths:
        parts.append(positions[cursor:cursor + int(n)].astype(np_dtype, copy=True))
        cursor += int(n)

    geometry = Geometry(kind=GEOM_STREAMLINE, parts=parts)
    for name, values in (trx.data_per_vertex or {}).items():
        geometry.vertex_attributes[str(name)] = np.asarray(values._data).squeeze()
    for name, values in (trx.data_per_streamline or {}).items():
        geometry.object_attributes[str(name)] = np.asarray(values).squeeze()
    # Every array above is copied out, so the file can be released here.
    # A ``.trx`` is a zip that ``load`` extracts to a scratch directory and
    # memory-maps; leaving it open keeps that directory alive for the life
    # of the process (and on Windows keeps it undeletable), which a loop
    # over a directory of tractograms turns into a real leak.
    try:
        trx.close()
    except Exception:  # noqa: BLE001 - nothing to release
        pass
    return geometry


# =====================================================================
# Registry
# =====================================================================

# Formats read straight into memory.  Everything absent from here is
# staged through a scratch store by FileSource, which is slower but
# needs no new code when a format is added to ``ingest``.
_NATIVE: dict[str, Any] = {
    "trk": read_trk,
    "tck": read_tck,
    "trx": read_trx,
}

_EXTENSIONS: dict[str, str] = {
    ".trk": "trk",
    ".tck": "tck",
    ".trx": "trx",
}


def native_formats() -> tuple[str, ...]:
    """Formats with an in-memory reader, so no staging is needed."""
    return tuple(sorted(_NATIVE))


def has_native_reader(fmt: str) -> bool:
    return fmt in _NATIVE


def resolve_reader_format(path: str | Path, explicit: str | None) -> str:
    """The format name for ``path``, from ``--format`` or the extension."""
    if explicit and explicit != "auto":
        return explicit
    suffix = Path(path).suffix.lower()
    if suffix in _EXTENSIONS:
        return _EXTENSIONS[suffix]

    from zarr_vectors_tools.cli._args import FORMAT_REGISTRY

    for fmt in FORMAT_REGISTRY.values():
        if suffix in fmt.exts:
            return fmt.name
    raise IngestError(
        f"cannot tell what format {Path(path).name!r} is from its extension; "
        f"pass format= explicitly. Known: "
        f"{', '.join(sorted(set(_NATIVE) | set(FORMAT_REGISTRY)))}"
    )


def read_geometry(path: str | Path, fmt: str, **options: Any) -> Geometry:
    """Read ``path`` with the native reader for ``fmt``."""
    try:
        reader = _NATIVE[fmt]
    except KeyError:
        raise IngestError(
            f"no native reader for {fmt!r}; it can still be merged, by staging "
            f"through a scratch store. Native readers: {', '.join(native_formats())}"
        ) from None
    return reader(path, **options)


def ingest_to_store(
    path: str | Path, fmt: str, target: str | Path, **options: Any,
) -> str:
    """Ingest ``path`` into a scratch store and return its path.

    The staging half of the two-tier registry.  The grid this store gets
    is irrelevant to the merge — the merge re-bins everything onto the
    *target's* grid — so a caller who has no opinion should not have to
    invent one, but the ingesters that require ``chunk_shape`` genuinely
    require it and inventing a number in their units would be worse than
    saying so.
    """
    from zarr_vectors_tools.cli._args import FORMAT_REGISTRY, load_ingest_func

    if fmt not in FORMAT_REGISTRY:
        raise IngestError(
            f"cannot stage {fmt!r}: it is neither a native reader "
            f"({', '.join(native_formats())}) nor a known ingest format "
            f"({', '.join(sorted(FORMAT_REGISTRY))})"
        )
    func = load_ingest_func(FORMAT_REGISTRY[fmt])
    try:
        func(str(path), str(target), **options)
    except TypeError as exc:
        if "chunk_shape" in str(exc):
            raise IngestError(
                f"staging {fmt!r} through a scratch store needs chunk_shape=; "
                f"the {fmt} ingester has no default for it. Pass it as a reader "
                f"option, e.g. chunk_shape=(100.0, 100.0, 100.0)."
            ) from exc
        raise
    return str(target)


# =====================================================================
# Turning a label column into groups
# =====================================================================


def derive_groups(
    geometry: Geometry,
    attribute: str,
    *,
    names: Mapping[Any, str] | None = None,
    drop_attribute: bool = False,
) -> Geometry:
    """Turn a per-object label column into named object groups.

    A bundle atlas usually arrives as one tractogram plus an integer code
    per streamline plus a lookup table mapping codes to names — three
    artefacts that only mean something together, and that a store can hold
    as one thing.  Groups are that one thing: named, first-class, and
    readable without the writing application's source next to them, which
    is exactly what a bare ``label_id`` column is not.

    Args:
        geometry: The geometry to group; modified in place and returned.
        attribute: Name of the per-object attribute holding the labels.
        names: Optional ``{code: name}`` lookup.  A code with no entry
            keeps a ``"<attribute>_<code>"`` name rather than being
            dropped — an unlabelled bundle is still a bundle.
        drop_attribute: Remove the label column once grouped.  Off by
            default: the column is the only thing that survives an export
            back to a format with no group concept.

    Returns:
        ``geometry``, with :attr:`Geometry.groups` populated.
    """
    try:
        labels = np.asarray(geometry.object_attributes[attribute])
    except KeyError:
        raise IngestError(
            f"no object attribute {attribute!r} to group by; this geometry has "
            f"{sorted(geometry.object_attributes) or 'none'}"
        ) from None
    if labels.ndim > 1:
        labels = labels.reshape(len(labels), -1)[:, 0]
    if len(labels) != len(geometry.parts):
        raise IngestError(
            f"attribute {attribute!r} has {len(labels)} values but there are "
            f"{len(geometry.parts)} objects"
        )

    lookup = {_key(k): str(v) for k, v in (names or {}).items()}
    # One grouping pass.  ``flatnonzero(labels == code)`` inside a loop
    # over the distinct labels is O(labels x objects) -- on a whole-brain
    # tractogram with a few thousand bundles that is tens of billions of
    # comparisons for work that is one sort.
    for code, members in _group_members(labels).items():
        geometry.groups[lookup.get(_key(code), f"{attribute}_{_key(code)}")] = members
    if drop_attribute:
        geometry.object_attributes.pop(attribute, None)
    return geometry


def _group_members(
    labels: npt.NDArray[Any],
) -> dict[Any, npt.NDArray[np.int64]]:
    """``{label: member indices}`` in one pass over ``labels``.

    ``np.unique(..., return_inverse=True)`` plus one stable argsort gives
    every group's members as contiguous slices, so the cost is a sort
    rather than a scan per distinct label.
    """
    values = np.asarray(labels).ravel()
    if values.size == 0:
        return {}
    uniques, inverse = np.unique(values, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    order = np.argsort(inverse, kind="stable")
    boundaries = np.flatnonzero(np.diff(inverse[order])) + 1
    out: dict[Any, npt.NDArray[np.int64]] = {}
    for chunk in np.split(order, boundaries):
        if chunk.size:
            out[uniques[inverse[chunk[0]]]] = chunk.astype(np.int64)
    return out


def _key(value: Any) -> Any:
    """A label's canonical key, so ``53``, ``53.0`` and ``"53"`` agree.

    A LUT read from JSON has string or int keys; the column read from a
    TRK is float32.  Without this the join silently finds nothing and
    every bundle comes out named ``label_id_53.0``.
    """
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return value
    number = float(value)
    return int(number) if number.is_integer() else number


def load_lut(path: str | Path, *, invert: bool = True) -> dict[Any, str]:
    """Read a ``{name: code}`` JSON lookup table into ``{code: name}``.

    ``invert`` is on by default because that is the direction these files
    are written in — a bundle atlas ships ``{"ac.trk": 21, ...}`` — while
    the direction a grouping needs is the other one.
    """
    import json

    raw = json.loads(Path(path).read_text())
    if not invert:
        return {_key(k): str(v) for k, v in raw.items()}
    out: dict[Any, str] = {}
    for name, code in raw.items():
        stem = Path(str(name)).stem
        out[_key(code)] = stem
    return out


def kinds_with_topology() -> frozenset[str]:
    """Geometry kinds whose meaning is not carried by positions alone."""
    from zarr_vectors.constants import GEOM_GRAPH, GEOM_MESH, GEOM_SKELETON

    return frozenset({GEOM_MESH, GEOM_GRAPH, GEOM_SKELETON})


def object_bearing_kinds() -> frozenset[str]:
    """Kinds this module can move between stores today."""
    return frozenset({GEOM_STREAMLINE, GEOM_POLYLINE, GEOM_POINT_CLOUD})


def check_supported(kind: str) -> None:
    """Raise unless ``kind`` round-trips through :class:`Geometry` intact.

    Meshes, graphs and skeletons carry their meaning in ``faces`` and
    ``edges``, and a batch of positions is not those.  Refusing is the
    point: a merge that quietly dropped a mesh's faces would produce a
    store that reads as a valid point cloud and is silently no longer a
    surface.
    """
    if kind in kinds_with_topology():
        raise IngestError(
            f"merging {kind!r} is not supported yet: its topology (faces/edges) "
            f"is indexed into a vertex array, and re-indexing it across two "
            f"stores' id spaces is not implemented. Supported: "
            f"{', '.join(sorted(object_bearing_kinds()))}."
        )


def _unused(_: Sequence[Any]) -> None:  # pragma: no cover - import hygiene
    pass
