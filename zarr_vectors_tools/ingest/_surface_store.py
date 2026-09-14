"""Write a cortical surface store: shared by the GIFTI and FreeSurfer ingesters.

A surface store holds one mesh **object per hemisphere**, and every format
that describes cortex reduces to the same few things per hemisphere: one
anatomical surface to chunk, other surfaces with the same topology, per-vertex
scalar maps, and per-vertex parcellation codes with a label table.  The
readers produce :class:`HemisphereSurface` values; this module turns any set
of them into a store, so the two ingesters cannot disagree about the layout.

The join key
------------
Every vertex carries ``zv_join_key = hemisphere << 32 | source_vertex``, with
``left = 0`` and ``right = 1`` fixed rather than taken from the object id.
The store reorders vertices into spatial chunks, and a great deal of surface
data arrives later and is indexed by the ORIGINAL vertex number: a CIFTI
dscalar, a metric computed by another tool, a second parcellation.  The key
lets all of it land on the right vertex through the existing streaming join
(:func:`~zarr_vectors_tools.ingest.attach.attach_attributes`), however many
chunks the vertices were scattered across, and the fixed hemisphere codes keep
a right-hemisphere-only store's keys identical to a two-hemisphere store's.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from zarr_vectors.exceptions import IngestError

from zarr_vectors_tools.ingest.attach import DEFAULT_KEY_ATTRIBUTE

__all__ = [
    "HEMISPHERE_CODES",
    "HemisphereSurface",
    "map_name",
    "surface_keys",
    "write_surface_store",
]

#: Fixed hemisphere codes used in the join key.  Never the object id: a store
#: holding only the right hemisphere still keys it as 1.
HEMISPHERE_CODES: dict[str, int] = {"left": 0, "right": 1}

#: CIFTI brain-structure name for each hemisphere's cortex.
CIFTI_STRUCTURES: dict[str, str] = {
    "left": "CIFTI_STRUCTURE_CORTEX_LEFT",
    "right": "CIFTI_STRUCTURE_CORTEX_RIGHT",
}

_KEY_SHIFT = 32

#: Filename suffixes that say what kind of file it is, not what it measures.
_SUFFIXES = re.compile(
    r"(\.(surf|shape|func|label|time|dscalar|dlabel|dtseries|ptseries|pscalar))*"
    r"\.(gii|nii)$",
    re.IGNORECASE,
)
#: Tokens that identify the subject, hemisphere, mesh or file type.
_DROP_TOKENS = {
    "l", "r", "lh", "rh", "left", "right", "fs", "lr", "fslr", "fsaverage",
    "fsaverage5", "fsaverage6", "fsnative", "native", "msmall", "msmsulc",
    "surf", "shape", "func", "label", "gii", "nii",
}
_DROP_PREFIXES = (
    "sub-", "ses-", "hemi-", "space-", "den-", "run-", "acq-", "task-", "res-",
)


def _identifying(token: str) -> bool:
    """Whether a filename token names the subject, hemisphere or mesh."""
    t = token.lower()
    return (
        re.fullmatch(r"\d+k?", t) is not None
        or t in _DROP_TOKENS
        or t.startswith(_DROP_PREFIXES)
        or "fs_lr" in t
        or t.startswith("fsaverage")
    )


def map_name(filename: str) -> str:
    """A clean attribute name from a BIDS, HCP or FreeSurfer style filename.

    ``sub-01_hemi-L_thickness.shape.gii``,
    ``100307.L.thickness.32k_fs_LR.shape.gii`` and
    ``100307.thickness.32k_fs_LR.dscalar.nii`` all become ``thickness``;
    a BIDS ``desc-`` entity wins where there is one, because it is the
    author's own name for the map.

    The two conventions separate differently.  HCP and FreeSurfer put dots
    between parts and allow underscores inside one (``MyelinMap_BC``,
    ``32k_fs_LR``), so a dotted name is split on dots only and what remains
    is joined (``lh.aparc.a2009s`` -> ``aparc_a2009s``).  BIDS puts
    underscores between entities and the measurement last.
    """
    stem = _SUFFIXES.sub("", filename)
    desc = re.search(r"(?:^|[._])desc-([A-Za-z0-9]+)", stem)
    if desc:
        return desc.group(1)
    if "." in stem:
        keep = [t for t in stem.split(".") if t and not _identifying(t)]
        if keep:
            return "_".join(keep)
    tokens = [t for t in re.split(r"[._]", stem) if t]
    keep = [t for t in tokens if not _identifying(t)]
    return keep[-1] if keep else stem


def surface_keys(
    hemisphere: str, vertex_indices: npt.ArrayLike,
) -> npt.NDArray[np.int64]:
    """Join keys for ``vertex_indices`` of ``hemisphere``.

    Args:
        hemisphere: ``"left"`` or ``"right"``.
        vertex_indices: Vertex numbers in the SOURCE mesh.

    Returns:
        ``(N,)`` int64 keys matching the store's ``zv_join_key`` attribute.
    """
    try:
        code = HEMISPHERE_CODES[hemisphere]
    except KeyError:
        raise IngestError(
            f"hemisphere must be 'left' or 'right', got {hemisphere!r}"
        ) from None
    idx = np.asarray(vertex_indices, dtype=np.int64)
    return (np.int64(code) << np.int64(_KEY_SHIFT)) | idx


@dataclass
class HemisphereSurface:
    """One hemisphere's cortical surface, as every surface reader produces it.

    Attributes:
        hemisphere: ``"left"`` or ``"right"``.
        vertices: ``(V, 3)`` geometry positions, in the store's space.
        faces: ``(F, 3)`` triangles indexing ``vertices``.
        geometry: Name of the surface ``vertices`` came from.
        scalars: ``{name: (V,) or (V, C)}`` continuous maps.
        labels: ``{name: (V,) int}`` parcellation codes.
        label_tables: ``{name: {code: {"name", "rgba"}}}`` for ``labels``.
        alternates: ``{surface name: (V, 3)}`` other surfaces of the same
            topology (white, pial, inflated, sphere...).
        sources: ``{attribute: description}`` recorded in the header.
    """

    hemisphere: str
    vertices: npt.NDArray[np.floating]
    faces: npt.NDArray[np.integer]
    geometry: str = "surface"
    scalars: dict[str, npt.NDArray] = field(default_factory=dict)
    labels: dict[str, npt.NDArray] = field(default_factory=dict)
    label_tables: dict[str, dict[Any, dict[str, Any]]] = field(
        default_factory=dict,
    )
    alternates: dict[str, npt.NDArray] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)

    @property
    def n_vertices(self) -> int:
        return int(np.asarray(self.vertices).shape[0])

    def validate(self) -> None:
        """Refuse a hemisphere whose parts do not describe one mesh.

        Every message names the offending array and both lengths, because
        the usual cause is a map from a different mesh resolution (a 32k
        fs_LR metric beside a native or 164k surface), and that is only
        diagnosable from the numbers.
        """
        if self.hemisphere not in HEMISPHERE_CODES:
            raise IngestError(
                f"hemisphere must be 'left' or 'right', got {self.hemisphere!r}"
            )
        v = np.asarray(self.vertices)
        f = np.asarray(self.faces)
        if v.ndim != 2 or v.shape[1] != 3:
            raise IngestError(
                f"{self.hemisphere} {self.geometry}: vertices must be (V, 3), "
                f"got {v.shape}"
            )
        if f.ndim != 2 or f.shape[1] != 3:
            raise IngestError(
                f"{self.hemisphere} {self.geometry}: faces must be (F, 3) "
                f"triangles, got {f.shape}"
            )
        if f.size and (int(f.min()) < 0 or int(f.max()) >= v.shape[0]):
            raise IngestError(
                f"{self.hemisphere} {self.geometry}: faces index vertices "
                f"{int(f.min())}..{int(f.max())} but the surface has "
                f"{v.shape[0]}"
            )
        n = v.shape[0]
        for kind, arrays in (("scalar", self.scalars), ("label", self.labels)):
            for name, values in arrays.items():
                length = int(np.asarray(values).shape[0])
                if length != n:
                    raise IngestError(
                        f"{self.hemisphere} {kind} {name!r} has {length} "
                        f"values but the {self.geometry} surface has {n} "
                        f"vertices -- it belongs to a different mesh "
                        f"(check the resolution: 32k, 164k or native)"
                    )
        for name, coords in self.alternates.items():
            shape = np.asarray(coords).shape
            if shape != (n, 3):
                raise IngestError(
                    f"{self.hemisphere} surface {name!r} has shape {shape} but "
                    f"the {self.geometry} geometry is ({n}, 3); surfaces of "
                    f"one hemisphere must share a topology"
                )


def _complete_columns(
    hemispheres: list[HemisphereSurface],
    attr: str,
    fill: Any,
) -> tuple[dict[str, npt.NDArray], list[str]]:
    """Concatenate one kind of per-vertex array across hemispheres.

    A map present for only one hemisphere is filled for the other rather than
    dropped, so the column stays aligned with every vertex.  Returns the
    columns and the names that needed filling, for the summary.
    """
    names: list[str] = []
    for hemi in hemispheres:
        for name in getattr(hemi, attr):
            if name not in names:
                names.append(name)
    columns: dict[str, npt.NDArray] = {}
    filled: list[str] = []
    for name in names:
        present = [
            np.asarray(getattr(h, attr)[name]) for h in hemispheres
            if name in getattr(h, attr)
        ]
        exemplar = present[0]
        parts = []
        for hemi in hemispheres:
            values = getattr(hemi, attr).get(name)
            if values is None:
                shape = (hemi.n_vertices, *exemplar.shape[1:])
                values = np.full(shape, fill, dtype=exemplar.dtype)
                if name not in filled:
                    filled.append(name)
            parts.append(np.asarray(values, dtype=exemplar.dtype))
        columns[name] = np.concatenate(parts, axis=0)
    return columns, filled


def write_surface_store(
    output_path: str | Path,
    hemispheres: list[HemisphereSurface],
    chunk_shape: tuple[float, ...],
    *,
    source: str,
    space: str,
    bin_shape: tuple[float, ...] | None = None,
    dtype: str = "float32",
    c_ras: npt.ArrayLike | None = None,
    extra_header: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write hemispheres as one surface store.

    Args:
        output_path: New store to create.
        hemispheres: One or two :class:`HemisphereSurface`, at most one each.
        chunk_shape: Spatial chunk size in the geometry's units (mm).
        source: ``"gifti"`` or ``"freesurfer"``, recorded in the header.
        space: Coordinate space of the geometry, recorded in the header.
        bin_shape: Optional intra-chunk sub-binning.
        dtype: Position dtype.
        c_ras: FreeSurfer ``c_ras`` offset, recorded when known.
        extra_header: Additional header fields.

    Returns:
        The mesh writer's summary plus ``hemispheres``, ``geometry``,
        ``scalars``, ``labels``, ``alternates`` and ``filled`` (maps present
        for only one hemisphere, completed with a fill value).
    """
    from zarr_vectors.building import (
        create_groupings_array,
        get_resolution_level,
        open_store,
        write_groupings,
    )
    from zarr_vectors.constants import GROUPS
    from zarr_vectors.types.meshes import write_mesh

    from zarr_vectors_tools.headers.formats import SurfaceHeader
    from zarr_vectors_tools.headers.registry import HeaderRegistry
    from zarr_vectors_tools.ingest.attach import attach_attributes

    if not hemispheres:
        raise IngestError("no hemisphere surface to write")
    seen = [h.hemisphere for h in hemispheres]
    if len(set(seen)) != len(seen):
        raise IngestError(
            f"each hemisphere may appear once; got {seen}. Pass one surface "
            f"per hemisphere as the geometry and the rest as alternates."
        )
    geometries = {h.geometry for h in hemispheres}
    if len(geometries) > 1:
        raise IngestError(
            f"both hemispheres must use the same surface as geometry, got "
            f"{sorted(geometries)}; mixing a pial left with a white right "
            f"makes every cross-hemisphere comparison wrong"
        )
    for hemi in hemispheres:
        hemi.validate()

    ordered = sorted(hemispheres, key=lambda h: HEMISPHERE_CODES[h.hemisphere])

    # ---- geometry, objects, keys ----------------------------------------
    positions: list[npt.NDArray] = []
    faces: list[npt.NDArray] = []
    object_ids: list[npt.NDArray] = []
    keys: list[npt.NDArray] = []
    offset = 0
    for object_id, hemi in enumerate(ordered):
        n = hemi.n_vertices
        positions.append(np.asarray(hemi.vertices, dtype=np.float64))
        faces.append(np.asarray(hemi.faces, dtype=np.int64) + offset)
        object_ids.append(np.full(n, object_id, dtype=np.int64))
        keys.append(surface_keys(hemi.hemisphere, np.arange(n)))
        offset += n

    scalars, filled_scalars = _complete_columns(ordered, "scalars", np.nan)
    labels, filled_labels = _complete_columns(ordered, "labels", -1)
    alternates, filled_alternates = _complete_columns(
        ordered, "alternates", np.nan,
    )
    key_column = np.concatenate(keys)

    # Single-column attributes go straight to the mesh writer.  Multi-column
    # ones do NOT: core's write_mesh records a (V, C) vertex attribute as
    # 1-column rows, so it reads back flattened and a C-column read is
    # refused.  They are attached by key afterwards instead, through a path
    # that records the width -- and the key makes that independent of the
    # order the mesh writer put the vertices in.
    single: dict[str, npt.NDArray] = {DEFAULT_KEY_ATTRIBUTE: key_column}
    multi: dict[str, npt.NDArray] = {}
    for name, column in {**scalars, **labels}.items():
        (multi if np.asarray(column).ndim > 1 else single)[name] = column
    for surface, column in alternates.items():
        multi[f"coords_{surface}"] = np.asarray(column, dtype=np.float32)

    summary = dict(write_mesh(
        str(output_path),
        np.concatenate(positions, axis=0),
        np.concatenate(faces, axis=0),
        chunk_shape=tuple(float(c) for c in chunk_shape),
        bin_shape=bin_shape,
        vertex_attributes=single,
        object_ids=np.concatenate(object_ids),
        dtype=dtype,
    ))

    if multi:
        attach_attributes(
            str(output_path), multi, keys=key_column, missing="error",
        )

    # ---- name the objects -------------------------------------------------
    # Objects already ARE hemispheres; a group per hemisphere gives them
    # names a reader can ask for, and lets ``compose.split --by groups`` cut
    # a store into one per hemisphere.
    level_group = get_resolution_level(open_store(str(output_path), mode="r+"), 0)
    create_groupings_array(level_group)
    write_groupings(
        level_group, {i: [i] for i in range(len(ordered))},
    )
    try:
        meta = dict(level_group.read_array_meta(GROUPS) or {})
        meta["group_names"] = [h.hemisphere for h in ordered]
        level_group.write_array_meta(GROUPS, meta)
    except Exception:  # noqa: BLE001 - names are additive, never load-bearing
        pass

    # ---- header -----------------------------------------------------------
    label_tables: dict[str, dict[str, dict[str, Any]]] = {}
    for hemi in ordered:
        for name, table in hemi.label_tables.items():
            merged = label_tables.setdefault(name, {})
            for code, entry in table.items():
                merged.setdefault(str(int(code)), dict(entry))
    sources: dict[str, str] = {}
    for hemi in ordered:
        sources.update(hemi.sources)

    header = SurfaceHeader(
        source=source,
        space=space,
        geometry=ordered[0].geometry,
        hemispheres=[
            {
                "object_id": object_id,
                "hemisphere": hemi.hemisphere,
                "structure": CIFTI_STRUCTURES[hemi.hemisphere],
                "n_vertices": hemi.n_vertices,
                "n_faces": int(np.asarray(hemi.faces).shape[0]),
            }
            for object_id, hemi in enumerate(ordered)
        ],
        alternates={f"coords_{name}": name for name in alternates},
        scalars={name: sources.get(name, name) for name in scalars},
        label_tables=label_tables,
        key_attribute=DEFAULT_KEY_ATTRIBUTE,
        c_ras=None if c_ras is None else [float(v) for v in np.asarray(c_ras)],
        extra=dict(extra_header or {}),
    )
    HeaderRegistry(str(output_path)).add("surface", header)

    summary.update({
        "hemispheres": [h.hemisphere for h in ordered],
        "geometry": ordered[0].geometry,
        "space": space,
        "scalars": sorted(scalars),
        "labels": sorted(labels),
        "alternates": sorted(alternates),
        "filled": sorted(set(filled_scalars + filled_labels + filled_alternates)),
    })
    return summary
