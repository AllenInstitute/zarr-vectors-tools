"""Ingest GIFTI cortical surfaces, metrics and parcellations.

GIFTI is how HCP, fMRIPrep and Connectome Workbench exchange surfaces, and a
subject is always a SET of files rather than one:

.. code-block:: text

    sub-01_hemi-L_midthickness.surf.gii     geometry
    sub-01_hemi-L_inflated.surf.gii         same mesh, another shape
    sub-01_hemi-L_thickness.shape.gii       per-vertex scalar
    sub-01_hemi-L_desc-aparc.label.gii      per-vertex parcellation
    ... and the same for hemi-R

:func:`ingest_gifti` takes that set -- a directory, a list of files, or a
single surface -- sorts every file by what it is and which hemisphere it
describes, and writes one surface store (see
:mod:`~zarr_vectors_tools.convert.ingest._surface_store` for the layout).

How each file is classified
---------------------------
- **Kind.** A file with a ``POINTSET`` array is a surface; one with a
  ``LABEL`` array is a parcellation; anything else holding per-vertex values
  (``SHAPE``, ``NONE``, statistics, ``TIME_SERIES``) is a scalar map.
- **Hemisphere.** From the image's ``AnatomicalStructurePrimary`` metadata,
  falling back to the filename (``hemi-L``, ``.L.``, ``lh.``...).
- **Surface.** ``GeometricType`` first for the shapes that are not anatomy
  (inflated, sphere, flat), because HCP writes an inflated surface with
  ``AnatomicalStructureSecondary=MidThickness`` -- it was inflated FROM the
  midthickness -- and reading that field first would chunk a balloon.  Then
  the filename, then ``AnatomicalStructureSecondary``.

Coordinates are stored as the file has them.  GIFTI declares a dataspace but
positions are already in it, so the dataspace is recorded, not applied.

Requires ``nibabel``::

    pip install 'zarr-vectors-tools[surfaces]'
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from zarr_vectors.exceptions import IngestError
from zarr_vectors.typing import BinShape, ChunkShape

from zarr_vectors_tools.convert.ingest._surface_store import (
    HemisphereSurface,
    map_name,
    write_surface_store,
)
from zarr_vectors_tools.convert.ingest._tabular import sanitise_name

__all__ = [
    "ANATOMICAL_PREFERENCE",
    "GiftiFile",
    "classify_gifti",
    "ingest_gifti",
]

#: Surfaces that describe cortex in anatomical space, in the order they are
#: preferred as the chunked geometry.  Midthickness first: it lies between
#: the white and pial boundaries, so it is the least biased surface to
#: project volume data onto and the one HCP analyses use.
ANATOMICAL_PREFERENCE: tuple[str, ...] = (
    "midthickness", "pial", "white", "smoothwm", "orig", "surface",
)

#: Shapes that are NOT in anatomical space.  Never chosen as geometry while
#: an anatomical surface exists: querying an inflated brain by bounding box
#: answers a question about a balloon.
_NON_ANATOMICAL = ("inflated", "very_inflated", "sphere", "flat")

_GEOMETRIC_TYPES = {
    "inflated": "inflated",
    "veryinflated": "very_inflated",
    "spherical": "sphere",
    "flat": "flat",
}

_SECONDARY_STRUCTURES = {
    "midthickness": "midthickness",
    "pial": "pial",
    "graywhite": "white",
}

#: Filename tokens, most specific first so ``very_inflated`` is not read as
#: ``inflated``.
_FILENAME_SURFACES: tuple[tuple[str, str], ...] = (
    (r"very[_-]?inflated", "very_inflated"),
    (r"mid[_-]?thickness|graymid", "midthickness"),
    (r"inflated", "inflated"),
    (r"sphere|spherical", "sphere"),
    (r"flat", "flat"),
    (r"smoothwm", "smoothwm"),
    (r"pial", "pial"),
    (r"white|graywhite", "white"),
    (r"orig", "orig"),
)

_HEMI_META = {
    "cortexleft": "left", "cortex_left": "left", "left": "left",
    "cortexright": "right", "cortex_right": "right", "right": "right",
}

_HEMI_FILENAME: tuple[tuple[str, str], ...] = (
    (r"(^|[._-])hemi-(l|left)([._-]|$)", "left"),
    (r"(^|[._-])hemi-(r|right)([._-]|$)", "right"),
    (r"(^|[._])(lh)([._]|$)", "left"),
    (r"(^|[._])(rh)([._]|$)", "right"),
    (r"(^|[._])(l)([._]|$)", "left"),
    (r"(^|[._])(r)([._]|$)", "right"),
    (r"(^|[._-])left([._-]|$)", "left"),
    (r"(^|[._-])right([._-]|$)", "right"),
)

@dataclass
class GiftiFile:
    """One classified GIFTI file."""

    path: Path
    kind: str                 # "surface" | "scalar" | "label"
    hemisphere: str | None
    surface: str | None       # for kind == "surface"
    image: Any = field(repr=False, default=None)


def _load(path: Path):
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise IngestError(
            "reading GIFTI needs nibabel. Install with: "
            "pip install 'zarr-vectors-tools[surfaces]'"
        ) from exc
    try:
        return nib.load(str(path))
    except Exception as exc:
        raise IngestError(f"failed to read GIFTI '{path}': {exc}") from exc


def _intent(darray) -> str:
    import nibabel as nib

    return str(nib.nifti1.intent_codes.label.get(darray.intent, "unknown"))


def _hemisphere(image, path: Path) -> str | None:
    primary = str(image.meta.get("AnatomicalStructurePrimary", "") or "")
    hemi = _HEMI_META.get(primary.strip().lower())
    if hemi:
        return hemi
    name = path.name.lower()
    for pattern, value in _HEMI_FILENAME:
        if re.search(pattern, name):
            return value
    return None


def _surface_name(image, path: Path) -> str:
    pointset = next(
        (d for d in image.darrays if _intent(d) == "pointset"), None,
    )
    meta = dict(pointset.meta) if pointset is not None else {}
    name = path.name.lower()
    # Some writers tag a very-inflated surface GeometricType=Inflated; the
    # filename is the only thing that tells it from the inflated one.
    if re.search(_FILENAME_SURFACES[0][0], name):
        return _FILENAME_SURFACES[0][1]
    geometric = str(meta.get("GeometricType", "") or "").strip().lower()
    if geometric in _GEOMETRIC_TYPES:
        return _GEOMETRIC_TYPES[geometric]
    for pattern, value in _FILENAME_SURFACES:
        if re.search(pattern, name):
            return value
    secondary = str(meta.get("AnatomicalStructureSecondary", "") or "")
    return _SECONDARY_STRUCTURES.get(secondary.strip().lower(), "surface")


def classify_gifti(path: str | Path) -> GiftiFile:
    """Read one GIFTI file and say what it is.

    Args:
        path: A ``.gii`` file.

    Returns:
        A :class:`GiftiFile` with the loaded image attached.

    Raises:
        IngestError: If the file cannot be read or holds no per-vertex data.
    """
    path = Path(path)
    image = _load(path)
    intents = [_intent(d) for d in image.darrays]
    if not intents:
        raise IngestError(f"GIFTI '{path.name}' holds no data arrays")
    if "pointset" in intents:
        if "triangle" not in intents:
            raise IngestError(
                f"GIFTI '{path.name}' has vertex positions but no triangles; "
                f"a point set is not a surface"
            )
        kind = "surface"
    elif "label" in intents:
        kind = "label"
    else:
        kind = "scalar"
    return GiftiFile(
        path=path,
        kind=kind,
        hemisphere=_hemisphere(image, path),
        surface=_surface_name(image, path) if kind == "surface" else None,
        image=image,
    )


def _dataspace(image) -> str | None:
    import nibabel as nib

    for darray in image.darrays:
        coordsys = getattr(darray, "coordsys", None)
        if coordsys is None:
            continue
        code = getattr(coordsys, "dataspace", 0)
        label = nib.nifti1.xform_codes.label.get(code)
        if label and label != "unknown":
            return str(label)
    return None


def _scalar_maps(
    entry: GiftiFile, used: set[str],
) -> dict[str, npt.NDArray]:
    """One attribute per named map, or one multi-column attribute otherwise.

    Distinctly named arrays (a ``thickness`` and a ``curvature`` in one file)
    stay separate attributes.  Unnamed arrays, and any time series, are one
    measurement over several columns and become a single ``(V, T)``
    attribute named after the file -- a 1200-frame resting-state run is not
    1200 attributes.
    """
    darrays = list(entry.image.darrays)
    names = [str(d.meta.get("Name", "") or "").strip() for d in darrays]
    series = any(_intent(d) == "time series" for d in darrays)
    distinct = all(names) and len(set(names)) == len(names)

    if len(darrays) == 1:
        values = np.asarray(darrays[0].data)
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        return {sanitise_name(map_name(entry.path.name), used): values}
    if distinct and not series:
        return {
            sanitise_name(name, used): np.asarray(d.data)
            for name, d in zip(names, darrays)
        }
    stacked = np.stack([np.asarray(d.data).ravel() for d in darrays], axis=1)
    return {sanitise_name(map_name(entry.path.name), used): stacked}


def _label_maps(
    entry: GiftiFile, used: set[str],
) -> tuple[dict[str, npt.NDArray], dict[str, dict[int, dict[str, Any]]]]:
    table: dict[int, dict[str, Any]] = {}
    labeltable = getattr(entry.image, "labeltable", None)
    if labeltable is not None:
        for label in labeltable.labels:
            rgba = label.rgba
            table[int(label.key)] = {
                "name": str(label.label),
                "rgba": [float(c) if c is not None else 0.0 for c in rgba],
            }
    darrays = [d for d in entry.image.darrays if _intent(d) == "label"]
    maps: dict[str, npt.NDArray] = {}
    tables: dict[str, dict[int, dict[str, Any]]] = {}
    for index, darray in enumerate(darrays):
        base = (
            map_name(entry.path.name) if len(darrays) == 1
            else str(darray.meta.get("Name", "") or f"{map_name(entry.path.name)}_{index}")
        )
        name = sanitise_name(base, used)
        maps[name] = np.asarray(darray.data).astype(np.int32).ravel()
        tables[name] = table
    return maps, tables


def _collect_inputs(input_path: Any) -> list[Path]:
    if isinstance(input_path, (list, tuple)):
        paths = [Path(p) for p in input_path]
    else:
        path = Path(input_path)
        if path.is_dir():
            paths = sorted(p for p in path.iterdir() if p.suffix.lower() == ".gii")
            if not paths:
                raise IngestError(f"no .gii files in {path}")
        else:
            paths = [path]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise IngestError(f"input file(s) not found: {missing}")
    return paths


def ingest_gifti(
    input_path: str | Path | list[str | Path],
    output_path: str | Path,
    chunk_shape: ChunkShape,
    *,
    bin_shape: BinShape | None = None,
    dtype: str = "float32",
    geometry: str | None = None,
    hemisphere: str | None = None,
) -> dict[str, Any]:
    """Ingest a set of GIFTI files into one cortical surface store.

    Args:
        input_path: A directory of ``.gii`` files, a list of files, or a
            single surface file.
        output_path: Store to create.
        chunk_shape: Spatial chunk size in mm (e.g. ``(20, 20, 20)``: a whole
            cortex is roughly 140 x 180 x 120 mm).
        bin_shape: Optional intra-chunk sub-binning.
        dtype: Position dtype.
        geometry: Which surface to chunk (``"pial"``, ``"white"``...).
            Default: the first available of ``midthickness``, ``pial``,
            ``white``, ``smoothwm``, ``orig``.  Every other surface of the
            same hemisphere becomes a ``coords_<name>`` attribute.
        hemisphere: Assign every file with no hemisphere information to this
            one (``"left"``/``"right"``).  Only needed for files that carry
            neither ``AnatomicalStructurePrimary`` nor a hemisphere token in
            their name.

    Returns:
        The surface-store summary: mesh counts plus ``hemispheres``,
        ``geometry``, ``space``, ``scalars``, ``labels``, ``alternates``,
        ``filled`` and ``files`` (how each input was classified).

    Raises:
        IngestError: If a file's hemisphere cannot be determined, a map does
            not match its surface's vertex count, or a hemisphere has no
            surface to use as geometry.
    """
    paths = _collect_inputs(input_path)
    if hemisphere is not None and hemisphere not in ("left", "right"):
        raise IngestError(
            f"hemisphere must be 'left' or 'right', got {hemisphere!r}"
        )

    entries = [classify_gifti(p) for p in paths]
    for entry in entries:
        if entry.hemisphere is None:
            if hemisphere is None:
                raise IngestError(
                    f"cannot tell which hemisphere '{entry.path.name}' "
                    f"describes: it has no AnatomicalStructurePrimary and no "
                    f"hemisphere in its name (hemi-L, .L., lh.). Rename it, "
                    f"or pass hemisphere='left' or 'right'."
                )
            entry.hemisphere = hemisphere

    dataspaces = {_dataspace(e.image) for e in entries if e.kind == "surface"}
    dataspaces.discard(None)
    space = dataspaces.pop() if len(dataspaces) == 1 else (
        "unknown" if not dataspaces else "mixed"
    )

    hemispheres: list[HemisphereSurface] = []
    for hemi in ("left", "right"):
        mine = [e for e in entries if e.hemisphere == hemi]
        if not mine:
            continue
        # Names are made unique within a hemisphere only: the left and right
        # "thickness" files are one measurement and must share one attribute.
        used: set[str] = {"zv_join_key"}
        surfaces: dict[str, GiftiFile] = {}
        for entry in (e for e in mine if e.kind == "surface"):
            clash = surfaces.get(entry.surface)
            if clash is not None:
                raise IngestError(
                    f"{hemi} hemisphere: both '{clash.path.name}' and "
                    f"'{entry.path.name}' look like the {entry.surface} "
                    f"surface. Ingest them separately, or pass the files to "
                    f"keep as a list."
                )
            surfaces[entry.surface] = entry
        if not surfaces:
            raise IngestError(
                f"{hemi} hemisphere has maps "
                f"({', '.join(e.path.name for e in mine)}) but no .surf.gii "
                f"to put them on"
            )
        if geometry is not None:
            if geometry not in surfaces:
                raise IngestError(
                    f"geometry={geometry!r} requested but the {hemi} "
                    f"hemisphere has only {sorted(surfaces)}"
                )
            chosen = geometry
        else:
            anatomical = [s for s in ANATOMICAL_PREFERENCE if s in surfaces]
            if not anatomical:
                raise IngestError(
                    f"{hemi} hemisphere has only non-anatomical surfaces "
                    f"({sorted(surfaces)}); chunking an inflated or spherical "
                    f"surface puts the spatial index on a shape that is not "
                    f"the brain. Pass geometry=... to do it anyway."
                )
            chosen = anatomical[0]

        geo = surfaces[chosen].image
        vertices = np.asarray(geo.agg_data("pointset"), dtype=np.float64)
        faces = np.asarray(geo.agg_data("triangle"), dtype=np.int64)

        surface = HemisphereSurface(
            hemisphere=hemi, vertices=vertices, faces=faces, geometry=chosen,
        )
        for name, entry in surfaces.items():
            if name == chosen:
                continue
            surface.alternates[name] = np.asarray(
                entry.image.agg_data("pointset"), dtype=np.float32,
            )
        for entry in (e for e in mine if e.kind == "scalar"):
            for name, values in _scalar_maps(entry, used).items():
                surface.scalars[name] = values
                surface.sources[name] = entry.path.name
        for entry in (e for e in mine if e.kind == "label"):
            maps, tables = _label_maps(entry, used)
            surface.labels.update(maps)
            surface.label_tables.update(tables)
            for name in maps:
                surface.sources[name] = entry.path.name
        hemispheres.append(surface)

    summary = write_surface_store(
        output_path, hemispheres, chunk_shape,
        source="gifti", space=space, bin_shape=bin_shape, dtype=dtype,
    )
    summary["files"] = {
        e.path.name: {
            "kind": e.kind, "hemisphere": e.hemisphere, "surface": e.surface,
        }
        for e in entries
    }
    return summary
