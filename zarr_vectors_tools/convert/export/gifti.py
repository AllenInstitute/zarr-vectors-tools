"""Export a cortical surface store back to GIFTI.

A surface store (written by
:func:`~zarr_vectors_tools.convert.ingest.gifti.ingest_gifti` or
:func:`~zarr_vectors_tools.convert.ingest.freesurfer.ingest_freesurfer`)
goes back out as the set of files Connectome Workbench, FreeView, fMRIPrep
and nilearn read, one directory per store:

.. code-block:: text

    out/
      hemi-L_midthickness.surf.gii    the geometry
      hemi-L_pial.surf.gii            every alternate surface, same triangles
      hemi-L_inflated.surf.gii
      hemi-L_thickness.shape.gii      morphometry
      hemi-L_MyelinMap.func.gii       any other continuous map
      hemi-L_aparc.label.gii          a parcellation, with its label table
      ... and the same for hemi-R

Every vertex is written in the SOURCE mesh's numbering (the store's join key
restores it), so vertex ``i`` of every file is vertex ``i`` of the files the
store was ingested from, and CIFTI data for that mesh still applies.

File names
----------
``[<prefix>_]hemi-<L|R>_<name>.<kind>.gii``.  A BIDS label may only hold
letters and digits, so a name that is not one (``aparc.a2009s``,
``MyelinMap_BC``) is written after a dot instead, the way FreeSurfer and HCP
write it: ``hemi-L.aparc.a2009s.label.gii``.  Either way
:func:`~zarr_vectors_tools.convert.ingest._surface_store.map_name` reads the
attribute name back unchanged, so ingesting the directory again rebuilds the
same store.  Any file for which that is not true (a prefix that is not a
BIDS entity combined with a dotted name, say) is listed in the summary's
``warnings``.

Which kind of file
------------------
- **Surfaces** are ``.surf.gii``: a float32 ``POINTSET`` and an int32
  ``TRIANGLE`` array, with ``GeometricType`` (``Anatomical``, ``Inflated``,
  ``VeryInflated``, ``Spherical``, ``Flat``) and, for the three surfaces
  Workbench names, ``AnatomicalStructureSecondary`` (``MidThickness``,
  ``Pial``, ``GrayWhite``).
- **Parcellations** -- attributes with a label table in the header -- are
  ``.label.gii``: int32 codes under ``NIFTI_INTENT_LABEL`` and the table's
  names and RGBA colours.
- **Continuous maps** are ``.shape.gii`` with ``NIFTI_INTENT_SHAPE`` when
  they describe the shape of cortex, and ``.func.gii`` with
  ``NIFTI_INTENT_NONE`` otherwise, following HCP, which writes thickness,
  curvature, sulcal depth and areal distortion as shape files and myelin
  and functional maps as func files.  :func:`metric_kind` decides, from the
  file the map was ingested from first and its name second.  A CIFTI time
  series keeps ``NIFTI_INTENT_TIME_SERIES`` and its ``TimeStep``.  A map
  with several columns is one data array per column.

Every file carries ``AnatomicalStructurePrimary`` (``CortexLeft`` /
``CortexRight``, from the CIFTI structure the header records for the
hemisphere), which is how Workbench assigns a file to a hemisphere.

Coordinates
-----------
Positions are written exactly as the store holds them, in the space its
header records, and that space goes into each anatomical surface's
coordinate system (``scanner`` as ``NIFTI_XFORM_SCANNER_ANAT``, a GIFTI
dataspace such as ``talairach`` as itself).  A FreeSurfer store ingested in
scanner RAS therefore exports in scanner RAS, lined up with the T1 and any
tractogram of the subject, which is also how fMRIPrep writes its GIFTI
surfaces.  ``c_ras`` is not re-applied or removed, and FreeSurfer's
``VolGeom*`` metadata is deliberately not written: FreeSurfer tools read
those as "these coordinates are surface RAS, add ``c_ras``", which would
shift scanner-space coordinates a second time.  A store kept in FreeSurfer
surface RAS has no NIfTI code for that space and is written as
``NIFTI_XFORM_UNKNOWN``.  Inflated, spherical and flat surfaces are in no
anatomical space and are always written as unknown.

Levels
------
``level=0`` is the mesh as ingested.  A coarser level can be exported too:
it is a decimated mesh with fewer vertices, and its files are consistent
with each other (surfaces, maps and labels share its vertices), but its
vertex numbering is its own -- metrics, CIFTI files and label files made
for the full-resolution mesh do not apply to it.

What does not round trip
------------------------
GIFTI holds float32 and int32 only, so a float64 map is written as float32.
Triangles come back as the same set of triangles, with the same winding,
but not necessarily in the source file's order.  The store does not record
a map's original intent, so a GIFTI time series (other than one attached
from CIFTI) is written as ``NIFTI_INTENT_NONE``.

Requires ``nibabel``::

    pip install 'zarr-vectors-tools[surfaces]'
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from zarr_vectors.exceptions import ExportError

from zarr_vectors_tools.convert.ingest._surface_store import map_name
from zarr_vectors_tools.convert.ingest.gifti import ANATOMICAL_PREFERENCE

__all__ = [
    "export_gifti",
    "gifti_filename",
    "metric_kind",
]

_HEMISPHERE_ALIASES = {
    "left": "left", "lh": "left", "l": "left",
    "right": "right", "rh": "right", "r": "right",
}
_TAGS = {"left": "L", "right": "R"}
_PRIMARY = {"left": "CortexLeft", "right": "CortexRight"}

#: ``GeometricType`` for the shapes that are not anatomy.  These are also
#: the values the GIFTI ingest reads back into these names.
_NON_ANATOMICAL = {
    "inflated": "Inflated",
    "very_inflated": "VeryInflated",
    "sphere": "Spherical",
    "flat": "Flat",
}
#: ``AnatomicalStructureSecondary`` for the surfaces Workbench has a name for.
_SECONDARY = {"midthickness": "MidThickness", "pial": "Pial", "white": "GrayWhite"}

#: Surface names the GIFTI ingest recognises from a file and so reads back
#: unchanged; anything else re-ingests under whichever of these its name
#: resembles, or as ``surface``.
_RECOGNISED_SURFACES = frozenset(ANATOMICAL_PREFERENCE) | frozenset(_NON_ANATOMICAL)

#: Words that mark a map as describing the shape of cortex.  Substrings, so
#: HCP's ``corrThickness``, ``curvature`` and ``ArealDistortion`` match too.
_MORPHOMETRY = ("thickness", "curv", "sulc", "area", "volume", "jacobian")

_BIDS_LABEL = re.compile(r"[A-Za-z0-9]+")
#: A FreeSurfer per-vertex file (``lh.thickness``), as the ingest records it.
_FREESURFER_MORPH = re.compile(r"^[lr]h\.[^/\\]+$")

_INTENT_SHAPE = "NIFTI_INTENT_SHAPE"
_INTENT_NONE = "NIFTI_INTENT_NONE"
_INTENT_SERIES = "NIFTI_INTENT_TIME_SERIES"


def _nibabel():
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ExportError(
            "writing GIFTI needs nibabel. Install with: "
            "pip install 'zarr-vectors-tools[surfaces]'"
        ) from exc
    return nib


def metric_kind(
    name: str,
    source: str | None = None,
    *,
    series: bool = False,
) -> tuple[str, str]:
    """The file kind and NIfTI intent a continuous map is written with.

    The file the map came from decides first, because it records what the
    producing tool thought the map was: a ``.shape.gii`` stays a shape file
    and a ``.func.gii`` a func file, and a FreeSurfer ``?h.<map>`` file is
    morphometry by definition.  With no such evidence (a CIFTI ``dscalar``,
    or a map attached by key) the name decides: thickness, curvature,
    sulcal depth, area, volume and Jacobian maps are shape, everything else
    is func.

    Args:
        name: The attribute name.
        source: The source description the header records for it, usually
            the file it was ingested from.
        series: Whether the header records it as a time series.

    Returns:
        ``(kind, intent)``: ``("shape", "NIFTI_INTENT_SHAPE")``,
        ``("func", "NIFTI_INTENT_NONE")`` or
        ``("func", "NIFTI_INTENT_TIME_SERIES")``.
    """
    if series:
        return "func", _INTENT_SERIES
    described = (source or "").lower()
    if ".shape." in described:
        return "shape", _INTENT_SHAPE
    if ".time." in described:
        return "func", _INTENT_SERIES
    if ".func." in described:
        return "func", _INTENT_NONE
    if _FREESURFER_MORPH.match(described) and not described.endswith(".annot"):
        return "shape", _INTENT_SHAPE
    if any(word in name.lower() for word in _MORPHOMETRY):
        return "shape", _INTENT_SHAPE
    return "func", _INTENT_NONE


def gifti_filename(
    hemisphere: str, name: str, kind: str, prefix: str | None = None,
) -> str:
    """The file name one surface or map is written under.

    Args:
        hemisphere: ``"left"`` or ``"right"``.
        name: Surface or attribute name.
        kind: ``"surf"``, ``"shape"``, ``"func"`` or ``"label"``.
        prefix: Optional leading part, e.g. ``"sub-01"``.

    Returns:
        ``[<prefix>_]hemi-<L|R>_<name>.<kind>.gii``, or with a dot before a
        name that is not a valid BIDS label (letters and digits only).
    """
    tag = f"hemi-{_TAGS[hemisphere]}"
    stem = f"{prefix}_{tag}" if prefix else tag
    separator = "_" if _BIDS_LABEL.fullmatch(name) else "."
    return f"{stem}{separator}{name}.{kind}.gii"


# ---------------------------------------------------------------------------
# Resolving what to write
# ---------------------------------------------------------------------------

def _surface_header(store_path: Path) -> Any:
    from zarr_vectors_tools.headers.registry import HeaderRegistry

    if not store_path.exists():
        raise ExportError(f"no store at {store_path}")
    registry = HeaderRegistry(str(store_path))
    if not registry.has("surface"):
        raise ExportError(
            f"{store_path} is not a surface store (it has headers "
            f"{sorted(registry.available_formats) or 'none'}); GIFTI export "
            f"reads stores written by the GIFTI or FreeSurfer ingest"
        )
    return registry.get("surface")


def _resolve_hemispheres(
    header: Any, hemispheres: Sequence[str] | str | None,
) -> list[str]:
    present = [h["hemisphere"] for h in header.hemispheres]
    if hemispheres is None:
        return present
    if isinstance(hemispheres, str):
        hemispheres = [hemispheres]
    chosen: list[str] = []
    for value in hemispheres:
        side = _HEMISPHERE_ALIASES.get(str(value).lower())
        if side is None:
            raise ExportError(
                f"hemisphere must be left, right, lh or rh, got {value!r}"
            )
        if side not in present:
            raise ExportError(
                f"the store has no {side} hemisphere; it has {present}"
            )
        if side not in chosen:
            chosen.append(side)
    # Store order, so the summary and the files come out the same way
    # however the caller spelled the list.
    return [side for side in present if side in chosen]


def _resolve_surfaces(
    header: Any, surfaces: Sequence[str] | str | None,
) -> list[tuple[str, str | None]]:
    """``(surface name, coordinate attribute)``; ``None`` is the geometry."""
    available: list[tuple[str, str | None]] = [(header.geometry, None)]
    available += [(surface, attr) for attr, surface in header.alternates.items()]
    if surfaces is None:
        return available
    if isinstance(surfaces, str):
        surfaces = [surfaces]
    by_name = {surface: (surface, attr) for surface, attr in available}
    by_attr = {attr: (surface, attr) for surface, attr in available if attr}
    chosen: list[tuple[str, str | None]] = []
    for value in surfaces:
        entry = by_name.get(value) or by_attr.get(value)
        if entry is None:
            raise ExportError(
                f"the store has no {value!r} surface; it has "
                f"{[surface for surface, _ in available]} "
                f"(geometry {header.geometry!r})"
            )
        if entry not in chosen:
            chosen.append(entry)
    return chosen


def _vertex_attribute_names(store_path: Path, level: int) -> set[str]:
    from zarr_vectors.building import (
        get_resolution_level,
        list_resolution_levels,
        open_store,
    )
    from zarr_vectors.constants import VERTEX_ATTRIBUTES

    root = open_store(str(store_path))
    levels = list_resolution_levels(root)
    if level not in levels:
        raise ExportError(
            f"level {level} is not in {store_path}; it has levels {levels}"
        )
    level_group = get_resolution_level(root, level)
    try:
        return set(level_group[VERTEX_ATTRIBUTES].children())
    except Exception:  # noqa: BLE001 - a level with no vertex attributes
        return set()


def _resolve_maps(
    header: Any,
    at_level: set[str],
    attribute_names: Sequence[str] | str | None,
    level: int,
) -> list[str]:
    """The per-vertex maps to write: every one by default, in a stable order.

    The default is every vertex attribute at the level, not only those the
    header lists, because a metric joined on later by key
    (:func:`~zarr_vectors_tools.convert.ingest.attach.attach_attributes`)
    is a map of the same vertices without a header entry.
    """
    reserved = {header.key_attribute, *header.alternates}
    maps = sorted(name for name in at_level if name not in reserved)
    if attribute_names is None:
        return maps
    if isinstance(attribute_names, str):
        attribute_names = [attribute_names]
    chosen: list[str] = []
    for name in attribute_names:
        if name == header.key_attribute:
            raise ExportError(
                f"{name!r} is the store's join key, not a map; the files are "
                f"already in source vertex order"
            )
        if name in header.alternates:
            raise ExportError(
                f"{name!r} holds a surface's coordinates; ask for it with "
                f"surfaces=[{header.alternates[name]!r}]"
            )
        if name not in maps:
            raise ExportError(
                f"attribute {name!r} is not at level {level}; it has "
                f"{maps or 'no maps'}"
            )
        if name not in chosen:
            chosen.append(name)
    return chosen


def _output_directory(output_dir: str | Path) -> Path:
    out = Path(output_dir)
    if out.suffix.lower() == ".gii":
        raise ExportError(
            f"{out} names a file, but GIFTI export writes a directory: one "
            f".gii per surface and per map"
        )
    if out.exists() and not out.is_dir():
        raise ExportError(f"{out} exists and is not a directory")
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Building images
# ---------------------------------------------------------------------------

def _dataspace_code(nib: Any, space: str) -> int:
    """The NIfTI xform code for a header's space; 0 when NIfTI has none."""
    try:
        return int(nib.nifti1.xform_codes.code[space])
    except KeyError:
        return 0


def _primary_structure(entry: dict[str, Any]) -> str:
    """GIFTI's ``AnatomicalStructurePrimary`` for a header hemisphere entry.

    Taken from the CIFTI structure the header records, because GIFTI's
    names are the same words (``CIFTI_STRUCTURE_CORTEX_LEFT`` is
    ``CortexLeft``); the hemisphere is the fallback for a header without one.
    """
    structure = str(entry.get("structure") or "")
    if structure.startswith("CIFTI_STRUCTURE_"):
        words = structure.removeprefix("CIFTI_STRUCTURE_").split("_")
        return "".join(word.capitalize() for word in words)
    return _PRIMARY[entry["hemisphere"]]


def _image(nib: Any, primary: str) -> Any:
    image = nib.gifti.GiftiImage()
    image.meta["AnatomicalStructurePrimary"] = primary
    return image


def _surface_image(
    nib: Any,
    primary: str,
    surface: str,
    coords: npt.NDArray,
    faces: npt.NDArray,
    dataspace: int,
) -> Any:
    from nibabel.gifti import GiftiCoordSystem, GiftiDataArray

    meta: dict[str, str] = {}
    if surface in _NON_ANATOMICAL:
        meta["GeometricType"] = _NON_ANATOMICAL[surface]
        # Not in any anatomical space: labelling a balloon "scanner" would
        # invite overlaying it on the T1.
        dataspace = 0
    elif surface in _RECOGNISED_SURFACES:
        meta["GeometricType"] = "Anatomical"
    if surface in _SECONDARY:
        meta["AnatomicalStructureSecondary"] = _SECONDARY[surface]

    image = _image(nib, primary)
    image.add_gifti_data_array(GiftiDataArray(
        np.ascontiguousarray(coords, dtype=np.float32),
        intent="NIFTI_INTENT_POINTSET",
        datatype="NIFTI_TYPE_FLOAT32",
        meta=meta,
        # The positions are already in the space, so the transform to it
        # is the identity.
        coordsys=GiftiCoordSystem(dataspace, dataspace, np.eye(4)),
    ))
    image.add_gifti_data_array(GiftiDataArray(
        np.ascontiguousarray(faces, dtype=np.int32),
        intent="NIFTI_INTENT_TRIANGLE",
        datatype="NIFTI_TYPE_INT32",
    ))
    return image


def _gifti_values(name: str, values: npt.NDArray) -> tuple[npt.NDArray, str]:
    """``values`` in a GIFTI data type: float32, or int32 for integers."""
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.floating):
        return values.astype(np.float32), "NIFTI_TYPE_FLOAT32"
    if np.issubdtype(values.dtype, np.integer) or values.dtype == bool:
        if values.size:
            info = np.iinfo(np.int32)
            low, high = int(values.min()), int(values.max())
            if low < info.min or high > info.max:
                raise ExportError(
                    f"attribute {name!r} holds integers {low}..{high}, "
                    f"outside the int32 range GIFTI can store"
                )
        return values.astype(np.int32), "NIFTI_TYPE_INT32"
    raise ExportError(
        f"attribute {name!r} has dtype {values.dtype}, which GIFTI cannot "
        f"store; it holds float32 and int32 data"
    )


def _metric_image(
    nib: Any,
    primary: str,
    name: str,
    values: npt.NDArray,
    intent: str,
    series: dict[str, Any] | None,
) -> Any:
    from nibabel.gifti import GiftiDataArray

    data, datatype = _gifti_values(name, values)
    image = _image(nib, primary)
    if series and str(series.get("unit", "")).upper() == "SECOND":
        image.meta["TimeStep"] = repr(float(series["step"]))
    columns = [data] if data.ndim == 1 else [data[:, c] for c in range(data.shape[1])]
    for column in columns:
        # A single map is named so Workbench shows it by name.  Columns of
        # one map stay unnamed: the GIFTI ingest reads distinctly named
        # arrays as separate maps, and these are one.
        meta = {"Name": name} if len(columns) == 1 else {}
        image.add_gifti_data_array(GiftiDataArray(
            np.ascontiguousarray(column), intent=intent, datatype=datatype,
            meta=meta,
        ))
    return image


def _label_image(
    nib: Any,
    primary: str,
    name: str,
    codes: npt.NDArray,
    table: dict[str, dict[str, Any]],
) -> Any:
    from nibabel.gifti import GiftiDataArray, GiftiLabel, GiftiLabelTable

    data, _ = _gifti_values(name, codes)
    if data.ndim != 1:
        raise ExportError(
            f"parcellation {name!r} has {data.shape[1]} columns; a GIFTI "
            f"label file holds one code per vertex"
        )
    labels = GiftiLabelTable()
    # The whole table for every hemisphere, as HCP writes it, so a code
    # means the same parcel in both files.
    for code in sorted(table, key=int):
        entry = table[code]
        red, green, blue, alpha = (float(c) for c in entry.get("rgba", (0, 0, 0, 0)))
        label = GiftiLabel(key=int(code), red=red, green=green, blue=blue, alpha=alpha)
        label.label = str(entry.get("name", code))
        labels.labels.append(label)
    image = _image(nib, primary)
    image.labeltable = labels
    image.add_gifti_data_array(GiftiDataArray(
        data, intent="NIFTI_INTENT_LABEL", datatype="NIFTI_TYPE_INT32",
        meta={"Name": name},
    ))
    return image


def _unmeasured(values: npt.NDArray, is_label: bool) -> bool:
    """Whether a hemisphere's column is only the fill the store wrote.

    The store completes a map present for one hemisphere with ``NaN`` (or
    ``-1`` for codes) on the other.  That map was never measured there, so
    no file is written for it -- which is also what makes re-ingesting the
    directory fill it the same way again.
    """
    values = np.asarray(values)
    if values.size == 0:
        return False
    if is_label:
        return bool(np.all(values == -1))
    if np.issubdtype(values.dtype, np.floating):
        return bool(np.all(np.isnan(values)))
    return False


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_gifti(
    store_path: str | Path,
    output_dir: str | Path,
    *,
    level: int = 0,
    hemispheres: Sequence[str] | str | None = None,
    surfaces: Sequence[str] | str | None = None,
    attribute_names: Sequence[str] | str | None = None,
    prefix: str | None = None,
) -> dict[str, Any]:
    """Write a cortical surface store as a directory of GIFTI files.

    Args:
        store_path: A store written by the GIFTI or FreeSurfer ingest.
        output_dir: Directory to write into; created if missing.  Files of
            the same name are replaced, and nothing else in it is touched.
        level: Resolution level.  ``0`` is the mesh as ingested; a coarser
            level is a decimated mesh with its own vertex numbering.
        hemispheres: ``left`` / ``right`` (or ``lh`` / ``rh``).  Default:
            every hemisphere in the store.
        surfaces: Surfaces to write, by name (``"pial"``) or attribute
            (``"coords_pial"``).  Default: the geometry and every alternate.
        attribute_names: Per-vertex maps and parcellations to write.
            Default: every vertex attribute at the level except the join
            key and the surface coordinates.
        prefix: Leading file-name part, e.g. ``"sub-01"`` gives
            ``sub-01_hemi-L_pial.surf.gii``.

    Returns:
        Summary dict with ``files`` (paths written, in order),
        ``file_count``, ``hemispheres``, ``geometry``, ``surfaces``,
        ``scalars``, ``labels``, ``level``, ``space``, ``vertex_count`` and
        ``face_count`` (summed over hemispheres), ``skipped``
        (``{hemisphere: [names]}`` of maps and surfaces that are only fill
        on that hemisphere, so were not written) and ``warnings``.

    Raises:
        ExportError: If the store is not a surface store, the level,
            a hemisphere, surface or attribute asked for is not in it, an
            attribute cannot be stored as GIFTI, or ``output_dir`` is a file.
    """
    from zarr_vectors_tools.algorithms.surfaces import read_hemisphere

    nib = _nibabel()
    store = Path(store_path)
    header = _surface_header(store)
    if prefix is not None and (not prefix or re.search(r"[/\\]", prefix)):
        raise ExportError(
            f"prefix must be a non-empty file-name part without path "
            f"separators, got {prefix!r}"
        )
    sides = _resolve_hemispheres(header, hemispheres)
    surface_columns = _resolve_surfaces(header, surfaces)
    maps = _resolve_maps(header, _vertex_attribute_names(store, level),
                         attribute_names, level)
    out = _output_directory(output_dir)

    dataspace = _dataspace_code(nib, header.space)
    series_info: dict[str, Any] = dict(header.extra.get("series") or {})
    entries = {h["hemisphere"]: h for h in header.hemispheres}
    coordinate_attributes = [attr for _, attr in surface_columns if attr is not None]

    files: list[str] = []
    warnings: list[str] = []
    skipped: dict[str, list[str]] = {}
    vertex_count = face_count = 0

    def save(image: Any, filename: str) -> None:
        path = out / filename
        try:
            nib.save(image, str(path))
        except Exception as exc:
            raise ExportError(f"failed to write GIFTI '{path}': {exc}") from exc
        files.append(str(path))

    for side in sides:
        primary = _primary_structure(entries[side])
        n_source = int(entries[side]["n_vertices"])
        try:
            result = read_hemisphere(
                store, side, level=level,
                attributes=[*coordinate_attributes, *maps],
            )
        except ValueError as exc:
            raise ExportError(str(exc)) from exc
        vertices = result["vertices"]
        faces = result["faces"]
        if level == 0 and (
            len(vertices) != n_source
            or not np.array_equal(result["source_vertex"], np.arange(len(vertices)))
        ):
            # Every GIFTI consumer indexes vertices by position, so a store
            # that cannot put each vertex back at its source number would
            # write files that look right and are wrong everywhere.
            raise ExportError(
                f"level 0 of the {side} hemisphere does not hold source "
                f"vertices 0..{n_source - 1} exactly once (it has "
                f"{len(vertices)} vertices); the store's join key is damaged"
            )
        vertex_count += len(vertices)
        face_count += len(faces)

        for surface, attr in surface_columns:
            coords = vertices if attr is None else result["attributes"][attr]
            if attr is not None and _unmeasured(coords, is_label=False):
                skipped.setdefault(side, []).append(surface)
                continue
            filename = gifti_filename(side, surface, "surf", prefix)
            if surface not in _RECOGNISED_SURFACES:
                warnings.append(
                    f"{filename}: ingest_gifti does not recognise the surface "
                    f"name {surface!r} and will read it back under another name"
                )
            save(_surface_image(nib, primary, surface, coords, faces, dataspace), filename)

        for name in maps:
            values = result["attributes"][name]
            is_label = name in header.label_tables
            if _unmeasured(values, is_label):
                skipped.setdefault(side, []).append(name)
                continue
            if is_label:
                kind = "label"
                image = _label_image(nib, primary, name, values, header.label_tables[name])
            else:
                kind, intent = metric_kind(
                    name, header.scalars.get(name), series=name in series_info,
                )
                image = _metric_image(
                    nib, primary, name, values, intent, series_info.get(name),
                )
            filename = gifti_filename(side, name, kind, prefix)
            read_back = map_name(filename)
            if read_back != name:
                warnings.append(
                    f"{filename}: ingest_gifti will read this map back as "
                    f"{read_back!r}, not {name!r}"
                )
            save(image, filename)

    return {
        "files": files,
        "file_count": len(files),
        "hemispheres": sides,
        "geometry": header.geometry,
        "surfaces": [surface for surface, _ in surface_columns],
        "scalars": [name for name in maps if name not in header.label_tables],
        "labels": [name for name in maps if name in header.label_tables],
        "level": level,
        "space": header.space,
        "vertex_count": vertex_count,
        "face_count": face_count,
        "skipped": skipped,
        "warnings": warnings,
    }
