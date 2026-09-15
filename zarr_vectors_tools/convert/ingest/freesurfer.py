"""Ingest a FreeSurfer subject's cortical surfaces into a surface store.

A ``recon-all`` subject directory holds each hemisphere as separate,
extensionless files that share one vertex numbering:

.. code-block:: text

    subject/
      surf/   lh.white  lh.pial  lh.inflated  lh.sphere     surfaces
              lh.thickness  lh.curv  lh.sulc  lh.area        morphometry
      label/  lh.aparc.annot  lh.aparc.a2009s.annot          parcellations
      ... and rh.* for the right hemisphere

:func:`ingest_freesurfer` reads the ones asked for and writes one surface
store (layout in :mod:`~zarr_vectors_tools.convert.ingest._surface_store`).

Coordinates
-----------
FreeSurfer writes surfaces in *surface RAS* ("tkregister" space), which is
centred on the conformed volume rather than on the scanner.  Everything else
in a neuroimaging pipeline -- the T1, the DWI, tractography in TRK or TRX,
fMRIPrep's GIFTI surfaces -- is in *scanner RAS*.  The two differ by one
translation, ``c_ras``, recorded in each surface file's footer.

``space="auto"`` (the default) adds ``c_ras`` to the anatomical surfaces when
it is known, so the store lines up with tracts and volumes of the same
subject.  It is taken from the surface footer when the footer is marked
valid, and otherwise from the subject's ``mri/orig.mgz`` (or ``T1.mgz``),
which is where FreeSurfer's own tools read it.  An invalid footer's ``cras``
is never used: ``fsaverage`` ships footers marked ``valid = 0`` whose
``cras`` is a 2 mm artefact, while its volumes record an offset of zero.

Inflated and spherical surfaces are left alone: they are not in any
anatomical space, and shifting them means nothing.  ``c_ras`` is
recorded in the header either way, so the shift can always be undone.

Midthickness
------------
``recon-all`` does not write a midthickness surface, but it is the most
useful geometry: it lies halfway through cortex, so it is the least biased
surface for projecting volume data and the one HCP-style analyses use.  When
no ``?h.midthickness`` file exists it is computed as the vertex-wise mean of
white and pial, which is how Connectome Workbench builds it too.

Requires ``nibabel``::

    pip install 'zarr-vectors-tools[surfaces]'
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from zarr_vectors.exceptions import IngestError
from zarr_vectors.typing import BinShape, ChunkShape

from zarr_vectors_tools.convert.ingest._surface_store import (
    HemisphereSurface,
    write_surface_store,
)
from zarr_vectors_tools.convert.ingest._tabular import sanitise_name

__all__ = [
    "DEFAULT_ALTERNATES",
    "DEFAULT_ANNOTATIONS",
    "DEFAULT_MORPHOMETRY",
    "find_freesurfer_surf_dir",
    "ingest_freesurfer",
]

#: Other surfaces carried as ``coords_<name>`` attributes when present.
DEFAULT_ALTERNATES: tuple[str, ...] = ("white", "pial", "inflated", "sphere")
#: Per-vertex morphometry maps read from ``surf/`` when present.
DEFAULT_MORPHOMETRY: tuple[str, ...] = ("thickness", "curv", "sulc", "area")
#: Parcellations read from ``label/`` when present.
DEFAULT_ANNOTATIONS: tuple[str, ...] = ("aparc",)

_HEMIS = {"lh": "left", "rh": "right", "left": "left", "right": "right"}
_PREFIX = {"left": "lh", "right": "rh"}

#: Surfaces in anatomical space, the only ones ``c_ras`` applies to.
_ANATOMICAL = {"white", "pial", "midthickness", "smoothwm", "orig", "graymid"}

#: Tolerance (mm) for the two hemispheres' ``c_ras`` agreeing.  They come
#: from one conformed volume, so any real difference means two subjects.
_CRAS_TOLERANCE = 1e-3


def _nib():
    try:
        import nibabel.freesurfer as fs
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise IngestError(
            "reading FreeSurfer files needs nibabel. Install with: "
            "pip install 'zarr-vectors-tools[surfaces]'"
        ) from exc
    return fs


def find_freesurfer_surf_dir(path: str | Path) -> Path | None:
    """The ``surf/`` directory for a subject directory or a ``surf/`` path.

    Returns ``None`` when ``path`` does not look like FreeSurfer output, so
    callers (the CLI's format detection) can ask without catching errors.
    """
    path = Path(path)
    if not path.is_dir():
        return None
    for candidate in (path / "surf", path):
        if any((candidate / f"{h}.white").exists() for h in ("lh", "rh")):
            return candidate
    return None


def _read_geometry(path: Path) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray | None]:
    fs = _nib()
    try:
        with warnings.catch_warnings():
            # A surface written without a volume footer makes nibabel warn
            # "Unknown extension code"; the missing c_ras is handled below.
            warnings.simplefilter("ignore")
            coords, faces, info = fs.read_geometry(str(path), read_metadata=True)
    except Exception as exc:
        raise IngestError(f"failed to read FreeSurfer surface '{path}': {exc}") from exc
    cras = info.get("cras") if info else None
    # nibabel returns the footer whether or not FreeSurfer marked it valid.
    # An invalid footer's cras is not an offset, it is whatever was there.
    if cras is not None and not str(info.get("valid", "")).strip().startswith("1"):
        cras = None
    return (
        np.asarray(coords, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
        None if cras is None else np.asarray(cras, dtype=np.float64),
    )


def _volume_cras(surf_dir: Path) -> tuple[npt.NDArray | None, str | None]:
    """``c_ras`` from the subject's conformed volume, when there is one.

    Returns the offset and the file it came from.  ``orig.mgz`` first: every
    surface is built from it, so its offset is the surfaces' by definition.
    """
    mri = surf_dir.parent / "mri"
    for name in ("orig.mgz", "T1.mgz", "brain.mgz"):
        path = mri / name
        if not path.exists():
            continue
        try:
            import nibabel as nib

            header = nib.load(str(path)).header
            return np.asarray(header["Pxyz_c"], dtype=np.float64), f"mri/{name}"
        except Exception:  # noqa: BLE001 - an unreadable volume is just absent
            continue
    return None, None


def _read_annot(path: Path) -> tuple[npt.NDArray, dict[int, dict[str, Any]]]:
    """Codes and label table from a ``.annot``.

    nibabel returns each vertex's row in the colour table, with ``-1`` for a
    vertex whose stored colour is not in the table.  Those rows are the
    codes; the table maps each to its structure name and RGBA in ``[0, 1]``
    (the file stores transparency, ``alpha = 1 - T / 255``).
    """
    fs = _nib()
    try:
        codes, ctab, names = fs.read_annot(str(path))
    except Exception as exc:
        raise IngestError(f"failed to read annotation '{path}': {exc}") from exc
    table: dict[int, dict[str, Any]] = {}
    for row, name in enumerate(names):
        r, g, b, t = (float(c) for c in ctab[row, :4])
        table[row] = {
            "name": name.decode() if isinstance(name, bytes) else str(name),
            "rgba": [r / 255.0, g / 255.0, b / 255.0, 1.0 - t / 255.0],
        }
    if (np.asarray(codes) < 0).any():
        table.setdefault(-1, {"name": "unassigned", "rgba": [0.0, 0.0, 0.0, 0.0]})
    return np.asarray(codes, dtype=np.int32), table


def _resolve(
    requested: tuple[str, ...] | list[str] | None,
    default: tuple[str, ...],
) -> tuple[tuple[str, ...], bool]:
    """The names to read, and whether a missing one is an error.

    Defaults are opportunistic -- not every subject has ``?h.sulc`` -- but a
    name the caller asked for by hand must exist.
    """
    if requested is None:
        return default, False
    return tuple(requested), True


def ingest_freesurfer(
    input_path: str | Path,
    output_path: str | Path,
    chunk_shape: ChunkShape,
    *,
    bin_shape: BinShape | None = None,
    dtype: str = "float32",
    geometry: str = "midthickness",
    alternates: tuple[str, ...] | list[str] | None = None,
    morphometry: tuple[str, ...] | list[str] | None = None,
    annotations: tuple[str, ...] | list[str] | None = None,
    hemispheres: tuple[str, ...] | list[str] | None = None,
    space: str = "auto",
) -> dict[str, Any]:
    """Ingest a FreeSurfer subject into one cortical surface store.

    Args:
        input_path: The subject directory (holding ``surf/`` and ``label/``)
            or its ``surf/`` directory.
        output_path: Store to create.
        chunk_shape: Spatial chunk size in mm.
        bin_shape: Optional intra-chunk sub-binning.
        dtype: Position dtype.
        geometry: Surface to chunk: ``midthickness`` (default; computed from
            white and pial when there is no file), ``white``, ``pial``,
            ``smoothwm`` or ``orig``.
        alternates: Other surfaces to carry as ``coords_<name>``.  Default
            ``white, pial, inflated, sphere``, each only if present; named
            explicitly, each must exist.
        morphometry: Per-vertex maps from ``surf/``.  Default
            ``thickness, curv, sulc, area``, each only if present.
        annotations: Parcellations from ``label/`` (``aparc``,
            ``aparc.a2009s``, ``aparc.DKTatlas``...).  Default ``aparc`` if
            present.  Stored as int32 codes; the names and colours go in the
            header's label table.
        hemispheres: ``lh``/``rh`` (or ``left``/``right``).  Default: both,
            whichever exist.
        space: ``"auto"`` (scanner RAS when ``c_ras`` is known from a valid
            surface footer or ``mri/orig.mgz``, else surface RAS),
            ``"scanner"`` (require ``c_ras``) or ``"surface"`` (leave
            FreeSurfer's coordinates untouched).

    Returns:
        The surface-store summary plus ``subject``, ``c_ras`` and
        ``c_ras_source`` (``"surface footer"``, ``"mri/orig.mgz"``... or
        ``None``).

    Raises:
        IngestError: If the directory is not FreeSurfer output, a requested
            file is missing, ``space="scanner"`` is asked for without
            ``c_ras``, or the hemispheres record different ``c_ras``.
    """
    surf_dir = find_freesurfer_surf_dir(input_path)
    if surf_dir is None:
        raise IngestError(
            f"'{input_path}' is not a FreeSurfer subject: expected "
            f"surf/lh.white or surf/rh.white inside it"
        )
    label_dir = surf_dir.parent / "label"
    if space not in ("auto", "scanner", "surface"):
        raise IngestError(
            f"space must be 'auto', 'scanner' or 'surface', got {space!r}"
        )
    if geometry not in _ANATOMICAL:
        raise IngestError(
            f"geometry={geometry!r} is not an anatomical surface; chunking "
            f"{geometry} puts the spatial index on a shape that is not the "
            f"brain. Use one of {sorted(_ANATOMICAL)} and carry {geometry} "
            f"as an alternate."
        )

    alt_names, alt_required = _resolve(alternates, DEFAULT_ALTERNATES)
    morph_names, morph_required = _resolve(morphometry, DEFAULT_MORPHOMETRY)
    annot_names, annot_required = _resolve(annotations, DEFAULT_ANNOTATIONS)

    if hemispheres is None:
        wanted = [h for h in ("left", "right")
                  if (surf_dir / f"{_PREFIX[h]}.white").exists()
                  or (surf_dir / f"{_PREFIX[h]}.{geometry}").exists()]
    else:
        wanted = []
        for h in hemispheres:
            if h not in _HEMIS:
                raise IngestError(
                    f"hemisphere must be lh, rh, left or right, got {h!r}"
                )
            wanted.append(_HEMIS[h])

    def need(path: Path, what: str) -> Path:
        if not path.exists():
            raise IngestError(f"{what}: {path} does not exist")
        return path

    surfaces: list[HemisphereSurface] = []
    cras_seen: dict[str, npt.NDArray | None] = {}
    computed_midthickness = False
    for hemi in wanted:
        prefix = _PREFIX[hemi]

        # ---- geometry -----------------------------------------------------
        geo_file = surf_dir / f"{prefix}.{geometry}"
        if geo_file.exists():
            vertices, faces, cras = _read_geometry(geo_file)
        elif geometry == "midthickness":
            white, faces, cras = _read_geometry(
                need(surf_dir / f"{prefix}.white",
                     f"{hemi} midthickness is computed from white and pial"),
            )
            pial, _, _ = _read_geometry(
                need(surf_dir / f"{prefix}.pial",
                     f"{hemi} midthickness is computed from white and pial"),
            )
            if pial.shape != white.shape:
                raise IngestError(
                    f"{hemi} white has {len(white)} vertices but pial has "
                    f"{len(pial)}; they are not the same subject's surfaces"
                )
            vertices = 0.5 * (white + pial)
            computed_midthickness = True
        else:
            need(geo_file, f"{hemi} geometry")
            raise AssertionError("unreachable")  # pragma: no cover
        cras_seen[hemi] = cras

        surface = HemisphereSurface(
            hemisphere=hemi, vertices=vertices, faces=faces, geometry=geometry,
        )

        # ---- alternates ---------------------------------------------------
        for name in alt_names:
            if name == geometry:
                continue
            path = surf_dir / f"{prefix}.{name}"
            if not path.exists():
                if alt_required:
                    need(path, f"{hemi} alternate surface {name!r}")
                continue
            coords, alt_faces, _ = _read_geometry(path)
            if alt_faces.shape != faces.shape:
                raise IngestError(
                    f"{hemi} {name} has {len(alt_faces)} faces but "
                    f"{geometry} has {len(faces)}; surfaces of one hemisphere "
                    f"must share a topology"
                )
            surface.alternates[name] = coords

        # ---- morphometry --------------------------------------------------
        used: set[str] = {"zv_join_key"}
        for name in morph_names:
            path = surf_dir / f"{prefix}.{name}"
            if not path.exists():
                if morph_required:
                    need(path, f"{hemi} morphometry {name!r}")
                continue
            try:
                values = np.asarray(_nib().read_morph_data(str(path)))
            except Exception as exc:
                raise IngestError(
                    f"failed to read morphometry '{path}': {exc}"
                ) from exc
            attr = sanitise_name(name, used)
            surface.scalars[attr] = values.astype(np.float32)
            surface.sources[attr] = path.name

        # ---- annotations --------------------------------------------------
        for name in annot_names:
            candidates = [label_dir / f"{prefix}.{name}.annot",
                          surf_dir / f"{prefix}.{name}.annot"]
            path = next((c for c in candidates if c.exists()), None)
            if path is None:
                if annot_required:
                    need(candidates[0], f"{hemi} annotation {name!r}")
                continue
            codes, table = _read_annot(path)
            attr = sanitise_name(name, used)
            surface.labels[attr] = codes
            surface.label_tables[attr] = table
            surface.sources[attr] = path.name

        surfaces.append(surface)

    if not surfaces:
        raise IngestError(f"no hemisphere surfaces found in {surf_dir}")

    # ---- coordinate space ---------------------------------------------------
    known = {h: c for h, c in cras_seen.items() if c is not None}
    if len(known) == 2:
        left, right = known["left"], known["right"]
        if not np.allclose(left, right, atol=_CRAS_TOLERANCE):
            raise IngestError(
                f"the hemispheres record different c_ras ({left.tolist()} vs "
                f"{right.tolist()}): they come from different subjects or "
                f"different conformed volumes"
            )
    c_ras = next(iter(known.values()), None)
    complete = len(known) == len(surfaces)
    c_ras_source = "surface footer" if complete else None
    if not complete:
        c_ras, c_ras_source = _volume_cras(surf_dir)
        complete = c_ras is not None
    if space == "scanner" and not complete:
        raise IngestError(
            "space='scanner' needs c_ras, but the surface files carry no "
            "valid volume footer and there is no mri/orig.mgz beside surf/. "
            "Use space='surface' to keep FreeSurfer's own coordinates."
        )
    apply_cras = space == "scanner" or (space == "auto" and complete)
    if apply_cras:
        for surface in surfaces:
            surface.vertices = surface.vertices + c_ras
            for name in list(surface.alternates):
                if name in _ANATOMICAL:
                    surface.alternates[name] = surface.alternates[name] + c_ras
    resolved_space = "scanner" if apply_cras else "surface"

    subject = surf_dir.parent.name if surf_dir.name == "surf" else surf_dir.name
    summary = write_surface_store(
        output_path, surfaces, chunk_shape,
        source="freesurfer", space=resolved_space, bin_shape=bin_shape,
        dtype=dtype, c_ras=c_ras,
        extra_header={
            "subject": subject,
            "c_ras_applied": bool(apply_cras),
            "c_ras_source": c_ras_source,
            "midthickness_computed": computed_midthickness,
            "shifted_surfaces": sorted(
                {geometry} | (_ANATOMICAL & set().union(
                    *(s.alternates for s in surfaces)
                ))
            ) if apply_cras else [],
        },
    )
    summary["subject"] = subject
    summary["c_ras"] = None if c_ras is None else [float(v) for v in c_ras]
    summary["c_ras_source"] = c_ras_source
    return summary
