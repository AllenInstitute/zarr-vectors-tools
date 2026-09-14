"""Attach CIFTI-2 surface data to an existing cortical surface store.

CIFTI is how HCP-style pipelines ship results: one ``.dscalar.nii`` holds
myelin, thickness and curvature for both hemispheres; a ``.dlabel.nii`` holds
a parcellation such as the HCP multi-modal atlas; a ``.dtseries.nii`` holds
a resting-state run.  None of them carry geometry -- they index the vertices
of a surface the analyst already has, which is exactly what a surface store
is for.  So CIFTI is an **attach**, not an ingest: build the store from the
GIFTI or FreeSurfer surfaces first, then add CIFTI maps onto it.

How values find their vertex
----------------------------
A CIFTI brain-model axis lists, per structure, the SOURCE vertex numbers it
has data for.  Cortex rarely covers every vertex: the medial wall is left out,
so a 32k fs_LR hemisphere has 32,492 vertices but about 29,700 values.  Each
value's key is ``surface_keys(hemisphere, vertex)``, the same key the surface
store stamped on every vertex when it was written, and the streaming join in
:func:`~zarr_vectors_tools.ingest.attach.attach_attributes` does the rest.
Vertices with no value -- the medial wall -- get ``NaN`` (scalars) or ``-1``
(labels).

Subcortical voxels in the same file have no vertex to land on and are skipped;
the summary reports how many.

Requires ``nibabel``::

    pip install 'zarr-vectors-tools[surfaces]'
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from zarr_vectors.exceptions import IngestError

from zarr_vectors_tools.ingest._surface_store import (
    CIFTI_STRUCTURES,
    map_name,
    surface_keys,
)
from zarr_vectors_tools.ingest._tabular import sanitise_name

__all__ = ["CIFTI_SUFFIXES", "attach_cifti", "is_cifti_path"]

#: Dense CIFTI kinds that map values onto surface vertices.
CIFTI_SUFFIXES: tuple[str, ...] = (".dscalar.nii", ".dlabel.nii", ".dtseries.nii")

_STRUCTURE_HEMISPHERE = {v: k for k, v in CIFTI_STRUCTURES.items()}


def is_cifti_path(path: str | Path) -> bool:
    """Whether ``path`` names a dense CIFTI file this module can attach."""
    return str(path).lower().endswith(CIFTI_SUFFIXES)


def _load(path: Path):
    try:
        import nibabel as nib
        from nibabel import cifti2
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise IngestError(
            "reading CIFTI needs nibabel. Install with: "
            "pip install 'zarr-vectors-tools[surfaces]'"
        ) from exc
    try:
        image = nib.load(str(path))
    except Exception as exc:
        raise IngestError(f"failed to read CIFTI '{path}': {exc}") from exc
    if not isinstance(image, cifti2.Cifti2Image):
        raise IngestError(f"'{path.name}' is not a CIFTI-2 file")
    return image, cifti2


def _surface_header(store_path: str | Path):
    from zarr_vectors_tools.headers.formats import SurfaceHeader
    from zarr_vectors_tools.headers.registry import HeaderRegistry

    registry = HeaderRegistry(str(store_path))
    try:
        header = registry.get("surface")
    except KeyError:
        header = None
    if not isinstance(header, SurfaceHeader):
        raise IngestError(
            f"'{store_path}' is not a surface store (no surface header). "
            f"Build one from the surfaces first: zvtools convert with a "
            f"GIFTI directory or a FreeSurfer subject."
        )
    return registry, header


def _keys_and_columns(
    brain_models, header, cifti_name: str,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], dict[str, Any]]:
    """Join keys for the cortical brain models, and the columns holding them.

    Returns ``(keys, columns, report)``: ``columns`` are the brain-model
    positions whose values belong to ``keys``, in the same order.  Refuses a
    hemisphere whose mesh resolution differs from the store's: joining a 32k
    map onto a 164k surface would succeed for the first 32k vertex numbers
    and put every value on the wrong vertex.
    """
    store_vertices = {
        h["hemisphere"]: int(h["n_vertices"]) for h in header.hemispheres
    }
    keys: list[npt.NDArray[np.int64]] = []
    columns: list[npt.NDArray[np.int64]] = []
    report: dict[str, Any] = {
        "hemispheres": [], "skipped_structures": [], "voxels_skipped": 0,
    }
    for structure, where, sub_axis in brain_models.iter_structures():
        structure = str(structure)
        hemisphere = _STRUCTURE_HEMISPHERE.get(structure)
        start = where.start or 0
        stop = where.stop if where.stop is not None else len(brain_models)
        if hemisphere is None:
            if sub_axis.volume_mask.any():
                report["voxels_skipped"] += int(sub_axis.volume_mask.sum())
            report["skipped_structures"].append(structure)
            continue
        if hemisphere not in store_vertices:
            report["skipped_structures"].append(structure)
            continue
        declared = int(brain_models.nvertices[structure])
        if declared != store_vertices[hemisphere]:
            raise IngestError(
                f"{cifti_name}: {structure} is defined on a {declared}-vertex "
                f"mesh but the store's {hemisphere} hemisphere has "
                f"{store_vertices[hemisphere]} vertices. The file belongs to a "
                f"different mesh (32k vs 164k fs_LR, or fsaverage vs native); "
                f"resample it to the store's surface first."
            )
        vertex = np.asarray(sub_axis.vertex, dtype=np.int64)
        keys.append(surface_keys(hemisphere, vertex))
        columns.append(np.arange(start, stop, dtype=np.int64))
        report["hemispheres"].append({
            "hemisphere": hemisphere,
            "values": int(len(vertex)),
            "vertices": store_vertices[hemisphere],
        })
    if not keys:
        raise IngestError(
            f"{cifti_name} has no cortical surface data for this store's "
            f"hemispheres ({sorted(store_vertices)}); it holds "
            f"{report['skipped_structures'] or 'nothing'}"
        )
    return np.concatenate(keys), np.concatenate(columns), report


def attach_cifti(
    store_path: str | Path,
    cifti_path: str | Path,
    *,
    name: str | None = None,
    missing: str = "fill",
    overwrite: bool = False,
    shard_shape: int | tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Attach a dense CIFTI file's cortical maps to a surface store.

    Args:
        store_path: A store written by the GIFTI or FreeSurfer ingest.
        cifti_path: A ``.dscalar.nii``, ``.dlabel.nii`` or ``.dtseries.nii``.
        name: Attribute name.  For a time series or a single map it names
            the one attribute written (default: from the filename).  For a
            multi-map ``dscalar`` it keeps the maps together as one ``(V, M)``
            attribute under this name instead of one attribute per map.
        missing: ``"fill"`` (default) gives vertices without a value -- the
            medial wall, a hemisphere the file lacks -- ``NaN`` or ``-1``;
            ``"error"`` refuses the attach instead.
        overwrite: Replace attributes that already exist.
        shard_shape: Passed to
            :func:`~zarr_vectors_tools.ingest.attach.attach_attributes`.

    Returns:
        The attach summary plus ``attributes``, ``kind``, ``hemispheres``
        (values per hemisphere), ``skipped_structures`` and
        ``voxels_skipped``.

    Raises:
        IngestError: If the store is not a surface store, the file's mesh
            resolution differs from the store's, or it has no cortical data
            for the store's hemispheres.

    Note:
        Attributes are written to level 0.  Attach before building a
        pyramid, or rebuild it afterwards so coarse levels carry the maps.
    """
    from zarr_vectors_tools.ingest.attach import attach_attributes

    cifti_path = Path(cifti_path)
    if not cifti_path.exists():
        raise IngestError(f"CIFTI file not found: {cifti_path}")
    registry, header = _surface_header(store_path)
    image, cifti2 = _load(cifti_path)

    axes = [image.header.get_axis(i) for i in range(image.ndim)]
    brain_axis = next(
        (i for i, a in enumerate(axes) if isinstance(a, cifti2.BrainModelAxis)),
        None,
    )
    if brain_axis is None or image.ndim != 2:
        raise IngestError(
            f"{cifti_path.name} is not a dense CIFTI (dscalar, dlabel or "
            f"dtseries): parcellated and connectivity files have no per-vertex "
            f"axis"
        )
    map_axis = axes[1 - brain_axis]
    data = np.asarray(image.dataobj)
    if brain_axis == 0:
        data = data.T  # always (maps, brain models) from here on

    keys, columns, report = _keys_and_columns(
        axes[brain_axis], header, cifti_path.name,
    )
    values = data[:, columns].T  # (K, maps)

    base = name or map_name(cifti_path.name)
    used: set[str] = {header.key_attribute}
    attributes: dict[str, npt.NDArray] = {}
    label_tables: dict[str, dict[str, dict[str, Any]]] = {}
    series: dict[str, Any] = {}

    if isinstance(map_axis, cifti2.LabelAxis):
        kind = "label"
        map_names = [str(n) for n in map_axis.name]
        if name is not None and len(map_names) > 1:
            raise IngestError(
                f"{cifti_path.name} holds {len(map_names)} parcellations; "
                f"name= can only name one. Leave it unset to use the map names."
            )
        for index, map_label in enumerate(map_axis.label):
            if len(map_names) == 1:
                attr = sanitise_name(base, used)
            else:
                attr = sanitise_name(map_names[index] or f"{base}_{index}", used)
            attributes[attr] = np.rint(values[:, index]).astype(np.int32)
            table = {
                str(int(code)): {
                    "name": str(label_name),
                    "rgba": [float(c) for c in rgba],
                }
                for code, (label_name, rgba) in map_label.items()
            }
            table.setdefault(
                "-1", {"name": "no data (medial wall)", "rgba": [0.0, 0.0, 0.0, 0.0]},
            )
            label_tables[attr] = table
    elif isinstance(map_axis, cifti2.SeriesAxis):
        kind = "series"
        attr = sanitise_name(base, used)
        attributes[attr] = values.astype(np.float32)
        series[attr] = {
            "start": float(map_axis.start),
            "step": float(map_axis.step),
            "size": int(map_axis.size),
            "unit": str(map_axis.unit),
        }
    elif isinstance(map_axis, cifti2.ScalarAxis):
        kind = "scalar"
        map_names = [str(n).strip() for n in map_axis.name]
        distinct = all(map_names) and len(set(map_names)) == len(map_names)
        if values.shape[1] == 1:
            attributes[sanitise_name(base, used)] = values[:, 0].astype(np.float32)
        elif name is None and distinct:
            for index, map_label in enumerate(map_names):
                attributes[sanitise_name(map_label, used)] = (
                    values[:, index].astype(np.float32)
                )
        else:
            attributes[sanitise_name(base, used)] = values.astype(np.float32)
    else:
        raise IngestError(
            f"{cifti_path.name}: unsupported map axis "
            f"{type(map_axis).__name__}"
        )

    summary = dict(attach_attributes(
        str(store_path), attributes, keys=keys, level=0,
        key_attribute=header.key_attribute, missing=missing,
        overwrite=overwrite, shard_shape=shard_shape,
    ))

    for attr in attributes:
        if kind == "label":
            header.label_tables[attr] = label_tables[attr]
            header.scalars.pop(attr, None)
        else:
            header.scalars[attr] = cifti_path.name
            header.label_tables.pop(attr, None)
    if series:
        header.extra.setdefault("series", {}).update(series)
    registry.add("surface", header)

    summary.update({
        "attributes": sorted(attributes),
        "kind": kind,
        **report,
    })
    return summary
