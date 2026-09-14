"""Synthetic cortical surface files for the surface ingest tests.

Everything is written with nibabel's own writers, so the tests exercise the
real file formats offline: a subdivided, gently curved sheet stands in for
each hemisphere, and the maps on it are simple functions of the vertex number
so any value that lands on the wrong vertex is detectable.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import numpy.typing as npt

#: Where each synthetic hemisphere sits along x, so the two do not overlap.
SHIFT = {"left": -50.0, "right": 10.0}
PRIMARY = {"left": "CortexLeft", "right": "CortexRight"}
CRAS = np.array([1.5, -17.25, 20.0])

#: A tiny parcellation: code -> (name, rgba in [0, 1]).  No black entry, so a
#: FreeSurfer annotation round trip does not confuse a label with "unset".
PARCELS = {
    1: ("precentral", (0.9, 0.1, 0.1, 1.0)),
    2: ("postcentral", (0.1, 0.1, 0.9, 1.0)),
    3: ("insula", (0.2, 0.8, 0.2, 1.0)),
}


def sheet(hemisphere: str, nx: int = 12, ny: int = 10):
    """``(vertices, faces)`` for one synthetic hemisphere."""
    xs, ys = np.meshgrid(
        np.linspace(0.0, 40.0, nx), np.linspace(0.0, 30.0, ny), indexing="ij",
    )
    vertices = np.column_stack([
        xs.ravel() + SHIFT[hemisphere],
        ys.ravel(),
        5.0 * np.sin(xs.ravel() / 7.0),
    ]).astype(np.float32)
    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a = i * ny + j
            b, c = a + 1, a + ny
            faces += [[a, c, b], [b, c, c + 1]]
    return vertices, np.asarray(faces, dtype=np.int32)


def thickness(n: int) -> npt.NDArray[np.float32]:
    return np.linspace(1.0, 4.0, n).astype(np.float32)


def parcels(n: int) -> npt.NDArray[np.int32]:
    return (np.arange(n) % 3 + 1).astype(np.int32)


# ---------------------------------------------------------------------------
# GIFTI
# ---------------------------------------------------------------------------

def write_surf_gii(
    path: Path,
    vertices,
    faces,
    *,
    primary: str | None = None,
    geometric: str = "Anatomical",
    secondary: str | None = None,
    dataspace: int = 3,
) -> Path:
    import nibabel as nib
    from nibabel.gifti import GiftiDataArray, GiftiImage

    image = GiftiImage()
    if primary:
        image.meta["AnatomicalStructurePrimary"] = primary
    meta = {"GeometricType": geometric}
    if secondary:
        meta["AnatomicalStructureSecondary"] = secondary
    points = GiftiDataArray(
        np.asarray(vertices, dtype=np.float32), intent="NIFTI_INTENT_POINTSET",
        datatype="NIFTI_TYPE_FLOAT32", meta=meta,
    )
    points.coordsys.dataspace = dataspace
    image.add_gifti_data_array(points)
    image.add_gifti_data_array(GiftiDataArray(
        np.asarray(faces, dtype=np.int32), intent="NIFTI_INTENT_TRIANGLE",
        datatype="NIFTI_TYPE_INT32",
    ))
    nib.save(image, str(path))
    return path


def write_shape_gii(
    path: Path,
    arrays: list[npt.ArrayLike],
    *,
    primary: str | None = None,
    names: list[str] | None = None,
    intent: str = "NIFTI_INTENT_SHAPE",
) -> Path:
    import nibabel as nib
    from nibabel.gifti import GiftiDataArray, GiftiImage

    image = GiftiImage()
    if primary:
        image.meta["AnatomicalStructurePrimary"] = primary
    for index, values in enumerate(arrays):
        meta = {"Name": names[index]} if names else {}
        image.add_gifti_data_array(GiftiDataArray(
            np.asarray(values, dtype=np.float32), intent=intent,
            datatype="NIFTI_TYPE_FLOAT32", meta=meta,
        ))
    nib.save(image, str(path))
    return path


def write_label_gii(path: Path, codes, *, primary: str | None = None) -> Path:
    import nibabel as nib
    from nibabel.gifti import (
        GiftiDataArray,
        GiftiImage,
        GiftiLabel,
        GiftiLabelTable,
    )

    image = GiftiImage()
    if primary:
        image.meta["AnatomicalStructurePrimary"] = primary
    table = GiftiLabelTable()
    for code, (name, (r, g, b, a)) in {0: ("???", (0, 0, 0, 0)), **PARCELS}.items():
        label = GiftiLabel(key=code, red=r, green=g, blue=b, alpha=a)
        label.label = name
        table.labels.append(label)
    image.labeltable = table
    image.add_gifti_data_array(GiftiDataArray(
        np.asarray(codes, dtype=np.int32), intent="NIFTI_INTENT_LABEL",
        datatype="NIFTI_TYPE_INT32",
    ))
    nib.save(image, str(path))
    return path


def write_gifti_subject(root: Path, hemispheres=("left", "right")) -> Path:
    """A BIDS-style fMRIPrep derivative directory for one subject."""
    root.mkdir(parents=True, exist_ok=True)
    for hemisphere in hemispheres:
        tag = "L" if hemisphere == "left" else "R"
        primary = PRIMARY[hemisphere]
        vertices, faces = sheet(hemisphere)
        stem = f"sub-01_hemi-{tag}"
        write_surf_gii(root / f"{stem}_midthickness.surf.gii", vertices, faces,
                       primary=primary)
        write_surf_gii(root / f"{stem}_pial.surf.gii", vertices + [0, 0, 1],
                       faces, primary=primary)
        # HCP tags an inflated surface with the structure it was inflated
        # FROM; the reader must still see it as inflated.
        write_surf_gii(root / f"{stem}_inflated.surf.gii", vertices * 1.5,
                       faces, primary=primary, geometric="Inflated",
                       secondary="MidThickness")
        write_shape_gii(root / f"{stem}_thickness.shape.gii",
                        [thickness(len(vertices))])
        write_label_gii(root / f"{stem}_desc-aparc.label.gii",
                        parcels(len(vertices)), primary=primary)
    return root


# ---------------------------------------------------------------------------
# FreeSurfer
# ---------------------------------------------------------------------------

def _volume_info(cras) -> OrderedDict:
    return OrderedDict([
        ("head", np.array([2, 0, 20], dtype=np.int32)),
        ("valid", "1  # volume info valid"),
        ("filename", "../mri/filled-pretess255.mgz"),
        ("volume", np.array([256, 256, 256])),
        ("voxelsize", np.array([1.0, 1.0, 1.0])),
        ("xras", np.array([-1.0, 0.0, 0.0])),
        ("yras", np.array([0.0, 0.0, -1.0])),
        ("zras", np.array([0.0, 1.0, 0.0])),
        ("cras", np.asarray(cras, dtype=float)),
    ])


def write_freesurfer_subject(
    root: Path,
    *,
    hemispheres=("left", "right"),
    with_cras: bool = True,
    morphometry=("thickness", "curv"),
) -> Path:
    """A minimal ``recon-all`` subject: surf/ and label/ for each hemisphere."""
    import nibabel.freesurfer as fs

    surf = root / "surf"
    label = root / "label"
    surf.mkdir(parents=True, exist_ok=True)
    label.mkdir(parents=True, exist_ok=True)
    info = _volume_info(CRAS) if with_cras else None
    for hemisphere in hemispheres:
        prefix = "lh" if hemisphere == "left" else "rh"
        vertices, faces = sheet(hemisphere)
        white = vertices.astype(np.float64)
        pial = white + [0.0, 0.0, 2.0]
        for name, coords in (
            ("white", white), ("pial", pial), ("inflated", white * 1.5),
            ("sphere", white / np.linalg.norm(white, axis=1, keepdims=True)),
        ):
            kwargs = {"volume_info": info} if info is not None else {}
            fs.write_geometry(str(surf / f"{prefix}.{name}"), coords, faces, **kwargs)
        n = len(vertices)
        maps = {"thickness": thickness(n), "curv": np.cos(np.arange(n) / 5.0)}
        for name in morphometry:
            fs.write_morph_data(str(surf / f"{prefix}.{name}"),
                                np.asarray(maps[name], dtype=np.float32))
        codes = parcels(n) - 1  # rows of the colour table
        ctab = np.array([
            [int(r * 255), int(g * 255), int(b * 255), int((1 - a) * 255), 0]
            for _, (r, g, b, a) in PARCELS.values()
        ], dtype=np.int32)
        names = [name for name, _ in PARCELS.values()]
        fs.write_annot(str(label / f"{prefix}.aparc.annot"), codes, ctab, names)
    return root


# ---------------------------------------------------------------------------
# CIFTI
# ---------------------------------------------------------------------------

def brain_models(n_vertices: dict[str, int], present: dict[str, npt.ArrayLike],
                 *, voxels: int = 0):
    """A brain-model axis over the given cortex vertices (+ optional voxels)."""
    from nibabel import cifti2

    axis = None
    for hemisphere, vertices in present.items():
        part = cifti2.BrainModelAxis.from_surface(
            np.asarray(vertices), n_vertices[hemisphere], PRIMARY[hemisphere],
        )
        axis = part if axis is None else axis + part
    if voxels:
        mask = np.zeros((4, 4, 4), dtype=bool)
        mask.reshape(-1)[:voxels] = True
        part = cifti2.BrainModelAxis.from_mask(mask, name="thalamus_left",
                                               affine=np.eye(4))
        axis = axis + part
    return axis


def write_cifti(path: Path, map_axis, brain_axis, data) -> Path:
    import nibabel as nib
    from nibabel import cifti2

    header = cifti2.Cifti2Header.from_axes((map_axis, brain_axis))
    nib.save(cifti2.Cifti2Image(np.asarray(data, dtype=np.float32), header),
             str(path))
    return path


# ---------------------------------------------------------------------------
# Reading a surface store back, keyed by source vertex
# ---------------------------------------------------------------------------

def read_surface_columns(
    store: Path, columns: dict[str, int], *, level: int = 0,
) -> tuple[npt.NDArray, dict[str, npt.NDArray]]:
    """Positions and attribute columns of every vertex, sorted by join key.

    ``columns`` maps attribute name to its column count.  Sorting by
    ``zv_join_key`` puts rows in (hemisphere, source vertex) order, so a
    test can compare against the arrays it wrote regardless of how the
    store chunked them.
    """
    from zarr_vectors.building import (
        get_resolution_level,
        list_chunk_keys,
        open_store,
        read_chunk_attributes,
        read_chunk_vertices,
    )

    group = get_resolution_level(open_store(str(store)), level)
    wanted = {"zv_join_key": 1, **columns}
    positions: list[npt.NDArray] = []
    values: dict[str, list[npt.NDArray]] = {name: [] for name in wanted}
    for chunk in list_chunk_keys(group, "vertices"):
        for block in read_chunk_vertices(group, chunk, ndim=3):
            positions.append(np.asarray(block))
        for name, ncols in wanted.items():
            for block in read_chunk_attributes(
                group, name, chunk, ncols=ncols if ncols > 1 else None,
            ):
                block = np.asarray(block)
                values[name].append(
                    block.reshape(len(block), ncols) if ncols > 1 else block.ravel()
                )
    joined = {name: np.concatenate(parts) for name, parts in values.items()}
    order = np.argsort(joined["zv_join_key"].astype(np.int64), kind="stable")
    return (
        np.concatenate(positions)[order],
        {name: column[order] for name, column in joined.items()},
    )
