"""Convert an MRI volume (and optional label volumes) to OME-Zarr 0.5.

Reads anything nibabel reads (``.mgz``, ``.nii``, ``.nii.gz``) and writes an
OME-Zarr 0.5 image -- Zarr v3, one multiscale pyramid, labels under
``labels/`` -- in the same world space as the source's affine.

Why the volume is reoriented
----------------------------
OME-Zarr places an array in space with a per-axis scale and translation only.
FreeSurfer's conformed volumes are stored LIA (left, inferior, anterior), so
their voxel-to-world affine permutes and flips axes, which a scale and a
translation cannot say.  The volume is therefore reoriented to RAS+ first
(``nibabel.as_closest_canonical``), which for any axis-aligned acquisition
makes the affine diagonal.  Axes are written ``z, y, x`` with ``x`` = right,
``y`` = anterior, ``z`` = superior, in millimetres of the SCANNER RAS space --
the space a FreeSurfer surface store is in after ``c_ras`` is applied, so the
two overlay without further transforms.  A volume whose affine is oblique is
refused rather than silently resampled.

Coordinates refer to voxel centres.  Each coarser level averages 2x2x2 blocks
(intensity images) or takes the most common value in each block (labels), and
its translation moves to the block centre so every level stays registered.

Usage::

    python scripts/volume_to_ome_zarr.py bert/mri/T1.mgz bert_T1.ome.zarr \\
        --label aparc_aseg=bert/mri/aparc+aseg.mgz
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import zarr
from zarr.codecs import ZstdCodec

OME_VERSION = "0.5"
AXES = [
    {"name": "z", "type": "space", "unit": "millimeter"},
    {"name": "y", "type": "space", "unit": "millimeter"},
    {"name": "x", "type": "space", "unit": "millimeter"},
]


def load_ras(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(data zyx, scale zyx, translation zyx)`` for a volume, in scanner RAS."""
    image = nib.as_closest_canonical(nib.load(str(path)))
    affine = image.affine
    linear = affine[:3, :3]
    if not np.allclose(linear, np.diag(np.diag(linear)), atol=1e-4):
        raise SystemExit(
            f"error: {path.name} is oblique (its affine rotates after "
            f"reorientation); OME-Zarr has no rotation, so it would have to be "
            f"resampled first"
        )
    data = np.asarray(image.dataobj)
    if data.ndim != 3:
        raise SystemExit(f"error: {path.name} has {data.ndim} dimensions, expected 3")
    # RAS voxel order (i=x, j=y, k=z) -> OME's z, y, x.
    zyx = np.ascontiguousarray(data.transpose(2, 1, 0))
    scale = np.diag(linear)[::-1].astype(float)
    translation = affine[:3, 3][::-1].astype(float)
    return zyx, scale, translation


def _pad_even(data: np.ndarray) -> np.ndarray:
    pad = [(0, s % 2) for s in data.shape]
    return np.pad(data, pad, mode="edge") if any(p[1] for p in pad) else data


def downsample_mean(data: np.ndarray) -> np.ndarray:
    d = _pad_even(data).astype(np.float32)
    z, y, x = (s // 2 for s in d.shape)
    out = d.reshape(z, 2, y, 2, x, 2).mean(axis=(1, 3, 5))
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        out = np.clip(np.rint(out), info.min, info.max)
    return out.astype(data.dtype)


def downsample_mode(data: np.ndarray) -> np.ndarray:
    """Most common label in each 2x2x2 block; ties go to the first voxel.

    Averaging label codes invents labels that exist nowhere in the source, and
    striding shifts the level by half a voxel against the image pyramid.
    """
    d = _pad_even(data)
    z, y, x = (s // 2 for s in d.shape)
    blocks = d.reshape(z, 2, y, 2, x, 2).transpose(0, 2, 4, 1, 3, 5).reshape(-1, 8)
    best = blocks[:, 0].copy()
    best_count = np.zeros(len(blocks), dtype=np.int8)
    for k in range(8):
        candidate = blocks[:, k]
        count = (blocks == candidate[:, None]).sum(axis=1, dtype=np.int8)
        better = count > best_count
        best[better] = candidate[better]
        best_count[better] = count[better]
    return best.reshape(z, y, x)


def write_multiscale(
    group: zarr.Group,
    data: np.ndarray,
    scale: np.ndarray,
    translation: np.ndarray,
    *,
    name: str,
    levels: int,
    chunk: int,
    labels: bool,
) -> list[dict]:
    datasets = []
    current = data
    for level in range(levels):
        factor = 2 ** level
        level_scale = scale * factor
        # Voxel i of this level covers source voxels [i*f, i*f + f); its
        # centre is (f - 1) / 2 source voxels past the first one.
        level_translation = translation + scale * (factor - 1) / 2.0
        chunks = tuple(min(chunk, s) for s in current.shape)
        shards = tuple(-(-s // c) * c for s, c in zip(current.shape, chunks))
        array = group.create_array(
            str(level), shape=current.shape, dtype=current.dtype,
            chunks=chunks, shards=shards, compressors=ZstdCodec(level=5),
            fill_value=0, dimension_names=["z", "y", "x"],
        )
        array[:] = current
        datasets.append({
            "path": str(level),
            "coordinateTransformations": [
                {"type": "scale", "scale": [float(v) for v in level_scale]},
                {"type": "translation",
                 "translation": [float(v) for v in level_translation]},
            ],
        })
        if level + 1 < levels:
            if min(current.shape) < 2:
                break
            current = downsample_mode(current) if labels else downsample_mean(current)
    return [{
        "name": name,
        "axes": AXES,
        "datasets": datasets,
        "type": "mode" if labels else "mean",
        "metadata": {"method": "2x2x2 block " + ("mode" if labels else "mean")},
    }]


def label_dtype(data: np.ndarray) -> np.dtype:
    """Viewers (Neuroglancer) want unsigned segmentation ids."""
    if data.min() < 0:
        raise SystemExit("error: label volume has negative values")
    top = int(data.max())
    for dtype in (np.uint8, np.uint16, np.uint32):
        if top <= np.iinfo(dtype).max:
            return np.dtype(dtype)
    return np.dtype(np.uint64)


def convert(
    image_path: Path,
    output: Path,
    label_paths: dict[str, Path],
    *,
    levels: int,
    chunk: int,
    overwrite: bool,
) -> dict:
    if output.exists():
        if not overwrite:
            raise SystemExit(f"error: {output} exists; pass --overwrite")
        if not (output / "zarr.json").exists():
            raise SystemExit(f"error: refusing to delete {output}: not a Zarr store")
        shutil.rmtree(output)

    data, scale, translation = load_ras(image_path)
    root = zarr.open_group(str(output), mode="w", zarr_format=3)
    name = image_path.name.split(".")[0]
    multiscales = write_multiscale(
        root, data, scale, translation,
        name=name, levels=levels, chunk=chunk, labels=False,
    )
    root.attrs["ome"] = {"version": OME_VERSION, "multiscales": multiscales}
    summary = {
        "image": str(image_path), "shape_zyx": list(data.shape),
        "dtype": str(data.dtype), "levels": len(multiscales[0]["datasets"]),
        "scale_zyx": scale.tolist(), "translation_zyx": translation.tolist(),
        "labels": {},
    }

    if label_paths:
        labels_group = root.create_group("labels")
        labels_group.attrs["ome"] = {
            "version": OME_VERSION, "labels": list(label_paths),
        }
        for label_name, path in label_paths.items():
            label_data, label_scale, label_translation = load_ras(path)
            if label_data.shape != data.shape or not (
                np.allclose(label_scale, scale) and np.allclose(label_translation, translation)
            ):
                raise SystemExit(
                    f"error: label {path.name} is not on the image's grid "
                    f"(shape {label_data.shape} vs {data.shape}); resample it first"
                )
            label_data = label_data.astype(label_dtype(label_data))
            group = labels_group.create_group(label_name)
            label_multiscales = write_multiscale(
                group, label_data, scale, translation,
                name=label_name, levels=levels, chunk=chunk, labels=True,
            )
            group.attrs["ome"] = {
                "version": OME_VERSION,
                "multiscales": label_multiscales,
                "image-label": {"source": {"image": "../../"}},
            }
            summary["labels"][label_name] = {
                "path": str(path), "dtype": str(label_data.dtype),
                "n_labels": int(len(np.unique(label_data))),
            }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("image", type=Path, help="volume to convert (.mgz, .nii, .nii.gz)")
    parser.add_argument("output", type=Path, help="OME-Zarr store to write (*.ome.zarr)")
    parser.add_argument(
        "--label", action="append", default=[], metavar="NAME=PATH",
        help="label volume on the same grid, written under labels/NAME (repeatable)",
    )
    parser.add_argument("--levels", type=int, default=4,
                        help="pyramid levels including full resolution (default: 4)")
    parser.add_argument("--chunk", type=int, default=64,
                        help="chunk edge in voxels; one shard per level (default: 64)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    labels: dict[str, Path] = {}
    for item in args.label:
        if "=" not in item:
            raise SystemExit(f"error: --label wants NAME=PATH, got {item!r}")
        label_name, path = item.split("=", 1)
        labels[label_name] = Path(path)

    summary = convert(
        args.image, args.output, labels,
        levels=args.levels, chunk=args.chunk, overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
