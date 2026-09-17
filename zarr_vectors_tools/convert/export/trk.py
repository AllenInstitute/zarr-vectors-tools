"""Export zarr vectors streamlines to TrackVis TRK format.

Requires ``nibabel``: ``pip install nibabel``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from zarr_vectors.exceptions import ExportError
from zarr_vectors.typing import ChunkCoords

from zarr_vectors_tools.convert.export._streamlines import (
    read_streamline_level,
    stored_space,
    trackvis_to_rasmm,
    trk_reference,
)


def export_trk(
    store_path: str | Path,
    output_path: str | Path,
    *,
    level: int = 0,
    object_ids: list[int] | None = None,
    group_ids: list[int] | None = None,
    chunks: list[ChunkCoords] | None = None,
    affine: np.ndarray | None = None,
    attribute_names: Sequence[str] | None = None,
    object_attribute_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Export zarr vectors streamlines to a TRK file.

    Writes back what the ingest kept: per-vertex attributes as TRK scalars,
    per-object attributes as TRK properties, and the reference image
    (``vox_to_ras``, voxel size, dimensions, voxel order) from the store's
    TRK header.  Positions are converted from the space the store holds
    them in, so a store ingested in TrackVis voxel millimetres and one
    registered to RAS both write a file that loads in the same place.

    Args:
        store_path: Path to the zarr vectors store.
        output_path: Path for the output .trk file.
        level: Resolution level to export.
        object_ids: Optional object ID filter.
        group_ids: Optional group ID filter.
        chunks: Optional whitelist of chunk coordinate tuples. Filters at
            the *segment* level: only vertex groups stored in listed
            chunks are emitted, and each surviving contiguous run is
            written as its own streamline. The output ``streamline_count``
            can therefore exceed the source object count.
        affine: 4×4 voxel-to-RAS affine for a store with no TRK header,
            with the positions taken as voxel coordinates.  When given it
            overrides the stored header; ``None`` (default) uses the header,
            or the identity when there is none.
        attribute_names: Per-vertex attributes to write as scalars.
            ``None`` (default) writes every numeric one; ``[]`` writes none.
        object_attribute_names: Per-object attributes to write as
            properties, with the same ``None`` / ``[]`` meaning.

    Returns:
        Summary dict with ``streamline_count``, ``vertex_count``,
        ``attributes_carried``, ``object_attributes_carried``, ``space``
        (of the stored positions) and ``attributes_skipped`` (non-numeric
        attributes left out, when there were any).

    Raises:
        ExportError: If nibabel is not installed, a requested attribute is
            missing or not numeric, TRK cannot hold what was asked (it takes
            ten scalar and ten property names of up to 20 characters), or
            the write fails.
    """
    try:
        import nibabel as nib
        from nibabel.streamlines import Field
        from nibabel.streamlines.trk import TrkFile
    except ImportError as e:
        raise ExportError(
            "nibabel is required for TRK export. "
            "Install with: pip install nibabel"
        ) from e

    data, skipped = read_streamline_level(
        store_path, level=level, object_ids=object_ids, group_ids=group_ids,
        chunks=chunks, attribute_names=attribute_names,
        object_attribute_names=object_attribute_names,
    )
    streamlines = data.streamlines
    space, trk, trx = stored_space(store_path)
    # A header written without an affine carries no reference to use.
    if trk is not None and trk.vox_to_ras is None:
        trk = None

    if affine is not None or (trk is None and trx is None):
        # No stored reference: the positions are voxel coordinates of
        # ``affine`` (the identity when none was passed).  Fill the header's
        # voxel size / dimensions / voxel->RAS explicitly -- left to
        # nibabel they stay (1,1,1)/identity and the affine is only baked
        # into the points, which loses the physical frame: freeview then
        # refuses to place the tract, and any metre->millimetre factor in
        # the affine shrinks a whole brain to a sub-millimetre speck.
        to_rasmm = np.asarray(
            np.eye(4) if affine is None else affine, dtype=np.float32,
        )
        all_pts = np.concatenate(streamlines, axis=0)
        dims = np.clip(
            np.ceil(np.abs(all_pts).max(axis=0)).astype(np.int64) + 1, 1, 32767,
        ).astype(np.int16)
        header = {
            Field.VOXEL_SIZES: tuple(
                float(v) for v in np.linalg.norm(to_rasmm[:3, :3], axis=0)
            ),
            Field.DIMENSIONS: tuple(int(v) for v in dims),
            Field.VOXEL_TO_RASMM: to_rasmm,
        }
    elif trk is not None:
        header = trk_reference(trk)
        # Positions still in TrackVis voxel millimetres are mapped to RAS so
        # nibabel can map them back through the same header -- the result
        # is the stored coordinates, bit for bit up to float32 rounding.
        to_rasmm = trackvis_to_rasmm(trk) if space == "voxmm" else np.eye(4)
    else:
        # A TRX-ingested store: RAS positions, and a reference image.
        from nibabel.orientations import aff2axcodes

        reference = trx.affine if trx.affine is not None else np.eye(4)
        header = {
            Field.VOXEL_TO_RASMM: np.asarray(reference, dtype=np.float32),
            Field.VOXEL_SIZES: tuple(
                float(v) for v in np.linalg.norm(reference[:3, :3], axis=0)
            ),
            Field.DIMENSIONS: tuple(int(v) for v in trx.dimensions),
            Field.VOXEL_ORDER: "".join(aff2axcodes(reference)).encode("latin1"),
        }
        to_rasmm = np.eye(4)

    data_per_point = {
        name: data.per_streamline(values.astype(np.float32))
        for name, values in data.vertex_attributes.items()
    }
    data_per_streamline = {
        name: values.astype(np.float32).reshape(len(values), -1)
        for name, values in data.object_attributes.items()
    }

    try:
        tractogram = nib.streamlines.Tractogram(
            streamlines=streamlines,
            data_per_point=data_per_point,
            data_per_streamline=data_per_streamline,
            affine_to_rasmm=np.asarray(to_rasmm, dtype=np.float32),
        )
    except Exception as e:
        raise ExportError(f"Failed to assemble TRK tractogram: {e}") from e

    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        TrkFile(tractogram=tractogram, header=header).save(str(output_path))
    except ValueError as e:
        # nibabel's TRK writer refuses more than ten names, or a name over 20
        # characters, per table.
        raise ExportError(
            f"TRK cannot hold these attributes ({e}); choose which to write "
            f"with attribute_names= / object_attribute_names= "
            f"(--attribute / --object-attribute)"
        ) from e
    except Exception as e:
        raise ExportError(f"Failed to write TRK '{output_path}': {e}") from e

    summary: dict[str, Any] = {
        "streamline_count": len(streamlines),
        "vertex_count": int(data.lengths.sum()),
        "attributes_carried": sorted(data_per_point),
        "object_attributes_carried": sorted(data_per_streamline),
        "space": space,
    }
    left_out = sorted(skipped["vertex"] + skipped["object"])
    if left_out:
        summary["attributes_skipped"] = left_out
    return summary
