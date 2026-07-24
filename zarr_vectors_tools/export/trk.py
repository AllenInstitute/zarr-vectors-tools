"""Export zarr vectors streamlines to TrackVis TRK format.

Requires ``nibabel``: ``pip install nibabel``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from zarr_vectors.exceptions import ExportError
from zarr_vectors.types.polylines import read_polylines
from zarr_vectors.typing import BoundingBox, ChunkCoords


def export_trk(
    store_path: str | Path,
    output_path: str | Path,
    *,
    level: int = 0,
    object_ids: list[int] | None = None,
    group_ids: list[int] | None = None,
    chunks: list[ChunkCoords] | None = None,
    affine: np.ndarray | None = None,
) -> dict[str, Any]:
    """Export zarr vectors streamlines to a TRK file.

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
        affine: 4×4 voxel-to-RAS affine matrix. If None, uses identity.

    Returns:
        Summary dict with ``streamline_count``, ``vertex_count``.

    Raises:
        ExportError: If nibabel is not installed or export fails.
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

    result = read_polylines(
        str(store_path),
        level=level,
        object_ids=object_ids,
        group_ids=group_ids,
        chunks=chunks,
    )

    poly_list = result["polylines"]
    n_streamlines = len(poly_list)

    if n_streamlines == 0:
        raise ExportError("No streamlines to export")

    # Reconstruct full streamlines by concatenating segments
    streamlines: list[np.ndarray] = []
    for segments in poly_list:
        full = np.concatenate(segments, axis=0).astype(np.float32)
        streamlines.append(full)

    total_vertices = sum(len(s) for s in streamlines)

    if affine is None:
        affine = np.eye(4, dtype=np.float32)
    affine = np.asarray(affine, dtype=np.float32)

    # Fill the header's voxel size / dimensions / voxel->RAS explicitly. The
    # streamlines are in the store's own (voxel) coordinates and ``affine`` is the
    # voxel->RAS matrix in MILLIMETRES. Without an explicit header nibabel leaves
    # voxel_sizes/dimensions at (1,1,1)/identity and merely bakes the affine into
    # the points, which loses the physical frame -- freeview then refuses to place
    # the tract, and any metre->millimetre factor in the affine silently shrinks a
    # whole brain to a sub-millimetre speck.
    voxel_sizes = np.linalg.norm(affine[:3, :3], axis=0).astype(np.float32)
    if streamlines:
        all_pts = np.concatenate(streamlines, axis=0)
        dims = np.clip(
            np.ceil(np.abs(all_pts).max(axis=0)).astype(np.int64) + 1, 1, 32767
        ).astype(np.int16)
    else:
        dims = np.array([1, 1, 1], dtype=np.int16)
    header = {
        Field.VOXEL_SIZES: tuple(float(v) for v in voxel_sizes),
        Field.DIMENSIONS: tuple(int(v) for v in dims),
        Field.VOXEL_TO_RASMM: affine,
    }

    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        tractogram = nib.streamlines.Tractogram(
            streamlines=streamlines,
            affine_to_rasmm=affine,
        )
        trk_file = TrkFile(tractogram=tractogram, header=header)
        trk_file.save(str(output_path))
    except Exception as e:
        raise ExportError(f"Failed to write TRK '{output_path}': {e}") from e

    return {
        "streamline_count": n_streamlines,
        "vertex_count": total_vertices,
    }
