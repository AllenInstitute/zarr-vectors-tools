"""Export zarr vectors streamlines to TrackVis TRK format.

Requires ``nibabel``: ``pip install nibabel``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from zarr_vectors.exceptions import ExportError
from zarr_vectors.types.polylines import read_polylines
from zarr_vectors.typing import BoundingBox


def export_trk(
    store_path: str | Path,
    output_path: str | Path,
    *,
    level: int = 0,
    object_ids: list[int] | None = None,
    group_ids: list[int] | None = None,
    affine: np.ndarray | None = None,
) -> dict[str, Any]:
    """Export zarr vectors streamlines to a TRK file.

    Args:
        store_path: Path to the zarr vectors store.
        output_path: Path for the output .trk file.
        level: Resolution level to export.
        object_ids: Optional object ID filter.
        group_ids: Optional group ID filter.
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

    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        tractogram = nib.streamlines.Tractogram(
            streamlines=streamlines,
            affine_to_rasmm=affine,
        )
        trk_file = TrkFile(tractogram=tractogram)
        trk_file.save(str(output_path))
    except Exception as e:
        raise ExportError(f"Failed to write TRK '{output_path}': {e}") from e

    return {
        "streamline_count": n_streamlines,
        "vertex_count": total_vertices,
    }
