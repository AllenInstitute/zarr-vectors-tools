"""Export zarr vectors streamlines to TRX format.

Requires ``trx-python``: ``pip install trx-python``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from zarr_vectors.exceptions import ExportError
from zarr_vectors.types.polylines import read_polylines
from zarr_vectors.typing import BoundingBox


def export_trx(
    store_path: str | Path,
    output_path: str | Path,
    *,
    level: int = 0,
    object_ids: list[int] | None = None,
    group_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Export zarr vectors streamlines to a TRX file.

    Args:
        store_path: Path to the zarr vectors store.
        output_path: Path for the output .trx file.
        level: Resolution level to export.
        object_ids: Optional object ID filter.
        group_ids: Optional group ID filter.

    Returns:
        Summary dict with ``streamline_count``, ``vertex_count``.

    Raises:
        ExportError: If trx-python is not installed or export fails.
    """
    try:
        from trx.trx_file_memmap import TrxFile
    except ImportError as e:
        raise ExportError(
            "trx-python is required for TRX export. "
            "Install with: pip install trx-python"
        ) from e

    try:
        result = read_polylines(
            str(store_path),
            level=level,
            object_ids=object_ids,
            group_ids=group_ids,
        )
    except Exception as e:
        raise ExportError(f"Failed to read store: {e}") from e

    poly_list = result["polylines"]
    n_streamlines = len(poly_list)

    if n_streamlines == 0:
        raise ExportError("No streamlines to export")

    # Reconstruct full streamlines by concatenating segments
    streamlines: list[np.ndarray] = []
    for segments in poly_list:
        full = np.concatenate(segments, axis=0).astype(np.float32)
        streamlines.append(full)

    # Build positions + offsets arrays (TRX layout)
    all_positions = np.concatenate(streamlines, axis=0)
    offsets = np.zeros(n_streamlines, dtype=np.int64)
    cum = 0
    for i, s in enumerate(streamlines):
        offsets[i] = cum
        cum += len(s)

    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        trx = TrxFile(
            nb_vertices=len(all_positions),
            nb_streamlines=n_streamlines,
        )
        trx.streamlines._data = all_positions
        trx.streamlines._offsets = offsets
        trx.save(str(output_path))
    except Exception as e:
        raise ExportError(f"Failed to write TRX '{output_path}': {e}") from e

    return {
        "streamline_count": n_streamlines,
        "vertex_count": len(all_positions),
    }
