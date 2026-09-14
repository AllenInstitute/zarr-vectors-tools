"""Export zarr vectors streamlines to TRX format.

Requires ``trx-python``: ``pip install trx-python``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from zarr_vectors.exceptions import ExportError
from zarr_vectors.types.polylines import read_polylines
from zarr_vectors.typing import BoundingBox, ChunkCoords


def export_trx(
    store_path: str | Path,
    output_path: str | Path,
    *,
    level: int = 0,
    object_ids: list[int] | None = None,
    group_ids: list[int] | None = None,
    chunks: list[ChunkCoords] | None = None,
) -> dict[str, Any]:
    """Export zarr vectors streamlines to a TRX file.

    Args:
        store_path: Path to the zarr vectors store.
        output_path: Path for the output .trx file.
        level: Resolution level to export.
        object_ids: Optional object ID filter.
        group_ids: Optional group ID filter.
        chunks: Optional whitelist of chunk coordinate tuples. Filters at
            the *segment* level: only vertex groups stored in listed
            chunks are emitted, and each surviving contiguous run is
            written as its own streamline. The output ``streamline_count``
            can therefore exceed the source object count.

    Returns:
        Summary dict with ``streamline_count``, ``vertex_count``.

    Raises:
        ExportError: If trx-python is not installed or export fails.
    """
    try:
        from trx.trx_file_memmap import TrxFile, save as trx_save
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
            chunks=chunks,
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

    # Build positions + offsets + lengths arrays (TRX layout).  All three are
    # required: the reader slices ``_data`` by ``_lengths``, so a file written
    # with the zero-filled default comes back as N empty streamlines.
    all_positions = np.concatenate(streamlines, axis=0).astype(np.float32)
    lengths = np.asarray([len(s) for s in streamlines], dtype=np.uint32)
    offsets = np.concatenate(
        [[0], np.cumsum(lengths[:-1])],
    ).astype(np.uint32)

    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        trx = TrxFile(
            nb_vertices=len(all_positions),
            nb_streamlines=n_streamlines,
        )
        # REPLACE the pre-allocated arrays rather than assigning into them.
        # ``TrxFile`` allocates ``_data`` as **float16**, so writing into it
        # silently quantises every coordinate (measured: 7.8e-4 mm of error on
        # ordinary RASmm positions).  Rebinding hands the saver the float32
        # buffer instead, and the round-trip is then exact.
        trx.streamlines._data = np.ascontiguousarray(
            all_positions, dtype=np.float32,
        )
        trx.streamlines._offsets = offsets
        trx.streamlines._lengths = lengths
        # Module-level ``save``, not ``TrxFile.save`` -- there is no such
        # method, so every export raised AttributeError, wrapped as
        # "Failed to write TRX" by the except below and never noticed
        # because the test asserted only that *something* was raised.
        trx_save(trx, str(output_path))
        trx.close()
    except Exception as e:
        raise ExportError(f"Failed to write TRX '{output_path}': {e}") from e

    return {
        "streamline_count": n_streamlines,
        "vertex_count": len(all_positions),
    }
