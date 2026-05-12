"""Global ↔ (chunk, local) vertex index mapping.

Workaround for the missing core ``chunk_local_to_global`` API
(catalog Add 4). Builds the forward direction by streaming the
per-chunk vertex counts via the existing public ``read_chunk_vertices``.

Retire / re-point at the core helper once it lands.
"""

from __future__ import annotations

import numpy as np

from zarr_vectors.core.arrays import list_chunk_keys, read_chunk_vertices
from zarr_vectors.typing import ChunkCoords


def build_chunk_to_global_offset(
    level_group,
    *,
    ndim: int = 3,
    dtype: str = "float32",
) -> tuple[dict[ChunkCoords, int], list[ChunkCoords], int]:
    """Stream chunk vertex counts and build the global-index offset table.

    Args:
        level_group: Resolution level group.
        ndim: Coordinate dimensionality (used by ``read_chunk_vertices``).
        dtype: Vertex dtype (used by ``read_chunk_vertices``).

    Returns:
        Tuple of:
          - ``offsets``: ``{chunk_key: start_global_index}``. The vertices
            in this chunk occupy ``[start, start + chunk_vertex_count)``
            in global ordering.
          - ``chunk_keys``: deterministic order matching the offset build.
          - ``total_vertices``: sum of per-chunk vertex counts.
    """
    chunk_keys = list_chunk_keys(level_group)
    offsets: dict[ChunkCoords, int] = {}
    running = 0
    for key in chunk_keys:
        offsets[key] = running
        try:
            groups = read_chunk_vertices(level_group, key, dtype=dtype, ndim=ndim)
            for vg in groups:
                running += len(vg)
        except Exception:
            # Treat unreadable chunks as empty; consistent with read_mesh.
            pass
    return offsets, chunk_keys, running


def resolve_endpoint(
    endpoint: tuple[ChunkCoords, int],
    offsets: dict[ChunkCoords, int],
) -> int:
    """Translate a ``(chunk_coords, local_index)`` endpoint to a global ID.

    Raises:
        KeyError: If the chunk is absent from the offset table.
    """
    chunk_key, local = endpoint
    return offsets[chunk_key] + int(local)


def resolve_endpoints(
    endpoints: list[tuple[ChunkCoords, int]],
    offsets: dict[ChunkCoords, int],
) -> np.ndarray:
    """Vectorised :func:`resolve_endpoint` for a list of pairs."""
    return np.array(
        [offsets[c] + int(li) for c, li in endpoints],
        dtype=np.int64,
    )
