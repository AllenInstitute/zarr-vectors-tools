"""Yield neighbouring chunk keys for halo/DDA algorithms.

Workaround for the missing core ``neighbouring_chunk_keys`` API
(catalog Add 1). Mechanical itertools walk; retire once the public
helper lands in ``zarr_vectors.spatial.chunking``.
"""

from __future__ import annotations

from itertools import product
from typing import Iterable

from zarr_vectors.typing import ChunkCoords


def neighbouring_chunk_keys(
    key: ChunkCoords,
    *,
    halo: int = 1,
    occupied_keys: Iterable[ChunkCoords] | None = None,
    include_self: bool = False,
) -> list[ChunkCoords]:
    """Return chunk keys within ``halo`` cells of ``key`` on every axis.

    Args:
        key: Centre chunk coordinate.
        halo: Inclusive halo radius (1 yields the 26 neighbours of a 3D
            cube, 124 for a 5×5×5 box, and so on).
        occupied_keys: If supplied, the result is filtered to keys that
            appear in this iterable.
        include_self: When True the centre key is included in the result.

    Returns:
        List of chunk coordinates. Order is the natural itertools.product
        ordering.
    """
    if halo < 0:
        raise ValueError(f"halo must be non-negative, got {halo}")

    occupied: set[ChunkCoords] | None = (
        set(occupied_keys) if occupied_keys is not None else None
    )

    deltas = list(product(range(-halo, halo + 1), repeat=len(key)))
    out: list[ChunkCoords] = []
    for delta in deltas:
        if all(d == 0 for d in delta):
            if not include_self:
                continue
        candidate: ChunkCoords = tuple(k + d for k, d in zip(key, delta))
        if occupied is not None and candidate not in occupied:
            continue
        out.append(candidate)
    return out
