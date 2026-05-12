"""Read per-edge (link) attributes from a chunked store.

Workaround for the missing core ``read_chunk_link_attributes`` API
(catalog Add 7). The core ships ``write_chunk_link_attributes`` and a
vertex-attribute reader, but no reader for the ``link_attributes/`` path
where edge weights / types live. Without this, every consumer that
catches the resulting ``ArrayError`` silently falls back to unit weights
— see ``graph_search._build_adjacency`` before this helper landed.

Retire once a public reader lands in ``zarr_vectors.core.arrays``.
"""

from __future__ import annotations

import numpy as np

from zarr_vectors.constants import LINK_ATTRIBUTES
from zarr_vectors.typing import ChunkCoords


def read_chunk_link_attributes(
    level_group,
    attr_name: str,
    chunk_coords: ChunkCoords,
    link_group_sizes: list[int],
) -> list[np.ndarray] | None:
    """Read a per-edge attribute for a single chunk.

    Args:
        level_group: Resolution level group.
        attr_name: Edge-attribute name as written by ``write_graph``
            (``edge_attributes={name: ...}``).
        chunk_coords: Spatial chunk coordinates.
        link_group_sizes: Sizes of the chunk's link groups in order;
            the flat attribute buffer is split using these. Pass the
            result of ``[len(g) for g in read_chunk_links(...)]``.

    Returns:
        List of 1-D arrays aligned with the link groups, or ``None`` if
        the attribute isn't stored for this chunk. The dtype comes from
        the array's ``link_attributes/<name>/.zattrs``.
    """
    full_name = f"{LINK_ATTRIBUTES}/{attr_name}"

    try:
        meta = level_group.read_array_meta(full_name)
    except Exception:
        return None
    dtype = np.dtype(meta.get("dtype", "float32"))

    # Chunk-key format is the same as everywhere else: "x_y_z" joined by
    # underscores. Reuse the same private helper the rest of the core uses
    # by importing it lazily — it's not part of the public surface but is
    # stable across the relevant versions.
    from zarr_vectors.core.arrays import _chunk_key  # type: ignore[attr-defined]

    key = _chunk_key(chunk_coords)
    try:
        raw = level_group.read_bytes(full_name, key)
    except Exception:
        return None

    flat = np.frombuffer(raw, dtype=dtype)
    expected = int(sum(link_group_sizes))
    if expected and len(flat) != expected:
        # Partial / stale write: refuse to guess.
        return None

    out: list[np.ndarray] = []
    cursor = 0
    for n in link_group_sizes:
        out.append(np.ascontiguousarray(flat[cursor:cursor + n]))
        cursor += n
    return out
