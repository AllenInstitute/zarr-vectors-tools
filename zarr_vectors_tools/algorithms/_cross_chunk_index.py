"""Per-chunk index into the global cross-chunk array.

Workaround for the missing core ``read_cross_chunk_links_for_chunk``
API (catalog Add 2). Reads the global cross-link list once and
groups it by chunk on the tools side.

Retire once the core stores a chunk-keyed sidecar at write time.
"""

from __future__ import annotations

from zarr_vectors.core.arrays import read_cross_chunk_links
from zarr_vectors.typing import ChunkCoords, CrossChunkLink


def build_cross_chunk_index(
    level_group,
) -> tuple[list[CrossChunkLink], dict[ChunkCoords, list[int]]]:
    """Read all cross-chunk links and group by participating chunk.

    Args:
        level_group: Resolution level group.

    Returns:
        Tuple of:
          - ``links``: the flat list returned by
            :func:`zarr_vectors.core.arrays.read_cross_chunk_links`.
          - ``per_chunk``: ``{chunk_key: [row_indices into ``links``]}``.
            A link touching two chunks appears in both chunks' lists.
            Stores with no cross-chunk links return an empty list and
            an empty dict.
    """
    try:
        links = read_cross_chunk_links(level_group)
    except Exception:
        return [], {}

    per_chunk: dict[ChunkCoords, list[int]] = {}
    for i, ((chunk_a, _), (chunk_b, _)) in enumerate(links):
        per_chunk.setdefault(chunk_a, []).append(i)
        if chunk_b != chunk_a:
            per_chunk.setdefault(chunk_b, []).append(i)
    return links, per_chunk
