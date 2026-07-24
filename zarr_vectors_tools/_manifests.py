"""Re-derive every per-chunk array's ``nonempty_chunks`` manifest.

Workaround for the missing core ``Group.rebuild_nonempty_manifests`` API.
Core ships the per-array half (``Group.derive_nonempty_chunks``) but no
level-wide coordinator; this walks the level and calls it for each array.
Retire once the public helper lands.

Why it is needed at all: ``nonempty_chunks`` is ONE attribute shared by
every cell of an array, so stamping it is a read-modify-write of state
outside the cell being written.  Two workers writing *disjoint* cells
still race on it, and the loser's key vanishes from the manifest even
though its payload landed.  ``list_chunks`` trusts the manifest, so the
loss is silent: the cell is on disk and invisible.

Core solves this for the links family only — ``write_chunk_links`` takes
``record_presence=False`` and ``finalize_links`` re-derives the manifest
afterwards.  Every other per-chunk writer (``write_chunk_vertices``,
``write_chunk_attributes``, ``write_chunk_fragment_attributes``,
``write_chunk_link_attributes``) stamps unconditionally with no opt-out
and no coordinator, which is what leaves this gap for parallel writers.
Link arrays are included here anyway: re-deriving them is idempotent and
cheap, and it keeps the helper correct for callers that never reach
``finalize_links``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from zarr_vectors.constants import (
    FRAGMENT_ATTRIBUTES,
    LINK_ATTRIBUTES,
    LINK_FRAGMENTS,
    LINKS,
    VERTEX_ATTRIBUTES,
    VERTEX_FRAGMENTS,
    VERTICES,
)

if TYPE_CHECKING:
    from zarr_vectors.core.store import FsGroup

__all__ = ["rebuild_nonempty_manifests", "per_chunk_array_paths"]


def _is_per_chunk_array(name: str) -> bool:
    """Whether ``name`` is a per-spatial-chunk array in this level.

    Mirrors ``zarr_vectors.core.arrays._is_per_chunk_array`` (private).
    The test is depth-aware rather than a prefix match because the link
    families nest a group above their arrays — ``links/<delta>`` is a
    GROUP whose children are one array per relative-offset segment — and
    a prefix test would match the group too:

        vertices                          array
        links/0                           GROUP
        links/0/0.0.+1                    array
        link_attributes/weight/0          GROUP
        link_attributes/weight/0/0.0.+1   array

    Excludes ``object_index`` / ``object_attributes`` / ``groups``, which
    are plain arrays with no spatial chunk grid and therefore no manifest.
    """
    if name in {VERTICES, VERTEX_FRAGMENTS, LINK_FRAGMENTS}:
        return True
    parts = name.split("/")
    if len(parts) == 2 and parts[0] in (VERTEX_ATTRIBUTES, FRAGMENT_ATTRIBUTES):
        return True
    # links/<delta>/<offsets>
    if len(parts) == 3 and parts[0] == LINKS:
        return True
    # link_attributes/<name>/<delta>/<offsets>
    if len(parts) == 4 and parts[0] == LINK_ATTRIBUTES:
        return True
    return False


def per_chunk_array_paths(level_group: FsGroup) -> list[str]:
    """Every per-chunk array path in ``level_group``, recursively.

    Walks the group hierarchy rather than guessing names: the link
    families nest two levels deeper than the rest, and their offsets
    segments are only discoverable by listing.
    """
    import zarr

    names: list[str] = []

    def _walk(prefix: str, group: "zarr.Group") -> None:
        for name in group.array_keys():
            path = f"{prefix}{name}"
            if _is_per_chunk_array(path):
                names.append(path)
        for name in group.group_keys():
            _walk(f"{prefix}{name}/", group[name])

    _walk("", level_group.zarr_group)
    return sorted(names)


def rebuild_nonempty_manifests(level_group: FsGroup) -> list[str]:
    """Re-derive ``nonempty_chunks`` for every per-chunk array in the level.

    Run once, single-process, after parallel workers finish and BEFORE
    any sharding pass — ``derive_nonempty_chunks`` reads the store
    listing, which only resolves per-cell keys while each cell is still
    its own object.

    Sharded arrays are skipped, not rebuilt: a shard packs many cells
    into one object whose inner index is not derivable from key names, so
    the listing finds nothing and the manifest would be rewritten to
    empty — turning a stale manifest into a destroyed one.

    Returns the array paths whose manifests were rebuilt.
    """
    rebuilt: list[str] = []
    for name in per_chunk_array_paths(level_group):
        arr = level_group.zarr_group[name]
        if getattr(arr, "shards", None) is not None:
            continue
        level_group.derive_nonempty_chunks(name)
        rebuilt.append(name)
    return rebuilt
