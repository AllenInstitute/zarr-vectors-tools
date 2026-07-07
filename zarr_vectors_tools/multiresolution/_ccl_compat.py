"""Cross-chunk-link (CCL) compatibility adapter.

The rich coarsening strategies in this package were written against an
earlier core CCL storage design (the "v0.8 partitioned per-``k{K}`` leaf"
layout: ``create_cross_chunk_link_kN_arrays`` / ``list_cross_chunk_link_leaves``
/ ``read_cross_chunk_link_leaf`` / ``stamp_ccl_capabilities`` +
``CROSS_CHUNK_LINK_SHARD_AXIS``). The core that ships ``types.skeletons``
(AllenInstitute/zarr-vectors-py PR #26) instead stores cross-chunk links with
the *cells / directed / duplicate* model (one file per chunk-tuple cell under
``cross_chunk_links/<delta>/``, ``read_cross_chunk_links`` /
``read_cross_chunk_links_for_tuple`` / ``write_cross_chunk_link_cells`` /
``finalize_cross_chunk_links`` + ``format_capabilities``).

Both share the same *logical* model — a leaf/cell is a tuple of chunk
coordinates, and a record is a tuple of ``(chunk_coords, vertex_index)``
endpoints — so this module re-implements the six v0.8-shaped symbols the
strategies import as thin wrappers over the core cells API. The strategy code
is left untouched; only its import source changes.

Nothing here duplicates on-disk format concerns: creation, per-cell writes,
reads and capability stamping all delegate to core.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from zarr_vectors.constants import CAP_MULTISCALE_LINKS
from zarr_vectors.core.arrays import (
    create_cross_chunk_links_array,
    finalize_cross_chunk_links,  # re-exported for the post-write reconcile pass
    read_cross_chunk_links_for_tuple,
    write_cross_chunk_link_cells as _core_write_cross_chunk_link_cells,
)
from zarr_vectors.core.paths import cross_chunk_links_path, parse_cell_key

if TYPE_CHECKING:
    from zarr_vectors.core.store import FsGroup
    from zarr_vectors.typing import ChunkCoords

__all__ = [
    "COARSEN_SKELETON",
    "CROSS_CHUNK_LINK_SHARD_AXIS",
    "create_cross_chunk_link_kN_arrays",
    "list_cross_chunk_link_leaves",
    "read_cross_chunk_link_leaf",
    "stamp_ccl_capabilities",
    "finalize_cross_chunk_links",
    "write_cross_chunk_link_cells",
]


def write_cross_chunk_link_cells(
    level_group,
    links,
    sid_ndim,
    *,
    delta: int = 0,
    link_width: int | None = None,
    directed: bool = False,
    store: str = "canonical",
    chunk_grid_shape=None,
    chunk_origin=None,
):
    """Core per-cell CCL writer, tolerating the v0.8 geometry kwargs.

    The old per-``k{K}`` layout needed ``chunk_grid_shape`` / ``chunk_origin``
    to size the sharded arrays; the core cells layout derives each cell key
    from the record's chunk coords, so those two args are accepted for
    signature compatibility and ignored.
    """
    return _core_write_cross_chunk_link_cells(
        level_group,
        links,
        sid_ndim,
        delta=delta,
        link_width=link_width,
        directed=directed,
        store=store,
    )


# Coarsening-method tag for skeleton-aware pyramids (was a core constant in the
# v0.8 design; skeleton coarsening is a tools-owned strategy, so it lives here).
COARSEN_SKELETON: str = "skeleton_simplify"

# Outer-shard width per axis used by the strategies purely to group per-cell
# writers into race-safe task partitions (one worker owns each shard's cells).
# Core's cells layout is flat (one file per cell), so this only bounds the
# task-partition granularity; it is not an on-disk value here.
CROSS_CHUNK_LINK_SHARD_AXIS: int = 4


def create_cross_chunk_link_kN_arrays(
    level_group: FsGroup,
    *,
    sid_ndim: int,
    link_width: int,
    chunk_grid_shape: tuple[int, ...] | None = None,
    chunk_origin: tuple[int, ...] | None = None,
    max_K: int | None = None,
    min_K: int = 1,
    directed: bool = False,
) -> None:
    """Pre-create the intra-level (``delta=0``) cross-chunk-link family.

    In the core cells layout there are no per-``k{K}`` sized arrays to
    pre-allocate — a single ``cross_chunk_links/0`` family holds one flat file
    per touched chunk-tuple cell, created on demand by the writers. This wrapper
    just stamps the family meta (``sid_ndim`` / ``link_width`` / ``directed``)
    so decentralized ``write_cross_chunk_link_cells`` workers agree on it. The
    ``chunk_grid_shape`` / ``chunk_origin`` / ``max_K`` / ``min_K`` args are
    accepted for signature compatibility and ignored (the cells layout is not
    grid-sized).
    """
    create_cross_chunk_links_array(
        level_group,
        delta=0,
        link_width=link_width,
        sid_ndim=sid_ndim,
        directed=directed,
        exist_ok=True,
    )


def list_cross_chunk_link_leaves(
    level_group: FsGroup,
    *,
    delta: int = 0,
    involves: ChunkCoords | None = None,
) -> list[tuple[ChunkCoords, ...]]:
    """List populated cross-chunk-link cells (leaves) at ``delta``.

    Each returned element is the ``link_width``-tuple of chunk coordinates
    keying one populated cell — the same value expected by
    :func:`read_cross_chunk_link_leaf`. When ``involves`` is given, only cells
    that contain that chunk in any slot are returned. Sorted by ``(len, key)``.
    """
    full = cross_chunk_links_path(delta)
    if not level_group.array_exists(full):
        return []
    meta = level_group.read_array_meta(full) or {}
    if "link_width" not in meta or "sid_ndim" not in meta:
        return []
    link_width = int(meta["link_width"])
    sid_ndim = int(meta["sid_ndim"])

    target = tuple(int(x) for x in involves) if involves is not None else None
    out: set[tuple[ChunkCoords, ...]] = set()
    for cell_key in level_group.list_chunks(full):
        chunks = parse_cell_key(cell_key, sid_ndim=sid_ndim, link_width=link_width)
        leaf = tuple(tuple(int(y) for y in c) for c in chunks)
        if target is not None and target not in set(leaf):
            continue
        out.add(leaf)
    return sorted(out, key=lambda t: (len(t), t))


def read_cross_chunk_link_leaf(
    level_group: FsGroup,
    chunks: tuple[ChunkCoords, ...],
    *,
    delta: int = 0,
) -> list[tuple[tuple[ChunkCoords, int], ...]]:
    """Read the records stored in the single cell keyed by ``chunks``.

    ``chunks`` is a ``link_width``-tuple of chunk coordinates (as produced by
    :func:`list_cross_chunk_link_leaves`). Each returned record is a tuple of
    ``(chunk_coords, vertex_index)`` endpoints in original input order.
    """
    return read_cross_chunk_links_for_tuple(level_group, chunks, delta=delta)


def stamp_ccl_capabilities(root_group) -> None:
    """Mark the store's root metadata as carrying multiscale links (idempotent).

    The v0.8 design stamped a dedicated ``partitioned_cross_chunk_links`` token;
    the core cells layout has no such capability, so only the recognized
    ``multiscale_links`` token is stamped.
    """
    attrs = root_group.attrs.to_dict()
    zv = attrs.get("zarr_vectors", {})
    caps = list(zv.get("format_capabilities", []))
    if CAP_MULTISCALE_LINKS not in caps:
        caps.append(CAP_MULTISCALE_LINKS)
        zv["format_capabilities"] = caps
        root_group.attrs.update({"zarr_vectors": zv})
