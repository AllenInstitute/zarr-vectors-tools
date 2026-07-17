"""Offsets-aware link reads shared by the algorithms and strategies.

Core stores connectivity as ONE family under ``links/<delta>/<offsets>/``:
an intra-chunk link is simply one whose offsets are all zero, not a
separate family.  Two consequences drive everything here.

**The double-count trap.**  Before the merge, ``read_links`` meant
*intra-chunk only* and had a sibling ``read_cross_chunk_links``, so the
idiom was "read_chunk_links per chunk, then read_cross_chunk_links".
``read_links`` now returns the WHOLE family, so that idiom unions every
intra edge twice — silently doubling degree, with no error.  Code ported
from the old shape must either use ``read_links`` alone (dropping the
per-chunk loop) or use :func:`read_cross_links` here.  There is exactly
one definition of the cross filter, and it is in this module; that is why
this module does not re-export a bare ``read_links``.

**Groups are not arrays.**  ``links/<delta>`` is a group whose children
are one array per offsets segment.  Naming the group where an array is
expected fails *silently*: ``list_chunks`` returns ``[]`` for it, and the
batch reader skips any prefetch entry that is not an array.  Hence
:func:`link_prefetch_plan`, which enumerates segments so the plan names
arrays and prefetch actually happens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Sequence

from zarr_vectors.constants import LINK_FRAGMENTS
from zarr_vectors.core.arrays import (
    link_family_policy,
    list_link_offsets,
    read_links,
)
from zarr_vectors.core.paths import (
    is_intra,
    link_attributes_path,
    links_group_path,
    links_path,
    parse_offsets,
)

if TYPE_CHECKING:
    from zarr_vectors.core.store import FsGroup
    from zarr_vectors.typing import ChunkCoords

__all__ = [
    "chunk_key_str",
    "cross_offset_segments",
    "link_offset_segments",
    "link_prefetch_plan",
    "list_link_cells",
    "read_cross_links",
    "require_link_width",
]


def chunk_key_str(chunk: Sequence[int]) -> str:
    """Format chunk coords as the dotted on-disk cell key."""
    return ".".join(str(int(c)) for c in chunk)


def _segment_offsets(
    level_group: FsGroup, seg: str, delta: int, link_width: int, sid_ndim: int | None,
) -> tuple[ChunkCoords, ...] | None:
    """Offsets for one segment, from the array's meta or the path.

    Prefers the array's own ``offsets`` meta because it decodes without
    ``sid_ndim``; falls back to parsing the segment name.  The two agree —
    core writes both — so this is a robustness ladder, not a choice.
    """
    meta = level_group.read_array_meta(f"{links_group_path(delta)}/{seg}") or {}
    raw = meta.get("offsets")
    if raw is not None:
        return tuple(tuple(int(c) for c in o) for o in raw)
    if sid_ndim is None:
        return None
    try:
        return parse_offsets(seg, sid_ndim=sid_ndim, link_width=link_width)
    except ValueError:
        return None


def link_offset_segments(
    level_group: FsGroup, *, delta: int = 0,
) -> list[tuple[str, tuple[ChunkCoords, ...]]]:
    """``(segment, offsets)`` for every array under ``links/<delta>/``.

    Includes the all-zero (intra) segment — it is not a separate family.
    Returns ``[]`` when the family is absent or carries no policy.
    """
    policy = link_family_policy(level_group, delta)
    if policy is None:
        return []
    link_width, sid_ndim, _directed, _store = policy
    out: list[tuple[str, tuple[ChunkCoords, ...]]] = []
    for seg in list_link_offsets(level_group, delta):
        offsets = _segment_offsets(level_group, seg, delta, link_width, sid_ndim)
        if offsets is not None:
            out.append((seg, offsets))
    return out


def cross_offset_segments(
    level_group: FsGroup, *, delta: int = 0,
) -> list[tuple[str, tuple[ChunkCoords, ...]]]:
    """``(segment, offsets)`` for the NON-intra arrays under ``links/<delta>/``.

    Filters on the segment's offsets rather than on decoded records: the
    segment name already says whether its records straddle chunks, so this
    costs one listing, not a decode.
    """
    return [
        (seg, offsets)
        for seg, offsets in link_offset_segments(level_group, delta=delta)
        if not is_intra(offsets)
    ]


def read_cross_links(
    level_group: FsGroup, *, delta: int = 0,
) -> list[tuple[tuple[ChunkCoords, int], ...]]:
    """Every record under ``links/<delta>`` spanning two or more chunks.

    The replacement for the deleted ``read_cross_chunk_links``.  Filters on
    decoded chunk identity rather than on the offsets segment, because
    ``store="duplicate"`` and ``perm_idx`` make the segment-to-record
    mapping non-obvious and ``read_links`` is the only public reader that
    reverses ``perm_idx`` back to input order.

    Prefer plain ``read_links`` when you want the whole family: unioning
    this with a per-chunk ``read_chunk_links`` loop is the intended use,
    and unioning ``read_links`` with one is the double-count bug.
    """
    return [
        record
        for record in read_links(level_group, delta=delta)
        if len({tuple(cc) for cc, _vi in record}) > 1
    ]


def list_link_cells(
    level_group: FsGroup,
    *,
    delta: int = 0,
    involves: ChunkCoords | None = None,
) -> list[tuple[ChunkCoords, ...]]:
    """Chunk tuples of every populated cross-chunk cell at ``delta``.

    The offsets-layout successor to ``list_cross_chunk_link_leaves``.  A
    cell is ``(offsets_segment, source_chunk)``; endpoint 0 IS the source
    and endpoint k is ``source + o_k``, which is the inverse of the
    placement arithmetic in ``boundary.partition_records_by_offset``.

    Cells are *listed*, never decoded, so this stays O(cells) rather than
    O(records) — the property the parallel coarseners depend on when they
    bucket cells to target chunks in a single pass.  Core's
    ``iter_link_cells`` decodes every cell and is the wrong tool here.

    Intra cells are excluded: callers bucket cells that *straddle* chunks,
    which is what the old per-``k{K}`` leaf listing returned.  When
    ``involves`` is given, only cells containing that chunk are returned.

    ``delta == 0`` only — for ``delta != 0`` the endpoints live on a
    different chunk grid and reconstruction needs ``boundary.anchor_chunk``
    with both levels' scales.
    """
    if delta != 0:
        raise NotImplementedError(
            "list_link_cells supports delta=0 only; cross-level cells need "
            "anchor_chunk with both levels' scales."
        )
    target = tuple(int(x) for x in involves) if involves is not None else None
    out: set[tuple[ChunkCoords, ...]] = set()
    for _seg, offsets in cross_offset_segments(level_group, delta=delta):
        for key in level_group.list_chunks(links_path(delta, offsets)):
            try:
                src = tuple(int(p) for p in key.split("."))
            except ValueError:
                continue
            cell = (src,) + tuple(
                tuple(s + int(o) for s, o in zip(src, off)) for off in offsets
            )
            if target is not None and target not in set(cell):
                continue
            out.add(cell)
    return sorted(out, key=lambda t: (len(t), t))


def link_prefetch_plan(
    level_group: FsGroup,
    chunk_keys: Iterable[ChunkCoords],
    *,
    delta: int = 0,
    attrs: Sequence[str] = (),
) -> list[tuple[str, list[str]]]:
    """``(array_path, [chunk_key])`` prefetch entries for the link family.

    One entry per offsets segment.  ``links/<delta>`` is a GROUP, so
    naming it prefetches NOTHING — the batch reader skips any plan node
    that is not an array, and every read then silently falls back to a
    serial GET.  Paths are composed via ``links_path`` rather than
    string-joined so the offsets convention keeps one definition.

    Prefetching a cell that does not exist is free, so the same key list
    is used for every segment: callers that want the whole level want
    exactly that, and a missing cell is simply omitted from the results.
    """
    keys = [chunk_key_str(cc) for cc in chunk_keys]
    plan: list[tuple[str, list[str]]] = [(LINK_FRAGMENTS, keys)]
    for _seg, offsets in link_offset_segments(level_group, delta=delta):
        plan.append((links_path(delta, offsets), keys))
        for name in attrs:
            plan.append((link_attributes_path(name, delta, offsets), keys))
    return plan


def require_link_width(
    level_group: FsGroup, expected: int, *, what: str, delta: int = 0,
) -> int:
    """Assert the level's link family has ``link_width == expected``.

    Raises ``NotImplementedError`` otherwise.  A store with no links
    family is treated as absent rather than as ``expected``: guessing the
    caller's width for a store that has no connectivity at all is how a
    graph store gets silently processed as a triangle mesh.
    """
    policy = link_family_policy(level_group, delta)
    if policy is None:
        raise NotImplementedError(
            f"{what} needs a links family with link_width={expected}; "
            f"this level has no links/{delta} family."
        )
    link_width = policy[0]
    if link_width != expected:
        raise NotImplementedError(
            f"{what} supports link_width={expected} only; "
            f"store has link_width={link_width}."
        )
    return link_width
