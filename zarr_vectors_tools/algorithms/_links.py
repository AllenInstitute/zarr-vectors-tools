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

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from zarr_vectors.building import (
    apply_perm_inverse,
    is_intra,
    link_attributes_path,
    link_family_policy,
    links_group_path,
    links_has_perm,
    links_path,
    list_link_offsets,
    parse_offsets,
    read_chunk_links,
    read_links,
    read_links_for_tuple,
)
from zarr_vectors.constants import LINK_FRAGMENTS

if TYPE_CHECKING:
    from zarr_vectors.building import Group
    from zarr_vectors.typing import ChunkCoords

__all__ = [
    "chunk_key_str",
    "cross_offset_segments",
    "link_cell_prefetch_plan",
    "link_cell_reads",
    "link_offset_segments",
    "link_prefetch_plan",
    "list_link_cells",
    "prune_prefetch_plan",
    "read_cross_links",
    "read_link_cell_records",
    "require_link_width",
]


def chunk_key_str(chunk: Sequence[int]) -> str:
    """Format chunk coords as the dotted on-disk cell key."""
    return ".".join(str(int(c)) for c in chunk)


def _segment_offsets(
    level_group: Group, seg: str, delta: int, link_width: int, sid_ndim: int | None,
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
    level_group: Group, *, delta: int = 0,
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
    level_group: Group, *, delta: int = 0,
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
    level_group: Group, *, delta: int = 0,
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
    level_group: Group,
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


def link_cell_reads(
    level_group: Group, *, delta: int = 0,
) -> tuple[list[tuple[ChunkCoords, ...]], list[int], dict[str, Any]]:
    """Every cross-chunk cell, plus what a worker needs to read it blind.

    The coordinator-side half of :func:`read_link_cell_records`.  Returns
    ``(cells, segments, spec)``: ``cells`` exactly as :func:`list_link_cells`
    lists them (same tuples, same order), ``segments[i]`` the index into
    ``spec["arrays"]`` of the offsets array cell ``i`` lives in, and ``spec``
    the family policy (``link_width``, ``directed``, ``store``), the intra
    array's path (``intra_path``, ``None`` if the level has none) and, per
    cross array, its ``path``, ``offsets``, ``dtype`` and ``has_perm``.

    ``spec`` is small and picklable, so it travels to workers once per phase.
    With it a worker reads a cell's records from nothing but the cell itself:
    no family-policy lookup, no segment listing, no per-segment metadata and
    no presence probe -- which :func:`read_links_for_tuple` repeats for every
    cell and which dominated the chunk-local coarseners' Phase A on a local
    filesystem (every lookup is a ``zarr.json`` open).  Costs the same
    listings :func:`list_link_cells` already paid.

    ``delta == 0`` only, for the reason :func:`list_link_cells` gives.
    """
    if delta != 0:
        raise NotImplementedError(
            "link_cell_reads supports delta=0 only; cross-level cells need "
            "anchor_chunk with both levels' scales."
        )
    spec: dict[str, Any] = {
        "link_width": None, "directed": False, "store": "canonical",
        "intra_path": None, "arrays": [],
    }
    policy = link_family_policy(level_group, delta)
    if policy is None:
        return [], [], spec
    link_width, sid_ndim, directed, store = policy
    spec.update(link_width=link_width, directed=directed, store=store)
    found: list[tuple[tuple[int, ...], tuple[ChunkCoords, ...], int]] = []
    for seg in list_link_offsets(level_group, delta):
        path = f"{links_group_path(delta)}/{seg}"
        meta = level_group.read_array_meta(path) or {}
        raw = meta.get("offsets")
        if raw is not None:
            offsets = tuple(tuple(int(c) for c in o) for o in raw)
        elif sid_ndim is not None:
            try:
                offsets = parse_offsets(seg, sid_ndim=sid_ndim, link_width=link_width)
            except ValueError:
                continue
        else:
            continue
        if is_intra(offsets):
            spec["intra_path"] = links_path(delta, offsets)
            continue
        index = len(spec["arrays"])
        spec["arrays"].append({
            "path": links_path(delta, offsets),
            "offsets": [list(o) for o in offsets],
            "dtype": str(meta.get("dtype", "int64")),
            # The rule read_links_for_tuple applies: trust the array's stamp,
            # else recompute it from the family policy.
            "has_perm": bool(meta.get(
                "has_perm",
                links_has_perm(offsets, delta=delta, directed=directed, store=store),
            )),
        })
        for key in level_group.list_chunks(links_path(delta, offsets)):
            try:
                src = tuple(int(p) for p in key.split("."))
            except ValueError:
                continue
            cell = (src,) + tuple(
                tuple(s + int(o) for s, o in zip(src, off)) for off in offsets
            )
            found.append((cell, offsets, index))
    # list_link_cells' order and de-duplication (a cell names one array).
    by_cell: dict[tuple[ChunkCoords, ...], int] = {}
    for cell, _offsets, index in found:
        by_cell.setdefault(cell, index)
    cells = sorted(by_cell, key=lambda t: (len(t), t))
    return cells, [by_cell[c] for c in cells], spec


def link_cell_prefetch_plan(
    cells: Sequence[Sequence[ChunkCoords]],
    segments: Sequence[int],
    spec: dict[str, Any],
) -> list[tuple[str, list[str]]]:
    """``batched_reads`` entries for exactly the cells named, grouped by array.

    Every entry exists (the cells came from :func:`link_cell_reads`), so the
    plan needs no pruning against the presence manifests.
    """
    keys: dict[str, list[str]] = {}
    for cell, seg in zip(cells, segments):
        keys.setdefault(spec["arrays"][seg]["path"], []).append(chunk_key_str(cell[0]))
    return [(path, sorted(set(ks))) for path, ks in keys.items()]


def read_link_cell_records(
    level_group: Group,
    cell: Sequence[ChunkCoords],
    segment: int,
    spec: dict[str, Any],
) -> list[tuple[tuple[ChunkCoords, int], ...]]:
    """``read_links_for_tuple(level_group, cell, delta=0)`` from a known cell.

    ``cell``, ``segment`` and ``spec`` come from :func:`link_cell_reads`.
    Decodes through the public per-cell reader
    :func:`~zarr_vectors.building.read_chunk_links` and returns the records
    ``read_links_for_tuple`` would -- same order, input endpoint order,
    ``perm_idx`` reversed -- without resolving the family policy, the level
    scales or the presence manifest again.  At ``delta == 0`` the tuple's
    anchor is its own source chunk, so the listed cell is the one that
    function would resolve; the single exception, an unsorted tuple on an
    undirected canonical family (which that function sorts first), is
    handed to it unchanged.
    """
    link_width = int(spec["link_width"])
    chunks = tuple(tuple(int(c) for c in ch) for ch in cell)
    if (
        not spec["directed"] and spec["store"] == "canonical"
        and tuple(sorted(chunks)) != chunks
    ):
        return read_links_for_tuple(level_group, chunks, delta=0)
    array = spec["arrays"][segment]
    offsets = tuple(tuple(int(c) for c in o) for o in array["offsets"])
    has_perm = bool(array["has_perm"])
    ncols = link_width + 1 if has_perm else link_width
    groups = read_chunk_links(
        level_group, chunks[0], dtype=array["dtype"], link_width=link_width,
        delta=0, offsets=offsets,
    )
    if not groups:
        return []
    rows = np.concatenate(
        [np.asarray(g, dtype=np.int64).reshape(-1, ncols) for g in groups], axis=0,
    )
    if rows.size == 0:
        return []
    if not has_perm:
        return [tuple(zip(chunks, vis)) for vis in rows[:, :link_width].tolist()]
    return [
        tuple(apply_perm_inverse(
            list(zip(chunks, row[1:1 + link_width])), int(row[0]), link_width,
        ))
        for row in rows.tolist()
    ]


def link_prefetch_plan(
    level_group: Group,
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


def prune_prefetch_plan(
    level_group: Group, plan: list[tuple[str, list[str]]],
) -> list[tuple[str, list[str]]]:
    """Drop the cells ``plan`` names that the presence manifests say are absent.

    ``batched_reads`` treats a missing cell as free, and against an object
    store it nearly is; against a local directory the direct reader pays a
    failed ``open()`` per absent cell, which on NTFS costs about what a hit
    does.  A links family has one array per offsets segment, so a plan from
    :func:`link_prefetch_plan` is mostly absent cells -- measured on a
    100k-streamline pyramid level: 2,600 opens per target chunk, most of them
    misses.  One listing per array, cached for the life of an enclosing
    ``cached_nodes`` block, prunes them.  An array that cannot be listed is
    left to the reader as it was.
    """
    out: list[tuple[str, list[str]]] = []
    for name, keys in plan:
        try:
            present = set(level_group.list_chunks(name))
        except Exception:
            out.append((name, keys))
            continue
        kept = [k for k in keys if k in present]
        if kept:
            out.append((name, kept))
    return out


def require_link_width(
    level_group: Group, expected: int, *, what: str, delta: int = 0,
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
