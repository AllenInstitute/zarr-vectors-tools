"""Per-object face groups, so a reader can fetch one object's faces.

A level stores each chunk's intra-chunk links (a mesh's faces) as flat rows
plus a ``link_fragments`` sidecar giving the row range of each link group.
The format lets groups be anything.  When there is exactly one group per
vertex fragment, in fragment order (empty where a fragment has no
intra-chunk faces), a reader drawing one object can range-read just its
rows instead of the whole cell.  The intra array declares that layout with
``{"link_groups": "per_vertex_fragment"}``.

The face bytes are the same whichever way they are grouped, and
zarr-vectors-py's readers concatenate the groups, so declaring or regrouping
changes nothing for readers that do not use it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

LINK_GROUPS_ATTR = "link_groups"
PER_VERTEX_FRAGMENT = "per_vertex_fragment"


def fragment_of_rows(fragments: Any, n_rows: int | None = None) -> npt.NDArray[np.int64]:
    """Fragment index of each vertex row of a chunk (``-1`` where none).

    ``fragments`` is a decoded fragment index (``num_fragments``,
    ``is_range``, ``range``, ``indices``).
    """
    spans = []
    top = 0
    for f in range(fragments.num_fragments):
        if fragments.is_range(f):
            start, count = (int(x) for x in fragments.range(f))
            rows = np.arange(start, start + count, dtype=np.int64)
        else:
            rows = np.asarray(fragments.indices(f), dtype=np.int64)
        spans.append(rows)
        if rows.size:
            top = max(top, int(rows.max()) + 1)
    out = np.full(max(top, n_rows or 0), -1, dtype=np.int64)
    for f, rows in enumerate(spans):
        out[rows] = f
    return out


def split_faces_by_fragment(
    faces: npt.NDArray[np.integer],
    fragment_of: npt.NDArray[np.int64],
    num_fragments: int,
) -> list[npt.NDArray[np.integer]] | None:
    """One group of rows per fragment, in fragment order (empty where a
    fragment has none), or ``None`` when some row's corners do not all lie in
    one fragment -- then no per-fragment grouping exists."""
    faces = np.asarray(faces)
    width = faces.shape[1] if faces.ndim == 2 else 3
    faces = faces.reshape(-1, width)
    if faces.size and (faces.min() < 0 or faces.max() >= len(fragment_of)):
        return None
    owner = fragment_of[faces] if faces.size else np.zeros((0, width), np.int64)
    if (owner < 0).any() or (owner != owner[:, :1]).any():
        return None
    order = np.argsort(owner[:, 0], kind="stable")
    sorted_faces = faces[order]
    bounds = np.searchsorted(owner[order, 0], np.arange(num_fragments + 1))
    return [sorted_faces[bounds[f]:bounds[f + 1]] for f in range(num_fragments)]


def intra_links_name(sid_ndim: int, link_width: int) -> str:
    from zarr_vectors.core.paths import intra_offsets, links_path

    return links_path(0, intra_offsets(sid_ndim, link_width))


def declare_per_fragment_groups(level_group: Any, sid_ndim: int, link_width: int) -> None:
    """Record on the level's intra array that its groups follow the fragments."""
    level_group.write_array_meta(
        intra_links_name(sid_ndim, link_width),
        {LINK_GROUPS_ATTR: PER_VERTEX_FRAGMENT},
    )


def index_mesh_faces(
    store_path: str,
    levels: list[int] | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Regroup an existing store's intra-chunk faces one group per vertex
    fragment, and declare it on each level where every chunk allows it.

    Cells are rewritten as they go -- a regrouped cell is valid whether or not
    the level ends up declared -- and a level is declared only once all of its
    chunks have been regrouped.  A level where some face's corners lie in two
    fragments is left undeclared, and the report says which chunk.
    """
    from zarr_vectors.building import (
        get_resolution_level,
        open_store,
        read_root_metadata,
    )
    from zarr_vectors.core.arrays import (
        read_chunk_links,
        read_vertex_fragment_index,
        write_chunk_links,
    )
    from zarr_vectors.core.paths import links_group_path
    from zarr_vectors.core.store import list_resolution_levels

    root = open_store(str(store_path), mode="r" if dry_run else "r+")
    ndim = len(read_root_metadata(root).chunk_shape)
    if levels is None:
        levels = list_resolution_levels(root)
    reports: list[dict[str, Any]] = []
    for level in levels:
        report: dict[str, Any] = {"level": level, "chunks": 0, "regrouped": 0}
        reports.append(report)
        try:
            level_group = get_resolution_level(root, level)
        except Exception as e:  # noqa: BLE001
            report["skipped"] = f"no level {level} ({e})"
            continue
        family = level_group.read_array_meta(links_group_path(0))
        if not family:
            report["skipped"] = "no links/0 family"
            continue
        link_width = int(family.get("link_width", 2))
        name = intra_links_name(ndim, link_width)
        meta = level_group.read_array_meta(name)
        if not meta:
            report["skipped"] = f"no {name} array"
            continue
        if meta.get(LINK_GROUPS_ATTR) == PER_VERTEX_FRAGMENT:
            report["skipped"] = "already declared"
            continue
        dtype = np.dtype(meta.get("dtype", "int64"))
        keys = meta.get("nonempty_chunks")
        if keys is None:
            report["skipped"] = f"{name} records no nonempty_chunks"
            continue
        ok = True
        for key in keys:
            cc = tuple(int(x) for x in str(key).split("."))
            groups = read_chunk_links(level_group, cc, delta=0, link_width=link_width)
            if not groups:
                continue
            report["chunks"] += 1
            fragments = read_vertex_fragment_index(level_group, cc)
            faces = np.concatenate([np.asarray(g).reshape(-1, link_width) for g in groups])
            fragment_of = fragment_of_rows(fragments)
            split = split_faces_by_fragment(faces, fragment_of, fragments.num_fragments)
            if split is None:
                report["skipped"] = f"chunk {key}: a face spans fragments"
                ok = False
                break
            aligned = len(groups) == len(split) and all(
                len(a) == len(b) and np.array_equal(np.asarray(a).reshape(-1, link_width), b)
                for a, b in zip(groups, split)
            )
            if aligned:
                continue
            report["regrouped"] += 1
            if not dry_run:
                write_chunk_links(level_group, cc, split, dtype=dtype, delta=0,
                                  link_width=link_width)
        if ok:
            report["declared"] = not dry_run
            if not dry_run:
                declare_per_fragment_groups(level_group, ndim, link_width)
    return reports
