"""Which of the two skeleton layouts a store holds.

Two writers produce ``links_convention = "implicit_sequential_with_branches"``
stores, and they join a tree across a chunk boundary differently:

* :data:`LAYOUT_SPLIT` -- the Neuroglancer precomputed ingests.  A tree is cut
  at every chunk face into pieces; the vertex on the face is written into
  both chunks, and a cross-chunk link joins the two copies.  A segment is
  read back by id (``read_skeleton_by_segment_id``) and the copies joined.
* :data:`LAYOUT_LINKED` -- SWC ingest, through core's
  ``write_graph(kind="skeleton")``.  Every vertex is stored once, and an edge
  whose ends sit in different chunks is a cross-chunk link record
  ``[child, parent]`` that *replaces* the parent the reader would otherwise
  imply -- the rule ``read_graph`` applies.

The skeleton coarsener keeps the layout of the level it reads, and the
exporters need it to pick a reader: the pull-by-segment-id reader is right
for the first and wrong for the second, which has no duplicated vertices
to join.  ``segment_id`` cannot tell them apart, since both carry one.

The ingests record the layout on the root (:func:`mark_skeleton_layout`).
:func:`skeleton_layout` falls back, for a store written before they did, on
what each writer leaves behind: an SWC header or no per-fragment
``segment_id`` means linked, a per-fragment ``segment_id`` without an SWC
header means split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = [
    "LAYOUT_LINKED",
    "LAYOUT_SPLIT",
    "mark_skeleton_layout",
    "skeleton_layout",
]

#: Pieces split at chunk faces, the face vertex duplicated (precomputed).
LAYOUT_SPLIT: str = "split_at_chunk_faces"
#: Every vertex once, cross-chunk edges as link records (SWC).
LAYOUT_LINKED: str = "linked_across_chunks"

#: Root attribute block this package owns; core's ``zarr_vectors`` block is
#: rewritten wholesale by its own writers, so the marker lives beside it.
_TOOLS_ATTRS_KEY = "zarr_vectors_tools"
_LAYOUT_KEY = "skeleton_layout"
_LAYOUTS = (LAYOUT_SPLIT, LAYOUT_LINKED)


def _root(store: Any, mode: str) -> Any:
    from zarr_vectors.building import open_store

    if isinstance(store, (str, Path)):
        return open_store(str(store), mode=mode)
    return store


def mark_skeleton_layout(store: Any, layout: str) -> None:
    """Record ``layout`` on the store's root.

    ``store`` is a path or an open root group.  Pass the handle the writer
    already holds: a root's attributes are written back whole from a
    handle's in-memory copy, so a stale handle written to later would drop
    a marker set through another one.
    """
    if layout not in _LAYOUTS:
        raise ValueError(f"layout must be one of {_LAYOUTS}, got {layout!r}")
    root = _root(store, mode="r+")
    block = dict(root.attrs.get(_TOOLS_ATTRS_KEY) or {})
    block[_LAYOUT_KEY] = layout
    root.attrs.update({_TOOLS_ATTRS_KEY: block})


def skeleton_layout(store: Any) -> str | None:
    """The store's skeleton layout, or ``None`` for a store of other geometry.

    ``store`` is a path or an open root group.
    """
    from zarr_vectors.building import get_resolution_level, read_root_metadata
    from zarr_vectors.constants import FRAGMENT_ATTRIBUTES, LINKS_IMPLICIT_BRANCHES

    root = _root(store, mode="r")
    if read_root_metadata(root).links_convention != LINKS_IMPLICIT_BRANCHES:
        return None
    block = root.attrs.get(_TOOLS_ATTRS_KEY)
    if isinstance(block, dict) and block.get(_LAYOUT_KEY) in _LAYOUTS:
        return str(block[_LAYOUT_KEY])
    # Written before the marker.
    from zarr_vectors_tools.headers.registry import HeaderRegistry

    try:
        if HeaderRegistry(root).has("swc"):
            return LAYOUT_LINKED
    except Exception:  # noqa: BLE001 - no headers is an answer, not an error
        pass
    try:
        level0 = get_resolution_level(root, 0)
        if level0.array_exists(f"{FRAGMENT_ATTRIBUTES}/segment_id"):
            return LAYOUT_SPLIT
    except Exception:  # noqa: BLE001 - no level 0 to look at
        pass
    return LAYOUT_LINKED
