"""Per-object skeleton metrics, written as object attributes.

:func:`compute_skeleton_metrics` walks one level of a skeleton store and
returns, for every object, the metrics below.  With ``write=True`` (the
default) each metric is also stored as ``object_attributes/<name>``, one row
per object id, so it can drive the ``"attribute"`` sparsity strategy or be
exported alongside the geometry.

Metrics (all in the store's coordinate units, at the level asked for):

``cable_length`` (float64)
    Sum of Euclidean edge lengths over the whole object, cross-chunk edges
    included.  A coarse level's decimation shortens it; it is a per-level
    value, not a property of the neuron.
``node_count`` (int64)
    Distinct vertices.  The boundary vertex igneous duplicates into both
    chunks it borders counts once.
``leaf_count`` (int64)
    Vertices with at most one neighbour: path ends, plus an isolated vertex.
    Undirected, so an unbranched root is a tip like any other end.
``branch_count`` (int64)
    Vertices with three or more neighbours.  A soma with three primary
    neurites counts once.
``component_count`` (int64)
    Connected pieces.  An object is a forest when a cutout or a sparse source
    separates it; a single tree reports 1.
``max_strahler`` (int64)
    Horton-Strahler order at the root of the object's highest-order
    component: tips are 1 and a vertex whose two highest-order children tie
    is one higher.  Strahler order depends on the root, so the root is fixed
    by rule: the object's stored root where the store records one (the
    ``write_graph`` layout, e.g. SWC ingest — this matches the per-node
    ``strahler`` the SWC ingester writes), otherwise the tip with the
    smallest ``(x, y, z)``.  Decimation keeps tips and their coordinates, so
    that rule picks the same vertex at every pyramid level and on any chunk
    grid.
``extent`` (float64, ``(O, D)``)
    Per-axis size of the object's axis-aligned bounding box.  Being a vector
    it cannot rank objects on its own; the other six are scalars.

An object a coarser level's sparsity drop has emptied reads 0 for every
count, ``cable_length`` and ``max_strahler``, and NaN for ``extent`` — there
is no box around nothing.

Two skeleton layouts share ``links_convention =
"implicit_sequential_with_branches"`` and are told apart by the links
family's ``directed`` flag:

* **chunked** (``directed=True``) — written chunk by chunk by
  :func:`zarr_vectors.types.skeletons.write_skeleton_chunk`: the precomputed
  EM ingesters and every level the skeleton coarsener writes.  Each fragment
  is a path, a branch link attaches a path's first vertex, and cross-chunk
  links are undirected coincidence matches, so they are added as edges and
  never re-parent anything.
* **graph** (``directed=False``) — written whole by
  :func:`zarr_vectors.types.graphs.write_graph` with ``kind="skeleton"``.
  One fragment per object per chunk, parents implied as the previous row,
  and every stored ``[child, parent]`` record, cross-chunk ones included,
  *replaces* the implied parent — exactly the rule ``read_graph`` applies.

Cost.  One read of every chunk the level holds (vertices, fragment index,
intra-chunk links) plus the cross-chunk link cells touching it, each cell
read by both of its chunks; nothing is re-read per object.  Chunks are
summarised independently — so ``executor`` can fan them out — into per-object
totals and a compressed topology that keeps only tips, branch points,
cross-chunk endpoints and piece roots, with each unbranched run collapsed to
one edge.  Strahler order is unchanged by that collapse, and the compressed
graph is what the coordinator holds: a few percent of the vertices for a
neuron.  Object ownership comes from the chunked layout's per-fragment
``object_id`` attribute when present; otherwise, and whenever ``object_ids``
is given, from the object manifests.

:func:`compute_tree_metrics_vectorized` is the same machinery exposed as a
drop-in for ``convert.ingest._tree_enrichments.compute_tree_metrics``: per
node depth, Strahler order and node kind, from a parent array, without a
per-node Python loop.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from zarr_vectors.building import (
    OBJECT_INDEX,
    create_object_attributes_array,
    get_resolution_level,
    link_family_policy,
    links_path,
    list_chunk_keys,
    open_store,
    read_all_object_manifests,
    read_chunk_fragment_attributes,
    read_chunk_links,
    read_chunk_vertices,
    read_object_attributes,
    read_object_manifests,
    read_root_metadata,
    read_vertex_fragment_index,
    write_object_attributes,
)
from zarr_vectors.constants import LINKS_IMPLICIT_BRANCHES, OBJECT_ATTRIBUTES, VERTICES
from zarr_vectors.exceptions import ArrayError

from zarr_vectors_tools.algorithms._links import cross_offset_segments

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "SKELETON_METRICS",
    "compute_skeleton_metrics",
    "compute_tree_metrics_vectorized",
]

#: Every metric :func:`compute_skeleton_metrics` knows, in column order.
SKELETON_METRICS: tuple[str, ...] = (
    "cable_length",
    "node_count",
    "leaf_count",
    "branch_count",
    "component_count",
    "max_strahler",
    "extent",
)

# The two metrics that need each object's connectivity assembled across
# chunks.  The rest are sums, extrema and degrees, so a request for only those
# skips the compressed topology entirely.
_TOPOLOGY_METRICS = frozenset({"component_count", "max_strahler"})

_LAYOUT_CHUNKED = "chunked"
_LAYOUT_GRAPH = "graph"

# node_kind codes, as ``convert.ingest._tree_enrichments`` defines them.
_SOMA, _BRANCH, _CONTINUATION, _TERMINAL = 0, 1, 2, 3


# ===================================================================
# Store pass
# ===================================================================

def compute_skeleton_metrics(
    store_path: str | Path,
    *,
    level: int = 0,
    metrics: Sequence[str] | None = None,
    write: bool = True,
    object_ids: Sequence[int] | None = None,
    executor: Callable[..., list] | None = None,
) -> pd.DataFrame:
    """Per-object skeleton metrics for one level, optionally stored.

    See the module docstring for what each metric measures, how the root
    Strahler order is taken from is chosen, and what the pass costs.

    Args:
        store_path: Path to a skeleton store (``links_convention =
            "implicit_sequential_with_branches"``).
        level: Resolution level to measure.
        metrics: Names from :data:`SKELETON_METRICS`.  ``None`` computes all.
        write: Store each metric as ``object_attributes/<name>`` on
            ``level``, replacing an existing array of that name.  An
            ``extent`` attribute is ``(O, D)``; the rest are ``(O,)``.
        object_ids: Restrict the pass to these objects (and the chunks their
            manifests name).  Cannot be combined with ``write=True``: a
            partial column would leave every other object a fabricated 0.
        executor: Optional ``map``-like ``(func, items, shared) -> list``
            callable (e.g. ``dask_executor``) that summarises chunks in
            parallel.  ``None`` runs serially in-process.

    Returns:
        DataFrame indexed by ``object_id``: a ``segment_id`` column when the
        level carries ``object_attributes/segment_id``, then one column per
        requested metric, with ``extent`` split into ``extent_<axis>``.

    Raises:
        ValueError: The store is not a skeleton store, a metric name is
            unknown, ``object_ids`` names an object the level does not hold
            or is combined with ``write=True``, or the level's links are not
            two-endpoint records.
    """
    import pandas as pd

    wanted = _resolve_metrics(metrics)
    if write and object_ids is not None:
        raise ValueError(
            "object_ids computes a subset of the objects, and writing it would "
            "store 0 for every object left out; pass write=False with object_ids"
        )

    # Reads run under one node cache: a pass issues several metadata lookups
    # per chunk and per link cell, and resolving each afresh was most of its
    # runtime.  The cache is read-only by contract, so writing opens the store
    # again afterwards.
    read_root = open_store(str(store_path), mode="r")
    with read_root.cached_nodes():
        values, ids, segment_ids, axis_names = _read_pass(
            read_root, store_path, level=level, wanted=wanted,
            object_ids=object_ids, executor=executor,
        )

    columns: dict[str, np.ndarray] = {}
    if segment_ids is not None:
        columns["segment_id"] = segment_ids
    for name in SKELETON_METRICS:
        if name not in wanted:
            continue
        if name == "extent":
            for axis, axis_name in enumerate(axis_names):
                columns[f"extent_{axis_name}"] = values["extent"][ids, axis]
        else:
            columns[name] = values[name][ids]
    frame = pd.DataFrame(columns, index=pd.Index(ids, name="object_id"))

    if write and len(values["extent"]) > 0:
        level_group = get_resolution_level(open_store(str(store_path), mode="r+"), level)
        for name in SKELETON_METRICS:
            if name in wanted:
                create_object_attributes_array(level_group, name)
                write_object_attributes(level_group, name, values[name])
    return frame


def _read_pass(
    root: Any,
    store_path: str | Path,
    *,
    level: int,
    wanted: frozenset[str],
    object_ids: Sequence[int] | None,
    executor: Callable[..., list] | None,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray | None, list[str]]:
    """Everything :func:`compute_skeleton_metrics` reads, for an open store.

    Returns ``(values, ids, segment_ids, axis_names)``: one full-length array
    per metric, the object ids reported, their segment ids (or ``None``), and
    the names the ``extent`` columns take.
    """
    root_meta = read_root_metadata(root)
    if root_meta.links_convention != LINKS_IMPLICIT_BRANCHES:
        raise ValueError(
            f"compute_skeleton_metrics needs a skeleton store "
            f"(links_convention={LINKS_IMPLICIT_BRANCHES!r}); {store_path} has "
            f"geometry_types={list(root_meta.geometry_types or [])} and "
            f"links_convention={root_meta.links_convention!r}"
        )
    ndim = int(root_meta.sid_ndim)
    level_group = get_resolution_level(root, level)
    if OBJECT_INDEX not in level_group:
        raise ValueError(
            f"level {level} of {store_path} has no object_index, so there are no "
            f"objects to measure"
        )
    n_objects = int((level_group.read_array_meta(OBJECT_INDEX) or {}).get("num_objects", 0))

    policy = link_family_policy(level_group, 0)
    if policy is not None and int(policy[0]) != 2:
        raise ValueError(
            f"skeleton links are two-endpoint records; level {level} of "
            f"{store_path} stores link_width={policy[0]}"
        )
    layout = _LAYOUT_CHUNKED if policy is not None and policy[2] else _LAYOUT_GRAPH
    link_store = str(policy[3]) if policy is not None else "canonical"

    requested: npt.NDArray[np.int64] | None = None
    owners_by_chunk: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] | None
    if object_ids is not None:
        requested = np.unique(np.asarray(list(object_ids), dtype=np.int64))
        outside = requested[(requested < 0) | (requested >= n_objects)]
        if outside.size:
            raise ValueError(
                f"object id(s) {outside.tolist()} are not at level {level}, which "
                f"holds {n_objects} objects"
            )
        owners_by_chunk = _owners_from_manifests(
            read_object_manifests(level_group, ids=requested.tolist()).items(),
        )
        chunks = sorted(owners_by_chunk)
    else:
        all_chunks = sorted(
            tuple(int(c) for c in cc) for cc in list_chunk_keys(level_group, VERTICES)
        )
        if layout == _LAYOUT_CHUNKED and _has_fragment_object_ids(level_group, all_chunks):
            # The chunked writers stamp every fragment with its object id, so
            # ownership is read with the chunk and the O(fragments) manifest
            # decode -- the slow step on a store of many segments -- is skipped.
            owners_by_chunk = None
            chunks = all_chunks
        else:
            owners_by_chunk = _owners_from_manifests(
                enumerate(read_all_object_manifests(level_group)),
            )
            chunks = sorted(owners_by_chunk)

    cells_by_chunk: dict[tuple[int, ...], list] = {cc: [] for cc in chunks}
    if policy is not None:
        # Cells are listed, never decoded, here: ``(source, offset)`` names
        # the one on-disk cell a record lives in and both of its chunks.
        for _segment, offsets in cross_offset_segments(level_group):
            (offset,) = offsets
            for key in level_group.list_chunks(links_path(0, offsets)):
                source = tuple(int(p) for p in key.split("."))
                other = tuple(s + int(o) for s, o in zip(source, offset))
                # A canonical record lives in one cell, so both of its chunks
                # must see it: the child's chunk to drop an implied parent,
                # either to mark an endpoint it has to keep.  A duplicate-store
                # family files a copy under every endpoint chunk, so each chunk
                # reads only the cells it is the source of.
                touching = (source,) if link_store == "duplicate" else (source, other)
                for cc in touching:
                    if cc in cells_by_chunk:
                        cells_by_chunk[cc].append((source, offset))

    topology = bool(wanted & _TOPOLOGY_METRICS)
    shared: dict[str, Any] = {
        "store_path": str(store_path),
        "level": int(level),
        "ndim": ndim,
        "layout": layout,
        "link_store": link_store,
        "has_links": policy is not None,
        "vertex_dtype": str(
            np.dtype((level_group.read_array_meta(VERTICES) or {}).get("dtype", "float32"))
        ),
        "n_objects": n_objects,
        "topology": topology,
    }
    payloads = [
        {
            "chunk": cc,
            "cells": cells_by_chunk[cc],
            "owners": None if owners_by_chunk is None else owners_by_chunk[cc],
        }
        for cc in chunks
    ]
    if executor is None:
        # In-process, the open level is reused rather than reopened per chunk;
        # a parallel executor's workers open the store from ``store_path``.
        shared_local = dict(shared, level_group=level_group)
        summaries = [_summarise_chunk(p, shared=shared_local) for p in payloads]
    else:
        summaries = list(executor(_summarise_chunk, payloads, shared))

    values = _assemble(summaries, n_objects=n_objects, ndim=ndim, topology=topology)
    ids = np.arange(n_objects, dtype=np.int64) if requested is None else requested
    segment_ids = None
    if level_group.array_exists(f"{OBJECT_ATTRIBUTES}/segment_id"):
        stored = np.asarray(read_object_attributes(level_group, "segment_id"))
        if len(stored) == n_objects:
            segment_ids = stored.reshape(n_objects, -1)[:, 0][ids]
    return values, ids, segment_ids, _axis_names(root_meta, ndim)



def _resolve_metrics(metrics: Sequence[str] | None) -> frozenset[str]:
    """Validate requested metric names, refusing unknown ones by name."""
    if metrics is None:
        return frozenset(SKELETON_METRICS)
    if isinstance(metrics, str):
        metrics = [metrics]
    names = list(metrics)
    unknown = [m for m in names if m not in SKELETON_METRICS]
    if unknown:
        raise ValueError(
            f"unknown skeleton metric(s) {unknown}; choose from {list(SKELETON_METRICS)}"
        )
    if not names:
        raise ValueError(f"metrics is empty; choose from {list(SKELETON_METRICS)}")
    return frozenset(names)


def _axis_names(root_meta: Any, ndim: int) -> list[str]:
    """Axis names for the ``extent_<axis>`` columns, falling back to ``d<i>``."""
    dims = list(getattr(root_meta, "spatial_index_dims", None) or [])
    names = [str(d.get("name")) for d in dims if isinstance(d, dict) and d.get("name")]
    if len(names) == ndim and len(set(names)) == ndim:
        return names
    return [f"d{i}" for i in range(ndim)]


def _has_fragment_object_ids(level_group: Any, chunks: Sequence[tuple[int, ...]]) -> bool:
    """Whether the level stamps fragments with ``object_id``, probed on a real chunk.

    Probing a chunk rather than listing the attribute group is what the
    skeleton coarsener does too: an attribute array can be declared and yet
    never written.
    """
    if not chunks:
        return False
    probe = read_chunk_fragment_attributes(
        level_group, "object_id", chunks[0], dtype=np.uint64, default=None,
    )
    return probe is not None


def _owners_from_manifests(
    manifests: Iterable[tuple[int, Sequence[tuple[Sequence[int], int]]]],
) -> dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]]:
    """Invert object manifests into per-chunk ``(fragment_index, object_id)`` arrays.

    The first object to name a fragment owns it, matching how the coarseners
    attribute a fragment several manifests reference.
    """
    pairs: dict[tuple[int, ...], dict[int, int]] = {}
    for oid, manifest in manifests:
        for chunk, fragment in manifest:
            owned = pairs.setdefault(tuple(int(c) for c in chunk), {})
            owned.setdefault(int(fragment), int(oid))
    return {
        cc: (
            np.fromiter(owned.keys(), dtype=np.int64, count=len(owned)),
            np.fromiter(owned.values(), dtype=np.int64, count=len(owned)),
        )
        for cc, owned in pairs.items()
    }


# ===================================================================
# Per-chunk summary
# ===================================================================

def _summarise_chunk(payload: dict[str, Any], shared: dict[str, Any] | None = None) -> dict:
    """Summarise one chunk: per-object totals plus its compressed topology.

    Picklable and self-contained so a parallel executor can run it: the
    payload names the chunk, the cross-chunk cells touching it and, when the
    level has no per-fragment ``object_id``, which object owns each fragment.
    """
    sh = shared or {}
    if sh.get("level_group") is None:
        # A worker process: open the store itself, under the same read-only
        # node cache the in-process pass runs in.
        root = open_store(sh["store_path"], mode="r")
        with root.cached_nodes():
            level_group = get_resolution_level(root, sh["level"])
            return _summarise_chunk(payload, shared=dict(sh, level_group=level_group))
    level_group = sh["level_group"]
    ndim = int(sh["ndim"])
    layout = sh["layout"]
    cc = tuple(int(c) for c in payload["chunk"])

    try:
        groups = read_chunk_vertices(level_group, cc, dtype=sh["vertex_dtype"], ndim=ndim)
    except ArrayError:
        groups = []
    positions, fragment_of_row, parent = _chunk_rows(level_group, cc, groups, ndim)
    n_rows = len(positions)
    rows = np.arange(n_rows, dtype=np.int64)

    fragment_owner = np.full(len(groups), -1, dtype=np.int64)
    if payload["owners"] is not None:
        fragments, oids = payload["owners"]
        inside = fragments < len(groups)
        fragment_owner[fragments[inside]] = oids[inside]
    elif groups:
        stamped = read_chunk_fragment_attributes(
            level_group, "object_id", cc, dtype=np.uint64, default=None,
        )
        if stamped is None:
            raise ValueError(
                f"chunk {cc} at level {sh['level']} has no fragment_attributes/object_id "
                f"although the level's first chunk does"
            )
        stamped = np.asarray(stamped).reshape(-1).astype(np.int64)
        count = min(len(stamped), len(groups))
        fragment_owner[:count] = stamped[:count]
    if fragment_owner.size and int(fragment_owner.max()) >= int(sh["n_objects"]):
        raise ValueError(
            f"chunk {cc} names object {int(fragment_owner.max())}, but the level holds "
            f"{sh['n_objects']} objects"
        )
    owner = np.where(fragment_of_row >= 0, fragment_owner[np.maximum(fragment_of_row, 0)], -1)

    if sh["has_links"] and n_rows:
        records = _intra_records(level_group, cc)
        if len(records):
            _require_rows(records, n_rows, cc, "an intra-chunk link")
            # Both layouts store an intra record as [child, parent].  In the
            # chunked layout the child always starts a path, so this only adds
            # a parent; in the graph layout it replaces the implied one.
            parent[records[:, 0]] = records[:, 1]

    cross_end = np.zeros(n_rows, dtype=bool)
    cross_child = np.zeros(n_rows, dtype=bool)
    emitted: list[tuple[tuple[int, ...], np.ndarray, tuple[int, ...], np.ndarray]] = []
    for source, offset in payload["cells"]:
        source = tuple(int(c) for c in source)
        other = tuple(s + int(o) for s, o in zip(source, offset))
        rows_source, rows_other, source_first = _cross_cell(level_group, source, offset)
        here = rows_source if source == cc else rows_other
        _require_rows(here.reshape(-1, 1), n_rows, cc, "a cross-chunk link")
        cross_end[here] = True
        if layout == _LAYOUT_GRAPH:
            # write_graph files a cross-chunk edge as [child, parent]; the
            # child's parent is in another chunk, so the previous row is not.
            children = rows_source[source_first] if source == cc else rows_other[~source_first]
            parent[children] = -1
            cross_child[children] = True
        # Emitted once, by the lower chunk -- and, for a duplicate store, from
        # the copy that chunk is the source of.
        if cc == min(source, other) and (sh["link_store"] != "duplicate" or source == cc):
            emitted.append((source, rows_source, other, rows_other))

    owned = owner >= 0
    has_parent = parent >= 0
    same_object = np.zeros(n_rows, dtype=bool)
    same_object[has_parent] = owner[parent[has_parent]] == owner[has_parent]
    # An edge to a row of another object -- or to a row no requested object
    # owns -- is not part of anything being measured.
    parent[~(has_parent & owned & same_object)] = -1

    children = rows[parent >= 0]
    degree = (parent >= 0).astype(np.int64)
    degree += np.bincount(parent[children], minlength=n_rows)[:n_rows]

    owned_rows = rows[owned]
    objects, local = np.unique(owner[owned_rows], return_inverse=True)
    local = np.asarray(local, dtype=np.int64).reshape(-1)
    local_of_row = np.full(n_rows, -1, dtype=np.int64)
    local_of_row[owned_rows] = local
    edge_length = np.linalg.norm(positions[children] - positions[parent[children]], axis=1)
    cable = np.bincount(
        local_of_row[children], weights=edge_length, minlength=len(objects),
    )
    node_count = np.bincount(local, minlength=len(objects)).astype(np.int64)
    lower = np.full((len(objects), ndim), np.inf)
    upper = np.full((len(objects), ndim), -np.inf)
    if len(owned_rows):
        np.minimum.at(lower, local, positions[owned_rows])
        np.maximum.at(upper, local, positions[owned_rows])

    preferred = np.zeros(n_rows, dtype=bool)
    if layout == _LAYOUT_GRAPH:
        # write_graph keeps exactly one parentless row per object; a row made
        # parentless only because its parent lives in another chunk is not it.
        preferred = owned & (parent < 0) & ~cross_child
    key = owned & ((degree != 2) | cross_end | (parent < 0) | preferred)
    key_rows = rows[key]

    summary: dict[str, Any] = {
        "chunk": cc,
        "n_rows": n_rows,
        "objects": objects.astype(np.int64),
        "cable": cable,
        "node_count": node_count,
        "lower": lower,
        "upper": upper,
        "key_row": key_rows,
        "key_owner": owner[key_rows],
        "key_degree": degree[key_rows],
        "key_position": positions[key_rows],
        "key_preferred": preferred[key_rows],
        "cross": emitted,
    }
    if sh["topology"]:
        # Collapse every unbranched run: each key row's parent edge becomes an
        # edge to its nearest key ancestor.  Piece roots are key rows, so the
        # jumping below always lands on one.
        nearest_key = np.where(key | ~owned, rows, parent)
        nearest_key = _pointer_jump(nearest_key, what=f"chunk {cc}")
        piece_root = _pointer_jump(np.where(parent >= 0, parent, rows), what=f"chunk {cc}")
        key_children = key_rows[parent[key_rows] >= 0]
        summary["edge_child"] = key_children
        summary["edge_parent"] = nearest_key[parent[key_children]]
        summary["key_piece_root"] = piece_root[key_rows]
    return summary


def _chunk_rows(
    level_group: Any,
    cc: tuple[int, ...],
    groups: list[np.ndarray],
    ndim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A chunk's row positions, owning fragment per row, and implied parents.

    Link records address rows of the chunk's whole vertex buffer, not of one
    fragment, so rows are placed by the fragment index.  Every writer here
    tiles the buffer in fragment order, which is handled without a
    per-fragment loop; an index that does not tile falls back to one.
    """
    counts = np.fromiter((len(g) for g in groups), dtype=np.int64, count=len(groups))
    n_rows = int(counts.sum())
    if not groups:
        return (
            np.zeros((0, ndim), dtype=np.float64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
        )
    fragment_index = read_vertex_fragment_index(level_group, cc)
    if fragment_index.tiles(n_rows):
        positions = np.concatenate(groups, axis=0).astype(np.float64, copy=False)
        fragment_of_row = np.repeat(np.arange(len(groups), dtype=np.int64), counts)
        parent = np.arange(-1, n_rows - 1, dtype=np.int64)
        starts = np.cumsum(counts) - counts
        parent[starts[counts > 0]] = -1
        return positions.reshape(n_rows, ndim), fragment_of_row, parent

    n_rows = max(fragment_index.vertex_extent, n_rows)
    positions = np.zeros((n_rows, ndim), dtype=np.float64)
    fragment_of_row = np.full(n_rows, -1, dtype=np.int64)
    parent = np.full(n_rows, -1, dtype=np.int64)
    for fragment, block in enumerate(groups):
        if fragment_index.is_range(fragment):
            start, count = fragment_index.range(fragment)
            idx = np.arange(int(start), int(start) + int(count), dtype=np.int64)
        else:
            idx = np.asarray(fragment_index.indices(fragment), dtype=np.int64)
        if idx.size == 0:
            continue
        positions[idx] = np.asarray(block, dtype=np.float64).reshape(-1, ndim)
        fragment_of_row[idx] = fragment
        parent[idx[1:]] = idx[:-1]
    return positions, fragment_of_row, parent


def _intra_records(level_group: Any, cc: tuple[int, ...]) -> np.ndarray:
    """The chunk's intra-chunk link records as one ``(M, 2)`` array."""
    try:
        link_groups = read_chunk_links(level_group, cc)
    except ArrayError:
        return np.zeros((0, 2), dtype=np.int64)
    blocks = [
        np.asarray(g, dtype=np.int64).reshape(len(g), -1)[:, -2:]
        for g in link_groups if len(g)
    ]
    if not blocks:
        return np.zeros((0, 2), dtype=np.int64)
    records = np.concatenate(blocks, axis=0)
    # A negative endpoint is a "no parent" row, not an edge.
    return records[(records >= 0).all(axis=1)]


def _cross_cell(
    level_group: Any,
    source: tuple[int, ...],
    offset: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode one cross-chunk link cell of a two-endpoint family.

    Read with the per-cell reader rather than ``read_links_for_tuple``, which
    re-derives the level's scales for every call -- a few milliseconds a
    cell, and most of a whole-level pass.  The rows come back as placed:
    the source chunk's endpoint first, then the one at ``source + offset``,
    with a leading ``perm_idx`` column when the family canonically sorts.
    For two endpoints that index is a Lehmer code with two values, 0 for
    input order and 1 for swapped, which is decoded here in one comparison
    instead of one ``apply_perm_inverse`` call per record.

    Returns:
        ``(rows_source, rows_other, source_first)``: the chunk-local row of
        each record's endpoint in ``source`` and in the other chunk, and
        whether the source endpoint was the record's first (for a
        ``write_graph`` store, its child).
    """
    groups = read_chunk_links(level_group, source, offsets=(tuple(int(o) for o in offset),))
    blocks = [np.asarray(g, dtype=np.int64).reshape(len(g), -1) for g in groups if len(g)]
    if not blocks:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0, dtype=bool)
    rows = np.concatenate(blocks, axis=0)
    if rows.shape[1] == 3:
        perm = rows[:, 0]
        if perm.size and (int(perm.min()) < 0 or int(perm.max()) > 1):
            raise ValueError(
                f"cross-chunk cell {source} + {tuple(offset)} carries perm_idx "
                f"{int(perm.max())}, which no two-endpoint record can have"
            )
        return rows[:, 1], rows[:, 2], perm == 0
    return rows[:, 0], rows[:, 1], np.ones(len(rows), dtype=bool)


def _require_rows(records: np.ndarray, n_rows: int, cc: tuple[int, ...], what: str) -> None:
    """Refuse a link record that names a row the chunk does not hold."""
    if records.size and (int(records.max()) >= n_rows or int(records.min()) < 0):
        bad = int(records.max()) if int(records.max()) >= n_rows else int(records.min())
        raise ValueError(
            f"{what} names row {bad} of chunk {cc}, which holds {n_rows} rows"
        )


def _pointer_jump(pointer: np.ndarray, *, what: str) -> np.ndarray:
    """Follow ``pointer`` to its fixed points by repeated squaring.

    Converges in ``ceil(log2(longest chain))`` rounds on an acyclic pointer
    graph.  A stored parent structure that loops never converges, so the
    round count is capped and the loop reported rather than spun forever.
    """
    limit = max(1, int(len(pointer)).bit_length() + 1)
    for _ in range(limit + 1):
        nxt = pointer[pointer]
        if np.array_equal(nxt, pointer):
            return pointer
        pointer = nxt
    raise ValueError(f"{what}: the stored parent structure contains a cycle")


# ===================================================================
# Coordinator
# ===================================================================

def _assemble(
    summaries: list[dict],
    *,
    n_objects: int,
    ndim: int,
    topology: bool,
) -> dict[str, np.ndarray]:
    """Combine chunk summaries into one value array per metric."""
    summaries = sorted(summaries, key=lambda s: tuple(s["chunk"]))
    cable = np.zeros(n_objects, dtype=np.float64)
    node_count = np.zeros(n_objects, dtype=np.int64)
    lower = np.full((n_objects, ndim), np.inf)
    upper = np.full((n_objects, ndim), -np.inf)
    if summaries:
        objects = np.concatenate([s["objects"] for s in summaries])
        cable += np.bincount(
            objects, weights=np.concatenate([s["cable"] for s in summaries]),
            minlength=n_objects,
        )[:n_objects]
        node_count += np.bincount(
            objects, weights=np.concatenate([s["node_count"] for s in summaries]),
            minlength=n_objects,
        )[:n_objects].astype(np.int64)
        if len(objects):
            np.minimum.at(lower, objects, np.concatenate([s["lower"] for s in summaries]))
            np.maximum.at(upper, objects, np.concatenate([s["upper"] for s in summaries]))

    # --- key vertices, addressed by one global id ------------------------
    chunk_index = {tuple(s["chunk"]): i for i, s in enumerate(summaries)}
    key_counts = np.array([len(s["key_row"]) for s in summaries], dtype=np.int64)
    n_keys = int(key_counts.sum())
    stride = max((int(s["n_rows"]) for s in summaries), default=0) + 1
    key_chunk = np.repeat(np.arange(len(summaries), dtype=np.int64), key_counts)

    def _cat(field: str, dtype: Any, shape: tuple[int, ...] = ()) -> np.ndarray:
        if not summaries:
            return np.zeros((0, *shape), dtype=dtype)
        return np.concatenate([np.asarray(s[field], dtype=dtype) for s in summaries])

    key_row = _cat("key_row", np.int64)
    key_address = key_chunk * stride + key_row  # ascending: chunks sorted, rows sorted
    key_owner = _cat("key_owner", np.int64)
    key_degree = _cat("key_degree", np.int64)
    key_position = _cat("key_position", np.float64, (ndim,)).reshape(n_keys, ndim)
    key_preferred = _cat("key_preferred", bool)

    def _global_id(chunk: np.ndarray, row: np.ndarray) -> np.ndarray:
        address = chunk * stride + row
        at = np.searchsorted(key_address, address)
        found = at < n_keys
        found[found] = key_address[at[found]] == address[found]
        return np.where(found, at, -1)

    # --- cross-chunk links ------------------------------------------------
    link_parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for s in summaries:
        for chunk_a, rows_a, chunk_b, rows_b in s["cross"]:
            ia = chunk_index.get(tuple(chunk_a))
            ib = chunk_index.get(tuple(chunk_b))
            if ia is None or ib is None or not len(rows_a):
                # The other end lies in a chunk this pass did not read, which
                # only happens under object_ids -- and then belongs to an
                # object nobody asked about.
                continue
            link_parts.append((
                np.full(len(rows_a), ia, dtype=np.int64), np.asarray(rows_a, dtype=np.int64),
                np.full(len(rows_b), ib, dtype=np.int64), np.asarray(rows_b, dtype=np.int64),
            ))
    if link_parts:
        link_a = _global_id(
            np.concatenate([p[0] for p in link_parts]), np.concatenate([p[1] for p in link_parts]),
        )
        link_b = _global_id(
            np.concatenate([p[2] for p in link_parts]), np.concatenate([p[3] for p in link_parts]),
        )
    else:
        link_a = link_b = np.zeros(0, dtype=np.int64)
    keep = (link_a >= 0) & (link_b >= 0)
    keep[keep] = key_owner[link_a[keep]] == key_owner[link_b[keep]]
    link_a, link_b = link_a[keep], link_b[keep]
    coincident = np.all(key_position[link_a] == key_position[link_b], axis=1)
    link_length = np.linalg.norm(key_position[link_a] - key_position[link_b], axis=1)
    cable += np.bincount(key_owner[link_a], weights=link_length, minlength=n_objects)[:n_objects]

    # A zero-length link joins the two copies of a boundary vertex; merged,
    # they are one vertex whose degree is the sum of both copies'.
    merged = _hook_components(n_keys, link_a[coincident], link_b[coincident])[0]
    is_merged_rep = merged == np.arange(n_keys, dtype=np.int64)
    node_count -= np.bincount(key_owner[~is_merged_rep], minlength=n_objects)[:n_objects]
    degree = np.bincount(merged, weights=key_degree, minlength=n_keys).astype(np.int64)
    degree += np.bincount(merged[link_a[~coincident]], minlength=n_keys)
    degree += np.bincount(merged[link_b[~coincident]], minlength=n_keys)
    leaf = is_merged_rep & (degree <= 1)
    branch = is_merged_rep & (degree >= 3)

    extent = upper - lower
    extent[node_count == 0] = np.nan
    values: dict[str, np.ndarray] = {
        "cable_length": cable,
        "node_count": node_count,
        "leaf_count": np.bincount(key_owner[leaf], minlength=n_objects)[:n_objects],
        "branch_count": np.bincount(key_owner[branch], minlength=n_objects)[:n_objects],
        "extent": extent,
    }
    values["leaf_count"] = values["leaf_count"].astype(np.int64)
    values["branch_count"] = values["branch_count"].astype(np.int64)
    if not topology:
        return values

    # --- topology: components and Strahler order ---------------------------
    edge_counts = np.array([len(s["edge_child"]) for s in summaries], dtype=np.int64)
    edge_chunk = np.repeat(np.arange(len(summaries), dtype=np.int64), edge_counts)
    edge_child = _global_id(edge_chunk, _cat("edge_child", np.int64))
    edge_parent = _global_id(edge_chunk, _cat("edge_parent", np.int64))
    piece_root = _global_id(key_chunk, _cat("key_piece_root", np.int64))
    # Every piece is already a tree, so components start from the piece
    # labels and only the cross-chunk links are left to hook.  A link that
    # does not merge two components closes a cycle; it stays in the counts
    # above but is left out of the tree Strahler order is taken over.
    component, spanning = _hook_components(n_keys, link_a, link_b, label0=piece_root)

    values["component_count"] = np.zeros(n_objects, dtype=np.int64)
    values["max_strahler"] = np.zeros(n_objects, dtype=np.int64)
    if n_keys == 0:
        return values
    pairs = np.unique(np.stack([key_owner, component], axis=1), axis=0)
    values["component_count"] += np.bincount(pairs[:, 0], minlength=n_objects)[:n_objects]

    tree_rep = _hook_components(
        n_keys, link_a[coincident & spanning], link_b[coincident & spanning],
    )[0]
    nodes = np.flatnonzero(tree_rep == np.arange(n_keys, dtype=np.int64))
    compact = np.full(n_keys, -1, dtype=np.int64)
    compact[nodes] = np.arange(len(nodes), dtype=np.int64)
    joined = spanning & ~coincident
    u = compact[tree_rep[np.concatenate([edge_child, link_a[joined]])]]
    v = compact[tree_rep[np.concatenate([edge_parent, link_b[joined]])]]
    loops = u == v
    u, v = u[~loops], v[~loops]

    n_nodes = len(nodes)
    node_component = np.unique(component[nodes], return_inverse=True)[1].reshape(-1)
    node_preferred = np.bincount(
        compact[tree_rep], weights=key_preferred.astype(np.float64), minlength=n_nodes,
    ) > 0
    tree_degree = np.bincount(u, minlength=n_nodes) + np.bincount(v, minlength=n_nodes)
    root_class = np.where(node_preferred, 0, np.where(tree_degree <= 1, 1, 2))
    position = key_position[nodes]
    sort_keys = [nodes] + [position[:, a] for a in range(ndim - 1, -1, -1)] + [root_class]
    rank = np.empty(n_nodes, dtype=np.int64)
    rank[np.lexsort(sort_keys)] = np.arange(n_nodes, dtype=np.int64)

    forest = _euler_forest(n_nodes, u, v, node_component, rank)
    strahler = _strahler(forest["parent"], forest["tin"], forest["tout"], forest["slots"])
    roots = forest["roots"]
    np.maximum.at(values["max_strahler"], key_owner[nodes[roots]], strahler[roots])
    return values


# ===================================================================
# Forest engine
# ===================================================================

def _hook_components(
    n: int,
    u: np.ndarray,
    v: np.ndarray,
    *,
    label0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Connected components by root hooking, plus the edges that did the joining.

    Each round every component root touched by an edge to a smaller root
    hooks onto the smallest such root, then every label is shortcut to its
    root.  Labels only ever point to smaller labels, so the hooks form a
    forest, and one hooking edge per merge is a spanning forest of the
    input: an edge never used for a hook lies inside a component already,
    i.e. closes a cycle.

    Args:
        n: Node count.
        u, v: Edge endpoints.
        label0: Optional starting labels, for nodes already known to share a
            component (each piece of a chunk).  Must point at nodes whose own
            label is themselves.

    Returns:
        ``(label, spanning)``: a component label per node and, per edge,
        whether it was one of the spanning forest's joining edges.
    """
    u = np.asarray(u, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    if label0 is None:
        label = np.arange(n, dtype=np.int64)
    else:
        label = _pointer_jump(np.asarray(label0, dtype=np.int64).copy(), what="component labels")
    spanning = np.zeros(len(u), dtype=bool)
    while len(u):
        label_u = label[u]
        label_v = label[v]
        live = np.flatnonzero(label_u != label_v)
        if live.size == 0:
            break
        high = np.maximum(label_u[live], label_v[live])
        low = np.minimum(label_u[live], label_v[live])
        order = np.lexsort((low, high))
        first = np.ones(len(order), dtype=bool)
        first[1:] = high[order][1:] != high[order][:-1]
        winners = order[first]
        label[high[winners]] = low[winners]
        spanning[live[winners]] = True
        label = _pointer_jump(label, what="component labels")
    return label, spanning


def _euler_forest(
    n: int,
    u: np.ndarray,
    v: np.ndarray,
    component: np.ndarray,
    rank: np.ndarray,
    *,
    with_depth: bool = False,
) -> dict[str, Any]:
    """Root every tree of a forest at its lowest-rank node and number its Euler tour.

    The tour visits each undirected edge twice.  Listing a node's arcs in
    sorted order, the arc after ``u -> v`` is the one leaving ``v`` just
    after ``v -> u``; cut at the root, that successor chain is the tour, and
    :func:`_rank_lists` numbers it.  The arc into a node that comes before
    the arc back out makes the tail its parent, and every descendant's arcs
    lie between the two -- which is what lets a subtree query be one
    prefix-sum difference.

    Args:
        n: Node count.
        u, v: Edge endpoints.  Must form a forest without repeated edges.
        component: ``(n,)`` dense component ids ``0..C-1``.
        rank: ``(n,)`` root priority; the lowest rank in a component roots it.
        with_depth: Also fill ``depth_delta``.

    Returns:
        Dict with ``parent`` (``-1`` at roots), ``tin`` / ``tout`` (a node's
        subtree occupies slots ``[tin, tout)``), ``slots`` (slot count),
        ``roots`` (one per component, in component order), ``depth_delta``
        (``+1`` / ``-1`` per slot, for depth by prefix sum; zeros unless
        ``with_depth``).
    """
    u = np.asarray(u, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    component = np.asarray(component, dtype=np.int64)
    n_components = int(component.max()) + 1 if n else 0

    by_component = np.lexsort((np.asarray(rank), component))
    first = np.ones(n, dtype=bool)
    first[1:] = component[by_component][1:] != component[by_component][:-1]
    roots = by_component[first]

    size = np.bincount(component, minlength=n_components)
    tour_length = 2 * (size - 1)
    offset = np.zeros(n_components, dtype=np.int64)
    offset[1:] = np.cumsum(tour_length + 1)[:-1]
    slots = int((tour_length + 1).sum())

    n_edges = len(u)
    tail = np.concatenate([u, v])
    head = np.concatenate([v, u])
    order = np.argsort(tail * n + head, kind="stable")
    tail, head = tail[order], head[order]
    degree = np.bincount(tail, minlength=n)
    start = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(degree, out=start[1:])
    # Arc ``e`` and arc ``e + n_edges`` are the two directions of edge ``e``,
    # so each arc's reverse is found through the sort permutation rather than
    # by searching for it -- a binary search per arc was most of the runtime.
    sorted_at = np.empty_like(order)
    sorted_at[order] = np.arange(len(order), dtype=np.int64)
    reverse = sorted_at[(order + n_edges) % max(2 * n_edges, 1)]
    successor = start[head] + (reverse - start[head] + 1) % np.maximum(degree[head], 1)

    tin = np.empty(n, dtype=np.int64)
    tout = np.empty(n, dtype=np.int64)
    root_component = component[roots]
    tin[roots] = offset[root_component]
    tout[roots] = offset[root_component] + tour_length[root_component] + 1
    parent = np.full(n, -1, dtype=np.int64)
    depth_delta = np.zeros(slots if with_depth else 0, dtype=np.int64)
    if len(tail) == 0:
        return {
            "parent": parent, "tin": tin, "tout": tout, "slots": slots,
            "roots": roots, "depth_delta": depth_delta,
        }

    # Cut each tour where it returns to its root: the arc into the root from
    # the root's last neighbour.
    with_arcs = roots[degree[roots] > 0]
    successor[reverse[start[with_arcs] + degree[with_arcs] - 1]] = -1

    distance = _rank_lists(successor)
    arcs = np.arange(len(tail), dtype=np.int64)
    arc_component = component[tail]
    position = tour_length[arc_component] - 1 - distance
    down = position < position[reverse]
    down_arcs = arcs[down]
    child = head[down_arcs]
    parent[child] = tail[down_arcs]
    tin[child] = offset[arc_component[down_arcs]] + 1 + position[down_arcs]
    tout[child] = offset[arc_component[down_arcs]] + 1 + position[reverse[down_arcs]]
    if with_depth:
        depth_delta[offset[arc_component] + 1 + position] = np.where(down, 1, -1)
    return {
        "parent": parent, "tin": tin, "tout": tout, "slots": slots,
        "roots": roots, "depth_delta": depth_delta,
    }


# Below this many elements plain pointer jumping is cheaper than laying out a
# ruling set; the constant only moves runtime, never the result.
_RULING_SET_MIN = 1 << 14
_RULER_SPACING = 32


def _rank_lists(successor: np.ndarray) -> np.ndarray:
    """Distance from every element of a set of linked lists to its list's end.

    Pointer jumping alone costs ``log2(length)`` passes over every element,
    each a random gather -- about a second for a 2M-arc tour.  A ruling set
    does the same in roughly two passes: every ``_RULER_SPACING``-th element
    (drawn at random, so no tour structure can line the rulers up) walks
    forward to the next ruler with all walkers advancing in lockstep, which
    labels every element with its ruler and offset in one visit each; only
    the short list of rulers is then pointer-jumped.

    Args:
        successor: ``(A,)`` next element, ``-1`` at a list end.  Must be a
            union of simple lists (no cycles).

    Returns:
        ``(A,)`` int64 distance to the end; a list's last element is 0.
    """
    successor = np.asarray(successor, dtype=np.int64)
    n = len(successor)
    if n < _RULING_SET_MIN:
        is_last = successor < 0
        jump = np.where(is_last, np.arange(n, dtype=np.int64), successor)
        distance = (~is_last).astype(np.int64)
        while n and not is_last[jump].all():
            distance = distance + distance[jump]
            jump = jump[jump]
        return distance

    has_predecessor = np.zeros(n, dtype=bool)
    has_predecessor[successor[successor >= 0]] = True
    is_ruler = np.random.default_rng(0).random(n) < 1.0 / _RULER_SPACING
    is_ruler[~has_predecessor] = True  # every list head starts a walk
    rulers = np.flatnonzero(is_ruler)
    n_rulers = len(rulers)
    ruler_id = np.full(n, -1, dtype=np.int64)
    ruler_id[rulers] = np.arange(n_rulers, dtype=np.int64)

    owner = np.empty(n, dtype=np.int64)
    offset = np.zeros(n, dtype=np.int64)
    owner[rulers] = np.arange(n_rulers, dtype=np.int64)
    next_ruler = np.full(n_rulers, -1, dtype=np.int64)
    gap = np.zeros(n_rulers, dtype=np.int64)

    walker = np.arange(n_rulers, dtype=np.int64)
    current = successor[rulers]
    step = 1
    while walker.size:
        ended = current < 0
        reached = np.zeros(len(current), dtype=bool)
        reached[~ended] = is_ruler[current[~ended]]
        stopped = ended | reached
        if stopped.any():
            gap[walker[stopped]] = step
            hit = stopped & reached
            next_ruler[walker[hit]] = ruler_id[current[hit]]
            walker, current = walker[~stopped], current[~stopped]
            if not walker.size:
                break
        owner[current] = walker
        offset[current] = step
        current = successor[current]
        step += 1

    # A ruler is ``gap`` elements before the next ruler, or ``gap - 1`` before
    # its list's last element when the walk ran off the end.
    last_ruler = next_ruler < 0
    jump = np.where(last_ruler, np.arange(n_rulers, dtype=np.int64), next_ruler)
    ruler_distance = np.where(last_ruler, 0, gap)
    while not last_ruler[jump].all():
        ruler_distance = ruler_distance + ruler_distance[jump]
        jump = jump[jump]
    ruler_distance = ruler_distance + gap[jump] - 1
    return ruler_distance[owner] - offset


def _strahler(parent: np.ndarray, tin: np.ndarray, tout: np.ndarray, slots: int) -> np.ndarray:
    """Strahler order of every node of a rooted forest, one pass per order.

    Order ``k + 1`` is reached exactly where some node of the subtree has two
    children of order at least ``k``: that node rises to ``k + 1`` and order
    never falls towards the root.  So each pass marks the nodes with two
    such children and lifts every node whose ``[tin, tout)`` interval holds a
    mark.  Passes equal the highest order -- ``log2`` of the node count at
    most -- rather than the tree's depth.
    """
    n = len(parent)
    order = np.ones(n, dtype=np.int64)
    reached = np.ones(n, dtype=bool)
    has_parent = parent >= 0
    k = 1
    while True:
        counts = np.bincount(parent[reached & has_parent], minlength=n)
        forks = np.flatnonzero(counts[:n] >= 2)
        if forks.size == 0:
            return order
        marks = np.zeros(slots + 1, dtype=np.int64)
        marks[tin[forks] + 1] = 1
        prefix = np.cumsum(marks)
        reached = (prefix[tout] - prefix[tin]) > 0
        k += 1
        order[reached] = k


def compute_tree_metrics_vectorized(
    parents: npt.ArrayLike,
    root_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Topological depth, Strahler order and node kind, without a per-node loop.

    A drop-in for ``convert.ingest._tree_enrichments.compute_tree_metrics``
    (same arguments, dtypes and codes), which walks child lists in Python and
    costs about 0.2 s per 100k nodes.  For a parents array that encodes a
    single tree rooted at ``root_idx`` the two agree exactly; nodes outside
    ``root_idx``'s tree get depth and Strahler 0 in both.

    Args:
        parents: ``(N,)`` parent index per node, ``-1`` (or the node itself)
            at a root.
        root_idx: Root of the tree to measure.

    Returns:
        ``(topological_depth uint16, strahler uint8, node_kind uint8)``.

    Raises:
        ValueError: A parent index is out of range, or the parents loop.
    """
    parents = np.asarray(parents, dtype=np.int64).reshape(-1)
    n = len(parents)
    if n == 0:
        return (
            np.zeros((0,), dtype=np.uint16),
            np.zeros((0,), dtype=np.uint8),
            np.zeros((0,), dtype=np.uint8),
        )
    if int(parents.max()) >= n:
        raise ValueError(f"parents names node {int(parents.max())}, but there are {n} nodes")
    nodes = np.arange(n, dtype=np.int64)
    has_parent = (parents >= 0) & (parents != nodes)
    child = nodes[has_parent]
    parent_of_child = parents[has_parent]

    tree_root = _pointer_jump(np.where(has_parent, parents, nodes), what="parents")
    component = np.unique(tree_root, return_inverse=True)[1].reshape(-1)
    rank = nodes + 1
    rank[int(root_idx)] = 0
    forest = _euler_forest(n, child, parent_of_child, component, rank, with_depth=True)
    strahler = _strahler(forest["parent"], forest["tin"], forest["tout"], forest["slots"])
    # Each down arc adds one level and each up arc removes it, so a node's
    # depth is the running sum up to the slot where the tour first enters it.
    depth = np.cumsum(forest["depth_delta"])[forest["tin"]]

    outside = component != component[int(root_idx)]
    depth[outside] = 0
    strahler[outside] = 0

    n_children = np.bincount(parent_of_child, minlength=n)
    node_kind = np.full(n, _CONTINUATION, dtype=np.uint8)
    node_kind[n_children == 0] = _TERMINAL
    node_kind[n_children >= 2] = _BRANCH
    node_kind[int(root_idx)] = _SOMA
    return depth.astype(np.uint16), strahler.astype(np.uint8), node_kind
