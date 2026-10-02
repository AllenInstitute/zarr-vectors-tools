"""A graph level's edges as arrays, and the array kernels the graph algorithms share.

The graph algorithms used to build a Python adjacency list, one tuple per
edge end, from :func:`read_links`' per-record tuples, and then walk it one
vertex at a time.  At ten million edges both halves take hours.  Here the
link family is read as arrays (``read_link_arrays``), each endpoint's
``(chunk, local index)`` becomes a global vertex index in one call
(``link_endpoints_to_rows``), and the edges are held as a compressed sparse
row adjacency: ``indptr``, ``indices``, ``weights``.  A
``store="duplicate"`` family's records come once per copy, as they do from
``read_links``.

A skeleton store (``links_convention =
"implicit_sequential_with_branches"``) does not store most of its edges:
consecutive vertices of a fragment are joined implicitly, and the stored
records only add to or override those.  :func:`read_edges` rebuilds the full
edge set the way core's ``read_graph`` does (see :func:`_skeleton_edges`).

The kernels here need only numpy.

Global vertex order is ``chunk_local_to_global_offsets``': chunks in their
sorted order, vertices in chunk order -- the order ``read_graph`` returns
positions in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Adjacency:
    """Both directions of every edge, grouped by source vertex.

    ``indices[indptr[v]:indptr[v + 1]]`` are ``v``'s neighbours and
    ``weights`` the matching edge weights.  A parallel edge appears once per
    copy and a self-loop twice, as in the adjacency lists this replaces, so
    degrees and weight sums are unchanged.
    """

    n: int
    indptr: npt.NDArray[np.int64]
    indices: npt.NDArray[np.int64]
    weights: npt.NDArray[np.float64]

    @property
    def degree(self) -> npt.NDArray[np.int64]:
        return np.diff(self.indptr)

    def sources(self) -> npt.NDArray[np.int64]:
        """The source vertex of each entry of ``indices``."""
        return np.repeat(np.arange(self.n, dtype=np.int64), self.degree)

    def gather(self, vertices: npt.NDArray[np.int64]) -> tuple[npt.NDArray[np.int64], ...]:
        """``(positions, owner)``: the entries of every vertex in ``vertices``.

        ``owner[i]`` is the vertex of ``vertices`` whose row ``positions[i]``
        belongs to, so ``indices[positions]`` are their neighbours.
        """
        starts = self.indptr[vertices]
        counts = self.indptr[vertices + 1] - starts
        total = int(counts.sum())
        if total == 0:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty
        run_start = np.repeat(np.cumsum(counts) - counts, counts)
        positions = np.repeat(starts, counts) + (np.arange(total, dtype=np.int64) - run_start)
        return positions, np.repeat(vertices, counts)

    def to_lists(self) -> list[list[tuple[int, float]]]:
        """The adjacency as ``adj[v] = [(neighbour, weight), ...]``."""
        neighbours = self.indices.tolist()
        weights = self.weights.tolist()
        bounds = self.indptr.tolist()
        return [
            list(zip(neighbours[bounds[v]:bounds[v + 1]], weights[bounds[v]:bounds[v + 1]]))
            for v in range(self.n)
        ]


def read_edges(
    level_group: Any, *, weight_attr: str | None = None,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64] | None, int]:
    """``(a, b, weights, n_vertices)`` for every edge of the level at ``delta=0``.

    ``a`` and ``b`` are global vertex indices.  On a graph store every edge
    is a stored record and they come in ``read_links`` record order.  On a
    skeleton store the edges the links convention implies are added, as
    :func:`_skeleton_edges` describes.  ``weights`` is the ``weight_attr``
    link attribute in the same order, or ``None`` without one, or when its
    length disagrees with the records (a partial or stale write, whose rows
    cannot be trusted).

    Raises:
        ValueError: If the links are not two-endpoint records, an endpoint
            names a chunk the level does not list, or ``weight_attr`` is
            stored on a skeleton store, whose implied edges carry no value
            for it.
    """
    from zarr_vectors.building import (
        chunk_local_to_global_offsets,
        link_endpoints_to_rows,
        link_family_policy,
        read_link_arrays,
        read_link_attributes,
    )
    from zarr_vectors.constants import VERTEX_FRAGMENTS

    from zarr_vectors_tools.algorithms._links import chunk_key_str, link_prefetch_plan

    offsets, chunk_keys, n_vertices = chunk_local_to_global_offsets(level_group)
    tree = _implies_parents(level_group)
    attrs = (weight_attr,) if weight_attr is not None else ()
    plan = link_prefetch_plan(level_group, chunk_keys, attrs=attrs)
    if tree:
        plan.append((VERTEX_FRAGMENTS, [chunk_key_str(cc) for cc in chunk_keys]))
    with level_group.batched_reads(plan):
        chunks, vi = read_link_arrays(level_group, delta=0)
        weights: npt.NDArray[np.float64] | None = None
        if weight_attr is not None and len(vi):
            try:
                weights = np.asarray(
                    read_link_attributes(level_group, weight_attr, delta=0), dtype=np.float64,
                ).reshape(-1)
            except Exception:  # noqa: BLE001 - no such attribute: unit weights
                weights = None
            if weights is not None and len(weights) != len(vi):
                weights = None
        implied = (
            _implied_parents(level_group, chunk_keys, offsets, n_vertices) if tree else None
        )

    if vi.shape[0] == 0:
        rows = np.zeros((0, 2), dtype=np.int64)
    else:
        if vi.shape[1] != 2:
            raise ValueError(
                f"the graph algorithms read two-endpoint links; this level's links "
                f"have {vi.shape[1]} endpoints"
            )
        if tree:
            # A negative endpoint is a "no parent" row, not an edge.
            stored = (vi >= 0).all(axis=1)
            chunks, vi = chunks[stored], vi[stored]
            if weights is not None:
                weights = weights[stored]
        rows = np.asarray(link_endpoints_to_rows(chunks, vi, offsets), dtype=np.int64)
        if (rows < 0).any():
            # Raised, not dropped: the global offsets are cumulative over the
            # chunks the presence manifest lists, so a chunk missing from it
            # shifts every later chunk's vertex numbers too, and the surviving
            # edges would silently point at the wrong vertices.
            raise ValueError(
                f"{int((rows < 0).sum())} link endpoint(s) name a chunk the level's "
                f"vertices presence manifest does not list, so global vertex numbers "
                f"cannot be trusted; run zarr_vectors.building.rebuild_presence on the "
                f"level (a parallel writer that skipped it leaves this state)"
            )
    if implied is None:
        return rows[:, 0].copy(), rows[:, 1].copy(), weights, int(n_vertices)

    policy = link_family_policy(level_group, 0)
    directed = bool(policy[2]) if policy is not None else False
    spans = (chunks[:, 0] != chunks[:, 1]).any(axis=1) if len(rows) else np.zeros(0, bool)
    a, b, record = _skeleton_edges(implied, rows, spans, directed=directed)
    if weights is not None:
        if (record < 0).any():
            raise ValueError(
                f"link attribute {weight_attr!r} is stored only for this skeleton "
                f"level's explicit links; the {int((record < 0).sum())} parent links "
                f"its links convention implies have no value for it"
            )
        weights = weights[record]
    return a, b, weights, int(n_vertices)


def _implies_parents(level_group: Any) -> bool:
    """Whether the store's links convention implies parent links.

    Read from the store root, where core keeps ``links_convention``; the
    level carries no copy.  ``implicit_sequential_with_branches`` is the one
    convention core's ``read_graph`` reconstructs edges for.
    """
    import zarr
    from zarr_vectors.constants import LINKS_IMPLICIT_BRANCHES

    group = level_group.zarr_group
    parent_path = group.path.rsplit("/", 1)[0] if "/" in group.path else ""
    try:
        root = zarr.open_group(store=group.store, path=parent_path, mode="r")
        convention = (root.attrs.get("zarr_vectors") or {}).get("links_convention")
    except Exception:  # noqa: BLE001 - a bare level: its links are all stored
        return False
    return convention == LINKS_IMPLICIT_BRANCHES


def _implied_parents(
    level_group: Any, chunk_keys: list, offsets: dict, n_vertices: int,
) -> npt.NDArray[np.int64]:
    """Each vertex's implied parent, as a global index, or -1.

    The implicit-sequential rule: a row's parent is the row before it in the
    same fragment, and a fragment's first row has none -- the row before it
    belongs to another fragment.  Rows are placed by the fragment index, so
    an index that does not tile the chunk is honoured too.
    """
    from zarr_vectors.building import read_vertex_fragment_index

    parent = np.full(n_vertices, -1, dtype=np.int64)
    for i, cc in enumerate(chunk_keys):
        base = int(offsets[cc])
        n_rows = (int(offsets[chunk_keys[i + 1]]) if i + 1 < len(chunk_keys) else n_vertices) - base
        if n_rows <= 0:
            continue
        try:
            fragment_index = read_vertex_fragment_index(level_group, cc)
        except Exception:  # noqa: BLE001 - no fragment index: no implied links
            continue
        ranges = fragment_index.ranges()
        if ranges is not None:
            starts, counts = ranges[:, 0].astype(np.int64), ranges[:, 1].astype(np.int64)
            # Every row after a fragment's first, in one pass.
            follow = np.maximum(counts - 1, 0)
            run_start = np.repeat(np.cumsum(follow) - follow, follow)
            child = np.repeat(starts, follow) + 1 + (
                np.arange(int(follow.sum()), dtype=np.int64) - run_start
            )
            prev = child - 1
        else:
            parts = [
                np.asarray(fragment_index.indices(f), dtype=np.int64)
                if not fragment_index.is_range(f)
                else np.arange(*_range_bounds(fragment_index.range(f)), dtype=np.int64)
                for f in range(fragment_index.num_fragments)
            ]
            child = np.concatenate([p[1:] for p in parts]) if parts else np.zeros(0, np.int64)
            prev = np.concatenate([p[:-1] for p in parts]) if parts else np.zeros(0, np.int64)
        inside = (child < n_rows) & (prev >= 0) & (prev < n_rows)
        parent[base + child[inside]] = base + prev[inside]
    return parent


def _range_bounds(start_count: tuple[int, int]) -> tuple[int, int]:
    start, count = start_count
    return int(start), int(start) + int(count)


def _skeleton_edges(
    implied: npt.NDArray[np.int64],
    rows: npt.NDArray[np.int64],
    spans: npt.NDArray[np.bool_],
    *,
    directed: bool,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """``(a, b, record)``: a skeleton level's edges, implied and stored.

    ``rows`` are the stored records as global ``[child, parent]`` pairs and
    ``spans`` marks those whose endpoints lie in different chunks.  The two
    skeleton layouts are told apart by the links family's ``directed`` flag:

    * written by ``write_graph`` (SWC ingest; undirected family): every
      record, cross-chunk ones included, replaces its child's implied
      parent -- the rule core's ``read_graph`` applies;
    * written chunk by chunk (precomputed ingest and the skeleton coarsener;
      directed family): an intra-chunk record attaches a path's first row,
      which has no implied parent, so it only adds one; a cross-chunk record
      joins two paths' ends and is added as an edge of its own, re-parenting
      nothing.

    ``record[i]`` is the stored record edge ``i`` came from, or -1 for an
    implied edge.  Each vertex's parent edge comes first, ordered by child.
    """
    parent = implied.copy()
    record_of = np.full(len(parent), -1, dtype=np.int64)
    overrides = ~spans if directed else np.ones(len(rows), dtype=bool)
    which = np.flatnonzero(overrides)
    parent[rows[which, 0]] = rows[which, 1]
    record_of[rows[which, 0]] = which

    child = np.flatnonzero(parent >= 0)
    extra = np.flatnonzero(~overrides)
    a = np.concatenate([child, rows[extra, 0]])
    b = np.concatenate([parent[child], rows[extra, 1]])
    return a, b, np.concatenate([record_of[child], extra])


def build_csr(
    n: int,
    a: npt.NDArray[np.int64],
    b: npt.NDArray[np.int64],
    weights: npt.NDArray[np.float64] | None = None,
) -> Adjacency:
    """The undirected adjacency of edges ``a[i] -- b[i]``."""
    src = np.concatenate([a, b])
    dst = np.concatenate([b, a])
    w = (
        np.ones(len(src), dtype=np.float64) if weights is None
        else np.concatenate([weights, weights]).astype(np.float64, copy=False)
    )
    order = np.argsort(src, kind="stable")
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=indptr[1:])
    return Adjacency(n, indptr, dst[order], w[order])


def read_adjacency(level_group: Any, *, weight_attr: str | None = None) -> Adjacency:
    a, b, weights, n = read_edges(level_group, weight_attr=weight_attr)
    return build_csr(n, a, b, weights)


# ===================================================================
# Kernels
# ===================================================================

def component_roots(
    n: int, a: npt.NDArray[np.int64], b: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """The smallest vertex of each vertex's connected component.

    Hook-and-jump union-find over the whole edge list at once: every round
    points the larger root of each edge at the smaller, then shortcuts every
    vertex to its root.  Roots only decrease, so it stops when a round
    changes nothing -- in a number of rounds that grows with the logarithm
    of the component diameter in practice, each one a pass over the edges.
    """
    parent = np.arange(n, dtype=np.int64)
    if len(a) == 0:
        return parent
    while True:
        ra, rb = parent[a], parent[b]
        differ = ra != rb
        if not differ.any():
            return parent
        lo = np.minimum(ra[differ], rb[differ])
        hi = np.maximum(ra[differ], rb[differ])
        np.minimum.at(parent, hi, lo)
        _compress(parent)
        # Edges whose ends now share a root never need looking at again.
        keep = parent[a] != parent[b]
        a, b = a[keep], b[keep]


def _compress(parent: npt.NDArray[np.int64]) -> None:
    """Point every vertex at its root, in place."""
    while True:
        grand = parent[parent]
        if np.array_equal(grand, parent):
            return
        parent[:] = grand


def bfs_levels(
    adjacency: Adjacency, source: int, max_distance: int | None = None,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int64]]:
    """Hop distances and a shortest-path parent from ``source``, a frontier at a time.

    Each vertex's parent is its lowest-numbered neighbour one hop nearer
    the source.  Vertices beyond ``max_distance`` stay ``-1``.
    """
    n = adjacency.n
    distances = np.full(n, -1, dtype=np.int32)
    predecessors = np.full(n, -1, dtype=np.int64)
    distances[source] = 0
    frontier = np.array([source], dtype=np.int64)
    depth = 0
    while frontier.size and (max_distance is None or depth < max_distance):
        positions, owner = adjacency.gather(frontier)
        neighbours = adjacency.indices[positions]
        fresh = distances[neighbours] == -1
        neighbours, owner = neighbours[fresh], owner[fresh]
        if not len(neighbours):
            break
        order = np.lexsort((owner, neighbours))
        neighbours, owner = neighbours[order], owner[order]
        first = np.ones(len(neighbours), dtype=bool)
        first[1:] = neighbours[1:] != neighbours[:-1]
        frontier = neighbours[first]
        depth += 1
        distances[frontier] = depth
        predecessors[frontier] = owner[first]
    return distances, predecessors


def k_core(adjacency: Adjacency) -> npt.NDArray[np.int64]:
    """Per-vertex coreness by degree peeling, a batch of vertices at a time.

    At each ``k`` every live vertex of degree at most ``k`` is removed at
    once and its live neighbours' degrees drop; the vertices that drop to
    ``k`` or below are the next batch.  Coreness is unique, so batching
    changes nothing about the answer: a vertex removed at ``k`` would have
    been removed at ``k`` one at a time too.
    """
    n = adjacency.n
    degree = adjacency.degree.astype(np.int64).copy()
    alive = np.ones(n, dtype=bool)
    coreness = np.zeros(n, dtype=np.int64)
    remaining = n
    k = 0
    while remaining:
        live = np.flatnonzero(alive)
        k = max(k, int(degree[live].min()))
        batch = live[degree[live] <= k]
        while batch.size:
            coreness[batch] = k
            alive[batch] = False
            remaining -= len(batch)
            positions, _owner = adjacency.gather(batch)
            touched = adjacency.indices[positions]
            touched = touched[alive[touched]]
            if not touched.size:
                break
            np.subtract.at(degree, touched, 1)
            candidates = np.unique(touched)
            batch = candidates[degree[candidates] <= k]
    return coreness


def label_propagation_round(
    adjacency: Adjacency,
    labels: npt.NDArray[np.int64],
    sources: npt.NDArray[np.int64],
    rng: np.random.Generator,
) -> npt.NDArray[np.int64]:
    """One synchronous round: each vertex takes its neighbours' commonest label.

    Ties are broken at random, seeded, among the tied labels.  A vertex with
    no neighbours keeps its label.
    """
    if not len(adjacency.indices):
        return labels.copy()
    neighbour_labels = labels[adjacency.indices]
    # Count (vertex, label) pairs: sort by vertex, then label.
    order = np.lexsort((neighbour_labels, sources))
    v, lab = sources[order], neighbour_labels[order]
    start = np.ones(len(v), dtype=bool)
    start[1:] = (v[1:] != v[:-1]) | (lab[1:] != lab[:-1])
    pair_start = np.flatnonzero(start)
    counts = np.diff(np.append(pair_start, len(v)))
    pair_vertex, pair_label = v[pair_start], lab[pair_start]
    # Best pair per vertex: most neighbours, then a random draw among ties.
    draw = rng.random(len(pair_start))
    best = np.lexsort((draw, counts, pair_vertex))
    last = np.ones(len(best), dtype=bool)
    last[:-1] = pair_vertex[best][1:] != pair_vertex[best][:-1]
    winners = best[last]
    out = labels.copy()
    out[pair_vertex[winners]] = pair_label[winners]
    return out


def compact_labels(
    labels: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.int64]]:
    """Labels renumbered 0.. in order of each label's lowest vertex, and their sizes."""
    if not len(labels):
        return np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.int64)
    unique, first, inverse = np.unique(labels, return_index=True, return_inverse=True)
    rank = np.empty(len(unique), dtype=np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(len(unique))
    compact = rank[inverse.reshape(-1)]
    return compact.astype(np.uint32), np.bincount(compact, minlength=len(unique)).astype(np.int64)


def modularity(adjacency: Adjacency, labels: npt.NDArray[np.int64]) -> float:
    """Q = (1/2m) Σ_ij [A_ij - k_i k_j / 2m] δ(c_i, c_j)."""
    if adjacency.n == 0:
        return 0.0
    strength = np.bincount(adjacency.sources(), weights=adjacency.weights, minlength=adjacency.n)
    two_m = float(strength.sum())
    if two_m == 0:
        return 0.0
    same = labels[adjacency.sources()] == labels[adjacency.indices]
    intra = float(adjacency.weights[same].sum())
    community = np.bincount(labels, weights=strength)
    return intra / two_m - float((community ** 2).sum()) / (two_m * two_m)


def contract(adjacency: Adjacency, labels: npt.NDArray[np.int64], new_n: int) -> Adjacency:
    """Communities as super-vertices; the weights between them summed.

    Each direction of every edge is already an entry, so the super-graph's
    entries are the sums over entries and need no mirroring.  A community's
    internal weight becomes a self-loop entry holding both directions.
    """
    src = labels[adjacency.sources()]
    dst = labels[adjacency.indices]
    key = src * new_n + dst
    unique, inverse = np.unique(key, return_inverse=True)
    weights = np.bincount(inverse.reshape(-1), weights=adjacency.weights)
    s, d = unique // new_n, unique % new_n
    indptr = np.zeros(new_n + 1, dtype=np.int64)
    np.cumsum(np.bincount(s, minlength=new_n), out=indptr[1:])
    return Adjacency(new_n, indptr, d.astype(np.int64), weights.astype(np.float64))
