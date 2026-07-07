"""ETL: precomputed spatial-index ``.frags`` skeletons → multiscale
zarr-vectors store.

Source: a precomputed skeleton layer whose ``info`` carries a
``spatial_index`` block.  Per spatial chunk, a ``<bbox>.frags`` file is a
seung-lab **MapBuffer** keyed by uint64 segment id; each value is a
precomputed skeleton *piece* (vertices in nm, edges, per-vertex
attributes such as ``radius`` / ``cross_sectional_area``).

Pipeline (all per-chunk, bounded RAM):

1. **Extract** one ``.frags`` chunk → ``{segment_id: piece}``
   (:class:`PrecomputedFragsReader`, cloud-volume + mapbuffer).
2. **Partition + write** each segment's geometry into origin-aligned
   zarr chunks, dropping cross-chunk edges, one fragment per connected
   piece (:func:`pieces_from_chunk` → ``write_skeleton_chunk``).  Emits
   ``(segment_id, chunk, fragment)`` records.
3. **Reduce** the records into a segment-id-preserving object index
   (``build_object_index``).
4. **Pyramid**: skeleton-aware path-simplification levels
   (``build_skeleton_pyramid``).

The reader is dependency-injected: :func:`run_ingest` accepts any object
exposing ``info`` and ``read_chunk(key)``, so the whole pipeline is
testable offline with :class:`InMemoryFragsReader`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import numpy.typing as npt

from zarr_vectors.types import skeletons as sk
from zarr_vectors.typing import ChunkCoords
from zarr_vectors_tools.multiresolution.object_index import build_object_index
from zarr_vectors_tools.multiresolution.skeleton_graph import split_components
from zarr_vectors_tools.multiresolution.strategies.skeletons import (
    build_skeleton_pyramid,
    coarsen_skeleton_level,
)


# Per-worker cache for lazily constructed external readers used by dask tasks.
# Keyed by (base_url, frags_dir).
_WORKER_READER_CACHE: dict[tuple[str, str], Any] = {}


# ===================================================================
# Source description
# ===================================================================

@dataclass
class SkeletonInfo:
    """Spatial-index geometry of a precomputed skeleton layer."""
    base_url: str
    resolution_nm: tuple[float, ...]       # nm per voxel (spatial index)
    chunk_size_nm: tuple[float, ...]        # spatial chunk size, nm
    vertex_attributes: list[dict[str, Any]] = field(default_factory=list)
    bounds_nm: tuple[list[float], list[float]] | None = None
    frags_dir: str = ""                     # subdir holding .frags (often "")

    @property
    def chunk_size_voxels(self) -> tuple[int, ...]:
        return tuple(
            int(round(c / r)) for c, r in zip(self.chunk_size_nm, self.resolution_nm)
        )

    @property
    def attribute_names(self) -> list[str]:
        return [a["id"] for a in self.vertex_attributes]

    @property
    def attribute_dtypes(self) -> dict[str, str]:
        return {a["id"]: a.get("data_type", "float32") for a in self.vertex_attributes}


def parse_frag_key(key: str) -> tuple[int, ...]:
    """Parse a ``.frags`` key back to its per-axis voxel start corner.

    ``[dir/]x0-x1_y0-y1_z0-z1.frags`` → ``(x0, y0, z0)``.
    """
    name = key.rsplit("/", 1)[-1]
    if name.endswith(".frags"):
        name = name[: -len(".frags")]
    return tuple(int(part.split("-")[0]) for part in name.split("_"))


def frag_key(info: SkeletonInfo, voxel_start: Sequence[int]) -> str:
    """Build a ``.frags`` filename from a per-axis voxel start corner.

    ``<x0>-<x1>_<y0>-<y1>_<z0>-<z1>.frags`` where ``x1 = x0 +
    chunk_size_voxels`` (matches igneous ``SpatialIndex`` naming).
    """
    cs = info.chunk_size_voxels
    parts = [f"{int(s)}-{int(s) + int(c)}" for s, c in zip(voxel_start, cs)]
    name = "_".join(parts) + ".frags"
    return f"{info.frags_dir.rstrip('/')}/{name}" if info.frags_dir else name


def enumerate_frag_keys(
    info: SkeletonInfo,
    anchor_voxel: Sequence[int],
    counts: Sequence[int],
) -> list[str]:
    """Enumerate a block of ``.frags`` keys from an anchor corner.

    ``anchor_voxel`` is one valid chunk's voxel start (e.g. from a known
    filename); ``counts`` is the per-axis number of chunks.  Keys step by
    ``chunk_size_voxels`` per axis.  Avoids bucket-wide listing for test
    cutouts.
    """
    csv = info.chunk_size_voxels
    starts = [
        [int(anchor_voxel[a]) + i * csv[a] for i in range(int(counts[a]))]
        for a in range(len(anchor_voxel))
    ]
    keys: list[str] = []
    for x in starts[0]:
        for y in starts[1]:
            for z in starts[2]:
                keys.append(frag_key(info, (x, y, z)))
    return keys


# ===================================================================
# Readers
# ===================================================================

class PrecomputedFragsReader:
    """Read ``.frags`` chunks from a precomputed skeleton layer.

    Requires the optional ``ingest`` extras (``cloud-volume``,
    ``mapbuffer``, ``cloud-files``).
    """

    def __init__(self, base_url: str, *, frags_dir: str = ""):
        from cloudfiles import CloudFiles  # noqa: F401  (import check)

        self.base_url = base_url.rstrip("/")
        info = self._read_info()
        si = info["spatial_index"]
        self.info = SkeletonInfo(
            base_url=self.base_url,
            resolution_nm=tuple(float(x) for x in si["resolution"]),
            chunk_size_nm=tuple(float(x) for x in si["chunk_size"]),
            vertex_attributes=info.get("vertex_attributes", []) or [],
            frags_dir=frags_dir,
        )
        self._raw_info = info

    def _read_info(self) -> dict[str, Any]:
        from cloudfiles import CloudFiles
        cf = CloudFiles(self.base_url)
        info = cf.get_json("info")
        if info is None:
            raise FileNotFoundError(f"no info at {self.base_url}/info")
        return info

    def read_chunk(self, key: str) -> dict[int, dict[str, npt.NDArray]]:
        from cloudfiles import CloudFiles
        from cloudvolume import Skeleton
        from mapbuffer import MapBuffer

        cf = CloudFiles(self.base_url)
        raw = cf.get(key)
        if raw is None:
            return {}
        mb = MapBuffer(raw)
        attr_names = self.info.attribute_names
        out: dict[int, dict[str, npt.NDArray]] = {}
        for label in mb.keys():
            buf = mb[label]
            skel = Skeleton.from_precomputed(
                buf, vertex_attributes=self.info.vertex_attributes or None,
            )
            piece: dict[str, npt.NDArray] = {
                "vertices": np.asarray(skel.vertices, dtype=np.float32),
                "edges": np.asarray(skel.edges, dtype=np.int64).reshape(-1, 2),
            }
            for name in attr_names:
                val = getattr(skel, name, None)
                if val is not None:
                    piece[name] = np.asarray(val)
            out[int(label)] = piece
        return out


class InMemoryFragsReader:
    """Offline reader for tests: holds ``{key: {seg_id: piece}}``."""

    def __init__(self, info: SkeletonInfo, chunks: dict[str, dict[int, dict[str, npt.NDArray]]]):
        self.info = info
        self._chunks = chunks

    def read_chunk(self, key: str) -> dict[int, dict[str, npt.NDArray]]:
        return self._chunks.get(key, {})


# ===================================================================
# Extract → per-zarr-chunk pieces
# ===================================================================

def pieces_from_chunk(
    chunk: dict[int, dict[str, npt.NDArray]],
    *,
    chunk_shape_nm: tuple[float, ...],
    attribute_names: Sequence[str],
    gid_start: int = 0,
    origin: npt.NDArray | None = None,
    fixed_cell: ChunkCoords | None = None,
) -> tuple[dict[ChunkCoords, list[dict[str, Any]]], list[tuple[int, int]], int]:
    """Turn one ``.frags`` chunk into per-(origin-aligned)-zarr-chunk pieces.

    For each segment, edges that cross a zarr-chunk boundary are split out
    as **cross-chunk edges** (rather than dropped): each such edge's two
    endpoints get a global id and the edge is recorded so the level-0
    writer can store it in ``cross_chunk_links``.  The coarsener later
    uses those links to merge the object's pieces once both endpoints
    land in the same coarser chunk.  Intra-chunk connectivity is split
    into rooted components, one fragment per component.

    Returns ``(by_chunk, cross_edges, next_gid)`` where ``cross_edges``
    is a list of ``(gid_a, gid_b)`` and each emitted piece carries an
    ``"anchors"`` map ``{gid: local_vertex_index}`` for its endpoints.
    """
    cs = np.asarray(chunk_shape_nm, dtype=np.float64)
    off = None if origin is None else np.asarray(origin, dtype=np.float64)
    out: dict[ChunkCoords, list[dict[str, Any]]] = defaultdict(list)
    cross: list[tuple[int, int]] = []
    gid = int(gid_start)
    for seg_id, piece in chunk.items():
        verts = np.asarray(piece["vertices"], dtype=np.float32)
        n = len(verts)
        if n == 0:
            continue
        # Shift to the stored (grid-aligned, origin-0) coordinate frame.
        if off is not None:
            verts = (verts.astype(np.float64) - off).astype(np.float32)
        edges = np.asarray(piece.get("edges", np.zeros((0, 2))), dtype=np.int64).reshape(-1, 2)
        attrs = {nm: np.asarray(piece[nm]) for nm in attribute_names if nm in piece}

        if fixed_cell is not None:
            # Aligned mode: this whole .frag belongs to one zarr chunk
            # (its grid cell), so all edges are intra-chunk and pieces are
            # NOT phase-split.  Cross-chunk connectivity is recovered
            # separately by exact coincident-vertex matching across
            # adjacent .frags (see _detect_coincident_cross_edges).
            for comp in split_components(verts, edges, attrs):
                if len(comp["positions"]) == 0:
                    continue
                out[tuple(int(c) for c in fixed_cell)].append({
                    "segment_id": int(seg_id),
                    "positions": comp["positions"],
                    "edges": comp["edges"],
                    "attributes": comp["attributes"],
                })
            continue

        chunk_of = np.floor(verts / cs).astype(np.int64)
        if len(edges) > 0:
            same = np.all(chunk_of[edges[:, 0]] == chunk_of[edges[:, 1]], axis=1)
            intra = edges[same]
            crossing = edges[~same]
        else:
            intra = edges
            crossing = np.zeros((0, 2), np.int64)

        # Global ids for crossing endpoints (one per distinct vertex).
        gmap: dict[int, int] = {}
        if len(crossing) > 0:
            for v in np.unique(crossing.reshape(-1)).tolist():
                gmap[int(v)] = gid
                gid += 1

        comps = split_components(verts, intra, attrs, vertex_ids=np.arange(n))
        for comp in comps:
            cpos = comp["positions"]
            if len(cpos) == 0:
                continue
            cc = tuple(int(np.floor(cpos[0, a] / cs[a])) for a in range(len(cs)))
            pd: dict[str, Any] = {
                "segment_id": int(seg_id),
                "positions": cpos,
                "edges": comp["edges"],
                "attributes": comp["attributes"],
            }
            if gmap:
                vids = comp["vertex_ids"].tolist()
                anchors = {gmap[ov]: local
                           for local, ov in enumerate(vids) if ov in gmap}
                if anchors:
                    pd["anchors"] = anchors
            out[cc].append(pd)
        for a, b in crossing.tolist():
            cross.append((gmap[int(a)], gmap[int(b)]))
    return out, cross, gid


def _detect_coincident_cross_edges(
    pieces_by_cc: dict[ChunkCoords, list[dict[str, Any]]],
    *,
    chunk_shape_nm: tuple[float, ...],
    boundary_offset_nm: npt.NDArray,
    gid_start: int = 0,
) -> tuple[list[tuple[int, int]], int]:
    """Find cross-chunk edges from exact coincident boundary vertices.

    igneous splits a skeleton at ``.frag`` boundaries by **duplicating**
    the boundary vertex bit-identically into both adjacent chunks.  Those
    vertices sit on a chunk face: in integer-nm coordinates,
    ``(coord - boundary_offset) % chunk_shape == 0`` along the split axis
    (mirroring cloud-volume's mesh boundary dedup, which uses
    ``mod(verts - offset, chunk_size) == 0``).  For precomputed skeletons
    the offset is the half-voxel voxel-center shift (``0.5 · resolution``).

    One vectorized pass (no per-object loop):

    1. Select only vertices ON a chunk face via the exact modular test
       across all chunks/fragments — duplicates can only occur there.
    2. Group globally by ``(segment_id, coord)`` with a single lexsort;
       keying on segment id makes each group a same-object coincidence
       (no false links where two neurons touch a shared face).
    3. For each group spanning ≥2 distinct chunks, link one
       representative per chunk (star to the first) with a cross-chunk
       edge and mark those vertices as anchors.

    Returns ``(cross_edges, next_gid)``.
    """
    cs = np.rint(np.asarray(chunk_shape_nm, dtype=np.float64)).astype(np.int64)
    off = np.rint(np.asarray(boundary_offset_nm, dtype=np.float64)).astype(np.int64)

    chunk_keys = list(pieces_by_cc.keys())
    seg_cols: list[npt.NDArray] = []
    coord_cols: list[npt.NDArray] = []
    cid_col: list[npt.NDArray] = []
    pid_col: list[npt.NDArray] = []
    loc_col: list[npt.NDArray] = []
    for cid, cc in enumerate(chunk_keys):
        for pi, p in enumerate(pieces_by_cc[cc]):
            pos = np.asarray(p["positions"])
            if len(pos) == 0:
                continue
            coord = np.rint(pos).astype(np.int64)
            # On a chunk face along some axis (exact, integer-nm).
            on_face = np.any(np.mod(coord - off, cs) == 0, axis=1)
            idx = np.flatnonzero(on_face)
            if len(idx) == 0:
                continue
            seg_cols.append(np.full(len(idx), int(p["segment_id"]), dtype=np.int64))
            coord_cols.append(coord[idx])
            cid_col.append(np.full(len(idx), cid, dtype=np.int64))
            pid_col.append(np.full(len(idx), pi, dtype=np.int64))
            loc_col.append(idx.astype(np.int64))

    cross: list[tuple[int, int]] = []
    gid = int(gid_start)
    if not seg_cols:
        return cross, gid

    seg = np.concatenate(seg_cols)
    coord = np.concatenate(coord_cols, axis=0)
    cid_a = np.concatenate(cid_col)
    pid_a = np.concatenate(pid_col)
    loc_a = np.concatenate(loc_col)

    # Group by (seg, x, y, z) via one lexsort; group boundaries via diff.
    key = np.column_stack([seg, coord])
    order = np.lexsort([key[:, i] for i in range(key.shape[1] - 1, -1, -1)])
    sk_sorted = key[order]
    change = np.any(sk_sorted[1:] != sk_sorted[:-1], axis=1)
    starts = np.concatenate([[0], np.flatnonzero(change) + 1])
    ends = np.concatenate([starts[1:], [len(order)]])
    sizes = ends - starts
    for gi in np.flatnonzero(sizes > 1):
        members = order[starts[gi]:ends[gi]]
        # one representative per distinct chunk
        per_chunk: dict[int, int] = {}
        for m in members:
            c = int(cid_a[m])
            per_chunk.setdefault(c, int(m))
        if len(per_chunk) < 2:
            continue
        reps = list(per_chunk.values())
        gids = []
        for m in reps:
            g = gid
            gid += 1
            cc = chunk_keys[int(cid_a[m])]
            pieces_by_cc[cc][int(pid_a[m])].setdefault("anchors", {})[g] = int(loc_a[m])
            gids.append(g)
        for k in range(1, len(reps)):
            cross.append((gids[0], gids[k]))
    return cross, gid


# ===================================================================
# Driver
# ===================================================================

def _l0_extract_write(payload: dict, shared: dict | None = None) -> dict:
    """Worker (aligned mode): read one ``.frag``, write its single level-0 zarr
    chunk, and return the chunk's object-index ``records`` plus its
    boundary-vertex records (small) for cross-chunk-edge matching.

    Boundary vertices (on a chunk face) are tagged as anchors so
    ``write_skeleton_chunk`` reports their post-write chunk-local indices;
    only those tiny per-face records leave the worker, so the coordinator
    never holds bulk vertex data.  Picklable / runs under the executor;
    writes a disjoint chunk file.

    The ``reader`` arrives via ``shared`` (scattered to the workers once), not in
    the per-``.frag`` payload — so an in-memory reader holding many chunks is
    transported a single time, never re-pickled per task.
    """
    import numpy as np
    from zarr_vectors.core.store import get_resolution_level, open_store
    from zarr_vectors.types import skeletons as sk

    sh = shared or {}
    if "reader" in sh:
        # Backward-compatible path (tests/in-memory readers).
        reader = sh["reader"]
    else:
        spec = sh.get("reader_spec")
        if not isinstance(spec, dict):
            raise RuntimeError("_l0_extract_write expected shared['reader'] or shared['reader_spec']")
        base_url = str(spec["base_url"])
        frags_dir = str(spec.get("frags_dir", ""))
        cache_key = (base_url, frags_dir)
        reader = _WORKER_READER_CACHE.get(cache_key)
        if reader is None:
            reader = PrecomputedFragsReader(base_url, frags_dir=frags_dir)
            _WORKER_READER_CACHE[cache_key] = reader

    key = payload["key"]
    chunk = reader.read_chunk(key)
    if not chunk:
        return {"records": [], "boundary": [], "n_segments": 0, "read": 0}
    chunk_shape_nm = tuple(payload["chunk_shape_nm"])
    attribute_names = payload["attribute_names"]
    attr_dtypes = {n: np.dtype(d) for n, d in payload["attr_dtypes"].items()}
    origin = np.asarray(payload["origin"]) if payload["origin"] is not None else None
    by_cc, _cr, _gid = pieces_from_chunk(
        chunk, chunk_shape_nm=chunk_shape_nm,
        attribute_names=attribute_names, gid_start=0,
        origin=origin, fixed_cell=tuple(payload["fixed_cell"]),
    )
    cs = np.rint(np.asarray(chunk_shape_nm, dtype=np.float64)).astype(np.int64)
    off = np.rint(np.asarray(payload["boundary_offset_nm"], dtype=np.float64)).astype(np.int64)

    root = open_store(payload["store_path"], mode="r+")
    lg = get_resolution_level(root, 0)
    records: list = []
    boundary: list = []
    for cc, pieces in by_cc.items():
        tag_meta: list = []
        for pi, p in enumerate(pieces):
            pos = np.asarray(p["positions"])
            if len(pos) == 0:
                continue
            coord = np.rint(pos).astype(np.int64)
            on_face = np.any(np.mod(coord - off, cs) == 0, axis=1)
            for vidx in np.flatnonzero(on_face):
                tag = ("b", pi, int(vidx))
                p.setdefault("anchors", {})[tag] = int(vidx)
                tag_meta.append(
                    (int(p["segment_id"]),
                     tuple(int(c) for c in coord[vidx]), tag)
                )
        recs, alocs = sk.write_skeleton_chunk(lg, cc, pieces, attr_dtypes=attr_dtypes)
        records.extend(recs)
        for seg, coordt, tag in tag_meta:
            loc = alocs.get(tag)
            if loc is not None:
                boundary.append((seg, coordt, tuple(int(c) for c in cc), int(loc[1])))
    return {
        "records": records,
        "boundary": boundary,
        "n_segments": len(chunk),
        "read": 1,
    }


def _match_coincident_links(boundary: list) -> list:
    """Coordinator: match the per-chunk boundary-vertex records into
    cross-chunk links.  Groups by ``(segment_id, coord)`` and, for each group
    spanning ≥2 chunks, links one representative per chunk (star) — the same
    grouping :func:`_detect_coincident_cross_edges` does globally, but on the
    small gathered boundary records (post-write chunk-local indices), so no
    bulk vertex data is held.

    Returns ``[((ccA, viA), (ccB, viB)), ...]`` for
    :func:`write_skeleton_cross_chunk_links`.
    """
    if not boundary:
        return []
    ndim = len(boundary[0][1])
    seg = np.fromiter((b[0] for b in boundary), dtype=np.int64, count=len(boundary))
    coord = np.array([b[1] for b in boundary], dtype=np.int64).reshape(-1, ndim)
    key = np.column_stack([seg, coord])
    order = np.lexsort([key[:, i] for i in range(key.shape[1] - 1, -1, -1)])
    ks = key[order]
    change = np.any(ks[1:] != ks[:-1], axis=1)
    starts = np.concatenate([[0], np.flatnonzero(change) + 1])
    ends = np.concatenate([starts[1:], [len(order)]])
    links: list = []
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        per_chunk: dict = {}
        for m in order[s:e]:
            ccloc = (boundary[m][2], boundary[m][3])
            per_chunk.setdefault(ccloc[0], ccloc)
        reps = list(per_chunk.values())
        if len(reps) < 2:
            continue
        for k in range(1, len(reps)):
            links.append((reps[0], reps[k]))
    return links


def run_ingest(
    reader: Any,
    out_store: str | Path,
    keys: Iterable[str],
    *,
    bounds_nm: tuple[list[float], list[float]],
    attribute_names: Sequence[str] | None = None,
    attribute_dtypes: dict[str, str] | None = None,
    strides: Sequence[int] = (),
    chunk_scale_factors: Sequence[int | tuple[int, ...]] | None = None,
    sparsity_factors: Sequence[float] | None = None,
    sparsity_strategy: str = "length",
    drop_interior_below: int = 0,
    backend: str | None = None,
    align: bool = True,
    progress: bool = True,
    workers: int | None = None,
    executor: Any = None,
) -> dict[str, Any]:
    """Run the full ETL: extract + level-0 write + object index + pyramid.

    Args:
        reader: Object with ``.info`` (:class:`SkeletonInfo`) and
            ``read_chunk(key)``.
        out_store: Output zarr-vectors store path.
        keys: ``.frags`` keys to ingest (the test cutout, or all chunks).
        bounds_nm: ``(min_corner, max_corner)`` in nm for the store.
        attribute_names / attribute_dtypes: Per-vertex attributes to
            carry (defaults from ``reader.info``).
        strides: Per-pyramid-level decimation strides (keep every k-th
            vertex).  Empty → level 0 only.
        chunk_scale_factors / sparsity_factors: Aligned with ``strides``
            (defaults: 2 per axis, 1.0 = keep all).

    Returns:
        Summary dict (level-0 counts + pyramid summary).
    """
    info: SkeletonInfo = reader.info
    chunk_shape_nm = tuple(float(x) for x in info.chunk_size_nm)
    if attribute_names is None:
        attribute_names = info.attribute_names
    if attribute_dtypes is None:
        attribute_dtypes = {
            n: info.attribute_dtypes.get(n, "float32") for n in attribute_names
        }

    # Align the (spec origin-0) chunk grid to the source .frag grid by
    # shifting stored coordinates so the block's min corner — itself a
    # .frag boundary — maps to 0.  This makes each .frag piece fall wholly
    # inside one zarr chunk at level 0 (no phase-split).  The offset is
    # recorded as the NGFF world translation.  ``align=False`` keeps
    # absolute coordinates on the origin-0 grid (phase-split → cross-chunk
    # edges that the coarsener merges upward).
    ndim = len(chunk_shape_nm)
    if align:
        grid_origin = np.asarray(bounds_nm[0], dtype=np.float64)
        store_bounds = ([0.0] * ndim,
                        (np.asarray(bounds_nm[1], dtype=np.float64) - grid_origin).tolist())
        coord_off = grid_origin.tolist()
    else:
        grid_origin = np.zeros(ndim, dtype=np.float64)
        store_bounds = (list(bounds_nm[0]), list(bounds_nm[1]))
        coord_off = None
    root, lg = sk.init_skeleton_store(
        str(out_store), chunk_shape=chunk_shape_nm,
        bounds=store_bounds, ndim=ndim,
        attribute_dtypes=attribute_dtypes, backend=backend,
        coordinate_offset=coord_off,
    )
    np_attr_dtypes = {n: np.dtype(attribute_dtypes[n]) for n in attribute_names}
    origin_voxel = np.rint(grid_origin / np.asarray(info.resolution_nm)).astype(np.int64)
    csv = np.asarray(info.chunk_size_voxels, dtype=np.int64)
    keys = list(keys)

    # Build the map-like executor: an explicit one wins, else a Dask local
    # cluster when ``workers`` is set, else serial in-process.  It drives the
    # per-.frag extract+write (aligned mode) and is threaded into the pyramid.
    _dask_cm = None
    if executor is not None:
        ex = executor
    elif workers:
        from zarr_vectors_tools.ingest._parallel import dask_executor
        _dask_cm = dask_executor(workers)
        ex = _dask_cm.__enter__()
    else:
        def ex(func, items, shared=None):
            return [func(it, shared=shared) for it in items]

    import os as _os
    import time as _time
    _timing = bool(_os.environ.get("ZV_TIMING"))
    _phase: dict[str, float] = {}
    _t0 = _time.perf_counter()

    try:
        records: list[tuple[int, ChunkCoords, int]] = []
        cc_links: list = []
        n_chunks = 0
        n_segments = 0

        if align:
            # Parallel per-.frag extract + level-0 write — each .frag maps to
            # exactly one zarr chunk, written to disjoint files.  Workers
            # return records + tiny boundary-vertex records; cross-chunk links
            # are matched from the gathered boundary records (bounded — the
            # coordinator never holds bulk vertices).
            boundary_offset = (
                0.5 * np.asarray(info.resolution_nm, dtype=np.float64)
            ).tolist()
            payloads = []
            for key in keys:
                vs = np.asarray(parse_frag_key(key), dtype=np.int64)
                fixed_cell = tuple(((vs - origin_voxel) // csv).tolist())
                payloads.append({
                    "key": key, "store_path": str(out_store),
                    "chunk_shape_nm": list(chunk_shape_nm),
                    "attribute_names": list(attribute_names),
                    "attr_dtypes": {n: str(attribute_dtypes[n]) for n in attribute_names},
                    "origin": grid_origin.tolist(),
                    "fixed_cell": list(fixed_cell),
                    "boundary_offset_nm": boundary_offset,
                })
            # Send only a lightweight reader spec to workers.  The worker task
            # lazily builds and caches the concrete reader per process,
            # avoiding huge graph/scatter payloads from heavy reader objects.
            reader_shared: dict[str, Any]
            if isinstance(reader, PrecomputedFragsReader):
                reader_shared = {
                    "reader_spec": {
                        "base_url": reader.base_url,
                        "frags_dir": reader.info.frags_dir,
                    }
                }
            else:
                # In-memory/offline readers are already lightweight.
                reader_shared = {"reader": reader}

            boundary: list = []
            for r in ex(_l0_extract_write, payloads, reader_shared):
                records.extend(r["records"])
                boundary.extend(r["boundary"])
                n_chunks += r["read"]
                n_segments += r["n_segments"]
            _phase["l0_extract_write"] = _time.perf_counter() - _t0
            _t1 = _time.perf_counter()
            # Deterministic order (independent of worker completion) for the
            # object index: sorted by (chunk, fragment).
            records.sort(key=lambda rec: (tuple(rec[1]), rec[2]))
            cc_links = _match_coincident_links(boundary)
            sk.write_skeleton_cross_chunk_links(lg, cc_links, ndim=ndim)
            _phase["l0_cross_links"] = _time.perf_counter() - _t1
        else:
            # Non-aligned (phase-split) mode: serial accumulation + global
            # coincident detection (rare; alignment / cross-edge tests).
            pieces_by_cc: dict[ChunkCoords, list[dict[str, Any]]] = defaultdict(list)
            cross_edges: list[tuple[int, int]] = []
            gid = 0
            for ki, key in enumerate(keys):
                chunk = reader.read_chunk(key)
                if not chunk:
                    continue
                n_chunks += 1
                n_segments += len(chunk)
                by_cc, cr, gid = pieces_from_chunk(
                    chunk, chunk_shape_nm=chunk_shape_nm,
                    attribute_names=attribute_names, gid_start=gid,
                    origin=None, fixed_cell=None,
                )
                for cc, pieces in by_cc.items():
                    pieces_by_cc[cc].extend(pieces)
                cross_edges += cr
                if progress:
                    print(f"  [{ki + 1}/{len(keys)}] {key}: "
                          f"{len(chunk)} segs → {len(by_cc)} zarr chunks", flush=True)
            gid_loc: dict[int, tuple[ChunkCoords, int]] = {}
            for cc, pieces in sorted(pieces_by_cc.items()):
                recs, alocs = sk.write_skeleton_chunk(
                    lg, cc, pieces, attr_dtypes=np_attr_dtypes,
                )
                records += recs
                gid_loc.update(alocs)
            for ga, gb in cross_edges:
                la = gid_loc.get(ga)
                lb = gid_loc.get(gb)
                if la is None or lb is None or tuple(la[0]) == tuple(lb[0]):
                    continue
                cc_links.append((la, lb))
            sk.write_skeleton_cross_chunk_links(lg, cc_links, ndim=ndim)

        _t2 = _time.perf_counter()
        oid_of = build_object_index(lg, records, ndim=ndim)
        sk.finalize_skeleton_store(root)
        _phase["l0_object_index"] = _time.perf_counter() - _t2

        summary: dict[str, Any] = {
            "frag_chunks_read": n_chunks,
            "source_segment_pieces": n_segments,
            "objects": len(oid_of),
            "level0_fragments": len(records),
            "level0_cross_chunk_edges": len(cc_links),
        }

        if strides:
            _t3 = _time.perf_counter()
            # Adaptive per-level scheduling: use at most one worker per target
            # chunk (and no Dask when a level has only one target chunk).
            if executor is None and workers:
                from zarr_vectors.constants import VERTICES
                from zarr_vectors.core.arrays import list_chunk_keys
                from zarr_vectors.core.store import get_resolution_level, open_store
                from zarr_vectors_tools.ingest._parallel import dask_executor

                if _dask_cm is not None:
                    _dask_cm.__exit__(None, None, None)
                    _dask_cm = None

                stride_list = list(strides)
                n_levels = len(stride_list)
                csf_list = (
                    list(chunk_scale_factors)
                    if chunk_scale_factors is not None
                    else [2] * n_levels
                )
                spf_list = (
                    list(sparsity_factors)
                    if sparsity_factors is not None
                    else [1.0] * n_levels
                )

                levels: list[dict[str, Any]] = []
                boundary_offset_nm = (
                    0.5 * np.asarray(info.resolution_nm, dtype=np.float64)
                ).tolist()
                max_workers = int(workers)
                root_for_counts = open_store(str(out_store), mode="r")

                for i in range(n_levels):
                    csf = csf_list[i]
                    spf = float(spf_list[i])
                    if isinstance(csf, (tuple, list)):
                        scale = tuple(int(s) for s in csf)
                    else:
                        scale = tuple(int(csf) for _ in range(ndim))

                    src_level_group = get_resolution_level(root_for_counts, i)
                    src_chunks = list_chunk_keys(src_level_group, VERTICES)
                    target_chunks = {
                        tuple(int(cc[a] // scale[a]) for a in range(ndim))
                        for cc in src_chunks
                    }
                    target_chunk_count = max(1, len(target_chunks))
                    level_workers = min(max_workers, target_chunk_count)

                    if progress:
                        print(
                            f"  [pyramid] L{i}->L{i+1}: target_chunks={target_chunk_count} "
                            f"workers={level_workers}",
                            flush=True,
                        )

                    if level_workers <= 1:
                        lv = coarsen_skeleton_level(
                            str(out_store), source_level=i, target_level=i + 1,
                            stride=int(stride_list[i]),
                            sparsity_factor=spf,
                            chunk_scale_factor=csf,
                            sparsity_strategy=sparsity_strategy,
                            drop_interior_below=int(drop_interior_below or 0),
                            boundary_offset_nm=boundary_offset_nm,
                            executor=None,
                        )
                    else:
                        with dask_executor(level_workers) as level_ex:
                            lv = coarsen_skeleton_level(
                                str(out_store), source_level=i, target_level=i + 1,
                                stride=int(stride_list[i]),
                                sparsity_factor=spf,
                                chunk_scale_factor=csf,
                                sparsity_strategy=sparsity_strategy,
                                drop_interior_below=int(drop_interior_below or 0),
                                boundary_offset_nm=boundary_offset_nm,
                                executor=level_ex,
                            )
                    levels.append(lv)

                summary["pyramid"] = {"levels": levels, "num_levels": n_levels + 1}
            else:
                summary["pyramid"] = build_skeleton_pyramid(
                    str(out_store),
                    strides=list(strides),
                    chunk_scale_factors=list(chunk_scale_factors) if chunk_scale_factors else None,
                    sparsity_factors=list(sparsity_factors) if sparsity_factors else None,
                    sparsity_strategy=sparsity_strategy,
                    drop_interior_below=int(drop_interior_below or 0),
                    # 0.5·resolution = the half-voxel face offset the boundary /
                    # interior test uses (matches the L0 coincident-vertex check).
                    boundary_offset_nm=(
                        0.5 * np.asarray(info.resolution_nm, dtype=np.float64)
                    ).tolist(),
                    executor=ex,
                )
            _phase["pyramid"] = _time.perf_counter() - _t3

        if _timing:
            print(f"  [timing] total={_time.perf_counter() - _t0:.1f}s "
                  f"workers={workers or 'serial'}", flush=True)
            for k, v in _phase.items():
                print(f"    {k:18s} {v:7.1f}s", flush=True)
            for i, lv in enumerate(summary.get("pyramid", {}).get("levels", [])):
                t = lv.get("timings", {})
                map_t = t.get("map_phase_a", t.get("map", 0.0))
                red_t = t.get("reduce_phase_c", 0.0)
                attrs_t = t.get("object_attrs", 0.0)
                cross_t = t.get("cross_links_phase_b", 0.0)
                print(f"    L{i}->L{i+1} setup={t.get('setup', 0):6.1f}s "
                    f"mapA={map_t:6.1f}s reduceC={red_t:6.1f}s "
                    f"attrs={attrs_t:6.1f}s crossB={cross_t:6.1f}s "
                    f"fin={t.get('finalize', 0):6.1f}s "
                    f"chunks={t.get('n_target_chunks', 0)}", flush=True)
        return summary
    finally:
        if _dask_cm is not None:
            _dask_cm.__exit__(None, None, None)


# ===================================================================
# CLI
# ===================================================================

def _parse_int3(s: str) -> tuple[int, int, int]:
    parts = [int(x) for x in s.replace(",", " ").split()]
    if len(parts) != 3:
        raise ValueError(f"expected 3 ints, got {s!r}")
    return tuple(parts)  # type: ignore[return-value]


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m zarr_vectors.ingest.precomputed_skeletons`` CLI.

    Ingests a rectangular block of ``.frags`` chunks (anchored at a known
    chunk's voxel corner) into a multiscale zarr-vectors skeleton store.
    """
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", help="precomputed skeleton layer URL "
                   "(e.g. gs://flywire_v141_m783/skeletons_mip_1)")
    p.add_argument("out_store", help="output zarr-vectors store path")
    p.add_argument("--anchor", required=True, type=_parse_int3,
                   help="voxel corner of one .frags chunk, 'x y z'")
    p.add_argument("--counts", default="1 1 1", type=_parse_int3,
                   help="number of chunks per axis, 'nx ny nz'")
    p.add_argument("--frags-dir", default="",
                   help="subdir holding .frags (default: layer root)")
    p.add_argument("--strides", default="", help="comma list of decimation "
                   "strides (keep every k-th vertex), one per pyramid level")
    p.add_argument("--chunk-scales", default="",
                   help="comma list of per-level chunk-grid multipliers")
    p.add_argument("--sparsity", default="",
                   help="comma list of per-level object-drop factors")
    p.add_argument("--sparsity-strategy", default="length")
    p.add_argument("--no-align", action="store_true",
                   help="keep absolute coords on the origin-0 grid instead "
                   "of aligning the chunk grid to the .frag grid")
    p.add_argument("--workers", type=int, default=0,
                   help="scale the ingest across N worker processes via a Dask "
                   "local cluster (requires the 'parallel' extra); 0 = serial")
    p.add_argument("--drop-interior-below", type=int, default=0,
                   help="LOD: at each coarse level drop objects whose entire "
                   "decimated skeleton is <= N vertices AND fully chunk-interior "
                   "(no boundary-touching vertex, so they can't extend into a "
                   "neighbour chunk); 0 = keep all")
    args = p.parse_args(argv)

    reader = PrecomputedFragsReader(args.source, frags_dir=args.frags_dir)
    info = reader.info
    res = np.asarray(info.resolution_nm)
    anchor = np.asarray(args.anchor)
    counts = np.asarray(args.counts)
    csv = np.asarray(info.chunk_size_voxels)
    bounds = (
        [float(anchor[a] * res[a]) for a in range(3)],
        [float((anchor[a] + counts[a] * csv[a]) * res[a]) for a in range(3)],
    )
    keys = enumerate_frag_keys(info, tuple(args.anchor), tuple(args.counts))

    def _floats(s):
        return [float(x) for x in s.split(",")] if s else []

    def _ints(s):
        return [int(x) for x in s.split(",")] if s else None

    summary = run_ingest(
        reader, args.out_store, keys, bounds_nm=bounds,
        strides=_ints(args.strides) or (),
        chunk_scale_factors=_ints(args.chunk_scales),
        sparsity_factors=_floats(args.sparsity) or None,
        sparsity_strategy=args.sparsity_strategy,
        drop_interior_below=args.drop_interior_below,
        align=not args.no_align,
        workers=args.workers or None,
    )
    import json
    print(json.dumps({k: v for k, v in summary.items() if k != "pyramid"}, indent=2))
    if "pyramid" in summary:
        for s in summary["pyramid"]["levels"]:
            print(f"  level {s.get('method')}: objs={s['object_count']} "
                  f"verts={s['vertex_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
