"""Mesh coarsening strategies.

Two approaches:

1. **Vertex clustering**: merge vertices within spatial bins into
   metanodes, remap face indices, remove degenerate faces (collapsed
   to edges or points), and merge duplicate faces.

2. **Quadric error decimation** (simplified): iterative edge collapse
   using quadric error metrics for vertex placement.  Falls back to
   vertex clustering when the optional ``pyfqmr`` is not available.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors_tools.multiresolution.metanodes import generate_metanodes

# ===================================================================
# Vertex clustering
# ===================================================================

def coarsen_mesh_cluster(
    vertices: npt.NDArray[np.floating],
    faces: npt.NDArray[np.integer],
    bin_size: float | tuple[float, ...],
    *,
    vertex_attributes: dict[str, npt.NDArray] | None = None,
    agg_mode: str = "mean",
) -> dict[str, Any]:
    """Coarsen a mesh by clustering vertices within spatial bins.

    Vertices in the same bin merge into a metanode (centroid).  Face
    indices are remapped; faces that collapse to degenerate triangles
    (two or more vertices in the same bin) are removed.

    Args:
        vertices: ``(V, D)`` vertex positions.
        faces: ``(F, L)`` face index array.
        bin_size: Spatial bin edge length.
        vertex_attributes: Per-vertex attributes to aggregate.
        agg_mode: Attribute aggregation mode.

    Returns:
        Dict with:
        - ``vertices``: ``(K, D)`` metanode positions
        - ``faces``: ``(E, L)`` remapped faces (non-degenerate only)
        - ``vertex_attributes``: aggregated attributes
        - ``children``: list of K arrays of original vertex indices
        - ``vertex_count``, ``face_count``
        - ``vertex_reduction``: V / K
        - ``face_reduction``: F / E
        - ``degenerate_faces_removed``: count
    """
    n_verts, ndim = vertices.shape
    n_faces, link_width = faces.shape

    if n_verts == 0:
        return _empty_mesh_coarsen(ndim, link_width)

    # Generate metanodes from vertices
    meta_result = generate_metanodes(
        vertices, bin_size,
        attributes=vertex_attributes,
        agg_mode=agg_mode,
    )

    meta_verts = meta_result["metanode_positions"]
    children = meta_result["children"]
    meta_attrs = meta_result["metanode_attributes"]
    n_meta = len(meta_verts)

    # Build vertex → metanode mapping
    vert_to_meta = np.empty(n_verts, dtype=np.int64)
    for m_idx in range(n_meta):
        for c in children[m_idx]:
            vert_to_meta[c] = m_idx

    # Remap face indices
    remapped_faces = vert_to_meta[faces]  # (F, L)

    # Remove degenerate faces: faces where any two vertices map to same metanode
    keep_mask = np.ones(n_faces, dtype=bool)
    for i in range(link_width):
        for j in range(i + 1, link_width):
            keep_mask &= remapped_faces[:, i] != remapped_faces[:, j]

    valid_faces = remapped_faces[keep_mask]

    # Remove duplicate faces (sort vertex indices per face, then unique)
    if len(valid_faces) > 0:
        sorted_faces = np.sort(valid_faces, axis=1)
        _, unique_idx = np.unique(sorted_faces, axis=0, return_index=True)
        valid_faces = valid_faces[np.sort(unique_idx)]

    degenerate_count = n_faces - int(keep_mask.sum())
    duplicate_count = int(keep_mask.sum()) - len(valid_faces)

    return {
        "vertices": meta_verts,
        "faces": valid_faces,
        "vertex_attributes": meta_attrs,
        "children": children,
        "vertex_count": n_meta,
        "face_count": len(valid_faces),
        "vertex_reduction": n_verts / max(n_meta, 1),
        "face_reduction": n_faces / max(len(valid_faces), 1),
        "degenerate_faces_removed": degenerate_count + duplicate_count,
    }


# ===================================================================
# Quadric error decimation (simplified)
# ===================================================================

def coarsen_mesh_quadric(
    vertices: npt.NDArray[np.floating],
    faces: npt.NDArray[np.integer],
    target_face_count: int | None = None,
    target_ratio: float | None = None,
    *,
    vertex_attributes: dict[str, npt.NDArray] | None = None,
) -> dict[str, Any]:
    """Coarsen a mesh using quadric error edge collapse.

    Uses ``pyfqmr`` if available; otherwise falls back to vertex
    clustering with an estimated bin size.

    Args:
        vertices: ``(V, 3)`` vertex positions (3D only).
        faces: ``(F, 3)`` triangle face indices.
        target_face_count: Desired face count.  Mutually exclusive
            with ``target_ratio``.
        target_ratio: Fraction of faces to keep (0–1).
        vertex_attributes: Per-vertex attributes (interpolated if
            pyfqmr is used, aggregated otherwise).

    Returns:
        Dict with vertex/face arrays, counts, and method used.
    """
    n_verts = len(vertices)
    n_faces = len(faces)

    if target_ratio is not None:
        target_face_count = max(1, int(n_faces * target_ratio))
    elif target_face_count is None:
        target_face_count = max(1, n_faces // 4)

    # Try pyfqmr
    try:
        return _quadric_pyfqmr(vertices, faces, target_face_count, vertex_attributes)
    except ImportError:
        pass

    # Fallback: estimate bin size from target reduction
    reduction = n_faces / max(target_face_count, 1)
    # Rough heuristic: bin_size ≈ extent * reduction^(1/3)
    extent = np.max(vertices, axis=0) - np.min(vertices, axis=0)
    mean_extent = float(np.mean(extent))
    estimated_bin = mean_extent * (reduction ** (1.0 / 3.0)) / (n_verts ** (1.0 / 3.0)) * 2

    result = coarsen_mesh_cluster(
        vertices, faces, estimated_bin,
        vertex_attributes=vertex_attributes,
    )
    result["method"] = "vertex_clustering_fallback"
    return result


def _quadric_pyfqmr(
    vertices: npt.NDArray,
    faces: npt.NDArray,
    target_faces: int,
    vertex_attributes: dict[str, npt.NDArray] | None,
) -> dict[str, Any]:
    """Quadric decimation via pyfqmr."""
    import pyfqmr

    mesh = pyfqmr.Simplify()
    mesh.setMesh(
        np.ascontiguousarray(vertices, dtype=np.float64),
        np.ascontiguousarray(faces, dtype=np.int32),
    )
    mesh.simplify_mesh(target_count=target_faces, aggressiveness=7)

    new_verts = np.array(mesh.getMesh()[0], dtype=np.float32)
    new_faces = np.array(mesh.getMesh()[1], dtype=np.int64)

    result: dict[str, Any] = {
        "vertices": new_verts,
        "faces": new_faces,
        "vertex_count": len(new_verts),
        "face_count": len(new_faces),
        "vertex_reduction": len(vertices) / max(len(new_verts), 1),
        "face_reduction": len(faces) / max(len(new_faces), 1),
        "method": "quadric_pyfqmr",
    }

    # Attributes: nearest-vertex interpolation
    if vertex_attributes:
        from scipy.spatial import cKDTree
        tree = cKDTree(vertices)
        _, nearest = tree.query(new_verts)
        result["vertex_attributes"] = {
            name: data[nearest] for name, data in vertex_attributes.items()
        }
    else:
        result["vertex_attributes"] = {}

    return result


def _empty_mesh_coarsen(ndim: int, link_width: int) -> dict[str, Any]:
    return {
        "vertices": np.zeros((0, ndim), dtype=np.float32),
        "faces": np.zeros((0, link_width), dtype=np.int64),
        "vertex_attributes": {},
        "children": [],
        "vertex_count": 0,
        "face_count": 0,
        "vertex_reduction": 0,
        "face_reduction": 0,
        "degenerate_faces_removed": 0,
    }


# ===================================================================
# Level coarsener (registry strategy)
# ===================================================================
#
# Why this exists: without it a mesh store falls through
# ``select_coarsener_key`` to ``per_object``, which materialises
# ``np.concatenate`` of every fragment each object's manifest names.  On a
# store whose objects share chunk-wide fragments that is quadratic in the
# worst case -- measured on a 1720-neuron MICrONS store: 65.2M vertices on
# disk, 49.9 BILLION vertices across the per-object concatenations, a 766x
# amplification that OOM-killed a 251 GB machine.  It also hardcodes
# ``link_width=2`` when it rebuilds links, so triangles could not survive it
# even with infinite memory.
#
# This strategy is chunk-local instead: peak memory is one target chunk, and
# faces are streamed as numpy arrays through ``iter_link_cells`` /
# ``write_chunk_links`` rather than materialised as Python tuples.

#: Level-metadata tag recorded for levels this strategy produced.
COARSEN_MESH_CLUSTER: str = "mesh_cluster"

#: Dispatch key for :func:`~zarr_vectors_tools.multiresolution.coarsen.coarsen_level`.
COARSENER_KEY: str = "mesh"


def _mesh_target_chunk(cc: tuple, scale: tuple[int, ...]) -> tuple:
    return tuple(int(c) // int(s) for c, s in zip(cc, scale))


def _median_edge_length(
    src_group, ndim: int, link_width: int, *, max_faces: int = 50_000,
) -> float:
    """Median face-edge length of a level, from a sample of its intra-chunk faces.

    The clustering bin has to be scaled to the MESH, not to the store's
    ``bin_shape``: that is the spatial-index bin (32 um on a MICrONS store)
    while mesh vertices sit ~100 nm apart, so binning at ``bin_shape`` collapses
    every triangle to a point and the level comes out empty.  Edge length is the
    mesh's own resolution and needs no configuration to find.
    """
    from zarr_vectors.building import iter_link_cells, read_chunk_vertices
    from zarr_vectors.exceptions import ArrayError

    lengths: list[npt.NDArray] = []
    seen = 0
    cache: dict[tuple, npt.NDArray] = {}
    for _segment, offsets, source_chunk, groups in iter_link_cells(src_group, 0):
        # Intra-chunk faces only: all three corners in one chunk, so one
        # vertex read answers the whole record.
        if any(any(int(o) for o in off) for off in offsets):
            continue
        cc = tuple(int(x) for x in source_chunk)
        if cc not in cache:
            try:
                parts = read_chunk_vertices(src_group, cc, ndim=ndim)
                cache[cc] = (np.concatenate(parts, axis=0).astype(np.float64)
                             if parts else np.zeros((0, ndim)))
            except (ArrayError, Exception):  # noqa: BLE001
                continue
        pos = cache[cc]
        for g in groups:
            g = np.asarray(g, dtype=np.int64)
            if g.ndim != 2 or g.shape[1] != link_width or g.size == 0:
                continue
            g = g[: max(1, max_faces - seen)]
            if g.max() >= pos.shape[0]:
                continue
            for a in range(link_width):
                b = (a + 1) % link_width
                lengths.append(np.linalg.norm(pos[g[:, a]] - pos[g[:, b]], axis=1))
            seen += g.shape[0]
        if seen >= max_faces:
            break
    if not lengths:
        return 0.0
    return float(np.median(np.concatenate(lengths)))


def _bin_keys(positions: npt.NDArray, bin_shape: npt.NDArray) -> npt.NDArray:
    """One int64 key per vertex identifying its spatial bin."""
    coords = np.floor(positions / bin_shape).astype(np.int64)
    return np.ascontiguousarray(coords).view(
        np.dtype((np.void, coords.dtype.itemsize * coords.shape[1]))
    ).ravel()


def coarsen_mesh_level(
    store_path,
    source_level: int,
    target_level: int,
    *,
    coarsen_factor: float = 2.0,
    sparsity_factor: float = 1.0,
    chunk_scale_factor=1,
    sparsity_strategy: str = "random",
    sparsity_seed: int | None = None,
    compressor: Any = None,
    bin_size: float | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """Coarsen one mesh level by chunk-local vertex clustering.

    Vertices within a bin of ``source bin_shape x coarsen_factor`` collapse to
    their centroid; faces are remapped onto the centroids and dropped when two
    or more of their corners land in the same bin.  Object ids and the group
    taxonomy carry across unchanged, and each surviving object keeps one
    fragment per target chunk.
    """
    from contextlib import nullcontext

    from zarr_vectors.building import (
        LevelMetadata,
        create_attribute_array,
        create_links_array,
        create_links_family,
        create_object_index_array,
        create_resolution_level,
        create_vertices_array,
        finalize_links,
        get_level_chunk_shape,
        get_resolution_level,
        iter_link_cells,
        link_family_policy,
        list_chunk_keys,
        open_store,
        read_all_object_manifests,
        read_chunk_attributes,
        read_chunk_vertices,
        read_level_metadata,
        read_root_metadata,
        read_vertex_fragment_index,
        write_chunk_attributes,
        write_chunk_links,
        write_chunk_vertices,
        write_links,
        write_object_index,
    )
    from zarr_vectors.constants import VERTEX_ATTRIBUTES, VERTICES
    from zarr_vectors.exceptions import ArrayError

    from ..groupings import propagate_groupings, surviving_oids_from
    from ..object_selection import apply_sparsity
    from .skeleton_bins import default_attr_agg

    root = open_store(str(store_path), mode="r+")
    root_meta = read_root_metadata(root)
    ndim = len(root_meta.chunk_shape)
    if isinstance(chunk_scale_factor, (list, tuple)):
        scale = tuple(int(v) for v in chunk_scale_factor)
    else:
        scale = (int(chunk_scale_factor),) * ndim
    src_group = get_resolution_level(root, source_level)

    try:
        _src_lm = read_level_metadata(root, source_level)
    except Exception:  # noqa: BLE001
        _src_lm = None
    root_bin = tuple(float(b) for b in root_meta.effective_bin_shape)

    try:
        link_width = int(link_family_policy(src_group, 0).get("link_width", 3))
    except Exception:  # noqa: BLE001
        link_width = 3

    # The clustering bin is a multiple of the SOURCE LEVEL's own edge length,
    # so ``coarsen_factor`` is a true ratio against the level below and each
    # level compounds. ``bin_size`` overrides it outright.
    if bin_size is not None:
        cluster_bin = float(bin_size)
    else:
        edge = _median_edge_length(src_group, ndim, link_width)
        cluster_bin = edge * float(coarsen_factor)
    if not np.isfinite(cluster_bin) or cluster_bin <= 0:
        # Nothing to measure (no faces): fall back to the index bin, which
        # at least produces a valid, if very coarse, level.
        cluster_bin = float(root_bin[0]) * float(coarsen_factor)
    bin_arr = np.full(ndim, cluster_bin, dtype=np.float64)

    # LevelMetadata.bin_shape drives the NGFF scale transform, and every level
    # >= 1 must declare one. Record the index-bin scaling the coarsening
    # corresponds to, following the polyline/per-fragment convention, rather
    # than the clustering bin, which is a geometry scale and not comparable.
    src_bin = getattr(_src_lm, "bin_shape", None) if _src_lm else None
    source_bin = tuple(float(b) for b in src_bin) if src_bin else root_bin
    target_bin = tuple(b * float(coarsen_factor) for b in source_bin)

    # ---- objects + sparsity ------------------------------------------
    try:
        src_manifests = read_all_object_manifests(src_group)
    except Exception:  # noqa: BLE001
        src_manifests = []
    has_objects = len(src_manifests) > 0
    n_src_objects = len(src_manifests)
    if has_objects and sparsity_factor > 1.0 and n_src_objects > 1:
        alive = np.array([len(m) > 0 for m in src_manifests], dtype=bool)
        keep_oids = sorted(int(o) for o in apply_sparsity(
            n_src_objects, 1.0 / float(sparsity_factor), sparsity_strategy,
            seed=sparsity_seed, alive_mask=alive, relative_to="alive",
        ))
    else:
        keep_oids = list(range(n_src_objects))
    keep_set = set(keep_oids)

    # Which object owns each source fragment.  With one fragment per
    # (chunk, object) -- what write_mesh produces -- this is exact; a fragment
    # shared by several objects is attributed to the first that names it,
    # which only affects which object's manifest lists the coarse fragment.
    owner: dict[tuple, int] = {}
    for oid in keep_oids:
        for cc, f in src_manifests[oid]:
            owner.setdefault((tuple(int(x) for x in cc), int(f)), oid)

    # ---- per-vertex attributes to carry --------------------------------
    # Without this a mesh pyramid keeps geometry and drops every scalar on
    # it: thickness, curvature, myelin and parcel labels all vanish above
    # level 0, which makes a coarse surface unshadeable and a coarse EM mesh
    # unlabellable.  ``.children()``, not iteration -- each attribute is a
    # flat array node and iterating the group yields sub-groups only.
    vattr_names: list[str] = []
    vattr_dtypes: dict[str, Any] = {}
    vattr_ncols: dict[str, int] = {}
    vattr_channels: dict[str, list[str] | None] = {}
    if VERTEX_ATTRIBUTES in src_group:
        for _name in src_group[VERTEX_ATTRIBUTES].children():
            try:
                _meta = src_group.read_array_meta(f"{VERTEX_ATTRIBUTES}/{_name}")
            except ArrayError:
                continue
            _cn = _meta.get("channel_names")
            vattr_names.append(_name)
            vattr_dtypes[_name] = np.dtype(_meta.get("dtype", "float32"))
            vattr_ncols[_name] = len(_cn) if _cn else 1
            vattr_channels[_name] = _cn
    # A continuous quantity averages over the cluster; a categorical code
    # takes the value of the member nearest the centroid, so a parcel label
    # is always one that genuinely occurred rather than the mean of two.
    vattr_modes = default_attr_agg(vattr_dtypes)

    # ---- pass 1: cluster each TARGET chunk's vertices ------------------
    # Source chunks are grouped by target chunk first, so a bin that several
    # source chunks contribute to still collapses to ONE metavertex.
    src_chunks = [tuple(int(x) for x in cc)
                  for cc in list_chunk_keys(src_group, VERTICES)]
    by_target: dict[tuple, list[tuple]] = {}
    for cc in sorted(src_chunks):
        by_target.setdefault(_mesh_target_chunk(cc, scale), []).append(cc)

    # (source chunk) -> int64 array mapping chunk-local vertex id -> metavertex
    # index within its target chunk (-1 = vertex belongs to no kept object).
    meta_of: dict[tuple, npt.NDArray] = {}
    # (target chunk) -> centroid positions
    out_vertices: dict[tuple, npt.NDArray] = {}
    # (target chunk) -> per-object fragment index, and the fragments themselves
    out_frag_of: dict[tuple, dict[int, int]] = {}
    out_frag_members: dict[tuple, list[list[int]]] = {}
    n_src_verts = 0

    out_attributes: dict[tuple, dict[str, npt.NDArray]] = {}

    for cc_t, members in sorted(by_target.items()):
        chunk_pos: list[npt.NDArray] = []
        chunk_owner: list[npt.NDArray] = []
        chunk_attrs: dict[str, list[npt.NDArray]] = {n: [] for n in vattr_names}
        spans: list[tuple[tuple, int, int]] = []
        cursor = 0
        for cc in members:
            try:
                frag_index = read_vertex_fragment_index(src_group, cc)
                groups = read_chunk_vertices(
                    src_group, cc, dtype=np.float32, ndim=ndim,
                )
            except ArrayError:
                continue
            if not groups:
                continue
            n_rows = sum(int(np.asarray(g).shape[0]) for g in groups)
            pos = np.zeros((n_rows, ndim), dtype=np.float32)
            own = np.full(n_rows, -1, dtype=np.int64)
            # Attribute cells are per fragment and 1:1 with the vertices, so
            # they scatter to the same rows the positions do.
            attr_cells: dict[str, list] = {}
            for name in vattr_names:
                try:
                    attr_cells[name] = read_chunk_attributes(
                        src_group, name, cc, dtype=vattr_dtypes[name],
                        ncols=vattr_ncols[name],
                    )
                except ArrayError:
                    attr_cells[name] = []
            attr_rows = {
                name: np.zeros(
                    (n_rows,) if vattr_ncols[name] == 1
                    else (n_rows, vattr_ncols[name]),
                    dtype=vattr_dtypes[name],
                )
                for name in vattr_names
            }
            at = 0
            for f, g in enumerate(groups):
                g = np.asarray(g, dtype=np.float32)
                if g.size == 0:
                    continue
                if frag_index.is_range(f):
                    start, count = frag_index.range(f)
                    idx = np.arange(int(start), int(start) + int(count))
                else:
                    idx = np.asarray(frag_index.indices(f), dtype=np.int64)
                if idx.size != g.shape[0]:
                    idx = np.arange(at, at + g.shape[0], dtype=np.int64)
                pos[idx] = g
                for name in vattr_names:
                    cells = attr_cells[name]
                    if f < len(cells):
                        block = np.asarray(cells[f], dtype=vattr_dtypes[name])
                        if block.shape[0] == idx.size:
                            attr_rows[name][idx] = block
                o = owner.get((cc, f))
                if o is not None and o in keep_set:
                    own[idx] = o
                at += g.shape[0]
            n_src_verts += n_rows
            chunk_pos.append(pos)
            chunk_owner.append(own)
            for name in vattr_names:
                chunk_attrs[name].append(attr_rows[name])
            spans.append((cc, cursor, cursor + n_rows))
            cursor += n_rows

        if not chunk_pos:
            continue
        pos_all = np.concatenate(chunk_pos, axis=0)
        own_all = np.concatenate(chunk_owner, axis=0)
        del chunk_pos, chunk_owner

        alive_mask = own_all >= 0
        keys = _bin_keys(pos_all, bin_arr)
        # Cluster by (object, bin), not by bin alone.  Two reasons, and both
        # matter: a bin holding two cells' surfaces must not weld them into one
        # metavertex, and a metavertex belonging to exactly one object is what
        # lets each object's vertices sit CONTIGUOUSLY in the chunk buffer.  A
        # face's stored index is an offset into that chunk's fragments
        # concatenated in write order (see meshes.read_mesh), so contiguous
        # per-object fragments are what keep the remapped faces pointing at the
        # vertices they were remapped onto.
        pair = np.empty(
            int(alive_mask.sum()), dtype=[("o", np.int64), ("k", keys.dtype)],
        )
        pair["o"] = own_all[alive_mask]
        pair["k"] = keys[alive_mask]
        uniq, inverse = np.unique(pair, return_inverse=True)
        n_meta = int(uniq.size)
        meta_idx = np.full(pos_all.shape[0], -1, dtype=np.int64)
        meta_idx[alive_mask] = inverse

        counts = np.bincount(inverse, minlength=n_meta).astype(np.float64)
        cent = np.zeros((n_meta, ndim), dtype=np.float64)
        for d in range(ndim):
            cent[:, d] = np.bincount(
                inverse, weights=pos_all[alive_mask, d].astype(np.float64),
                minlength=n_meta,
            )
        cent /= np.maximum(counts, 1)[:, None]
        out_vertices[cc_t] = cent.astype(np.float32)

        if vattr_names:
            alive_positions = pos_all[alive_mask].astype(np.float64)
            # Rank each member by distance to its cluster centroid, ties
            # broken on source order, so "nearest" is deterministic across
            # workers.  The first member of each group is the representative.
            d2 = np.sum((alive_positions - cent[inverse]) ** 2, axis=1)
            n_alive = int(alive_positions.shape[0])
            order = np.lexsort(
                (np.arange(n_alive, dtype=np.int64), d2, inverse),
            )
            group_starts = np.concatenate(
                [[0], np.cumsum(counts.astype(np.int64))[:-1]],
            ).astype(np.int64)
            representative = order[group_starts] if n_meta else order[:0]
            chunk_out: dict[str, npt.NDArray] = {}
            for name in vattr_names:
                values = np.concatenate(chunk_attrs[name], axis=0)[alive_mask]
                ncols = vattr_ncols[name]
                flat = values.reshape(n_alive, -1)
                if vattr_modes.get(name) == "mean":
                    acc = np.zeros((n_meta, flat.shape[1]), dtype=np.float64)
                    for c in range(flat.shape[1]):
                        acc[:, c] = np.bincount(
                            inverse, weights=flat[:, c].astype(np.float64),
                            minlength=n_meta,
                        )
                    acc /= np.maximum(counts, 1)[:, None]
                    if np.issubdtype(vattr_dtypes[name], np.integer):
                        acc = np.rint(acc)
                    agg = acc.astype(vattr_dtypes[name])
                else:
                    agg = flat[representative].astype(vattr_dtypes[name])
                chunk_out[name] = (
                    agg.reshape(n_meta) if ncols == 1
                    else agg.reshape(n_meta, ncols)
                )
            out_attributes[cc_t] = chunk_out

        # np.unique sorted by object first, so each object's metavertices are
        # one contiguous run: fragment f is [start, stop) of the centroid array.
        meta_owner = uniq["o"]
        frag_of: dict[int, int] = {}
        spans_out: list[tuple[int, int]] = []
        if n_meta:
            starts = np.concatenate((
                [0], np.flatnonzero(np.diff(meta_owner)) + 1,
            )).astype(np.int64)
            stops = np.append(starts[1:], n_meta)
            for f, (a, b) in enumerate(zip(starts, stops)):
                frag_of[int(meta_owner[a])] = f
                spans_out.append((int(a), int(b)))
        out_frag_of[cc_t] = frag_of
        out_frag_members[cc_t] = spans_out

        for cc, lo, hi in spans:
            meta_of[cc] = meta_idx[lo:hi]

    # ---- write the target level ---------------------------------------
    # Scale the SOURCE level's chunk shape, not the root's: at level 2 the
    # root shape is two doublings behind, and a target grid sized from it does
    # not contain the coords ``source // scale`` produces.
    src_chunk_shape = get_level_chunk_shape(root_meta, _src_lm)
    target_chunk_shape_override = None
    if any(s != 1 for s in scale):
        target_chunk_shape_override = tuple(
            float(c) * float(s) for c, s in zip(src_chunk_shape, scale)
        )
    total_vertices = int(sum(v.shape[0] for v in out_vertices.values()))
    level_meta = LevelMetadata(
        level=target_level,
        vertex_count=total_vertices,
        arrays_present=(
            [VERTICES, "links", "object_index"]
            + (["vertex_attributes"] if vattr_names else [])
        ),
        bin_shape=target_bin,
        bin_ratio=tuple(
            max(1, int(round(float(t) / float(r))))
            for t, r in zip(target_bin, root_bin)
        ),
        chunk_shape=target_chunk_shape_override,
        object_sparsity=(1.0 / sparsity_factor),
        coarsening_method=COARSEN_MESH_CLUSTER,
        parent_level=source_level,
        preserves_object_ids=has_objects,
        inherited_num_objects=n_src_objects if has_objects else 0,
        shared_fragments=False,
    )
    level_group = create_resolution_level(root, target_level, level_meta)

    ctx = (level_group.batched_writes(compressor=compressor)
           if compressor else nullcontext())
    with ctx:
        create_vertices_array(level_group, dtype="float32")
        if has_objects:
            create_object_index_array(level_group)
        for name in vattr_names:
            create_attribute_array(
                level_group, name, dtype=str(vattr_dtypes[name]),
                channel_names=vattr_channels[name],
            )

    for cc_t in sorted(out_vertices):
        cent = out_vertices[cc_t]
        spans = out_frag_members[cc_t]
        # One contiguous fragment per object, written in centroid order, so the
        # concatenation of the fragments IS the centroid array and the face
        # indices computed against it stay valid.
        blocks = ([cent[a:b] for a, b in spans] if spans else [cent])
        write_chunk_vertices(level_group, cc_t, blocks, dtype=np.float32)
        # Attributes are cut on exactly the same spans, so a fragment's
        # values stay paired with its vertices.
        for name in vattr_names:
            column = out_attributes.get(cc_t, {}).get(name)
            if column is None:
                continue
            attr_blocks = (
                [column[a:b] for a, b in spans] if spans else [column]
            )
            write_chunk_attributes(
                level_group, name, cc_t, attr_blocks,
                dtype=vattr_dtypes[name],
            )

    # ---- pass 2: remap faces, streaming cell by cell -------------------
    # Grouped by (base target chunk, per-endpoint chunk offsets) so each write
    # is one numpy array; nothing here ever becomes a Python tuple per face.
    pending: dict[tuple, list[npt.NDArray]] = {}
    faces_in = faces_out = degenerate = 0
    for _segment, offsets, source_chunk, groups in iter_link_cells(src_group, 0):
        src_cc = tuple(int(x) for x in source_chunk)
        for g in groups:
            g = np.asarray(g, dtype=np.int64)
            if g.ndim != 2 or g.size == 0:
                continue
            if g.shape[1] == link_width + 1:
                # Cross-offset rows carry a leading perm_idx column that undoes
                # the canonical endpoint sort (links_has_perm).  Endpoint k of
                # the remaining columns still pairs with offsets[k-1], which is
                # all the remap needs; winding is not preserved across
                # clustering anyway, and write_links re-canonicalises.
                g = g[:, 1:]
            elif g.shape[1] != link_width:
                continue
            faces_in += g.shape[0]
            ends_chunk: list[tuple] = []
            ends_meta: list[npt.NDArray] = []
            bad = False
            for k in range(link_width):
                # The cell's own chunk is endpoint 0's; ``offsets`` names
                # endpoints 1.. relative to it, so it has link_width - 1
                # entries and must not be indexed by k directly.
                off = ((0,) * ndim if k == 0
                       else (offsets[k - 1] if k - 1 < len(offsets)
                             else (0,) * ndim))
                cc_k = tuple(int(a) + int(b) for a, b in zip(src_cc, off))
                lut = meta_of.get(cc_k)
                if lut is None:
                    bad = True
                    break
                col = g[:, k]
                ok = (col >= 0) & (col < lut.size)
                m = np.full(col.shape, -1, dtype=np.int64)
                m[ok] = lut[col[ok]]
                ends_chunk.append(_mesh_target_chunk(cc_k, scale))
                ends_meta.append(m)
            if bad:
                continue
            keep = np.ones(g.shape[0], dtype=bool)
            for m in ends_meta:
                keep &= m >= 0
            # A face whose corners collapsed into one bin is degenerate; two
            # corners in the same bin leaves a line, not a triangle.
            for a in range(link_width):
                for b in range(a + 1, link_width):
                    same = (ends_meta[a] == ends_meta[b]) & (
                        np.asarray(ends_chunk[a]) == np.asarray(ends_chunk[b])
                    ).all()
                    keep &= ~same
            degenerate += int((~keep).sum())
            if not keep.any():
                continue
            # The offsets segment names endpoints 2..L relative to the cell's
            # own chunk, which is endpoint 1's -- hence link_width - 1 entries,
            # matching the ``0.0.0_0.0.+1`` naming write_links produces.
            base = ends_chunk[0]
            off_key = tuple(
                tuple(int(c) - int(b) for c, b in zip(cc_k, base))
                for cc_k in ends_chunk[1:]
            )
            arr = np.stack([m[keep] for m in ends_meta], axis=1)
            pending.setdefault((base, off_key), []).append(arr)

    create_links_family(
        level_group, delta=0, link_width=link_width, sid_ndim=ndim,
    )
    zero = (0,) * ndim
    # Cross-offset records first: ``write_links`` canonicalises endpoint order
    # and records the ``perm_idx`` that restores winding, and it allocates its
    # own offsets arrays. It takes the tuple-of-tuples form, which is why only
    # the boundary faces go through it -- on a real store that is a fraction of
    # a percent of them (0.31% at 128 um chunks), while the intra-chunk
    # majority stays as numpy arrays below.
    cross_records: list[list[tuple]] = []
    for (base, off_key), arrs in sorted(pending.items()):
        if all(off == zero for off in off_key):
            continue
        stacked = np.unique(np.concatenate(arrs, axis=0), axis=0)
        faces_out += int(stacked.shape[0])
        # off_key names endpoints 1.. relative to the cell, so endpoint 0 is
        # the cell's own chunk.
        ends = [tuple(int(b) for b in base)] + [
            tuple(int(b) + int(o) for b, o in zip(base, off)) for off in off_key
        ]
        cross_records.extend(
            [(ends[k], int(row[k])) for k in range(link_width)]
            for row in stacked
        )
    if cross_records:
        write_links(
            level_group, cross_records, ndim, delta=0, link_width=link_width,
        )
    # Intra-chunk faces: one numpy array per cell, never materialised as
    # Python objects.
    for (base, off_key), arrs in sorted(pending.items()):
        if not all(off == zero for off in off_key):
            continue
        # Distinct source triangles can collapse onto one triple of
        # metavertices; keep each triple once.
        stacked = np.unique(np.concatenate(arrs, axis=0), axis=0)
        faces_out += int(stacked.shape[0])
        create_links_array(
            level_group, link_width, delta=0, sid_ndim=ndim,
            offsets=[zero] * (link_width - 1),
        )
        write_chunk_links(
            level_group, base, [stacked], dtype=np.int64, delta=0,
            link_width=link_width,
        )
    finalize_links(level_group, delta=0)

    # ---- manifests + groupings ----------------------------------------
    if has_objects:
        new_manifests: dict[int, list] = {oid: [] for oid in keep_oids}
        for cc_t, frag_of in sorted(out_frag_of.items()):
            for oid, frag_idx in frag_of.items():
                new_manifests.setdefault(oid, []).append((cc_t, frag_idx))
        write_object_index(
            level_group, new_manifests, sid_ndim=ndim,
            total_objects=n_src_objects,
        )
        propagate_groupings(
            src_group, level_group,
            surviving_oids=surviving_oids_from(keep_oids, sparsity_factor),
        )

    return {
        "vertex_count": total_vertices,
        "object_count": len(keep_oids),
        "objects_kept": len(keep_oids),
        "method": COARSEN_MESH_CLUSTER,
        "preserves_object_ids": has_objects,
        "shared_fragments": False,
        "vertices_in": int(n_src_verts),
        "faces_in": int(faces_in),
        "faces_out": int(faces_out),
        "degenerate_faces_removed": int(degenerate),
        "cluster_bin": float(cluster_bin),
        "attributes_carried": list(vattr_names),
    }


def _mesh_coarsener(
    store_path, source_level, target_level, **kwargs,
) -> dict[str, Any]:
    """Registry adapter — drops the kwargs this strategy has no use for."""
    return coarsen_mesh_level(
        store_path, source_level, target_level,
        coarsen_factor=kwargs.get("coarsen_factor", 2.0),
        sparsity_factor=kwargs.get("sparsity_factor", 1.0),
        chunk_scale_factor=kwargs.get("chunk_scale_factor", 1),
        sparsity_strategy=kwargs.get("sparsity_strategy", "random"),
        sparsity_seed=kwargs.get("sparsity_seed"),
        compressor=kwargs.get("compressor"),
        bin_size=kwargs.get("bin_size"),
    )
