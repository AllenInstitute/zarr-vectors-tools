"""Store-level mesh coarsening by quadric edge collapse, object by object.

WHY OBJECT-LOCAL AND NOT CHUNK-LOCAL
------------------------------------
``strategies.meshes`` is chunk-local because the generic per-object coarsener
once OOM-killed a 251 GB machine, a 766x read amplification. That amplification
was a property of the retired chunk-wide-fragment layout, not of the data:
under the current one-fragment-per-(chunk, object) layout the measured
amplification is 1.000x, the largest object in the 1720-neuron MICrONS store is
153,510 vertices (5.5 MB gathered), and all 1720 together are 2.35 GB. Every
one of the 114,624,311 faces at level 0 has its three corners inside a single
object, so objects are an exact partition of both vertices and faces.

Working per object rather than per chunk buys three things chunk-locality
cannot: an edge collapse needs the ring of faces around a vertex, which a chunk
boundary cuts; the surface can be welded (a chunk-wide weld would fuse two
neurons that happen to touch, 0.22-0.32% of a chunk's vertices are exact
coincidences between DIFFERENT cells); and there are no seams to reconcile,
because the object is re-split into chunks only after it has been simplified.

WHAT IT WRITES
--------------
One fragment per (target chunk, object), vertices contiguous within it, faces
as links -- the numpy path for the intra-chunk majority (98.2% at level 0) and
the tuple path for the cross-chunk remainder, which is the split the incumbent
already makes and the only one ``write_links`` supports for canonicalisation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from .mesh_decimate import decimate_mesh

#: Level-metadata tag recorded for levels this strategy produced.
COARSEN_MESH_DECIMATE: str = "mesh_quadric_collapse"

#: Dispatch key for coarsen_level.
COARSENER_KEY: str = "mesh_decimate"

#: Bytes per vertex and per face in the raw store layout, measured: positions
#: are 3 x float32 and a face is 3 x int64 plus ~0.7% index overhead.
BYTES_PER_VERTEX = 12.0
BYTES_PER_FACE = 24.0


def predict_bytes(n_vertices: int, n_faces: int) -> float:
    """On-disk bytes for a level, from its vertex and face counts.

    Linear and accurate to under a percent on the real store, which is what
    lets a caller's *data size* target be converted to a face target
    analytically instead of by trial decimation.
    """
    return (BYTES_PER_VERTEX * n_vertices + BYTES_PER_FACE * n_faces) * 1.007


def faces_for_byte_budget(budget: float, v_over_f: float) -> int:
    """Invert :func:`predict_bytes` at a known vertex/face ratio."""
    per_face = (BYTES_PER_VERTEX * v_over_f + BYTES_PER_FACE) * 1.007
    return max(4, int(budget / per_face))



def tube_floor_faces(
    path_length_nm: float,
    radius_nm: float,
    *,
    n_radial: int = 3,
    max_aspect: float = 3.0,
) -> int:
    """Fewest faces at which a tubular arbour is still a surface.

    A tube of ``n_radial`` segments costs ``2 * n_radial`` faces per
    longitudinal step, so a face budget FIXES the step length:

        step = 2 * n_radial * path_length / faces

    and the step divided by the tube diameter is the aspect ratio of the
    quads being laid down. Past roughly 3:1 the triangles are slivers lying
    along the centreline and the tube has stopped reading as a tube, however
    faithful the vertex positions are.

    Measured on the 37.0 mm / 224 nm-radius test axon: 391k faces gives a
    568 nm step (1.3:1, still round), while 97k gives 2292 nm (5.1:1, visibly
    collapsed). The floor at 3:1 is 165k faces. This is a property of the
    geometry, not of the simplifier -- no algorithm renders 37 mm of 448 nm
    tube in 97k faces -- which is why it belongs in the STOP rule rather than
    in the decimator.
    """
    diameter = 2.0 * max(radius_nm, 1e-9)
    max_step = max_aspect * diameter
    return int(np.ceil(2 * n_radial * path_length_nm / max_step))


# ===================================================================
# reading one level into per-object meshes
# ===================================================================

def _unpermute(tri: npt.NDArray, perm: npt.NDArray, link_width: int):
    """Undo the canonical endpoint sort recorded as a Lehmer ``perm_idx``.

    ``zarr_vectors.spatial.boundary.apply_perm_inverse`` does this one row at a
    time in Python. There are only ``link_width!`` distinct permutations (6 for
    triangles), so decoding each once and applying it to all the rows that
    carry it is the same operation without the per-face interpreter cost.
    """
    import math

    from zarr_vectors.building import apply_perm_inverse

    out = np.empty_like(tri)
    n_perm = math.factorial(link_width)
    for p in np.unique(perm):
        pi = int(p)
        sel = perm == pi
        if pi < 0 or pi >= n_perm:
            out[sel] = tri[sel]          # unrecognised: leave order as read
            continue
        # apply_perm_inverse on the column identity gives the scatter target
        # for each canonical position, which is all that varies per row.
        order = apply_perm_inverse(list(range(link_width)), pi, link_width)
        for canon_pos, orig_pos in enumerate(order):
            out[sel, canon_pos] = tri[sel, orig_pos]
    return out


def _vertex_attribute_spec(src_group):
    """``{name: (dtype, ncols, channel_names)}`` for the level's vertex attrs."""
    from zarr_vectors.constants import VERTEX_ATTRIBUTES
    from zarr_vectors.exceptions import ArrayError

    spec: dict[str, tuple[Any, int, Any]] = {}
    if VERTEX_ATTRIBUTES not in src_group:
        return spec
    for name in src_group[VERTEX_ATTRIBUTES].children():
        try:
            meta = src_group.read_array_meta(f"{VERTEX_ATTRIBUTES}/{name}")
        except ArrayError:
            continue
        channels = meta.get("channel_names")
        spec[name] = (
            np.dtype(meta.get("dtype", "float32")),
            len(channels) if channels else 1,
            channels,
        )
    return spec


def _read_level_objects(src_group, ndim: int, link_width: int, attr_spec=None):
    """Gather a level into {oid: (vertices, faces)} in object-local indices.

    Returns the per-object meshes plus the map needed to interpret face
    endpoints: ``row_of[(chunk, local_idx)] -> (oid, object_local_index)``,
    materialised as one array per chunk.  When ``attr_spec`` is given, each
    object's per-vertex attribute columns come back alongside its vertices,
    in the same row order.
    """
    from zarr_vectors.building import (
        iter_link_cells,
        list_chunk_keys,
        read_all_object_manifests,
        read_chunk_attributes,
        read_chunk_vertices,
        read_vertex_fragment_index,
    )
    from zarr_vectors.constants import VERTICES
    from zarr_vectors.exceptions import ArrayError

    attr_spec = attr_spec or {}

    manifests = read_all_object_manifests(src_group)
    owner: dict[tuple, int] = {}
    for oid, frags in enumerate(manifests):
        for cc, f in frags:
            owner.setdefault((tuple(int(x) for x in cc), int(f)), oid)

    # ---- pass 1: vertices, and the (chunk,row) -> (oid, obj row) map ----
    obj_pos: dict[int, list] = {}
    obj_attrs: dict[int, dict[str, list]] = {}
    obj_n: dict[int, int] = {}
    chunk_oid: dict[tuple, npt.NDArray] = {}
    chunk_row: dict[tuple, npt.NDArray] = {}

    for cc in sorted(tuple(int(x) for x in c)
                     for c in list_chunk_keys(src_group, VERTICES)):
        try:
            frag_index = read_vertex_fragment_index(src_group, cc)
            groups = read_chunk_vertices(src_group, cc, dtype=np.float32,
                                         ndim=ndim)
        except ArrayError:
            continue
        if not groups:
            continue
        n_rows = sum(int(np.asarray(g).shape[0]) for g in groups)
        pos = np.zeros((n_rows, ndim), np.float32)
        oid_of = np.full(n_rows, -1, np.int64)
        row_of = np.full(n_rows, -1, np.int64)
        attr_cells: dict[str, list] = {}
        attr_rows: dict[str, npt.NDArray] = {}
        for name, (dt, ncols, _cn) in attr_spec.items():
            try:
                attr_cells[name] = read_chunk_attributes(
                    src_group, name, cc, dtype=dt, ncols=ncols,
                )
            except ArrayError:
                attr_cells[name] = []
            attr_rows[name] = np.zeros(
                (n_rows,) if ncols == 1 else (n_rows, ncols), dtype=dt,
            )
        at = 0
        for f, g in enumerate(groups):
            g = np.asarray(g, np.float32)
            if g.size == 0:
                continue
            # The fragment index gives each fragment's ROW POSITIONS in the
            # chunk buffer; a face's stored local_idx addresses that buffer,
            # so placement must follow the index, not the read order.
            if frag_index.is_range(f):
                start, count = frag_index.range(f)
                idx = np.arange(int(start), int(start) + int(count))
            else:
                idx = np.asarray(frag_index.indices(f), np.int64)
            if idx.size != g.shape[0]:
                idx = np.arange(at, at + g.shape[0], dtype=np.int64)
            pos[idx] = g
            for name, (dt, _ncols, _cn) in attr_spec.items():
                cells = attr_cells[name]
                if f < len(cells):
                    block = np.asarray(cells[f], dtype=dt)
                    if block.shape[0] == idx.size:
                        attr_rows[name][idx] = block
            o = owner.get((cc, f))
            if o is not None:
                oid_of[idx] = o
                base = obj_n.get(o, 0)
                row_of[idx] = np.arange(base, base + len(idx))
                obj_pos.setdefault(o, []).append(pos[idx])
                for name in attr_spec:
                    obj_attrs.setdefault(o, {}).setdefault(
                        name, [],
                    ).append(attr_rows[name][idx])
                obj_n[o] = base + len(idx)
            at += g.shape[0]
        chunk_oid[cc] = oid_of
        chunk_row[cc] = row_of

    objects = {o: [np.concatenate(p, axis=0), None]
               for o, p in obj_pos.items() if p}
    attributes = {
        o: {
            name: np.concatenate(blocks, axis=0)
            for name, blocks in per_name.items() if blocks
        }
        for o, per_name in obj_attrs.items()
    }

    # ---- pass 2: faces, mapped into object-local index space ------------
    faces_of: dict[int, list] = {}
    for _seg, offsets, source_chunk, groups in iter_link_cells(src_group, 0):
        src_cc = tuple(int(x) for x in source_chunk)
        for g in groups:
            g = np.asarray(g, np.int64)
            if g.ndim != 2 or g.size == 0:
                continue
            perm = None
            if g.shape[1] == link_width + 1:
                # Cross-chunk rows are stored with their endpoints in
                # CANONICAL order plus a Lehmer perm_idx that restores the
                # order they were written in. Dropping the column without
                # undoing the permutation silently scrambles the corner order
                # of every cross-chunk face -- the winding inverts on some,
                # which flips the face normal, which poisons the quadrics.
                # Measured: it dragged a two-sphere test level from 99.4% of
                # true volume down to 48.8%.
                perm = g[:, 0]
                g = g[:, 1:]
            elif g.shape[1] != link_width:
                continue
            cols_o, cols_r = [], []
            bad = False
            for k in range(link_width):
                off = ((0,) * len(src_cc) if k == 0
                       else (offsets[k - 1] if k - 1 < len(offsets)
                             else (0,) * len(src_cc)))
                cc_k = tuple(int(a) + int(b) for a, b in zip(src_cc, off))
                oid_of = chunk_oid.get(cc_k)
                row_of = chunk_row.get(cc_k)
                if oid_of is None:
                    bad = True
                    break
                col = g[:, k]
                ok = (col >= 0) & (col < oid_of.size)
                o = np.full(col.shape, -1, np.int64)
                r = np.full(col.shape, -1, np.int64)
                o[ok] = oid_of[col[ok]]
                r[ok] = row_of[col[ok]]
                cols_o.append(o)
                cols_r.append(r)
            if bad:
                continue
            same = np.ones(len(g), bool)
            for k in range(1, link_width):
                same &= cols_o[k] == cols_o[0]
            same &= cols_o[0] >= 0
            for k in range(link_width):
                same &= cols_r[k] >= 0
            if not same.any():
                continue
            oids = cols_o[0][same]
            tri = np.stack([c[same] for c in cols_r], axis=1)
            if perm is not None:
                tri = _unpermute(tri, np.asarray(perm)[same], link_width)
            for o in np.unique(oids):
                faces_of.setdefault(int(o), []).append(tri[oids == o])

    for o in list(objects):
        fl = faces_of.get(o)
        objects[o][1] = (np.concatenate(fl, axis=0) if fl
                         else np.zeros((0, link_width), np.int64))
    meshes = {o: (vp, ff) for o, (vp, ff) in objects.items()}
    return meshes, attributes


# ===================================================================
# writing one level from per-object meshes
# ===================================================================

def _drop_duplicate_faces(faces: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Faces with repeated and degenerate triangles removed, in order.

    Quadric collapse can leave the same triangle twice where a fold closes
    up: on FreeSurfer's bert, 65 per hemisphere per level.  Each one is a
    face with no edge of its own, so a closed cortical sheet's Euler
    characteristic read 67 instead of 2, and a renderer draws the triangle
    twice.  A triangle is identified by its corners whatever their order,
    and the first copy, with its winding, is the one kept.
    """
    if len(faces) == 0:
        return faces
    ordered = np.sort(faces, axis=1)
    degenerate = np.any(ordered[:, 1:] == ordered[:, :-1], axis=1)
    _, first = np.unique(ordered, axis=0, return_index=True)
    keep = np.zeros(len(faces), dtype=bool)
    keep[first] = True
    keep &= ~degenerate
    return faces if keep.all() else faces[keep]


def _write_level_objects(
    root, level_group, meshes: dict[int, tuple], chunk_shape, ndim: int,
    link_width: int, n_src_objects: int, dtype_links=np.int64,
    attributes: dict[int, dict] | None = None,
    attr_spec: dict | None = None,
):
    """Re-split simplified per-object meshes onto the target chunk grid."""
    from zarr_vectors.building import (
        create_links_array,
        create_links_family,
        finalize_links,
        write_chunk_attributes,
        write_chunk_links,
        write_chunk_vertices,
        write_links,
        write_object_index,
    )

    attributes = attributes or {}
    attr_spec = attr_spec or {}

    cs = np.asarray(chunk_shape, np.float64)

    # ---- assign every vertex to a chunk, per object --------------------
    per_chunk: dict[tuple, list[tuple[int, npt.NDArray]]] = {}
    vert_chunk: dict[int, npt.NDArray] = {}
    for oid, (v, f) in sorted(meshes.items()):
        if len(v) == 0:
            continue
        cc = np.floor(np.asarray(v, np.float64) / cs).astype(np.int64)
        vert_chunk[oid] = cc
        # Group rows by chunk with np.unique rather than a dict keyed on a
        # per-vertex Python tuple: the latter is O(V) interpreter work and
        # costs ~a minute per level at 12.6M vertices.
        uniq, inv = np.unique(cc, axis=0, return_inverse=True)
        inv = np.asarray(inv).ravel()
        order = np.argsort(inv, kind="stable")
        bounds = np.flatnonzero(np.diff(inv[order])) + 1
        for u, rows in enumerate(np.split(order, bounds)):
            if rows.size:
                key = tuple(int(x) for x in uniq[inv[rows[0]]])
                per_chunk.setdefault(key, []).append((oid, rows))

    # ---- write vertices; fragment order fixes the local index space -----
    local_of: dict[int, npt.NDArray] = {
        oid: np.full(len(v), -1, np.int64) for oid, (v, f) in meshes.items()}
    chunk_of: dict[int, npt.NDArray] = {
        oid: np.zeros((len(v), ndim), np.int64) for oid, (v, f) in meshes.items()}
    manifests: dict[int, list] = {}

    for cc in sorted(per_chunk):
        members = sorted(per_chunk[cc])          # by oid: deterministic order
        blocks, base = [], 0
        attr_blocks: dict[str, list] = {name: [] for name in attr_spec}
        for frag, (oid, rows) in enumerate(members):
            v = meshes[oid][0][rows]
            blocks.append(np.asarray(v, np.float32))
            # Attribute rows are cut on exactly the same selection as the
            # vertices, so a fragment's values stay paired with its points.
            for name, (dt, ncols, _cn) in attr_spec.items():
                column = attributes.get(oid, {}).get(name)
                if column is None:
                    shape = (len(rows),) if ncols == 1 else (len(rows), ncols)
                    attr_blocks[name].append(np.zeros(shape, dtype=dt))
                else:
                    attr_blocks[name].append(
                        np.asarray(column, dtype=dt)[rows]
                    )
            local_of[oid][rows] = np.arange(base, base + len(rows))
            chunk_of[oid][rows] = np.asarray(cc, np.int64)
            manifests.setdefault(oid, []).append((cc, frag))
            base += len(rows)
        write_chunk_vertices(level_group, cc, blocks, dtype=np.float32)
        for name, (dt, _ncols, _cn) in attr_spec.items():
            write_chunk_attributes(
                level_group, name, cc, attr_blocks[name], dtype=dt,
            )

    # ---- faces: bucket by (base chunk, per-corner offsets) --------------
    pending: dict[tuple, list[npt.NDArray]] = {}
    for oid, (v, f) in sorted(meshes.items()):
        if len(f) == 0:
            continue
        f = _drop_duplicate_faces(np.asarray(f, dtype=np.int64))
        loc = local_of[oid]
        ch = chunk_of[oid]
        cor_c = [ch[f[:, k]] for k in range(link_width)]
        cor_l = [loc[f[:, k]] for k in range(link_width)]
        base = cor_c[0]
        offs = np.stack([cor_c[k] - base for k in range(1, link_width)], axis=1)
        # group rows sharing the same (base chunk, offsets) signature
        sig = np.concatenate([base] + [offs[:, k] for k in range(link_width - 1)],
                             axis=1)
        uniq, inv = np.unique(sig, axis=0, return_inverse=True)
        for u in range(len(uniq)):
            sel = inv == u
            b = tuple(int(x) for x in uniq[u, :ndim])
            ok = tuple(tuple(int(x) for x in uniq[u, ndim * (k + 1):ndim * (k + 2)])
                       for k in range(link_width - 1))
            arr = np.stack([c[sel] for c in cor_l], axis=1)
            pending.setdefault((b, ok), []).append(arr)

    create_links_family(level_group, delta=0, link_width=link_width,
                        sid_ndim=ndim)
    zero = (0,) * ndim
    n_faces = 0

    cross: list[list[tuple]] = []
    for (b, ok), arrs in sorted(pending.items()):
        if all(o == zero for o in ok):
            continue
        stacked = np.concatenate(arrs, axis=0)
        n_faces += len(stacked)
        ends = [tuple(b)] + [tuple(int(x) + int(y) for x, y in zip(b, o))
                             for o in ok]
        cross.extend([(ends[k], int(row[k])) for k in range(link_width)]
                     for row in stacked)
    if cross:
        write_links(level_group, cross, ndim, delta=0, link_width=link_width)

    for (b, ok), arrs in sorted(pending.items()):
        if not all(o == zero for o in ok):
            continue
        stacked = np.concatenate(arrs, axis=0)
        n_faces += len(stacked)
        create_links_array(level_group, link_width, delta=0, sid_ndim=ndim,
                           offsets=[zero] * (link_width - 1))
        write_chunk_links(level_group, b, [stacked], dtype=dtype_links,
                          delta=0, link_width=link_width)
    finalize_links(level_group, delta=0)

    write_object_index(level_group, manifests, sid_ndim=ndim,
                       total_objects=n_src_objects)
    return n_faces, manifests


def _resample_attributes(
    source_vertices: npt.NDArray,
    new_vertices: npt.NDArray,
    source_attributes: dict | None,
    attr_spec: dict,
) -> dict:
    """Carry per-vertex attributes onto a simplified mesh.

    An edge collapse MOVES vertices -- the survivor sits at the quadric
    optimum, not on any input vertex -- so there is no index map to follow
    and the honest carry is the value of the nearest source vertex.  That is
    what ``meshes._quadric_pyfqmr`` already does, and it keeps a categorical
    label a value that genuinely occurred rather than a blend of two.

    Falls back to leaving the attributes off when SciPy is absent, rather
    than failing a whole pyramid over it.
    """
    if not attr_spec or not source_attributes or len(new_vertices) == 0:
        return {}
    try:
        from scipy.spatial import cKDTree
    except ImportError:  # pragma: no cover - the mesh extra pulls scipy
        return {}
    tree = cKDTree(np.asarray(source_vertices, dtype=np.float64))
    _dist, nearest = tree.query(np.asarray(new_vertices, dtype=np.float64))
    nearest = np.asarray(nearest, dtype=np.int64)
    out: dict[str, npt.NDArray] = {}
    for name, (dt, _ncols, _cn) in attr_spec.items():
        column = source_attributes.get(name)
        if column is None:
            continue
        values = np.asarray(column)
        if len(values) == 0:
            continue
        out[name] = values[np.clip(nearest, 0, len(values) - 1)].astype(dt)
    return out


# ===================================================================
# the strategy
# ===================================================================

def coarsen_mesh_decimate_level(
    store_path,
    source_level: int,
    target_level: int,
    *,
    coarsen_factor: float = 4.0,
    size_factor: float | None = None,
    chunk_scale_factor=1,
    sparsity_factor: float = 1.0,
    compressor: Any = None,
    max_normal_turn: float = 100.0,
    max_passes: int = 200,
    min_component_faces: int = 0,
    object_floors: dict | None = None,
    min_faces: int = 4,
    verbose: bool = False,
    **_ignored: Any,
) -> dict[str, Any]:
    """Coarsen one mesh level by per-object quadric edge collapse.

    ``size_factor`` (or ``coarsen_factor`` if it is not given) is the desired
    reduction in ON-DISK DATA SIZE for the level as a whole. It is converted
    to a per-object face target through :func:`predict_bytes`, which is linear
    and measured accurate to under a percent, so the requested factor is hit
    directly rather than searched for.
    """
    from contextlib import nullcontext

    from zarr_vectors.building import (
        LevelMetadata,
        create_attribute_array,
        create_object_index_array,
        create_resolution_level,
        create_vertices_array,
        get_level_chunk_shape,
        get_resolution_level,
        link_family_policy,
        open_store,
        read_all_object_manifests,
        read_level_metadata,
        read_root_metadata,
    )

    from ..groupings import propagate_groupings, surviving_oids_from

    k = float(size_factor if size_factor is not None else coarsen_factor)
    if k <= 1.0:
        raise ValueError(f"reduction factor must exceed 1, got {k}")
    # This strategy simplifies every object and drops none.  Accepting a
    # sparsity factor and stamping ``object_sparsity = 1 / factor`` on the
    # level -- which it did -- tells a reader that half the cells are gone
    # while all of them are present, and the object count in the summary
    # contradicts it.  Refuse rather than mislead; ``method="mesh"`` (vertex
    # clustering) is the mesh strategy that does apply object selection.
    if float(sparsity_factor) > 1.0:
        raise ValueError(
            f"sparsity_factor={sparsity_factor} is not supported by the "
            f"quadric mesh coarsener: it simplifies every object and drops "
            f"none. Use method='mesh' for object-dropping mesh levels, or "
            f"leave sparsity_factor at 1.0."
        )

    root = open_store(str(store_path), mode="r+")
    root_meta = read_root_metadata(root)
    ndim = len(root_meta.chunk_shape)
    src_group = get_resolution_level(root, source_level)
    try:
        src_lm = read_level_metadata(root, source_level)
    except Exception:  # noqa: BLE001
        src_lm = None
    try:
        link_width = int(link_family_policy(src_group, 0).get("link_width", 3))
    except Exception:  # noqa: BLE001
        link_width = 3

    # Per-vertex attributes come back with the meshes and are resampled onto
    # the simplified vertices below.  Without this a quadric level keeps the
    # surface and loses every scalar on it, which for a cortical surface is
    # the whole payload.
    attr_spec = _vertex_attribute_spec(src_group)
    meshes, src_attributes = _read_level_objects(
        src_group, ndim, link_width, attr_spec,
    )
    n_src_objects = len(read_all_object_manifests(src_group))

    v_in = sum(len(v) for v, f in meshes.values())
    f_in = sum(len(f) for v, f in meshes.values())
    if f_in == 0:
        raise ValueError(f"level {source_level} has no faces to coarsen")

    # ---- turn the size factor into per-object face targets --------------
    # Budget is shared proportionally to each object's current face count, so
    # a small object is not driven below the floor to pay for a large one.
    budget = predict_bytes(v_in, f_in) / k
    ratio = v_in / f_in
    total_target = faces_for_byte_budget(budget, ratio)
    share = total_target / f_in

    out: dict[int, tuple] = {}
    out_attributes: dict[int, dict] = {}
    stats = []
    floors = 0
    budget_bound = 0
    at_own_floor = 0
    for oid, (v, f) in sorted(meshes.items()):
        if len(f) == 0:
            continue
        tgt = max(min_faces, int(round(len(f) * share)))
        # Each cell has its OWN tube floor -- across the 1720-cell corpus the
        # per-cell floor runs from a median 42,980 faces to 89,107 at p95, so a
        # single global floor would over-decimate the long arbours and
        # under-decimate the short ones. A cell already at or below its floor
        # is passed through untouched rather than ground down further.
        if object_floors:
            fl = int(object_floors.get(oid, 0))
            if fl > 0:
                if len(f) <= fl:
                    out[oid] = (np.asarray(v, np.float32), np.asarray(f, np.int64))
                    # Passed through untouched, so its attributes are too.
                    out_attributes[oid] = src_attributes.get(oid, {})
                    at_own_floor += 1
                    continue
                tgt = max(tgt, fl)
        r = decimate_mesh(v, f, target_faces=tgt,
                          max_normal_turn=max_normal_turn, do_weld=True,
                          max_passes=max_passes,
                          min_component_faces=min_component_faces)
        out[oid] = (r["vertices"], r["faces"])
        out_attributes[oid] = _resample_attributes(
            v, r["vertices"], src_attributes.get(oid), attr_spec,
        )
        floors += int(r["hit_floor"] and not r["target_met"])
        budget_bound += int(r.get("pass_budget_bound", False))
        stats.append((oid, len(f), r["face_count"], r["hit_floor"]))
        if verbose:
            print(f"  obj {oid}: {len(f):,} -> {r['face_count']:,} faces"
                  f"{' (floor)' if r['hit_floor'] else ''}", flush=True)

    # ---- level metadata --------------------------------------------------
    if isinstance(chunk_scale_factor, (list, tuple)):
        scale = tuple(int(x) for x in chunk_scale_factor)
    else:
        scale = (int(chunk_scale_factor),) * ndim
    src_chunk_shape = get_level_chunk_shape(root_meta, src_lm)
    target_chunk_shape = tuple(float(c) * float(s)
                               for c, s in zip(src_chunk_shape, scale))
    # Cap the growth at the extent of the data. Doubling the chunk every level
    # suits a corpus that spans many chunks; on a single cell it runs away --
    # the test neuron is 0.66 mm across while level 4 and 5 chunks reached
    # 2.05 mm and 4.10 mm, so each coarse level became ONE chunk holding the
    # whole object, which no viewer can cull spatially and which defeats the
    # store's own spatial index.
    # The cap must land on an integer multiple of the ROOT chunk shape --
    # create_resolution_level rejects a level whose chunk_shape is not one
    # ("nested chunk grids are required"). Clamping straight to the data extent
    # produced e.g. 1000 against a root of 600 and raised MetadataError, so the
    # cap is rounded down to a whole multiple, floored at 1x.
    lo, hi = root_meta.bounds
    extent = [float(b) - float(a) for a, b in zip(lo, hi)]
    capped = []
    for c, e, r in zip(target_chunk_shape, extent, root_meta.chunk_shape):
        r = float(r)
        mult_target = max(1, int(round(c / r)))
        mult_cap = max(1, int(np.floor(max(e, r) / r)))
        capped.append(r * min(mult_target, mult_cap))
    target_chunk_shape = tuple(capped)
    root_bin = tuple(float(b) for b in root_meta.effective_bin_shape)
    src_bin = getattr(src_lm, "bin_shape", None) if src_lm else None
    source_bin = tuple(float(b) for b in src_bin) if src_bin else root_bin
    # The geometric scale really applied is the linear one implied by the area
    # reduction, sqrt(k) -- recorded so the NGFF transform is not the ~340x
    # misstatement the clustering strategy leaves behind.
    target_bin = tuple(b * float(np.sqrt(k)) for b in source_bin)

    v_out = sum(len(v) for v, f in out.values())
    f_out = sum(len(f) for v, f in out.values())

    level_meta = LevelMetadata(
        level=target_level,
        vertex_count=v_out,
        arrays_present=(
            ["vertices", "links", "object_index"]
            + (["vertex_attributes"] if attr_spec else [])
        ),
        bin_shape=target_bin,
        bin_ratio=tuple(max(1, int(round(t / r)))
                        for t, r in zip(target_bin, root_bin)),
        chunk_shape=(target_chunk_shape if any(s != 1 for s in scale) else None),
        object_sparsity=1.0 / float(sparsity_factor),
        coarsening_method=COARSEN_MESH_DECIMATE,
        parent_level=source_level,
        preserves_object_ids=True,
        inherited_num_objects=n_src_objects,
        shared_fragments=False,
    )
    level_group = create_resolution_level(root, target_level, level_meta)
    ctx = (level_group.batched_writes(compressor=compressor)
           if compressor else nullcontext())
    with ctx:
        create_vertices_array(level_group, dtype="float32")
        create_object_index_array(level_group)
        for name, (dt, _ncols, channels) in attr_spec.items():
            create_attribute_array(
                level_group, name, dtype=str(dt), channel_names=channels,
            )

    written, manifests = _write_level_objects(
        root, level_group, out, target_chunk_shape, ndim, link_width,
        n_src_objects, attributes=out_attributes, attr_spec=attr_spec)
    propagate_groupings(src_group, level_group,
                        surviving_oids=surviving_oids_from(
                            sorted(out), sparsity_factor))

    achieved = predict_bytes(v_in, f_in) / max(predict_bytes(v_out, f_out), 1)
    return {
        "vertex_count": v_out,
        "face_count": f_out,
        "faces_written": written,
        "vertices_in": v_in,
        "faces_in": f_in,
        "object_count": len(out),
        "objects_at_floor": floors,
        "objects_at_own_tube_floor": at_own_floor,
        "objects_pass_budget_bound": budget_bound,
        "requested_size_factor": k,
        "achieved_size_factor": round(achieved, 3),
        "bytes_in": int(predict_bytes(v_in, f_in)),
        "bytes_out": int(predict_bytes(v_out, f_out)),
        "method": COARSEN_MESH_DECIMATE,
        "preserves_object_ids": True,
        "shared_fragments": False,
        "attributes_carried": sorted(attr_spec),
    }


def build_mesh_pyramid_to_floor(
    store_path,
    size_factor: float = 4.0,
    *,
    max_levels: int = 12,
    min_faces: int = 256,
    tolerance: float = 0.75,
    chunk_scale_factor=2,
    max_normal_turn: float = 100.0,
    min_component_faces: int = 0,
    tube_floor: int | None = None,
    object_floors: dict | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Add levels at ``size_factor``x data-size reduction each, until the floor.

    "The floor" is where the surface can no longer be reduced by the requested
    factor and still be a mesh. Three things stop the ladder, and which one
    fired is reported per level:

    ``target_missed``  the level came out more than ``tolerance`` short of the
        requested factor. Every remaining edge collapse is refused by the link
        condition or the normal-flip guard, so the geometry is as coarse as it
        can be without pinching a tube shut or folding a face over. This is the
        real floor and normally the one that fires.
    ``min_faces``      the absolute face budget is exhausted.
    ``not_a_mesh``     the level failed :func:`mesh_floor_report` -- an orphan
        appeared, or a component fell below the 4 faces a closed surface needs.
        This should never fire; it is a backstop against a bug, not a design
        stop, and it deletes the offending level rather than publishing it.

    ``tube_floor``    the next level would fall below the face count at which
        the geometry can still be a surface at all -- see
        :func:`tube_floor_faces`. Checked BEFORE the level is built, so a level
        that cannot look right is never written rather than written and judged.
        This is the stop that matters for thin neurites: the size-factor tests
        below are blind to quality and happily emitted a level at 3.99x whose
        axon had disintegrated into 5:1 slivers.

    The level that trips a stop is kept if it is still a valid mesh (it is the
    coarsest honest representation), and discarded only in the ``not_a_mesh``
    case.
    """
    from zarr_vectors.building import (
        list_resolution_levels,
        open_store,
        read_level_metadata,
        remove_resolution_level,
    )

    root = open_store(str(store_path), mode="r+")
    levels = sorted(list_resolution_levels(root))
    src = max(levels)
    reports: list[dict[str, Any]] = []

    for step in range(max_levels):
        tgt = src + 1
        # Refuse to build a level that would breach the tube floor. Predicting
        # the face count is exact enough for this: the byte model is linear and
        # measured accurate to under a percent, so the next level lands at
        # ~1/size_factor of the current one.
        if tube_floor is not None:
            try:
                cur_lm = read_level_metadata(
                    open_store(str(store_path), mode="r"), src)
                cur_faces = int(getattr(cur_lm, "vertex_count", 0) * 2)
            except Exception:  # noqa: BLE001
                cur_faces = 0
            projected = cur_faces / float(size_factor)
            if cur_faces and projected < tube_floor:
                reports.append({"level": tgt, "stop": "tube_floor",
                                "projected_faces": int(projected),
                                "tube_floor": int(tube_floor),
                                "not_built": True})
                if verbose:
                    print(f"  level {tgt}: NOT BUILT -- projected "
                          f"{int(projected):,} faces is below the tube floor of "
                          f"{int(tube_floor):,}; level {src} is the coarsest "
                          f"level this geometry supports", flush=True)
                break
        if tgt in list_resolution_levels(open_store(str(store_path), mode="r")):
            remove_resolution_level(open_store(str(store_path), mode="r+"), tgt)
        try:
            r = coarsen_mesh_decimate_level(
                store_path, src, tgt,
                size_factor=size_factor,
                chunk_scale_factor=chunk_scale_factor,
                max_normal_turn=max_normal_turn,
                min_component_faces=min_component_faces,
                object_floors=object_floors,
            )
        except Exception as e:  # noqa: BLE001
            reports.append({"level": tgt, "stop": "error", "error": repr(e)})
            break

        r["level"] = tgt
        achieved = r["achieved_size_factor"]
        r["stop"] = None

        n_obj = max(int(r.get("object_count", 1)), 1)
        if r.get("objects_at_own_tube_floor", 0) >= 0.9 * n_obj:
            r["stop"] = "tube_floor"
        elif r["face_count"] < min_faces:
            r["stop"] = "min_faces"
        elif achieved < size_factor * tolerance:
            r["stop"] = "target_missed"

        reports.append(r)
        if verbose:
            print(f"  level {tgt}: {r['faces_in']:>10,} -> {r['face_count']:>9,} faces "
                  f"({r['vertex_count']:>9,} v)  {achieved:5.2f}x of {size_factor}x "
                  f"requested  {r['bytes_out'] / 1e6:8.1f} MB"
                  + (f"  STOP: {r['stop']}" if r["stop"] else ""), flush=True)

        if r["stop"]:
            break
        src = tgt

    return reports


def _mesh_decimate_coarsener(store_path, source_level, target_level, **kw):
    """Registry adapter."""
    return coarsen_mesh_decimate_level(
        store_path, source_level, target_level,
        coarsen_factor=kw.get("coarsen_factor", 4.0),
        size_factor=kw.get("size_factor"),
        chunk_scale_factor=kw.get("chunk_scale_factor", 1),
        sparsity_factor=kw.get("sparsity_factor", 1.0),
        compressor=kw.get("compressor"),
        max_normal_turn=kw.get("max_normal_turn", 100.0),
        max_passes=kw.get("max_passes", 200),
        min_component_faces=kw.get("min_component_faces", 0),
        object_floors=kw.get("object_floors"),
        verbose=kw.get("verbose", False),
    )
