"""Competitor single-file formats, implemented to be as fast as the
format allows.

Design rule for this module: **never hand zarr-vectors an easy win.**
Every reader here is vectorised (``np.frombuffer`` / ``pandas`` C parser,
never a Python loop over lines), and every single-object replace uses the
cheapest operation the format's on-disk structure permits -- an in-place
seek-and-overwrite for the fixed-width binary formats, rather than the
read-modify-rewrite an off-the-shelf tool would do.  Where a format
genuinely cannot do better than a full rewrite, that is a property of the
format and is documented at the function.

Formats
-------
=========== ========= =============== ==================================
Format      Encoding  Geometry        Single-object replace
=========== ========= =============== ==================================
PLY         binary    point cloud     in-place seek + 12-byte write
STL         binary    mesh            in-place seek + 50 B/triangle run
TRK         binary    streamlines     full rewrite (variable-length records)
TRX         zip       streamlines     n/a -- used for partial *reads* only
CSV         text      points, graph   size comparison only
OBJ         text      mesh            size comparison only
SWC         text      skeletons       size comparison only
GraphML     text      graph           size comparison only
=========== ========= =============== ==================================
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

# ====================================================================
# PLY -- binary little-endian point cloud
# ====================================================================

_PLY_HEADER = (
    "ply\n"
    "format binary_little_endian 1.0\n"
    "element vertex {n}\n"
    "property float x\n"
    "property float y\n"
    "property float z\n"
    "end_header\n"
)
PLY_ITEMSIZE = 12  # 3 x float32


def ply_write(path, positions) -> None:
    pos = np.ascontiguousarray(positions, dtype="<f4")
    header = _PLY_HEADER.format(n=len(pos)).encode("ascii")
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(pos.tobytes())


def _ply_data_offset(path) -> int:
    """Byte offset of the first vertex record."""
    with open(path, "rb") as fh:
        head = fh.read(4096)
    return head.index(b"end_header\n") + len(b"end_header\n")


def ply_read_full(path):
    """Read every vertex into a materialised, writable ``(n, 3)`` array.

    ``np.frombuffer`` alone would hand back a read-only view over the file
    buffer, which is faster than anything zarr-vectors can return but is
    not the same product: the zarr-vectors readers return owned arrays.
    The copy keeps the two sides comparable and costs a few milliseconds
    even at a million vertices."""
    off = _ply_data_offset(path)
    buf = Path(path).read_bytes()
    return np.array(np.frombuffer(buf, dtype="<f4", offset=off).reshape(-1, 3))


def ply_read_objects(path, object_ids):
    """PLY has no index: the file is read, then rows are taken by position."""
    return ply_read_full(path)[np.asarray(object_ids, dtype=np.int64)]


def ply_read_bbox(path, low, high):
    pts = ply_read_full(path)
    mask = np.all((pts >= np.asarray(low)) & (pts <= np.asarray(high)), axis=1)
    return pts[mask]


def ply_replace_object(path, index, new_xyz, data_offset=None):
    """Best case for a fixed-width binary format: seek to the record and
    overwrite 12 bytes.  Assumes the caller already knows the row index,
    which is the most generous assumption available to PLY."""
    off = _ply_data_offset(path) if data_offset is None else data_offset
    payload = np.asarray(new_xyz, dtype="<f4").tobytes()
    with open(path, "r+b") as fh:
        fh.seek(off + int(index) * PLY_ITEMSIZE)
        fh.write(payload)


# ====================================================================
# STL -- binary triangle soup (mesh)
# ====================================================================

STL_HEADER_BYTES = 84                     # 80-byte comment + uint32 count
STL_TRI = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
STL_ITEMSIZE = STL_TRI.itemsize           # 50


def stl_write(path, vertices, faces) -> None:
    tris = np.zeros(len(faces), dtype=STL_TRI)
    v = np.asarray(vertices, dtype="<f4")[np.asarray(faces)]
    tris["v"] = v
    e1, e2 = v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]
    nrm = np.cross(e1, e2)
    norm = np.linalg.norm(nrm, axis=1, keepdims=True)
    tris["n"] = np.divide(nrm, norm, out=np.zeros_like(nrm), where=norm > 0)
    with open(path, "wb") as fh:
        fh.write(b"\0" * 80)
        fh.write(struct.pack("<I", len(faces)))
        fh.write(tris.tobytes())


def stl_read_full(path):
    """Read every triangle into a materialised ``(n_tri, 3, 3)`` array.

    ``tris["v"]`` is a 50-byte-strided view into the record buffer; it is
    made contiguous for the same reason as in :func:`ply_read_full`."""
    buf = Path(path).read_bytes()
    tris = np.frombuffer(buf, dtype=STL_TRI, offset=STL_HEADER_BYTES)
    return np.ascontiguousarray(tris["v"])


def stl_read_objects(path, tri_slices):
    """STL carries no object index -- the file is read, then the triangle
    ranges belonging to the requested objects are taken by position."""
    tris = stl_read_full(path)
    return [tris[s0:s1] for s0, s1 in tri_slices]


def stl_read_bbox(path, low, high):
    tris = stl_read_full(path)
    centroid = tris.mean(axis=1)
    mask = np.all((centroid >= np.asarray(low))
                  & (centroid <= np.asarray(high)), axis=1)
    return tris[mask]


def stl_replace_object(path, tri_start, new_tri_vertices):
    """In-place overwrite of one object's contiguous triangle run.  Only
    possible because our writer emits each patch contiguously and the
    caller knows the run's start index -- the most generous assumption
    available to STL."""
    v = np.asarray(new_tri_vertices, dtype="<f4")
    tris = np.zeros(len(v), dtype=STL_TRI)
    tris["v"] = v
    e1, e2 = v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]
    nrm = np.cross(e1, e2)
    norm = np.linalg.norm(nrm, axis=1, keepdims=True)
    tris["n"] = np.divide(nrm, norm, out=np.zeros_like(nrm), where=norm > 0)
    with open(path, "r+b") as fh:
        fh.seek(STL_HEADER_BYTES + int(tri_start) * STL_ITEMSIZE)
        fh.write(tris.tobytes())


# ====================================================================
# TRK -- TrackVis streamlines (nibabel)
# ====================================================================

def _trk_header(streamlines):
    import numpy as _np
    from nibabel.streamlines import Field
    affine = _np.eye(4, dtype=_np.float32)
    pts = _np.concatenate(streamlines, axis=0)
    dims = _np.clip(_np.ceil(_np.abs(pts).max(axis=0)).astype(_np.int64) + 1,
                    1, 32767).astype(_np.int16)
    return {
        Field.VOXEL_SIZES: (1.0, 1.0, 1.0),
        Field.DIMENSIONS: tuple(int(d) for d in dims),
        Field.VOXEL_TO_RASMM: affine,
    }, affine


def trk_write(path, streamlines) -> None:
    import nibabel as nib
    from nibabel.streamlines.trk import TrkFile
    header, affine = _trk_header(streamlines)
    tractogram = nib.streamlines.Tractogram(streamlines=list(streamlines),
                                            affine_to_rasmm=affine)
    TrkFile(tractogram=tractogram, header=header).save(str(path))


def trk_read_full(path):
    """Read every streamline.

    Returns the flat point buffer plus per-streamline lengths rather than
    a Python list of per-streamline arrays: nibabel already holds the data
    in that form, and materialising a list would charge TRK for a Python
    loop the format does not require."""
    import nibabel as nib
    sl = nib.streamlines.load(str(path)).streamlines
    return sl.get_data(), np.asarray(sl._lengths, dtype=np.int64)


def trk_read_objects(path, object_ids):
    """TRK records are variable length and unindexed, so reaching the k-th
    streamline means walking every record before it.  nibabel's loader
    does exactly that."""
    import nibabel as nib
    sl = nib.streamlines.load(str(path)).streamlines
    return [np.asarray(sl[int(i)]) for i in object_ids]


def trk_read_bbox(path, low, high):
    """Whole-file read, then a vectorised per-streamline "any vertex inside"
    test over the flat point buffer -- no Python loop over streamlines, so
    the cost measured is the format's (read everything) and not the
    baseline implementation's."""
    import nibabel as nib
    sl = nib.streamlines.load(str(path)).streamlines
    data = sl.get_data()
    lengths = np.asarray(sl._lengths, dtype=np.int64)
    offsets = np.asarray(sl._offsets, dtype=np.int64)
    inside = np.all((data >= np.asarray(low)) & (data <= np.asarray(high)),
                    axis=1)
    csum = np.concatenate([[0], np.cumsum(inside)])
    hit = np.flatnonzero(csum[offsets + lengths] - csum[offsets] > 0)
    return [np.asarray(sl[int(i)]) for i in hit]


def trk_replace_object(path, index, new_vertices):
    """Full read-modify-rewrite.  A variable-length record cannot be
    overwritten in place unless the replacement has exactly the same point
    count, and even then locating its byte offset requires walking every
    preceding record -- so the file is O(N) either way."""
    import nibabel as nib
    from nibabel.streamlines.trk import TrkFile
    sl = list(nib.streamlines.load(str(path)).streamlines)
    sl[int(index)] = np.asarray(new_vertices, dtype=np.float32)
    header, affine = _trk_header(sl)
    tractogram = nib.streamlines.Tractogram(streamlines=sl,
                                            affine_to_rasmm=affine)
    TrkFile(tractogram=tractogram, header=header).save(str(path))


# ====================================================================
# TRX -- the only competitor with a native partial read
# ====================================================================

def trx_write(path, streamlines, *, deflate: bool = False,
              positions_dtype=np.float32) -> None:
    """Write a TRX archive.

    TRX defaults to float32 members written STORED, because its whole
    point is that a reader can memory-map them -- which is also why the
    default costs exactly what TRK costs.  It will happily deflate
    instead (still lossless, no longer mappable), and it will store
    float16 positions (lossy, ~0.25 units of error at this domain size).
    Both are the format's own options, so the lossless one belongs in any
    honest size comparison against a compressed store.
    """
    import zipfile

    import nibabel as nib
    from nibabel.streamlines import Tractogram
    from trx.trx_file_memmap import TrxFile
    from trx.trx_file_memmap import save as trx_save
    ref = nib.Nifti1Image(np.zeros((1, 1, 1), dtype=np.uint8), affine=np.eye(4))
    tg = Tractogram(streamlines=list(streamlines), affine_to_rasmm=np.eye(4))
    obj = TrxFile.from_tractogram(tg, reference=ref, dtype_dict={
        "positions": positions_dtype, "offsets": np.uint32,
        "dpv": {}, "dps": {},
    })
    trx_save(obj, str(path), compression_standard=(
        zipfile.ZIP_DEFLATED if deflate else zipfile.ZIP_STORED))


def trx_read_full(path):
    from trx.trx_file_memmap import load as trx_load
    return [np.asarray(s) for s in trx_load(str(path)).streamlines]


def trx_read_objects(path, object_ids):
    from trx.trx_file_memmap import load as trx_load
    obj = trx_load(str(path))
    return [np.asarray(s) for s in obj.select(list(object_ids)).streamlines]


# ====================================================================
# Text formats -- size comparison only
# ====================================================================

def csv_points_write(path, positions) -> None:
    import pandas as pd
    pd.DataFrame(np.asarray(positions), columns=["x", "y", "z"]).to_csv(
        path, index=False, float_format="%.6g")


def csv_graph_write(nodes_path, edges_path, positions, edges) -> None:
    import pandas as pd
    pos = np.asarray(positions)
    pd.DataFrame({"node_id": np.arange(len(pos)),
                  "x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2]}).to_csv(
        nodes_path, index=False, float_format="%.6g")
    e = np.asarray(edges)
    pd.DataFrame({"source": e[:, 0], "target": e[:, 1]}).to_csv(
        edges_path, index=False)


def graphml_write(path, positions, edges) -> None:
    import networkx as nx
    g = nx.Graph()
    pos = np.asarray(positions)
    for i, p in enumerate(pos):
        g.add_node(int(i), x=float(p[0]), y=float(p[1]), z=float(p[2]))
    g.add_edges_from((int(s), int(t)) for s, t in np.asarray(edges))
    nx.write_graphml(g, str(path))


def obj_write(path, vertices, faces) -> None:
    """Vectorised OBJ writer -- ``np.savetxt`` on two blocks, no per-line
    Python formatting."""
    v = np.asarray(vertices)
    f = np.asarray(faces) + 1
    with open(path, "wb") as fh:
        np.savetxt(fh, v, fmt="v %.6f %.6f %.6f")
        np.savetxt(fh, f, fmt="f %d %d %d")


def swc_write(path, polylines, radii) -> None:
    """SWC for a set of unbranched chains: one row per vertex, carrying an
    explicit sample id and parent id -- the cost the implicit sequential
    link avoids."""
    rows = []
    sample = 1
    for poly, rad in zip(polylines, radii):
        n = len(poly)
        ids = np.arange(sample, sample + n)
        parents = ids - 1
        parents[0] = -1
        block = np.empty((n, 7))
        block[:, 0] = ids
        block[:, 1] = 2                       # structure identifier: axon
        block[:, 2:5] = poly
        block[:, 5] = rad
        block[:, 6] = parents
        rows.append(block)
        sample += n
    arr = np.concatenate(rows) if rows else np.empty((0, 7))
    with open(path, "wb") as fh:
        fh.write(b"# id type x y z radius parent\n")
        np.savetxt(fh, arr, fmt="%d %d %.6f %.6f %.6f %.4f %d")


# ====================================================================
# Neuroglancer precomputed annotations
# ====================================================================
#
# The closest competitor in kind: precomputed is itself a chunked spatial
# format, so unlike PLY or TRK it can answer a bounding-box query without
# reading everything.  What it cannot do is share storage between the two
# access patterns -- the spatial index and the by-id index are separate
# copies of the same annotations, which is where its size goes.
#
# Layout (https://github.com/google/neuroglancer, annotations spec):
#
#   info                 JSON: dimensions, bounds, one spatial level, by_id
#   spatial0/<x>_<y>_<z> multiple-annotation encoding, one file per chunk
#   by_id/<id>           single-annotation encoding, one file per annotation
#
# Multiple-annotation encoding: uint64 count, then count packed geometries,
# then count uint64 ids.  Single-annotation encoding: the geometry alone
# (relationships would follow, and there are none here).

_PC_INFO = "info"
_PC_SPATIAL = "spatial0"
_PC_BY_ID = "by_id"


def precomputed_write(root, positions, chunk_shape, bounds, *, with_by_id=True):
    """Write a POINT annotation store.

    ``with_by_id=False`` writes the spatial index alone, which is what a
    viewer needs to render but not to resolve an annotation id -- the two
    are measured separately because the second copy is most of the cost.
    """
    import json

    root = Path(root)
    (root / _PC_SPATIAL).mkdir(parents=True, exist_ok=True)
    pos = np.ascontiguousarray(positions, dtype="<f4")
    lower, upper = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    cs = np.asarray(chunk_shape, float)
    grid = np.ceil((upper - lower) / cs).astype(np.int64)

    (root / _PC_INFO).write_text(json.dumps({
        "@type": "neuroglancer_annotations_v1",
        "dimensions": {d: [1e-9, "m"] for d in "xyz"},
        "lower_bound": lower.tolist(),
        "upper_bound": upper.tolist(),
        "annotation_type": "POINT",
        "properties": [],
        "relationships": [],
        "by_id": {"key": _PC_BY_ID},
        "spatial": [{
            "key": _PC_SPATIAL,
            "grid_shape": grid.tolist(),
            "chunk_size": cs.tolist(),
            "limit": int(len(pos)),
        }],
    }))

    cell = np.clip(((pos - lower) // cs).astype(np.int64), 0, grid - 1)
    flat = (cell[:, 0] * grid[1] + cell[:, 1]) * grid[2] + cell[:, 2]
    order = np.argsort(flat, kind="stable")
    flat_s, pos_s, ids_s = flat[order], pos[order], order.astype("<u8")
    starts = np.flatnonzero(np.r_[True, flat_s[1:] != flat_s[:-1]])
    for b, s in enumerate(starts):
        e = starts[b + 1] if b + 1 < len(starts) else len(flat_s)
        cc = cell[order[s]]
        key = "_".join(str(int(c)) for c in cc)
        with open(root / _PC_SPATIAL / key, "wb") as fh:
            fh.write(struct.pack("<Q", int(e - s)))
            fh.write(pos_s[s:e].tobytes())
            fh.write(ids_s[s:e].tobytes())

    if with_by_id:
        by_id = root / _PC_BY_ID
        by_id.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(pos):
            (by_id / str(i)).write_bytes(p.tobytes())


def _pc_grid(root):
    import json
    info = json.loads((Path(root) / _PC_INFO).read_text())
    spatial = info["spatial"][0]
    return (np.asarray(info["lower_bound"], float),
            np.asarray(spatial["chunk_size"], float),
            np.asarray(spatial["grid_shape"], np.int64))


def _pc_decode(blob):
    n = struct.unpack_from("<Q", blob, 0)[0]
    pts = np.frombuffer(blob, dtype="<f4", offset=8, count=3 * n).reshape(n, 3)
    return np.array(pts)


def precomputed_read_full(root):
    d = Path(root) / _PC_SPATIAL
    out = [_pc_decode(f.read_bytes()) for f in sorted(d.iterdir()) if f.is_file()]
    return np.concatenate(out) if out else np.zeros((0, 3), np.float32)


def precomputed_read_one(root, annotation_id):
    """One file, one read -- precomputed's by-id index is O(1) by design."""
    blob = (Path(root) / _PC_BY_ID / str(int(annotation_id))).read_bytes()
    return np.frombuffer(blob, dtype="<f4", count=3)


def precomputed_read_bbox(root, low, high):
    """Read only the spatial chunks the box touches, then filter."""
    lower, cs, grid = _pc_grid(root)
    lo = np.clip(((np.asarray(low, float) - lower) // cs).astype(np.int64),
                 0, grid - 1)
    hi = np.clip(((np.asarray(high, float) - lower) // cs).astype(np.int64),
                 0, grid - 1)
    d = Path(root) / _PC_SPATIAL
    out = []
    for i in range(lo[0], hi[0] + 1):
        for j in range(lo[1], hi[1] + 1):
            for k in range(lo[2], hi[2] + 1):
                f = d / f"{i}_{j}_{k}"
                if f.exists():
                    out.append(_pc_decode(f.read_bytes()))
    if not out:
        return np.zeros((0, 3), np.float32)
    pts = np.concatenate(out)
    mask = np.all((pts >= np.asarray(low)) & (pts <= np.asarray(high)), axis=1)
    return pts[mask]


# ====================================================================
# Other chunked formats
# ====================================================================
#
# Precomputed is one way to chunk points; these are the others a reader
# might reasonably reach for.  All three are given the same treatment:
# points are sorted into the same spatial grid zarr-vectors uses, so the
# comparison is between *layouts over the same partition*, not between
# partitions.  Each then answers a bounding box by touching only the
# cells it intersects, which is exactly the capability being claimed.
#
# Plain Zarr is the control that matters most.  It is the same library,
# the same codec and the same chunk grid with none of the zarr-vectors
# layout on top, so the gap between it and the store is the cost of the
# fragment index, object index and manifest machinery -- not of chunking.


def spatial_cells(positions, chunk_shape, bounds):
    """Sort points into spatial cells.

    Returns ``(order, sorted_positions, cell_of_row, cell_index)`` where
    ``cell_index`` maps a cell tuple to its ``(start, stop)`` row range in
    the sorted array.
    """
    pos = np.asarray(positions, dtype="<f4")
    lower = np.asarray(bounds[0], float)
    cs = np.asarray(chunk_shape, float)
    grid = np.ceil((np.asarray(bounds[1], float) - lower) / cs).astype(np.int64)
    cell = np.clip(((pos - lower) // cs).astype(np.int64), 0, grid - 1)
    flat = (cell[:, 0] * grid[1] + cell[:, 1]) * grid[2] + cell[:, 2]
    order = np.argsort(flat, kind="stable")
    flat_s = flat[order]
    starts = np.flatnonzero(np.r_[True, flat_s[1:] != flat_s[:-1]])
    stops = np.r_[starts[1:], len(flat_s)]
    index = {}
    for s, e in zip(starts, stops):
        index[tuple(int(c) for c in cell[order[s]])] = (int(s), int(e))
    return order, pos[order], cell[order], index


def cells_in_bbox(low, high, chunk_shape, bounds):
    lower = np.asarray(bounds[0], float)
    cs = np.asarray(chunk_shape, float)
    grid = np.ceil((np.asarray(bounds[1], float) - lower) / cs).astype(np.int64)
    lo = np.clip(((np.asarray(low, float) - lower) // cs).astype(np.int64), 0, grid - 1)
    hi = np.clip(((np.asarray(high, float) - lower) // cs).astype(np.int64), 0, grid - 1)
    return [(i, j, k)
            for i in range(lo[0], hi[0] + 1)
            for j in range(lo[1], hi[1] + 1)
            for k in range(lo[2], hi[2] + 1)]


def _bbox_filter(pts, low, high):
    m = np.all((pts >= np.asarray(low)) & (pts <= np.asarray(high)), axis=1)
    return pts[m]


# ----- Parquet: one row group per spatial cell -------------------------

def parquet_write(path, positions, chunk_shape, bounds) -> None:
    """One row group per cell, so the footer's per-group min/max statistics
    are a usable spatial index."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    _, pos, _, index = spatial_cells(positions, chunk_shape, bounds)
    schema = pa.schema([("x", pa.float32()), ("y", pa.float32()),
                        ("z", pa.float32())])
    with pq.ParquetWriter(str(path), schema, compression="zstd") as wr:
        for s, e in sorted(index.values()):
            block = pos[s:e]
            wr.write_table(pa.table(
                {"x": block[:, 0], "y": block[:, 1], "z": block[:, 2]},
                schema=schema))


def parquet_read_full(path):
    import pyarrow.parquet as pq
    t = pq.read_table(str(path))
    return np.column_stack([t.column(c).to_numpy() for c in ("x", "y", "z")])


def parquet_read_bbox(path, low, high):
    """Select row groups by their footer statistics, read only those."""
    import pyarrow.parquet as pq
    f = pq.ParquetFile(str(path))
    lo, hi = np.asarray(low, float), np.asarray(high, float)
    want = []
    for g in range(f.metadata.num_row_groups):
        rg = f.metadata.row_group(g)
        keep = True
        for c, (a, b) in enumerate(zip(lo, hi)):
            st = rg.column(c).statistics
            if st is not None and (st.max < a or st.min > b):
                keep = False
                break
        if keep:
            want.append(g)
    if not want:
        return np.zeros((0, 3), np.float32)
    t = f.read_row_groups(want, columns=["x", "y", "z"])
    pts = np.column_stack([t.column(c).to_numpy() for c in ("x", "y", "z")])
    return _bbox_filter(pts, low, high)


# ----- Plain Zarr: the same grid, none of the layout --------------------

def zarr_write(path, positions, chunk_shape, bounds) -> None:
    """Points sorted into cell order in one array, plus a cell offset table.

    Chunked on the row axis at the mean cell size, so a cell read touches
    a small number of zarr chunks -- the ordinary way anyone would put a
    spatially-sorted point cloud into zarr.
    """
    import json

    import zarr

    _, pos, _, index = spatial_cells(positions, chunk_shape, bounds)
    n_cells = max(1, len(index))
    rows = max(1, int(np.ceil(len(pos) / n_cells)))
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    g = zarr.open_group(store=str(root), mode="w")
    arr = g.create_array("positions", shape=pos.shape, chunks=(rows, 3),
                         dtype="<f4", compressors=zarr.codecs.ZstdCodec())
    arr[:] = pos
    (root / "cells.json").write_text(json.dumps(
        {"_".join(map(str, k)): v for k, v in index.items()}))


def _zarr_open(path):
    import json

    import zarr
    root = Path(path)
    g = zarr.open_group(store=str(root), mode="r")
    cells = {tuple(int(p) for p in k.split("_")): tuple(v)
             for k, v in json.loads((root / "cells.json").read_text()).items()}
    return g["positions"], cells


def zarr_read_full(path):
    arr, _ = _zarr_open(path)
    return arr[:]


def zarr_read_bbox(path, low, high, chunk_shape, bounds):
    arr, cells = _zarr_open(path)
    parts = [arr[s:e] for c in cells_in_bbox(low, high, chunk_shape, bounds)
             if (se := cells.get(c)) for s, e in (se,)]
    if not parts:
        return np.zeros((0, 3), np.float32)
    return _bbox_filter(np.concatenate(parts), low, high)


# ----- HDF5: chunked dataset over the same ordering --------------------

def hdf5_write(path, positions, chunk_shape, bounds) -> None:
    import h5py
    _, pos, _, index = spatial_cells(positions, chunk_shape, bounds)
    n_cells = max(1, len(index))
    rows = max(1, int(np.ceil(len(pos) / n_cells)))
    keys = sorted(index)
    with h5py.File(str(path), "w") as f:
        f.create_dataset("positions", data=pos, chunks=(rows, 3),
                         compression="gzip", compression_opts=4)
        f.create_dataset("cell_keys", data=np.asarray(keys, dtype=np.int64))
        f.create_dataset("cell_range",
                         data=np.asarray([index[k] for k in keys], np.int64))


def _hdf5_cells(f):
    return {tuple(int(x) for x in k): (int(a), int(b))
            for k, (a, b) in zip(f["cell_keys"][:], f["cell_range"][:])}


def hdf5_read_full(path):
    import h5py
    with h5py.File(str(path), "r") as f:
        return f["positions"][:]


def hdf5_read_bbox(path, low, high, chunk_shape, bounds):
    import h5py
    with h5py.File(str(path), "r") as f:
        cells = _hdf5_cells(f)
        ds = f["positions"]
        parts = [ds[s:e] for c in cells_in_bbox(low, high, chunk_shape, bounds)
                 if (se := cells.get(c)) for s, e in (se,)]
    if not parts:
        return np.zeros((0, 3), np.float32)
    return _bbox_filter(np.concatenate(parts), low, high)
