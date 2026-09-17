#!/usr/bin/env python
"""Measure everything the paper figure needs, once, into a tidy CSV.

    python run_sweep.py                 # full sweep, 10^3..10^6, ~20 min
    python run_sweep.py --quick         # smoke test, ~1 min
    python run_sweep.py --workdir /scratch
    python run_large.py                 # 10^3..10^7, ~1.5 h, resumable

Measurement is deliberately separated from plotting.  This script writes
``results/measurements.csv`` and ``results/environment.json``; nothing
here draws anything.  ``make_figure.py`` turns that CSV into the figure
and the supplementary table, so the figure can be regenerated anywhere
without re-running the sweep, and the sweep can be re-run on reference
hardware without a plotting stack installed.

What is measured
----------------
Five ``--blocks`` produce six row groups in the CSV's ``block`` column.

``storage``   on-disk bytes for every (geometry, N, format) combination,
              including gzip of the text formats and both zarr-vectors
              codec settings.
``timing``    -> ``bulk``   write-everything and read-everything, per
                            geometry.
              -> ``object`` fetch one object by id, and read-modify-write
                            one object.
``spatial``   bounding-box query, swept three ways: by selected fraction
              at fixed N, at a fixed ~100-object selection across N
              (``spatial_fixed``), and at a fixed ~100-vertex box across
              N (``spatial_volume``), which is one box volume for every
              geometry rather than one per geometry.
``chunked``   the same bulk and single-object operations against the
              other chunked layouts (Parquet, Zarr, HDF5, precomputed).
``fidelity``  write, read back, compare -- per geometry and competitor.

One sweep variable: every block sweeps the **vertex** count (``SIZES``).
Objects follow from it, because each geometry's objects are a fixed size
(1 vertex per point, 12 per streamline, 144 per mesh patch), and both
counts are recorded on every row -- ``n_vertices`` is what the panels
plot and ``n_objects`` is what the key converts it to.

Every zarr-vectors store in the timing blocks uses the same deployed
configuration: ``compressor='zstd'``, fixed 200-unit chunks, and a
per-object index.  The storage block additionally reports the
uncompressed and un-indexed variants so the cost of each can be read off.

Timing measurements are ``TIMING_RUNS`` repeats each, fixed rather than
adaptive -- ``LARGE_TIMING_RUNS`` at the top of the large sweep, where a
single repeat is a whole-dataset write of 10^7 vertices.

Sizes
-----
``--sizes`` sets the swept vertex counts; the default is one point per
decade to 10^6 and ``--large`` extends that by one more.  The sub-blocks
that hold N fixed rather than sweeping it -- the selected-fraction
spatial sweep at 10^6, round-trip fidelity at 10^5, and the chunk-count
and fragmentation blocks at 10^6 -- stay pinned wherever the sweep ends,
so extending the range adds points to the curves without moving anything
that was measured at a fixed size.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _formats as F  # noqa: E402
import _harness as H  # noqa: E402
from zarr_vectors.building import open_store, read_object_manifests  # noqa: E402
from zarr_vectors.ops import EditSession, FragmentRef, VertexRef  # noqa: E402
from zarr_vectors.types.graphs import read_graph, write_graph  # noqa: E402
from zarr_vectors.types.meshes import read_mesh, write_mesh  # noqa: E402
from zarr_vectors.types.points import read_points, write_points  # noqa: E402
from zarr_vectors.types.polylines import (  # noqa: E402
    read_polylines,
    write_polylines,
)

RESULTS = Path(__file__).resolve().parent / "results"

# One point per decade.  Span matters more than resolution inside a
# decade -- these curves are close to straight on log-log, so extra
# points bought precision on a slope that was never in doubt while making
# every panel dense enough to be hard to read.
#
# The default stops at 10^6 because that is what a decade costs: 10^7
# roughly triples the sweep's wall time for one more point per line, and
# most of the time this script is run it is being run to check that
# nothing regressed, not to produce a figure.  The extra decade is not
# dropped, only moved -- ``run_large.py`` drives this same script over
# ``LARGE_SIZES`` block by block, into its own results directory, so the
# long run is opt-in rather than the price of every run.
SIZES = [1_000, 10_000, 100_000, 1_000_000]
LARGE_SIZES = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
QUICK_SIZES = [1_000, 10_000]

# Above this the sweep is in "large" territory: single writes run to tens
# of seconds, competitor text writers to minutes, and the repeat counts
# below step down accordingly.  Named rather than inlined because three
# separate decisions key off it.
LARGE_N = 10_000_000

# Objects come along for the ride: every geometry has a fixed number of
# vertices per object (1 per point, 12 per streamline, 144 per mesh
# patch), so a vertex target fixes the object count too and both are
# recorded on every row.  Sweeping vertices rather than objects is what
# keeps the three geometries on one x range -- an object sweep puts the
# mesh line at 144 k to 144 M vertices and the point line at 10^3 to
# 10^6, which is two curves that never overlap.

# Repeats per timing measurement.  Fixed rather than adaptive: five runs
# of everything, including the sub-millisecond calls that ``H.repeat``
# would otherwise keep sampling up to 200 times.  Confidence intervals on
# the cheap operations are correspondingly wider.
TIMING_RUNS = 5

# ... except at the top of the large sweep, where "five runs of
# everything" is five whole-dataset writes of 10^7 vertices per geometry
# per implementation.  Three still gives a confidence interval (t on 2
# degrees of freedom is wide, and the CSV records ``n_runs`` per row so
# the widening is visible rather than implied), and it is the difference
# between the timing block taking an hour and taking most of an evening.
LARGE_TIMING_RUNS = 3


def timing_runs(n: int) -> int:
    """Repeats for the timing block at vertex target ``n``."""
    return LARGE_TIMING_RUNS if n >= LARGE_N else TIMING_RUNS

# Selected-volume fractions for the spatial sweep at fixed N.
FRACTIONS = [0.001, 0.01, 0.1, 0.5, 1.0]
FRACTION_N = 1_000_000
FIXED_SELECTION = 100          # objects targeted by the fixed-selection sweep
FIXED_VERTICES = 100           # vertices targeted by the equal-box sweep

CODEC = "zstd"                 # deployed configuration for every timing block


# --------------------------------------------------------------------
# Run counts -- fewer repeats where a single run costs seconds.
# --------------------------------------------------------------------

def n_runs(op: str, n: int) -> int:
    """Floor on repeats for the vertex-swept blocks.

    ``H.repeat`` adds more where a call is cheap enough that seven
    samples would be mostly clock noise, so this only needs to say when
    to *stop* early.  The timing block does not use it -- it takes
    exactly ``timing_runs(n)`` samples of everything.
    """
    if n >= LARGE_N:
        # A single write of 10^7 vertices is tens of seconds and a single
        # bbox query still returns in milliseconds, so the floor splits
        # further here rather than holding at the 300k tier's 3 and 5.
        return 2 if op in ("write_full", "replace_one") else 3
    if n >= 300_000:
        return 3 if op in ("write_full", "replace_one") else 5
    return 7


# --------------------------------------------------------------------
# zarr-vectors adapters
# --------------------------------------------------------------------

def _cs(chunk):
    """Chunk shape: the deployed one unless a sweep asks for another."""
    return H.CHUNK if chunk is None else (float(chunk),) * 3


def _bs(chunk):
    """Bin shape, held at a quarter of the chunk side.

    The deployed configuration is a 200-unit chunk over a 50-unit bin, so
    a chunk-shape sweep that left the bin at 50 would be sweeping two
    things at once -- and would ask for bins larger than the chunk that
    contains them at the fine end.
    """
    return H.BIN if chunk is None else (float(chunk) / 4.0,) * 3


def zv_write_points(path, data, *, codec=CODEC, object_ids=False,
                    chunk=None):
    """A point cloud needs no object index.

    Each point is addressed by where it is, not by an id, so the store is
    written without one: ``read_points(bbox=...)`` finds a point and
    ``VertexRef.from_position`` edits it.  Giving every point its own
    object costs 36x on write, 16x on read and 4.5 bytes per point, and
    buys nothing that a spatial lookup does not already provide -- the
    storage block measures that variant separately so the price is on
    record.
    """
    write_points(str(path), data["positions"], chunk_shape=_cs(chunk),
                 bin_shape=_bs(chunk), bounds=H.BOUNDS, compressor=codec,
                 object_ids=data["object_ids"] if object_ids else None)


def zv_write_streamlines(path, data, *, codec=CODEC, object_ids=True,
                    chunk=None):
    write_polylines(str(path), data["polylines"], chunk_shape=_cs(chunk),
                    bin_shape=_bs(chunk), bounds=H.BOUNDS, compressor=codec)


def zv_write_skeletons(path, data, *, codec=CODEC, object_ids=True,
                    chunk=None):
    write_polylines(str(path), data["polylines"], chunk_shape=_cs(chunk),
                    bin_shape=_bs(chunk), bounds=H.BOUNDS, compressor=codec,
                    geometry_type="skeleton",
                    vertex_attributes={"radius": data["radii"]})


def zv_write_graphs(path, data, *, codec=CODEC, object_ids=True,
                    chunk=None):
    write_graph(str(path), data["positions"], data["edges"],
                chunk_shape=_cs(chunk), bin_shape=_bs(chunk), bounds=H.BOUNDS,
                compressor=codec)


def zv_write_meshes(path, data, *, codec=CODEC, object_ids=True,
                    chunk=None):
    write_mesh(str(path), data["vertices"], data["faces"],
               chunk_shape=_cs(chunk), bin_shape=_bs(chunk), bounds=H.BOUNDS,
               compressor=codec,
               object_ids=data["object_ids"] if object_ids else None)


ZV_WRITERS = {
    "points": zv_write_points,
    "streamlines": zv_write_streamlines,
    "skeletons": zv_write_skeletons,
    "graphs": zv_write_graphs,
    "meshes": zv_write_meshes,
}


POINT_EPS = 0.5   # half-width of the box used to address a single point


def _point_bbox(pos):
    return ((np.asarray(pos) - POINT_EPS).tolist(),
            (np.asarray(pos) + POINT_EPS).tolist())


def zv_replace_point(path, index, data):
    """Read one point, perturb it, write it back -- addressed by position.

    ``VertexRef.from_position`` resolves the chunk and offset from the
    coordinate itself, so this works on a store with no object index.
    """
    target = data["positions"][int(index)]
    got = read_points(str(path), bbox=_point_bbox(target))
    new = np.asarray(got["positions"][0], dtype=np.float32) + np.float32(0.25)
    root = open_store(str(path), mode="r+")
    with EditSession(root, atomic=True, refresh_pyramid=False) as ed:
        ref = VertexRef.from_position(root, level=0, pos=target.tolist())
        ed.edit_vertex(ref, new_pos=np.clip(new, 0.0, H.DOMAIN - 1).tolist(),
                       atomic=True)


def zv_replace_streamline(path, oid):
    """Read one streamline, perturb it, write it back.

    A streamline that crosses a chunk boundary is stored as several
    fragments; each is rewritten with its own slice of the new vertices,
    so the operation is a true whole-object replacement regardless of how
    the object is split.
    """
    frags = read_polylines(str(path), object_ids=[int(oid)])["polylines"][0]
    root = open_store(str(path), mode="r+")
    manifest = read_object_manifests(root["0"], ids=[int(oid)])[int(oid)]
    with EditSession(root, atomic=True, refresh_pyramid=False) as ed:
        for (chunk, frag_idx), frag in zip(manifest, frags):
            new = np.clip(np.asarray(frag, dtype=np.float32) + np.float32(0.25),
                          0.0, H.DOMAIN - 1)
            ed.edit_fragment(FragmentRef(level=0, chunk=chunk, fragment=frag_idx),
                             new_vertices=new)


def bbox_for_fraction(fraction: float):
    """Centred axis-aligned box covering ``fraction`` of the domain volume."""
    side = H.DOMAIN * float(fraction) ** (1 / 3)
    lo = np.full(3, (H.DOMAIN - side) / 2.0)
    return lo.tolist(), (lo + side).tolist()


# --------------------------------------------------------------------
# Row accumulation
# --------------------------------------------------------------------

FIELDS = ["block", "geometry", "n_target", "n_vertices", "n_objects", "op",
          "impl", "unit", "value", "ci_hw", "n_runs", "detail"]


class Rows(list):
    def add(self, **kw):
        row = {f: kw.get(f, "") for f in FIELDS}
        self.append(row)
        return row


def log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------
# Block 1 -- storage
# --------------------------------------------------------------------

def block_storage(rows, sizes, ws):
    for geom in GEOM_ALL:
        for n in sizes:
            data = H.GENERATORS[geom](n)
            nv, no = data["n_vertices"], data["n_objects"]
            log(f"  storage {geom:12s} N={n:>9,}  ({nv:,} verts, {no:,} objects)")

            variants = [("zarr-vectors (no codec)", dict(codec=None)),
                        ("zarr-vectors (zstd)", dict(codec="zstd"))]
            if geom == "points" and n <= PER_POINT_INDEX_MAX_N:
                variants.append(("zarr-vectors (zstd, per-point object index)",
                                 dict(codec="zstd", object_ids=True)))
            elif geom == "points":
                log(f"    skip per-point object index at N={n:,} "
                    f"(grouping is quadratic in points per chunk)")
            elif geom == "meshes":
                variants.append(("zarr-vectors (zstd, no object index)",
                                 dict(codec="zstd", object_ids=False)))
            for label, kw in variants:
                tag = f"{geom}_{n}_{label}"
                store = ws.store(tag)
                ZV_WRITERS[geom](store, data, **kw)
                rows.add(block="storage", geometry=geom, n_target=n,
                         n_vertices=nv, n_objects=no, op="bytes", impl=label,
                         unit="bytes", value=H.path_bytes(store), n_runs=1)
                ws.clear(tag)

            for label, kind, suffix, cap, writer in _competitor_files(geom, data):
                if cap is not None and n > cap:
                    log(f"    skip {label} at N={n:,} (writer does not scale)")
                    continue
                tag = f"{geom}_{n}_{label}"
                path = ws.file(tag, suffix)
                writer(path)
                # A competitor may legitimately need more than one file
                # (node table + edge table); every file it wrote lives in
                # this tag's own directory, so the directory is its size.
                written = sorted(p for p in path.parent.rglob("*") if p.is_file())
                rows.add(block="storage", geometry=geom, n_target=n,
                         n_vertices=nv, n_objects=no, op="bytes", impl=label,
                         unit="bytes", value=H.path_bytes(path.parent),
                         n_runs=1, detail=kind)
                # Gzip every competitor, binary included.  Reporting a
                # Zstd-compressed store against an uncompressed binary
                # file is the same asymmetry as reporting it against raw
                # ASCII, and it flattered the store: TRK gzips to 10.92
                # bytes per vertex, below anything zarr-vectors managed.
                if "(gzip)" not in label:
                    rows.add(block="storage", geometry=geom, n_target=n,
                             n_vertices=nv, n_objects=no, op="bytes",
                             impl=f"{label} (gzip)", unit="bytes",
                             value=sum(H.gzip_bytes(f) for f in written),
                             n_runs=1, detail=f"{kind}-gzip")
                ws.clear(tag)


# Some competitor writers do not scale to the top of the sweep -- the
# networkx GraphML serialiser builds the whole XML tree in memory and
# takes minutes at a million nodes.  A cap here drops that one point
# rather than silently distorting the run; the omission is reported.
GRAPHML_MAX_N = 100_000

# The per-point object index is quadratic in the points per chunk -- the
# writer groups vertices by object with one full-array mask per object --
# so it costs ~15 s at 10^6 points and roughly half an hour at 10^7.  It
# is measured to put a price on record, and that price is already on
# record at 10^6; the top size is skipped and the skip is logged.  See
# the README's "a per-point object index is expensive" section.
PER_POINT_INDEX_MAX_N = 1_000_000


def _competitor_files(geom, data):
    """``(label, kind, suffix, max_n, writer)`` for every format compared."""
    if geom == "points":
        return [
            ("PLY", "binary", ".ply", None,
             lambda p: F.ply_write(p, data["positions"])),
            ("CSV", "text", ".csv", None,
             lambda p: F.csv_points_write(p, data["positions"])),
            ("precomputed (spatial only)", "chunked", "", None,
             lambda p: F.precomputed_write(p, data["positions"], H.CHUNK,
                                           H.BOUNDS, with_by_id=False)),
        ]
    if geom == "streamlines":
        return [
            ("TRK", "binary", ".trk", None,
             lambda p: F.trk_write(p, data["polylines"])),
            ("TRX", "binary", ".trx", None,
             lambda p: F.trx_write(p, data["polylines"])),
            # TRX's own lossless compression setting -- the like-for-like
            # comparison against a Zstd-compressed store.
            ("TRX deflate", "binary", ".trx", None,
             lambda p: F.trx_write(p, data["polylines"], deflate=True)),
        ]
    if geom == "skeletons":
        return [("SWC", "text", ".swc", None,
                 lambda p: F.swc_write(p, data["polylines"], data["radii"]))]
    if geom == "graphs":
        return [
            ("CSV edge-list", "text", ".csv", None,
             lambda p: F.csv_graph_write(p.parent / "nodes.csv", p,
                                         data["positions"], data["edges"])),
            ("GraphML", "text", ".graphml", GRAPHML_MAX_N,
             lambda p: F.graphml_write(p, data["positions"], data["edges"])),
        ]
    if geom == "meshes":
        return [
            ("STL", "binary", ".stl", None,
             lambda p: F.stl_write(p, data["vertices"], data["faces"])),
            ("OBJ", "text", ".obj", None,
             lambda p: F.obj_write(p, data["vertices"], data["faces"])),
        ]
    raise KeyError(geom)


# --------------------------------------------------------------------
# Block 2 + 3 -- bulk and per-object timing
# --------------------------------------------------------------------

GEOM_ALL = ("points", "streamlines", "skeletons", "graphs", "meshes")
TIMED_GEOMS = ("points", "streamlines", "meshes")
COMPETITOR = {"points": "PLY", "streamlines": "TRK", "meshes": "STL"}
COMPETITOR_SUFFIX = {"points": ".ply", "streamlines": ".trk", "meshes": ".stl"}


def block_timing(rows, sizes, ws):
    """Write, read and per-object timings, per geometry, per vertex count.

    Each object stays the same size at every point on the curve, so the
    object count moves in proportion with the vertex count and both are
    on every row: ``n_vertices`` is what the panels plot, ``n_objects``
    is what the key converts it to.
    """
    for geom in TIMED_GEOMS:
        for n_obj in sizes:
            data = H.GENERATORS[geom](n_obj)
            nv, no = data["n_vertices"], data["n_objects"]
            comp = COMPETITOR[geom]
            log(f"  timing  {geom:12s} N={n_obj:>9,}  "
                f"({nv:,} verts, {no:,} objects)")

            def add(op, impl, mean, hw, runs, detail="", _n=n_obj, _nv=nv,
                    _no=no, _geom=geom):
                rows.add(block="bulk" if op in ("write_full", "read_full")
                         else "object", geometry=_geom, n_target=_n,
                         n_vertices=_nv, n_objects=_no, op=op, impl=impl,
                         unit="s", value=round(mean, 6), ci_hw=round(hw, 6),
                         n_runs=runs, detail=detail)

            # ---- write everything -----------------------------------
            r = timing_runs(n_obj)
            m, hw, k = H.repeat(
                lambda tag: ZV_WRITERS[geom](ws.store(tag), data),
                r,
                setup=lambda i: f"w_{geom}_{n_obj}_{i}",
                teardown=lambda tag: ws.clear(tag))
            add("write_full", "zarr-vectors", m, hw, k)

            m, hw, k = H.repeat(
                lambda tag: _comp_write(geom, ws.file(tag, COMPETITOR_SUFFIX[geom]),
                                        data),
                r,
                setup=lambda i: f"cw_{geom}_{n_obj}_{i}",
                teardown=lambda tag: ws.clear(tag))
            add("write_full", comp, m, hw, k)

            # ---- persistent inputs for the read/edit ops -------------
            store = ws.store(f"s_{geom}_{n_obj}")
            ZV_WRITERS[geom](store, data)
            cfile = ws.file(f"c_{geom}_{n_obj}", COMPETITOR_SUFFIX[geom])
            _comp_write(geom, cfile, data)

            # ---- read everything ------------------------------------
            m, hw, k = H.repeat(lambda: _zv_read_full(geom, store), r)
            add("read_full", "zarr-vectors", m, hw, k)
            m, hw, k = H.repeat(lambda: _comp_read_full(geom, cfile), r)
            add("read_full", comp, m, hw, k)

            # ---- fetch one object by id -----------------------------
            oid = no // 2
            m, hw, k = H.repeat(lambda: _zv_fetch_one(geom, store, oid, data),
                                r)
            over = _fetch_overshoot(geom, store, oid, data)
            api = {"points": "bbox= around the point",
                   "streamlines": "object_ids=",
                   "meshes": "chunks= then filter (no object_ids= on read_mesh)"}
            add("fetch_one", "zarr-vectors", m, hw, k,
                detail=f"{api[geom]}; reads {over:.0f}x the object")
            m, hw, k = H.repeat(lambda: _comp_fetch_one(geom, cfile, oid, data),
                                r)
            add("fetch_one", comp, m, hw, k, detail=_fetch_note(geom))

            # ---- read-modify-write one object -----------------------
            if geom in ("points", "streamlines"):
                if geom == "points":
                    def fn(i, _s=store, _d=data, _n=no):
                        return zv_replace_point(_s, (oid + i * 7) % _n, _d)
                else:
                    def fn(i, _s=store, _n=no):
                        return zv_replace_streamline(_s, (oid + i * 7) % _n)
                # Each repeat targets a different object so that no run
                # benefits from the previous run's warm chunk.
                m, hw, k = H.repeat(fn, r, setup=lambda i: i)
                add("replace_one", "zarr-vectors", m, hw, k,
                    detail="VertexRef.from_position" if geom == "points"
                    else "edit_fragment via object manifest")
                m, hw, k = H.repeat(
                    lambda i: _comp_replace_one(geom, cfile,
                                                (oid + i * 7) % no, data),
                    r, setup=lambda i: i)
                add("replace_one", comp, m, hw, k, detail=_replace_note(geom))

            ws.clear(f"s_{geom}_{n_obj}")
            ws.clear(f"c_{geom}_{n_obj}")


def _comp_write(geom, path, data):
    if geom == "points":
        return F.ply_write(path, data["positions"])
    if geom == "streamlines":
        return F.trk_write(path, data["polylines"])
    return F.stl_write(path, data["vertices"], data["faces"])


def _zv_read_full(geom, store):
    if geom == "points":
        return read_points(str(store))
    if geom == "streamlines":
        return read_polylines(str(store))
    return read_mesh(str(store))


def _comp_read_full(geom, path):
    if geom == "points":
        return F.ply_read_full(path)
    if geom == "streamlines":
        return F.trk_read_full(path)
    return F.stl_read_full(path)


def _mesh_object_chunks(data, oid, chunk=None):
    """Every chunk the object's vertices land in.

    A patch that straddles a boundary lives in more than one chunk, so
    asking for a single chunk returns *part* of the object -- 120 of 144
    vertices at N = 10^3.  The caller knows the object's extent (it is
    the bounding box any viewer already has), so the honest equivalent
    fetches all of them.
    """
    v = data["vertices"][data["object_ids"] == oid]
    cc = np.floor(v / np.asarray(_cs(chunk))).astype(np.int64)
    return [tuple(int(x) for x in c) for c in np.unique(cc, axis=0)]


def _zv_fetch_one(geom, store, oid, data, chunk=None):
    """Fetch exactly one object, by the cheapest route the public API has.

    All three geometries return the whole object and nothing less.  What
    differs is how much extra comes with it: points and streamlines
    return the object alone, while ``read_mesh`` has no ``object_ids=``
    filter, so a mesh patch has to arrive inside its chunks and be
    filtered locally.  ``fetch_overshoot`` records that amplification per
    row so the mesh series is read for what it is.
    """
    if geom == "points":
        return read_points(str(store),
                           bbox=_point_bbox(data["positions"][int(oid)]))
    if geom == "streamlines":
        return read_polylines(str(store), object_ids=[int(oid)])
    got = read_mesh(str(store), chunks=_mesh_object_chunks(data, oid, chunk))
    want = data["vertices"][data["object_ids"] == oid]
    keep = np.isin(got["vertices"].view([("", got["vertices"].dtype)] * 3),
                   want.view([("", want.dtype)] * 3))
    return {"vertices": got["vertices"][keep.ravel()]}


def _fetch_overshoot(geom, store, oid, data, chunk=None):
    """Vertices read to deliver the object, over vertices in the object."""
    got = _zv_fetch_one(geom, store, oid, data, chunk)
    if geom == "points":
        n_read, n_want = len(got["positions"]), 1
    elif geom == "streamlines":
        n_read = sum(sum(len(f) for f in o) for o in got["polylines"])
        n_want = len(data["polylines"][int(oid)])
    else:
        n_read = len(read_mesh(
            str(store),
            chunks=_mesh_object_chunks(data, oid, chunk))["vertices"])
        n_want = int((data["object_ids"] == oid).sum())
    return n_read / max(1, n_want)


def _fetch_note(geom):
    if geom == "points":
        return "fixed-width record: seek + 12-byte read"
    if geom == "streamlines":
        return "variable-length records: full scan"
    return "no object index: full read"


def _comp_fetch_one(geom, path, oid, data):
    if geom == "points":
        # Best case available to PLY: the row index is known, so seek
        # straight to the record instead of reading the file.
        off = F._ply_data_offset(path)
        with open(path, "rb") as fh:
            fh.seek(off + int(oid) * F.PLY_ITEMSIZE)
            return np.frombuffer(fh.read(F.PLY_ITEMSIZE), dtype="<f4")
    if geom == "streamlines":
        return F.trk_read_objects(path, [int(oid)])
    per = len(data["faces"]) // data["n_objects"]
    return F.stl_read_objects(path, [(oid * per, (oid + 1) * per)])


def _replace_note(geom):
    if geom == "points":
        return "in-place seek + 12-byte overwrite"
    return "variable-length records: full read-modify-rewrite"


def _comp_replace_one(geom, path, oid, data):
    if geom == "points":
        new = data["positions"][int(oid)] + np.float32(0.25)
        return F.ply_replace_object(path, int(oid), new)
    new = np.asarray(data["polylines"][int(oid)], dtype=np.float32) + 0.25
    return F.trk_replace_object(path, int(oid), new)


# --------------------------------------------------------------------
# Block 4 -- spatial queries
# --------------------------------------------------------------------

def _zv_bbox(geom, store, low, high):
    if geom == "points":
        return read_points(str(store), bbox=(low, high))
    if geom == "streamlines":
        return read_polylines(str(store), bbox=(low, high))
    return read_mesh(str(store), bbox=(low, high))


def _comp_bbox(geom, path, low, high):
    if geom == "points":
        return F.ply_read_bbox(path, low, high)
    if geom == "streamlines":
        return F.trk_read_bbox(path, low, high)
    return F.stl_read_bbox(path, low, high)


def block_spatial(rows, sizes, ws):
    # --- (a) selected fraction sweep at fixed N ----------------------
    n = FRACTION_N if FRACTION_N in sizes else max(sizes)
    for geom in TIMED_GEOMS:
        data = H.GENERATORS[geom](n)
        store = ws.store(f"sp_{geom}")
        ZV_WRITERS[geom](store, data)
        cfile = ws.file(f"spc_{geom}", COMPETITOR_SUFFIX[geom])
        _comp_write(geom, cfile, data)
        for frac in FRACTIONS:
            low, high = bbox_for_fraction(frac)
            log(f"  spatial {geom:12s} N={n:>9,} fraction={frac}")
            m, hw, k = H.repeat(lambda: _zv_bbox(geom, store, low, high), 7)
            rows.add(block="spatial_fraction", geometry=geom, n_target=n,
                     n_vertices=data["n_vertices"], n_objects=data["n_objects"],
                     op="bbox_query", impl="zarr-vectors", unit="s",
                     value=round(m, 6), ci_hw=round(hw, 6), n_runs=k,
                     detail=frac)
            m, hw, k = H.repeat(lambda: _comp_bbox(geom, cfile, low, high), 7)
            rows.add(block="spatial_fraction", geometry=geom, n_target=n,
                     n_vertices=data["n_vertices"], n_objects=data["n_objects"],
                     op="bbox_query", impl=COMPETITOR[geom], unit="s",
                     value=round(m, 6), ci_hw=round(hw, 6), n_runs=k,
                     detail=frac)
        ws.clear(f"sp_{geom}")
        ws.clear(f"spc_{geom}")

    # --- (b) fixed ~100-object selection across N --------------------
    for geom in TIMED_GEOMS:
        for n in sizes:
            data = H.GENERATORS[geom](n)
            frac = min(1.0, FIXED_SELECTION / max(1, data["n_objects"]))
            low, high = bbox_for_fraction(frac)
            store = ws.store(f"fx_{geom}_{n}")
            ZV_WRITERS[geom](store, data)
            cfile = ws.file(f"fxc_{geom}_{n}", COMPETITOR_SUFFIX[geom])
            _comp_write(geom, cfile, data)
            log(f"  fixed   {geom:12s} N={n:>9,} fraction={frac:.5f}")
            r = n_runs("bbox_query", n)
            m, hw, k = H.repeat(lambda: _zv_bbox(geom, store, low, high), r)
            rows.add(block="spatial_fixed", geometry=geom, n_target=n,
                     n_vertices=data["n_vertices"], n_objects=data["n_objects"],
                     op="bbox_query", impl="zarr-vectors", unit="s",
                     value=round(m, 6), ci_hw=round(hw, 6), n_runs=k,
                     detail=round(frac, 6))
            m, hw, k = H.repeat(lambda: _comp_bbox(geom, cfile, low, high), r)
            rows.add(block="spatial_fixed", geometry=geom, n_target=n,
                     n_vertices=data["n_vertices"], n_objects=data["n_objects"],
                     op="bbox_query", impl=COMPETITOR[geom], unit="s",
                     value=round(m, 6), ci_hw=round(hw, 6), n_runs=k,
                     detail=round(frac, 6))
            ws.clear(f"fx_{geom}_{n}")
            ws.clear(f"fxc_{geom}_{n}")

    # --- (c) equal box across geometries -----------------------------
    # (b) sizes the box to ~100 *objects*, which is a different box per
    # geometry -- 100 mesh patches is 14 400 vertices and 100 points is
    # 100 -- so its volume curve is per geometry and the three series
    # cannot be read against each other.  This one sizes the box to
    # ~100 *vertices* instead: one volume for all three at every N, and
    # the same expected payload, since every geometry spreads N vertices
    # over the same domain.  What differs is only how those vertices are
    # grouped, which is the thing being compared.
    for geom in TIMED_GEOMS:
        for n in sizes:
            data = H.GENERATORS[geom](n)
            frac = min(1.0, FIXED_VERTICES / max(1, data["n_vertices"]))
            low, high = bbox_for_fraction(frac)
            store = ws.store(f"eq_{geom}_{n}")
            ZV_WRITERS[geom](store, data)
            cfile = ws.file(f"eqc_{geom}_{n}", COMPETITOR_SUFFIX[geom])
            _comp_write(geom, cfile, data)
            log(f"  equal   {geom:12s} N={n:>9,} fraction={frac:.6f}")
            r = n_runs("bbox_query", n)
            for impl, fn in (("zarr-vectors",
                              lambda: _zv_bbox(geom, store, low, high)),
                             (COMPETITOR[geom],
                              lambda: _comp_bbox(geom, cfile, low, high))):
                m, hw, k = H.repeat(fn, r)
                rows.add(block="spatial_volume", geometry=geom, n_target=n,
                         n_vertices=data["n_vertices"],
                         n_objects=data["n_objects"], op="bbox_query",
                         impl=impl, unit="s", value=round(m, 6),
                         ci_hw=round(hw, 6), n_runs=k, detail=round(frac, 8))
            ws.clear(f"eq_{geom}_{n}")
            ws.clear(f"eqc_{geom}_{n}")



# --------------------------------------------------------------------
# Block 6 -- chunk count
# --------------------------------------------------------------------
#
# Everywhere else the grid is fixed at 200 units over a 1000-unit domain
# -- 125 chunks, whatever the dataset size -- so nothing in the sweep
# says what the chunk count itself costs.  This block holds the dataset
# fixed and moves the grid instead: the same 10^6 vertices cut 8 ways up
# to 8000, and one object fetched out of each.  It is the price of the
# partition, separated from the price of the data.

CHUNK_SWEEP_N = 1_000_000     # dataset size the chunk sweep holds fixed
CHUNK_SIDES = [500.0, 250.0, 200.0, 125.0, 100.0, 50.0]   # -> 8 .. 8000


def _grid_chunks(side):
    """Chunks over the domain at this side length."""
    return int(round((H.DOMAIN / side) ** 3))


def block_chunks(rows, ws):
    for geom in TIMED_GEOMS:
        data = H.GENERATORS[geom](CHUNK_SWEEP_N)
        nv, no = data["n_vertices"], data["n_objects"]
        oid = no // 2
        for side in CHUNK_SIDES:
            n_chunks = _grid_chunks(side)
            tag = f"ck_{geom}_{int(side)}"
            store = ws.store(tag)
            log(f"  chunks  {geom:12s} side={side:>6.1f}  "
                f"{n_chunks:>6,} chunks  ({nv:,} verts)")
            t_write, _ = H.timed(ZV_WRITERS[geom], store, data, chunk=side)
            m, hw, k = H.repeat(
                lambda: _zv_fetch_one(geom, store, oid, data, side),
                n_runs("fetch_one", nv))
            over = _fetch_overshoot(geom, store, oid, data, side)
            for op, val, half, runs in (("fetch_one", m, hw, k),
                                        ("write_full", t_write, 0.0, 1)):
                rows.add(block="chunks", geometry=geom, n_target=n_chunks,
                         n_vertices=nv, n_objects=no, op=op,
                         impl="zarr-vectors", unit="s", value=round(val, 6),
                         ci_hw=round(half, 6), n_runs=runs,
                         detail=f"chunk side {side:g}; reads {over:.0f}x "
                                f"the object")
            ws.clear(tag)


# --------------------------------------------------------------------
# Block 7 -- fragmentation
# --------------------------------------------------------------------
#
# The chunk-count block sweeps the grid under the generator's own
# objects, which are small: a 12-vertex streamline is ~32 units across
# and a mesh patch 40, against chunks of 50 to 500 units.  They sit
# inside one cell almost however the grid is drawn, so that block cannot
# say what it costs to *span* chunks.
#
# This block sweeps the same grid over the same 10^6-vertex datasets but
# fetches a planted probe instead: a single point, a long streamline, and
# a large mesh patch.  The point occupies one chunk at every grid, so its
# curve is the per-chunk floor -- and should fall as finer chunks carry
# less of the store's other data with them.  The streamline and the patch
# occupy more cells the finer the grid gets, so theirs is the cost of
# reassembling one object out of many.  Payload is fixed: each probe
# carries the same vertices whatever the grid.

FRAG_N = 1_000_000            # background dataset the probes are planted in
FRAG_SIDES = [500.0, 250.0, 200.0, 125.0, 100.0, 50.0]     # 8 .. 8000 chunks
FRAG_LINE_VERTS = 1024        # vertices in the long streamline probe
FRAG_MESH_SIDE = 48           # 48 x 48 = 2304 vertices in the patch probe
FRAG_MESH_SPAN = 600.0        # units the patch spans


def _long_streamline():
    """One streamline that runs across the domain, not a 32-unit stub.

    A helix: 800 units of travel along x wrapped around a 600-unit
    circle, so it threads a large number of cells at any grid without
    leaving the domain.
    """
    t = np.linspace(0.0, 1.0, FRAG_LINE_VERTS)
    return np.stack([100.0 + 800.0 * t,
                     500.0 + 300.0 * np.sin(4.0 * np.pi * t),
                     500.0 + 300.0 * np.cos(4.0 * np.pi * t)],
                    axis=1).astype(np.float32)


def _large_patch():
    """A 48x48 patch spanning ``FRAG_MESH_SPAN`` units, gently curved."""
    side, span = FRAG_MESH_SIDE, FRAG_MESH_SPAN
    u, v = np.meshgrid(np.linspace(0.0, span, side),
                       np.linspace(0.0, span, side), indexing="ij")
    w = H.DOMAIN / 2.0 + 100.0 * np.sin(3.0 * np.pi * u / span)
    verts = np.stack([u + (H.DOMAIN - span) / 2.0,
                      v + (H.DOMAIN - span) / 2.0, w], -1).reshape(-1, 3)
    i = np.arange(side - 1)
    ii, jj = np.meshgrid(i, i, indexing="ij")
    a = (ii * side + jj).ravel()
    b, c, d = a + 1, a + side, a + side + 1
    faces = np.concatenate([np.stack([a, b, c], 1), np.stack([b, d, c], 1)])
    return (np.clip(verts, 0.0, H.DOMAIN - 1.0).astype(np.float32),
            faces.astype(np.int64))


def _spans(points, side):
    """Chunks a set of vertices occupies at this grid."""
    cc = np.floor(np.asarray(points, float) / side).astype(np.int64)
    return int(len(np.unique(cc, axis=0)))


def block_fragments(rows, ws):
    point_data = H.GENERATORS["points"](FRAG_N)
    line_base = H.GENERATORS["streamlines"](FRAG_N)["polylines"]
    lines = list(line_base) + [_long_streamline()]
    line_oid = len(line_base)

    mesh = H.GENERATORS["meshes"](FRAG_N)
    pv, pf = _large_patch()
    mesh_oid = int(mesh["object_ids"].max()) + 1
    mesh_data = {
        "vertices": np.concatenate([mesh["vertices"], pv]),
        "faces": np.concatenate([mesh["faces"], pf + len(mesh["vertices"])]),
        "object_ids": np.concatenate(
            [mesh["object_ids"], np.full(len(pv), mesh_oid, np.int64)]),
    }

    def add(geom, side, m, hw, runs, span, n_read, note):
        rows.add(block="fragments", geometry=geom,
                 n_target=_grid_chunks(side), n_vertices=FRAG_N,
                 n_objects=span, op="fetch_one", impl="zarr-vectors",
                 unit="s", value=round(m, 6), ci_hw=round(hw, 6), n_runs=runs,
                 detail=f"chunk side {side:g}; {note} spans {span} chunks; "
                        f"{n_read} vertices read")

    for side in FRAG_SIDES:
        log(f"  frag    grid side {side:>6.1f}  "
            f"({_grid_chunks(side):>6,} chunks)")

        # ---- one point: one chunk, at every grid --------------------
        store = ws.store("fr_points")
        zv_write_points(store, point_data, chunk=side)
        pos = point_data["positions"][len(point_data["positions"]) // 2]
        box = _point_bbox(pos)
        n_read = len(read_points(str(store), bbox=box)["positions"])
        m, hw, runs = H.repeat(lambda: read_points(str(store), bbox=box), 7)
        add("points", side, m, hw, runs, 1, n_read, "one point")
        ws.clear("fr_points")

        # ---- one long streamline -----------------------------------
        store = ws.store("fr_lines")
        write_polylines(str(store), lines, chunk_shape=(side,) * 3,
                        bin_shape=(side / 4,) * 3, bounds=H.BOUNDS,
                        compressor=CODEC)
        got = read_polylines(str(store), object_ids=[line_oid])
        n_read = sum(sum(len(f) for f in o) for o in got["polylines"])
        n_frag = sum(len(o) for o in got["polylines"])
        m, hw, runs = H.repeat(
            lambda: read_polylines(str(store), object_ids=[line_oid]), 7)
        add("streamlines", side, m, hw, runs, n_frag, n_read,
            f"{FRAG_LINE_VERTS}-vertex streamline")
        ws.clear("fr_lines")

        # ---- one large mesh patch ----------------------------------
        store = ws.store("fr_mesh")
        zv_write_meshes(store, mesh_data, chunk=side)
        chunks = _mesh_object_chunks(mesh_data, mesh_oid, side)
        n_read = len(read_mesh(str(store), chunks=chunks)["vertices"])
        m, hw, runs = H.repeat(
            lambda: _zv_fetch_one("meshes", store, mesh_oid, mesh_data, side),
            5)
        add("meshes", side, m, hw, runs, len(chunks), n_read,
            f"{FRAG_MESH_SIDE ** 2}-vertex patch")
        ws.clear("fr_mesh")


# --------------------------------------------------------------------
# Block 8 -- versus other chunked formats
# --------------------------------------------------------------------
#
# The single-file blocks compare against formats that must read
# everything to answer anything, so zarr-vectors wins any partial-read
# comparison eventually and the only question is where.  This block asks
# the harder one: against layouts that also chunk, does the store still
# earn its keep?
#
# All four sort points into the *same* spatial grid zarr-vectors uses, so
# what is being compared is the layout over a partition, not the choice
# of partition.  Plain Zarr is the control that matters most -- same
# library, same codec, same grid, none of the zarr-vectors fragment /
# object / manifest machinery on top.

CHUNKED_GEOM = "points"
CHUNKED_FORMATS = ("precomputed annotations", "Parquet", "plain Zarr", "HDF5")

# Precomputed's ``by_id`` index is one file per annotation, so a write at
# N = 10^7 creates ten million files -- per repeat, and again for the
# persistent copy.  That is hours of inode churn to measure a curve whose
# shape is already settled by 10^6, so the format is dropped above this
# and the omission is logged.  The other three chunked formats scale fine.
PRECOMPUTED_MAX_N = 1_000_000
_CHUNKED_SUFFIX = {"precomputed annotations": "", "Parquet": ".parquet",
                   "plain Zarr": "", "HDF5": ".h5"}


def _chunked_write(fmt, path, data):
    pos = data["positions"]
    if fmt == "precomputed annotations":
        return F.precomputed_write(path, pos, H.CHUNK, H.BOUNDS)
    if fmt == "Parquet":
        return F.parquet_write(path, pos, H.CHUNK, H.BOUNDS)
    if fmt == "plain Zarr":
        return F.zarr_write(path, pos, H.CHUNK, H.BOUNDS)
    return F.hdf5_write(path, pos, H.CHUNK, H.BOUNDS)


def _chunked_read_full(fmt, path):
    return {"precomputed annotations": F.precomputed_read_full,
            "Parquet": F.parquet_read_full,
            "plain Zarr": F.zarr_read_full,
            "HDF5": F.hdf5_read_full}[fmt](path)


def _chunked_read_bbox(fmt, path, low, high):
    if fmt == "precomputed annotations":
        return F.precomputed_read_bbox(path, low, high)
    if fmt == "Parquet":
        return F.parquet_read_bbox(path, low, high)
    if fmt == "plain Zarr":
        return F.zarr_read_bbox(path, low, high, H.CHUNK, H.BOUNDS)
    return F.hdf5_read_bbox(path, low, high, H.CHUNK, H.BOUNDS)


def block_chunked(rows, sizes, ws):
    geom = CHUNKED_GEOM
    if geom not in GEOM_ALL:
        return
    for n in sizes:
        data = H.GENERATORS[geom](n)
        nv, no = data["n_vertices"], data["n_objects"]
        frac = min(1.0, FIXED_SELECTION / max(1, no))
        low, high = bbox_for_fraction(frac)
        log(f"  chunked {geom} N={n:>9,} (box fraction {frac:.6f})")

        for fmt in CHUNKED_FORMATS:
            if fmt == "precomputed annotations" and n > PRECOMPUTED_MAX_N:
                log(f"    skip {fmt} at N={n:,} "
                    f"(by_id is one file per annotation)")
                continue
            sfx = _CHUNKED_SUFFIX[fmt]
            tag = fmt.replace(" ", "_")

            def add(op, mean, hw, runs, detail=""):
                rows.add(block="chunked", geometry=geom, n_target=n,
                         n_vertices=nv, n_objects=no, op=op, impl=fmt,
                         unit="s", value=round(mean, 6), ci_hw=round(hw, 6),
                         n_runs=runs, detail=detail)

            m, hw, k = H.repeat(
                lambda t_: _chunked_write(fmt, ws.file(t_, sfx), data),
                n_runs("write_full", n),
                setup=lambda i: f"cw_{tag}_{n}_{i}",
                teardown=lambda t_: ws.clear(t_))
            add("write_full", m, hw, k)

            target = ws.file(f"c_{tag}_{n}", sfx)
            _chunked_write(fmt, target, data)
            rows.add(block="storage", geometry=geom, n_target=n,
                     n_vertices=nv, n_objects=no, op="bytes", impl=fmt,
                     unit="bytes", value=H.path_bytes(target.parent),
                     n_runs=1, detail="chunked")

            m, hw, k = H.repeat(lambda: _chunked_read_full(fmt, target),
                                n_runs("read_full", n))
            add("read_full", m, hw, k)

            m, hw, k = H.repeat(
                lambda: _chunked_read_bbox(fmt, target, low, high),
                n_runs("bbox_query", n))
            add("bbox_query", m, hw, k, f"box fraction {frac:.6f}")

            if fmt == "precomputed annotations":
                m, hw, k = H.repeat(
                    lambda: F.precomputed_read_one(target, no // 2),
                    n_runs("fetch_one", n))
                add("fetch_one", m, hw, k, "by_id: one file, one read")
            ws.clear(f"c_{tag}_{n}")

# --------------------------------------------------------------------
# Block 5 -- round-trip fidelity
# --------------------------------------------------------------------

def _sorted_rows(a):
    """Lexicographically sorted copy, so a comparison is insensitive to the
    order chunks happen to be walked in."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    return a[np.lexsort((a[:, 2], a[:, 1], a[:, 0]))]


def _max_abs_error(source, roundtrip):
    a, b = _sorted_rows(source), _sorted_rows(roundtrip)
    if a.shape != b.shape:
        return float("inf"), a.shape[0], b.shape[0]
    return float(np.abs(a - b).max()), a.shape[0], b.shape[0]


def block_fidelity(rows, sizes, ws):
    """Does the data survive the round trip, for every geometry type?

    Claim one of the paper -- that the format works across geometry types
    -- is a correctness statement, not a timing one, so it is measured as
    the largest coordinate discrepancy between what went in and what came
    back out.  Vertices are compared in sorted order because neither side
    promises to preserve input ordering.
    """
    n = 100_000 if 100_000 in sizes else max(sizes)
    for geom in GEOM_ALL:
        data = H.GENERATORS[geom](n)
        source = (data["positions"] if geom in ("points", "graphs")
                  else data["vertices"] if geom == "meshes"
                  else np.concatenate(data["polylines"]))
        store = ws.store(f"fid_{geom}")
        ZV_WRITERS[geom](store, data)
        back = _zv_read_back(geom, store)
        err, n_in, n_out = _max_abs_error(source, back)
        log(f"  fidelity {geom:12s} max|Δ| = {err:g}  ({n_in:,} in, {n_out:,} out)")
        rows.add(block="fidelity", geometry=geom, n_target=n,
                 n_vertices=data["n_vertices"], n_objects=data["n_objects"],
                 op="roundtrip_max_abs_error", impl="zarr-vectors (zstd)",
                 unit="store units", value=err, n_runs=1,
                 detail=f"{n_in} in / {n_out} out")
        ws.clear(f"fid_{geom}")

        for label, reader in _fidelity_competitors(geom, data, ws):
            path = ws.file(f"fidc_{geom}_{label}", f".{label.lower()}")
            got = reader(path)
            err, n_in, n_out = _max_abs_error(source, got)
            log(f"  fidelity {geom:12s} {label:4s} max|Δ| = {err:g}")
            rows.add(block="fidelity", geometry=geom, n_target=n,
                     n_vertices=data["n_vertices"],
                     n_objects=data["n_objects"],
                     op="roundtrip_max_abs_error", impl=label,
                     unit="store units", value=err, n_runs=1,
                     detail=f"{n_in} in / {n_out} out")
            ws.clear(f"fidc_{geom}_{label}")


def _zv_read_back(geom, store):
    if geom == "points":
        return read_points(str(store))["positions"]
    if geom == "graphs":
        return read_graph(str(store))["positions"]
    if geom == "meshes":
        return read_mesh(str(store))["vertices"]
    out = read_polylines(str(store))["polylines"]
    return np.concatenate([np.concatenate(frags) for frags in out])


def _fidelity_competitors(geom, data, ws):
    """Binary competitors we already have readers for.  The text formats
    are size-only in this suite and are not round-tripped."""
    if geom == "points":
        def _ply(p):
            F.ply_write(p, data["positions"])
            return F.ply_read_full(p)
        return [("PLY", _ply)]
    if geom == "streamlines":
        def _trk(p):
            F.trk_write(p, data["polylines"])
            return F.trk_read_full(p)[0]

        def _trx(p):
            F.trx_write(p, data["polylines"])
            return np.concatenate(F.trx_read_full(p))
        return [("TRK", _trk), ("TRX", _trx)]
    if geom == "meshes":
        def _stl(p):
            F.stl_write(p, data["vertices"], data["faces"])
            # STL stores triangles, not a vertex table: every vertex is
            # repeated once per incident face, so the round trip is
            # compared on the unique positions it recovers.
            return np.unique(F.stl_read_full(p).reshape(-1, 3), axis=0)
        return [("STL", _stl)]
    return []


# --------------------------------------------------------------------

def _size(token: str) -> int:
    """Parse one ``--sizes`` entry.

    Accepts what anyone would type at a shell for a vertex count that
    runs to eight digits: ``10000000``, ``10_000_000`` and ``1e7`` are
    the same number, and a float that is not one is an error rather than
    a silent truncation.
    """
    t = token.strip().replace("_", "")
    try:
        v = float(t)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a size: {token!r}") from None
    if v != int(v) or v < 1:
        raise argparse.ArgumentTypeError(f"not a positive whole size: {token!r}")
    return int(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="small sizes only, for a smoke test")
    ap.add_argument("--large", action="store_true",
                    help="extend the sweep by one decade, to 10^7; hours "
                         "rather than minutes, so see run_large.py, which "
                         "drives this block by block and can resume")
    ap.add_argument("--sizes", default=None,
                    help="comma-separated vertex counts to sweep, "
                         "overriding --quick/--large, e.g. '1000,10000'. "
                         "Underscores and 1e6 notation are accepted")
    ap.add_argument("--workdir", default=None,
                    help="parent directory for scratch stores")
    ap.add_argument("--out", default=str(RESULTS / "measurements.csv"))
    ap.add_argument("--blocks",
                    default="storage,timing,spatial,chunks,fragments,"
                            "chunked,fidelity")
    ap.add_argument("--geometries", default=None,
                    help="comma-separated subset to measure, e.g. 'points'; "
                         "used with --append to re-measure one geometry")
    ap.add_argument("--append", action="store_true",
                    help="add rows to an existing measurements.csv instead "
                         "of replacing it (used to run one block on its own)")
    ap.add_argument("--replace", action="store_true",
                    help="like --append, but first drop the rows this run "
                         "supersedes: any existing row whose (block, "
                         "geometry, op, impl) this run re-measured. Use it "
                         "to re-run one block into an existing sweep without "
                         "leaving two generations of the same series in the "
                         "file")
    args = ap.parse_args()

    if args.sizes:
        try:
            sizes = sorted({_size(t) for t in args.sizes.split(",")
                            if t.strip()})
        except argparse.ArgumentTypeError as exc:
            ap.error(str(exc))
    elif args.quick:
        sizes = QUICK_SIZES
    elif args.large:
        sizes = LARGE_SIZES
    else:
        sizes = SIZES
    if not sizes:
        ap.error("--sizes named no sizes")
    wanted = {b.strip() for b in args.blocks.split(",") if b.strip()}
    if args.geometries:
        keep = {g.strip() for g in args.geometries.split(",") if g.strip()}
        global GEOM_ALL, TIMED_GEOMS
        GEOM_ALL = tuple(g for g in GEOM_ALL if g in keep)
        TIMED_GEOMS = tuple(g for g in TIMED_GEOMS if g in keep)
        log(f"geometries restricted to {sorted(keep)}")
    rows = Rows()
    ws = H.Workspace(args.workdir)
    log(f"scratch: {ws.root}")
    try:
        if "storage" in wanted:
            log("== storage ==")
            block_storage(rows, sizes, ws)
        if "timing" in wanted:
            log("== bulk + per-object timing ==")
            block_timing(rows, sizes, ws)
        if "spatial" in wanted:
            log("== spatial queries ==")
            block_spatial(rows, sizes, ws)
        if "chunks" in wanted:
            log("== chunk-count sweep ==")
            block_chunks(rows, ws)
        if "fragments" in wanted:
            log("== fragmentation ==")
            block_fragments(rows, ws)
        if "chunked" in wanted:
            log("== versus other chunked formats ==")
            block_chunked(rows, sizes, ws)
        if "fidelity" in wanted:
            log("== round-trip fidelity ==")
            block_fidelity(rows, sizes, ws)
    finally:
        ws.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merge = (args.append or args.replace) and out.exists()

    kept = []
    if merge:
        with open(out, newline="") as fh:
            existing = list(csv.DictReader(fh))
        if args.replace:
            # Supersede on the series key rather than on the block name.
            # Dropping whole blocks would take the chunked block's
            # storage rows with it, and keying on size would leave behind
            # sizes this run no longer measures -- the object sweep
            # covers different sizes than the vertex sweep it replaces.
            fresh = {(r["block"], r["geometry"], r["op"], r["impl"])
                     for r in rows}
            kept = [r for r in existing
                    if (r["block"], r["geometry"], r["op"], r["impl"])
                    not in fresh]
            log(f"replacing {len(existing) - len(kept)} superseded rows "
                f"of {len(existing)}")
        else:
            kept = existing

    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(kept)
        writer.writerows(rows)
    log(f"wrote {out}  ({len(rows)} new rows, {len(kept)} carried over)")

    env = {}
    env_path = out.parent / "environment.json"
    if merge and env_path.exists():
        # Keep the provenance of the blocks that were not re-run, and
        # record that the file now mixes two runs.
        env = json.loads(env_path.read_text())
        # The carried-over rows were measured against whatever was
        # installed then.  If that differs from what is installed now,
        # keep the old provenance rather than overwriting it with a
        # version those rows were never measured under -- and only then
        # is the file a mixed run.  Re-running block by block in one
        # sitting, against one install, is not.
        # Compare through JSON, not against the live objects: a tuple
        # and the list it was serialised as are the same provenance, and
        # a naive != would report chunk_shape as having changed on every
        # merge.
        now = json.loads(json.dumps(H.environment()))
        was = {k: v for k, v in env.items()
               if k in now and now[k] != v and k != "mixed_run"}
        if was:
            env["previous_run"] = was
            env["mixed_run"] = sorted(set(env.get("mixed_run", [])) | wanted)
    env.update(H.environment())
    env["sizes"] = sizes
    env["timing_runs"] = {str(n): timing_runs(n) for n in sizes}
    env["verts_per_object"] = dict(H.VERTS_PER_OBJECT)
    env["codec"] = CODEC
    env["workdir"] = str(args.workdir or "(system temp)")
    env["workdir_fstype"] = H.fstype(args.workdir or "/tmp")
    env["fractions"] = FRACTIONS
    env["fraction_n"] = FRACTION_N
    env["fixed_selection"] = FIXED_SELECTION
    env["chunk_sweep_n"] = CHUNK_SWEEP_N
    env["chunk_sides"] = CHUNK_SIDES
    env_path.write_text(json.dumps(env, indent=2))
    log(f"wrote {env_path}")


if __name__ == "__main__":
    main()
