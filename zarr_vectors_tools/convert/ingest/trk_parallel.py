"""Memory-bounded parallel TRK → zarr-vectors ingest.

Processes large TRK (TrackVis) files without loading them whole into RAM.
Coordinate convention: by default the raw TrackVis voxmm coordinates are stored
as-is and the ``vox_to_ras`` affine is recorded in CRS metadata (a viewer that
understands the affine renders it registered).  Pass ``register_to_rasmm=True``
to instead bake that affine into the vertices at ingest — matching
``nibabel.streamlines.load(...).streamlines`` — and store an identity CRS
affine.  That is the layout viewers which can only scale/translate (not rotate),
e.g. neuroglancer, need: NGFF transforms can't undo the affine's axis flips, so
the geometry itself must carry them.

Pipeline
--------
Phase 0  Parse 1000-byte header; resolve the registration affine.
Phase 1  Offset-index scan (~6 s for 5.6M streamlines), also sampling one
         vertex per streamline to size ``chunk_shape`` from ``num_chunks``;
         partition into N parts.
Phase A  Parallel over N parts: bin streamlines → spatial chunks, a block of
         streamlines at a time, write .npz, report each part's exact bbox.
Grid     Single-process: union those bboxes into the store bounds and lay out
         the chunk grid, then create the store.  The grid comes from the
         geometry, not the header's declared FOV — see ingest_trk_parallel.
Phase B  Parallel over batches of the S chunks: assemble from all N parts →
         write level-0.
Coord    Single-process: reconstruct manifests + cross-chunk links from the
         part directories; the links are written in parallel, whole offsets
         arrays per task.
Phase 5  Store CRS/affine metadata + TRKHeader for round-trip.
Phase 6  Build multiscale pyramid via coarsen.build_pyramid.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from zarr_vectors.building import (
    LevelMetadata,
    create_attribute_array,
    create_fragment_attribute_array,
    create_links_array,
    create_links_family,
    create_object_index_array,
    create_resolution_level,
    create_store,
    create_vertices_array,
    get_resolution_level,
    level_grid_layout,
    links_group_path,
    links_has_perm,
    open_store,
    rebuild_presence,
    refresh_arrays_present,
    update_level_metadata,
    write_chunk_attributes,
    write_chunk_fragment_attributes,
    write_chunk_links,
    write_chunk_vertices,
    write_object_index,
)
from zarr_vectors.exceptions import IngestError

from zarr_vectors_tools.convert.ingest._cell_limits import check_vertex_cell_limit
from zarr_vectors_tools.convert.ingest._polyline_enrichments import (
    arc_length_normalized,
    compute_tangents,
    index_normalized,
)
from zarr_vectors_tools.multiresolution.coarsen import (
    build_pyramid,
    validate_rdp_tolerances,
)

# Per-vertex synthetic attribute generators.  ``_VERTEX_CARRY`` names must be
# computed on the whole streamline in Phase A (before the chunk split, which
# would otherwise destroy the along-streamline context) and carried through it
# via a parallel memmap; the rest (x/y/z/random) are derived at chunk-write
# time in Phase B from the already-assembled positions.  All stored float32.
_VERTEX_CARRY = ("arc_length", "index", "tangent")
_VERTEX_ATTR_NCOLS = {
    "arc_length": 1, "index": 1, "tangent": 3,
    "x": 1, "y": 1, "z": 1, "random": 1,
}
_TANGENT_CHANNELS = ["dx", "dy", "dz"]


def _carry_attr_path(npz_path: str, name: str) -> str:
    """Path of the Phase-A per-part memmap carrying vertex attribute ``name``."""
    return npz_path.replace(".npz", f".attr_{name}.npy")


def _index_path(npz_path: str) -> str:
    """Per-part segment table, sorted by target chunk (see _SEG_COLS)."""
    return npz_path.replace(".npz", ".index.npy")


def _chunk_dir_path(npz_path: str) -> str:
    """Per-part directory of ``(chunk, slice)`` into the sorted segment table."""
    return npz_path.replace(".npz", ".chunks.npy")


#: Columns of the per-part sorted segment table.  One row per segment, in
#: ascending ``(chunk, poly_id, index within the polyline)`` order, which is
#: also the order Phase B must emit fragments in.
_SEG_POLY_ID = 0
_SEG_WITHIN = 1
_SEG_VTX_START = 2
_SEG_VTX_COUNT = 3
_SEG_COLS = 4


# ---------------------------------------------------------------------------
# TRK header parsing (no nibabel required)
# ---------------------------------------------------------------------------

def parse_trk_header(path: str | Path) -> dict[str, Any]:
    """Parse the 1000-byte TRK header using struct.

    Returns a dict with keys: dim, voxel_size, origin, n_scalars, n_properties,
    scalar_names and property_names (``[(name, columns), ...]``, see
    :func:`~zarr_vectors_tools.convert.ingest._trk_names.decode_trk_names`),
    vox_to_ras (4×4 float32 array), voxel_order (str), n_count, version, hdr_size.
    """
    from zarr_vectors_tools.convert.ingest._trk_names import (
        TRK_N_PROPERTIES,
        TRK_N_SCALARS,
        TRK_PROPERTY_NAMES,
        TRK_SCALAR_NAMES,
        decode_trk_names,
    )

    path = Path(path)
    with open(path, "rb") as f:
        raw = f.read(1000)

    if len(raw) < 1000:
        raise ValueError(f"File too short to be a valid TRK: {path}")

    id_string = raw[0:6]
    if id_string[:5] != b"TRACK":
        raise ValueError(f"Not a TRK file (bad magic): {path}")

    dim = struct.unpack_from("<3h", raw, 6)
    voxel_size = struct.unpack_from("<3f", raw, 12)
    origin = struct.unpack_from("<3f", raw, 24)
    n_scalars = struct.unpack_from("<h", raw, TRK_N_SCALARS)[0]
    # 238, after the ten 20-byte scalar names that start at 38.  This was read
    # from 236 -- the last two bytes of the tenth scalar name, almost always
    # zero -- so a file with per-streamline properties parsed as having none,
    # and the offset scan then read property bytes as the next point count.
    n_properties = struct.unpack_from("<h", raw, TRK_N_PROPERTIES)[0]
    vox_to_ras_flat = struct.unpack_from("<16f", raw, 440)
    vox_to_ras = np.array(vox_to_ras_flat, dtype=np.float32).reshape(4, 4)
    voxel_order_raw = raw[948:952]
    voxel_order = voxel_order_raw.rstrip(b"\x00").decode("ascii", errors="replace")
    n_count = struct.unpack_from("<i", raw, 988)[0]
    version = struct.unpack_from("<i", raw, 992)[0]
    hdr_size = struct.unpack_from("<i", raw, 996)[0]

    return {
        "dim": tuple(int(d) for d in dim),
        "voxel_size": tuple(float(v) for v in voxel_size),
        "origin": tuple(float(o) for o in origin),
        "n_scalars": int(n_scalars),
        "n_properties": int(n_properties),
        "scalar_names": decode_trk_names(raw, TRK_SCALAR_NAMES, int(n_scalars)),
        "property_names": decode_trk_names(
            raw, TRK_PROPERTY_NAMES, int(n_properties), unnamed="property",
        ),
        "vox_to_ras": vox_to_ras,
        "voxel_order": voxel_order,
        "n_count": int(n_count),
        "version": int(version),
        "hdr_size": int(hdr_size),
    }


def _compute_bounds_from_header(header: dict[str, Any]) -> tuple[list[float], list[float]]:
    """Compute voxmm bounding box from header dim and voxel_size."""
    dim = header["dim"]
    vs = header["voxel_size"]
    lo = [0.0, 0.0, 0.0]
    hi = [float(dim[i]) * float(vs[i]) for i in range(3)]
    return lo, hi


def _trackvis_to_rasmm_affine(input_path: str | Path) -> np.ndarray:
    """4×4 affine mapping the TRK's on-disk trackvis-voxmm coords to RASmm.

    Uses nibabel's canonical transform, which correctly handles ``voxel_order``
    and the voxel-corner→center convention (a naive ``vox_to_ras @
    (voxmm / voxel_size)`` gets those subtly wrong).  Baking this into the
    geometry makes the stored streamlines match what
    ``nibabel.streamlines.load(...).streamlines`` returns — i.e. registered to
    the source image's world space — so neuroglancer renders them aligned.
    """
    try:
        import nibabel as nib
        from nibabel.streamlines.trk import get_affine_trackvis_to_rasmm
    except ImportError as e:  # pragma: no cover - env without nibabel
        raise IngestError(
            "nibabel is required to register TRK streamlines to RASmm world "
            "space (so neuroglancer aligns them with the source image). "
            "Install with: pip install nibabel"
        ) from e
    hdr = nib.streamlines.load(str(input_path), lazy_load=True).header
    return np.asarray(get_affine_trackvis_to_rasmm(hdr), dtype=np.float64)


def _transform_bounds(
    bounds: tuple[list[float], list[float]], affine: np.ndarray,
) -> tuple[list[float], list[float]]:
    """Axis-aligned bbox of a min/max corner pair after applying ``affine``.

    The affine maps the voxmm box to a rotated parallelepiped; its
    axis-aligned bounding box is the min/max over the 8 transformed corners.
    """
    lo, hi = bounds
    a = np.asarray(affine, dtype=np.float64)
    corners = np.array(
        [[x, y, z] for x in (lo[0], hi[0])
         for y in (lo[1], hi[1]) for z in (lo[2], hi[2])],
        dtype=np.float64,
    )
    out = corners @ a[:3, :3].T + a[:3, 3]
    return out.min(axis=0).tolist(), out.max(axis=0).tolist()


def _bounds_of_points(
    points: npt.NDArray, fallback: tuple[list[float], list[float]],
) -> tuple[list[float], list[float]]:
    """Axis-aligned bbox of a point cloud, or ``fallback`` if it is empty."""
    if points is None or len(points) == 0:
        return fallback
    p = np.asarray(points, dtype=np.float64)
    return p.min(axis=0).tolist(), p.max(axis=0).tolist()


def _bounds_contain(
    outer: tuple[list[float], list[float]],
    inner: tuple[list[float], list[float]],
) -> bool:
    """True when ``outer`` fully contains ``inner`` (inclusive)."""
    return bool(
        np.all(np.asarray(outer[0]) <= np.asarray(inner[0]))
        and np.all(np.asarray(outer[1]) >= np.asarray(inner[1]))
    )


def _compute_chunk_shape(
    bounds: tuple[list[float], list[float]],
    num_chunks: int | tuple[int, int, int] | None = None,
) -> tuple[float, float, float]:
    """Derive near-isotropic chunk shape from bounds and target chunk count.

    Args:
        bounds: (min_corner, max_corner) in voxmm.
        num_chunks: Integer total target chunk count (near-isotropic default),
            or explicit (nx, ny, nz) 3-tuple, or None (uses default of 125).

    Returns:
        Chunk shape (cx, cy, cz) in mm.
    """
    lo, hi = bounds
    extent = [hi[i] - lo[i] for i in range(3)]

    if isinstance(num_chunks, tuple):
        nx, ny, nz = num_chunks
    else:
        T = int(num_chunks) if num_chunks is not None else 125
        vol = extent[0] * extent[1] * extent[2]
        if vol <= 0:
            return (10.0, 10.0, 10.0)
        s = (T / vol) ** (1.0 / 3.0)
        nx = max(1, round(extent[0] * s))
        ny = max(1, round(extent[1] * s))
        nz = max(1, round(extent[2] * s))

    # Round for a readable chunk size, but never to zero: a sub-millimetre
    # tractogram (or any extent smaller than the chunk count) rounded straight
    # to ``(0, 0, 0)``, and every downstream ``floor(p / chunk_shape)`` then
    # divides by zero.  Fall back to the exact quotient, and to 1 only when the
    # extent itself is degenerate.
    def _edge(extent_axis: float, n: int) -> float:
        exact = float(extent_axis) / float(max(n, 1))
        rounded = round(exact)
        if rounded >= 1:
            return float(rounded)
        return exact if exact > 0 else 1.0

    return (_edge(extent[0], nx), _edge(extent[1], ny), _edge(extent[2], nz))


# ---------------------------------------------------------------------------
# Offset index: scan the file to build streamline byte offsets
# ---------------------------------------------------------------------------

def build_offset_index(path: str | Path, header: dict[str, Any]) -> dict[str, npt.NDArray]:
    """Scan the TRK file and record each streamline's byte offset.

    Returns dict with:
        byte_offset  int64 (O,)  byte position of each streamline's n_points int32
        n_points     int32 (O,)  point count per streamline
        nbytes       int64 (O,)  byte span of each streamline record (4 + n_pts*pt_stride + props)
        first_point  float32 (O, 3)  each streamline's first vertex, on-disk voxmm
    """
    n_scalars = header["n_scalars"]
    n_properties = header["n_properties"]
    pt_stride = (3 + n_scalars) * 4
    prop_bytes = n_properties * 4

    byte_offsets = []
    n_points_list = []
    # One vertex per streamline, collected as raw bytes and viewed as float32
    # at the end — cheaper than 5M struct.unpack calls, and 12 bytes per
    # streamline is a rounding error next to the offsets themselves.
    first_pts = bytearray()

    path = Path(path)
    # Big buffer so the read/skip pair below stays inside one buffer fill for
    # typical record sizes rather than round-tripping to the OS twice.
    with open(path, "rb", buffering=1 << 20) as f:
        f.seek(1000)
        while True:
            pos = f.tell()
            b = f.read(4)
            if len(b) < 4:
                break
            n = struct.unpack("<i", b)[0]
            if n <= 0:
                break
            # The first vertex costs nothing extra: the file is already
            # positioned on it, so this reads 12 bytes it would have skipped.
            # Its bbox over all streamlines is what sizes chunk_shape, because
            # the header's declared FOV can be nothing like where the tracts
            # actually are (see ingest_trk_parallel).
            pt = f.read(12)
            if len(pt) < 12:
                break  # truncated final record — drop it, same as a short count
            n_points_list.append(n)
            byte_offsets.append(pos)
            first_pts += pt
            f.seek(n * pt_stride + prop_bytes - 12, 1)

    byte_offsets_arr = np.array(byte_offsets, dtype=np.int64)
    n_points_arr = np.array(n_points_list, dtype=np.int32)
    pt_stride_arr = np.int64(pt_stride)
    prop_bytes_arr = np.int64(prop_bytes)
    nbytes_arr = (4 + n_points_arr.astype(np.int64) * pt_stride_arr + prop_bytes_arr)

    return {
        "byte_offset": byte_offsets_arr,
        "n_points": n_points_arr,
        "nbytes": nbytes_arr,
        "first_point": np.frombuffer(bytes(first_pts), dtype="<f4").reshape(-1, 3),
    }


def partition_offset_index(
    offset_index: dict[str, npt.NDArray],
    n_parts: int,
) -> list[dict[str, Any]]:
    """Split offset index into N byte-balanced, non-overlapping parts.

    Each part dict has: byte_offset, n_points (arrays for that part),
    poly_id_base (global index of first streamline in part).
    """
    cum_bytes = np.concatenate([[0], np.cumsum(offset_index["nbytes"])])
    total_bytes = int(cum_bytes[-1])
    n_streamlines = len(offset_index["byte_offset"])

    # Compute end indices for each part independently, then build non-overlapping
    # [start, end) ranges. Each boundary snaps to the streamline that first crosses
    # the target byte mark, so no streamline appears in more than one part.
    end_indices: list[int] = []
    for i in range(n_parts):
        target_end = total_bytes * (i + 1) // n_parts
        end_idx = int(np.searchsorted(cum_bytes, target_end, side="right"))
        end_indices.append(min(end_idx, n_streamlines))

    start_indices = [0] + end_indices[:-1]

    parts = []
    for start_idx, end_idx in zip(start_indices, end_indices):
        if start_idx >= end_idx:
            continue
        parts.append({
            "poly_id_base": int(start_idx),
            "byte_offset": offset_index["byte_offset"][start_idx:end_idx],
            "n_points": offset_index["n_points"][start_idx:end_idx],
        })
    return parts


# ---------------------------------------------------------------------------
# Phase A worker: read one part, bin into spatial chunks, write .npz
# ---------------------------------------------------------------------------

#: The object attributes each ``object_attrs`` generator writes.
_OBJECT_ATTR_OUTPUTS: dict[str, tuple[str, ...]] = {
    "length": ("length",),
    "endpoints": ("start", "end"),
    "orientation": ("orientation",),
    "tortuosity": ("tortuosity",),
    "vertex_count": ("vertex_count",),
}


def _plan_file_data(
    header: dict[str, Any],
    keep_scalars: bool,
    keep_properties: bool,
    vertex_attrs: set[str],
    object_attrs: set[str],
) -> tuple[list[tuple[str, str, int, int]], list[tuple[str, int, int]]]:
    """Which of the file's own scalars and properties to store, and where.

    Returns ``(scalars, properties)``: each scalar as ``(attribute name,
    carry key, first column, width)`` and each property as ``(attribute
    name, first column, width)``, columns counted within the file's scalar
    or property block.  The carry key names the Phase-A scratch file; it is
    not the attribute name, which comes from the file and need not be a
    safe filename.

    A file name that collides with another stored attribute is refused
    rather than letting one silently overwrite the other.
    """
    scalars: list[tuple[str, str, int, int]] = []
    if keep_scalars:
        first = 0
        for i, (name, width) in enumerate(header["scalar_names"]):
            scalars.append((name, f"file_scalar_{i}", first, width))
            first += width
    properties: list[tuple[str, int, int]] = []
    if keep_properties:
        first = 0
        for name, width in header["property_names"]:
            properties.append((name, first, width))
            first += width

    generated_objects = {
        out for attr in object_attrs for out in _OBJECT_ATTR_OUTPUTS.get(attr, (attr,))
    }
    for kind, names, taken, option in (
        ("scalar", [s[0] for s in scalars], set(vertex_attrs), "keep_scalars"),
        ("property", [p[0] for p in properties], generated_objects, "keep_properties"),
    ):
        seen: set[str] = set()
        for name in names:
            if name in seen:
                raise IngestError(
                    f"the TRK header names two {kind} columns {name!r}; pass "
                    f"{option}=False to ingest without them"
                )
            seen.add(name)
            if name in taken:
                raise IngestError(
                    f"the TRK file's {kind} {name!r} has the same name as an "
                    f"attribute this ingest generates; stop generating it, or "
                    f"pass {option}=False"
                )
    return scalars, properties


# ---------------------------------------------------------------------------
# Resuming a run
# ---------------------------------------------------------------------------

def _looks_like_store(path: Path) -> bool:
    return path.is_dir() and ((path / "zarr.json").exists() or (path / ".zattrs").exists())


def _part_files(npz_path: str) -> list[Path]:
    """The files Phase A writes for one part, which Phase B reads back."""
    return [
        Path(npz_path),
        Path(npz_path.replace(".npz", ".verts.npy")),
        Path(_index_path(npz_path)),
        Path(_chunk_dir_path(npz_path)),
    ]


class _RunRecord:
    """What a run given ``intermediate_dir`` has finished, kept beside its files.

    A whole-brain ingest spends hours in Phase A and in the pyramid, and a
    run that died in either used to start again from the file scan.  With an
    ``intermediate_dir`` every stage records itself here as it completes --
    the offset index, Phase A's part files, level 0, each pyramid level --
    with the options that produced it, and ``resume=True`` skips what is
    recorded and still on disk.  Without a directory it records nothing and
    every method answers "not done".
    """

    FILE = "trk_parallel_run.json"
    OFFSETS = "offset_index.npz"

    def __init__(self, directory: str | Path | None, *, resume: bool) -> None:
        self.directory = None if directory is None else Path(directory)
        self.resume = resume
        self.state: dict[str, Any] = {}
        self.earlier_level0_attempt = False
        if self.directory is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / self.FILE
        if resume and path.exists():
            try:
                self.state = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                self.state = {}
        # The store is created only after Phase A, so a store at the output
        # path is this ingest's own leftover only if the recorded run got
        # that far.  Anything else there is not ours to remove.
        self.earlier_level0_attempt = bool(self.state.get("phase_a"))

    def get(self, key: str) -> Any:
        return self.state.get(key)

    def set(self, key: str, value: Any) -> None:
        self.state[key] = value
        self._save()

    def forget(self, *keys: str) -> None:
        if any(self.state.pop(key, None) is not None for key in keys):
            self._save()

    def _save(self) -> None:
        if self.directory is None:
            return
        path = self.directory / self.FILE
        scratch = path.with_suffix(".json.tmp")
        scratch.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        os.replace(scratch, path)

    @staticmethod
    def input_fingerprint(input_path: Path) -> dict[str, Any]:
        stat = input_path.stat()
        return {
            "path": str(input_path.resolve()),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    def offset_index(self, input_path: Path, header: dict[str, Any], log: Any) -> dict:
        """The offset index, from the record when the file has not changed.

        The scan reads a few bytes of every streamline in the file, which on
        a large tractogram is minutes of I/O before anything else starts.
        """
        fingerprint = self.input_fingerprint(input_path)
        cached = None if self.directory is None else self.directory / self.OFFSETS
        if (
            self.resume and cached is not None and cached.exists()
            and self.state.get("input") == fingerprint
        ):
            log("  reusing the offset index from an earlier run")
            with np.load(cached) as data:
                return {key: np.asarray(data[key]) for key in data.files}
        index = build_offset_index(input_path, header)
        if cached is not None:
            np.savez(cached, **index)
            if self.state.get("input") != fingerprint:
                self.state = {}
            self.set("input", fingerprint)
        return index

    def check_plan(self, plan: dict[str, Any]) -> None:
        """Refuse to resume onto work done with different options."""
        recorded = (self.state.get("phase_a") or {}).get("plan")
        if not self.resume or recorded is None or recorded == plan:
            return
        changed = sorted(k for k in plan if recorded.get(k) != plan[k])
        raise IngestError(
            f"cannot resume: the run recorded in {self.directory} used different "
            f"{', '.join(changed)}; rerun with the same options, or give a fresh "
            f"intermediate_dir"
        )

    def finished_parts(self, plan: dict[str, Any]) -> list[str] | None:
        phase_a = self.state.get("phase_a") if self.resume else None
        if not phase_a or phase_a.get("plan") != plan:
            return None
        parts = list(phase_a.get("parts") or [])
        if not parts or not all(f.exists() for p in parts for f in _part_files(p)):
            return None
        return parts

    def level0_done(self, plan: dict[str, Any], output_path: Path) -> bool:
        level0 = self.state.get("level0") if self.resume else None
        return bool(level0 and level0.get("plan") == plan and _looks_like_store(output_path))

    def pyramid_start(
        self, pyramid_plan: dict[str, Any], root: Any, level0_done: bool, log: Any,
    ) -> int:
        """How many pyramid levels to keep; removes any level past them.

        Levels are kept only when level 0 was itself kept and the recorded
        pyramid options match.  Anything on disk past the last recorded level
        was being written when the run stopped, and is removed.
        """
        from zarr_vectors.building import list_resolution_levels, remove_resolution_level

        pyramid = self.state.get("pyramid") if self.resume else None
        keep = 0
        if level0_done and pyramid and pyramid.get("plan") == pyramid_plan:
            keep = int(pyramid.get("levels_done", 0))
        existing = set(list_resolution_levels(root))
        while keep > 0 and not all(level in existing for level in range(1, keep + 1)):
            keep -= 1
        for level in sorted(existing, reverse=True):
            if level > keep:
                remove_resolution_level(root, level)
        if keep:
            log(f"  keeping pyramid levels 1-{keep} from an earlier run")
        self.set("pyramid", {"plan": pyramid_plan, "levels_done": keep})
        return keep


#: Vertices Phase A reads and bins per step.  Bounds a worker's working
#: memory (about 100 bytes per vertex across the step's arrays) whatever the
#: part size; a streamline longer than this is one step on its own.
_PHASE_A_BLOCK_VERTICES = 1 << 19


def _line_lengths(
    verts: npt.NDArray, line_first: npt.NDArray, line_counts: npt.NDArray,
) -> npt.NDArray:
    """Path length of each streamline in a block of concatenated streamlines.

    Equal, bit for bit, to ``np.sum(np.linalg.norm(np.diff(line, axis=0),
    axis=1))`` per streamline.  The norms are row-wise, so taking them over
    the whole block changes nothing.  The sum is not: numpy sums pairwise,
    so its rounding depends on the run it is given, and a cumulative or
    ``reduceat`` sum over the block differs in the last bit.  Streamlines
    of equal length are therefore summed together as the rows of one 2-D
    array, which hands each streamline's norms to the same pairwise sum
    one at a time.
    """
    out = np.zeros(line_counts.shape[0], dtype=verts.dtype)
    n_steps = line_counts - 1
    if not np.any(n_steps > 0):
        return out
    steps = verts[1:] - verts[:-1]
    norms = np.linalg.norm(steps, axis=1)
    del steps
    order = np.argsort(n_steps, kind="stable")
    sorted_steps = n_steps[order]
    bounds = np.flatnonzero(np.diff(sorted_steps)) + 1
    for group in np.split(order, bounds):
        n = int(n_steps[group[0]])
        if n <= 0:
            continue
        rows = line_first[group][:, None] + np.arange(n)
        out[group] = np.sum(norms[rows], axis=1)
    return out


def _phase_a_worker(
    part_index: int,
    shared: dict[str, Any],
) -> str:
    """Process one byte-range part of the TRK file.

    Reads streamlines, bins each into spatial chunks via split_polyline_at_boundaries,
    and writes a .npz with per-streamline segment descriptors + raw vertices.

    Returns the path to the written .npz file.
    """
    trk_path = shared["trk_path"]
    header = shared["header"]
    chunk_shape = shared["chunk_shape"]
    affine_tv2ras = shared["affine_tv2ras"]
    intermediate_dir = shared["intermediate_dir"]
    dtype = shared["dtype"]
    compute_length = shared["compute_length"]
    compute_endpoints = shared["compute_endpoints"]
    compute_vertex_count = shared.get("compute_vertex_count", False)
    vertex_carry = shared.get("vertex_carry", [])
    # The file's own per-point scalars, as (carry key, first column, width),
    # and whether to keep its per-streamline properties.
    file_scalars = shared.get("file_scalars", [])
    keep_properties = bool(shared.get("keep_properties", False))
    part_specs = shared["part_specs"]

    part_spec = part_specs[part_index]
    poly_id_base = part_spec["poly_id_base"]
    byte_offsets = part_spec["byte_offset"]
    n_points_arr = part_spec["n_points"]

    n_scalars = header["n_scalars"]
    n_properties = header["n_properties"] if keep_properties else 0
    pt_stride = (3 + n_scalars) * 4
    np_dtype = np.dtype(dtype)

    npz_path = str(Path(intermediate_dir) / f"part_{part_index:06d}.npz")
    verts_npy_path = npz_path.replace(".npz", ".verts.npy")

    # Pre-allocate the vertex output file as a memory-mapped array.
    # split_polyline_at_boundaries never drops vertices (it only re-assigns them
    # to chunks), so the total output vertex count == total input vertex count.
    total_pts = int(n_points_arr.sum())
    # open_memmap writes the .npy header so np.load(path, mmap_mode='r') works.
    verts_mm = np.lib.format.open_memmap(
        verts_npy_path, mode="w+", dtype=np.float32, shape=(total_pts, 3)
    )

    # Parallel memmaps for per-vertex attributes that must be computed on the
    # whole streamline (before the split) and carried through it in the same
    # vertex order as verts_mm.  split_polyline_at_boundaries preserves order
    # (segments are contiguous slices), so slicing the full attribute array by
    # a per-streamline cursor keeps it aligned 1:1 with the stored vertices.
    carry_mm: dict[str, npt.NDArray] = {}
    for name in vertex_carry:
        ncols = _VERTEX_ATTR_NCOLS[name]
        shape = (total_pts,) if ncols == 1 else (total_pts, ncols)
        carry_mm[name] = np.lib.format.open_memmap(
            _carry_attr_path(npz_path, name), mode="w+",
            dtype=np.float32, shape=shape,
        )
    for key, _first, width in file_scalars:
        shape = (total_pts,) if width == 1 else (total_pts, width)
        carry_mm[key] = np.lib.format.open_memmap(
            _carry_attr_path(npz_path, key), mode="w+",
            dtype=np.float32, shape=shape,
        )

    n_lines = int(len(byte_offsets))
    offsets = np.asarray(byte_offsets, dtype=np.int64)
    counts = np.asarray(n_points_arr, dtype=np.int64)
    # Vertex position of each streamline's first point within this part.
    line_first = np.concatenate([[0], np.cumsum(counts)])
    words_per_point = 3 + n_scalars
    # The property block is in the file whether or not it is kept.
    file_property_words = int(header["n_properties"])
    chunk_size = np.array(chunk_shape, dtype=np.float64)

    seg_poly_blocks: list[npt.NDArray] = []
    seg_chunk_blocks: list[npt.NDArray] = []
    seg_count_blocks: list[npt.NDArray] = []
    lengths = np.zeros(n_lines, dtype=np.float32)
    starts = np.zeros((n_lines, 3), dtype=np.float32)
    ends = np.zeros((n_lines, 3), dtype=np.float32)
    properties = np.zeros((n_lines, n_properties), dtype=np.float32)

    # Whole blocks of streamlines at a time: one read, then every per-vertex
    # step as one array operation.  A block is bounded by vertex count, so a
    # part never has to fit in memory -- only one block of it does.  Each
    # result is the one the per-streamline loop this replaced computed:
    # every step is element-wise or row-wise, which does not depend on how
    # many rows share a call, and the one reduction that does -- a
    # streamline's length, a pairwise sum -- is still taken one streamline
    # per row (see _line_lengths).
    write_cursor = 0
    with open(trk_path, "rb") as f:
        s0 = 0
        while s0 < n_lines:
            budget = line_first[s0] + _PHASE_A_BLOCK_VERTICES
            s1 = int(np.searchsorted(line_first, budget, side="right")) - 1
            s1 = min(max(s1, s0 + 1), n_lines)
            block_counts = counts[s0:s1]
            block_first = line_first[s0:s1] - line_first[s0]
            n_block = int(line_first[s1] - line_first[s0])

            # One read covering every record in the block: count words,
            # points and property words, all 4 bytes wide.
            first_byte = int(offsets[s0])
            stop_byte = (
                int(offsets[s1 - 1]) + 4 + int(block_counts[-1]) * pt_stride
                + file_property_words * 4
            )
            words = np.empty((stop_byte - first_byte) // 4, dtype="<f4")
            f.seek(first_byte)
            if f.readinto(words.view(np.uint8)) != words.nbytes:
                raise IngestError(
                    f"{trk_path}: the file ends inside the record of "
                    f"streamline {poly_id_base + s1 - 1}"
                )
            count_word = (offsets[s0:s1] - first_byte) // 4
            is_point = np.ones(words.shape[0], dtype=bool)
            is_point[count_word] = False
            if file_property_words:
                property_word = (
                    count_word + 1 + block_counts * words_per_point
                )[:, None] + np.arange(file_property_words)
                is_point[property_word.ravel()] = False
                if n_properties:
                    properties[s0:s1] = words[property_word]
            points = words[is_point].reshape(n_block, words_per_point)
            del words, is_point

            verts = points[:, :3].astype(np_dtype)
            if affine_tv2ras is not None:
                # Bake trackvis-voxmm → RASmm so the stored geometry is
                # registered to the source image's world space (see module
                # docstring).  A rigid/orthogonal transform, so streamline
                # lengths and endpoints computed below are unaffected.
                verts = (
                    verts @ affine_tv2ras[:3, :3].T + affine_tv2ras[:3, 3]
                ).astype(np_dtype)

            stop = write_cursor + n_block
            verts_mm[write_cursor:stop] = verts
            for key, first, width in file_scalars:
                column = points[:, 3 + first:3 + first + width]
                carry_mm[key][write_cursor:stop] = column[:, 0] if width == 1 else column
            del points
            # Per-vertex carry attributes follow each streamline's own
            # geometry, so they are the one step still taken per streamline.
            if vertex_carry:
                for i in range(s1 - s0):
                    a = int(block_first[i])
                    b = a + int(block_counts[i])
                    line = verts[a:b]
                    for name in vertex_carry:
                        if name == "arc_length":
                            value = arc_length_normalized(line)
                        elif name == "index":
                            value = index_normalized(len(line))
                        else:  # tangent
                            value = compute_tangents(line)
                        carry_mm[name][write_cursor + a:write_cursor + b] = value

            # Segments: a new one wherever the chunk changes or a streamline
            # starts -- split_polyline_at_boundaries, over the whole block.
            # Absolute, unclamped chunk coords: the array grid is derived
            # *from* these (via the exact bounds reported below) rather than
            # the other way round, so every coord is in range by
            # construction.  Clamping to a header-derived grid is what
            # silently mis-binned 87% of a real tractogram into its edge
            # chunks.
            chunk_of = np.floor(verts / chunk_size).astype(np.int64)
            new_segment = np.empty(n_block, dtype=bool)
            new_segment[0] = True
            np.any(chunk_of[1:] != chunk_of[:-1], axis=1, out=new_segment[1:])
            new_segment[block_first] = True
            seg_start = np.flatnonzero(new_segment)
            line_of_segment = np.searchsorted(block_first, seg_start, side="right") - 1
            seg_poly_blocks.append(poly_id_base + s0 + line_of_segment)
            seg_chunk_blocks.append(chunk_of[seg_start])
            seg_count_blocks.append(np.diff(np.append(seg_start, n_block)))
            del chunk_of, new_segment

            if compute_length:
                lengths[s0:s1] = _line_lengths(verts, block_first, block_counts)
            if compute_endpoints:
                starts[s0:s1] = verts[block_first].astype(np.float32)
                ends[s0:s1] = verts[block_first + block_counts - 1].astype(np.float32)

            write_cursor = stop
            s0 = s1

    seg_poly_ids = np.concatenate(seg_poly_blocks or [np.zeros(0, np.int64)])
    seg_chunks = np.concatenate(seg_chunk_blocks or [np.zeros((0, 3), np.int64)])
    seg_vertex_counts = np.concatenate(seg_count_blocks or [np.zeros(0, np.int64)])
    del seg_poly_blocks, seg_chunk_blocks, seg_count_blocks

    # Exact bbox of everything this part wrote, in stored (post-affine) space.
    # One vectorised pass over the part's own memmap, so the coordinator can
    # size the chunk grid from the geometry instead of the header's claim
    # about it.  Taken before the memmap is released, below.
    if write_cursor > 0:
        _written = verts_mm[:write_cursor]
        vert_min = _written.min(axis=0).astype(np.float64)
        vert_max = _written.max(axis=0).astype(np.float64)
    else:
        vert_min = np.full(3, np.inf)
        vert_max = np.full(3, -np.inf)

    # Flush and release the memmap — pages can now be evicted by the OS.
    del verts_mm
    # Same flush/release as verts_mm; the .npy files stay on disk for Phase B.
    carry_mm.clear()

    # Sort this part's segments by target chunk and write the result as two
    # plain .npy files a Phase B worker can memory-map.
    #
    # Phase B used to open every part's COMPRESSED npz once per spatial
    # chunk and mask the whole segment table to find that chunk's rows:
    # chunks x parts decompressions of the same data, each followed by an
    # O(all segments) pass, plus a recomputation of every segment's position
    # within its polyline.  On a 5000-chunk grid with 32 parts that is
    # 160,000 full decodes of a table that never changes.  Sorting it once,
    # here, in parallel, turns the lookup into a slice.
    seg_counts_arr = np.asarray(seg_vertex_counts, dtype=np.int64)
    seg_polys_arr = np.asarray(seg_poly_ids, dtype=np.int64)
    n_segments = int(seg_polys_arr.shape[0])
    if n_segments:
        chunk_cols = seg_chunks
        # Position within the polyline: segments are appended in polyline
        # order, so a run of equal poly ids numbers 0, 1, 2...
        starts_of_poly = np.concatenate(
            [[0], np.flatnonzero(np.diff(seg_polys_arr)) + 1],
        )
        group_of = np.searchsorted(
            starts_of_poly, np.arange(n_segments), side="right",
        ) - 1
        within_poly = np.arange(n_segments) - starts_of_poly[group_of]
        vtx_starts_arr = np.concatenate(
            [[0], np.cumsum(seg_counts_arr)[:-1]],
        ).astype(np.int64)

        order = np.lexsort((
            within_poly, seg_polys_arr,
            chunk_cols[:, 2], chunk_cols[:, 1], chunk_cols[:, 0],
        ))
        sorted_chunks = chunk_cols[order]
        table = np.empty((n_segments, _SEG_COLS), dtype=np.int64)
        table[:, _SEG_POLY_ID] = seg_polys_arr[order]
        table[:, _SEG_WITHIN] = within_poly[order]
        table[:, _SEG_VTX_START] = vtx_starts_arr[order]
        table[:, _SEG_VTX_COUNT] = seg_counts_arr[order]

        change = np.any(sorted_chunks[1:] != sorted_chunks[:-1], axis=1)
        group_start = np.concatenate([[0], np.flatnonzero(change) + 1])
        group_stop = np.concatenate([group_start[1:], [n_segments]])
        directory = np.column_stack([
            sorted_chunks[group_start], group_start, group_stop,
        ]).astype(np.int64)
    else:
        table = np.zeros((0, _SEG_COLS), dtype=np.int64)
        directory = np.zeros((0, 5), dtype=np.int64)
    np.save(_index_path(npz_path), table, allow_pickle=False)
    np.save(_chunk_dir_path(npz_path), directory, allow_pickle=False)

    save_dict: dict[str, Any] = {
        "poly_id_base": np.int64(poly_id_base),
        "n_streamlines": np.int64(len(byte_offsets)),
        "vert_min": vert_min,
        "vert_max": vert_max,
        "seg_poly_ids": seg_polys_arr,
        "seg_chunk_x": seg_chunks[:, 0].astype(np.int32),
        "seg_chunk_y": seg_chunks[:, 1].astype(np.int32),
        "seg_chunk_z": seg_chunks[:, 2].astype(np.int32),
        "seg_vertex_counts": seg_counts_arr.astype(np.int32),
    }
    if compute_length:
        save_dict["lengths"] = lengths
    if compute_endpoints:
        save_dict["starts"] = starts
        save_dict["ends"] = ends
    if compute_vertex_count:
        save_dict["vertex_counts"] = counts.astype(np.uint32)
    if n_properties:
        save_dict["properties"] = properties

    np.savez_compressed(npz_path, **save_dict)
    return npz_path


# ---------------------------------------------------------------------------
# Level-0 layout: where every segment lands, from the Phase-A part files
# ---------------------------------------------------------------------------

#: Columns of the level-0 layout table, one row per (chunk, part) pair that
#: holds segments, ascending by (chunk, part) -- the order Phase B writes
#: fragments in.
_LAY_CHUNK = 0          # index into the sorted occupied-chunk list
_LAY_PART = 1
_LAY_START = 2          # [start, stop) rows of the part's segment table
_LAY_STOP = 3
_LAY_VERTICES = 4       # vertices those rows hold
_LAY_BASE_FRAGMENT = 5  # fragments ahead of this part in the chunk
_LAY_BASE_ROW = 6       # vertex rows ahead of this part in the chunk
_LAY_COLS = 7


def _plan_level0(part_npz_paths: list[str]) -> dict[str, npt.NDArray]:
    """Where Phase B puts every part's segments, from the part directories.

    Each part's directory already gives, per chunk, the slice of its sorted
    segment table bound there.  Put side by side and ordered by (chunk,
    part), those slices are exactly the fragment order Phase B writes a
    cell in, so each slice's first fragment index and first vertex row
    follow by cumulative sums -- which is all the coordinator needs to
    reconstruct manifests and cross-chunk links without being handed a
    per-segment record back from every Phase B task.

    Returns ``chunks`` (C, 3) occupied chunk coords ascending,
    ``chunk_vertices`` and ``chunk_fragments`` (C,), and ``layout``
    (R, _LAY_COLS).
    """
    rows = []
    for part_idx, npz_path in enumerate(part_npz_paths):
        directory = np.load(_chunk_dir_path(npz_path))
        if directory.shape[0] == 0:
            continue
        table = np.load(_index_path(npz_path), mmap_mode="r")
        try:
            cum = np.concatenate(
                [[0], np.cumsum(np.asarray(table[:, _SEG_VTX_COUNT]))],
            )
        finally:
            _close_memmap(table)
        block = np.empty((directory.shape[0], 8), dtype=np.int64)
        block[:, :3] = directory[:, :3]
        block[:, 3] = part_idx
        block[:, 4] = directory[:, 3]
        block[:, 5] = directory[:, 4]
        block[:, 6] = cum[directory[:, 4]] - cum[directory[:, 3]]
        rows.append(block)
    if not rows:
        return {
            "chunks": np.zeros((0, 3), dtype=np.int64),
            "chunk_vertices": np.zeros(0, dtype=np.int64),
            "chunk_fragments": np.zeros(0, dtype=np.int64),
            "layout": np.zeros((0, _LAY_COLS), dtype=np.int64),
        }
    raw = np.concatenate(rows)
    raw = raw[np.lexsort((raw[:, 3], raw[:, 2], raw[:, 1], raw[:, 0]))]
    first_of_chunk = np.ones(raw.shape[0], dtype=bool)
    first_of_chunk[1:] = np.any(raw[1:, :3] != raw[:-1, :3], axis=1)
    chunk_start = np.flatnonzero(first_of_chunk)
    chunk_id = np.cumsum(first_of_chunk) - 1

    layout = np.empty((raw.shape[0], _LAY_COLS), dtype=np.int64)
    layout[:, _LAY_CHUNK] = chunk_id
    layout[:, _LAY_PART] = raw[:, 3]
    layout[:, _LAY_START] = raw[:, 4]
    layout[:, _LAY_STOP] = raw[:, 5]
    layout[:, _LAY_VERTICES] = raw[:, 6]
    for column, per_row in (
        (_LAY_BASE_FRAGMENT, raw[:, 5] - raw[:, 4]),
        (_LAY_BASE_ROW, raw[:, 6]),
    ):
        ahead = np.cumsum(per_row) - per_row
        layout[:, column] = ahead - ahead[chunk_start][chunk_id]
    return {
        "chunks": raw[chunk_start, :3],
        "chunk_vertices": np.add.reduceat(raw[:, 6], chunk_start),
        "chunk_fragments": np.add.reduceat(raw[:, 5] - raw[:, 4], chunk_start),
        "layout": layout,
    }


def _phase_b_tasks(
    plan: dict[str, npt.NDArray], chunks_per_task: int,
) -> list[dict[str, npt.NDArray]]:
    """Cut the occupied chunks, in order, into Phase B tasks.

    Each task names its chunks and, per chunk, the part-table slices it
    reads, so a worker opens each part's files once per task instead of
    once per chunk and never searches a directory.
    """
    chunks = plan["chunks"]
    layout = plan["layout"]
    tasks = []
    for a in range(0, chunks.shape[0], chunks_per_task):
        b = min(a + chunks_per_task, chunks.shape[0])
        lo, hi = np.searchsorted(layout[:, _LAY_CHUNK], [a, b])
        rows = layout[lo:hi]
        slices = np.column_stack([
            rows[:, _LAY_CHUNK] - a, rows[:, _LAY_PART],
            rows[:, _LAY_START], rows[:, _LAY_STOP],
        ])
        tasks.append({"chunks": chunks[a:b].copy(), "slices": slices})
    return tasks


def _close_memmap(arr: Any) -> None:
    """Release a ``np.load(..., mmap_mode=...)`` mapping's file handle.

    Dropping the last Python reference is not enough on Windows: the handle
    survives until GC runs, and until it does the backing file cannot be
    deleted — which is what makes the intermediate-dir cleanup blow up at
    the very end of an otherwise successful ingest.  ``np.memmap`` exposes
    the mapping as ``._mmap``; a plain ndarray (no mmap) has none.
    """
    mm = getattr(arr, "_mmap", None)
    if mm is not None:
        mm.close()


def _ranges_index(starts: npt.NDArray, counts: npt.NDArray) -> npt.NDArray:
    """``concatenate([arange(s, s + c) for s, c in zip(starts, counts)])``."""
    total = int(counts.sum())
    ahead = np.cumsum(counts) - counts
    return np.arange(total, dtype=np.int64) + np.repeat(starts - ahead, counts)


# ---------------------------------------------------------------------------
# Phase B worker: assemble a batch of spatial chunks from all N parts
# ---------------------------------------------------------------------------

def _phase_b_worker(
    task: dict[str, npt.NDArray],
    shared: dict[str, Any],
) -> npt.NDArray:
    """Write a batch of spatial chunks' level-0 data from the Phase-A part files.

    ``task`` is one of :func:`_phase_b_tasks`: ``chunks`` (K, 3) and
    ``slices`` (R, 4) rows of ``(chunk position in task, part, start,
    stop)``, ascending by chunk then part.  Each chunk's fragments are its
    segments in ascending (part, poly_id, seg_idx_within_poly) order -- the
    determinism pin -- which is each part's table slice, in part order.
    Every part the task touches is opened once for the whole task, and
    closed before returning.

    Returns (K, 2) int64 ``(fragments, vertices)`` written per chunk, which
    the coordinator checks against the layout it builds manifests from.
    """
    part_npz_paths = shared["part_npz_paths"]
    store_path = shared["store_path"]
    dtype = shared["dtype"]
    out_dtype = np.dtype(dtype)
    vertex_attrs = shared.get("vertex_attrs", ())
    vertex_carry = shared.get("vertex_carry", ())
    # The file's own scalars: (attribute name, carry key, width).
    file_scalars = shared.get("file_scalars", ())
    carry_keys = list(vertex_carry) + [key for _name, key, _width in file_scalars]
    attr_seed = int(shared.get("attr_seed", 0))

    chunks = np.asarray(task["chunks"], dtype=np.int64).reshape(-1, 3)
    slices = np.asarray(task["slices"], dtype=np.int64).reshape(-1, 4)
    written = np.zeros((chunks.shape[0], 2), dtype=np.int64)
    if slices.shape[0] == 0:
        return written
    row_bounds = np.searchsorted(slices[:, 0], np.arange(chunks.shape[0] + 1))

    # part -> [segment table, vertices, {carry key: values}], all mapped.
    opened: dict[int, list[Any]] = {}
    try:
        for part_idx in np.unique(slices[:, 1]).tolist():
            npz_path = part_npz_paths[part_idx]
            # Registered before each next map, so a failure part-way still
            # closes every handle already taken (see the finally below).
            handles: list[Any] = [None, None, {}]
            opened[part_idx] = handles
            handles[0] = np.load(_index_path(npz_path), mmap_mode="r")
            handles[1] = np.load(npz_path.replace(".npz", ".verts.npy"), mmap_mode="r")
            # Carry-attribute memmaps for this part, gathered with the SAME
            # rows as the vertices (Phase A wrote them in identical order).
            for name in carry_keys:
                handles[2][name] = np.load(
                    _carry_attr_path(npz_path, name), mmap_mode="r",
                )

        root = open_store(str(store_path), mode="r+")
        # Level group must already exist (created in coordinator pre-phase-B)
        level_group = get_resolution_level(root, 0)

        for k in range(chunks.shape[0]):
            lo, hi = int(row_bounds[k]), int(row_bounds[k + 1])
            if lo == hi:
                continue
            chunk_coords = tuple(int(c) for c in chunks[k])
            vert_pieces, count_pieces, poly_pieces = [], [], []
            carry_pieces: dict[str, list[npt.NDArray]] = {n: [] for n in carry_keys}
            for _pos, part_idx, start, stop in slices[lo:hi].tolist():
                table, vertices, carry_mm = opened[part_idx]
                rows = np.asarray(table[start:stop])
                counts = rows[:, _SEG_VTX_COUNT]
                gather = _ranges_index(rows[:, _SEG_VTX_START], counts)
                vert_pieces.append(np.asarray(vertices[gather], dtype=out_dtype))
                for name in carry_keys:
                    carry_pieces[name].append(
                        np.asarray(carry_mm[name][gather], dtype=np.float32),
                    )
                count_pieces.append(counts)
                poly_pieces.append(rows[:, _SEG_POLY_ID])
            counts = np.concatenate(count_pieces)
            split_at = np.cumsum(counts)[:-1]
            chunk_verts = np.concatenate(vert_pieces)
            del vert_pieces
            # One contiguous buffer per chunk, fragments as views of it: the
            # bytes each fragment encodes to are the same as a copy's.
            vert_groups = np.split(chunk_verts, split_at)

            # record_presence=False: Phase B runs tasks across worker
            # PROCESSES.  ``nonempty_chunks`` is an array-wide attribute
            # living in the array's zarr.json, so stamping it per cell is a
            # read-modify-write that concurrent workers collide on (a
            # Windows atomic-rename hard-fail).  The coordinator re-derives
            # every manifest once, after Phase B, via rebuild_presence.
            write_chunk_vertices(
                level_group, chunk_coords, vert_groups, dtype=out_dtype,
                record_presence=False,
            )

            # Per-fragment segment id (global poly_id = streamline index in
            # file), one uint64 per fragment in vert_groups order.
            # Neuroglancer uses this to map a picked spatial fragment back to
            # the full streamline for pass-2 fetch.
            seg_ids = np.concatenate(poly_pieces).astype(np.uint64)
            write_chunk_fragment_attributes(
                level_group, "segment_id", chunk_coords, seg_ids, dtype=np.uint64,
                record_presence=False,
            )

            # Per-vertex attributes: one ragged group per fragment, aligned
            # 1:1 with vert_groups.  Coordinate/random generators are derived
            # here from the assembled positions (no Phase A carry); the
            # geometry-following ones come from the carried arrays.
            if vertex_attrs:
                _axis = {"x": 0, "y": 1, "z": 2}
                for name in sorted(vertex_attrs):
                    if name in _axis:
                        attr_groups = np.split(
                            chunk_verts[:, _axis[name]].astype(np.float32), split_at,
                        )
                    elif name == "random":
                        # Deterministic per (seed, chunk): reproducible across
                        # serial and parallel runs since each chunk is written
                        # by one worker.  Mask to 32-bit unsigned — chunk
                        # coords can be negative after --apply-affine, and
                        # SeedSequence rejects negative seeds.
                        rng = np.random.default_rng(
                            [attr_seed & 0xFFFFFFFF]
                            + [int(c) & 0xFFFFFFFF for c in chunk_coords]
                        )
                        attr_groups = [
                            rng.random(int(n), dtype=np.float32) for n in counts.tolist()
                        ]
                    else:  # carry attribute (arc_length / index / tangent)
                        attr_groups = np.split(
                            np.concatenate(carry_pieces[name]), split_at,
                        )
                    write_chunk_attributes(
                        level_group, name, chunk_coords, attr_groups,
                        dtype=np.float32, record_presence=False,
                    )
            for name, key, _width in file_scalars:
                write_chunk_attributes(
                    level_group, name, chunk_coords,
                    np.split(np.concatenate(carry_pieces[key]), split_at),
                    dtype=np.float32, record_presence=False,
                )
            written[k] = (counts.shape[0], chunk_verts.shape[0])
    finally:
        # Close every mapping explicitly: `del` drops a reference but leaves
        # the file handle open until GC, which on Windows is long enough to
        # block cleanup of the intermediate dir.
        for table, vertices, carry_mm in opened.values():
            _close_memmap(table)
            _close_memmap(vertices)
            for mm in carry_mm.values():
                _close_memmap(mm)
    return written


# ---------------------------------------------------------------------------
# Coordinator: reconstruct manifests and cross-chunk links
# ---------------------------------------------------------------------------

def _build_manifests_and_cross_links(
    part_npz_paths: list[str],
    plan: dict[str, npt.NDArray],
    n_total_streamlines: int,
) -> tuple[dict[int, list], tuple[npt.NDArray, ...]]:
    """Build object manifests and cross-chunk links from the level-0 layout.

    One part at a time: a part's segment table, re-ordered by (poly_id,
    seg_idx_within_poly), gives each streamline's fragments in order, and
    the layout gives each one's fragment index and first vertex row in its
    chunk.  Streamlines never span parts, so the part's links are complete.

    Returns:
        object_manifests  dict[poly_id → list[(chunk_coords, fragment_idx)]]
        cross_chunk_links  ``(src_chunk_id, src_vertex, dst_chunk_id,
            dst_vertex)`` int64 arrays, in (poly_id, segment) order; chunk
            ids index ``plan["chunks"]``.
    """
    chunks = plan["chunks"]
    layout = plan["layout"]
    # One tuple per chunk, shared by every manifest entry that names it.
    chunk_tuples = [tuple(c) for c in chunks.tolist()]
    object_manifests: dict[int, list] = {}
    link_pieces: list[tuple[npt.NDArray, ...]] = []

    by_part = np.argsort(layout[:, _LAY_PART], kind="stable")
    part_bounds = np.searchsorted(
        layout[by_part, _LAY_PART], np.arange(len(part_npz_paths) + 1),
    )
    for part_idx, npz_path in enumerate(part_npz_paths):
        rows = layout[by_part[part_bounds[part_idx]:part_bounds[part_idx + 1]]]
        if rows.shape[0] == 0:
            continue
        rows = rows[np.argsort(rows[:, _LAY_START], kind="stable")]
        table = np.load(_index_path(npz_path))
        n_segments = table.shape[0]
        span = rows[:, _LAY_STOP] - rows[:, _LAY_START]
        if int(span.sum()) != n_segments:
            raise IngestError(
                f"{npz_path}: its chunk directory covers {int(span.sum())} "
                f"of {n_segments} segments"
            )
        counts = table[:, _SEG_VTX_COUNT]
        ahead = np.cumsum(counts) - counts
        seg_chunk = np.repeat(rows[:, _LAY_CHUNK], span)
        seg_fragment = np.arange(n_segments) + np.repeat(
            rows[:, _LAY_BASE_FRAGMENT] - rows[:, _LAY_START], span,
        )
        seg_first = ahead + np.repeat(
            rows[:, _LAY_BASE_ROW] - ahead[rows[:, _LAY_START]], span,
        )

        order = np.lexsort((table[:, _SEG_WITHIN], table[:, _SEG_POLY_ID]))
        poly = table[order, _SEG_POLY_ID]
        seg_chunk = seg_chunk[order]
        seg_fragment = seg_fragment[order]
        seg_first = seg_first[order]
        seg_last = seg_first + counts[order] - 1
        del table, counts, ahead, order

        entries = list(zip(
            map(chunk_tuples.__getitem__, seg_chunk.tolist()),
            seg_fragment.tolist(),
        ))
        poly_start = np.flatnonzero(np.concatenate([[True], poly[1:] != poly[:-1]]))
        bounds = np.append(poly_start, n_segments).tolist()
        for i, poly_id in enumerate(poly[poly_start].tolist()):
            object_manifests[poly_id] = entries[bounds[i]:bounds[i + 1]]
        del entries

        # Consecutive segments of one streamline in different chunks.
        crossing = (poly[1:] == poly[:-1]) & (seg_chunk[1:] != seg_chunk[:-1])
        link_pieces.append((
            seg_chunk[:-1][crossing], seg_last[:-1][crossing],
            seg_chunk[1:][crossing], seg_first[1:][crossing],
        ))

    if len(object_manifests) < n_total_streamlines:
        for poly_id in range(n_total_streamlines):
            object_manifests.setdefault(poly_id, [])
    if link_pieces:
        cross_chunk_links = tuple(
            np.concatenate([piece[i] for piece in link_pieces]) for i in range(4)
        )
    else:
        cross_chunk_links = tuple(np.zeros(0, dtype=np.int64) for _ in range(4))
    return object_manifests, cross_chunk_links


def _cross_links_worker(
    task: list[tuple[tuple[int, ...], npt.NDArray, npt.NDArray]],
    shared: dict[str, Any],
) -> int:
    """Write whole ``links/0/<offsets>`` arrays of level-0 cross-chunk links.

    ``task`` is a list of ``(offset, source_chunks, rows)``: every record
    of one offsets array, ordered by source chunk and, within a chunk, in
    the coordinator's input order -- the cell layout ``write_links`` would
    produce.  A task owns its arrays outright (no other task writes a cell
    of them), so it can also stamp each array's ``nonempty_chunks`` itself:
    one batched flush per array, where a per-cell writer would rewrite the
    array's zarr.json once per cell and a presence rebuild would read every
    cell back.
    """
    root = open_store(str(shared["store_path"]), mode="r+")
    level_group = get_resolution_level(root, 0)
    written = 0
    for offset, sources, rows in task:
        cell_start = np.flatnonzero(np.concatenate(
            [[True], np.any(sources[1:] != sources[:-1], axis=1)],
        ))
        bounds = np.append(cell_start, rows.shape[0]).tolist()
        with level_group.batched_writes():
            for i, first in enumerate(cell_start.tolist()):
                write_chunk_links(
                    level_group, tuple(int(c) for c in sources[first]),
                    [rows[first:bounds[i + 1]]], np.int64,
                    delta=0, offsets=(tuple(offset),), link_width=2,
                )
        written += int(rows.shape[0])
    return written


def _write_level0_cross_links(
    level_group: Any,
    store_path: str,
    chunks: npt.NDArray,
    cross_chunk_links: tuple[npt.NDArray, ...],
    *,
    compressor: Any,
    executor: Any,
    n_tasks: int,
) -> int:
    """Write level 0's cross-chunk links into ``links/0``, one task per array group.

    Byte-for-byte what ``write_links(level_group, records, sid_ndim=3,
    delta=0, mode="replace", directed=True)`` writes for the same records,
    without building a Python record per link.  Directed canonical records
    file under their endpoint-0 chunk, in the array named by the offset to
    endpoint 1 (``links_has_perm`` is False for them, so a row is just the
    two vertex indices), each cell's rows in input order.

    The coordinator creates every offsets array first, serially (creating
    one from inside a worker races on the array's zarr.json), then hands
    whole arrays to tasks -- balanced by record count -- and finally stamps
    the family's counts, which it knows without reading anything back.
    """
    src_chunk, src_vertex, dst_chunk, dst_vertex = cross_chunk_links
    n_links = int(src_vertex.shape[0])
    if n_links == 0:
        return 0
    sources = chunks[src_chunk]
    offsets, offset_id = np.unique(
        chunks[dst_chunk] - sources, axis=0, return_inverse=True,
    )
    offset_id = offset_id.reshape(-1)
    offset_tuples = [tuple(o) for o in offsets.tolist()]
    if any(
        links_has_perm((o,), delta=0, directed=True, store="canonical")
        for o in offset_tuples
    ):  # pragma: no cover - core policy guard
        raise IngestError("directed cross-chunk links unexpectedly carry perm_idx")

    codec_ctx = (
        level_group.batched_writes(compressor=compressor)
        if compressor else nullcontext()
    )
    with codec_ctx:
        for offset in offset_tuples:
            create_links_array(
                level_group, link_width=2, delta=0, sid_ndim=3,
                offsets=(offset,), directed=True,
            )

    # Stable: by array, then source chunk (chunk ids ascend with coords),
    # then input order.
    order = np.lexsort((src_chunk, offset_id))
    offset_id = offset_id[order]
    sources = sources[order]
    rows = np.column_stack([src_vertex[order], dst_vertex[order]]).astype(np.int64)
    array_bounds = np.searchsorted(offset_id, np.arange(len(offset_tuples) + 1))
    sizes = np.diff(array_bounds)

    n_tasks = max(1, min(int(n_tasks), len(offset_tuples)))
    tasks: list[list] = [[] for _ in range(n_tasks)]
    load = [0] * n_tasks
    for a in np.argsort(-sizes, kind="stable").tolist():
        t = load.index(min(load))
        lo, hi = int(array_bounds[a]), int(array_bounds[a + 1])
        tasks[t].append((offset_tuples[a], sources[lo:hi], rows[lo:hi]))
        load[t] += int(sizes[a])
    tasks = [t for t in tasks if t]
    shared = {"store_path": str(store_path)}
    if executor is not None and len(tasks) > 1:
        written = sum(executor(_cross_links_worker, tasks, shared=shared))
    else:
        written = sum(_cross_links_worker(t, shared) for t in tasks)
    if written != n_links:
        raise IngestError(f"wrote {written} of {n_links} cross-chunk links")

    # The family counts write_links stamps; a canonical family's logical and
    # physical counts are both the record count.
    family = links_group_path(0)
    meta = level_group.read_array_meta(family)
    meta["num_links"] = n_links
    meta["num_physical_records"] = n_links
    level_group.write_array_meta(family, meta)
    return n_links


# ---------------------------------------------------------------------------
# Top-level ingest function
# ---------------------------------------------------------------------------

def ingest_trk_parallel(
    input_path: str | Path,
    output_path: str | Path,
    *,
    num_chunks: int | tuple[int, int, int] | None = None,
    n_parts: int | None = None,
    workers: int | None = None,
    executor: Any = None,
    dtype: str = "float32",
    compressor: Any = None,
    max_streamlines: int | None = None,
    compute_length: bool = False,
    compute_endpoints: bool = False,
    object_attrs: set[str] | None = None,
    vertex_attrs: set[str] | None = None,
    attr_seed: int = 0,
    keep_scalars: bool = True,
    keep_properties: bool = True,
    preserve_header: bool = True,
    register_to_rasmm: bool = False,
    build_multiscale: bool = True,
    pyramid_factors: list[tuple[float, float]] | None = None,
    chunk_scale_factors: list[int] | None = None,
    sparsity_strategy: str = "length",
    pyramid_coarsen_mode: str = "rdp",
    pyramid_rdp_tolerances: list[float] | None = None,
    intermediate_dir: str | Path | None = None,
    keep_intermediate: bool = False,
    resume: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Ingest a large TRK file into a zarr-vectors streamline store.

    Memory-bounded: processes the file in parallel byte-range parts, binning
    streamlines into spatial chunks without loading the whole file into RAM.

    Args:
        input_path: Path to the .trk file.
        output_path: Path for the new zarr-vectors store.
        num_chunks: Target total chunk count (integer, near-isotropic default),
            or explicit (nx, ny, nz) 3-tuple, or None (uses 125).
        n_parts: Number of file parts for Phase A parallelism. Controls how
            finely the input file is sliced — does NOT affect how many
            processes run simultaneously (that is ``workers``). Defaults to
            4× workers (fine-grained enough for good load balancing).
        workers: Number of Dask worker processes. None = cpu_count-1.
        executor: Injected executor (func, items, shared) callable. If None,
            either uses dask_executor (when workers>1) or runs serially.
        dtype: Numpy dtype for vertex positions.
        compressor: Codec for the level-0 per-chunk arrays.  ``None``
            (default) stores raw — vertices then cost exactly
            ``n_vertices * ndim * itemsize`` bytes, the same payload a .trk
            holds.  ``"zstd"`` / ``"blosc"`` roughly halve that (measured
            ~2.4x on HCP tract coordinates) at the cost of a slower,
            sync codec-encoding write path.  See
            :func:`zarr_vectors.encoding.compression.resolve_compressor`
            for accepted values (a full codec list also works).  The codec
            pipeline is fixed when each array is created, so every later
            per-cell write — including from parallel workers — encodes to
            match automatically.
        max_streamlines: If set, only ingest the first N streamlines from the
            file (in on-disk order). Useful for quick test runs on a subset
            of a large tractogram; leave None to ingest all streamlines.
        compute_length: Write per-streamline path length to object_attributes.
        compute_endpoints: Write start/end points to object_attributes.
        object_attrs: Synthetic per-object (per-streamline) attributes to
            generate for color-by-object testing.  Any of ``length``,
            ``endpoints``, ``orientation`` (start→end unit vector, 3ch),
            ``tortuosity``, ``vertex_count``.  ``compute_length`` /
            ``compute_endpoints`` are folded in as ``length`` / ``endpoints``.
        vertex_attrs: Synthetic per-vertex (per-point) attributes to generate
            for color-by-vertex testing.  Any of ``arc_length`` (0→1 along the
            streamline), ``x`` / ``y`` / ``z`` (coordinate), ``random``,
            ``index`` (0→1 within the streamline), ``tangent`` (unit direction,
            3ch).  Stored under ``vertex_attributes/<name>`` and carried through
            the pyramid to every level.
        attr_seed: Seed for the ``random`` vertex generator (default 0).
        keep_scalars: Store the file's own per-point scalars as vertex
            attributes, one per name in the header (a multi-column scalar
            stays one attribute).  Carried through the pyramid like the
            synthetic ones.
        keep_properties: Store the file's per-streamline properties as
            object attributes.
        preserve_header: Store TRKHeader (affine, dims) in the zarr store.
        register_to_rasmm: When True, bake the trackvis-voxmm→RASmm affine into
            the stored vertex positions (and chunk grid) and record an identity
            CRS affine, so the store is in registered world space with no
            transform left for the viewer to apply — the layout neuroglancer
            needs.  When False (default) the raw voxmm coordinates are kept and
            the real ``vox_to_ras`` affine is recorded in CRS metadata instead.
        build_multiscale: Build coarser pyramid levels after level 0.
        pyramid_factors: List of (coarsen_factor, sparsity_factor) per level.
            Default: [(8.0, 1.0), (8.0, 1.0)] for two coarser levels. In
            "decimate" mode, coarsen_factor is the decimation stride rather
            than an RDP epsilon multiplier.
        chunk_scale_factors: Per-axis chunk multiplier per level. Default [1, 1].
        sparsity_strategy: "length" or "random" for object dropping.
        pyramid_coarsen_mode: "rdp" (default, Douglas-Peucker simplification)
            or "decimate" (uniform per-object stride decimation, keeping
            every stride-th vertex plus endpoints).
        pyramid_rdp_tolerances: Explicit Douglas-Peucker tolerance per coarser
            level, in the stored coordinates' units (voxmm, or RAS mm with
            ``register_to_rasmm``), one per ``pyramid_factors`` entry.  None
            (default) derives each from the level's bin.  Refused, before
            anything is read, in "decimate" mode, without
            ``build_multiscale``, or when the count does not match.  See
            :func:`zarr_vectors_tools.multiresolution.coarsen.build_pyramid`.
        intermediate_dir: Directory for .npz scratch files. Default: temp dir.
        keep_intermediate: If True, don't delete the intermediate dir on exit.
        resume: Reuse what an earlier run with the same options finished in
            ``intermediate_dir``: the offset scan, Phase A's part files, a
            complete level 0, and the pyramid levels already built.  Needs
            ``intermediate_dir``, which a run given one always keeps and
            records its progress in.  A run whose options differ from the
            recorded ones is refused rather than mixed with them; a
            half-written level 0 or pyramid level is removed and rebuilt.
        progress: Print progress messages to stdout.

    Returns:
        Summary dict with vertex_count, streamline_count, chunk_count, etc.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if resume and intermediate_dir is None:
        raise IngestError(
            "resume needs intermediate_dir: the directory an earlier run kept "
            "its part files and progress record in"
        )
    record = _RunRecord(intermediate_dir, resume=resume)

    # The pyramid runs last, after the whole level-0 ingest, so a tolerance
    # list that cannot apply is refused here rather than hours later.  The
    # geometry check build_pyramid adds cannot fail: this store is always
    # streamlines.
    _pyramid_factors = pyramid_factors or [(8.0, 1.0), (8.0, 1.0)]
    if pyramid_rdp_tolerances is not None and not build_multiscale:
        raise ValueError(
            "pyramid_rdp_tolerances sets the coarser levels' tolerances, but "
            "build_multiscale=False builds no coarser levels"
        )
    pyramid_rdp_tolerances = validate_rdp_tolerances(
        pyramid_rdp_tolerances, n_levels=len(_pyramid_factors),
        coarsen_mode=pyramid_coarsen_mode, name="pyramid_rdp_tolerances",
    )

    # Normalize the attribute request.  ``compute_length`` / ``compute_endpoints``
    # are sugar for the corresponding object attrs, so callers (and the pyramid's
    # "length" sparsity auto-enable) keep working unchanged.
    object_attrs = set(object_attrs or ())
    if compute_length:
        object_attrs.add("length")
    if compute_endpoints:
        object_attrs.add("endpoints")
    vertex_attrs = set(vertex_attrs or ())
    vertex_carry = sorted(vertex_attrs.intersection(_VERTEX_CARRY))

    # Which per-part accumulators Phase A must produce for the object attrs.
    _need_length = bool(object_attrs & {"length", "tortuosity"})
    _need_endpoints = bool(object_attrs & {"endpoints", "orientation", "tortuosity"})
    _need_vertex_count = "vertex_count" in object_attrs

    def _log(msg: str) -> None:
        if progress:
            print(msg)

    # --- Phase 0: header + affine -----------------------------------------
    _log("Phase 0: parsing TRK header...")
    header = parse_trk_header(input_path)
    header_bounds = _compute_bounds_from_header(header)
    file_scalars, file_properties = _plan_file_data(
        header, keep_scalars, keep_properties, vertex_attrs, object_attrs,
    )
    # Register to RASmm world space by baking the trackvis-voxmm→RASmm affine
    # into the geometry (applied per streamline in Phase A).  Derive the store
    # bounds/chunk grid in that same RASmm space so the chunk layout matches the
    # stored coordinates.  Without this the store sits in voxmm — shifted to the
    # chunk-grid origin and axis-flipped relative to the source image.
    affine_tv2ras = _trackvis_to_rasmm_affine(input_path) if register_to_rasmm else None
    if affine_tv2ras is not None:
        header_bounds = _transform_bounds(header_bounds, affine_tv2ras)
    _log(f"  header bbox: {header_bounds[0]} -> {header_bounds[1]} mm")

    # --- Phase 1: offset index + partition --------------------------------
    _log("Phase 1: building offset index (scanning file)...")
    offset_index = record.offset_index(input_path, header, _log)
    n_streamlines = len(offset_index["byte_offset"])
    n_count_hdr = header["n_count"]
    _log(f"  found {n_streamlines} streamlines "
         f"(header says {n_count_hdr if n_count_hdr > 0 else 'unknown'})")

    if max_streamlines is not None and max_streamlines < n_streamlines:
        _log(f"  max_streamlines set: limiting to first {max_streamlines} "
             f"of {n_streamlines} streamlines")
        offset_index = {
            "byte_offset": offset_index["byte_offset"][:max_streamlines],
            "n_points": offset_index["n_points"][:max_streamlines],
            "nbytes": offset_index["nbytes"][:max_streamlines],
            # Sliced too, so a subset run sizes its chunks from the subset it
            # actually ingests rather than the whole file's spread.
            "first_point": offset_index["first_point"][:max_streamlines],
        }
        n_streamlines = len(offset_index["byte_offset"])

    # Chunk *size* only.  One vertex per streamline is a good enough sample of
    # where the tracts are, and the header's declared FOV is not: on a real
    # tractogram the header box was 152 mm across an axis whose tracts ran to
    # 186 mm, so chunks sized from it were both wrong-scaled and offset.  The
    # grid's origin and extent come later, from Phase A's exact bounds.
    seed_bounds = _bounds_of_points(offset_index["first_point"], header_bounds)
    if affine_tv2ras is not None:
        seed_bounds = _transform_bounds(seed_bounds, affine_tv2ras)
    chunk_shape = _compute_chunk_shape(seed_bounds, num_chunks)
    _log(f"  chunk_shape: {chunk_shape}")

    _n_workers = workers if workers and workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    # n_parts controls file split granularity independently of worker count.
    # More parts → better load balancing; excess parts are queued by the executor.
    _n_parts = n_parts if n_parts and n_parts > 0 else max(_n_workers * 4, 16)
    _n_parts = min(_n_parts, n_streamlines)

    part_specs = partition_offset_index(offset_index, _n_parts)
    _log(f"  partitioned into {len(part_specs)} parts")

    # Everything that decides what Phase A and level 0 write.  A resumed run
    # reuses their output only when this matches what was recorded.
    plan = {
        "input": record.input_fingerprint(input_path),
        "chunk_shape": [float(c) for c in chunk_shape],
        "n_parts": len(part_specs),
        "max_streamlines": max_streamlines,
        "dtype": str(np.dtype(dtype)),
        "compressor": repr(compressor),
        "register_to_rasmm": bool(register_to_rasmm),
        "vertex_attrs": sorted(vertex_attrs),
        "object_attrs": sorted(object_attrs),
        "keep_scalars": bool(keep_scalars),
        "keep_properties": bool(keep_properties),
        "attr_seed": int(attr_seed),
        "preserve_header": bool(preserve_header),
    }
    record.check_plan(plan)

    # --- Setup intermediate directory -------------------------------------
    _own_tempdir = intermediate_dir is None
    if _own_tempdir:
        # ignore_cleanup_errors: by the time cleanup runs the store is written
        # and the summary returned, so a straggling intermediate file must
        # never fail the ingest.  Every read site closes its handle explicitly
        # (see _close_memmap and the `with np.load(...)` blocks); this is the
        # backstop, and it costs temp space rather than correctness.
        _tmpdir_obj = tempfile.TemporaryDirectory(
            prefix="trk_parallel_", ignore_cleanup_errors=True,
        )
        _intermediate_dir = _tmpdir_obj.name
    else:
        _intermediate_dir = str(intermediate_dir)
        Path(_intermediate_dir).mkdir(parents=True, exist_ok=True)
        _tmpdir_obj = None

    try:
        # --- Phase A: per-part spatial binning ---------------------------
        # Runs before the store exists.  Binning needs only chunk_shape, and
        # deferring array creation until Phase A has reported the geometry's
        # real extent is what lets the grid be derived from the data.  It also
        # means a rejected grid (below) leaves no store behind at all.
        _log(f"Phase A: binning streamlines ({len(part_specs)} parts)...")

        shared_a = {
            "trk_path": str(input_path),
            "header": header,
            "chunk_shape": chunk_shape,
            "affine_tv2ras": affine_tv2ras,
            "intermediate_dir": _intermediate_dir,
            "dtype": dtype,
            "compute_length": _need_length,
            "compute_endpoints": _need_endpoints,
            "compute_vertex_count": _need_vertex_count,
            "vertex_carry": vertex_carry,
            "file_scalars": [(key, first, width) for _n, key, first, width in file_scalars],
            "keep_properties": bool(file_properties),
            "part_specs": part_specs,
        }

        def _run_phase_a_serial(items: list[int], shared: dict) -> list[str]:
            return [_phase_a_worker(i, shared) for i in items]

        part_indices = list(range(len(part_specs)))

        reused = record.finished_parts(plan)
        if reused is not None:
            part_npz_paths = reused
            _log(f"  reusing {len(part_npz_paths)} part files from an earlier run")
        else:
            record.forget("phase_a", "level0", "pyramid")
            if executor is not None:
                part_npz_paths = executor(_phase_a_worker, part_indices, shared=shared_a)
            else:
                part_npz_paths = _run_phase_a_serial(part_indices, shared_a)
            record.set("phase_a", {"plan": plan, "parts": [str(p) for p in part_npz_paths]})

        _log(f"  wrote {len(part_npz_paths)} intermediate files")

        # --- Grid layout, from the geometry ------------------------------
        # Union of the exact per-part bboxes.  A TRK header's dim x voxel_size
        # is a *declared* FOV and nothing checks it against the tracts: on a
        # real 5.6M-streamline file 87% of vertices sat outside it, and a grid
        # built from it therefore had to fold them into its edge chunks —
        # storing them under coords that do not contain them, and piling 415M
        # of them into one cell.  The data decides the grid instead.
        _vmin = np.full(3, np.inf)
        _vmax = np.full(3, -np.inf)
        for npz_path in part_npz_paths:
            with np.load(npz_path) as data:
                _vmin = np.minimum(_vmin, data["vert_min"])
                _vmax = np.maximum(_vmax, data["vert_max"])
        if not np.all(np.isfinite(_vmin)):   # no vertices at all
            bounds = header_bounds
        else:
            bounds = (_vmin.tolist(), _vmax.tolist())

        # Same (origin, grid_shape) the single-array layout derives for the
        # on-disk arrays: floor-based, +1 to hold a point exactly on the max
        # boundary.  Because ``bounds`` is the exact extent of what Phase A
        # binned, every chunk coord it recorded is inside this grid.
        _grid_origin, grid_shape = level_grid_layout(bounds, chunk_shape)
        _log(f"  bounds: {bounds[0]} -> {bounds[1]} mm")
        _log(f"  grid: {grid_shape} = {grid_shape[0]*grid_shape[1]*grid_shape[2]} chunks")
        if not _bounds_contain(header_bounds, bounds):
            _log("  note: geometry falls outside the TRK header's declared "
                 "bbox; grid sized to the geometry")

        # Enumerate all occupied chunk coords across all parts, totalling the
        # vertices bound for each — those totals are exactly the cell payloads
        # Phase B is about to write, so this is the last point at which an
        # over-limit chunk can be caught before hours of writing.
        # Read off the per-part chunk directories (see _plan_level0), which
        # also lay out every fragment Phase B writes and the coordinator
        # indexes.
        level0_plan = _plan_level0(part_npz_paths)
        chunk_vertex_counts: dict[tuple[int, int, int], int] = dict(zip(
            map(tuple, level0_plan["chunks"].tolist()),
            level0_plan["chunk_vertices"].tolist(),
        ))
        n_chunks = len(chunk_vertex_counts)
        _log(f"  {n_chunks} occupied spatial chunks")

        # A cell at or over 4 GiB is written without complaint and read back
        # truncated (see _cell_limits), so refuse the grid rather than emit a
        # store that fails much later, mid-pyramid, with a reshape error.
        # Ahead of store creation, so a refusal writes nothing at all.
        check_vertex_cell_limit(
            chunk_vertex_counts,
            ndim=3,
            itemsize=np.dtype(dtype).itemsize,
            grid_shape=grid_shape,
            chunk_shape=chunk_shape,
        )

        # --- Level 0 (store, Phase B, manifests, attributes, header) ------
        level0_done = record.level0_done(plan, output_path)
        if level0_done:
            _log("Level 0: complete from an earlier run, reusing it")
            root = open_store(str(output_path), mode="r+")
            n_cross_links = int(record.get("level0")["cross_chunk_link_count"])
        else:
            record.forget("level0", "pyramid")
            if record.earlier_level0_attempt and _looks_like_store(output_path):
                # What an interrupted run left of level 0 cannot be told apart
                # from a finished one cell by cell; it is rebuilt from the part
                # files, which is the cheap half.
                _log(f"  removing the partial store an earlier run left at {output_path}")
                shutil.rmtree(output_path)
            # --- Create zarr-vectors store ------------------------------------
            _log("Creating zarr-vectors store...")
            vox_to_ras = header["vox_to_ras"]
            if affine_tv2ras is not None:
                # Geometry is already baked to RASmm, so the store-space transform
                # is the identity — a crs-aware viewer applying it must be a no-op
                # (otherwise it would double-transform).  The original voxmm→RASmm
                # affine is preserved in the Phase-5 TRKHeader metadata for
                # provenance / round-trip export.
                crs_dict = {
                    "input_space": "RASmm",
                    "output_space": "RASmm",
                    "units": "mm",
                    "affine": np.eye(4, dtype=np.float64).flatten().tolist(),
                }
            else:
                crs_dict = {
                    "input_space": "voxmm",
                    "output_space": "RASmm",
                    "units": "mm",
                    "affine": vox_to_ras.flatten().tolist(),
                }
            root = create_store(
                str(output_path),
                bounds=bounds,
                chunk_shape=chunk_shape,
                # create_store warm-creates vertices/vertex_fragments, and a chunk
                # array's codecs are fixed at creation — so the compressor has to
                # be set HERE or the store's largest arrays stay raw forever.
                compressor=compressor,
                axes=[
                    {"name": "x", "type": "space", "unit": "mm"},
                    {"name": "y", "type": "space", "unit": "mm"},
                    {"name": "z", "type": "space", "unit": "mm"},
                ],
                geometry_types=["streamline"],
                links_convention="implicit_sequential",
                cross_chunk_strategy="explicit_links",
                crs=crs_dict,
            )

            # Pre-create level 0 arrays (Phase B workers need the level group)
            _log("Creating level-0 arrays...")
            level_meta = LevelMetadata(
                level=0,
                vertex_count=0,  # updated after Phase B
                # "fragment_attributes" listed up front (not just in the
                # best-effort re-stamp near the end of this function): segment_id
                # is created right below and neuroglancer's reader gates whether
                # it even attempts to fetch per-fragment segment ids on this
                # list being accurate — a level left showing the initial
                # placeholder here (e.g. if a later pipeline stage crashes
                # before the re-stamp runs, as happened when Phase 6 OOM'd on a
                # real run) silently falls back to a meaningless per-chunk
                # fragment index for picking/selection.
                arrays_present=(
                    ["vertices", "object_index", "fragment_attributes"]
                    + (["vertex_attributes"] if vertex_attrs or file_scalars else [])
                ),
            )
            level_group = create_resolution_level(root, 0, level_meta)
            # A chunk array's codec pipeline is fixed when the array is CREATED
            # (it lands in the array's zarr.json), and every later write — the
            # Phase B workers included, in their own processes — encodes to match
            # it via zarr's array API.  So the compressor only has to be active
            # here, around creation; there is nothing to thread into the workers.
            # ``nullcontext`` when no compressor is asked for, so the default path
            # is byte-for-byte what it was before this option existed.
            _codec_ctx = (
                level_group.batched_writes(compressor=compressor)
                if compressor else nullcontext()
            )
            with _codec_ctx:
                create_vertices_array(level_group, dtype=dtype)
                create_object_index_array(level_group)
                # Stamp the family policy only; streamline connectivity is
                # implicit-sequential within a fragment, so there are no intra
                # records and the only arrays that should appear under links/0 are
                # the ones the boundary-crossing records actually land in.
                create_links_family(
                    level_group, delta=0, link_width=2, sid_ndim=3, directed=True,
                )
                create_fragment_attribute_array(
                    level_group, "segment_id", dtype="uint64",
                )
                # Pre-create each per-vertex attribute array once, single-process,
                # so the concurrent Phase B workers only WRITE cells (creating here
                # would race on the array's zarr.json — same reason segment_id is
                # created up front).  Metadata-only; cells are written in Phase B.
                for _name in sorted(vertex_attrs):
                    _cn = _TANGENT_CHANNELS if _name == "tangent" else None
                    create_attribute_array(
                        level_group, _name, dtype="float32", channel_names=_cn,
                    )
                for _name, _key, _first, _width in file_scalars:
                    create_attribute_array(
                        level_group, _name, dtype="float32", ncols=_width,
                    )

            # --- Phase B: level-0 write, a batch of chunks per task ----------
            _log(f"Phase B: writing level-0 ({n_chunks} chunks)...")

            shared_b = {
                "part_npz_paths": part_npz_paths,
                "store_path": str(output_path),
                "dtype": dtype,
                "vertex_attrs": sorted(vertex_attrs),
                "vertex_carry": vertex_carry,
                "file_scalars": [(name, key, width) for name, key, _f, width in file_scalars],
                "attr_seed": int(attr_seed),
            }

            # A task opens every part it reads once, so batching chunks saves
            # about parts x files opens per chunk; with few parts there is
            # nothing to save and one chunk per task keeps tasks fine-grained.
            # Parallel runs keep a few tasks per worker for load balance.
            chunks_per_task = max(1, min(64, len(part_npz_paths) // 2))
            if executor is not None:
                chunks_per_task = max(1, min(
                    chunks_per_task, -(-n_chunks // (_n_workers * 4)),
                ))
            phase_b_tasks = _phase_b_tasks(level0_plan, chunks_per_task)
            if executor is not None:
                phase_b_results = executor(_phase_b_worker, phase_b_tasks, shared=shared_b)
            else:
                phase_b_results = [_phase_b_worker(t, shared_b) for t in phase_b_tasks]
            written = (
                np.concatenate(phase_b_results) if phase_b_results
                else np.zeros((0, 2), dtype=np.int64)
            )
            if not (
                np.array_equal(written[:, 0], level0_plan["chunk_fragments"])
                and np.array_equal(written[:, 1], level0_plan["chunk_vertices"])
            ):
                raise IngestError(
                    "Phase B wrote different fragment or vertex counts than the "
                    "part files lay out; the level-0 manifests would not match "
                    "the cells"
                )
            del phase_b_tasks, phase_b_results, written

            # Phase B workers each read-modify-write the shared per-chunk arrays'
            # ``nonempty_chunks`` manifest from separate processes, so those RMWs
            # race and can under-report even though every cell file is on disk.
            # Re-derive the manifests single-process from the on-disk cells before
            # anything downstream enumerates chunks (coarsening source scan,
            # algorithms readers).
            rebuild_presence(level_group)

            # --- Coordinator: manifests + cross-chunk links -------------------
            _log("Coordinator: building manifests and cross-chunk links...")
            object_manifests, cross_chunk_links = _build_manifests_and_cross_links(
                part_npz_paths, level0_plan, n_streamlines,
            )
            n_cross_links = int(cross_chunk_links[0].shape[0])

            _log(f"  writing object index ({n_streamlines} streamlines)...")
            write_object_index(level_group, object_manifests, sid_ndim=3)
            del object_manifests

            if n_cross_links:
                _log(f"  writing {n_cross_links} cross-chunk links...")
                # No capability stamp: these are cross-CHUNK links at delta 0.
                # Since the links merge that is just a non-zero offsets segment
                # in the ordinary links/0 family, and multiscale_links marks the
                # presence of delta != 0 (cross-LEVEL) arrays only.
                _write_level0_cross_links(
                    level_group, str(output_path), level0_plan["chunks"],
                    cross_chunk_links, compressor=compressor, executor=executor,
                    n_tasks=_n_workers if executor is not None else 1,
                )
            del cross_chunk_links

            # Object attributes (per-streamline).  Gather the per-part accumulators
            # once, then derive each requested attribute in global object order —
            # parts are concatenated in ascending part index, which is ascending
            # poly_id (== object id), so row i lines up with object i.
            obj_attrs_to_write: dict[str, npt.NDArray] = {}

            def _concat_part_arrays(key: str, cols: int) -> npt.NDArray:
                parts = []
                for npz_path in part_npz_paths:
                    with np.load(npz_path) as d:
                        if key in d:
                            parts.append(d[key])
                if parts:
                    return np.concatenate(parts, axis=0)
                return np.zeros((0,) if cols == 1 else (0, cols), dtype=np.float32)

            if object_attrs:
                from zarr_vectors_tools.convert.ingest._polyline_enrichments import (
                    orientation_from_endpoints,
                    tortuosity_from_endpoints,
                )

                lengths_all = _concat_part_arrays("lengths", 1) if _need_length else None
                starts_all = _concat_part_arrays("starts", 3) if _need_endpoints else None
                ends_all = _concat_part_arrays("ends", 3) if _need_endpoints else None

                if "length" in object_attrs and lengths_all is not None:
                    obj_attrs_to_write["length"] = lengths_all
                if "endpoints" in object_attrs and starts_all is not None:
                    obj_attrs_to_write["start"] = starts_all
                    obj_attrs_to_write["end"] = ends_all
                if "orientation" in object_attrs and starts_all is not None:
                    obj_attrs_to_write["orientation"] = orientation_from_endpoints(
                        starts_all, ends_all,
                    )
                if "tortuosity" in object_attrs and starts_all is not None:
                    obj_attrs_to_write["tortuosity"] = tortuosity_from_endpoints(
                        starts_all, ends_all, lengths_all,
                    )
                if "vertex_count" in object_attrs:
                    obj_attrs_to_write["vertex_count"] = _concat_part_arrays(
                        "vertex_counts", 1,
                    ).astype(np.uint32)

            if file_properties:
                # One row per streamline, every property column side by side;
                # each named property takes its own columns out of it.
                table = _concat_part_arrays("properties", int(header["n_properties"]))
                for prop_name, first, width in file_properties:
                    column = table[:, first:first + width]
                    obj_attrs_to_write[prop_name] = column[:, 0] if width == 1 else column

            if obj_attrs_to_write:
                from zarr_vectors.building import write_object_attributes

                from zarr_vectors_tools.multiresolution.coarsen import (
                    create_object_attributes_array,
                )
                for attr_name, data in obj_attrs_to_write.items():
                    create_object_attributes_array(level_group, attr_name)
                    write_object_attributes(level_group, attr_name, data)

            # --- Phase 5: header + affine metadata ---------------------------
            if preserve_header:
                try:
                    from zarr_vectors_tools.headers.formats import TRKHeader
                    from zarr_vectors_tools.headers.registry import HeaderRegistry
                    trk_header = TRKHeader(
                        voxel_size=header["voxel_size"],
                        dimensions=header["dim"],
                        vox_to_ras=vox_to_ras.flatten().tolist(),
                        voxel_order=header["voxel_order"],
                        n_scalars=header["n_scalars"],
                        scalar_names=[name for name, _width in header["scalar_names"]],
                        n_properties=header["n_properties"],
                        property_names=[name for name, _width in header["property_names"]],
                        n_count=n_streamlines,
                        origin=list(header["origin"]),
                        version=header["version"],
                        # What an export has to undo: the coordinates as the file
                        # had them, or already registered to RAS millimetres.
                        space="rasmm" if register_to_rasmm else "voxmm",
                    )
                    reg = HeaderRegistry(str(output_path))
                    reg.add("trk", trk_header)
                except Exception:
                    pass  # best-effort

            record.set("level0", {"plan": plan, "cross_chunk_link_count": n_cross_links})

        # --- Phase 6: multiscale pyramid ---------------------------------
        pyramid_summary: dict[str, Any] = {}
        if build_multiscale:
            _log("Phase 6: building multiscale pyramid...")
            _chunk_scale_factors = chunk_scale_factors or [1] * len(_pyramid_factors)
            pyramid_plan = {
                "factors": [[float(c), float(s)] for c, s in _pyramid_factors],
                "chunk_scale_factors": [
                    list(f) if isinstance(f, (tuple, list)) else int(f)
                    for f in _chunk_scale_factors
                ],
                "sparsity_strategy": sparsity_strategy,
                "coarsen_mode": pyramid_coarsen_mode,
                "rdp_tolerances": pyramid_rdp_tolerances,
                "compressor": repr(compressor),
            }
            start_level = record.pyramid_start(pyramid_plan, root, level0_done, _log)

            def _level_done(level: int, _summary: dict[str, Any]) -> None:
                record.set("pyramid", {"plan": pyramid_plan, "levels_done": level})

            pyramid_summary = build_pyramid(
                str(output_path),
                factors=_pyramid_factors,
                chunk_scale_factors=_chunk_scale_factors,
                sparsity_strategy=sparsity_strategy,
                coarsen_mode=pyramid_coarsen_mode,
                rdp_tolerances=pyramid_rdp_tolerances,
                start_level=start_level,
                on_level_done=_level_done,
                # Coarser levels create their own arrays, so they need the
                # same compressor to keep the store uniform — level 0's codec
                # does not propagate.
                compressor=compressor,
                executor=executor,
            )

        # Every vertex lands in exactly one chunk's cell.
        total_vertices = int(level0_plan["chunk_vertices"].sum())

        # Stamp level-0 metadata now that we know the true vertex count and
        # which arrays are present (create_resolution_level runs before Phase B
        # and leaves vertex_count=0 as a placeholder).
        # Was a raw ``import zarr`` inside ``except Exception: pass`` --
        # the only reach past the API left in this package -- because core
        # offered nothing between create_resolution_level and
        # read_level_metadata.  ``refresh_arrays_present`` also derives the
        # family list from what is on disk rather than from a hand-kept
        # candidate list, which is how ``vertex_fragments`` used to go
        # undeclared.
        _lvl0 = get_resolution_level(root, 0)
        refresh_arrays_present(_lvl0)
        update_level_metadata(_lvl0, vertex_count=total_vertices)

        _log("Done.")

        return {
            "streamline_count": n_streamlines,
            "vertex_count": total_vertices,
            "chunk_count": n_chunks,
            # Matches the key core's write_polylines returns, so the CLI
            # prints the same field whichever ingest path produced it.
            # "cross-chunk link" still names the concept (a record whose
            # endpoints span chunks); the merge changed where it is stored,
            # not what it counts.
            "cross_chunk_link_count": n_cross_links,
            "n_parts": len(part_specs),
            "chunk_shape": chunk_shape,
            "bounds": bounds,
            "pyramid": pyramid_summary,
        }

    finally:
        if _tmpdir_obj is not None and not keep_intermediate:
            _tmpdir_obj.cleanup()
