"""Tractogram files read in one pass, as batches of streamlines in RAS+ mm.

The same arithmetic as the browser's reader (ngplatform's Zarr Vectors
plugin, ``src/virtual/``), so both make the same store from a file:

- ``.trk`` (TrackVis): voxmm points to RAS+ mm as nibabel does
  (``get_affine_trackvis_to_rasmm``), computed in float64 and stored as
  float32;
- ``.tck`` (MRtrix): float32 points, NaN between streamlines;
- ``.trx``: a zip of arrays (positions, offsets, ``dpv/``, ``dps/``).

Each reader yields ``Batch`` objects: ``points`` (float32, N x 3),
``offsets`` (N streamlines + 1), ``scalars`` (N points x values) and
``properties`` (streamlines x values).
"""

from __future__ import annotations

import json
import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

TRK_HEADER_SIZE = 1000


@dataclass
class Batch:
    points: np.ndarray
    offsets: np.ndarray
    scalars: np.ndarray
    properties: np.ndarray


@dataclass
class Tractogram:
    """A tractogram file opened for reading: what the store needs to know first."""

    path: Path
    scalar_names: list[str]
    property_names: list[str]
    expected_count: int
    extent: float
    batches: "callable[[], Iterator[Batch]]" = field(repr=False)


def _mat_mul(a: list[float], b: list[float]) -> list[float]:
    """Row-major 4 x 4 product, summed in the browser reader's order."""
    out = [0.0] * 16
    for r in range(4):
        for c in range(4):
            for k in range(4):
                out[4 * r + c] += a[4 * r + k] * b[4 * k + c]
    return out


_AXES = {"R": (0, 1), "L": (0, -1), "A": (1, 1), "P": (1, -1), "S": (2, 1), "I": (2, -1)}


def trk_to_ras(dims, voxel_size, vox_to_ras, voxel_order) -> list[float]:
    """nibabel's get_affine_trackvis_to_rasmm, row-major."""
    m = [1 / voxel_size[0], 0, 0, -0.5, 0, 1 / voxel_size[1], 0, -0.5, 0, 0, 1 / voxel_size[2], -0.5, 0, 0, 0, 1]
    affine = list(vox_to_ras)
    if affine[15] == 0:
        affine = [voxel_size[0], 0, 0, 0, 0, voxel_size[1], 0, 0, 0, 0, voxel_size[2], 0, 0, 0, 0, 1]
    affine_axes = []
    for c in range(3):
        column = [affine[c], affine[4 + c], affine[8 + c]]
        world = max(range(3), key=lambda r: abs(column[r]))
        affine_axes.append((world, 1 if column[world] >= 0 else -1))
    header = [_AXES.get(ch.upper()) for ch in (voxel_order or "RAS")[:3]]
    if all(h is not None for h in header) and len({h[0] for h in header}) == 3:
        reorder = [0.0] * 16
        reorder[15] = 1
        for i, (world, sign) in enumerate(header):
            j = next((j for j, (w, _) in enumerate(affine_axes) if w == world), -1)
            if j == -1:
                continue
            if sign == affine_axes[j][1]:
                reorder[4 * j + i] = 1
            else:
                reorder[4 * j + i] = -1
                reorder[4 * j + 3] = dims[i] - 1
        m = _mat_mul(reorder, m)
    return _mat_mul(affine, m)


def _names(raw: bytes, n: int) -> list[str]:
    out: list[str] = []
    for i in range(10):
        if len(out) >= n:
            break
        cell = raw[20 * i : 20 * i + 20]
        name, _, rest = cell.partition(b"\0")
        count = rest.split(b"\0")[0]
        repeat = int(count) if count.isdigit() and int(count) > 1 else 1
        label = name.decode("latin1").strip() or f"value{len(out)}"
        for k in range(repeat):
            if len(out) < n:
                out.append(label if repeat == 1 else f"{label}_{k}")
    while len(out) < n:
        out.append(f"value{len(out)}")
    return out


def _apply(points64: np.ndarray, t: list[float]) -> np.ndarray:
    x, y, z = points64[:, 0], points64[:, 1], points64[:, 2]
    out = np.empty(points64.shape, np.float32)
    for r in range(3):
        out[:, r] = ((t[4 * r] * x + t[4 * r + 1] * y) + t[4 * r + 2] * z) + t[4 * r + 3]
    return out


def open_trk(path: str | Path, per_batch: int = 100_000) -> Tractogram:
    path = Path(path)
    with open(path, "rb") as f:
        head = f.read(TRK_HEADER_SIZE)
    if len(head) < TRK_HEADER_SIZE or not head.startswith(b"TRACK"):
        raise ValueError(f"{path}: not a TrackVis .trk file")
    little = struct.unpack("<i", head[996:1000])[0] == TRK_HEADER_SIZE
    e = "<" if little else ">"
    if not little and struct.unpack(">i", head[996:1000])[0] != TRK_HEADER_SIZE:
        raise ValueError(f"{path}: a .trk header whose size is not 1000 bytes")
    dims = struct.unpack(e + "3h", head[6:12])
    voxel_size = [s if s > 0 else 1.0 for s in struct.unpack(e + "3f", head[12:24])]
    n_scalars = struct.unpack(e + "h", head[36:38])[0]
    n_properties = struct.unpack(e + "h", head[238:240])[0]
    vox_to_ras = list(struct.unpack(e + "16f", head[440:504]))
    voxel_order = head[948:952].split(b"\0")[0].decode("latin1").strip()
    count = struct.unpack(e + "i", head[988:992])[0]
    to_ras = trk_to_ras(dims, voxel_size, vox_to_ras, voxel_order)
    scalar_names = _names(head[38:238], n_scalars)
    property_names = _names(head[240:440], n_properties)

    def batches() -> Iterator[Batch]:
        data = np.fromfile(path, dtype=np.uint8, offset=TRK_HEADER_SIZE)
        words = data[: len(data) - len(data) % 4].view(e + "f4")
        ints = data[: len(data) - len(data) % 4].view(e + "i4")
        at = 0
        while at < len(words):
            starts, counts = [], []
            while at < len(words) and len(starts) < per_batch:
                n = int(ints[at])
                if n < 0 or at + 1 + n * (3 + n_scalars) + n_properties > len(words):
                    raise ValueError(f"{path}: a broken streamline record at byte {TRK_HEADER_SIZE + 4 * at}")
                starts.append(at + 1)
                counts.append(n)
                at += 1 + n * (3 + n_scalars) + n_properties
            offsets = np.zeros(len(counts) + 1, np.uint32)
            offsets[1:] = np.cumsum(counts)
            raw = np.concatenate([words[s : s + n * (3 + n_scalars)].reshape(n, 3 + n_scalars) for s, n in zip(starts, counts)]) if counts else np.zeros((0, 3 + n_scalars), np.float32)
            props = np.array([words[s + n * (3 + n_scalars) : s + n * (3 + n_scalars) + n_properties] for s, n in zip(starts, counts)], np.float32).reshape(len(counts), n_properties)
            points = _apply(raw[:, :3].astype(np.float64), to_ras)
            yield Batch(points, offsets, raw[:, 3:].astype(np.float32), props)

    extent = max(d * s for d, s in zip(dims, voxel_size))
    return Tractogram(path, scalar_names, property_names, count, float(extent), batches)


def open_tck(path: str | Path, per_batch: int = 100_000) -> Tractogram:
    path = Path(path)
    with open(path, "rb") as f:
        head = f.read(64 * 1024).decode("latin1")
    if not head.startswith("mrtrix tracks"):
        raise ValueError(f"{path}: not an MRtrix .tck file")
    end = head.find("\nEND")
    fields = {}
    for line in head[:end].split("\n")[1:]:
        key, sep, value = line.partition(":")
        if sep:
            fields[key.strip()] = value.strip()
    where = fields.get("file", "").split()
    if len(where) != 2 or where[0] != ".":
        raise ValueError(f"{path}: a .tck whose data is not in the same file")
    offset = int(where[1])
    datatype = fields.get("datatype", "Float32LE")
    dtype = ">f4" if datatype.upper().endswith("BE") else "<f4"
    count = int(fields.get("count", "0") or 0)

    def batches() -> Iterator[Batch]:
        data = np.fromfile(path, dtype=np.uint8, offset=offset)
        triplets = data[: len(data) - len(data) % 12].view(dtype).reshape(-1, 3)
        x = triplets[:, 0]
        stop = np.flatnonzero(np.isinf(x))
        if len(stop):
            triplets = triplets[: stop[0]]
            x = triplets[:, 0]
        breaks = np.flatnonzero(np.isnan(x))
        bounds = np.concatenate([[-1], breaks, [len(triplets)] if len(breaks) == 0 or breaks[-1] != len(triplets) - 1 else []]).astype(np.int64)
        lines = [(int(a) + 1, int(b)) for a, b in zip(bounds[:-1], bounds[1:])]
        for first in range(0, len(lines), per_batch):
            part = lines[first : first + per_batch]
            counts = [b - a for a, b in part]
            offsets = np.zeros(len(part) + 1, np.uint32)
            offsets[1:] = np.cumsum(counts)
            points = np.concatenate([triplets[a:b] for a, b in part]).astype(np.float32) if part else np.zeros((0, 3), np.float32)
            yield Batch(points, offsets, np.zeros((len(points), 0), np.float32), np.zeros((len(part), 0), np.float32))

    return Tractogram(path, [], [], count, 0.0, batches)


_DTYPES = {"float16", "float32", "float64", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"}


def _array_name(name: str):
    parts = name.split(".")
    if len(parts) < 2 or parts[-1] not in _DTYPES:
        return None
    if len(parts) >= 3 and parts[-2].isdigit():
        return ".".join(parts[:-2]), int(parts[-2]), parts[-1]
    return ".".join(parts[:-1]), 1, parts[-1]


def open_trx(path: str | Path, per_batch: int = 100_000) -> Tractogram:
    path = Path(path)
    archive = zipfile.ZipFile(path)
    entries = {i.filename.split(".trx/", 1)[-1]: i for i in archive.infolist()}
    header = json.loads(archive.read(entries["header.json"]))
    arrays = {}
    for name, info in entries.items():
        folder, _, base = name.rpartition("/")
        parsed = _array_name(base)
        if parsed is not None:
            arrays[(folder, parsed[0])] = (info, parsed[1], parsed[2])
    pos_info, columns, pos_dtype = arrays[("", "positions")]
    off_info, _, off_dtype = arrays[("", "offsets")]
    scalars = [(n, i, d) for (f, n), (i, c, d) in sorted(arrays.items()) if f == "dpv" and c == 1]
    properties = [(n, i, d) for (f, n), (i, c, d) in sorted(arrays.items()) if f == "dps" and c == 1]
    count = int(header.get("NB_STREAMLINES", 0))
    voxel_to_ras = np.asarray(header.get("VOXEL_TO_RASMM", np.eye(4)), np.float64)
    dims = header.get("DIMENSIONS", [0, 0, 0])
    sizes = [float(np.hypot.reduce(voxel_to_ras[:3, c])) or 1.0 for c in range(3)]
    extent = max([0.0] + [float(d) * s for d, s in zip(dims, sizes)])

    def batches() -> Iterator[Batch]:
        positions = np.frombuffer(archive.read(pos_info), dtype="<" + np.dtype(pos_dtype).str[1:]).reshape(-1, 3)
        offsets = np.frombuffer(archive.read(off_info), dtype="<" + np.dtype(off_dtype).str[1:]).astype(np.int64)
        n = count or len(offsets)
        if len(offsets) == n:
            offsets = np.concatenate([offsets, [len(positions)]])
        point_values = [np.frombuffer(archive.read(i), dtype="<" + np.dtype(d).str[1:]).astype(np.float64) for _, i, d in scalars]
        line_values = [np.frombuffer(archive.read(i), dtype="<" + np.dtype(d).str[1:]).astype(np.float64) for _, i, d in properties]
        for first in range(0, n, per_batch):
            last = min(n, first + per_batch)
            v0, v1 = int(offsets[first]), int(offsets[last])
            yield Batch(
                positions[v0:v1].astype(np.float64).astype(np.float32),
                (offsets[first : last + 1] - v0).astype(np.uint32),
                np.stack([v[v0:v1] for v in point_values], axis=1).astype(np.float32) if point_values else np.zeros((v1 - v0, 0), np.float32),
                np.stack([v[first:last] for v in line_values], axis=1).astype(np.float32) if line_values else np.zeros((last - first, 0), np.float32),
            )

    return Tractogram(path, [s[0] for s in scalars], [p[0] for p in properties], count, extent, batches)


def open_tractogram(path: str | Path) -> Tractogram:
    """A .trk, .tck or .trx, by its extension."""
    suffix = Path(path).suffix.lower()
    if suffix == ".trk":
        return open_trk(path)
    if suffix == ".tck":
        return open_tck(path)
    if suffix == ".trx":
        return open_trx(path)
    raise ValueError(f"{path}: not a tractogram (.trk, .tck or .trx)")
