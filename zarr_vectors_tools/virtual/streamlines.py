"""A virtual zarr-vectors streamline store: streamlines binned once into
spatial cells, every key of a format 0.9.2 store made from that when asked.

The same layout, byte for byte in its chunks, as the browser's
(ngplatform's Zarr Vectors plugin, ``src/virtual/streamlines.ts``):

- level 0: every streamline, split where it changes cell into fragments
  (one per run of vertices in a cell, in object then walk order; boundary
  vertices are not repeated), a bridge record (``links/0/<offset>``, the
  directed policy) for each crossing;
- level 1, for large tractograms: one streamline in ten, every fourth point
  (and the last), in cells twice as large, keeping level 0's ids;
- objects: a manifest per streamline, identity ids, columns (length,
  n_points, the file's per-streamline values); per-point values as vertex
  attributes.

Unsharded and uncompressed, so every cell is range-readable.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field

import numpy as np

from .tractograms import Tractogram

MAGIC = 0x5A564647
MANIFEST_ROWS = 16384
COLUMN_ROWS = 65536
NICE = [1, 2, 4, 5, 8, 10, 16, 20, 25, 32, 40, 50, 64, 100, 128, 200, 256]


def cell_size(extent: float) -> float:
    """A round cell size, mm, giving about 10 cells across ``extent``."""
    if not extent > 0:
        return 16
    want = extent / 10
    return next((n for n in NICE if n >= want), 256)


def fragment_index(counts) -> bytes:
    """fragment_index_v1 for fragments that are all ranges tiling the cell's rows."""
    f = len(counts)
    head = struct.pack("<IHHII", MAGIC, 1, 0, f, f)
    if f == 0:
        return head
    bitmap = bytearray(8 * -(-f // 64))
    for i in range(f):
        bitmap[i >> 3] |= 1 << (i & 7)
    counts = np.asarray(counts, np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    ranges = np.stack([starts, counts], axis=1).astype("<i8").tobytes()
    return head + bytes(bitmap) + ranges + struct.pack("<I", 0)


def element(payload: bytes) -> bytes:
    """One element of a per-cell vlen-bytes array."""
    return struct.pack("<II", 1, len(payload)) + payload


def vlen_chunk(items: list[bytes], rows: int) -> bytes:
    parts = [struct.pack("<I", rows)]
    for r in range(rows):
        item = items[r] if r < len(items) else b""
        parts.append(struct.pack("<I", len(item)))
        parts.append(item)
    return b"".join(parts)


@dataclass
class Level:
    path: str
    chunk: float
    value_names: list[str]
    keep_every: int
    point_step: int
    # Filled by add():
    cells: dict = field(default_factory=dict)  # (cx, cy, cz) -> index, in order first met
    cell_at: list = field(default_factory=list)
    cell_vertices: list = field(default_factory=list)
    cell_owners: list = field(default_factory=list)
    cell_counts: list = field(default_factory=list)
    cell_values: list = field(default_factory=list)
    cell_rows: list = field(default_factory=list)  # rows so far, per cell
    blocks: list = field(default_factory=list)  # per object: list of (cell, fragment)
    bridges: dict = field(default_factory=dict)  # offset name -> {cell: [a, b, a, b, ...]}
    lo: np.ndarray = field(default_factory=lambda: np.full(3, np.inf))
    hi: np.ndarray = field(default_factory=lambda: np.full(3, -np.inf))
    vertex_count: int = 0
    present: int = 0

    def _cell(self, key) -> int:
        i = self.cells.get(key)
        if i is None:
            i = len(self.cell_at)
            self.cells[key] = i
            self.cell_at.append(key)
            self.cell_vertices.append([])
            self.cell_owners.append([])
            self.cell_counts.append([])
            self.cell_values.append([[] for _ in self.value_names])
            self.cell_rows.append(0)
        return i

    def add(self, obj: int, points: np.ndarray, values: np.ndarray) -> None:
        n = len(points)
        if obj % self.keep_every != 0 or n == 0:
            self.blocks.append([])
            return
        if self.point_step > 1:
            keep = (np.arange(n) % self.point_step == 0) | (np.arange(n) == n - 1)
            points, values = points[keep], values[keep]
        cells = np.floor(points.astype(np.float64) / self.chunk).astype(np.int64)
        change = np.ones(len(points), bool)
        change[1:] = np.any(cells[1:] != cells[:-1], axis=1)
        starts = np.flatnonzero(change)
        ends = np.concatenate([starts[1:], [len(points)]])
        blocks = []
        prev = None
        for s, e in zip(starts, ends):
            key = tuple(int(v) for v in cells[s])
            ci = self._cell(key)
            fragment = len(self.cell_owners[ci])
            row = self.cell_rows[ci]
            self.cell_rows[ci] += int(e - s)
            self.cell_owners[ci].append(obj)
            self.cell_counts[ci].append(int(e - s))
            self.cell_vertices[ci].append(points[s:e])
            for k in range(len(self.value_names)):
                self.cell_values[ci][k].append(values[s:e, k])
            blocks.append((ci, fragment))
            if prev is not None:
                prev_cell, prev_last = prev
                offset = [key[k] - self.cell_at[prev_cell][k] for k in range(3)]
                name = ".".join(f"+{d}" if d > 0 else str(d) for d in offset)
                self.bridges.setdefault(name, {}).setdefault(prev_cell, []).extend([prev_last, row])
            prev = (ci, row + int(e - s) - 1)
        self.blocks.append(blocks)
        self.vertex_count += len(points)
        self.present += 1
        self.lo = np.minimum(self.lo, points.min(axis=0))
        self.hi = np.maximum(self.hi, points.max(axis=0))

    def finish(self) -> None:
        """Joins each cell's runs into arrays (they are kept as lists while reading)."""
        self.cell_row_offsets = []
        for ci in range(len(self.cell_at)):
            self.cell_vertices[ci] = np.concatenate(self.cell_vertices[ci]).astype("<f4")
            self.cell_values[ci] = [np.concatenate(v).astype("<f4") if v else np.zeros(0, "<f4") for v in self.cell_values[ci]]
            self.cell_owners[ci] = np.asarray(self.cell_owners[ci], "<u4")

    def grid(self, lo, hi):
        origin = [int(np.floor(v / self.chunk)) for v in lo]
        shape = [int(np.floor(v / self.chunk)) - origin[k] + 1 for k, v in enumerate(hi)]
        return origin, shape


def _group(attributes=None):
    return {"zarr_format": 3, "node_type": "group", "attributes": attributes or {}}


def _per_cell(shape, attributes):
    return {
        "zarr_format": 3,
        "node_type": "array",
        "shape": shape,
        "data_type": "variable_length_bytes",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [1] * len(shape)}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": "",
        "codecs": [{"name": "vlen-bytes", "configuration": {}}],
        "attributes": attributes,
    }


def _column(n, rows, data_type, fill, attributes=None):
    return {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [n],
        "data_type": data_type,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [rows]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": fill,
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "attributes": attributes or {},
    }


def _json(value) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode()


class StreamlineStore:
    """A tractogram as a read-only zarr-vectors store: ``get(key)`` and ``list(prefix)``."""

    def __init__(self, tractogram: Tractogram, name: str | None = None):
        self.name = name or tractogram.path.stem
        chunk = cell_size(tractogram.extent)
        self.levels = [Level("0", chunk, list(tractogram.scalar_names), 1, 1)]
        if tractogram.expected_count > 20_000:
            self.levels.append(Level("1", 2 * chunk, [], 10, 4))
        lengths, points_per, props = [], [], []
        obj = 0
        for batch in tractogram.batches():
            off = batch.offsets
            for s in range(len(off) - 1):
                a, b = int(off[s]), int(off[s + 1])
                pts = batch.points[a:b]
                vals = batch.scalars[a:b]
                for level in self.levels:
                    level.add(obj, pts, vals if level.value_names else np.zeros((b - a, 0), np.float32))
                d = np.diff(pts.astype(np.float64), axis=0)
                lengths.append(float(np.sum(np.sqrt(np.sum(d * d, axis=1)))) if len(d) else 0.0)
                points_per.append(b - a)
                props.append(batch.properties[s])
                obj += 1
        self.count = obj
        for level in self.levels:
            level.finish()
        base = self.levels[0]
        self.lo = [float(v) if np.isfinite(v) else 0.0 for v in base.lo]
        self.hi = [float(v) if np.isfinite(v) else 0.0 for v in base.hi]
        props = np.asarray(props, np.float32).reshape(self.count, len(tractogram.property_names))
        self.columns = {"length": np.asarray(lengths, "<f4"), "n_points": np.asarray(points_per, "<f4")}
        for k, name in enumerate(tractogram.property_names):
            self.columns[name] = props[:, k].astype("<f4")
        self.directories = set()
        for level in self.levels:
            p = level.path
            for a in ["vertices", "vertex_fragments", "fragment_attributes/segment_id", "object_index/manifests", "object_index/object_ids", "links/0"]:
                self._add_dir(f"{p}/{a}")
            for seg in level.bridges:
                self._add_dir(f"{p}/links/0/{seg}")
            for v in level.value_names:
                self._add_dir(f"{p}/vertex_attributes/{v}")
        for c in self.columns:
            self._add_dir(f"0/object_attributes/{c}")

    def _add_dir(self, path: str) -> None:
        parts = path.split("/")
        for i in range(1, len(parts) + 1):
            self.directories.add("/".join(parts[:i]))

    # ------------------------------------------------------------- listing

    def list(self, prefix: str) -> list[str] | None:
        """Names in a directory ("" or ending in "/"), sub-directories ending in "/"."""
        d = prefix.rstrip("/")
        if d and d not in self.directories:
            return None
        base = f"{d}/" if d else ""
        names = {"zarr.json"}
        for path in self.directories:
            if path != d and path.startswith(base):
                names.add(path[len(base):].split("/")[0] + "/")
        return sorted(names)

    def keys(self):
        """Every key of the store."""
        out = ["zarr.json"]
        for level in self.levels:
            origin, _ = level.grid(self.lo, self.hi)
            for d in sorted(x for x in self.directories if x.startswith(f"{level.path}/") or x == level.path):
                out.append(f"{d}/zarr.json")
            cells = ["/".join(str(level.cell_at[i][k] - origin[k]) for k in range(3)) for i in range(len(level.cell_at))]
            for array in ["vertices", "vertex_fragments", "fragment_attributes/segment_id"] + [f"vertex_attributes/{v}" for v in level.value_names]:
                out += [f"{level.path}/{array}/c/{c}" for c in cells]
            for seg, bucket in level.bridges.items():
                out += [f"{level.path}/links/0/{seg}/c/" + "/".join(str(level.cell_at[i][k] - origin[k]) for k in range(3)) for i in bucket]
            out += [f"{level.path}/object_index/manifests/c/{k}" for k in range(-(-self.count // MANIFEST_ROWS))]
            out += [f"{level.path}/object_index/object_ids/c/{k}" for k in range(-(-self.count // COLUMN_ROWS))]
        for c in self.columns:
            out += [f"0/object_attributes/{c}/c/{k}" for k in range(-(-self.count // COLUMN_ROWS))]
        return out

    # ---------------------------------------------------------------- keys

    def get(self, key: str) -> bytes | None:
        if key == "zarr.json":
            return _json(self._root())
        level_path, _, rest = key.partition("/")
        level = next((l for l in self.levels if l.path == level_path), None)
        if level is None:
            return None
        return self._level_key(level, rest)

    def _root(self):
        values = self.levels[0].value_names
        zv = {
            "zv_version": "0.9.2",
            "format_capabilities": ["preserved_object_ids"],
            "chunk_shape": [self.levels[0].chunk] * 3,
            "bounds": [self.lo, self.hi],
            "geometry_types": ["streamline"],
            "links_convention": "implicit_sequential",
            "object_index_convention": "standard",
            "cross_chunk_strategy": "explicit_links",
        }
        if values:
            zv["attribute_specs"] = {"vertex": {v: {"dtype": "float32"} for v in values}}
        return _group(
            {
                "zarr_vectors": zv,
                "multiscales": [
                    {
                        "version": "0.4",
                        "name": self.name,
                        "axes": [{"name": a, "type": "space", "unit": "millimeter"} for a in "xyz"],
                        "datasets": [
                            {
                                "path": l.path,
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [1, 1, 1]},
                                    {"type": "translation", "translation": [l.chunk / 2] * 3},
                                ],
                            }
                            for l in self.levels
                        ],
                        "metadata": {"format": "zarr_vectors"},
                    }
                ],
            }
        )

    def _level_key(self, level: Level, key: str) -> bytes | None:
        origin, shape = level.grid(self.lo, self.hi)
        nonempty = [".".join(str(v) for v in at) for at in level.cell_at]

        def per_cell(attributes):
            return _json(_per_cell(shape, {"nonempty_chunks": nonempty, "chunk_grid_origin": origin, **attributes}))

        if key == "zarr.json":
            attrs = {"level": int(level.path), "vertex_count": level.vertex_count, "arrays_present": ["vertices", "object_index"]}
            if level.path == "0":
                attrs.update(object_sparsity=1, coarsening_method="none", parent_level=None, fragments_tile=True)
            else:
                attrs.update(
                    chunk_shape=[level.chunk] * 3,
                    object_sparsity=level.present / max(1, self.count),
                    coarsening_method="subsample",
                    parent_level=0,
                    preserves_object_ids=True,
                    inherited_num_objects=self.count,
                )
            return _json(_group({"zarr_vectors_level": attrs}))
        fixed = {
            "vertices/zarr.json": lambda: per_cell({"zv_array": "vertices", "dtype": "float32", "encoding": "raw"}),
            "vertex_fragments/zarr.json": lambda: per_cell({"zv_array": "vertex_fragments", "encoding": "fragment_index_v1"}),
            "fragment_attributes/segment_id/zarr.json": lambda: per_cell({"zv_array": "fragment_attribute", "name": "segment_id", "dtype": "uint32", "row_shape": []}),
            "links/0/zarr.json": lambda: _json(
                _group(
                    {
                        "zv_array": "links_family",
                        "level_delta": 0,
                        "link_width": 2,
                        "directed": True,
                        "store": "canonical",
                        "sid_ndim": 3,
                        "num_links": sum(len(r) // 2 for b in level.bridges.values() for r in b.values()),
                        "num_physical_records": sum(len(r) // 2 for b in level.bridges.values() for r in b.values()),
                    }
                )
            ),
            "object_index/zarr.json": lambda: _json(
                _group({"zv_array": "object_index", "num_objects": self.count, "num_present": level.present, "sid_ndim": 3, "layout": "vlen_manifests_v2", "object_ids_sorted": True})
            ),
            "object_index/manifests/zarr.json": lambda: _json(
                {**_column(self.count, MANIFEST_ROWS, "variable_length_bytes", ""), "codecs": [{"name": "vlen-bytes", "configuration": {}}]}
            ),
            "object_index/object_ids/zarr.json": lambda: _json(_column(self.count, COLUMN_ROWS, "int64", 0)),
        }
        if key in fixed:
            return fixed[key]()
        if key.endswith("zarr.json"):
            d = key[: -len("zarr.json")].rstrip("/")
            if f"{level.path}/{d}" not in self.directories:
                return None
            if d.startswith("links/0/"):
                seg = d[len("links/0/"):]
                bucket = level.bridges[seg]
                return _json(
                    _per_cell(
                        shape,
                        {
                            "nonempty_chunks": [".".join(str(v) for v in level.cell_at[i]) for i in bucket],
                            "chunk_grid_origin": origin,
                            "zv_array": "links",
                            "dtype": "int32",
                            "offsets": [[int(v) for v in seg.split(".")]],
                            "has_perm": False,
                            "link_width": 2,
                            "level_delta": 0,
                        },
                    )
                )
            if d.startswith("vertex_attributes/"):
                return per_cell({"zv_array": "attribute", "name": d.split("/", 1)[1], "dtype": "float32", "row_shape": []})
            if d.startswith("object_attributes/") and level.path == "0":
                name = d.split("/", 1)[1]
                return _json(
                    _column(self.count, COLUMN_ROWS, "float32", "NaN", {"zv_array": "object_attribute", "name": name, "dtype": "float32", "shape": [self.count], "fill_sentinel_meaning": "absent"})
                )
            return _json(_group())
        return self._chunk(level, key, origin)

    def _chunk(self, level: Level, key: str, origin) -> bytes | None:
        array, sep, coords = key.partition("/c/")
        if not sep:
            return None
        parts = coords.split("/")
        if len(parts) == 3:
            try:
                at = tuple(int(p) + origin[k] for k, p in enumerate(parts))
            except ValueError:
                return None
            ci = level.cells.get(at)
            if ci is None:
                return None
            if array == "vertices":
                return element(level.cell_vertices[ci].tobytes())
            if array == "vertex_fragments":
                return element(fragment_index(level.cell_counts[ci]))
            if array == "fragment_attributes/segment_id":
                return element(level.cell_owners[ci].tobytes())
            if array.startswith("vertex_attributes/"):
                name = array.split("/", 1)[1]
                if name not in level.value_names:
                    return None
                return element(level.cell_values[ci][level.value_names.index(name)].tobytes())
            if array.startswith("links/0/"):
                rows = level.bridges.get(array[len("links/0/"):], {}).get(ci)
                if rows is None:
                    return None
                return element(np.asarray([1, 0], "<i8").tobytes() + np.asarray(rows, "<i4").tobytes())
            return None
        if len(parts) != 1 or not parts[0].isdigit():
            return None
        chunk = int(parts[0])
        if array == "object_index/manifests":
            first = chunk * MANIFEST_ROWS
            if first >= self.count:
                return None
            return vlen_chunk([self._manifest(level, o) for o in range(first, min(self.count, first + MANIFEST_ROWS))], MANIFEST_ROWS)
        first = chunk * COLUMN_ROWS
        if first >= self.count:
            return None
        if array == "object_index/object_ids":
            ids = np.zeros(COLUMN_ROWS, "<i8")
            n = min(COLUMN_ROWS, self.count - first)
            ids[:n] = np.arange(first, first + n)
            return ids.tobytes()
        if array.startswith("object_attributes/") and level.path == "0":
            values = self.columns.get(array.split("/", 1)[1])
            if values is None:
                return None
            out = np.full(COLUMN_ROWS, np.nan, "<f4")
            part = values[first : first + COLUMN_ROWS]
            out[: len(part)] = part
            return out.tobytes()
        return None

    def _manifest(self, level: Level, o: int) -> bytes:
        blocks = level.blocks[o]
        out = [struct.pack("<I", len(blocks))]
        for ci, fragment in blocks:
            out.append(struct.pack("<qqqBq", *level.cell_at[ci], 0, fragment))
        return b"".join(out)
