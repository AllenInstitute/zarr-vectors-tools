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
    cell_counts_flat: list = field(default_factory=list)  # one entry per fragment so far, per cell
    obj_runs: list = field(default_factory=list)  # per batch: runs of each object
    run_cell: list = field(default_factory=list)  # per batch: each run's cell
    run_frag: list = field(default_factory=list)  # per batch: each run's fragment in its cell
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
            self.cell_counts_flat.append([])
        return i

    def add_batch(self, first_obj: int, points: np.ndarray, offsets: np.ndarray, values: np.ndarray) -> None:
        """Bins a batch of streamlines (objects first_obj, first_obj + 1, ...), all at once."""
        offsets = np.asarray(offsets, np.int64)
        counts = np.diff(offsets)
        n = len(counts)
        objs = first_obj + np.arange(n)
        keep_obj = (objs % self.keep_every == 0) & (counts > 0)
        point_obj = np.repeat(np.arange(n), counts)
        index = np.arange(len(points)) - np.repeat(offsets[:-1], counts)
        keep = keep_obj[point_obj]
        if self.point_step > 1:
            keep &= (index % self.point_step == 0) | (index == counts[point_obj] - 1)
        P = points[keep]
        V = values[keep] if self.value_names else None
        pobj = point_obj[keep]
        runs_per_obj = np.zeros(n, np.int64)
        if len(P) == 0:
            self.obj_runs.append(runs_per_obj)
            return
        cells = np.floor(P.astype(np.float64) / self.chunk).astype(np.int64)
        change = np.ones(len(P), bool)
        change[1:] = (pobj[1:] != pobj[:-1]) | np.any(cells[1:] != cells[:-1], axis=1)
        starts = np.flatnonzero(change)
        lens = np.diff(np.append(starts, len(P)))
        robj = pobj[starts]
        rxyz = cells[starts]
        # Cell indices, new cells numbered in the order first met.
        uniq, first, inverse = np.unique(rxyz, axis=0, return_index=True, return_inverse=True)
        inverse = inverse.reshape(-1)
        index_of = np.empty(len(uniq), np.int64)
        for u in np.argsort(first, kind="stable"):
            index_of[u] = self._cell(tuple(int(v) for v in uniq[u]))
        rcell = index_of[inverse]
        # Each run's fragment number and first row in its cell, in scan order.
        order = np.argsort(rcell, kind="stable")
        sc = rcell[order]
        heads = np.flatnonzero(np.r_[True, sc[1:] != sc[:-1]])
        sizes = np.diff(np.append(heads, len(sc)))
        rank = np.arange(len(sc)) - np.repeat(heads, sizes)
        sorted_lens = lens[order]
        before = np.cumsum(sorted_lens) - sorted_lens
        within = before - np.repeat(before[heads], sizes)
        base_frag = np.array([len(self.cell_counts_flat[c]) for c in sc[heads]], np.int64)
        base_rows = np.array([self.cell_rows[c] for c in sc[heads]], np.int64)
        frag = np.empty(len(rcell), np.int64)
        row = np.empty(len(rcell), np.int64)
        frag[order] = np.repeat(base_frag, sizes) + rank
        row[order] = np.repeat(base_rows, sizes) + within
        # The cells' vertices, owners and run lengths, a group of runs at a time.
        run_of_point = np.repeat(np.arange(len(starts)), lens)
        point_order = np.argsort(rcell[run_of_point], kind="stable")
        P_sorted = P[point_order]
        V_sorted = V[point_order] if V is not None else None
        point_heads = np.r_[0, np.cumsum(np.add.reduceat(sorted_lens, heads))]
        for g, h in enumerate(heads):
            c = int(sc[h])
            runs = order[h : h + sizes[g]]
            a, b = point_heads[g], point_heads[g + 1]
            self.cell_vertices[c].append(P_sorted[a:b])
            if V_sorted is not None:
                for k in range(len(self.value_names)):
                    self.cell_values[c][k].append(V_sorted[a:b, k])
            self.cell_owners[c].append(objs[robj[runs]])
            self.cell_counts[c].append(lens[runs])
            self.cell_counts_flat[c].extend([0] * len(runs))
            self.cell_rows[c] += int(b - a)
        # Bridges: each crossing from one run to the next of the same streamline.
        cross = np.flatnonzero(robj[1:] == robj[:-1])
        if len(cross):
            src, dst = cross, cross + 1
            offsets3 = rxyz[dst] - rxyz[src]
            names_u, name_of = np.unique(offsets3, axis=0, return_inverse=True)
            name_of = name_of.reshape(-1)
            names = [".".join(f"+{d}" if d > 0 else str(d) for d in o) for o in names_u.tolist()]
            key = name_of.astype(np.int64) * (len(self.cell_at) + 1) + rcell[src]
            keys_u, key_first, key_of = np.unique(key, return_index=True, return_inverse=True)
            key_of = key_of.reshape(-1)
            by_key = np.argsort(key_of, kind="stable")
            key_heads = np.flatnonzero(np.r_[True, key_of[by_key][1:] != key_of[by_key][:-1]])
            key_sizes = np.diff(np.append(key_heads, len(by_key)))
            pair_rows = np.stack([row[src] + lens[src] - 1, row[dst]], axis=1)
            group_of_key = {int(key_of[by_key[h]]): (h, key_sizes[k]) for k, h in enumerate(key_heads)}
            for u in np.argsort(key_first, kind="stable"):
                h, size = group_of_key[int(u)]
                members = by_key[h : h + size]
                first_pair = members[0]
                bucket = self.bridges.setdefault(names[int(name_of[first_pair])], {})
                bucket.setdefault(int(rcell[src[first_pair]]), []).extend(pair_rows[members].ravel().tolist())
        # Each object's runs, in walk order: (cell, fragment).
        runs_per_obj = np.bincount(robj, minlength=n)
        self.obj_runs.append(runs_per_obj)
        self.run_cell.append(rcell)
        self.run_frag.append(frag)
        self.vertex_count += len(P)
        self.present += int(keep_obj.sum())
        self.lo = np.minimum(self.lo, P.min(axis=0))
        self.hi = np.maximum(self.hi, P.max(axis=0))

    def finish(self) -> None:
        """Joins what was gathered batch by batch into arrays."""
        for ci in range(len(self.cell_at)):
            self.cell_vertices[ci] = np.concatenate(self.cell_vertices[ci]).astype("<f4")
            self.cell_values[ci] = [np.concatenate(v).astype("<f4") if v else np.zeros(0, "<f4") for v in self.cell_values[ci]]
            self.cell_owners[ci] = np.concatenate(self.cell_owners[ci]).astype("<u4")
            self.cell_counts[ci] = np.concatenate(self.cell_counts[ci]).astype(np.int64)
        self.cell_counts_flat = []
        runs = np.concatenate(self.obj_runs) if self.obj_runs else np.zeros(0, np.int64)
        self.obj_start = np.concatenate([[0], np.cumsum(runs)]).astype(np.int64)
        self.run_cell = np.concatenate(self.run_cell) if self.run_cell else np.zeros(0, np.int64)
        self.run_frag = np.concatenate(self.run_frag) if self.run_frag else np.zeros(0, np.int64)

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
            off = np.asarray(batch.offsets, np.int64)
            n = len(off) - 1
            for level in self.levels:
                level.add_batch(obj, batch.points, off, batch.scalars)
            # Arc lengths, summed point by point in walk order (as the browser does).
            counts = np.diff(off)
            owner = np.repeat(np.arange(n), counts)
            d = np.diff(batch.points.astype(np.float64), axis=0)
            seg = np.sqrt((d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]) + d[:, 2] * d[:, 2])
            inside = owner[1:] == owner[:-1]
            lengths.append(np.bincount(owner[1:][inside], weights=seg[inside], minlength=n))
            points_per.append(counts)
            props.append(np.asarray(batch.properties, np.float32).reshape(n, -1))
            obj += n
        self.count = obj
        lengths = np.concatenate(lengths) if lengths else np.zeros(0)
        points_per = np.concatenate(points_per) if points_per else np.zeros(0)
        props = np.concatenate(props) if props else np.zeros((0, len(tractogram.property_names)), np.float32)
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
        a, b = int(level.obj_start[o]), int(level.obj_start[o + 1])
        out = [struct.pack("<I", b - a)]
        for ci, fragment in zip(level.run_cell[a:b].tolist(), level.run_frag[a:b].tolist()):
            out.append(struct.pack("<qqqBq", *level.cell_at[ci], 0, fragment))
        return b"".join(out)
