"""SWC neurons and point tables (.csv, .tsv, .xyz) as virtual zarr-vectors
0.9.2 skeleton and point-cloud stores.

The browser's layout (ngplatform's Zarr Vectors plugin,
``src/virtual/points.ts``), chunk for chunk: one cell holds everything,
moved to the cell's corner (``coordinate_offset``); a point table too large
to make objects of is split into cells about eight across instead, each
cell's points in file order.

A skeleton store has one object per tree (segment ids 1..N in file then
root order), each one fragment whose rows are its nodes in depth-first
pre-order from the root; a row's parent is the row before it unless a
``[child, parent]`` record in ``links/0/0.0.0`` says otherwise.  Per node
``radius`` and ``compartment`` (the SWC type), as ``ingest_swc`` writes
them; per tree ``nodes``, ``cable_length`` and ``branch_points``.

A point store's rows are the table's, the other numeric columns per-point
values; up to ``MAX_POINT_OBJECTS`` points each point is also an object
(segment id = row + 1) with those values as its columns.

Ids as ``ingest_swc`` writes them: identity ``object_ids``,
``object_attributes/segment_id`` and ``fragment_attributes/object_id``.
Unlike ``ingest_csv``, a table's x, y and z are found by name, text columns
are left out, and columns without a header are ``col<i>``.
"""

from __future__ import annotations

import json
import math
import re
import struct
from pathlib import Path

import numpy as np

from .meshes import mesh_unit
from .streamlines import _column, _group, element, fragment_index, vlen_chunk

MANIFEST_ROWS = 16384
COLUMN_ROWS = 65536
MAX_POINT_OBJECTS = 250_000
U64_MAX = 18446744073709551615
POINT_SUFFIXES = (".csv", ".tsv", ".xyz")


def _number(text: str) -> float:
    """JavaScript's Number(text), for the inputs tables hold."""
    s = text.strip()
    if s == "":
        return 0.0
    if "_" in s:
        return math.nan
    try:
        return float(s)
    except ValueError:
        return math.nan


def parse_swc(text: str) -> list[dict]:
    """Nodes (id, type, position, radius, parent): short lines skipped, a parent not in the file a root."""
    nodes = []
    for raw in text.split("\n"):
        line = raw.strip()
        if line == "" or line.startswith("#"):
            continue
        p = [_number(x) for x in line.split()]
        if len(p) < 7:
            continue
        if not all(math.isfinite(v) for v in p[:7]):
            raise ValueError(f"not an SWC line: {line}")
        nodes.append({"id": p[0], "type": p[1], "position": p[2:5], "radius": p[5], "parent": p[6]})
    if not nodes:
        raise ValueError("an SWC file without nodes")
    ids = set()
    for n in nodes:
        if n["id"] in ids:
            raise ValueError(f"node {n['id']:g} appears twice")
        ids.add(n["id"])
    for n in nodes:
        if n["parent"] not in ids:
            n["parent"] = -1
    return nodes


def parse_swc_collection(text: str):
    """Several SWC files as one (a folder's ``@all.swc``): each file's text after a
    ``# @file <name>`` line, its ids numbered apart.  Files that do not parse are
    left out.  Returns (nodes, file names, each node's file index, skipped names)."""
    parts = re.split(r"^# @file (.*)$", text, flags=re.M)
    nodes, files, owner, skipped = [], [], [], []
    base = 0
    for i in range(1, len(parts), 2):
        name = parts[i].strip()
        try:
            mine = parse_swc(parts[i + 1])
        except ValueError:
            skipped.append(name)
            continue
        renumber = {n["id"]: base + k + 1 for k, n in enumerate(mine)}
        for n in mine:
            nodes.append({**n, "id": renumber[n["id"]], "parent": -1 if n["parent"] == -1 else renumber[n["parent"]]})
            owner.append(len(files))
        files.append(name)
        base += len(mine)
    if not nodes:
        raise ValueError("no SWC file in the folder could be read")
    return nodes, files, owner, skipped


def collection_text(folder: Path, suffix: str) -> str:
    """A folder's files of one kind as one text, as the browser hands a collection over."""
    names = sorted(p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() == suffix and not p.name.startswith("@all"))
    return "".join(f"# @file {n}\n{(folder / n).read_text(errors='replace')}\n" for n in names)


def parse_point_table(text: str):
    """Positions (float64, n x 3), the other numeric columns [(name, float32 values)], and the unit if the names say it."""
    lines = [l.strip() for l in text.split("\n")]
    lines = [l for l in lines if l != "" and not l.startswith("#")]
    if not lines:
        raise ValueError("a point table without rows")
    if "," in lines[0]:
        split = lambda l: l.split(",")  # noqa: E731
    elif "\t" in lines[0]:
        split = lambda l: l.split("\t")  # noqa: E731
    else:
        split = lambda l: re.split(r"\s+", l)  # noqa: E731
    cells = lambda l: [re.sub(r'^"(.*)"$', r"\1", c.strip()) for c in split(l)]  # noqa: E731
    first = cells(lines[0])
    header = any(c != "" and not math.isfinite(_number(c)) for c in first)
    names = first if header else [f"col{i}" for i in range(len(first))]
    rows = [cells(l) for l in (lines[1:] if header else lines)]
    lower = [n.lower() for n in names]

    def find(axis):
        options = {axis, f"{axis}_position", f"position_{axis}", f"{axis}_um", f"{axis}_mm", f"{axis}_nm", f"centroid_{axis}"}
        return next((i for i, n in enumerate(lower) if n in options), -1)

    axes = [find(a) for a in "xyz"]
    if -1 in axes:
        axes = [0, 1, 2]
    if len(names) < 3:
        raise ValueError("a point table needs three columns (x, y, z)")
    value = lambda r, c: _number(r[c]) if c < len(r) else math.nan  # noqa: E731
    positions = np.array([[value(r, a) for a in axes] for r in rows], np.float64).reshape(-1, 3)
    if not np.all(np.isfinite(positions)):
        raise ValueError("a point table with x, y or z values that are not numbers")
    columns = []
    for c, name in enumerate(names):
        if c in axes:
            continue
        with np.errstate(over="ignore"):
            values = np.array([value(r, c) for r in rows], np.float64).astype("<f4")
        if not np.all(np.isfinite(values)):
            continue
        columns.append((re.sub(r"[^A-Za-z0-9_]", "_", name) or f"col{c}", values))
    m = re.search(r"_(um|mm|nm)$", lower[axes[0]] if axes[0] < len(lower) else "")
    unit = {"um": "micrometer", "mm": "millimeter", "nm": "nanometer"}[m.group(1)] if m else None
    return positions, columns, unit


class CellStore:
    """Points or a skeleton in one cell, as a read-only zarr-vectors store: ``get(key)``, ``list(prefix)``, ``keys()``."""

    def __init__(self, kind, positions, object_rows, vertex_attributes, object_columns, links, unit, name, cells_across=None, text_columns=()):
        self.kind = kind
        self.name = name
        self.object_rows = object_rows
        self.vertex_attributes = dict(vertex_attributes)
        self.object_columns = dict(object_columns)
        self.text_columns = dict(text_columns)
        self.links = np.asarray(links, "<i4").reshape(-1)
        self.unit = unit
        world = np.asarray(positions, np.float64).reshape(-1, 3)
        lo = world.min(axis=0) if len(world) else np.zeros(3)
        self.offset = [int(math.floor(v)) for v in lo]
        self.vertices = (world - self.offset).astype("<f4")
        self.lo = [float(v) for v in self.vertices.min(axis=0)] if len(world) else [0.0] * 3
        self.hi = [float(v) for v in self.vertices.max(axis=0)] if len(world) else [0.0] * 3
        span = max(self.hi)
        self.chunk = 1
        while self.chunk <= span:
            self.chunk *= 2
        if cells_across is not None and object_rows is None and len(self.links) == 0 and kind == "point_cloud":
            while self.chunk > 1 and self.chunk * cells_across > 2 * span:
                self.chunk //= 2
        # Each cell's rows, in order, by key "i.j.k" (sorted as strings, as the browser does).
        at = np.floor(self.vertices.astype(np.float64) / self.chunk).astype(np.int64) if len(world) else np.zeros((0, 3), np.int64)
        keys = [f"{i}.{j}.{k}" for i, j, k in at]
        groups: dict = {}
        for v, key in enumerate(keys):
            groups.setdefault(key, []).append(v)
        if not keys:
            groups["0.0.0"] = []
        self.cells = {key: np.asarray(groups[key], np.int64) for key in sorted(groups)}
        self.shape = [int(math.floor(h / self.chunk)) + 1 for h in self.hi]
        self.count = len(object_rows) if object_rows is not None else 0
        paths = ["0/vertices", "0/vertex_fragments"] + [f"0/vertex_attributes/{a}" for a in self.vertex_attributes]
        if kind == "skeleton":
            paths.append("0/links/0/0.0.0")
        if self.count > 0:
            paths += ["0/object_index/manifests", "0/object_index/object_ids", "0/object_attributes/segment_id", "0/fragment_attributes/object_id"]
            paths += [f"0/object_attributes/{c}" for c in self.object_columns]
            paths += [f"0/object_attributes/{c}" for c in self.text_columns]
        self.directories = set()
        for p in paths:
            parts = p.split("/")
            for i in range(1, len(parts) + 1):
                self.directories.add("/".join(parts[:i]))

    def list(self, prefix: str):
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
        out = ["zarr.json"] + [f"{d}/zarr.json" for d in sorted(self.directories)]
        arrays = ["vertices", "vertex_fragments"] + [f"vertex_attributes/{a}" for a in self.vertex_attributes]
        if self.kind == "skeleton":
            arrays.append("links/0/0.0.0")
        if self.count > 0:
            arrays.append("fragment_attributes/object_id")
            out += [f"0/object_index/manifests/c/{k}" for k in range(-(-self.count // MANIFEST_ROWS))]
            for a in ["object_index/object_ids", "object_attributes/segment_id"] + [f"object_attributes/{c}" for c in [*self.object_columns, *self.text_columns]]:
                out += [f"0/{a}/c/{k}" for k in range(-(-self.count // COLUMN_ROWS))]
        out += [f"0/{a}/c/{cell.replace('.', '/')}" for a in arrays for cell in self.cells]
        return [k for k in out if self.get(k) is not None]

    def _per_cell(self, attributes):
        return {
            "zarr_format": 3, "node_type": "array", "shape": self.shape, "data_type": "variable_length_bytes",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [1, 1, 1]}},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "fill_value": "", "codecs": [{"name": "vlen-bytes", "configuration": {}}],
            "attributes": {"nonempty_chunks": list(self.cells), "chunk_grid_origin": [0, 0, 0], **attributes},
        }

    def _cell_chunk(self, array, rows):
        """A cell's chunk of a per-cell array: its rows' vertices or values, its fragments, its links."""
        if rows is None:
            return None
        whole = len(rows) == len(self.vertices)
        gather = lambda values: values if whole else values[rows]  # noqa: E731
        if array == "vertices":
            return element(gather(self.vertices).tobytes())
        if array == "vertex_fragments":
            return element(fragment_index(self.object_rows if self.object_rows is not None else [len(rows)]))
        if array == "links/0/0.0.0":
            return element(self.links.tobytes()) if self.kind == "skeleton" else None
        if array == "fragment_attributes/object_id":
            return element(np.arange(self.count, dtype="<u8").tobytes()) if self.count else None
        name = array.split("/", 1)[1] if array.startswith("vertex_attributes/") else None
        values = self.vertex_attributes.get(name) if name else None
        return None if values is None else element(np.asarray(gather(np.asarray(values, "<f4")), "<f4").tobytes())

    def get(self, key: str):
        j = lambda v: json.dumps(v, separators=(",", ":")).encode()  # noqa: E731
        if key != "zarr.json" and re.sub(r"/(zarr\.json|c/.*)$", "", key) not in self.directories:
            return None
        m = re.match(r"^0/(vertices|vertex_fragments|vertex_attributes/[^/]+|links/0/0\.0\.0|fragment_attributes/object_id)/c/(\d+)/(\d+)/(\d+)$", key)
        if m:
            return self._cell_chunk(m.group(1), self.cells.get(f"{m.group(2)}.{m.group(3)}.{m.group(4)}"))
        c = self.chunk
        skeleton = self.kind == "skeleton"
        n_links = len(self.links) // 2
        fixed = {
            "zarr.json": lambda: j(_group({
                "zarr_vectors": {
                    "zv_version": "0.9.2", "format_capabilities": [], "chunk_shape": [c, c, c], "bounds": [self.lo, self.hi],
                    "geometry_types": [self.kind],
                    "links_convention": "implicit_sequential_with_branches" if skeleton else "implicit_sequential",
                    "object_index_convention": "standard", "cross_chunk_strategy": "explicit_links", "coordinate_offset": self.offset,
                    "attribute_specs": {"vertex": {a: {"dtype": "float32"} for a in self.vertex_attributes}},
                },
                **({"zarr_vectors_tools": {"skeleton_layout": "linked_across_chunks"}} if skeleton else {}),
                "multiscales": [{
                    "version": "0.4", "name": self.name,
                    "axes": [{"name": a, "type": "space", **({"unit": self.unit} if self.unit else {})} for a in "xyz"],
                    "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1, 1, 1]}, {"type": "translation", "translation": [c / 2] * 3}]}],
                    "metadata": {"format": "zarr_vectors"},
                }],
            })),
            "0/zarr.json": lambda: j(_group({"zarr_vectors_level": {
                "level": 0, "vertex_count": len(self.vertices),
                "arrays_present": ["vertices", "links", "object_index"] if skeleton else ["vertices", "vertex_attributes"] + (["object_index"] if self.count else []),
                "object_sparsity": 1, "coarsening_method": "none", "parent_level": None, "fragments_tile": True,
            }})),
            "0/vertices/zarr.json": lambda: j(self._per_cell({"zv_array": "vertices", "dtype": "float32", "encoding": "raw"})),
            "0/vertex_fragments/zarr.json": lambda: j(self._per_cell({"zv_array": "vertex_fragments", "encoding": "fragment_index_v1"})),
            "0/links/0/zarr.json": lambda: j(_group({"zv_array": "links_family", "level_delta": 0, "link_width": 2, "directed": True, "store": "canonical", "sid_ndim": 3, "num_links": n_links, "num_physical_records": n_links})),
            "0/links/0/0.0.0/zarr.json": lambda: j(self._per_cell({"zv_array": "links", "dtype": "int32", "offsets": [[0, 0, 0]], "has_perm": False, "link_width": 2, "level_delta": 0})),
            "0/object_index/zarr.json": lambda: j(_group({"zv_array": "object_index", "num_objects": self.count, "num_present": self.count, "sid_ndim": 3, "layout": "vlen_manifests_v2", "object_ids_sorted": True})),
            "0/object_index/manifests/zarr.json": lambda: j({**_column(self.count, MANIFEST_ROWS, "variable_length_bytes", ""), "codecs": [{"name": "vlen-bytes", "configuration": {}}]}),
            "0/object_index/object_ids/zarr.json": lambda: j(_column(self.count, COLUMN_ROWS, "int64", 0)),
            "0/object_attributes/segment_id/zarr.json": lambda: j(_column(self.count, COLUMN_ROWS, "uint64", U64_MAX, {"zv_array": "object_attribute", "name": "segment_id", "dtype": "uint64", "shape": [self.count], "fill_sentinel_meaning": "absent"})),
            "0/fragment_attributes/object_id/zarr.json": lambda: j(self._per_cell({"zv_array": "fragment_attribute", "name": "object_id", "dtype": "uint64", "row_shape": []})),
        }
        if key in fixed:
            return fixed[key]()
        if re.match(r"^0/(links|object_attributes|vertex_attributes|fragment_attributes)/zarr\.json$", key):
            return j(_group())
        m = re.match(r"^0/vertex_attributes/([^/]+)/zarr\.json$", key)
        if m:
            if m.group(1) not in self.vertex_attributes:
                return None
            return j(self._per_cell({"zv_array": "attribute", "name": m.group(1), "dtype": "float32", "row_shape": []}))
        m = re.match(r"^0/object_attributes/([^/]+)/zarr\.json$", key)
        if m and m.group(1) in self.text_columns:
            return j({**_column(self.count, COLUMN_ROWS, "string", ""), "codecs": [{"name": "vlen-utf8", "configuration": {}}]})
        if m and m.group(1) in self.object_columns:
            return j(_column(self.count, COLUMN_ROWS, "float32", "NaN", {"zv_array": "object_attribute", "name": m.group(1), "dtype": "float32", "shape": [self.count], "fill_sentinel_meaning": "absent"}))
        m = re.match(r"^0/(object_index/manifests|object_index/object_ids|object_attributes/[^/]+)/c/(\d+)$", key)
        if not m:
            return None
        array = m.group(1)
        first = int(m.group(2)) * (MANIFEST_ROWS if array == "object_index/manifests" else COLUMN_ROWS)
        if first >= self.count:
            return None
        n = min(COLUMN_ROWS, self.count - first)
        if array == "object_index/manifests":
            items = [struct.pack("<IqqqBq", 1, 0, 0, 0, 0, o) for o in range(first, min(self.count, first + MANIFEST_ROWS))]
            return vlen_chunk(items, MANIFEST_ROWS)
        if array == "object_index/object_ids":
            ids = np.zeros(COLUMN_ROWS, "<i8")
            ids[:n] = np.arange(first, first + n)
            return ids.tobytes()
        if array == "object_attributes/segment_id":
            ids = np.full(COLUMN_ROWS, U64_MAX, "<u8")
            ids[:n] = np.arange(first + 1, first + n + 1)
            return ids.tobytes()
        text = self.text_columns.get(array.split("/", 1)[1])
        if text is not None:
            return vlen_chunk([t.encode() for t in text[first : first + COLUMN_ROWS]], COLUMN_ROWS)
        values = self.object_columns.get(array.split("/", 1)[1])
        if values is None:
            return None
        out = np.full(COLUMN_ROWS, np.nan, "<f4")
        out[:n] = values[first : first + n]
        return out.tobytes()


def skeleton_store(nodes: list[dict], name: str, files=None) -> CellStore:
    """An SWC file's trees: one object each, its nodes depth-first from the root."""
    index = {n["id"]: i for i, n in enumerate(nodes)}
    children = [[] for _ in nodes]
    roots = []
    for i, n in enumerate(nodes):
        p = None if n["parent"] == -1 else index.get(n["parent"])
        if p is None:
            roots.append(i)
        else:
            children[p].append(i)
    order, object_rows = [], []
    row_of = [-1] * len(nodes)
    for root in roots:
        start = len(order)
        stack = [root]
        while stack:
            i = stack.pop()
            row_of[i] = len(order)
            order.append(i)
            stack.extend(reversed(children[i]))
        object_rows.append(len(order) - start)
    if len(order) != len(nodes):
        raise ValueError("an SWC file whose nodes' parents form a cycle")
    positions = np.array([nodes[i]["position"] for i in order], np.float64)
    radius = np.array([nodes[i]["radius"] for i in order], "<f4")
    compartment = np.array([nodes[i]["type"] for i in order], "<f4")
    cable = np.zeros(len(object_rows), "<f4")
    branches = np.zeros(len(object_rows), "<f4")
    pairs = []
    obj, end = 0, object_rows[0]
    for row, i in enumerate(order):
        while row >= end:
            obj += 1
            end += object_rows[obj]
        n = nodes[i]
        if len(children[i]) >= 2:
            branches[obj] += 1
        p = None if n["parent"] == -1 else index.get(n["parent"])
        if p is None:
            continue
        q = nodes[p]["position"]
        cable[obj] = np.float32(float(cable[obj]) + math.hypot(*(a - b for a, b in zip(n["position"], q))))
        if row_of[p] != row - 1:
            pairs += [row, row_of[p]]
    return CellStore(
        "skeleton",
        positions,
        object_rows,
        [("compartment", compartment), ("radius", radius)],
        [("nodes", np.asarray(object_rows, "<f4")), ("cable_length", cable), ("branch_points", branches)],
        pairs,
        "micrometer",
        name,
        text_columns=[] if files is None else [("file", [files[0][files[1][r]] for r in roots])],
    )


def point_store(positions, columns, unit, name: str) -> CellStore:
    """A point table: each point an object (up to MAX_POINT_OBJECTS), the other columns its values."""
    positions = np.asarray(positions, np.float64).reshape(-1, 3)
    n = len(positions)
    objects = n <= MAX_POINT_OBJECTS
    extent = float(np.max(positions.max(axis=0) - positions.min(axis=0))) if n else 0.0
    return CellStore(
        "point_cloud",
        positions,
        [1] * n if objects else None,
        columns,
        columns if objects else [],
        [],
        unit or mesh_unit(extent),
        name,
        cells_across=None if objects else 8,
    )


def is_point_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in (".swc",) + POINT_SUFFIXES


def open_point_file(path: str | Path, name: str | None = None) -> CellStore:
    path = Path(path)
    if path.name.lower() == "@all.swc" and not path.exists():
        # A folder's SWC files as one store, each neuron's file a property.
        nodes, files, owner, _ = parse_swc_collection(collection_text(path.parent, ".swc"))
        return skeleton_store(nodes, name or "neurons", (files, owner))
    text = path.read_text(errors="replace")
    stem = name or path.stem
    if path.suffix.lower() == ".swc":
        return skeleton_store(parse_swc(text), stem)
    positions, columns, unit = parse_point_table(text)
    return point_store(positions, columns, unit, stem)
