"""Mesh files (.obj, .stl, .ply, GIFTI .gii, FreeSurfer surfaces) as virtual
zarr-vectors 0.9.3 mesh stores.

The browser's layout (ngplatform's Zarr Vectors plugin,
``src/virtual/meshes.ts``), chunk for chunk: one cell holds the whole mesh;
faces are int32 rows of ``links/0/0.0.0_0.0.0`` in the file's winding; each
object (an .obj ``o``/``g`` group, or the whole file) is one fragment of the
vertices and one group of the faces (``fragment_link_groups``); ids start at
1; the mesh is moved to the cell's corner, the move kept as
``coordinate_offset``.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import numpy as np

from .streamlines import _column, _group, element, fragment_index, vlen_chunk

MANIFEST_ROWS = 16384
COLUMN_ROWS = 65536


def _single(positions, triangles, name):
    return positions, triangles, np.zeros(len(triangles) // 3, np.uint32), [name]


def _merge_corners(corners: np.ndarray):
    """Merges corners at exactly the same place, in the order they first appear."""
    bits = corners.astype("<f4").view("<u4").reshape(-1, 3)
    index: dict = {}
    triangles = np.empty(len(bits), np.uint32)
    kept = []
    for i, key in enumerate(map(tuple, bits)):
        v = index.get(key)
        if v is None:
            v = len(kept)
            index[key] = v
            kept.append(i)
        triangles[i] = v
    return corners.reshape(-1, 3)[kept].astype(np.float32).ravel(), triangles


def parse_stl(data: bytes, name: str):
    count = struct.unpack("<I", data[80:84])[0] if len(data) >= 84 else -1
    if count >= 0 and abs(84 + 50 * count - len(data)) <= 2:
        facets = np.frombuffer(data[84 : 84 + 50 * count].ljust(50 * count, b"\0"), np.dtype([("n", "<f4", 3), ("v", "<f4", 9), ("a", "<u2")]))
        corners = facets["v"].reshape(-1).astype(np.float32)
    else:
        text = data.decode(errors="replace")
        if not re.match(r"^\s*solid", text):
            raise ValueError("not an STL file")
        corners = np.array([float(x) for m in re.finditer(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", text) for x in m.groups()], np.float32)
    positions, triangles = _merge_corners(corners)
    return _single(positions, triangles, name)


def parse_obj(text: str, name: str):
    positions, triangles, owner, objects = [], [], [], []
    index: dict = {}
    current = -1

    def use(label):
        nonlocal current
        if label not in index:
            index[label] = len(objects)
            objects.append(label)
        current = index[label]

    for raw in text.split("\n"):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "v":
            positions += [float(parts[1]), float(parts[2]), float(parts[3])]
        elif parts[0] in ("o", "g"):
            if len(parts) > 1:
                use(" ".join(parts[1:]))
        elif parts[0] == "f":
            if current == -1:
                use(name)
            count = len(positions) // 3
            corners = []
            for p in parts[1:]:
                i = int(p.split("/")[0])
                corners.append(count + i if i < 0 else i - 1)
            if any(not 0 <= i < count for i in corners):
                raise ValueError(f"a face with a vertex out of range: {line}")
            for k in range(1, len(corners) - 1):
                triangles += [corners[0], corners[k], corners[k + 1]]
                owner.append(current)
    if not objects:
        objects.append(name)
    return np.array(positions, np.float32), np.array(triangles, np.uint32), np.array(owner, np.uint32), objects


_PLY = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
        "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4", "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}


def parse_ply(data: bytes, name: str):
    head = data[:65536].decode("latin1")
    if not head.startswith("ply"):
        raise ValueError("not a PLY file")
    end = head.index("end_header")
    body = head.index("\n", end) + 1
    fmt, elements = "ascii", []
    for line in head[:end].split("\n"):
        p = line.split()
        if not p:
            continue
        if p[0] == "format":
            fmt = p[1]
        elif p[0] == "element":
            elements.append({"name": p[1], "count": int(p[2]), "props": []})
        elif p[0] == "property" and elements:
            elements[-1]["props"].append((p[4], "list", p[2], p[3]) if p[1] == "list" else (p[2], p[1]))
    vertex = next(e for e in elements if e["name"] == "vertex")
    axes = [next(i for i, pr in enumerate(vertex["props"]) if pr[0] == a) for a in "xyz"]
    positions = np.zeros((vertex["count"], 3), np.float32)
    triangles: list = []
    if fmt == "ascii":
        tokens = data[body:].decode("latin1").split()
        t = 0
        for e in elements:
            for i in range(e["count"]):
                row = []
                for pr in e["props"]:
                    if pr[1] == "list":
                        n = int(float(tokens[t]))
                        row.append([float(x) for x in tokens[t + 1 : t + 1 + n]])
                        t += 1 + n
                    else:
                        row.append([float(tokens[t])])
                        t += 1
                if e is vertex:
                    positions[i] = [row[a][0] for a in axes]
                elif e["name"] == "face":
                    c = next(r for r, pr in zip(row, e["props"]) if pr[1] == "list")
                    triangles += [int(x) for k in range(1, len(c) - 1) for x in (c[0], c[k], c[k + 1])]
    else:
        e_ = "<" if fmt == "binary_little_endian" else ">"
        at = body
        for e in elements:
            for i in range(e["count"]):
                corners = None
                for k, pr in enumerate(e["props"]):
                    if pr[1] == "list":
                        size = np.dtype(_PLY[pr[2]]).itemsize
                        n = int(np.frombuffer(data, e_ + _PLY[pr[2]], 1, at)[0])
                        at += size
                        items = np.frombuffer(data, e_ + _PLY[pr[3]], n, at).astype(np.int64)
                        at += n * np.dtype(_PLY[pr[3]]).itemsize
                        if corners is None:
                            corners = items
                    else:
                        v = np.frombuffer(data, e_ + _PLY[pr[1]], 1, at)[0]
                        at += np.dtype(_PLY[pr[1]]).itemsize
                        if e is vertex and k in axes:
                            positions[i, axes.index(k)] = v
                if e["name"] == "face" and corners is not None:
                    triangles += [int(x) for k in range(1, len(corners) - 1) for x in (corners[0], corners[k], corners[k + 1])]
    return _single(positions.ravel(), np.array(triangles, np.uint32), name)


#: Surfaces in no anatomical space: left where the file puts them.
NOT_ANATOMICAL = re.compile(r"inflated|sphere|flat", re.I)
#: FreeSurfer surface file names (lh.pial, rh.white, ...).
FREESURFER_SURFACE = re.compile(r"^(lh|rh)\.(pial|white|orig|smoothwm|inflated|very_inflated|sphere|sphere\.reg|midthickness|graymid|pial\.T1|orig\.nofix|inflated_pre|qsphere\.nofix)$", re.I)


def parse_freesurfer(data: bytes, name: str):
    """A FreeSurfer triangle surface; an anatomical one moved by a valid footer's c_ras."""
    if data[:3] != b"\xff\xff\xfe":
        raise ValueError("not a FreeSurfer triangle surface")
    at = data.index(b"\n", 3)
    if data[at + 1 : at + 2] != b"\n":
        raise ValueError("a FreeSurfer surface without its created-by line")
    at += 2
    nv, nf = struct.unpack(">ii", data[at : at + 8])
    at += 8
    positions = np.frombuffer(data, ">f4", 3 * nv, at).astype(np.float32)
    at += 12 * nv
    triangles = np.frombuffer(data, ">i4", 3 * nf, at).astype(np.uint32)
    at += 12 * nf
    footer = data[at:].decode("latin1")
    cras = re.search(r"cras\s*=\s*(\S+)\s+(\S+)\s+(\S+)", footer)
    if re.search(r"valid\s*=\s*1", footer) and cras and not NOT_ANATOMICAL.search(name):
        c = np.array([float(v) for v in cras.groups()])
        positions = (positions.reshape(-1, 3).astype(np.float64) + c).astype(np.float32).ravel()
    return _single(positions, triangles, name)


def _cdata(text: str) -> str:
    m = re.match(r"^\s*<!\[CDATA\[([\s\S]*?)\]\]>\s*$", text)
    return (m.group(1) if m else text).strip()


def parse_gifti(text: str, name: str):
    """A GIFTI surface; an anatomical one moved by its matrix when that leads to a named space."""
    import base64
    import zlib

    if not re.search(r"<GIFTI[\s>]", text):
        raise ValueError("not a GIFTI file")
    positions = triangles = matrix = None
    geometric = ""
    for m in re.finditer(r"<DataArray\b([^>]*)>([\s\S]*?)</DataArray>", text):
        attrs = dict(re.findall(r'(\w+)\s*=\s*"([^"]*)"', m.group(1)))
        intent = attrs.get("Intent", "")
        if intent not in ("NIFTI_INTENT_POINTSET", "NIFTI_INTENT_TRIANGLE"):
            continue
        dims = [int(attrs.get("Dim0", 0)), int(attrs.get("Dim1", 1) or 1)]
        encoding = attrs.get("Encoding", "")
        body = _cdata((re.search(r"<Data>([\s\S]*?)</Data>", m.group(2)) or [None, ""])[1])
        is_float = attrs.get("DataType") == "NIFTI_TYPE_FLOAT32"
        if encoding == "ASCII":
            values = np.array([float(v) for v in body.split()], np.float32 if is_float else np.int32)
        elif encoding in ("Base64Binary", "GZipBase64Binary"):
            raw = base64.b64decode("".join(body.split()))
            if encoding == "GZipBase64Binary":
                raw = zlib.decompress(raw)
            endian = ">" if attrs.get("Endian") == "BigEndian" else "<"
            values = np.frombuffer(raw[: len(raw) // 4 * 4], endian + ("f4" if is_float else "i4")).astype(np.float32 if is_float else np.int32)
        else:
            raise ValueError(f"a GIFTI array stored as {encoding or 'nothing'} (ASCII and base64 only)")
        if attrs.get("ArrayIndexingOrder") == "ColumnMajorOrder" and dims[1] > 1:
            values = values.reshape(dims[1], dims[0]).T.ravel()
        if intent == "NIFTI_INTENT_POINTSET":
            positions = values.astype(np.float32)
            g = re.search(r"<Name>\s*(?:<!\[CDATA\[)?GeometricType(?:\]\]>)?\s*</Name>\s*<Value>\s*(?:<!\[CDATA\[)?([^<\]]*)", m.group(2))
            geometric = g.group(1) if g else ""
            x = re.search(r"<CoordinateSystemTransformMatrix>([\s\S]*?)</CoordinateSystemTransformMatrix>", m.group(2))
            xform = x.group(1) if x else ""
            target = _cdata((re.search(r"<TransformedSpace>([\s\S]*?)</TransformedSpace>", xform) or [None, ""])[1])
            data = [float(v) for v in _cdata((re.search(r"<MatrixData>([\s\S]*?)</MatrixData>", xform) or [None, ""])[1]).split()]
            if len(data) == 16 and target not in ("", "NIFTI_XFORM_UNKNOWN"):
                matrix = data
        else:
            triangles = values.astype(np.uint32)
    if positions is None:
        raise ValueError("a GIFTI file without a surface (no NIFTI_INTENT_POINTSET array)")
    identity = matrix is None or all(abs(v - (1 if i % 5 == 0 else 0)) < 1e-6 for i, v in enumerate(matrix))
    if not identity and not NOT_ANATOMICAL.search(geometric or name):
        p = positions.reshape(-1, 3).astype(np.float64)
        out = np.empty_like(p)
        for r in range(3):
            out[:, r] = ((matrix[4 * r] * p[:, 0] + matrix[4 * r + 1] * p[:, 1]) + matrix[4 * r + 2] * p[:, 2]) + matrix[4 * r + 3]
        positions = out.astype(np.float32).ravel()
    return _single(positions, triangles if triangles is not None else np.zeros(0, np.uint32), name)


def mesh_name(path: Path) -> str:
    """The name a mesh file is shown by: without .surf.gii, .gii, .obj, .stl or .ply."""
    return re.sub(r"\.(surf\.gii|gii|obj|stl|ply)$", "", path.name, flags=re.I)


def is_mesh_file(path: str | Path) -> bool:
    name = Path(path).name
    return bool(re.search(r"\.(obj|stl|ply|gii)$", name, re.I) or FREESURFER_SURFACE.match(name))


def parse_mesh(path: str | Path):
    path = Path(path)
    name = mesh_name(path)
    data = path.read_bytes()
    suffix = path.suffix.lower()
    if suffix == ".obj":
        return parse_obj(data.decode(), name)
    if suffix == ".stl":
        return parse_stl(data, name)
    if suffix == ".ply":
        return parse_ply(data, name)
    if suffix == ".gii":
        return parse_gifti(data.decode(), name)
    if FREESURFER_SURFACE.match(path.name):
        return parse_freesurfer(data, path.name)
    raise ValueError(f"{path}: not a mesh file (.obj, .stl, .ply, .gii or a FreeSurfer surface)")


def mesh_unit(extent: float) -> str:
    return "nanometer" if extent > 50_000 else "millimeter"


class MeshStore:
    """A mesh file as a read-only zarr-vectors mesh store: ``get(key)``, ``list(prefix)``, ``keys()``."""

    def __init__(self, path: str | Path, name: str | None = None):
        positions, triangles, owner, objects = parse_mesh(path)
        self.name = name or mesh_name(Path(path))
        n_obj = len(objects)
        faces_of = [np.flatnonzero(owner == o) for o in range(n_obj)]
        used = []
        for o in range(n_obj):
            idx = np.unique(triangles.reshape(-1, 3)[faces_of[o]].ravel()) if len(faces_of[o]) else np.zeros(0, np.int64)
            used.append(idx)
        if len(triangles) == 0:
            used[0] = np.arange(len(positions) // 3)
        self.vertex_counts = [len(u) for u in used]
        self.face_counts = [len(f) for f in faces_of]
        pos = positions.reshape(-1, 3)
        world = np.concatenate([pos[u].astype(np.float64) for u in used]) if used else np.zeros((0, 3))
        lo = world.min(axis=0) if len(world) else np.zeros(3)
        self.offset = [int(np.floor(v)) for v in lo]
        self.vertices = (world - self.offset).astype("<f4")
        self.lo = [float(v) for v in self.vertices.min(axis=0)] if len(world) else [0.0] * 3
        self.hi = [float(v) for v in self.vertices.max(axis=0)] if len(world) else [0.0] * 3
        self.chunk = 1
        while self.chunk <= max(self.hi):
            self.chunk *= 2
        rows, areas, base = [], [], 0
        tri = triangles.reshape(-1, 3)
        for o in range(n_obj):
            local = np.searchsorted(used[o], tri[faces_of[o]]) + base
            rows.append(local)
            p = world[local - 0] if len(local) else np.zeros((0, 3, 3))
            if len(local):
                p = world[local]
                areas.append(float(np.sum(0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1))))
            else:
                areas.append(0.0)
            base += self.vertex_counts[o]
        self.faces = (np.concatenate(rows) if rows else np.zeros((0, 3))).astype("<i4")
        self.count = n_obj
        self.columns = {
            "n_vertices": np.asarray(self.vertex_counts, "<f4"),
            "n_faces": np.asarray(self.face_counts, "<f4"),
            "area": np.asarray(areas, "<f4"),
        }
        self.directories = set()
        for p in ["0/vertices", "0/vertex_fragments", "0/link_fragments", "0/links/0/0.0.0_0.0.0", "0/object_index/manifests", "0/object_index/object_ids"] + [f"0/object_attributes/{c}" for c in self.columns]:
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
        out += [f"0/{a}/c/0/0/0" for a in ("vertices", "vertex_fragments", "link_fragments", "links/0/0.0.0_0.0.0")]
        out += [f"0/object_index/manifests/c/{k}" for k in range(-(-self.count // MANIFEST_ROWS))]
        out += [f"0/object_index/object_ids/c/{k}" for k in range(-(-self.count // COLUMN_ROWS))]
        for c in self.columns:
            out += [f"0/object_attributes/{c}/c/{k}" for k in range(-(-self.count // COLUMN_ROWS))]
        return out

    def _per_cell(self, attributes):
        return {
            "zarr_format": 3, "node_type": "array", "shape": [1, 1, 1], "data_type": "variable_length_bytes",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [1, 1, 1]}},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "fill_value": "", "codecs": [{"name": "vlen-bytes", "configuration": {}}],
            "attributes": {"nonempty_chunks": ["0.0.0"], "chunk_grid_origin": [0, 0, 0], **attributes},
        }

    def get(self, key: str):
        j = lambda v: json.dumps(v, separators=(",", ":")).encode()  # noqa: E731
        c = self.chunk
        extent = max(h - l for h, l in zip(self.hi, self.lo))
        n_faces = len(self.faces)
        fixed = {
            "zarr.json": lambda: j(_group({
                "zarr_vectors": {
                    "zv_version": "0.9.3", "format_capabilities": ["fragment_link_groups"], "chunk_shape": [c, c, c],
                    "bounds": [self.lo, self.hi], "coordinate_offset": self.offset, "geometry_types": ["mesh"],
                    "links_convention": "explicit", "object_index_convention": "standard", "cross_chunk_strategy": "explicit_links",
                },
                "multiscales": [{
                    "version": "0.4", "name": self.name,
                    "axes": [{"name": a, "type": "space", "unit": mesh_unit(extent)} for a in "xyz"],
                    "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1, 1, 1]}, {"type": "translation", "translation": [c / 2] * 3}]}],
                    "metadata": {"format": "zarr_vectors"},
                }],
            })),
            "0/zarr.json": lambda: j(_group({"zarr_vectors_level": {
                "level": 0, "vertex_count": len(self.vertices), "arrays_present": ["vertices", "links", "object_index"],
                "object_sparsity": 1, "coarsening_method": "none", "parent_level": None, "fragments_tile": True, "fragment_link_groups": True,
            }})),
            "0/vertices/zarr.json": lambda: j(self._per_cell({"zv_array": "vertices", "dtype": "float32", "encoding": "raw"})),
            "0/vertex_fragments/zarr.json": lambda: j(self._per_cell({"zv_array": "vertex_fragments", "encoding": "fragment_index_v1"})),
            "0/link_fragments/zarr.json": lambda: j(self._per_cell({"zv_array": "link_fragments", "encoding": "fragment_index_v1"})),
            "0/links/0/zarr.json": lambda: j(_group({"zv_array": "links_family", "level_delta": 0, "link_width": 3, "directed": False, "store": "canonical", "sid_ndim": 3, "num_links": n_faces, "num_physical_records": n_faces})),
            "0/links/0/0.0.0_0.0.0/zarr.json": lambda: j(self._per_cell({"zv_array": "links", "dtype": "int32", "offsets": [[0, 0, 0], [0, 0, 0]], "has_perm": False, "link_width": 3, "level_delta": 0})),
            "0/object_index/zarr.json": lambda: j(_group({"zv_array": "object_index", "num_objects": self.count, "num_present": self.count, "sid_ndim": 3, "layout": "vlen_manifests_v2", "object_ids_sorted": True})),
            "0/object_index/manifests/zarr.json": lambda: j({**_column(self.count, MANIFEST_ROWS, "variable_length_bytes", ""), "codecs": [{"name": "vlen-bytes", "configuration": {}}]}),
            "0/object_index/object_ids/zarr.json": lambda: j(_column(self.count, COLUMN_ROWS, "int64", 0)),
            "0/vertices/c/0/0/0": lambda: element(self.vertices.tobytes()),
            "0/vertex_fragments/c/0/0/0": lambda: element(fragment_index(self.vertex_counts)),
            "0/link_fragments/c/0/0/0": lambda: element(fragment_index(self.face_counts)),
            "0/links/0/0.0.0_0.0.0/c/0/0/0": lambda: element(self.faces.tobytes()),
        }
        if key in fixed:
            return fixed[key]()
        m = re.match(r"^0/object_attributes/([^/]+)/zarr\.json$", key)
        if m and m.group(1) in self.columns:
            return j(_column(self.count, COLUMN_ROWS, "float32", "NaN", {"zv_array": "object_attribute", "name": m.group(1), "dtype": "float32", "shape": [self.count], "fill_sentinel_meaning": "absent"}))
        if key in ("0/links/zarr.json", "0/object_attributes/zarr.json"):
            return j(_group())
        m = re.match(r"^0/(object_index/manifests|object_index/object_ids|object_attributes/[^/]+)/c/(\d+)$", key)
        if not m:
            return None
        array = m.group(1)
        first = int(m.group(2)) * (MANIFEST_ROWS if array == "object_index/manifests" else COLUMN_ROWS)
        if first >= self.count:
            return None
        if array == "object_index/manifests":
            items = [struct.pack("<IqqqBq", 1, 0, 0, 0, 0, o) for o in range(first, min(self.count, first + MANIFEST_ROWS))]
            return vlen_chunk(items, MANIFEST_ROWS)
        if array == "object_index/object_ids":
            ids = np.zeros(COLUMN_ROWS, "<i8")
            n = min(COLUMN_ROWS, self.count - first)
            ids[:n] = np.arange(first + 1, first + n + 1)
            return ids.tobytes()
        values = self.columns.get(array.split("/", 1)[1])
        if values is None:
            return None
        out = np.full(COLUMN_ROWS, np.nan, "<f4")
        part = values[first : first + COLUMN_ROWS]
        out[: len(part)] = part
        return out.tobytes()
