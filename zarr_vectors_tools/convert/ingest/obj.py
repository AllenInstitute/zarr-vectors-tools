"""Ingest meshes from Wavefront OBJ files into zarr vectors.

The parser is numpy only -- no external dependencies needed.  Lines are
tokenised with numpy and face indices decoded without a Python loop; text it
cannot parse exactly falls back to a line-by-line reader (see
:mod:`zarr_vectors_tools.convert.ingest._text_tokens`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from zarr_vectors.exceptions import IngestError
from zarr_vectors.types.meshes import write_mesh
from zarr_vectors.typing import BinShape, ChunkShape

from zarr_vectors_tools.convert.ingest._text_tokens import (
    TokenTable,
    iter_chunks,
    line_heads,
    parse_floats,
    read_plain_ascii,
    tokenize,
)

# An index longer than this may not fit in int64, where the line reader's
# int() never overflows; such a file takes the line reader.
_MAX_INDEX_DIGITS = 18


class _ParsedOBJ(NamedTuple):
    """What either reader extracts from an OBJ file.

    The fast reader fills these with arrays and the line reader with lists;
    :func:`ingest_obj` converts both the same way.
    """

    vertices: Any  # (V, 3) coordinates
    normals: Any  # (N, 3) ``vn`` vectors
    faces: Any  # (F, 3) triangles, or (F, 4) when every face is a quad
    vertex_object_ids: Any  # (V,) when auto_object_id, else empty
    object_names: list[str]


def ingest_obj(
    input_path: str | Path,
    output_path: str | Path,
    chunk_shape: ChunkShape,
    *,
    bin_shape: BinShape | None = None,
    dtype: str = "float32",
    encoding: str = "raw",
    draco_quantization_bits: int = 11,
    auto_object_id: bool = False,
) -> dict[str, Any]:
    """Ingest an OBJ file into a zarr vectors mesh store.

    Supports triangular and quad faces.  Polygon faces with more than
    4 vertices are fan-triangulated.

    Args:
        input_path: Path to the input .obj file.
        output_path: Path for the output zarr vectors store.
        chunk_shape: Spatial chunk size per dimension (3D).
        dtype: Dtype for position data.
        encoding: ``"raw"`` or ``"draco"``.
        draco_quantization_bits: For Draco encoding.
        auto_object_id: If True, parse ``o <name>`` / ``g <name>``
            directives and assign each vertex an integer object ID based
            on the most recently declared group. Object names are stored
            in ``OBJHeader.object_names``.

    Returns:
        Summary dict from :func:`write_mesh`.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise IngestError(f"Input file not found: {input_path}")

    try:
        parsed = _parse_obj(input_path, auto_object_id)
    except IngestError:
        raise
    except Exception as e:
        raise IngestError(f"Failed to parse OBJ '{input_path}': {e}") from e
    vertices, normals, faces, vertex_object_ids, object_names = parsed

    if len(vertices) == 0:
        raise IngestError(f"OBJ file has no vertices: {input_path}")

    np_dtype = np.dtype(dtype)
    if isinstance(vertices, np.ndarray) and np_dtype.kind != "f":
        # Converting Python floats to an integer dtype raises on a value out
        # of range, where casting a float array wraps it silently; keep the
        # error the line reader always gave.
        vertices = vertices.tolist()
    positions = np.asarray(vertices, dtype=np_dtype)

    if len(faces) == 0:
        raise IngestError(f"OBJ file has no faces: {input_path}")

    faces_arr = np.asarray(faces, dtype=np.int64)

    vertex_attributes: dict[str, np.ndarray] | None = None
    if len(normals) and len(normals) == len(vertices):
        vertex_attributes = {
            "normal": np.asarray(normals, dtype=np.float32),
        }

    object_ids_arr: np.ndarray | None = None
    if auto_object_id and len(vertex_object_ids):
        object_ids_arr = np.asarray(vertex_object_ids, dtype=np.int64)

    write_kwargs: dict[str, Any] = {
        "chunk_shape": chunk_shape,
        "bin_shape": bin_shape,
        "encoding": encoding,
        "vertex_attributes": vertex_attributes,
        "dtype": dtype,
        "draco_quantization_bits": draco_quantization_bits,
    }
    if object_ids_arr is not None:
        write_kwargs["object_ids"] = object_ids_arr

    result = write_mesh(str(output_path), positions, faces_arr, **write_kwargs)

    if auto_object_id and object_names:
        try:
            from zarr_vectors_tools.headers.formats import OBJHeader
            from zarr_vectors_tools.headers.registry import HeaderRegistry

            obj_header = OBJHeader(object_names=object_names)
            HeaderRegistry(str(output_path)).add("obj", obj_header)
        except Exception:
            pass

    return result


def _parse_obj(path: Path, auto_object_id: bool) -> _ParsedOBJ:
    parsed = _parse_obj_fast(path, auto_object_id)
    if parsed is not None:
        return parsed
    return _parse_obj_lines(path, auto_object_id)


def _parse_obj_fast(path: Path, auto_object_id: bool) -> _ParsedOBJ | None:
    """Vectorised :func:`_parse_obj_lines`, or ``None`` if it cannot be exact.

    Only coordinates go through Python, one ``float`` per field, because
    that is the conversion the line reader defines its values by.  Face
    indices are decoded in numpy when every one is a plain optionally
    negative decimal before any ``/``; anything else (``+1``, ``1_0``, an
    inline comment on a face line) sends the file to the line reader.
    """
    data = read_plain_ascii(path)
    if data is None:
        return None

    vertex_parts: list[np.ndarray] = []
    normal_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    arity_parts: list[np.ndarray] = []
    id_parts: list[np.ndarray] = []
    object_names: list[str] = []
    name_ids: dict[str, int] = {}
    n_vertices = 0  # declared so far, which is what a negative index counts back from
    current_obj_id = -1

    for arr in iter_chunks(data):
        table = tokenize(arr)
        heads = line_heads(table)
        head_len = np.zeros_like(heads)
        has_tokens = table.count > 0
        head_len[has_tokens] = (
            table.ends[table.first[has_tokens]] - heads[has_tokens]
        )
        c0 = arr[heads]
        c1 = arr[np.minimum(heads + 1, arr.size - 1)]
        # A line is kept by the keyword that is its whole first field, and
        # only with enough fields; shorter "v", "vn" and "f" lines have always
        # been skipped.  Comment lines start with "#", which no keyword does.
        one_char = has_tokens & (head_len == 1)
        is_v = one_char & (c0 == ord("v")) & (table.count >= 4)
        is_vn = (head_len == 2) & (c0 == ord("v")) & (c1 == ord("n")) & (table.count >= 4)
        is_f = one_char & (c0 == ord("f")) & (table.count >= 4)

        v_lines = np.flatnonzero(is_v)
        vertices = parse_floats(table, v_lines, 1)
        normals = parse_floats(table, np.flatnonzero(is_vn), 1)
        if vertices is None or normals is None:
            return None
        vertex_parts.append(vertices)
        normal_parts.append(normals)

        f_lines = np.flatnonzero(is_f)
        if f_lines.size:
            arity = table.count[f_lines] - 1
            # A face line is never a vertex line, so the running count at it
            # is the number of vertices declared above it.
            declared = n_vertices + np.cumsum(is_v, dtype=np.int64)[f_lines]
            indices = _face_indices(table, f_lines, arity, declared)
            if indices is None:
                return None
            index_parts.append(indices)
            arity_parts.append(arity)

        if auto_object_id:
            is_group = one_char & ((c0 == ord("o")) | (c0 == ord("g"))) & (table.count >= 2)
            group_lines = np.flatnonzero(is_group)
            group_ids = np.empty(group_lines.size, dtype=np.int64)
            for k, line in enumerate(group_lines.tolist()):
                first = table.first[line]
                lo = table.starts[first + 1]
                hi = table.ends[first + table.count[line] - 1]
                name = b" ".join(arr[lo:hi].tobytes().split()).decode("ascii")
                if name not in name_ids:
                    name_ids[name] = len(object_names)
                    object_names.append(name)
                group_ids[k] = name_ids[name]
            # Each vertex takes the id of the last o/g line above it.
            ids = np.full(v_lines.size, max(current_obj_id, 0), dtype=np.int64)
            if group_lines.size:
                above = np.searchsorted(group_lines, v_lines)
                ids[above > 0] = group_ids[above[above > 0] - 1]
                current_obj_id = int(group_ids[-1])
            id_parts.append(ids)

        n_vertices += v_lines.size

    arity = _concat(arity_parts, np.int64, (0,))
    return _ParsedOBJ(
        vertices=_concat(vertex_parts, np.float64, (0, 3)),
        normals=_concat(normal_parts, np.float64, (0, 3)),
        faces=_assemble_faces(_concat(index_parts, np.int64, (0,)), arity),
        vertex_object_ids=_concat(id_parts, np.int64, (0,)),
        object_names=object_names,
    )


def _face_indices(
    table: TokenTable, lines: np.ndarray, arity: np.ndarray, declared: np.ndarray,
) -> np.ndarray | None:
    """Zero-based vertex index of every corner of the given face lines.

    Returns the flat indices in file order, or ``None`` if any corner is not
    something this decoder reproduces ``int()`` on exactly.
    """
    arr = table.arr
    total = int(arity.sum())
    offsets = np.cumsum(arity) - arity
    token = np.repeat(table.first[lines] + 1 - offsets, arity) + np.arange(total)
    lo = table.starts[token]
    hi = table.ends[token]

    # Only the text before the first "/" is the vertex index (v, v/vt,
    # v//vn, v/vt/vn); whatever follows it is never read.
    slashes = np.flatnonzero(arr == ord("/"))
    if slashes.size:
        after = np.searchsorted(slashes, lo)
        slash = slashes[np.minimum(after, slashes.size - 1)]
        hi = np.where((after < slashes.size) & (slash < hi), slash, hi)

    negative = arr[lo] == ord("-")
    lo = lo + negative
    width = hi - lo
    if width.min() < 1 or width.max() > _MAX_INDEX_DIGITS:
        return None

    value = np.zeros(total, dtype=np.int64)
    last = arr.size - 1
    for j in range(int(width.max())):
        live = width > j
        digit = arr[np.minimum(lo + j, last)].astype(np.int64) - ord("0")
        if np.any(live & ((digit < 0) | (digit > 9))):
            return None
        value = np.where(live, value * 10 + digit, value)
    value = np.where(negative, -value, value)

    # OBJ counts from 1; a negative index counts back from the vertices
    # declared so far.  An index of 0 becomes -1, as it always has.
    before = np.repeat(declared, arity)
    return np.where(value < 0, before + value, value - 1)


def _assemble_faces(indices: np.ndarray, arity: np.ndarray) -> np.ndarray:
    """Build the face array the line reader builds, from flat corner indices.

    A file of nothing but quads keeps them.  Otherwise every face is
    fan-triangulated, which is what the line reader does to a polygon as it
    parses it and to each quad of a file that mixes quads with triangles:
    its split ``[0, 1, 2], [0, 2, 3]`` is the two-triangle fan.
    """
    if arity.size and np.all(arity == 4):
        return indices.reshape(-1, 4)
    if np.all(arity == 3):
        return indices.reshape(-1, 3)
    n_tri = arity - 2
    corner0 = np.repeat(np.cumsum(arity) - arity, n_tri)
    step = np.arange(int(n_tri.sum())) - np.repeat(np.cumsum(n_tri) - n_tri, n_tri) + 1
    return np.stack(
        [indices[corner0], indices[corner0 + step], indices[corner0 + step + 1]],
        axis=1,
    )


def _concat(parts: list[np.ndarray], dtype: type, empty_shape: tuple[int, ...]) -> np.ndarray:
    if not parts:
        return np.empty(empty_shape, dtype=dtype)
    return np.concatenate(parts, axis=0)


def _parse_obj_lines(path: Path, auto_object_id: bool) -> _ParsedOBJ:
    """Line-by-line OBJ reader: the reference the fast path must match."""
    vertices: list[list[float]] = []
    normals_list: list[list[float]] = []
    faces: list[list[int]] = []
    vertex_object_ids: list[int] = []
    object_names: list[str] = []
    current_obj_id = -1

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if auto_object_id:
                    vertex_object_ids.append(max(current_obj_id, 0))
            elif parts[0] in ("o", "g") and len(parts) >= 2 and auto_object_id:
                name = " ".join(parts[1:])
                if name not in object_names:
                    object_names.append(name)
                current_obj_id = object_names.index(name)
            elif parts[0] == "vn" and len(parts) >= 4:
                normals_list.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                # Parse face indices (OBJ is 1-indexed, may have v/vt/vn)
                face_indices: list[int] = []
                for p in parts[1:]:
                    idx_str = p.split("/")[0]
                    idx = int(idx_str)
                    # Handle negative indices (relative to end)
                    if idx < 0:
                        idx = len(vertices) + idx
                    else:
                        idx = idx - 1  # 0-index
                    face_indices.append(idx)

                if len(face_indices) == 3:
                    faces.append(face_indices)
                elif len(face_indices) == 4:
                    faces.append(face_indices)
                elif len(face_indices) > 4:
                    # Fan triangulate
                    for i in range(1, len(face_indices) - 1):
                        faces.append([
                            face_indices[0],
                            face_indices[i],
                            face_indices[i + 1],
                        ])

    # A uniform file is stored as it stands -- all triangles give link width
    # 3, all quads give 4, and the writer takes the width from the face array
    # itself.  A MIXED file has no single width, so its quads are split.
    face_sizes = set(len(f) for f in faces)
    if face_sizes not in ({3}, {4}):
        tri_faces: list[list[int]] = []
        for f in faces:
            if len(f) == 3:
                tri_faces.append(f)
            elif len(f) == 4:
                tri_faces.append([f[0], f[1], f[2]])
                tri_faces.append([f[0], f[2], f[3]])
        faces = tri_faces

    return _ParsedOBJ(vertices, normals_list, faces, vertex_object_ids, object_names)
