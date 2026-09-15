"""Ingest meshes from STL files (ASCII and binary) into zarr vectors.

Both parsers are numpy only -- no external dependencies needed.  Binary STL
is read as blocks of fixed 50-byte records; ASCII STL is tokenised with
numpy and falls back to a line-by-line reader for text it cannot parse
exactly (see :mod:`zarr_vectors_tools.convert.ingest._text_tokens`).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from zarr_vectors.exceptions import IngestError
from zarr_vectors.types.meshes import write_mesh
from zarr_vectors.typing import BinShape, ChunkShape

from zarr_vectors_tools.convert.ingest._text_tokens import (
    iter_chunks,
    line_heads,
    parse_floats,
    read_plain_ascii,
    starts_with,
    tokenize,
)

# One binary STL triangle: facet normal, three vertices, and the 16-bit
# "attribute byte count" that nothing reads.
_BINARY_FACET = np.dtype([
    ("normal", "<f4", (3,)),
    ("vertices", "<f4", (3, 3)),
    ("attribute", "<u2"),
])
_BINARY_HEADER_BYTES = 84
# Triangles decoded per read, so the transient record buffer stays at
# about 50 MB whatever the mesh size.
_FACETS_PER_READ = 1 << 20


def ingest_stl(
    input_path: str | Path,
    output_path: str | Path,
    chunk_shape: ChunkShape,
    *,
    bin_shape: BinShape | None = None,
    dtype: str = "float32",
    encoding: str = "raw",
    merge_vertices: bool = True,
    merge_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Ingest an STL file into a zarr vectors mesh store.

    STL stores each triangle with 3 independent vertices (no sharing),
    so ``merge_vertices=True`` (default) deduplicates vertices that are
    within ``merge_tolerance``.

    Args:
        input_path: Path to the input .stl file.
        output_path: Path for the output zarr vectors store.
        chunk_shape: Spatial chunk size per dimension (3D).
        dtype: Dtype for position data.
        encoding: ``"raw"`` or ``"draco"``.
        merge_vertices: If True, merge duplicate vertices.
        merge_tolerance: Distance threshold for merging.

    Returns:
        Summary dict from :func:`write_mesh`.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise IngestError(f"Input file not found: {input_path}")

    try:
        if _is_ascii_stl(input_path):
            raw_verts, raw_normals = _parse_ascii_stl(input_path)
        else:
            raw_verts, raw_normals = _parse_binary_stl(input_path)
    except IngestError:
        raise
    except Exception as e:
        raise IngestError(f"Failed to parse STL '{input_path}': {e}") from e

    if len(raw_verts) == 0:
        raise IngestError(f"STL file has no triangles: {input_path}")

    np_dtype = np.dtype(dtype)

    # raw_verts: (F*3, 3) — three vertices per face
    n_raw = len(raw_verts)
    n_faces = n_raw // 3

    if merge_vertices:
        positions, faces = _merge_vertices(raw_verts, n_faces, merge_tolerance)
    else:
        positions = raw_verts
        faces = np.arange(n_raw, dtype=np.int64).reshape(n_faces, 3)

    positions = positions.astype(np_dtype)
    # Per-face normals are parsed but not stored: the mesh writer derives
    # them from the geometry, so keeping them would be a second source of
    # truth that nothing reads.

    return write_mesh(
        str(output_path),
        positions,
        faces,
        chunk_shape=chunk_shape,
        bin_shape=bin_shape,
        encoding=encoding,
        dtype=dtype,
    )


def _is_ascii_stl(path: Path) -> bool:
    with open(path, "rb") as f:
        header = f.read(80)
    return header.lstrip().lower().startswith(b"solid")


def _parse_ascii_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    parsed = _parse_ascii_stl_fast(path)
    if parsed is not None:
        return parsed
    return _parse_ascii_stl_lines(path)


def _parse_ascii_stl_fast(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Vectorised :func:`_parse_ascii_stl_lines`, or ``None`` if it cannot be exact.

    Follows the line reader's rules rather than the STL grammar: a line whose
    stripped text starts with ``facet normal`` gives a normal from its third
    to fifth fields, one starting with ``vertex`` a vertex from its second to
    fourth, and every other line is ignored.
    """
    data = read_plain_ascii(path)
    if data is None:
        return None

    vertex_parts: list[np.ndarray] = []
    normal_parts: list[np.ndarray] = []
    for arr in iter_chunks(data):
        table = tokenize(arr)
        heads = line_heads(table)
        has_tokens = table.count > 0
        # Matched on the raw bytes, so "facet\tnormal" or "facet  normal"
        # does not count, just as it does not for str.startswith.
        is_facet = has_tokens & starts_with(arr, heads, b"facet normal")
        is_vertex = has_tokens & ~is_facet & starts_with(arr, heads, b"vertex")
        # Too few fields is an IndexError in the line reader; leave that
        # error to it.
        if np.any(table.count[is_facet] < 5) or np.any(table.count[is_vertex] < 4):
            return None
        normals = parse_floats(table, np.flatnonzero(is_facet), 2)
        vertices = parse_floats(table, np.flatnonzero(is_vertex), 1)
        if normals is None or vertices is None:
            return None
        normal_parts.append(normals)
        vertex_parts.append(vertices)

    return _stack_rows(vertex_parts), _stack_rows(normal_parts)


def _stack_rows(parts: list[np.ndarray]) -> np.ndarray:
    if not parts:
        return np.zeros((0, 3))
    return np.concatenate(parts, axis=0)


def _parse_ascii_stl_lines(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Line-by-line ASCII STL reader: the reference the fast path must match."""
    vertices: list[list[float]] = []
    normals: list[list[float]] = []

    with open(path) as f:
        current_normal: list[float] = [0, 0, 0]
        for line in f:
            line = line.strip()
            if line.startswith("facet normal"):
                parts = line.split()
                current_normal = [float(parts[2]), float(parts[3]), float(parts[4])]
                normals.append(current_normal)
            elif line.startswith("vertex"):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return (
        np.array(vertices, dtype=np.float64) if vertices else np.zeros((0, 3)),
        np.array(normals, dtype=np.float64) if normals else np.zeros((0, 3)),
    )


def _parse_binary_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a binary STL as ``(F*3, 3)`` float32 vertices and ``(F, 3)`` normals.

    Bytes after the declared triangles are ignored.
    """
    record = _BINARY_FACET.itemsize
    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        header = f.read(_BINARY_HEADER_BYTES)
        if len(header) < _BINARY_HEADER_BYTES:
            raise IngestError(
                f"Binary STL '{path}' is {size} bytes, too short for the "
                f"{_BINARY_HEADER_BYTES}-byte header and triangle count"
            )
        n_faces = int.from_bytes(header[80:84], "little")
        # The attribute word of the last triangle is never read, and the
        # parser has always accepted a file that stops short of it; any
        # other shortfall means missing coordinates.  Checked before
        # allocating, so a corrupt count is reported as such rather than as
        # a failed multi-gigabyte allocation.
        expected = _BINARY_HEADER_BYTES + record * n_faces
        if size < expected - 2:
            raise _truncated_stl(path, n_faces, expected, size)

        vertices = np.empty((n_faces * 3, 3), dtype=np.float32)
        normals = np.empty((n_faces, 3), dtype=np.float32)
        for first in range(0, n_faces, _FACETS_PER_READ):
            count = min(_FACETS_PER_READ, n_faces - first)
            want = count * record
            block = f.read(want)
            if len(block) < want - 2:
                # The file shrank after the size check.
                raise _truncated_stl(path, n_faces, expected, size)
            if len(block) < want:
                block += bytes(want - len(block))  # the missing attribute word
            facets = np.frombuffer(block, dtype=_BINARY_FACET)
            face_normals = normals[first:first + count]
            face_vertices = vertices[3 * first:3 * (first + count)]
            face_normals[...] = facets["normal"]
            face_vertices[...] = facets["vertices"].reshape(-1, 3)
            _quiet_nans(face_normals)
            _quiet_nans(face_vertices)

    return vertices, normals


def _truncated_stl(path: Path, n_faces: int, expected: int, size: int) -> IngestError:
    return IngestError(
        f"Binary STL '{path}' is truncated: its header declares {n_faces} "
        f"triangles, which take {expected} bytes, but the file is {size} bytes"
    )


def _quiet_nans(values: np.ndarray) -> None:
    """Set the quiet bit on NaN float32 values, in place.

    The parser used to hand every coordinate through ``struct.unpack`` as a
    Python float, and widening a signalling NaN to a double quiets it.
    Doing the same here keeps the parsed bytes identical to what that
    parser produced.
    """
    nan = np.isnan(values)
    if nan.any():
        values.view(np.uint32)[nan] |= np.uint32(0x00400000)


def _merge_vertices(
    raw_verts: np.ndarray, n_faces: int, tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Deduplicate vertices within tolerance using grid rounding."""
    if tolerance <= 0:
        unique, inverse = np.unique(raw_verts, axis=0, return_inverse=True)
    else:
        # Round to tolerance grid
        rounded = np.round(raw_verts / tolerance) * tolerance
        _, unique_idx, inverse = np.unique(
            rounded, axis=0, return_index=True, return_inverse=True,
        )
        unique = raw_verts[unique_idx]

    faces = inverse.reshape(n_faces, 3).astype(np.int64)
    return unique, faces
