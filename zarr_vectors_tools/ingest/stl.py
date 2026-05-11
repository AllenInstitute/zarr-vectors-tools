"""Ingest meshes from STL files (ASCII and binary) into zarr vectors.

Pure Python parser — no external dependencies needed.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

from zarr_vectors.exceptions import IngestError
from zarr_vectors.types.meshes import write_mesh
from zarr_vectors.typing import ChunkShape


def ingest_stl(
    input_path: str | Path,
    output_path: str | Path,
    chunk_shape: ChunkShape,
    *,
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
    face_normals = raw_normals  # (F, 3) per-face normals from STL

    return write_mesh(
        str(output_path),
        positions,
        faces,
        chunk_shape=chunk_shape,
        encoding=encoding,
        dtype=dtype,
    )


def _is_ascii_stl(path: Path) -> bool:
    with open(path, "rb") as f:
        header = f.read(80)
    return header.lstrip().lower().startswith(b"solid")


def _parse_ascii_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
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
    with open(path, "rb") as f:
        f.read(80)  # header
        n_faces = struct.unpack("<I", f.read(4))[0]

        vertices = np.empty((n_faces * 3, 3), dtype=np.float32)
        normals = np.empty((n_faces, 3), dtype=np.float32)

        for i in range(n_faces):
            nx, ny, nz = struct.unpack("<3f", f.read(12))
            normals[i] = [nx, ny, nz]
            for j in range(3):
                x, y, z = struct.unpack("<3f", f.read(12))
                vertices[i * 3 + j] = [x, y, z]
            f.read(2)  # attribute byte count

    return vertices, normals


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
