"""Ingest meshes from Wavefront OBJ files into zarr vectors.

Pure Python parser — no external dependencies needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from zarr_vectors.exceptions import IngestError
from zarr_vectors.types.meshes import write_mesh
from zarr_vectors.typing import ChunkShape


def ingest_obj(
    input_path: str | Path,
    output_path: str | Path,
    chunk_shape: ChunkShape,
    *,
    dtype: str = "float32",
    encoding: str = "raw",
    draco_quantization_bits: int = 11,
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

    Returns:
        Summary dict from :func:`write_mesh`.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise IngestError(f"Input file not found: {input_path}")

    vertices: list[list[float]] = []
    normals_list: list[list[float]] = []
    faces: list[list[int]] = []

    try:
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if parts[0] == "v" and len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
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

    except IngestError:
        raise
    except Exception as e:
        raise IngestError(f"Failed to parse OBJ '{input_path}': {e}") from e

    if not vertices:
        raise IngestError(f"OBJ file has no vertices: {input_path}")

    np_dtype = np.dtype(dtype)
    positions = np.array(vertices, dtype=np_dtype)

    if not faces:
        raise IngestError(f"OBJ file has no faces: {input_path}")

    # Determine link width (3 for tris, 4 for quads)
    face_sizes = set(len(f) for f in faces)
    if face_sizes == {3}:
        link_width = 3
    elif face_sizes == {4}:
        link_width = 4
    else:
        # Mixed — triangulate quads
        tri_faces: list[list[int]] = []
        for f in faces:
            if len(f) == 3:
                tri_faces.append(f)
            elif len(f) == 4:
                tri_faces.append([f[0], f[1], f[2]])
                tri_faces.append([f[0], f[2], f[3]])
        faces = tri_faces
        link_width = 3

    faces_arr = np.array(faces, dtype=np.int64)

    vertex_attributes: dict[str, np.ndarray] | None = None
    if normals_list and len(normals_list) == len(vertices):
        vertex_attributes = {
            "normal": np.array(normals_list, dtype=np.float32),
        }

    return write_mesh(
        str(output_path),
        positions,
        faces_arr,
        chunk_shape=chunk_shape,
        encoding=encoding,
        vertex_attributes=vertex_attributes,
        dtype=dtype,
        draco_quantization_bits=draco_quantization_bits,
    )
