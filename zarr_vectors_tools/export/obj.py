"""Export zarr vectors meshes to Wavefront OBJ format."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from zarr_vectors.exceptions import ExportError
from zarr_vectors.types.meshes import read_mesh
from zarr_vectors.typing import BoundingBox


def export_obj(
    store_path: str | Path,
    output_path: str | Path,
    *,
    level: int = 0,
    bbox: BoundingBox | None = None,
    object_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Export a zarr vectors mesh to a Wavefront OBJ file.

    Args:
        store_path: Path to the zarr vectors store.
        output_path: Path for the output .obj file.
        level: Resolution level to export.
        bbox: Optional bounding box filter.
        object_ids: Optional object ID filter.

    Returns:
        Summary dict with ``vertex_count``, ``face_count``.

    Raises:
        ExportError: If export fails.
    """
    try:
        result = read_mesh(
            str(store_path),
            level=level,
            bbox=bbox,
            object_ids=object_ids,
        )
    except Exception as e:
        raise ExportError(f"Failed to read store: {e}") from e

    vertices = result["vertices"]
    faces = result["faces"]
    n_verts = len(vertices)
    n_faces = len(faces)

    if n_verts == 0:
        raise ExportError("No vertices to export")

    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write("# OBJ exported by zarr-vectors\n")
            f.write(f"# {n_verts} vertices, {n_faces} faces\n")

            # Write vertices
            for v in vertices:
                if len(v) >= 3:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                elif len(v) == 2:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} 0.000000\n")

            # Write faces (OBJ is 1-indexed)
            for face in faces:
                indices = " ".join(str(int(idx) + 1) for idx in face)
                f.write(f"f {indices}\n")

    except Exception as e:
        raise ExportError(f"Failed to write OBJ '{output_path}': {e}") from e

    return {"vertex_count": n_verts, "face_count": n_faces}
