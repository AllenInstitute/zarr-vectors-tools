"""Export zarr vectors stores to Neuroglancer Precomputed format.

Requires ``cloud-volume``: ``pip install zarr-vectors-tools[neuroglancer]``.

Three public entry points mirroring the ingest module:

* :func:`export_precomputed_mesh` — write a ZVF mesh as a legacy
  unsharded precomputed mesh layer.
* :func:`export_precomputed_skeleton` — write a ZVF skeleton/graph as
  an unsharded precomputed skeleton layer.
* :func:`export_precomputed_annotations` — write a ZVF point or line
  store as a precomputed annotation layer.

When a :class:`NeuroglancerHeader` is present in the source store, the
original resolution, transform, and segment-ID mapping are restored.
Otherwise sensible defaults are used (resolution=1nm, identity
transform, sequential segment IDs starting at 1).

Sharded precomputed output is not implemented in this version — see
``igneous`` for that pipeline.

.. note::
    Legacy unsharded precomputed mesh and skeleton layers use
    ``<segment_id>:<lod>`` filenames, which are not valid on Windows.
    Local-filesystem mesh and skeleton export therefore only works on
    POSIX systems (Linux, macOS, WSL). Annotation layers and remote
    (gs://, s3://) targets are unaffected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from zarr_vectors.exceptions import ExportError
from zarr_vectors.types.graphs import read_graph
from zarr_vectors.types.lines import read_lines
from zarr_vectors.types.meshes import read_mesh
from zarr_vectors.types.points import read_points
from zarr_vectors.typing import BoundingBox, ChunkCoords

_INSTALL_HINT = (
    "cloud-volume is required for Neuroglancer Precomputed export. "
    "Install with: pip install zarr-vectors-tools[neuroglancer]"
)


def _import_cloudvolume():
    try:
        import cloudvolume  # type: ignore
    except ImportError as e:
        raise ExportError(_INSTALL_HINT) from e
    return cloudvolume


def _output_url(output_path: str | Path) -> str:
    """Build a ``file://`` URL for a local output directory."""
    p = Path(output_path)
    p.mkdir(parents=True, exist_ok=True)
    return "file://" + str(p.resolve()).replace("\\", "/")


def _load_header(store_path: str | Path):
    """Return a NeuroglancerHeader if present, else None."""
    try:
        from zarr_vectors_tools.headers.registry import HeaderRegistry
        reg = HeaderRegistry(str(store_path))
        if reg.has("neuroglancer"):
            return reg.get("neuroglancer")
    except Exception:
        return None
    return None


def _resolve_segment_ids(
    header: Any, object_count: int, *, default_start: int = 1
) -> list[int]:
    """Map ZVF object slots back to precomputed segment IDs.

    Uses the header's ``segment_ids`` list when available; otherwise
    falls back to ``default_start..default_start+object_count-1``.
    """
    if header is not None and header.segment_ids:
        ids = [int(s) for s in header.segment_ids]
        if len(ids) >= object_count:
            return ids[:object_count]
        return ids + list(
            range(default_start + len(ids), default_start + object_count)
        )
    return list(range(default_start, default_start + object_count))


# ===================================================================
# Meshes
# ===================================================================

def export_precomputed_mesh(
    store_path: str | Path,
    output_path: str | Path,
    *,
    level: int = 0,
    bbox: BoundingBox | None = None,
    object_ids: list[int] | None = None,
    chunks: list[ChunkCoords] | None = None,
    encoding: str = "raw",
    sharded: bool = False,
) -> dict[str, Any]:
    """Export a ZVF mesh as an unsharded precomputed mesh layer.

    Args:
        store_path: Source ZVF store.
        output_path: Destination directory for the precomputed layer.
        level: Resolution level to read.
        bbox: Optional spatial filter.
        object_ids: Optional object-ID filter.
        chunks: Optional chunk-whitelist filter.
        encoding: Currently only ``"raw"`` is supported on export.
        sharded: Sharded output is not implemented — raises if True.

    Returns:
        Dict with ``vertex_count``, ``face_count``, ``segment_count``,
        and ``segment_ids`` (the precomputed IDs written).
    """
    if sharded:
        raise NotImplementedError(
            "Sharded precomputed mesh export is not implemented. "
            "Use cloud-volume + igneous for sharded output."
        )
    if encoding != "raw":
        raise ExportError(
            f"Only encoding='raw' is supported on export (got {encoding!r})"
        )

    cloudvolume = _import_cloudvolume()

    header = _load_header(store_path)
    resolution = list(header.resolution) if header is not None else [1.0, 1.0, 1.0]

    try:
        mesh_data = read_mesh(
            str(store_path),
            level=level,
            bbox=bbox,
            object_ids=object_ids,
            chunks=chunks,
        )
    except Exception as e:
        raise ExportError(f"Failed to read mesh store: {e}") from e

    verts = np.asarray(mesh_data["vertices"], dtype=np.float32)
    faces_p = np.asarray(mesh_data["faces"], dtype=np.uint32)
    if verts.size == 0:
        raise ExportError("No vertices to export")

    # `read_mesh` does not currently split its output by ``object_ids``, so
    # we emit all data as a single precomputed segment. If the source
    # store carried multiple original segments in its header, surface a
    # warning so the user knows the per-segment split was collapsed.
    seg_ids = _resolve_segment_ids(header, 1)
    if header is not None and len(header.segment_ids) > 1:
        import warnings

        warnings.warn(
            f"Source store has {len(header.segment_ids)} original segments but "
            f"zarr-vectors' read_mesh does not yet support per-object splitting; "
            f"all geometry is being written as a single precomputed segment "
            f"(id={seg_ids[0]}).",
            stacklevel=2,
        )

    out_url = _output_url(output_path)
    info = cloudvolume.CloudVolume.create_new_info(
        num_channels=1,
        layer_type="segmentation",
        data_type="uint64",
        encoding="raw",
        resolution=resolution,
        voxel_offset=[0, 0, 0],
        volume_size=[1, 1, 1],
        chunk_size=[1, 1, 1],
        mesh="mesh",
    )
    cv = cloudvolume.CloudVolume(out_url, info=info, compress=False)
    cv.commit_info()
    mesh_dir = Path(_url_to_path(out_url)) / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    sid = seg_ids[0]
    _write_precomputed_mesh_fragment(mesh_dir, sid, verts, faces_p)

    return {
        "vertex_count": int(verts.shape[0]),
        "face_count": int(faces_p.shape[0]),
        "segment_count": 1,
        "segment_ids": [sid],
        "output_path": str(Path(output_path).resolve()),
    }


def _url_to_path(url: str) -> str:
    if url.startswith("file://"):
        return url[len("file://") :]
    return url


def _write_precomputed_mesh_fragment(
    mesh_dir: Path, segment_id: int, vertices: np.ndarray, faces: np.ndarray
) -> None:
    """Write one segment's mesh in legacy precomputed binary layout.

    Layout: ``uint32 N`` num vertices, ``float32 3*N`` xyz positions,
    ``uint32 3*M`` triangle indices. Plus a JSON manifest pointing at
    the binary fragment.
    """
    import json

    fragment_key = f"{segment_id}:0"
    fragment_path = mesh_dir / fragment_key
    manifest_path = mesh_dir / f"{segment_id}:0:manifest.json"

    n = np.uint32(len(vertices))
    with open(fragment_path, "wb") as f:
        f.write(n.tobytes())
        f.write(np.ascontiguousarray(vertices, dtype=np.float32).tobytes())
        f.write(np.ascontiguousarray(faces, dtype=np.uint32).tobytes())

    manifest_path.write_text(json.dumps({"fragments": [fragment_key]}))


# ===================================================================
# Skeletons
# ===================================================================

def export_precomputed_skeleton(
    store_path: str | Path,
    output_path: str | Path,
    *,
    level: int = 0,
    bbox: BoundingBox | None = None,
    object_ids: list[int] | None = None,
    chunks: list[ChunkCoords] | None = None,
) -> dict[str, Any]:
    """Export a ZVF skeleton/graph as an unsharded precomputed skeleton layer.

    Args:
        store_path: Source ZVF store.
        output_path: Destination directory for the precomputed layer.
        level, bbox, object_ids, chunks: Standard ZVF read filters.

    Returns:
        Dict with ``vertex_count``, ``edge_count``, ``segment_count``,
        ``segment_ids``.
    """
    cloudvolume = _import_cloudvolume()

    header = _load_header(store_path)
    resolution = list(header.resolution) if header is not None else [1.0, 1.0, 1.0]

    try:
        graph_data = read_graph(
            str(store_path),
            level=level,
            bbox=bbox,
            object_ids=object_ids,
            chunks=chunks,
        )
    except Exception as e:
        raise ExportError(f"Failed to read skeleton store: {e}") from e

    verts = np.asarray(graph_data["positions"], dtype=np.float32)
    edges_p = np.asarray(graph_data["edges"], dtype=np.uint32)
    if verts.size == 0:
        raise ExportError("No skeleton vertices to export")

    # Same single-segment collapse as the mesh exporter — see that
    # function's comment for context.
    seg_ids = _resolve_segment_ids(header, 1)
    if header is not None and len(header.segment_ids) > 1:
        import warnings

        warnings.warn(
            f"Source store has {len(header.segment_ids)} original segments but "
            f"zarr-vectors' read_graph does not yet support per-object splitting; "
            f"all geometry is being written as a single precomputed segment "
            f"(id={seg_ids[0]}).",
            stacklevel=2,
        )

    out_url = _output_url(output_path)
    info = cloudvolume.CloudVolume.create_new_info(
        num_channels=1,
        layer_type="segmentation",
        data_type="uint64",
        encoding="raw",
        resolution=resolution,
        voxel_offset=[0, 0, 0],
        volume_size=[1, 1, 1],
        chunk_size=[1, 1, 1],
        skeletons="skeletons",
    )
    cv = cloudvolume.CloudVolume(out_url, info=info, compress=False)
    cv.commit_info()

    skel_dir = Path(_url_to_path(out_url)) / "skeletons"
    skel_dir.mkdir(parents=True, exist_ok=True)
    _write_skeleton_metadata(skel_dir, header)

    sid = seg_ids[0]
    _write_precomputed_skeleton_fragment(skel_dir, sid, verts, edges_p)

    return {
        "vertex_count": int(verts.shape[0]),
        "edge_count": int(edges_p.shape[0]),
        "segment_count": 1,
        "segment_ids": [sid],
        "output_path": str(Path(output_path).resolve()),
    }


def _write_skeleton_metadata(skel_dir: Path, header: Any) -> None:
    """Write the precomputed skeleton ``info`` next to the fragments."""
    import json

    info: dict[str, Any] = {"@type": "neuroglancer_skeletons"}
    if header is not None:
        if header.transform is not None:
            info["transform"] = header.transform
        sm = header.skeleton_metadata or {}
        if sm.get("vertex_attributes"):
            info["vertex_attributes"] = sm["vertex_attributes"]
    (skel_dir / "info").write_text(json.dumps(info))


def _write_precomputed_skeleton_fragment(
    skel_dir: Path, segment_id: int, vertices: np.ndarray, edges: np.ndarray
) -> None:
    """Write one segment's skeleton in precomputed binary layout.

    Layout: ``uint32 num_vertices``, ``uint32 num_edges``,
    ``float32 3*V`` positions, ``uint32 2*E`` edge indices.
    """
    path = skel_dir / str(segment_id)
    with open(path, "wb") as f:
        f.write(np.uint32(len(vertices)).tobytes())
        f.write(np.uint32(len(edges)).tobytes())
        f.write(np.ascontiguousarray(vertices, dtype=np.float32).tobytes())
        f.write(np.ascontiguousarray(edges, dtype=np.uint32).tobytes())


# ===================================================================
# Annotations
# ===================================================================

def export_precomputed_annotations(
    store_path: str | Path,
    output_path: str | Path,
    *,
    level: int = 0,
    bbox: BoundingBox | None = None,
    object_ids: list[int] | None = None,
    annotation_type: str | None = None,
) -> dict[str, Any]:
    """Export a ZVF points or lines store as a precomputed annotation layer.

    The annotation type is inferred from the source store's
    :class:`NeuroglancerHeader` when present; otherwise the caller must
    pass ``annotation_type="POINT"`` or ``"LINE"`` explicitly.

    Args:
        store_path: Source ZVF store.
        output_path: Destination directory for the precomputed layer.
        level, bbox, object_ids: Standard ZVF read filters.
        annotation_type: Override the inferred type (``"POINT"`` or
            ``"LINE"``).

    Returns:
        Dict with ``annotation_count`` and ``output_path``.
    """
    _import_cloudvolume()  # surface install error early
    header = _load_header(store_path)

    resolved_type = (annotation_type or "").upper()
    if not resolved_type and header is not None:
        resolved_type = (header.annotation_type or "").upper()
    if resolved_type not in ("POINT", "LINE"):
        raise ExportError(
            "annotation_type must be 'POINT' or 'LINE' "
            "(pass it explicitly when no NeuroglancerHeader is present)"
        )

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    if resolved_type == "POINT":
        try:
            data = read_points(
                str(store_path),
                level=level,
                bbox=bbox,
                object_ids=object_ids,
            )
        except Exception as e:
            raise ExportError(f"Failed to read points store: {e}") from e
        positions = np.asarray(data["positions"], dtype=np.float32)
        annotation_count = int(positions.shape[0])
        _write_annotation_layer(
            out_dir,
            annotation_type="POINT",
            positions=positions,
            header=header,
        )
    else:
        try:
            data = read_lines(
                str(store_path),
                level=level,
                bbox=bbox,
                object_ids=object_ids,
            )
        except Exception as e:
            raise ExportError(f"Failed to read lines store: {e}") from e
        endpoints = np.asarray(data["endpoints"], dtype=np.float32)
        annotation_count = int(endpoints.shape[0])
        _write_annotation_layer(
            out_dir,
            annotation_type="LINE",
            endpoints=endpoints,
            header=header,
        )

    return {
        "annotation_count": annotation_count,
        "annotation_type": resolved_type,
        "output_path": str(out_dir.resolve()),
    }


def _write_annotation_layer(
    out_dir: Path,
    *,
    annotation_type: str,
    positions: np.ndarray | None = None,
    endpoints: np.ndarray | None = None,
    header: Any = None,
) -> None:
    """Write a minimal unsharded precomputed annotation layer.

    Writes:
    * ``info`` — JSON describing the layer
    * ``by_id/<aid>`` — per-annotation binary records
    * ``spatial0/0_0_0`` — single-cell spatial index containing all IDs

    The format follows the Neuroglancer precomputed annotation spec:
    https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md
    """
    import json

    resolution = list(header.resolution) if header is not None else [1.0, 1.0, 1.0]
    properties = (header.annotation_properties if header is not None else []) or []
    relationships = (header.relationships if header is not None else []) or []

    if annotation_type == "POINT":
        assert positions is not None
        n = int(positions.shape[0])
        coords = positions.reshape(n, 1, 3)
        lower = positions.min(axis=0).tolist() if n else [0.0, 0.0, 0.0]
        upper = positions.max(axis=0).tolist() if n else [1.0, 1.0, 1.0]
    else:
        assert endpoints is not None
        n = int(endpoints.shape[0])
        coords = endpoints.reshape(n, 2, 3)
        if n:
            flat = endpoints.reshape(-1, 3)
            lower = flat.min(axis=0).tolist()
            upper = flat.max(axis=0).tolist()
        else:
            lower, upper = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]

    # Guarantee a non-zero extent so neuroglancer's bbox math doesn't divide by zero.
    upper = [u if u > lo else lo + 1.0 for lo, u in zip(lower, upper)]

    info = {
        "@type": "neuroglancer_annotations_v1",
        "dimensions": {
            "x": [resolution[0], "nm"],
            "y": [resolution[1], "nm"],
            "z": [resolution[2], "nm"],
        },
        "lower_bound": lower,
        "upper_bound": upper,
        "annotation_type": annotation_type,
        "properties": properties,
        "relationships": [
            {"id": r.get("id") or r.get("key"), "key": r.get("key", r.get("id"))}
            for r in relationships
        ],
        "by_id": {"key": "by_id"},
        "spatial": [
            {
                "key": "spatial0",
                "grid_shape": [1, 1, 1],
                "chunk_size": [
                    upper[0] - lower[0],
                    upper[1] - lower[1],
                    upper[2] - lower[2],
                ],
                "limit": max(n, 1),
            }
        ],
    }
    (out_dir / "info").write_text(json.dumps(info))

    by_id_dir = out_dir / "by_id"
    spatial_dir = out_dir / "spatial0"
    by_id_dir.mkdir(parents=True, exist_ok=True)
    spatial_dir.mkdir(parents=True, exist_ok=True)

    # Spatial chunk: count (uint64 LE) + N * geometry + N * id (uint64 LE).
    spatial_buf = bytearray()
    spatial_buf += np.uint64(n).tobytes()
    spatial_buf += np.ascontiguousarray(coords, dtype=np.float32).tobytes()
    spatial_buf += np.arange(n, dtype=np.uint64).tobytes()
    (spatial_dir / "0_0_0").write_bytes(bytes(spatial_buf))

    # Per-annotation by_id record: geometry + relationship counts.
    for i in range(n):
        rec = bytearray()
        rec += np.ascontiguousarray(coords[i], dtype=np.float32).tobytes()
        # No properties or relationships: relationship counts are uint32 zero each.
        for _ in relationships:
            rec += np.uint32(0).tobytes()
        (by_id_dir / str(i)).write_bytes(bytes(rec))
