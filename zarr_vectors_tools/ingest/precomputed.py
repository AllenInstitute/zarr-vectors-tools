"""Ingest Neuroglancer Precomputed sources into zarr vectors.

Requires ``cloud-volume``: ``pip install zarr-vectors-tools[neuroglancer]``.

Three public entry points covering the vector-flavoured precomputed
data types:

* :func:`ingest_precomputed_mesh` — segment meshes (sharded or legacy).
* :func:`ingest_precomputed_skeleton` — segment skeletons.
* :func:`ingest_precomputed_annotations` — point / line annotation layers.

All three accept either a local path or a cloud-volume URL
(``file://``, ``gs://``, ``s3://``, ``https://`` — whatever cloud-volume
knows how to parse).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from zarr_vectors.exceptions import IngestError
from zarr_vectors.types.graphs import write_graph
from zarr_vectors.types.lines import write_lines
from zarr_vectors.types.meshes import write_mesh
from zarr_vectors.types.points import write_points
from zarr_vectors.typing import BinShape, BoundingBox, ChunkShape

_INSTALL_HINT = (
    "cloud-volume is required for Neuroglancer Precomputed ingest. "
    "Install with: pip install zarr-vectors-tools[neuroglancer]"
)


def _import_cloudvolume():
    try:
        import cloudvolume  # type: ignore
    except ImportError as e:
        raise IngestError(_INSTALL_HINT) from e
    return cloudvolume


def _normalise_source(source: str | Path) -> str:
    """Convert a Path or bare string into a cloud-volume URL.

    Bare local paths get a ``file://`` prefix; URLs are passed through.
    """
    s = str(source)
    if "://" in s:
        return s
    return "file://" + str(Path(s).resolve()).replace("\\", "/")


def _info_resolution(info: dict[str, Any]) -> tuple[float, float, float]:
    """Pull the (x, y, z) voxel resolution from a precomputed info dict.

    Falls back to (1, 1, 1) when the field is missing or malformed.
    """
    scales = info.get("scales") or []
    if scales and "resolution" in scales[0]:
        r = scales[0]["resolution"]
        if len(r) >= 3:
            return (float(r[0]), float(r[1]), float(r[2]))
    return (1.0, 1.0, 1.0)


def _flatten_transform(t: Any) -> list[float] | None:
    """Flatten a precomputed transform (3×4 list-of-lists) to 12 floats."""
    if t is None:
        return None
    try:
        arr = np.asarray(t, dtype=np.float64).flatten()
        if arr.size in (12, 16):
            return arr.tolist()
    except Exception:
        pass
    return None


# ===================================================================
# Meshes
# ===================================================================

def ingest_precomputed_mesh(
    source: str | Path,
    output_path: str | Path,
    chunk_shape: ChunkShape,
    *,
    segment_ids: Sequence[int] | None = None,
    lod: int = 0,
    bin_shape: BinShape | None = None,
    dtype: str = "float32",
    encoding: str = "raw",
    draco_quantization_bits: int = 11,
    preserve_header: bool = True,
) -> dict[str, Any]:
    """Ingest precomputed segment meshes into a zarr vectors mesh store.

    Each precomputed segment becomes one ZVF object, identified by its
    0-based position in the ``segment_ids`` list. Original uint64 IDs are
    preserved in the :class:`NeuroglancerHeader` for round-trip export.

    Args:
        source: cloud-volume URL or local precomputed path.
        output_path: Destination ZVF store path.
        chunk_shape: Spatial chunk size per dimension (3D).
        segment_ids: Segment IDs to ingest. If None, cloud-volume's
            mesh listing is queried.
        lod: Level-of-detail to pull for multi-resolution sharded meshes.
        bin_shape: Optional sub-chunk binning.
        dtype: Numpy dtype for vertex positions.
        encoding: ``"raw"`` or ``"draco"`` for the output ZVF store.
        draco_quantization_bits: Draco quantisation bits when
            ``encoding="draco"``.
        preserve_header: Store a :class:`NeuroglancerHeader` under
            ``/headers/neuroglancer/`` for round-trip export.

    Returns:
        Summary dict from :func:`write_mesh`, with an extra
        ``segment_count`` field.
    """
    cloudvolume = _import_cloudvolume()
    source_url = _normalise_source(source)

    try:
        cv = cloudvolume.CloudVolume(source_url, progress=False, use_https=True)
    except Exception as e:
        raise IngestError(f"Failed to open precomputed source '{source_url}': {e}") from e

    if not hasattr(cv, "mesh"):
        raise IngestError(f"Source '{source_url}' has no mesh data")

    if segment_ids is None:
        try:
            seg_ids = list(cv.mesh.get_all_ids())
        except Exception as e:
            raise IngestError(
                f"Could not list mesh segment IDs for '{source_url}'. "
                f"Pass segment_ids explicitly. ({e})"
            ) from e
    else:
        seg_ids = [int(s) for s in segment_ids]

    if not seg_ids:
        raise IngestError(f"No mesh segments found in '{source_url}'")

    np_dtype = np.dtype(dtype)
    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    all_obj_ids: list[np.ndarray] = []
    kept_seg_ids: list[int] = []
    vertex_offset = 0
    accepts_lod = "lod" in _get_kwargs(cv.mesh.get)

    for slot, sid in enumerate(seg_ids):
        try:
            mesh = cv.mesh.get(sid, lod=lod) if accepts_lod else cv.mesh.get(sid)
        except Exception as e:
            raise IngestError(f"Failed to fetch mesh for segment {sid}: {e}") from e

        # cv.mesh.get returns either a Mesh or {sid: Mesh}
        if isinstance(mesh, dict):
            mesh = mesh.get(sid) or next(iter(mesh.values()), None)
        if mesh is None or len(mesh.vertices) == 0:
            continue

        verts = np.asarray(mesh.vertices, dtype=np_dtype)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if faces.ndim != 2 or faces.shape[1] < 3:
            continue

        all_verts.append(verts)
        all_faces.append(faces + vertex_offset)
        all_obj_ids.append(np.full(len(verts), slot, dtype=np.int64))
        kept_seg_ids.append(int(sid))
        vertex_offset += len(verts)

    if not all_verts:
        raise IngestError(f"All mesh segments in '{source_url}' were empty")

    positions = np.concatenate(all_verts, axis=0)
    faces_arr = np.concatenate(all_faces, axis=0)
    object_ids = np.concatenate(all_obj_ids, axis=0)

    result = write_mesh(
        str(output_path),
        positions,
        faces_arr,
        chunk_shape=chunk_shape,
        bin_shape=bin_shape,
        encoding=encoding,
        object_ids=object_ids,
        dtype=dtype,
        draco_quantization_bits=draco_quantization_bits,
    )
    result["segment_count"] = len(kept_seg_ids)

    if preserve_header:
        _save_header(
            output_path,
            data_type="mesh",
            source_url=source_url,
            info=cv.info if hasattr(cv, "info") else {},
            segment_ids=kept_seg_ids,
            extra_mesh_metadata=_safe_get_mesh_metadata(cv),
        )

    return result


def _get_kwargs(fn) -> set[str]:
    try:
        import inspect
        return set(inspect.signature(fn).parameters)
    except Exception:
        return set()


def _safe_get_mesh_metadata(cv: Any) -> dict[str, Any]:
    """Pull mesh metadata (sharding, transform, LOD) from a CloudVolume."""
    meta: dict[str, Any] = {}
    try:
        info = cv.info.get("mesh") if isinstance(cv.info, dict) else None
        if isinstance(info, str):
            meta["mesh_dir"] = info
        try:
            mm = cv.mesh.meta.info if hasattr(cv.mesh, "meta") else None
            if isinstance(mm, dict):
                for key in ("transform", "vertex_quantization_bits",
                            "lod_scale_multiplier", "num_lods", "@type"):
                    if key in mm:
                        meta[key] = mm[key]
        except Exception:
            pass
    except Exception:
        pass
    return meta


# ===================================================================
# Skeletons
# ===================================================================

def ingest_precomputed_skeleton(
    source: str | Path,
    output_path: str | Path,
    chunk_shape: ChunkShape,
    *,
    segment_ids: Sequence[int] | None = None,
    bin_shape: BinShape | None = None,
    dtype: str = "float32",
    preserve_header: bool = True,
) -> dict[str, Any]:
    """Ingest precomputed segment skeletons into a zarr vectors skeleton store.

    Each precomputed skeleton becomes one ZVF object. Per-vertex
    attributes declared in the source's ``vertex_attributes`` info (e.g.
    ``radius``, ``vertex_types``) are forwarded to ``vertex_attributes``
    on the output store.

    Args:
        source: cloud-volume URL or local precomputed path.
        output_path: Destination ZVF store path.
        chunk_shape: Spatial chunk size per dimension.
        segment_ids: Skeleton IDs to ingest. If None, cloud-volume's
            skeleton listing is queried.
        bin_shape: Optional sub-chunk binning.
        dtype: Numpy dtype for vertex positions.
        preserve_header: Store a :class:`NeuroglancerHeader` for
            round-trip export.

    Returns:
        Summary dict from :func:`write_graph`, with an extra
        ``segment_count`` field.
    """
    cloudvolume = _import_cloudvolume()
    source_url = _normalise_source(source)

    try:
        cv = cloudvolume.CloudVolume(source_url, progress=False, use_https=True)
    except Exception as e:
        raise IngestError(f"Failed to open precomputed source '{source_url}': {e}") from e

    if not hasattr(cv, "skeleton"):
        raise IngestError(f"Source '{source_url}' has no skeleton data")

    if segment_ids is None:
        try:
            seg_ids = list(cv.skeleton.get_all_ids())
        except Exception as e:
            raise IngestError(
                f"Could not list skeleton segment IDs for '{source_url}'. "
                f"Pass segment_ids explicitly. ({e})"
            ) from e
    else:
        seg_ids = [int(s) for s in segment_ids]

    if not seg_ids:
        raise IngestError(f"No skeleton segments found in '{source_url}'")

    np_dtype = np.dtype(dtype)
    all_verts: list[np.ndarray] = []
    all_edges: list[np.ndarray] = []
    all_obj_ids: list[np.ndarray] = []
    per_attr: dict[str, list[np.ndarray]] = {}
    kept_seg_ids: list[int] = []
    vertex_offset = 0

    for slot, sid in enumerate(seg_ids):
        try:
            skel = cv.skeleton.get(sid)
        except Exception as e:
            raise IngestError(f"Failed to fetch skeleton for segment {sid}: {e}") from e

        if skel is None or len(skel.vertices) == 0:
            continue

        verts = np.asarray(skel.vertices, dtype=np_dtype)
        edges = np.asarray(skel.edges, dtype=np.int64)
        if edges.ndim != 2 or edges.shape[1] != 2:
            edges = np.zeros((0, 2), dtype=np.int64)

        all_verts.append(verts)
        all_edges.append(edges + vertex_offset)
        all_obj_ids.append(np.full(len(verts), slot, dtype=np.int64))

        # Forward per-vertex attributes (radius, vertex_types, ...)
        for attr_name in dir(skel):
            if attr_name.startswith("_") or attr_name in (
                "vertices", "edges", "id", "segid", "transform", "space",
                "extra_attributes", "from_path", "from_swc", "to_swc", "encode",
            ):
                continue
            try:
                val = getattr(skel, attr_name)
            except Exception:
                continue
            if callable(val):
                continue
            try:
                arr = np.asarray(val)
            except Exception:
                continue
            if arr.dtype.kind not in ("f", "i", "u", "b") or arr.shape[:1] != (len(verts),):
                continue
            per_attr.setdefault(attr_name, []).append(arr.astype(np.float32))

        kept_seg_ids.append(int(sid))
        vertex_offset += len(verts)

    if not all_verts:
        raise IngestError(f"All skeleton segments in '{source_url}' were empty")

    positions = np.concatenate(all_verts, axis=0)
    edges_arr = np.concatenate(all_edges, axis=0) if all_edges else np.zeros((0, 2), dtype=np.int64)
    object_ids = np.concatenate(all_obj_ids, axis=0)

    # Only keep attributes present on every segment.
    vertex_attributes: dict[str, np.ndarray] = {}
    for name, parts in per_attr.items():
        if len(parts) == len(kept_seg_ids):
            try:
                vertex_attributes[name] = np.concatenate(parts, axis=0)
            except Exception:
                continue

    result = write_graph(
        str(output_path),
        positions,
        edges_arr,
        chunk_shape=chunk_shape,
        bin_shape=bin_shape,
        kind="skeleton",
        vertex_attributes=vertex_attributes or None,
        object_ids=object_ids,
        dtype=dtype,
    )
    result["segment_count"] = len(kept_seg_ids)

    if preserve_header:
        _save_header(
            output_path,
            data_type="skeleton",
            source_url=source_url,
            info=cv.info if hasattr(cv, "info") else {},
            segment_ids=kept_seg_ids,
            extra_skeleton_metadata=_safe_get_skeleton_metadata(cv),
        )

    return result


def _safe_get_skeleton_metadata(cv: Any) -> dict[str, Any]:
    """Pull skeleton metadata (transform, vertex_attributes spec) from CloudVolume."""
    meta: dict[str, Any] = {}
    try:
        sm = cv.skeleton.meta.info if hasattr(cv.skeleton, "meta") else None
        if isinstance(sm, dict):
            for key in ("transform", "vertex_attributes", "@type", "spatial_index"):
                if key in sm:
                    meta[key] = sm[key]
    except Exception:
        pass
    return meta


# ===================================================================
# Annotations (points + line segments)
# ===================================================================

def ingest_precomputed_annotations(
    source: str | Path,
    output_path: str | Path,
    chunk_shape: ChunkShape,
    *,
    bbox: BoundingBox | None = None,
    bin_shape: BinShape | None = None,
    dtype: str = "float32",
    preserve_header: bool = True,
) -> dict[str, Any]:
    """Ingest a precomputed annotation layer into a zarr vectors store.

    Dispatches by ``info["annotation_type"]``:

    * ``"POINT"`` → :func:`write_points`
    * ``"LINE"``  → :func:`write_lines`

    Other annotation types (``AXIS_ALIGNED_BOUNDING_BOX``, ``ELLIPSOID``)
    are not yet supported and raise :class:`IngestError`.

    Args:
        source: cloud-volume URL or local precomputed annotation path.
        output_path: Destination ZVF store path.
        chunk_shape: Spatial chunk size per dimension.
        bbox: Optional spatial bbox restricting which annotations are
            pulled. Forwarded to cloud-volume's annotation query.
        bin_shape: Optional sub-chunk binning.
        dtype: Numpy dtype for positions.
        preserve_header: Store a :class:`NeuroglancerHeader` for
            round-trip export.

    Returns:
        Summary dict from the appropriate writer.
    """
    cloudvolume = _import_cloudvolume()
    source_url = _normalise_source(source)

    try:
        cv = cloudvolume.CloudVolume(source_url, progress=False, use_https=True)
    except Exception as e:
        raise IngestError(f"Failed to open precomputed source '{source_url}': {e}") from e

    info = cv.info if hasattr(cv, "info") else {}
    if not isinstance(info, dict):
        raise IngestError(f"Source '{source_url}' has no info JSON")

    annotation_type = str(info.get("annotation_type", "")).upper()
    if not annotation_type:
        raise IngestError(
            f"Source '{source_url}' is not an annotation layer "
            f"(no 'annotation_type' in info JSON)"
        )

    properties_spec = info.get("properties", []) or []
    relationships_spec = info.get("relationships", []) or []
    np_dtype = np.dtype(dtype)

    # cloud-volume exposes annotation queries via a few different APIs
    # depending on version. Try the modern path first, fall back to
    # the raw spatial-index reader.
    try:
        annotations = _fetch_annotations(cv, bbox=bbox)
    except Exception as e:
        raise IngestError(f"Failed to fetch annotations from '{source_url}': {e}") from e

    if not annotations:
        raise IngestError(f"No annotations returned from '{source_url}'")

    if annotation_type == "POINT":
        positions, vertex_attributes = _annotations_to_points(
            annotations, properties_spec, np_dtype,
        )
        result = write_points(
            str(output_path),
            positions,
            chunk_shape=chunk_shape,
            bin_shape=bin_shape,
            vertex_attributes=vertex_attributes or None,
            dtype=dtype,
        )
    elif annotation_type == "LINE":
        endpoints, object_attributes = _annotations_to_lines(
            annotations, properties_spec, np_dtype,
        )
        result = write_lines(
            str(output_path),
            endpoints,
            chunk_shape=chunk_shape,
            bin_shape=bin_shape,
            object_attributes=object_attributes or None,
            dtype=dtype,
        )
    else:
        raise IngestError(
            f"Annotation type '{annotation_type}' is not supported. "
            f"Supported: POINT, LINE."
        )

    if preserve_header:
        _save_header(
            output_path,
            data_type="annotation",
            annotation_type=annotation_type,
            source_url=source_url,
            info=info,
            annotation_properties=properties_spec,
            relationships=relationships_spec,
        )

    return result


def _fetch_annotations(cv: Any, *, bbox: BoundingBox | None) -> list[Any]:
    """Best-effort annotation fetch across cloud-volume API variants.

    Returns a list of annotation records. Each record has ``point``
    (for POINT) or ``pointA``/``pointB`` (for LINE), plus any property
    fields declared in the info JSON.
    """
    # Preferred: cv.annotation.all() / cv.annotation.get(bbox=...)
    ann = getattr(cv, "annotation", None)
    if ann is not None:
        if bbox is not None and hasattr(ann, "get"):
            try:
                return list(ann.get(bbox))
            except Exception:
                pass
        for method in ("all", "list", "get_all"):
            fn = getattr(ann, method, None)
            if callable(fn):
                try:
                    return list(fn())
                except Exception:
                    continue

    # Fallback: read the by_id index directly.
    raise RuntimeError(
        "cloud-volume's installed version does not expose an annotation "
        "fetch API this module knows how to call. Update cloud-volume or "
        "pre-extract the annotations into a CSV."
    )


def _annotation_field(ann: Any, name: str) -> Any:
    """Pull a field from either a dict-shaped or attribute-shaped record."""
    if isinstance(ann, dict):
        return ann.get(name)
    return getattr(ann, name, None)


def _annotations_to_points(
    annotations: list[Any],
    properties_spec: list[dict[str, Any]],
    np_dtype: np.dtype,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Extract (positions, vertex_attributes) from POINT annotations."""
    n = len(annotations)
    positions = np.zeros((n, 3), dtype=np_dtype)
    for i, ann in enumerate(annotations):
        p = _annotation_field(ann, "point")
        if p is None:
            continue
        arr = np.asarray(p, dtype=np_dtype).flatten()
        positions[i, : min(3, arr.size)] = arr[: min(3, arr.size)]

    vertex_attributes: dict[str, np.ndarray] = {}
    for prop in properties_spec:
        pname = prop.get("id") or prop.get("name")
        if not pname:
            continue
        try:
            vals = [_annotation_field(a, pname) for a in annotations]
            vertex_attributes[pname] = np.asarray(
                [0.0 if v is None else float(v) for v in vals],
                dtype=np.float32,
            )
        except Exception:
            continue
    return positions, vertex_attributes


def _annotations_to_lines(
    annotations: list[Any],
    properties_spec: list[dict[str, Any]],
    np_dtype: np.dtype,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Extract (endpoints, object_attributes) from LINE annotations."""
    n = len(annotations)
    endpoints = np.zeros((n, 2, 3), dtype=np_dtype)
    for i, ann in enumerate(annotations):
        a = _annotation_field(ann, "pointA")
        b = _annotation_field(ann, "pointB")
        if a is None or b is None:
            continue
        a_arr = np.asarray(a, dtype=np_dtype).flatten()
        b_arr = np.asarray(b, dtype=np_dtype).flatten()
        endpoints[i, 0, : min(3, a_arr.size)] = a_arr[: min(3, a_arr.size)]
        endpoints[i, 1, : min(3, b_arr.size)] = b_arr[: min(3, b_arr.size)]

    object_attributes: dict[str, np.ndarray] = {}
    for prop in properties_spec:
        pname = prop.get("id") or prop.get("name")
        if not pname:
            continue
        try:
            vals = [_annotation_field(a, pname) for a in annotations]
            object_attributes[pname] = np.asarray(
                [0.0 if v is None else float(v) for v in vals],
                dtype=np.float32,
            )
        except Exception:
            continue
    return endpoints, object_attributes


# ===================================================================
# Header save helper
# ===================================================================

def _save_header(
    output_path: str | Path,
    *,
    data_type: str,
    source_url: str,
    info: dict[str, Any],
    annotation_type: str = "",
    segment_ids: list[int] | None = None,
    annotation_properties: list[dict[str, Any]] | None = None,
    relationships: list[dict[str, Any]] | None = None,
    extra_mesh_metadata: dict[str, Any] | None = None,
    extra_skeleton_metadata: dict[str, Any] | None = None,
) -> None:
    """Best-effort write of a :class:`NeuroglancerHeader` to the store."""
    try:
        from zarr_vectors_tools.headers.formats import NeuroglancerHeader
        from zarr_vectors_tools.headers.registry import HeaderRegistry

        resolution = _info_resolution(info)
        transform = _flatten_transform(
            (extra_mesh_metadata or {}).get("transform")
            or (extra_skeleton_metadata or {}).get("transform")
            or info.get("transform")
        )

        header = NeuroglancerHeader(
            data_type=data_type,
            annotation_type=annotation_type,
            source_url=source_url,
            resolution=resolution,
            transform=transform,
            mesh_metadata=extra_mesh_metadata or {},
            skeleton_metadata=extra_skeleton_metadata or {},
            annotation_properties=annotation_properties or [],
            relationships=relationships or [],
            segment_ids=[str(s) for s in (segment_ids or [])],
        )
        HeaderRegistry(str(output_path)).add("neuroglancer", header)
    except Exception:
        pass  # header preservation is best-effort
