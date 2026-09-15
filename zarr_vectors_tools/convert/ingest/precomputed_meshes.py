"""Neuroglancer precomputed mesh layer → zarr-vectors mesh store.

A mesh layer holds one mesh per segment, in one of three layouts, told
apart by the ``@type`` (and ``sharding``) of the layer's ``info``:

* ``neuroglancer_legacy_mesh``: a ``<segment id>:0`` JSON manifest naming
  fragment files, each a vertex count, ``float32`` vertices and ``uint32``
  triangles.  One level of detail.
* ``neuroglancer_multilod_draco``: a ``<segment id>.index`` manifest and a
  ``<segment id>`` file of Draco-encoded fragments, several levels of detail.
* the same, ``sharded``: manifests and fragments packed into ``.shard``
  files.

Each segment becomes one object.  Objects are numbered by the rank of their
segment id, and the ids are stored ascending as the ``segment_id`` object
attribute -- the convention the EM skeleton ingests use, so
:func:`~zarr_vectors_tools.convert.export.precomputed.export_precomputed`
and lookups by segment id work on the result.

Meshing pipelines (igneous, zmesh) mesh each chunk of the volume separately,
so a vertex on a chunk face is written once per fragment that touches it.
``weld=True`` merges vertices with identical coordinates within a segment,
then drops the faces that collapse or repeat; without it an object's
fragments do not share vertices across their seams.  Multi-resolution
fragments are quantised to a grid whose ends lie on the fragment's faces,
so their seam vertices coincide exactly as well.

Memory holds every ingested segment's vertices and faces at once (about
20 bytes per vertex and 24 per triangle after welding), since the store is
written in one call.  Select segments with ``segment_ids`` to bound it.

Needs ``cloud-volume`` (and ``DracoPy`` for multi-resolution layers), from
the ``precomputed`` extra.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors_tools.convert.ingest.precomputed import layer_url

LEGACY_MESH_TYPE = "neuroglancer_legacy_mesh"
MULTILOD_MESH_TYPE = "neuroglancer_multilod_draco"
MESH_TYPES = (LEGACY_MESH_TYPE, MULTILOD_MESH_TYPE)
_SEGMENT_ID_ATTR = "segment_id"


def mesh_layout(info: dict[str, Any]) -> str:
    """``"legacy"``, ``"multilod"`` or ``"multilod_sharded"`` for a mesh ``info``."""
    kind = info.get("@type")
    if kind == LEGACY_MESH_TYPE:
        return "legacy"
    if kind == MULTILOD_MESH_TYPE:
        return "multilod_sharded" if info.get("sharding") else "multilod"
    raise ValueError(
        f"@type {kind!r} is not a precomputed mesh layer; expected one of {MESH_TYPES}"
    )


def mesh_source(url: str) -> Any:
    """A cloud-volume mesh source for the mesh directory at ``url``.

    ``PrecomputedMeshSource.from_cloudpath`` does not build its configuration
    with every argument cloud-volume 12 requires, so the source is assembled
    here the way the skeleton source's ``from_cloudpath`` does it.
    """
    from cloudvolume.cacheservice import CacheService
    from cloudvolume.cloudvolume import SharedConfiguration
    from cloudvolume.datasource.precomputed.mesh import PrecomputedMeshSource
    from cloudvolume.datasource.precomputed.metadata import PrecomputedMetadata

    config = SharedConfiguration(
        cdn_cache=False, compress=True, compress_level=None, green=False, mip=0,
        parallel=1, progress=False, secrets=None, spatial_index_db=None,
        cache_locking=False, codec_threads=1,
    )
    cache = CacheService(cloudpath=url, enabled=False, config=config, compress=True)
    root, directory = url.rstrip("/").rsplit("/", 1)
    meta = PrecomputedMetadata(root, config, cache, info={"mesh": directory})
    return PrecomputedMeshSource(meta, cache, config)


def list_mesh_segment_ids(url: str, info: dict[str, Any]) -> list[int]:
    """Every segment id with a mesh in the layer, ascending.

    Unsharded layers are listed by their manifest names; a sharded layer by
    reading the minishard indices of each ``.shard`` file.
    """
    from cloudfiles import CloudFiles

    layout = mesh_layout(info)
    ids: set[int] = set()
    names = [
        name for name in CloudFiles(url).list(flat=True)
        # A flat listing's top-level files; anything with a separator is
        # deeper (segment_properties/info, or another layer's keys).
        if "/" not in name and "\\" not in name
    ]
    if layout == "multilod_sharded":
        source = mesh_source(url)
        for name in sorted(n for n in names if n.endswith(".shard")):
            ids.update(int(label) for label in source.reader.list_labels(name, path=source.path))
        return sorted(ids)

    suffix = ":0" if layout == "legacy" else ".index"
    for name in names:
        stem = name[: -len(suffix)] if name.endswith(suffix) else ""
        if stem.isdigit():
            ids.add(int(stem))
    return sorted(ids)


def weld_vertices(
    vertices: npt.NDArray[np.floating], faces: npt.NDArray[np.integer],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64], int]:
    """Merge coincident vertices; drop the faces that collapse or repeat.

    Returns ``(vertices, faces, faces dropped)``.  Vertices no face uses are
    dropped too, and the survivors keep the order of their first use.
    """
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not len(faces):
        return vertices[:0], faces, 0
    unique, inverse = np.unique(vertices, axis=0, return_inverse=True)
    faces = inverse.reshape(-1)[faces]

    # A triangle is its corners whatever their order, as in the mesh
    # decimator; the first copy, with its winding, is the one kept.
    canonical = np.sort(faces, axis=1)
    keep = np.all(canonical[:, 1:] != canonical[:, :-1], axis=1)
    _, first = np.unique(canonical[keep], axis=0, return_index=True)
    survivors = np.flatnonzero(keep)[np.sort(first)]
    dropped = len(faces) - len(survivors)
    faces = faces[survivors]

    used, order = np.unique(faces.reshape(-1), return_index=True)
    used = used[np.argsort(order)]
    remap = np.full(len(unique), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return unique[used], remap[faces], dropped


def ingest_precomputed_meshes(
    source: str | Path,
    out_store: str | Path,
    chunk_shape: Sequence[float],
    *,
    segment_ids: Sequence[int] | None = None,
    lod: int = 0,
    weld: bool = True,
    batch_size: int = 64,
    read_workers: int = 4,
    segment_properties: bool = True,
    progress: bool = True,
) -> dict[str, Any]:
    """Ingest a precomputed mesh layer, one object per segment.

    Args:
        source: The mesh directory: a URL (``gs://``, ``s3://``, ``https://``,
            ``file://``, ``mem://``, optionally behind ``precomputed://``) or
            a local directory.  Not the segmentation layer above it.
        out_store: Output store path.
        chunk_shape: Level-0 chunk size, in the layer's units (nanometres).
        segment_ids: Only these segments.  Default: every segment in the
            layer.
        lod: Level of detail to read from a multi-resolution layer; 0 is the
            finest.  A legacy layer has only level 0.
        weld: Merge coincident vertices across fragment seams.
        batch_size: Segments per read.
        read_workers: Threads reading batches.
        segment_properties: Copy the layer's inline segment properties, when
            its ``info`` names them, as object attributes.
        progress: Print progress.

    Returns:
        Summary dict with ``layer``, ``layout``, ``lod``, ``object_count``,
        ``empty_segments`` (ids asked for whose mesh has no faces at
        ``lod``), ``vertex_count``, ``face_count``, ``vertices_welded``,
        ``faces_dropped`` and ``properties``, plus :func:`write_mesh`'s own.

    Raises:
        FileNotFoundError: No ``info`` at ``source``.
        ValueError: ``source`` is not a mesh layer, a segment asked for has
            no mesh, ``lod`` is out of range, or nothing is left to write.
    """
    from cloudfiles import CloudFiles
    from zarr_vectors.building import get_resolution_level, open_store
    from zarr_vectors.types.meshes import write_mesh

    url = layer_url(source)
    info = CloudFiles(url).get_json("info")
    if info is None:
        raise FileNotFoundError(f"no precomputed info file at {url}/info")
    layout = mesh_layout(info)
    if lod < 0:
        raise ValueError(f"lod must be 0 or more, not {lod}")
    if layout == "legacy" and lod:
        raise ValueError(f"{url} is a legacy mesh layer, which has only lod 0")
    if batch_size < 1 or read_workers < 1:
        raise ValueError("batch_size and read_workers must be at least 1")

    if segment_ids is None:
        ids = list_mesh_segment_ids(url, info)
        if not ids:
            raise ValueError(f"{url} holds no meshes")
    else:
        ids = sorted({int(s) for s in segment_ids})
        if not ids:
            raise ValueError("segment_ids is empty")
        missing = _missing_segments(url, layout, ids)
        if missing:
            shown = ", ".join(str(s) for s in missing[:10])
            more = f" and {len(missing) - 10} more" if len(missing) > 10 else ""
            raise ValueError(f"{url} has no mesh for segment id(s) {shown}{more}")
    _log(progress, f"reading {len(ids)} segment mesh(es) from {url} ({layout}, lod {lod})")

    written: list[int] = []
    empty: list[int] = []
    vertex_parts: list[npt.NDArray[np.float32]] = []
    face_parts: list[npt.NDArray[np.int64]] = []
    counts: list[int] = []
    raw_vertices = dropped = offset = 0
    for batch in _read_batches(url, layout, ids, lod, batch_size, read_workers):
        for segment, vertices, faces in batch:
            raw_vertices += len(vertices)
            if weld:
                vertices, faces, n_dropped = weld_vertices(vertices, faces)
                dropped += n_dropped
            if not len(faces):
                empty.append(segment)
                continue
            face_parts.append(faces + offset)
            vertex_parts.append(vertices)
            counts.append(len(vertices))
            offset += len(vertices)
            written.append(segment)
        _log(progress, f"  {len(written) + len(empty)}/{len(ids)} segments read")

    if not written:
        raise ValueError(
            f"none of the {len(ids)} segment mesh(es) read from {url} has faces at lod {lod}"
        )
    vertices = np.concatenate(vertex_parts)
    faces = np.concatenate(face_parts)
    object_ids = np.repeat(np.arange(len(written), dtype=np.int64), counts)
    del vertex_parts, face_parts

    summary = dict(write_mesh(
        str(out_store), vertices, faces,
        chunk_shape=tuple(float(c) for c in chunk_shape),
        object_ids=object_ids,
        object_attributes={_SEGMENT_ID_ATTR: np.asarray(written, dtype=np.uint64)},
    ))

    properties: list[str] = []
    if segment_properties and info.get("segment_properties"):
        inline = _inline_properties(url, info["segment_properties"])
        if inline:
            properties = _write_properties(
                get_resolution_level(open_store(str(out_store), mode="r+"), 0),
                inline, written, progress,
            )

    summary.update({
        "layer": url,
        "layout": layout,
        "lod": lod,
        "object_count": len(written),
        "empty_segments": empty,
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "vertices_welded": int(raw_vertices - len(vertices)) if weld else 0,
        "faces_dropped": int(dropped),
        "properties": properties,
    })
    _log(progress, f"wrote {len(written)} mesh object(s), {len(vertices)} vertices, "
                   f"{len(faces)} triangles")
    return summary


# ===================================================================
# Reading
# ===================================================================

def _read_batches(
    url: str, layout: str, ids: list[int], lod: int, batch_size: int, workers: int,
) -> Iterator[list[tuple[int, npt.NDArray[np.float32], npt.NDArray[np.int64]]]]:
    """``[(segment id, vertices, faces), ...]`` per batch, in ``ids`` order.

    Each thread keeps its own mesh source: cloud-volume's sources hold
    connection state that is not shared safely between threads.
    """
    local = threading.local()

    def read(batch: list[int]) -> list[tuple[int, Any, Any]]:
        if not hasattr(local, "source"):
            local.source = mesh_source(url)
        return _read_segments(local.source, url, layout, batch, lod)

    batches = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
    if workers == 1 or len(batches) == 1:
        for batch in batches:
            yield read(batch)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # map yields in submission order, holding at most ``workers`` batches
        # ahead of the one being consumed.
        yield from pool.map(read, batches)


def _read_segments(
    source: Any, url: str, layout: str, batch: list[int], lod: int,
) -> list[tuple[int, npt.NDArray[np.float32], npt.NDArray[np.int64]]]:
    from cloudvolume.exceptions import MeshDecodeError

    try:
        if layout == "legacy":
            meshes = source.get(batch, fuse=False, remove_duplicate_vertices=False)
        else:
            meshes = source.get(batch, lod=lod, concat=True)
    except MeshDecodeError as e:
        raise ValueError(f"cannot read lod {lod} from {url}: {e}") from e

    out = []
    for segment in batch:
        mesh = meshes.get(segment)
        if mesh is None:
            out.append((segment, np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64)))
            continue
        out.append((
            segment,
            np.asarray(mesh.vertices, dtype=np.float32).reshape(-1, 3),
            np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3),
        ))
    return out


def _missing_segments(url: str, layout: str, ids: list[int]) -> list[int]:
    if layout == "multilod_sharded":
        source = mesh_source(url)
        return [s for s in ids if source.reader.exists(s, source.path) is None]
    from cloudfiles import CloudFiles

    suffix = ":0" if layout == "legacy" else ".index"
    found = CloudFiles(url).exists([f"{s}{suffix}" for s in ids])
    return [s for s in ids if not found.get(f"{s}{suffix}")]


# ===================================================================
# Segment properties
# ===================================================================

def _inline_properties(url: str, directory: str) -> dict[str, Any] | None:
    from cloudfiles import CloudFiles

    props = CloudFiles(url).get_json(f"{str(directory).strip('/')}/info")
    return (props or {}).get("inline")


def _write_properties(
    level_group: Any, inline: dict[str, Any], written: list[int], progress: bool,
) -> list[str]:
    """Write the inline properties the ingested segments have; return their names.

    A property called ``segment_id`` is skipped: that attribute is the
    object numbering itself.
    """
    from zarr_vectors_tools.convert.ingest.precomputed_plain_skeletons import (
        _write_segment_properties,
    )

    kept = dict(inline)
    kept["properties"] = [
        p for p in inline.get("properties", []) if p.get("id") != _SEGMENT_ID_ATTR
    ]
    if not kept["properties"]:
        return []
    _write_segment_properties(
        level_group, seg_props_raw=kept,
        oid_of_seg={segment: oid for oid, segment in enumerate(written)},
        n_objects=len(written), progress=progress,
    )
    from zarr_vectors.constants import OBJECT_ATTRIBUTES

    try:
        present = set(level_group[OBJECT_ATTRIBUTES].children())
    except Exception:  # noqa: BLE001 - no attributes were written
        present = set()
    return [p["id"] for p in kept["properties"] if p["id"] in present]


def _log(progress: bool, message: str) -> None:
    if progress:
        print(f"[precomputed-meshes] {message}", flush=True)
