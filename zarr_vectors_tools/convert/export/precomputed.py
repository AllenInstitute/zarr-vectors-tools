"""Export skeleton and mesh stores as Neuroglancer precomputed layers.

The way back into Neuroglancer, CloudVolume and igneous.  An export writes
one kind of object into a *segmentation layer*: a directory or bucket prefix
laid out the way CloudVolume opens one::

    <layer>/info                                segmentation volume info
    <layer>/skeletons/info                      "@type": "neuroglancer_skeletons"
    <layer>/skeletons/<segment id>              one skeleton per segment
    <layer>/skeletons/segment_properties/info   ids, labels and numbers
    <layer>/mesh/info                           "@type": "neuroglancer_legacy_mesh"
    <layer>/mesh/<segment id>:0                 manifest naming the fragment
    <layer>/mesh/<segment id>.frag              the segment's triangles
    <layer>/mesh/segment_properties/info

What the format forces, and what this module does about it:

* **The layer's own info.**  ``CloudVolume(url)`` refuses a layer whose
  ``info`` has no ``scales``, and a bare ``neuroglancer_skeletons`` directory
  has none.  So a new layer gets a segmentation ``info`` with one placeholder
  scale -- one voxel per level-0 chunk over the store's bounds, no image
  chunks written, keyed ``placeholder``, which is CloudVolume's own marker
  for a scale without image data -- that names the ``skeletons`` or ``mesh``
  subdirectory.
  Exporting into an existing segmentation layer keeps its scales and writes
  into the subdirectory it already names, provided that subdirectory is a
  format this module writes.  Neuroglancer can also open either
  subdirectory on its own, as ``precomputed://<layer>/skeletons``.
* **Coordinates** are nanometres, the unit both formats are read in, under an
  identity ``transform``.  A store whose axes record a length unit is scaled
  from it; a cortical surface store is in millimetres; any other store that
  records no unit is taken to be in nanometres unless ``unit=`` says
  otherwise.  The store's coordinate offset is added back.
* **Segment ids.**  A store written by the precomputed ingesters keeps a
  ``segment_id`` per object, and that is the id.  Any other store numbers its
  objects from 0, and Neuroglancer treats segment 0 as background -- it
  never asks for its skeleton or mesh -- so those are exported as
  ``object id + 1``.  A stored segment id of 0 is refused for the same reason.
* **Meshes are single-resolution** (``neuroglancer_legacy_mesh``), one
  fragment per segment.  The multi-resolution Draco format and sharding are
  not written; export a coarser level to a second layer instead.  The legacy
  manifest is named ``<id>:0``, which Windows cannot hold as a file name, so
  on Windows a mesh layer goes to a bucket (``gs://``, ``s3://``), not to a
  local directory.

``output`` may be a URL (``gs://``, ``s3://``, ``file://``, optionally
behind Neuroglancer's ``precomputed://``) or a local directory.  Writing
needs ``cloud-files``, the ``precomputed`` extra.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from zarr_vectors.exceptions import ExportError

from zarr_vectors_tools.convert.ingest.precomputed import layer_url

_SKELETON_TYPE = "neuroglancer_skeletons"
_MESH_TYPE = "neuroglancer_legacy_mesh"
_PROPERTIES_TYPE = "neuroglancer_segment_properties"
_SKELETON_KEY = "skeletons"
_MESH_KEY = "mesh"
_PROPERTIES_DIR = "segment_properties"
#: The per-object attribute the precomputed ingesters keep segment ids in.
_SEGMENT_ID_ATTR = "segment_id"
_IDENTITY = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
#: Files handed to CloudFiles at once; bounds how many encoded segments are
#: held in memory while the export streams.
_UPLOAD_BATCH = 256
#: Largest placeholder chunk edge, in placeholder voxels.
_PLACEHOLDER_CHUNK = 1024

_SKELETON_GEOMETRIES = ("skeleton", "graph")
_MESH_GEOMETRY = "mesh"

#: Length units a store's axes may name (UDUNITS-2 spellings and the usual
#: abbreviations), as nanometres per unit.
_NM_PER_UNIT = {
    "angstrom": 0.1,
    "nanometer": 1.0, "nanometre": 1.0, "nm": 1.0,
    "micrometer": 1e3, "micrometre": 1e3, "micron": 1e3, "um": 1e3,
    "millimeter": 1e6, "millimetre": 1e6, "mm": 1e6,
    "centimeter": 1e7, "centimetre": 1e7, "cm": 1e7,
    "meter": 1e9, "metre": 1e9, "m": 1e9,
}

#: Integer types Neuroglancer takes for a skeleton vertex attribute or a
#: numeric segment property.  Anything wider has no precomputed spelling.
_PRECOMPUTED_INTS = ("uint8", "int8", "uint16", "int16", "uint32", "int32")


# ===================================================================
# Public API
# ===================================================================

def export_precomputed(
    store_path: str | Path,
    output: str | Path,
    *,
    level: int = 0,
    object_ids: Sequence[int] | None = None,
    segment_ids: Sequence[int] | None = None,
    attribute_names: Sequence[str] | None = None,
    object_attribute_names: Sequence[str] | None = None,
    unit: str | None = None,
    compress: str | None = None,
) -> dict[str, Any]:
    """Export a store as a Neuroglancer precomputed layer, by its geometry.

    Skeleton and graph stores become skeletons
    (:func:`export_precomputed_skeletons`); mesh stores become legacy
    meshes (:func:`export_precomputed_meshes`).  Every other geometry is
    refused.

    Args:
        store_path: Path to the zarr vectors store.
        output: Layer URL or local directory.
        level: Resolution level to export.
        object_ids: Objects to export, by the store's object id.
        segment_ids: Objects to export, by segment id.  Not with
            ``object_ids``.
        attribute_names: Per-vertex attributes to carry (skeletons only).
        object_attribute_names: Per-object attributes to write as segment
            properties.  Default: every one that has a precomputed spelling.
        unit: The store's coordinate unit, when its metadata does not say.
        compress: CloudFiles compression for the object files (``"gzip"``,
            ``"br"``...).  Default none, which any static file server can
            serve to Neuroglancer.

    Returns:
        The summary of the exporter that ran.
    """
    _root, root_meta, _level_group = _open_level(store_path, level)
    geometries = list(root_meta.geometry_types or [])
    common = dict(
        level=level, object_ids=object_ids, segment_ids=segment_ids,
        object_attribute_names=object_attribute_names, unit=unit, compress=compress,
    )
    if any(g in _SKELETON_GEOMETRIES for g in geometries):
        return export_precomputed_skeletons(
            store_path, output, attribute_names=attribute_names, **common,
        )
    if _MESH_GEOMETRY in geometries:
        if attribute_names:
            raise ExportError(
                "attribute_names applies to skeletons; a legacy precomputed mesh "
                "has no per-vertex attributes"
            )
        return export_precomputed_meshes(store_path, output, **common)
    raise ExportError(
        f"{store_path} holds {geometries or 'no geometry'}; precomputed export "
        f"writes skeleton, graph and mesh stores"
    )


def export_precomputed_skeletons(
    store_path: str | Path,
    output: str | Path,
    *,
    level: int = 0,
    object_ids: Sequence[int] | None = None,
    segment_ids: Sequence[int] | None = None,
    attribute_names: Sequence[str] | None = None,
    object_attribute_names: Sequence[str] | None = None,
    unit: str | None = None,
    compress: str | None = None,
) -> dict[str, Any]:
    """Write a skeleton (or graph) store's objects as precomputed skeletons.

    Each object becomes one ``neuroglancer_skeletons`` file: its vertices in
    nanometres, its edges, and one block per carried vertex attribute, in
    the order the layer's ``info`` lists them.

    An EM skeleton store (one the precomputed ingesters wrote) is read one
    segment at a time, touching only the chunks each segment lives in, and
    the boundary-crossing edges between its fragments are added back from
    the cross-chunk links -- without them a neuron that crosses a chunk
    comes out in pieces.  Any other store is read once and split by its
    object manifests.

    Args:
        store_path: Path to the zarr vectors store.
        output: Layer URL or local directory.
        level: Resolution level to export.
        object_ids: Objects to export, by the store's object id.  Default:
            every object with geometry at ``level``.
        segment_ids: Objects to export, by segment id (see the module
            docstring for how ids are assigned).  Not with ``object_ids``.
        attribute_names: Per-vertex attributes to carry.  Default: every one
            with a precomputed type (float32, or an integer of at most 32
            bits).  A named attribute without one is refused.
        object_attribute_names: Per-object attributes to write as segment
            properties.  Default: every scalar or string one.
        unit: The store's coordinate unit, when its metadata does not say.
        compress: CloudFiles compression for the skeleton files.

    Returns:
        Summary dict with ``layer`` (the URL written), ``directory``,
        ``object_count``, ``segment_ids`` (in the order written),
        ``node_count``, ``edge_count``, ``attributes_carried``,
        ``attributes_skipped``, ``properties``, ``unit`` and ``nm_per_unit``,
        and ``unit_assumed`` when the store did not say.

    Raises:
        ExportError: For a store that is not a skeleton or graph, an object
            that is not at ``level``, a segment id of 0, or a layer that
            already holds skeletons in a form this module does not write.
    """
    root, root_meta, level_group = _open_level(store_path, level)
    geometries = list(root_meta.geometry_types or [])
    if not any(g in _SKELETON_GEOMETRIES for g in geometries):
        raise ExportError(
            f"{store_path} holds {geometries or 'no geometry'}, not skeletons; "
            f"precomputed skeletons are written from skeleton and graph stores"
        )
    unit_name, nm_per_unit, assumed = _unit(store_path, root_meta, unit)
    plan, skipped = _attribute_plan(level_group, attribute_names)

    stored = _stored_segment_ids(level_group)
    em = stored is not None and _is_em_skeleton_store(root_meta)
    explicit = object_ids is not None or segment_ids is not None
    if em:
        pairs = _select(level, len(stored), stored, object_ids, segment_ids)
    else:
        from zarr_vectors.building import read_all_object_manifests

        manifests = read_all_object_manifests(level_group)
        pairs = _select(
            level, len(manifests), stored, object_ids, segment_ids,
            nonempty=[i for i, m in enumerate(manifests) if m],
        )
        # Read before the layer's info is settled: an attribute's width is
        # only certain once its rows are in hand.
        graph, plan, unaligned = _load_graph_level(
            store_path, level_group, level, plan, attribute_names is not None,
        )
        skipped += unaligned
    vertex_attributes = [
        {"id": name, "data_type": dtype.name, "num_components": ncols}
        for name, dtype, ncols in plan
    ]

    layer = _Layer(output, compress)
    layer_info, directory, existing = layer.prepare(
        _SKELETON_KEY, _bounds_nm(root_meta, root, nm_per_unit),
        np.asarray(root_meta.chunk_shape, dtype=np.float64) * nm_per_unit,
    )
    _check_existing_skeletons(layer.url, directory, existing, vertex_attributes)
    # Built before anything is written, so an unusable attribute is refused
    # while the layer is still untouched.
    properties = _segment_properties(store_path, root, em, pairs, object_attribute_names)

    if em:
        pieces = _em_skeletons(store_path, level_group, level, pairs, plan, explicit)
    else:
        # A graph store's positions come back in the stored frame; an EM
        # store's reader adds the offset itself.
        pieces = _manifest_skeletons(
            graph, level_group, level, pairs, plan, explicit, manifests, _coordinate_offset(root),
        )

    written: list[tuple[int, int]] = []
    node_count = edge_count = 0
    for oid, segment, positions, edges, columns in pieces:
        positions_nm = np.asarray(positions, dtype=np.float64) * nm_per_unit
        layer.put(
            f"{directory}/{segment}",
            _encode_skeleton(positions_nm, edges, columns, plan, segment),
        )
        written.append((oid, segment))
        node_count += len(positions)
        edge_count += len(edges)
    layer.flush()

    properties = _only_segments(properties, written)
    skeleton_info = dict(existing or {})
    skeleton_info.update({
        "@type": _SKELETON_TYPE,
        "transform": _IDENTITY,
        "vertex_attributes": vertex_attributes,
    })
    names = layer.finish(layer_info, _SKELETON_KEY, directory, skeleton_info, properties)

    summary: dict[str, Any] = {
        "layer": layer.url,
        "directory": directory,
        "object_count": len(written),
        "segment_ids": [segment for _oid, segment in written],
        "node_count": node_count,
        "edge_count": edge_count,
        "attributes_carried": [name for name, _dtype, _ncols in plan],
        "attributes_skipped": skipped,
        "properties": names,
        "unit": unit_name,
        "nm_per_unit": nm_per_unit,
    }
    if assumed:
        summary["unit_assumed"] = True
    return summary


def export_precomputed_meshes(
    store_path: str | Path,
    output: str | Path,
    *,
    level: int = 0,
    object_ids: Sequence[int] | None = None,
    segment_ids: Sequence[int] | None = None,
    object_attribute_names: Sequence[str] | None = None,
    unit: str | None = None,
    compress: str | None = None,
) -> dict[str, Any]:
    """Write a mesh store's objects as Neuroglancer legacy meshes.

    Each object becomes one fragment file -- ``uint32`` vertex count, the
    ``float32`` vertices in nanometres, the ``uint32`` triangle corners --
    and a ``<segment id>:0`` manifest naming it.  Quads and larger polygons
    are fan-triangulated.  The level is read once and split by the object
    manifests.

    Args:
        store_path: Path to the zarr vectors mesh store.
        output: Layer URL or local directory (not a local directory on
            Windows; see the module docstring).
        level: Resolution level to export.
        object_ids: Objects to export, by the store's object id.  Default:
            every object with geometry at ``level``.
        segment_ids: Objects to export, by segment id.  Not with
            ``object_ids``.
        object_attribute_names: Per-object attributes to write as segment
            properties.  Default: every scalar or string one.
        unit: The store's coordinate unit, when its metadata does not say.
        compress: CloudFiles compression for the fragment files.

    Returns:
        Summary dict with ``layer``, ``directory``, ``object_count``,
        ``segment_ids``, ``vertex_count``, ``face_count`` (triangles
        written), ``properties``, ``unit`` and ``nm_per_unit``, and
        ``unit_assumed`` when the store did not say.

    Raises:
        ExportError: For a store that is not a mesh, a Draco-encoded store,
            an object that is not at ``level``, a local output on Windows, or
            a layer that already holds meshes in another format.
    """
    root, root_meta, level_group = _open_level(store_path, level)
    geometries = list(root_meta.geometry_types or [])
    if _MESH_GEOMETRY not in geometries:
        raise ExportError(
            f"{store_path} holds {geometries or 'no geometry'}, not a mesh; "
            f"precomputed meshes are written from mesh stores"
        )
    url = layer_url(output)
    if url.startswith("file://") and sys.platform == "win32":
        raise ExportError(
            f"cannot write a legacy mesh layer to {url}: each segment's manifest "
            f"is named '<segment id>:0', and Windows does not allow ':' in a file "
            f"name.  Write to a bucket (gs://, s3://) or from a POSIX filesystem."
        )
    _refuse_draco(level_group)
    unit_name, nm_per_unit, assumed = _unit(store_path, root_meta, unit)

    from zarr_vectors.building import read_all_object_manifests

    manifests = read_all_object_manifests(level_group)
    stored = _stored_segment_ids(level_group)
    pairs = _select(
        level, len(manifests), stored, object_ids, segment_ids,
        nonempty=[i for i, m in enumerate(manifests) if m],
    )
    explicit = object_ids is not None or segment_ids is not None

    layer = _Layer(url, compress)
    layer_info, directory, existing = layer.prepare(
        _MESH_KEY, _bounds_nm(root_meta, root, nm_per_unit),
        np.asarray(root_meta.chunk_shape, dtype=np.float64) * nm_per_unit,
    )
    _check_existing_meshes(layer.url, directory, existing)
    properties = _segment_properties(store_path, root, False, pairs, object_attribute_names)

    written: list[tuple[int, int]] = []
    vertex_count = face_count = 0
    offset = _coordinate_offset(root)
    for oid, segment, vertices, faces in _manifest_meshes(
        store_path, level_group, level, pairs, explicit, manifests, offset,
    ):
        vertices_nm = np.asarray(vertices, dtype=np.float64) * nm_per_unit
        triangles = _triangulate(faces)
        fragment = f"{segment}.frag"
        layer.put(f"{directory}/{fragment}", _encode_mesh(vertices_nm, triangles, segment))
        layer.put_json(f"{directory}/{segment}:0", {"fragments": [fragment]})
        written.append((oid, segment))
        vertex_count += len(vertices)
        face_count += len(triangles)
    layer.flush()

    properties = _only_segments(properties, written)
    mesh_info = dict(existing or {})
    mesh_info["@type"] = _MESH_TYPE
    names = layer.finish(layer_info, _MESH_KEY, directory, mesh_info, properties)

    summary: dict[str, Any] = {
        "layer": layer.url,
        "directory": directory,
        "object_count": len(written),
        "segment_ids": [segment for _oid, segment in written],
        "vertex_count": vertex_count,
        "face_count": face_count,
        "properties": names,
        "unit": unit_name,
        "nm_per_unit": nm_per_unit,
    }
    if assumed:
        summary["unit_assumed"] = True
    return summary


# ===================================================================
# The store
# ===================================================================

def _open_level(store_path: str | Path, level: int) -> tuple[Any, Any, Any]:
    from zarr_vectors.building import get_resolution_level, open_store, read_root_metadata

    try:
        root = open_store(str(store_path))
        root_meta = read_root_metadata(root)
    except Exception as e:
        raise ExportError(f"Failed to read store: {e}") from e
    if root_meta.sid_ndim != 3:
        raise ExportError(
            f"{store_path} is {root_meta.sid_ndim}-dimensional; precomputed "
            f"skeletons and meshes are 3-D"
        )
    try:
        level_group = get_resolution_level(root, level)
    except Exception as e:
        raise ExportError(f"{store_path} has no level {level}: {e}") from e
    return root, root_meta, level_group


def _is_em_skeleton_store(root_meta: Any) -> bool:
    """Is this a store core's pull-by-segment-id reader can read?"""
    from zarr_vectors.constants import LINKS_IMPLICIT_BRANCHES

    return root_meta.links_convention == LINKS_IMPLICIT_BRANCHES


def _stored_segment_ids(level_group: Any) -> npt.NDArray[np.uint64] | None:
    from zarr_vectors.building import read_object_attributes

    try:
        values = read_object_attributes(level_group, _SEGMENT_ID_ATTR)
    except Exception:  # noqa: BLE001 - a store with no segment ids
        return None
    return np.asarray(values, dtype=np.uint64).reshape(-1)


def _coordinate_offset(root: Any) -> npt.NDArray[np.float64]:
    from zarr_vectors.building import get_coordinate_offset

    return np.asarray(get_coordinate_offset(root, 3), dtype=np.float64)


def _unit(
    store_path: str | Path, root_meta: Any, unit: str | None,
) -> tuple[str, float, bool]:
    """``(unit, nanometres per unit, assumed)`` for the store's coordinates."""
    assumed = False
    if unit is None:
        named = {
            axis.get("unit") for axis in root_meta.spatial_index_dims
            if axis.get("type", "space") == "space" and axis.get("unit")
        }
        if len(named) > 1:
            raise ExportError(
                f"{store_path} has axes in {sorted(named)}; pass unit= to say "
                f"which one the coordinates are in"
            )
        if named:
            unit = named.pop()
        elif _has_header(store_path, "surface"):
            # GIFTI and FreeSurfer geometry is millimetres by definition,
            # and the surface ingest does not stamp it on the axes.
            unit = "millimeter"
        else:
            unit, assumed = "nanometer", True
    factor = _NM_PER_UNIT.get(str(unit).lower())
    if factor is None:
        raise ExportError(
            f"unit {unit!r} is not a length unit precomputed export knows; "
            f"use one of {sorted(_NM_PER_UNIT)}"
        )
    return str(unit), factor, assumed


def _has_header(store_path: str | Path, name: str) -> bool:
    try:
        from zarr_vectors_tools.headers.registry import HeaderRegistry

        return bool(HeaderRegistry(str(store_path)).has(name))
    except Exception:  # noqa: BLE001 - no headers is an answer, not an error
        return False


def _bounds_nm(
    root_meta: Any, root: Any, nm_per_unit: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    offset = _coordinate_offset(root)
    lo = (np.asarray(root_meta.bounds[0], dtype=np.float64) + offset) * nm_per_unit
    hi = (np.asarray(root_meta.bounds[1], dtype=np.float64) + offset) * nm_per_unit
    return lo, hi


def _refuse_draco(level_group: Any) -> None:
    try:
        encoding = (level_group.read_array_meta("vertices") or {}).get("encoding")
    except Exception:  # noqa: BLE001 - no metadata means the raw default
        encoding = None
    if encoding == "draco":
        raise ExportError(
            "this mesh store is Draco-encoded, which keeps every object in a "
            "chunk in one bitstream, so its objects cannot be told apart; "
            "re-ingest with encoding='raw' to export it"
        )


def _select(
    level: int,
    n_objects: int,
    stored: npt.NDArray[np.uint64] | None,
    object_ids: Sequence[int] | None,
    segment_ids: Sequence[int] | None,
    nonempty: Sequence[int] | None = None,
) -> list[tuple[int, int]]:
    """``(object id, segment id)`` for each object to export, in order asked.

    ``nonempty`` lists the objects with geometry when the caller has read
    the manifests; without it every object is a candidate and the reader
    skips the empty ones.
    """
    if object_ids is not None and segment_ids is not None:
        raise ExportError("pass object_ids or segment_ids, not both")
    ids = (
        stored if stored is not None
        else np.arange(1, n_objects + 1, dtype=np.uint64)
    )

    if segment_ids is not None:
        wanted = list(dict.fromkeys(int(s) for s in segment_ids))
        order = np.argsort(ids, kind="stable")
        ranked = ids[order]
        found = np.searchsorted(ranked, np.asarray(wanted, dtype=np.uint64))
        oids: list[int] = []
        missing = []
        for segment, pos in zip(wanted, found.tolist()):
            if pos < len(ranked) and int(ranked[pos]) == segment:
                oids.append(int(order[pos]))
            else:
                missing.append(segment)
        if missing:
            raise ExportError(
                f"segment id(s) {missing} are not at level {level}"
                + ("" if stored is not None else
                   "; this store has no segment ids, so its objects are "
                   "numbered object id + 1")
            )
    elif object_ids is not None:
        oids = list(dict.fromkeys(int(o) for o in object_ids))
        missing = [o for o in oids if not 0 <= o < n_objects]
        if missing:
            raise ExportError(
                f"object id(s) {missing} are not at level {level}, which holds "
                f"{n_objects} objects"
            )
    else:
        oids = list(nonempty) if nonempty is not None else list(range(n_objects))

    if segment_ids is not None or object_ids is not None:
        if nonempty is not None:
            present = set(nonempty)
            empty = [o for o in oids if o not in present]
            if empty:
                raise ExportError(
                    f"object id(s) {empty} have no geometry at level {level}; "
                    f"a coarser level's sparsity may have dropped them"
                )

    pairs = [(oid, int(ids[oid])) for oid in oids]
    zero = [oid for oid, segment in pairs if segment == 0]
    if zero:
        raise ExportError(
            f"object id(s) {zero} have segment id 0, which Neuroglancer treats "
            f"as background and never draws"
        )
    if stored is not None:
        segments = [segment for _oid, segment in pairs]
        if len(set(segments)) != len(segments):
            raise ExportError(
                "the store's segment_id attribute repeats ids among the objects "
                "asked for; a precomputed layer has one file per segment"
            )
    return pairs


# ===================================================================
# Vertex attributes
# ===================================================================

def _vertex_attribute_names(level_group: Any) -> list[str]:
    from zarr_vectors.constants import VERTEX_ATTRIBUTES

    try:
        return sorted(level_group[VERTEX_ATTRIBUTES].children())
    except Exception:  # noqa: BLE001 - a level with no vertex attributes
        return []


def _precomputed_vertex_dtype(dtype: np.dtype) -> np.dtype | None:
    """The type a stored attribute is written as, or ``None`` if it has none."""
    if dtype.kind == "b":
        return np.dtype("uint8")
    if dtype.kind == "f":
        return np.dtype("float32")
    if dtype.kind in "iu" and dtype.name in _PRECOMPUTED_INTS:
        return np.dtype(dtype.name)
    return None


def _attribute_plan(
    level_group: Any, names: Sequence[str] | None,
) -> tuple[list[tuple[str, np.dtype, int]], list[str]]:
    """``[(name, precomputed dtype, components)]`` to carry, and the skipped."""
    from zarr_vectors.building import attribute_layout

    available = _vertex_attribute_names(level_group)
    explicit = names is not None
    wanted = list(names) if explicit else available
    missing = [name for name in wanted if name not in available]
    if missing:
        raise ExportError(
            f"vertex attribute(s) {missing} are not at this level; it has "
            f"{available or 'none'}"
        )
    plan: list[tuple[str, np.dtype, int]] = []
    skipped: list[str] = []
    for name in wanted:
        dtype, ncols = attribute_layout(level_group, name)
        target = _precomputed_vertex_dtype(np.dtype(dtype))
        if target is None:
            if explicit:
                raise ExportError(
                    f"vertex attribute {name!r} is {np.dtype(dtype)}; a precomputed "
                    f"skeleton attribute is float32 or an integer of at most 32 bits"
                )
            skipped.append(name)
            continue
        plan.append((name, target, int(ncols)))
    return plan, skipped


def _level_order_column(
    level_group: Any, name: str, n_nodes: int,
) -> npt.NDArray[Any] | None:
    """One vertex attribute in level order, ``(n_nodes, components)``.

    The whole-level readers return vertices chunk by chunk in
    ``list_chunk_keys`` order with each chunk's fragments concatenated, and
    attribute cells are stored the same way, so walking the same chunk list
    lines the rows up.  Core writes a ``(V, C)`` graph attribute as ``C * V``
    one-column rows, row-major, and declares one column, so a row count that
    is a multiple of the vertex count is read as that many components.
    ``None`` when the rows do not line up with the vertices at all.
    """
    from zarr_vectors.building import (
        attribute_layout,
        chunk_local_to_global_offsets,
        read_chunk_attributes,
    )

    dtype, stored_ncols = attribute_layout(level_group, name)
    _offsets, chunk_keys, _total = chunk_local_to_global_offsets(level_group)
    parts = [
        np.asarray(group).reshape(len(group), -1)
        for cc in chunk_keys
        for group in read_chunk_attributes(
            level_group, name, cc, dtype=dtype, ncols=stored_ncols,
        )
    ]
    if not parts or n_nodes == 0:
        return None
    values = np.concatenate(parts, axis=0)
    if values.size % n_nodes:
        return None
    return values.reshape(n_nodes, values.size // n_nodes)


def _load_graph_level(
    store_path: str | Path,
    level_group: Any,
    level: int,
    plan: list[tuple[str, np.dtype, int]],
    explicit: bool,
) -> tuple[dict[str, Any], list[tuple[str, np.dtype, int]], list[str]]:
    """A whole level of a graph store, with its attributes in level order.

    Returns the level (``positions``, ``edges``, ``columns``), the attribute
    plan with each width as read, and the attributes that did not line up
    with the vertices -- refused instead when they were asked for by name.
    """
    from zarr_vectors.types.graphs import read_graph

    try:
        result = read_graph(str(store_path), level=level)
    except Exception as e:
        raise ExportError(f"Failed to read store: {e}") from e
    positions = np.asarray(result["positions"], dtype=np.float64)
    n_nodes = len(positions)
    columns: dict[str, npt.NDArray[Any]] = {}
    kept: list[tuple[str, np.dtype, int]] = []
    unaligned: list[str] = []
    for name, dtype, _ncols in plan:
        values = _level_order_column(level_group, name, n_nodes)
        if values is None:
            if explicit:
                raise ExportError(
                    f"vertex attribute {name!r} does not have a value for each of "
                    f"the {n_nodes} vertices at level {level}"
                )
            unaligned.append(name)
            continue
        columns[name] = values
        kept.append((name, dtype, int(values.shape[1])))
    level_data = {
        "positions": positions,
        "edges": np.asarray(result["edges"], dtype=np.int64).reshape(-1, 2),
        "columns": columns,
    }
    return level_data, kept, unaligned


# ===================================================================
# Splitting a level into objects
# ===================================================================

def _fragment_rows(fragment_index: Any, fragment: int) -> npt.NDArray[np.int64]:
    """The chunk-buffer rows a fragment occupies, range or explicit."""
    if fragment_index.is_range(fragment):
        start, count = fragment_index.range(fragment)
        return np.arange(int(start), int(start) + int(count), dtype=np.int64)
    return np.asarray(fragment_index.indices(fragment), dtype=np.int64)


def _vertex_owner(
    level_group: Any, manifests: Sequence[Any], oids: Sequence[int], n_nodes: int,
) -> npt.NDArray[np.int64]:
    """Which requested object each level-order vertex belongs to (-1: none).

    The chunk's offset in level order plus the fragment's rows in the chunk
    locate every vertex a manifest names.  A vertex two objects both claim
    means their geometry is not separable, and is refused rather than
    written twice.
    """
    from zarr_vectors.building import chunk_local_to_global_offsets, read_vertex_fragment_index

    offsets, _keys, total = chunk_local_to_global_offsets(level_group)
    if total != n_nodes:
        raise ExportError(
            f"the level reads back {n_nodes} vertices but its chunks hold "
            f"{total}; its vertex order cannot be matched to the object manifests"
        )
    owner = np.full(n_nodes, -1, dtype=np.int64)
    indexes: dict[tuple[int, ...], Any] = {}
    for oid in oids:
        for cc, fragment in manifests[oid]:
            key = tuple(int(c) for c in cc)
            if key not in offsets:
                raise ExportError(f"object {oid} names chunk {key}, which has no vertices")
            if key not in indexes:
                indexes[key] = read_vertex_fragment_index(level_group, key)
            rows = _fragment_rows(indexes[key], int(fragment)) + int(offsets[key])
            claimed = owner[rows]
            clash = claimed[(claimed >= 0) & (claimed != oid)]
            if len(clash):
                raise ExportError(
                    f"objects {sorted({int(oid), *clash.tolist()})} share vertices "
                    f"in chunk {key}, so they cannot be written as separate segments"
                )
            owner[rows] = oid
    return owner


def _group_by_owner(
    owner: npt.NDArray[np.int64], links: npt.NDArray[np.int64], oids: Sequence[int],
) -> Iterator[tuple[int, npt.NDArray[np.int64], npt.NDArray[np.int64]]]:
    """Per object: its vertex rows, and its links re-indexed into them.

    One sort over the owners instead of a mask per object, so exporting every
    object of a level stays ``O(n log n)``.  A link whose corners belong to
    different objects belongs to none of them.
    """
    n = len(owner)
    node_order = np.argsort(owner, kind="stable")
    ranked = owner[node_order]
    local = np.empty(n, dtype=np.int64)
    local[node_order] = np.arange(n, dtype=np.int64) - np.searchsorted(ranked, ranked)

    if len(links):
        corner_owner = owner[links]
        link_owner = np.where(
            np.all(corner_owner == corner_owner[:, :1], axis=1), corner_owner[:, 0], -1,
        )
    else:
        link_owner = np.zeros(0, dtype=np.int64)
    link_order = np.argsort(link_owner, kind="stable")
    link_ranked = link_owner[link_order]

    for oid in oids:
        lo, hi = np.searchsorted(ranked, [oid, oid + 1])
        llo, lhi = np.searchsorted(link_ranked, [oid, oid + 1])
        yield oid, node_order[lo:hi], local[links[link_order[llo:lhi]]]


def _manifest_skeletons(
    level_data: dict[str, Any],
    level_group: Any,
    level: int,
    pairs: list[tuple[int, int]],
    plan: list[tuple[str, np.dtype, int]],
    explicit: bool,
    manifests: Sequence[Any],
    offset: npt.NDArray[np.float64],
) -> Iterator[tuple[int, int, Any, Any, list[Any]]]:
    """Skeletons of a store without per-segment reads: one read, then split."""
    positions = level_data["positions"] + offset
    columns = level_data["columns"]
    segment_of = dict(pairs)
    oids = [oid for oid, _segment in pairs]
    owner = _vertex_owner(level_group, manifests, oids, len(positions))
    for oid, nodes, local_edges in _group_by_owner(owner, level_data["edges"], oids):
        if len(nodes) == 0:
            if explicit:
                raise ExportError(f"object {oid} has no geometry at level {level}")
            continue
        yield (
            oid, segment_of[oid], positions[nodes], local_edges,
            [columns[name][nodes] for name, _dtype, _ncols in plan],
        )


def _manifest_meshes(
    store_path: str | Path,
    level_group: Any,
    level: int,
    pairs: list[tuple[int, int]],
    explicit: bool,
    manifests: Sequence[Any],
    offset: npt.NDArray[np.float64],
) -> Iterator[tuple[int, int, Any, Any]]:
    from zarr_vectors.types.meshes import read_mesh

    try:
        result = read_mesh(str(store_path), level=level)
    except Exception as e:
        raise ExportError(f"Failed to read store: {e}") from e
    vertices = np.asarray(result["vertices"], dtype=np.float64) + offset
    faces = np.asarray(result["faces"], dtype=np.int64)
    segment_of = dict(pairs)
    oids = [oid for oid, _segment in pairs]
    owner = _vertex_owner(level_group, manifests, oids, len(vertices))
    for oid, nodes, local_faces in _group_by_owner(owner, faces, oids):
        if len(nodes) == 0:
            if explicit:
                raise ExportError(f"object {oid} has no geometry at level {level}")
            continue
        yield oid, segment_of[oid], vertices[nodes], local_faces


def _em_skeletons(
    store_path: str | Path,
    level_group: Any,
    level: int,
    pairs: list[tuple[int, int]],
    plan: list[tuple[str, np.dtype, int]],
    explicit: bool,
) -> Iterator[tuple[int, int, Any, Any, list[Any]]]:
    """Skeletons of an EM store, one segment at a time.

    Core's ``read_skeleton_by_segment_id`` joins a segment's fragments by
    their implicit path edges and the intra-chunk branch links, but not by
    the links that cross a chunk boundary, so a neuron spanning two chunks
    reads back as two trees.  Those links are added here: for each chunk the
    segment occupies and each cross-chunk offset the level has, the one link
    cell naming both chunks is read, and its records are kept when both
    endpoints are this segment's vertices.
    """
    from zarr_vectors.types.skeletons import read_skeleton_by_segment_id

    from zarr_vectors_tools.algorithms._links import cross_offset_segments

    cross_offsets = [offsets for _seg, offsets in cross_offset_segments(level_group)]
    for oid, segment in pairs:
        result = read_skeleton_by_segment_id(str(store_path), segment, level=level)
        if result is None or len(result["positions"]) == 0:
            if explicit:
                raise ExportError(
                    f"object {oid} (segment {segment}) has no geometry at level "
                    f"{level}; a coarser level's sparsity may have dropped it"
                )
            continue
        positions = np.asarray(result["positions"], dtype=np.float64)
        n = len(positions)
        edges = np.asarray(result["edges"], dtype=np.int64).reshape(-1, 2)
        if cross_offsets:
            crossing = _crossing_edges(level_group, oid, cross_offsets)
            if len(crossing):
                edges = np.unique(np.concatenate([edges, crossing]), axis=0)
        columns = []
        for name, _dtype, ncols in plan:
            values = (result.get("attributes") or {}).get(name)
            if values is None or np.asarray(values).size != n * ncols:
                raise ExportError(
                    f"segment {segment} has no {name!r} value for each of its {n} "
                    f"vertices at level {level}; pass attribute_names without it"
                )
            columns.append(np.asarray(values).reshape(n, ncols))
        yield oid, segment, positions, edges, columns


def _crossing_edges(
    level_group: Any, oid: int, cross_offsets: Sequence[Sequence[Sequence[int]]],
) -> npt.NDArray[np.int64]:
    """An EM object's chunk-crossing edges, in its reader's vertex numbering.

    ``read_skeleton_by_segment_id`` concatenates fragments in manifest order,
    so a fragment's first vertex sits at the running count of those before
    it, and its chunk-local rows map onto the object from there.
    """
    from zarr_vectors.building import (
        read_links_for_tuple,
        read_object_manifests,
        read_vertex_fragment_index,
    )

    manifest = read_object_manifests(level_group, ids=[oid]).get(oid) or []
    vertex_of: dict[tuple[tuple[int, ...], int], int] = {}
    indexes: dict[tuple[int, ...], Any] = {}
    running = 0
    for cc, fragment in manifest:
        key = tuple(int(c) for c in cc)
        if key not in indexes:
            indexes[key] = read_vertex_fragment_index(level_group, key)
        rows = _fragment_rows(indexes[key], int(fragment))
        for i, row in enumerate(rows.tolist()):
            vertex_of[(key, row)] = running + i
        running += len(rows)

    chunks = set(indexes)
    found: list[tuple[int, int]] = []
    for a in chunks:
        for offsets in cross_offsets:
            b = tuple(int(x) + int(o) for x, o in zip(a, offsets[0]))
            if b == a or b not in chunks:
                continue
            for record in read_links_for_tuple(level_group, (a, b)):
                ends = [vertex_of.get((tuple(int(c) for c in cc), int(row))) for cc, row in record]
                if None not in ends:
                    found.append((ends[0], ends[1]))
    return np.asarray(found, dtype=np.int64).reshape(-1, 2)


def _triangulate(faces: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Triangles from ``(F, L)`` polygons, fanned from each first corner."""
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) == 0:
        return np.zeros((0, 3), dtype=np.int64)
    if faces.shape[1] == 3:
        return faces
    fans = [
        np.stack([faces[:, 0], faces[:, i], faces[:, i + 1]], axis=1)
        for i in range(1, faces.shape[1] - 1)
    ]
    return np.concatenate(fans, axis=0)


# ===================================================================
# Encoding
# ===================================================================

def _encode_skeleton(
    positions_nm: npt.NDArray[np.float64],
    edges: npt.NDArray[np.int64],
    columns: list[npt.NDArray[Any]],
    plan: list[tuple[str, np.dtype, int]],
    segment: int,
) -> bytes:
    """One ``neuroglancer_skeletons`` file.

    ``uint32`` vertex and edge counts, ``float32`` vertices, ``uint32`` edge
    endpoints, then each attribute in the order the ``info`` lists them, all
    little-endian.
    """
    n = len(positions_nm)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if len(edges) and (edges.min() < 0 or edges.max() >= n):
        raise ExportError(f"segment {segment} has an edge to a vertex it does not hold")
    parts = [
        np.array([n, len(edges)], dtype="<u4").tobytes(),
        np.asarray(positions_nm, dtype="<f4").reshape(n, 3).tobytes(),
        edges.astype("<u4").tobytes(),
    ]
    for values, (_name, dtype, ncols) in zip(columns, plan):
        parts.append(
            np.asarray(values).reshape(n, ncols).astype(dtype.newbyteorder("<")).tobytes()
        )
    return b"".join(parts)


def _encode_mesh(
    vertices_nm: npt.NDArray[np.float64], triangles: npt.NDArray[np.int64], segment: int,
) -> bytes:
    """One legacy mesh fragment: vertex count, vertices, triangle corners."""
    n = len(vertices_nm)
    if len(triangles) and (triangles.min() < 0 or triangles.max() >= n):
        raise ExportError(f"segment {segment} has a face on a vertex it does not hold")
    return b"".join([
        np.array([n], dtype="<u4").tobytes(),
        np.asarray(vertices_nm, dtype="<f4").reshape(n, 3).tobytes(),
        np.asarray(triangles, dtype="<u4").reshape(-1, 3).tobytes(),
    ])


# ===================================================================
# Segment properties
# ===================================================================

def _object_attribute_names(level_group: Any) -> list[str]:
    from zarr_vectors.constants import OBJECT_ATTRIBUTES

    try:
        return sorted(level_group[OBJECT_ATTRIBUTES].children())
    except Exception:  # noqa: BLE001 - a level with no object attributes
        return []


def _segment_properties(
    store_path: str | Path,
    root: Any,
    em: bool,
    written: list[tuple[int, int]],
    names: Sequence[str] | None,
) -> dict[str, Any]:
    """The ``inline`` segment-properties block for ``(object, segment)`` pairs.

    Properties are read from level 0, which is where ingest puts them and
    which every coarser level answers to: an EM store's rows are found by
    segment id, since sparsity renumbers a coarse level's objects, and any
    other store's by object id, which the pyramid builders preserve.
    """
    from zarr_vectors.building import get_resolution_level, read_object_attributes

    ids = [str(segment) for _oid, segment in written]
    level0 = get_resolution_level(root, 0)
    available = [n for n in _object_attribute_names(level0) if n != _SEGMENT_ID_ATTR]
    explicit = names is not None
    wanted = list(names) if explicit else available
    missing = [n for n in wanted if n not in available]
    if missing:
        raise ExportError(
            f"object attribute(s) {missing} are not at level 0; it has "
            f"{available or 'none'}"
        )

    if em and wanted:
        base = np.asarray(read_object_attributes(level0, _SEGMENT_ID_ATTR), dtype=np.uint64)
        wanted_ids = np.asarray([segment for _oid, segment in written], dtype=np.uint64)
        pos = np.searchsorted(base, wanted_ids)
        inside = pos < len(base)
        hit = np.zeros(len(pos), dtype=bool)
        hit[inside] = base[pos[inside]] == wanted_ids[inside]
        rows = np.where(hit, pos, -1)
    else:
        rows = np.asarray([oid for oid, _segment in written], dtype=np.int64)

    properties: list[dict[str, Any]] = []
    taken: set[str] = set()
    for name in wanted:
        column = np.asarray(read_object_attributes(level0, name))
        prop, why = _property(name, column, rows, taken)
        if prop is None:
            if explicit:
                raise ExportError(f"object attribute {name!r} {why}")
            continue
        properties.append(prop)
        if prop["type"] in ("label", "description"):
            taken.add(prop["type"])

    if not explicit and "label" not in taken and not em:
        labels = _header_labels(store_path)
        if labels:
            properties.insert(0, {
                "id": "label", "type": "label",
                "values": [labels.get(oid, "") for oid, _segment in written],
            })
    return {"ids": ids, "properties": properties}


def _property(
    name: str, column: npt.NDArray[Any], rows: npt.NDArray[np.int64], taken: set[str],
) -> tuple[dict[str, Any] | None, str]:
    """One segment property from an object attribute, or why it has none."""
    if column.ndim > 1 and column.shape[1] != 1:
        return None, f"has {column.shape[1]} columns; a segment property holds one value"
    column = column.reshape(-1)
    valid = (rows >= 0) & (rows < len(column))
    if len(column):
        picked = column[np.where(valid, rows, 0)]
    else:
        picked = np.zeros(len(rows), dtype=column.dtype)

    if column.dtype.kind in "SUO":
        text = [
            (value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value))
            .rstrip("\x00")
            if ok else ""
            for value, ok in zip(picked.tolist(), valid.tolist())
        ]
        # Neuroglancer shows one label and one description per segment;
        # further text columns are plain string properties.
        kind = "string"
        if name in ("label", "name") and "label" not in taken:
            kind = "label"
        elif name == "description" and "description" not in taken:
            kind = "description"
        return {"id": name, "type": kind, "values": text}, ""

    if column.dtype.kind not in "biuf":
        return None, f"is {column.dtype}, which no segment property type holds"
    # A segment the attribute has no row for reads 0, in the column's type.
    values = picked.copy()
    values[~valid] = 0
    if column.dtype.kind == "f":
        if not np.all(np.isfinite(values)):
            return None, "has non-finite values, which a JSON segment property cannot hold"
        dtype = "float32"
        out = [float(v) for v in values.astype(np.float32).tolist()]
    else:
        dtype = _integer_property_dtype(values)
        if dtype is None:
            return None, "has values outside 32 bits, which a segment property cannot hold"
        out = [int(v) for v in values.tolist()]
    return {"id": name, "type": "number", "data_type": dtype, "values": out}, ""


def _integer_property_dtype(values: npt.NDArray[Any]) -> str | None:
    if values.dtype.kind == "b":
        return "uint8"
    if values.dtype.name in _PRECOMPUTED_INTS:
        return values.dtype.name
    if len(values) == 0:
        return "int32"
    lo, hi = int(values.min()), int(values.max())
    if lo >= 0 and hi <= np.iinfo(np.uint32).max:
        return "uint32"
    if lo >= np.iinfo(np.int32).min and hi <= np.iinfo(np.int32).max:
        return "int32"
    return None


def _header_labels(store_path: str | Path) -> dict[int, str]:
    """Object names an ingest kept in a header rather than an attribute."""
    try:
        from zarr_vectors_tools.headers.registry import HeaderRegistry

        registry = HeaderRegistry(str(store_path))
        if registry.has("obj"):
            return dict(enumerate(registry.get("obj").object_names))
        if registry.has("surface"):
            return {
                int(entry["object_id"]): str(entry["hemisphere"])
                for entry in registry.get("surface").hemispheres
            }
    except Exception:  # noqa: BLE001 - names are a nicety, never load-bearing
        pass
    return {}


def _only_segments(inline: dict[str, Any], written: list[tuple[int, int]]) -> dict[str, Any]:
    """The properties of the segments actually written.

    A whole-level export skips objects a coarse level left empty, and the
    layer should not list segments it has no file for.
    """
    keep = {str(segment) for _oid, segment in written}
    rows = [k for k, sid in enumerate(inline["ids"]) if sid in keep]
    if len(rows) == len(inline["ids"]):
        return inline
    return {
        "ids": [inline["ids"][k] for k in rows],
        "properties": [
            {**prop, "values": [prop["values"][k] for k in rows]}
            for prop in inline["properties"]
        ],
    }


def _merge_properties(old: dict[str, Any] | None, new: dict[str, Any]) -> dict[str, Any]:
    """Union two ``inline`` blocks by segment id; this export's values win.

    A layer is often filled in more than one export.  A property is matched
    by id, except a label or description, which Neuroglancer allows once per
    layer and so are matched by type.  Segments a property does not cover
    get an empty string, or 0.
    """
    if not old or not old.get("ids"):
        return new
    old_ids = [str(i) for i in old["ids"]]
    known = set(old_ids)
    ids = old_ids + [i for i in new["ids"] if i not in known]
    row = {sid: k for k, sid in enumerate(ids)}

    def slot(prop: dict[str, Any]) -> Any:
        return prop["type"] if prop["type"] in ("label", "description") else prop["id"]

    merged: dict[Any, dict[str, Any]] = {}
    sources = ((old_ids, old.get("properties") or []), (new["ids"], new["properties"]))
    for source_ids, props in sources:
        for prop in props:
            key = slot(prop)
            fill: Any = 0 if prop["type"] == "number" else [] if prop["type"] == "tags" else ""
            target = merged.get(key)
            if (
                target is None
                or target["type"] != prop["type"]
                or target.get("data_type") != prop.get("data_type")
            ):
                target = {k: v for k, v in prop.items() if k != "values"}
                target["values"] = [fill] * len(ids)
                merged[key] = target
            for sid, value in zip(source_ids, prop.get("values") or []):
                target["values"][row[str(sid)]] = value
    return {"ids": ids, "properties": list(merged.values())}


# ===================================================================
# The layer
# ===================================================================

def _check_existing_skeletons(
    url: str, directory: str, existing: dict[str, Any] | None,
    vertex_attributes: list[dict[str, Any]],
) -> None:
    if existing is None:
        return
    where = f"{url}/{directory}"
    if existing.get("@type") != _SKELETON_TYPE:
        raise ExportError(f"{where} holds {existing.get('@type')!r}, not skeletons")
    for key, what in (("sharding", "sharded"), ("spatial_index", "spatially indexed")):
        if existing.get(key):
            raise ExportError(
                f"{where} is {what}; this exporter writes one unsharded file per "
                f"segment, which that layer would not read.  Export to a new layer."
            )
    if list(existing.get("transform") or _IDENTITY) != _IDENTITY:
        raise ExportError(
            f"{where} has a non-identity transform; skeletons in nanometres "
            f"would be misplaced in it"
        )
    theirs = [
        (a.get("id"), a.get("data_type"), int(a.get("num_components", 1)))
        for a in existing.get("vertex_attributes") or []
    ]
    ours = [(a["id"], a["data_type"], a["num_components"]) for a in vertex_attributes]
    if theirs != ours:
        raise ExportError(
            f"{where} carries vertex attributes {theirs}, and this export would "
            f"carry {ours}; one layer decodes every skeleton the same way, so pass "
            f"attribute_names to match or export to a new layer"
        )


def _check_existing_meshes(url: str, directory: str, existing: dict[str, Any] | None) -> None:
    if existing is None:
        return
    where = f"{url}/{directory}"
    kind = existing.get("@type", _MESH_TYPE)
    if kind != _MESH_TYPE or existing.get("sharding"):
        raise ExportError(
            f"{where} holds {'sharded ' if existing.get('sharding') else ''}{kind!r} "
            f"meshes; this exporter writes unsharded {_MESH_TYPE!r}.  Export to a "
            f"new layer."
        )


class _Layer:
    """The output layer: batched object uploads and its ``info`` files."""

    def __init__(self, output: str | Path, compress: str | None) -> None:
        try:
            from cloudfiles import CloudFiles
        except ImportError as e:
            raise ImportError(
                "precomputed export writes through cloud-files: "
                "pip install 'zarr-vectors-tools[precomputed]'"
            ) from e
        self.url = layer_url(output)
        self.cf = CloudFiles(self.url)
        self.compress = compress
        self._files: list[tuple[str, bytes]] = []
        self._jsons: list[tuple[str, Any]] = []

    def prepare(
        self,
        key: str,
        bounds_nm: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
        chunk_nm: npt.NDArray[np.float64],
    ) -> tuple[dict[str, Any], str, dict[str, Any] | None]:
        """The layer ``info`` to finish with, the subdirectory, and its info."""
        info = self.cf.get_json("info")
        if info is None:
            info = _placeholder_info(bounds_nm, chunk_nm)
        elif info.get("type") != "segmentation" or not info.get("scales"):
            kind = info.get("@type") or info.get("type") or "unrecognised"
            raise ExportError(
                f"{self.url} already holds a precomputed {kind} layer, not a "
                f"segmentation layer; export to an empty location or into a "
                f"segmentation layer"
            )
        directory = str(info.get(key) or key).strip("/")
        return info, directory, self.cf.get_json(f"{directory}/info")

    def put(self, path: str, content: bytes) -> None:
        self._files.append((path, content))
        if len(self._files) >= _UPLOAD_BATCH:
            self.flush()

    def put_json(self, path: str, content: Any) -> None:
        self._jsons.append((path, content))
        if len(self._jsons) >= _UPLOAD_BATCH:
            self.flush()

    def flush(self) -> None:
        if self._files:
            self.cf.puts(
                self._files, content_type="application/octet-stream",
                compress=self.compress,
            )
            self._files = []
        if self._jsons:
            # Manifests stay uncompressed, as Neuroglancer fetches them.
            self.cf.put_jsons(self._jsons)
            self._jsons = []

    def finish(
        self,
        layer_info: dict[str, Any],
        key: str,
        directory: str,
        kind_info: dict[str, Any],
        properties: dict[str, Any],
    ) -> list[str]:
        """Write the properties and both infos, last, so none names missing files.

        Returns the property ids now in the subdirectory's properties.
        """
        props_dir = str(kind_info.get("segment_properties") or _PROPERTIES_DIR).strip("/")
        old = self.cf.get_json(f"{directory}/{props_dir}/info")
        inline = _merge_properties((old or {}).get("inline"), properties)
        self.cf.put_json(
            f"{directory}/{props_dir}/info",
            {"@type": _PROPERTIES_TYPE, "inline": inline},
        )
        kind_info["segment_properties"] = props_dir
        self.cf.put_json(f"{directory}/info", kind_info)
        layer_info[key] = directory
        # The layer-level pointer is Neuroglancer's for the segmentation
        # layer as a whole; an existing one is left where it points.
        layer_info.setdefault("segment_properties", f"{directory}/{props_dir}")
        self.cf.put_json("info", layer_info)
        return [prop["id"] for prop in inline["properties"]]


def _placeholder_info(
    bounds_nm: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    chunk_nm: npt.NDArray[np.float64],
) -> dict[str, Any]:
    """A segmentation ``info`` whose one scale has no image data.

    One voxel per level-0 chunk keeps the grid Neuroglancer and CloudVolume
    see small -- a few chunk requests at most, all answered "missing" --
    while still spanning the store's bounds, so a viewer frames the data.
    """
    resolution = np.where(chunk_nm > 0, chunk_nm, 1.0)
    lo = np.floor(bounds_nm[0] / resolution).astype(np.int64)
    hi = np.ceil(bounds_nm[1] / resolution).astype(np.int64)
    size = np.maximum(hi - lo, 1)
    return {
        "@type": "neuroglancer_multiscale_volume",
        "type": "segmentation",
        "data_type": "uint64",
        "num_channels": 1,
        "scales": [{
            "key": "placeholder",
            "encoding": "raw",
            "resolution": [float(r) for r in resolution],
            "voxel_offset": [int(v) for v in lo],
            "size": [int(s) for s in size],
            "chunk_sizes": [[int(min(s, _PLACEHOLDER_CHUNK)) for s in size]],
        }],
    }
