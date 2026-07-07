"""ETL: plain precomputed skeletons (no ``.frags`` spatial index) → multiscale
zarr-vectors store.

Source: a precomputed skeleton layer whose ``info`` does **not** carry a
``spatial_index`` block.  Each segment's skeleton is stored as a separate file
``<segment-id>`` (or via sharding) under the layer root.  The full list of
segment IDs is discovered from the layer's ``segment_properties`` sub-directory
(inline ``ids`` array in ``segment_properties/info``).

Pipeline:

1. **Read** – parallel cloud reads of all (or a subset of) segment skeletons
   using the cloud-volume library.  Applies the layer transform so all
   coordinates are in physical nanometre space.
2. **Bounds** – if not supplied by the caller, compute the axis-aligned bounding
   box of all vertex positions and round outward to the target chunk grid.
3. **Spatial distribute** – assign each neuron's vertices to origin-aligned zarr
   chunks using the existing :func:`pieces_from_chunk` helper.  Cross-chunk
   edges (neurites that span a chunk boundary) are preserved and recorded.
4. **Write** – write each zarr chunk with :func:`write_skeleton_chunk`.
5. **Object index** – build the segment-id → dense-OID map and fragment manifests
   via ``build_object_index``.
6. **Segment properties → object attributes** – convert the precomputed inline
   property table to per-object zarr-vectors arrays (``write_object_attributes``).
7. **Pyramid** – optionally coarsen to multiple LOD levels
   (``build_skeleton_pyramid``).

Example::

    from zarr_vectors_tools.ingest.precomputed_plain_skeletons import (
        PlainPrecomputedReader, run_ingest_plain,
    )

    reader = PlainPrecomputedReader("precomputed://gs://allen_neuroglancer_ccf/Mouselight")
    summary = run_ingest_plain(
        reader,
        "/tmp/mouselight.zv",
        chunk_shape_nm=(1_000_000, 1_000_000, 1_000_000),
        strides=[8, 8, 8],
        chunk_scale_factors=[2, 2, 2],
        sparsity_factors=[1.0, 1.0, 4.0],
        read_workers=16,
        pyramid_workers=8,
    )
    print(summary)
"""

from __future__ import annotations

import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

from zarr_vectors.core.arrays import write_object_attributes
from zarr_vectors.types import skeletons as sk
from zarr_vectors.typing import ChunkCoords
from zarr_vectors_tools.multiresolution.object_index import build_object_index
from zarr_vectors_tools.multiresolution.strategies.skeletons import (
    build_skeleton_pyramid,
    coarsen_skeleton_level,
)

# Re-use the coordinate-distribution helper from the .frags ingest module.
from zarr_vectors_tools.ingest.precomputed_skeletons import pieces_from_chunk


# ===================================================================
# Source description
# ===================================================================

@dataclass
class PlainSkeletonInfo:
    """Metadata parsed from a plain precomputed skeleton ``info`` file."""

    base_url: str
    """Layer URL (no trailing slash)."""

    transform: npt.NDArray
    """4×4 float64 homogeneous affine: ``stored_nm = transform @ [x, y, z, 1]``."""

    vertex_attributes: list[dict[str, Any]] = field(default_factory=list)
    """Per-vertex attribute descriptors (same schema as precomputed ``vertex_attributes``)."""

    seg_props_path: str | None = None
    """Relative path to the segment-properties sub-directory, e.g. ``"segment_properties"``."""

    sharding: dict[str, Any] | None = None
    """Sharding spec if the layer uses the sharded format, else ``None``."""

    @property
    def attribute_names(self) -> list[str]:
        return [a["id"] for a in self.vertex_attributes]

    @property
    def attribute_dtypes(self) -> dict[str, str]:
        return {a["id"]: a.get("data_type", "float32") for a in self.vertex_attributes}


def _parse_4x3_transform(raw: list[float]) -> npt.NDArray:
    """Parse the 12-element flat 4×3 transform from the ``info`` file.

    The precomputed spec stores the transform as 12 numbers in row-major
    order representing a 4-column × 3-row matrix (i.e. the transposed
    [3×4] stored-model→model affine).  We expand it to a 4×4 homogeneous
    matrix for uniform application.

    Layout (per spec): ``[m00, m01, m02, m03, m10, m11, m12, m13,
    m20, m21, m22, m23]`` where ``model_xyz = M · stored_xyz + t``.
    """
    m = np.array(raw, dtype=np.float64).reshape(3, 4)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :4] = m
    return mat


# ===================================================================
# Cloud reader
# ===================================================================

class PlainPrecomputedReader:
    """Read plain (non-``.frags``) precomputed skeleton segments.

    Each skeleton is fetched individually via the cloud-volume library.
    Requires the optional ``ingest`` extras (``cloud-volume``,
    ``cloud-files``).

    Args:
        base_url: Layer URL, e.g.
            ``"precomputed://gs://allen_neuroglancer_ccf/Mouselight"``.
    """

    def __init__(self, base_url: str) -> None:
        from cloudfiles import CloudFiles  # noqa: F401 — import check

        self.base_url = base_url.rstrip("/")
        self._raw_info = self._read_info()
        self.info = self._build_info()
        self._seg_ids: list[int] | None = None
        self._seg_props_raw: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Internal helpers

    def _read_info(self) -> dict[str, Any]:
        from cloudfiles import CloudFiles

        # Strip the "precomputed://" scheme that cloud-volume uses but
        # CloudFiles does not need.
        url = self.base_url
        if url.startswith("precomputed://"):
            url = url[len("precomputed://"):]
        cf = CloudFiles(url)
        info = cf.get_json("info")
        if info is None:
            raise FileNotFoundError(f"no info at {self.base_url}/info")
        return info

    def _build_info(self) -> PlainSkeletonInfo:
        raw = self._raw_info
        transform_raw = raw.get("transform", [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])
        transform = _parse_4x3_transform(transform_raw)
        return PlainSkeletonInfo(
            base_url=self.base_url,
            transform=transform,
            vertex_attributes=raw.get("vertex_attributes", []) or [],
            seg_props_path=raw.get("segment_properties"),
            sharding=raw.get("sharding"),
        )

    # ------------------------------------------------------------------
    # Public API

    @property
    def segment_ids(self) -> list[int]:
        """Sorted list of segment IDs from the ``segment_properties`` sub-dir."""
        if self._seg_ids is None:
            self._seg_ids = self._load_seg_ids()
        return self._seg_ids

    @property
    def segment_properties_raw(self) -> dict[str, Any] | None:
        """Raw ``inline`` dict from ``segment_properties/info``, or ``None``."""
        if self._seg_props_raw is None and self.info.seg_props_path:
            self._seg_props_raw = self._load_seg_props_raw()
        return self._seg_props_raw

    def _load_seg_ids(self) -> list[int]:
        props_path = self.info.seg_props_path
        if props_path is None:
            raise RuntimeError(
                f"No 'segment_properties' entry in info at {self.base_url}. "
                "Pass explicit seg_ids= to run_ingest_plain."
            )
        from cloudfiles import CloudFiles

        url = self.base_url
        if url.startswith("precomputed://"):
            url = url[len("precomputed://"):]
        cf = CloudFiles(url)
        props_info = cf.get_json(f"{props_path}/info")
        if props_info is None:
            raise FileNotFoundError(
                f"segment_properties/info not found at {self.base_url}/{props_path}/info"
            )
        inline = props_info.get("inline")
        if inline is None:
            raise ValueError(
                f"segment_properties/info has no 'inline' key at {self.base_url}"
            )
        ids = [int(x) for x in inline["ids"]]
        return sorted(ids)

    def _load_seg_props_raw(self) -> dict[str, Any]:
        from cloudfiles import CloudFiles

        props_path = self.info.seg_props_path
        url = self.base_url
        if url.startswith("precomputed://"):
            url = url[len("precomputed://"):]
        cf = CloudFiles(url)
        props_info = cf.get_json(f"{props_path}/info")
        if props_info is None:
            return {}
        return props_info.get("inline", {})

    def read_skeleton(
        self, seg_id: int
    ) -> dict[str, npt.NDArray] | None:
        """Read one segment's skeleton, returning coordinates in nanometres.

        Uses the cloud-volume ``CloudVolume`` client which handles both
        sharded and unsharded formats transparently.

        Returns a dict with keys:

        - ``"vertices"`` – ``(N, 3)`` float32 in nm
        - ``"edges"``    – ``(M, 2)`` int64 ``[child, parent]``
        - plus one array per vertex attribute in ``self.info.vertex_attributes``

        Returns ``None`` if the segment has no skeleton in the store.
        """
        try:
            import cloudvolume as cv
        except ImportError as e:
            raise ImportError(
                "PlainPrecomputedReader.read_skeleton requires cloud-volume: "
                "pip install cloud-volume"
            ) from e

        try:
            vol = cv.CloudVolume(
                self.base_url,
                mip=0,
                use_https=True,
                parallel=False,
                progress=False,
                fill_missing=True,
            )
            skel = vol.skeleton.get(seg_id)
        except Exception:
            return None

        if skel is None or len(skel.vertices) == 0:
            return None

        # CloudVolume already returns vertices in nm when reading precomputed
        # skeletons (it applies the layer transform internally).  We re-apply
        # our parsed transform only if CloudVolume did NOT apply it
        # (i.e. returned raw stored coords).  In practice cv.skeleton.get
        # returns nm coords, so we use them directly.
        verts = np.asarray(skel.vertices, dtype=np.float32)
        edges = np.asarray(skel.edges, dtype=np.int64).reshape(-1, 2)

        piece: dict[str, npt.NDArray] = {
            "vertices": verts,
            "edges": edges,
        }
        for attr in self.info.vertex_attributes:
            name = attr["id"]
            val = getattr(skel, name, None)
            if val is not None:
                piece[name] = np.asarray(val)
        return piece


# ===================================================================
# Segment-properties parsing
# ===================================================================

_NUMPY_TYPE_MAP: dict[str, str] = {
    "uint8": "uint8", "int8": "int8",
    "uint16": "uint16", "int16": "int16",
    "uint32": "uint32", "int32": "int32",
    "float32": "float32",
}

_STRING_PROP_TYPES = {"label", "description", "string"}
_STRING_DTYPE = "S256"   # fixed-width bytes; callers decode as UTF-8


def parse_segment_properties(
    inline: dict[str, Any],
) -> tuple[list[int], dict[str, tuple[str, list, dict[str, Any]]]]:
    """Parse the ``inline`` block of a segment-properties ``info`` file.

    Args:
        inline: The ``inline`` value from the segment-properties ``info``
            JSON — must contain ``"ids"`` and ``"properties"`` keys.

    Returns:
        A 2-tuple ``(seg_ids, props)`` where:

        - ``seg_ids`` is a sorted list of ``int`` segment IDs.
        - ``props`` is a dict mapping ``prop_id → (type, values, meta)``
          where:
          - ``type`` is the precomputed property type string (``"label"``,
            ``"string"``, ``"description"``, ``"number"``, ``"tags"``).
          - ``values`` is a list of raw values aligned with ``seg_ids``.
          - ``meta`` is extra info: ``{"data_type": ...}`` for numbers,
            ``{"tags": [...]}`` for tags, empty dict otherwise.
    """
    ids = [int(x) for x in inline.get("ids", [])]
    props: dict[str, tuple[str, list, dict[str, Any]]] = {}
    for prop in inline.get("properties", []):
        pid = prop["id"]
        ptype = prop["type"]
        values = prop.get("values", [])
        meta: dict[str, Any] = {}
        if ptype == "number":
            meta["data_type"] = prop.get("data_type", "float32")
        elif ptype == "tags":
            meta["tags"] = prop.get("tags", [])
            meta["tag_descriptions"] = prop.get("tag_descriptions", [])
        if prop.get("description"):
            meta["description"] = prop["description"]
        props[pid] = (ptype, values, meta)
    return ids, props


def _build_object_attribute_array(
    ptype: str,
    values: list,
    meta: dict[str, Any],
    seg_ids_ordered: list[int],   # IDs in property table order
    oid_of_seg: dict[int, int],   # seg_id → dense OID
    n_objects: int,
) -> npt.NDArray | None:
    """Build a dense (n_objects,) array from a segment-properties column.

    Only segments present in ``oid_of_seg`` (i.e. those that have geometry)
    are included.  Segments in the property table but without geometry are
    silently skipped.

    Returns ``None`` if the property type is not mappable to a numpy array.
    """
    if ptype in _STRING_PROP_TYPES:
        out = np.zeros(n_objects, dtype=_STRING_DTYPE)
        for idx, sid in enumerate(seg_ids_ordered):
            oid = oid_of_seg.get(sid)
            if oid is None:
                continue
            raw = values[idx] if idx < len(values) else ""
            encoded = str(raw).encode("utf-8")[:256]
            out[oid] = encoded
        return out

    if ptype == "number":
        np_dtype = _NUMPY_TYPE_MAP.get(meta.get("data_type", "float32"), "float32")
        out = np.zeros(n_objects, dtype=np_dtype)
        for idx, sid in enumerate(seg_ids_ordered):
            oid = oid_of_seg.get(sid)
            if oid is None:
                continue
            out[oid] = values[idx] if idx < len(values) else 0
        return out

    if ptype == "tags":
        # Encode as uint64 bitmask: bit N is set when tag N is present.
        out = np.zeros(n_objects, dtype=np.uint64)
        for idx, sid in enumerate(seg_ids_ordered):
            oid = oid_of_seg.get(sid)
            if oid is None:
                continue
            tag_indices = values[idx] if idx < len(values) else []
            mask = np.uint64(0)
            for ti in tag_indices:
                if ti < 64:
                    mask |= np.uint64(1) << np.uint64(ti)
            out[oid] = mask
        return out

    return None


# ===================================================================
# Driver
# ===================================================================

def run_ingest_plain(
    reader: "PlainPrecomputedReader",
    out_store: str | Path,
    *,
    chunk_shape_nm: tuple[float, ...] = (1_000_000.0, 1_000_000.0, 1_000_000.0),
    bounds_nm: tuple[list[float], list[float]] | None = None,
    seg_ids: list[int] | None = None,
    attribute_names: Sequence[str] | None = None,
    attribute_dtypes: dict[str, str] | None = None,
    strides: Sequence[int] = (),
    chunk_scale_factors: Sequence[int | tuple[int, ...]] | None = None,
    sparsity_factors: Sequence[float] | None = None,
    sparsity_strategy: str = "length",
    drop_interior_below: int = 0,
    backend: str | None = None,
    progress: bool = True,
    read_workers: int = 8,
    pyramid_workers: int | None = None,
) -> dict[str, Any]:
    """Ingest a plain precomputed skeleton layer to a multiscale zarr-vectors store.

    Unlike :func:`zarr_vectors_tools.ingest.precomputed_skeletons.run_ingest`,
    this function does **not** assume the existence of ``.frags`` spatially-
    indexed chunk files.  Instead it:

    1. Reads all (or a subset of) skeletons in parallel using ``read_workers``
       threads.
    2. Computes bounds from the vertex data if ``bounds_nm`` is not supplied.
    3. Distributes vertices across zarr chunks, preserving cross-chunk
       connectivity via the standard :func:`pieces_from_chunk` mechanism.
    4. Writes the zarr-vectors store, object index, and segment-property-based
       object attributes.
    5. Builds a multi-LOD pyramid if ``strides`` is non-empty.

    Args:
        reader: A :class:`PlainPrecomputedReader` (or any object exposing
            ``.info`` (:class:`PlainSkeletonInfo`) and
            ``.read_skeleton(seg_id)``, ``.segment_ids``,
            ``.segment_properties_raw``).
        out_store: Output zarr-vectors store path.
        chunk_shape_nm: Target spatial chunk size in nm per axis.  Defaults
            to 1 mm³ ``(1_000_000, 1_000_000, 1_000_000)``.
        bounds_nm: Optional ``(min_corner, max_corner)`` in nm.  If
            ``None``, computed from the vertex data and rounded outward to
            the chunk grid.
        seg_ids: Optional explicit list of segment IDs to ingest.  Defaults
            to all IDs in the layer's segment_properties.
        attribute_names: Per-vertex attributes to carry.  Defaults to all
            attributes in ``reader.info.vertex_attributes``.
        attribute_dtypes: Override dtype map for per-vertex attributes.
        strides: Per-pyramid-level decimation strides (keep every k-th
            vertex).  Empty → level 0 only.
        chunk_scale_factors: Per-level spatial coarsening (default: 2 per axis).
        sparsity_factors: Per-level fraction of objects to keep (default: 1.0).
        sparsity_strategy: Sparsification strategy (``"length"`` or other).
        drop_interior_below: LOD: drop fully-interior objects with ≤ N vertices.
        backend: zarr-vectors storage backend override.
        progress: If ``True``, print progress messages.
        read_workers: Number of threads for parallel skeleton reads (IO-bound).
        pyramid_workers: Number of Dask *process* workers for pyramid building
            (CPU-bound).  ``None`` → serial.

    Returns:
        Summary dict with keys ``objects``, ``level0_fragments``,
        ``level0_cross_chunk_edges``, and optionally ``pyramid``.
    """
    info = reader.info
    ndim = 3

    # Resolve attribute info
    if attribute_names is None:
        attribute_names = info.attribute_names
    if attribute_dtypes is None:
        attribute_dtypes = {
            n: info.attribute_dtypes.get(n, "float32") for n in attribute_names
        }
    np_attr_dtypes = {n: np.dtype(attribute_dtypes[n]) for n in attribute_names}

    # Resolve segment ID list
    if seg_ids is None:
        seg_ids = reader.segment_ids
    if progress:
        print(f"  [plain-ingest] {len(seg_ids)} segments to ingest", flush=True)

    # ------------------------------------------------------------------
    # Phase 1 — parallel skeleton reads
    # ------------------------------------------------------------------
    all_pieces: dict[int, dict[str, npt.NDArray]] = {}

    def _read_one(sid: int) -> tuple[int, dict | None]:
        return sid, reader.read_skeleton(sid)

    if progress:
        print(f"  [plain-ingest] reading skeletons ({read_workers} threads)…", flush=True)

    with ThreadPoolExecutor(max_workers=read_workers) as pool:
        futures = {pool.submit(_read_one, sid): sid for sid in seg_ids}
        done = 0
        for fut in as_completed(futures):
            sid, piece = fut.result()
            if piece is not None and len(piece["vertices"]) > 0:
                all_pieces[sid] = piece
            done += 1
            if progress and done % max(1, len(seg_ids) // 20) == 0:
                print(
                    f"  [plain-ingest] read {done}/{len(seg_ids)} "
                    f"({len(all_pieces)} with geometry)",
                    flush=True,
                )

    if progress:
        print(
            f"  [plain-ingest] read complete: {len(all_pieces)}/{len(seg_ids)} "
            "have geometry",
            flush=True,
        )

    if not all_pieces:
        raise RuntimeError("No skeletons with geometry found — aborting.")

    # ------------------------------------------------------------------
    # Phase 2 — compute bounds if not provided
    # ------------------------------------------------------------------
    if bounds_nm is None:
        if progress:
            print("  [plain-ingest] computing bounds from vertices…", flush=True)
        all_verts = np.concatenate([p["vertices"] for p in all_pieces.values()], axis=0)
        vmin = all_verts.min(axis=0).astype(np.float64)
        vmax = all_verts.max(axis=0).astype(np.float64)
        cs = np.asarray(chunk_shape_nm, dtype=np.float64)
        # Round min inward (floor) and max outward (ceil), then expand by one
        # extra chunk on each side for safety.
        lo = np.floor(vmin / cs) * cs
        hi = (np.ceil(vmax / cs) + 1) * cs
        bounds_nm = (lo.tolist(), hi.tolist())
        if progress:
            print(f"  [plain-ingest] bounds (nm): {bounds_nm}", flush=True)
        del all_verts

    # Store origin shift: zarr coordinates are world - bounds_nm[0].
    grid_origin = np.asarray(bounds_nm[0], dtype=np.float64)
    store_bounds = (
        [0.0] * ndim,
        (np.asarray(bounds_nm[1], dtype=np.float64) - grid_origin).tolist(),
    )
    coord_off = grid_origin.tolist()

    # ------------------------------------------------------------------
    # Phase 3 — initialise zarr-vectors store
    # ------------------------------------------------------------------
    if progress:
        print(f"  [plain-ingest] initialising store: {out_store}", flush=True)

    root, lg = sk.init_skeleton_store(
        str(out_store),
        chunk_shape=chunk_shape_nm,
        bounds=store_bounds,
        ndim=ndim,
        attribute_dtypes=attribute_dtypes,
        backend=backend,
        coordinate_offset=coord_off,
    )

    # ------------------------------------------------------------------
    # Phase 4 — spatial distribute (serial, one neuron at a time)
    # ------------------------------------------------------------------
    if progress:
        print("  [plain-ingest] distributing vertices into zarr chunks…", flush=True)

    pieces_by_cc: dict[ChunkCoords, list[dict[str, Any]]] = defaultdict(list)
    cross_edges: list[tuple[int, int]] = []
    gid = 0

    for i, (seg_id, piece) in enumerate(all_pieces.items()):
        by_cc, cr, gid = pieces_from_chunk(
            {seg_id: piece},
            chunk_shape_nm=chunk_shape_nm,
            attribute_names=list(attribute_names),
            gid_start=gid,
            origin=grid_origin,
            fixed_cell=None,
        )
        for cc, ps in by_cc.items():
            pieces_by_cc[cc].extend(ps)
        cross_edges.extend(cr)

        if progress and (i + 1) % max(1, len(all_pieces) // 20) == 0:
            print(
                f"  [plain-ingest] distributed {i + 1}/{len(all_pieces)} neurons"
                f" → {len(pieces_by_cc)} chunks so far",
                flush=True,
            )

    if progress:
        print(
            f"  [plain-ingest] distribution complete: "
            f"{len(pieces_by_cc)} zarr chunks, "
            f"{len(cross_edges)} raw cross-chunk edge pairs",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Phase 5 — write zarr chunks (serial)
    # ------------------------------------------------------------------
    if progress:
        print("  [plain-ingest] writing zarr chunks…", flush=True)

    records: list[tuple[int, ChunkCoords, int]] = []
    gid_loc: dict[int, tuple[ChunkCoords, int]] = {}

    for cc, pieces in sorted(pieces_by_cc.items()):
        recs, alocs = sk.write_skeleton_chunk(
            lg, cc, pieces, attr_dtypes=np_attr_dtypes,
        )
        records.extend(recs)
        gid_loc.update(alocs)

    if progress:
        print(f"  [plain-ingest] {len(records)} fragments written", flush=True)

    # ------------------------------------------------------------------
    # Phase 6 — cross-chunk links
    # ------------------------------------------------------------------
    cc_links: list[tuple[tuple[ChunkCoords, int], tuple[ChunkCoords, int]]] = []
    for ga, gb in cross_edges:
        la = gid_loc.get(ga)
        lb = gid_loc.get(gb)
        if la is None or lb is None:
            continue
        if tuple(la[0]) == tuple(lb[0]):
            # Both endpoints ended up in the same chunk (e.g. very short edge).
            continue
        cc_links.append((la, lb))

    sk.write_skeleton_cross_chunk_links(lg, cc_links, ndim=ndim)

    if progress:
        print(f"  [plain-ingest] {len(cc_links)} cross-chunk links written", flush=True)

    # ------------------------------------------------------------------
    # Phase 7 — object index
    # ------------------------------------------------------------------
    if progress:
        print("  [plain-ingest] building object index…", flush=True)

    oid_of_seg = build_object_index(lg, records, ndim=ndim)

    sk.finalize_skeleton_store(root)

    if progress:
        print(f"  [plain-ingest] object index: {len(oid_of_seg)} objects", flush=True)

    # ------------------------------------------------------------------
    # Phase 8 — segment properties → object attributes
    # ------------------------------------------------------------------
    seg_props_raw = reader.segment_properties_raw
    if seg_props_raw:
        if progress:
            print("  [plain-ingest] writing segment properties as object attributes…",
                  flush=True)
        _write_segment_properties(
            lg,
            seg_props_raw=seg_props_raw,
            oid_of_seg=oid_of_seg,
            n_objects=len(oid_of_seg),
            progress=progress,
        )

    # ------------------------------------------------------------------
    # Phase 9 — pyramid
    # ------------------------------------------------------------------
    summary: dict[str, Any] = {
        "objects": len(oid_of_seg),
        "level0_fragments": len(records),
        "level0_cross_chunk_edges": len(cc_links),
    }

    if strides:
        if progress:
            print(
                f"  [plain-ingest] building pyramid ({len(list(strides))} levels)…",
                flush=True,
            )

        stride_list = list(strides)
        n_levels = len(stride_list)
        csf_list = list(chunk_scale_factors) if chunk_scale_factors is not None else [2] * n_levels
        spf_list = list(sparsity_factors) if sparsity_factors is not None else [1.0] * n_levels

        if pyramid_workers:
            from zarr_vectors_tools.ingest._parallel import dask_executor
            from zarr_vectors.constants import VERTICES
            from zarr_vectors.core.arrays import list_chunk_keys
            from zarr_vectors.core.store import get_resolution_level, open_store as _open

            # Use adaptive per-level worker count — same logic as run_ingest.
            root_for_counts = _open(str(out_store), mode="r")
            for i in range(n_levels):
                csf = csf_list[i]
                spf = float(spf_list[i])
                scale: tuple[int, ...]
                if isinstance(csf, (tuple, list)):
                    scale = tuple(int(s) for s in csf)
                else:
                    scale = tuple(int(csf) for _ in range(ndim))

                src_lg = get_resolution_level(root_for_counts, i)
                src_chunks = list_chunk_keys(src_lg, VERTICES)
                target_chunks = {
                    tuple(int(cc[a] // scale[a]) for a in range(ndim))
                    for cc in src_chunks
                }
                level_workers = min(int(pyramid_workers), max(1, len(target_chunks)))

                if progress:
                    print(
                        f"  [pyramid] L{i}→L{i+1}: "
                        f"target_chunks={len(target_chunks)} "
                        f"workers={level_workers}",
                        flush=True,
                    )

                if level_workers <= 1:
                    lv = coarsen_skeleton_level(
                        str(out_store), source_level=i, target_level=i + 1,
                        stride=int(stride_list[i]),
                        sparsity_factor=spf,
                        chunk_scale_factor=csf,
                        sparsity_strategy=sparsity_strategy,
                        drop_interior_below=int(drop_interior_below or 0),
                        executor=None,
                    )
                else:
                    with dask_executor(level_workers) as level_ex:
                        lv = coarsen_skeleton_level(
                            str(out_store), source_level=i, target_level=i + 1,
                            stride=int(stride_list[i]),
                            sparsity_factor=spf,
                            chunk_scale_factor=csf,
                            sparsity_strategy=sparsity_strategy,
                            drop_interior_below=int(drop_interior_below or 0),
                            executor=level_ex,
                        )
                summary.setdefault("pyramid", {"levels": [], "num_levels": 0})
                summary["pyramid"]["levels"].append(lv)
                summary["pyramid"]["num_levels"] = i + 2
        else:
            summary["pyramid"] = build_skeleton_pyramid(
                str(out_store),
                strides=stride_list,
                chunk_scale_factors=csf_list,
                sparsity_factors=spf_list,
                sparsity_strategy=sparsity_strategy,
                drop_interior_below=int(drop_interior_below or 0),
                executor=None,
            )

    return summary


# ===================================================================
# Segment-properties → object-attributes helper
# ===================================================================

def _write_segment_properties(
    level_group,
    *,
    seg_props_raw: dict[str, Any],
    oid_of_seg: dict[int, int],
    n_objects: int,
    progress: bool = True,
) -> None:
    """Convert a precomputed inline segment-properties block to object attributes.

    Writes one ``object_attributes/<prop_id>`` array per property.

    String properties (``"label"``, ``"description"``, ``"string"``) are
    stored as ``S256`` (fixed-width 256-byte) numpy arrays (UTF-8 encoded,
    zero-padded).

    Numeric properties (``"number"``) are stored with the dtype from the
    property definition.

    Tag properties (``"tags"``) are stored as a ``uint64`` bitmask where
    bit *N* corresponds to tag index *N* (up to 64 tags).

    Args:
        level_group: Level-0 zarr-vectors group.
        seg_props_raw: The ``inline`` dict from ``segment_properties/info``.
        oid_of_seg: ``{segment_id: dense_object_id}`` from ``build_object_index``.
        n_objects: Total number of objects in the store (length of OID space).
        progress: Print per-property messages.
    """
    seg_ids_ordered, props = parse_segment_properties(seg_props_raw)

    for prop_id, (ptype, values, meta) in props.items():
        arr = _build_object_attribute_array(
            ptype, values, meta,
            seg_ids_ordered=seg_ids_ordered,
            oid_of_seg=oid_of_seg,
            n_objects=n_objects,
        )
        if arr is None:
            if progress:
                print(
                    f"  [plain-ingest]  skipping property '{prop_id}' "
                    f"(type={ptype!r} not mappable)",
                    flush=True,
                )
            continue
        write_object_attributes(level_group, prop_id, arr)
        if progress:
            print(
                f"  [plain-ingest]  wrote object attribute '{prop_id}' "
                f"(type={ptype!r}, dtype={arr.dtype}, shape={arr.shape})",
                flush=True,
            )
