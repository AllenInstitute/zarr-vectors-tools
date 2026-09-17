"""Select streamlines by region: through a mask, or ending in one.

"Streamlines that pass through this ROI" is the core query of tractography.
A store answers it without reading the tractogram: the region's bounding box
names the few chunks it touches, and only those are read.  Every vertex read
is tested against the region, and so are points sampled along each segment,
so a coarse level whose simplified segments step over a small ROI still
finds the streamlines that cross it.

The one thing that reading only those chunks can miss is a segment whose two
ends both lie in chunks the region does not touch -- a segment longer than a
chunk, jumping clean over the region.  At level 0 vertices are far closer
together than a chunk; at a heavily simplified coarse level, select at a
finer level if the region is smaller than a chunk.

Regions are in world (RAS millimetre) coordinates -- a NIfTI mask with its
own affine, or a box.  A TRK store kept in TrackVis voxel millimetres is
mapped through its header's affine first, so the same mask selects the same
streamlines whichever way the store was ingested.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

_MODES = ("path", "endpoints", "both_endpoints")


def select_streamlines(
    store_path: str | Path,
    *,
    mask: Any = None,
    mask_affine: npt.ArrayLike | None = None,
    box: tuple[Sequence[float], Sequence[float]] | None = None,
    mode: str = "path",
    level: int = 0,
    sample_spacing: float | None = None,
) -> npt.NDArray[np.int64]:
    """Object ids of the streamlines that meet a region.

    Args:
        store_path: A streamline (or polyline) store.
        mask: The region as a NIfTI file path, a nibabel image, or a boolean
            (or nonzero-valued) 3-D array with ``mask_affine``.
        mask_affine: Voxel-to-RAS affine for an array ``mask``.
        box: The region as ``(min_corner, max_corner)`` in RAS millimetres,
            instead of a mask.
        mode: ``"path"`` -- any part of the streamline is inside;
            ``"endpoints"`` -- either end is; ``"both_endpoints"`` -- both are.
        level: Resolution level to select at.
        sample_spacing: Distance between the points tested along each
            segment, in millimetres.  Default: half the mask's smallest voxel
            edge, or half the box's smallest side.

    Returns:
        Sorted, unique object ids.

    Raises:
        ValueError: If neither or both of ``mask`` and ``box`` are given, the
            mode is unknown, or the store does not hold streamlines.
    """
    from zarr_vectors.building import (
        get_level_chunk_shape,
        get_resolution_level,
        open_store,
        read_level_metadata,
        read_root_metadata,
    )

    from zarr_vectors_tools.convert.export._streamlines import stored_space, trackvis_to_rasmm

    if mode not in _MODES:
        raise ValueError(f"mode must be one of {', '.join(_MODES)}; got {mode!r}")
    if (mask is None) == (box is None):
        raise ValueError("pass a region: exactly one of mask= or box=")

    root = open_store(str(store_path))
    root_meta = read_root_metadata(root)
    geometry = set(root_meta.geometry_types or [])
    if not geometry & {"streamline", "polyline"}:
        raise ValueError(
            f"{store_path} holds {sorted(geometry) or 'no geometry'}, not streamlines"
        )

    region = _Region.from_mask(mask, mask_affine) if box is None else _Region.from_box(box)
    spacing = float(sample_spacing) if sample_spacing is not None else region.default_spacing
    if spacing <= 0:
        raise ValueError(f"sample_spacing must be above 0, got {sample_spacing}")

    # Store coordinates to RAS: the identity unless the positions are still
    # TrackVis voxel millimetres.
    space, trk, _trx = stored_space(store_path)
    to_ras = trackvis_to_rasmm(trk) if space == "voxmm" and trk is not None else np.eye(4)

    level_group = get_resolution_level(root, level)
    try:
        level_meta = read_level_metadata(root, level)
    except Exception:  # noqa: BLE001 - a level without metadata uses the root shape
        level_meta = None
    chunk_shape = get_level_chunk_shape(root_meta, level_meta)
    lo, hi = _box_in_store(region.ras_bounds, np.linalg.inv(to_ras))
    chunks = _chunks_in_box(level_group, chunk_shape, lo, hi)
    if not chunks:
        return np.zeros(0, dtype=np.int64)

    touching = _objects_touching(store_path, level, chunks, region, to_ras, spacing)
    if mode == "path" or len(touching) == 0:
        return touching
    return _objects_by_endpoints(store_path, level, touching, region, to_ras, mode)


class _Region:
    """A region in RAS millimetres that can say whether points are inside."""

    def __init__(self, contains, ras_bounds, default_spacing) -> None:
        self.contains = contains
        self.ras_bounds = ras_bounds
        self.default_spacing = default_spacing

    @classmethod
    def from_box(cls, box) -> _Region:
        lo = np.asarray(box[0], dtype=np.float64).reshape(3)
        hi = np.asarray(box[1], dtype=np.float64).reshape(3)
        if np.any(hi < lo):
            raise ValueError(f"box max corner {hi.tolist()} is below its min corner {lo.tolist()}")

        def contains(points: npt.NDArray) -> npt.NDArray[np.bool_]:
            return np.all((points >= lo) & (points <= hi), axis=1)

        return cls(contains, (lo, hi), max(float(np.min(hi - lo)) / 2.0, 1e-6))

    @classmethod
    def from_mask(cls, mask, mask_affine) -> _Region:
        if isinstance(mask, (str, Path)):
            try:
                import nibabel as nib
            except ImportError as exc:
                raise ImportError(
                    "reading a NIfTI mask needs nibabel: "
                    "pip install 'zarr-vectors-tools[surfaces]'"
                ) from exc
            mask = nib.load(str(mask))
        if hasattr(mask, "affine") and hasattr(mask, "dataobj"):
            if mask_affine is not None:
                raise ValueError("mask_affine is for an array mask; an image has its own")
            mask_affine = mask.affine
            mask = np.asarray(mask.dataobj)
        if mask_affine is None:
            raise ValueError("an array mask needs mask_affine, its voxel-to-RAS affine")
        voxels = np.asarray(mask)
        if voxels.ndim == 4 and voxels.shape[3] == 1:
            voxels = voxels[..., 0]
        if voxels.ndim != 3:
            raise ValueError(f"a mask must be 3-D, got shape {voxels.shape}")
        inside = voxels != 0
        affine = np.asarray(mask_affine, dtype=np.float64).reshape(4, 4)
        to_voxel = np.linalg.inv(affine)
        shape = np.asarray(inside.shape)

        def contains(points: npt.NDArray) -> npt.NDArray[np.bool_]:
            ijk = np.rint(points @ to_voxel[:3, :3].T + to_voxel[:3, 3]).astype(np.int64)
            valid = np.all((ijk >= 0) & (ijk < shape), axis=1)
            hit = np.zeros(len(points), dtype=bool)
            hit[valid] = inside[tuple(ijk[valid].T)]
            return hit

        occupied = np.argwhere(inside)
        if len(occupied) == 0:
            bounds = (np.full(3, np.inf), np.full(3, -np.inf))
        else:
            # Voxel centres +/- half a voxel, all eight corners mapped to RAS.
            corners = np.array(np.meshgrid(*[
                [occupied[:, a].min() - 0.5, occupied[:, a].max() + 0.5] for a in range(3)
            ], indexing="ij")).reshape(3, -1).T
            ras = corners @ affine[:3, :3].T + affine[:3, 3]
            bounds = (ras.min(axis=0), ras.max(axis=0))
        voxel_size = np.linalg.norm(affine[:3, :3], axis=0)
        return cls(contains, bounds, float(voxel_size.min()) / 2.0)


def _box_in_store(ras_bounds, to_store: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    lo, hi = ras_bounds
    if not np.all(np.isfinite(lo)):
        return np.zeros(3), -np.ones(3)
    corners = np.array(np.meshgrid(*[[lo[a], hi[a]] for a in range(3)], indexing="ij"))
    corners = corners.reshape(3, -1).T
    store = corners @ to_store[:3, :3].T + to_store[:3, 3]
    return store.min(axis=0), store.max(axis=0)


def _chunks_in_box(level_group, chunk_shape, lo, hi) -> list[tuple[int, ...]]:
    """The level's chunks that the store-space box ``lo``..``hi`` touches.

    Chunk ``c`` spans ``c * chunk_shape`` to ``(c + 1) * chunk_shape`` from
    the store origin, as core bins them.
    """
    from zarr_vectors.building import list_chunk_keys
    from zarr_vectors.constants import VERTICES

    if np.any(hi < lo):
        return []
    cs = np.asarray(chunk_shape, dtype=np.float64)
    first = np.floor(np.asarray(lo) / cs).astype(np.int64)
    last = np.floor(np.asarray(hi) / cs).astype(np.int64)
    return sorted(
        tuple(int(c) for c in key)
        for key in list_chunk_keys(level_group, VERTICES)
        if np.all(np.asarray(key[-3:]) >= first) and np.all(np.asarray(key[-3:]) <= last)
    )


def _to_ras(points: npt.NDArray, to_ras: npt.NDArray) -> npt.NDArray:
    return points.astype(np.float64) @ to_ras[:3, :3].T + to_ras[:3, 3]


def _objects_touching(store_path, level, chunks, region, to_ras, spacing) -> npt.NDArray[np.int64]:
    """Objects with a vertex, or a sampled point on a segment, inside the region.

    Reads only ``chunks``.  The read crops streamlines at the chunk set's
    edge, so a segment leaving it is not sampled; every vertex inside the
    region is still read, because the chunk holding it touches the region.
    """
    from zarr_vectors.types.polylines import read_polylines

    result = read_polylines(str(store_path), level=level, chunks=list(chunks))
    points: list[npt.NDArray] = []
    owners: list[npt.NDArray] = []
    for oid, segments in zip(result["object_ids"], result["polylines"]):
        run = _to_ras(np.concatenate(segments, axis=0), to_ras)
        sampled = _along_segments(run, spacing)
        points.append(sampled)
        owners.append(np.full(len(sampled), int(oid), dtype=np.int64))
    if not points:
        return np.zeros(0, dtype=np.int64)
    everything = np.concatenate(points, axis=0)
    hit = region.contains(everything)
    return np.unique(np.concatenate(owners)[hit])


def _along_segments(run: npt.NDArray, spacing: float) -> npt.NDArray:
    """A run's vertices plus points every ``spacing`` along each segment."""
    if len(run) < 2:
        return run
    starts, steps = run[:-1], np.diff(run, axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    counts = np.maximum(np.ceil(lengths / spacing).astype(np.int64) - 1, 0)
    if counts.sum() == 0:
        return run
    which = np.repeat(np.arange(len(starts)), counts)
    offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    fraction = (offsets + 1) / (counts[which] + 1)
    interior = starts[which] + steps[which] * fraction[:, None]
    return np.concatenate([run, interior], axis=0)


def _objects_by_endpoints(store_path, level, candidates, region, to_ras, mode):
    """Of ``candidates``, those with one (or both) ends inside the region.

    An end inside the region is a vertex inside it, so every streamline that
    qualifies is already a candidate; only the candidates are read whole,
    to find their ends.
    """
    from zarr_vectors.types.polylines import read_polylines

    result = read_polylines(str(store_path), level=level, object_ids=candidates.tolist())
    starts, ends, ids = [], [], []
    for oid, segments in zip(result["object_ids"], result["polylines"]):
        whole = np.concatenate(segments, axis=0)
        starts.append(whole[0])
        ends.append(whole[-1])
        ids.append(int(oid))
    if not ids:
        return np.zeros(0, dtype=np.int64)
    at_start = region.contains(_to_ras(np.asarray(starts), to_ras))
    at_end = region.contains(_to_ras(np.asarray(ends), to_ras))
    keep = at_start | at_end if mode == "endpoints" else at_start & at_end
    return np.unique(np.asarray(ids, dtype=np.int64)[keep])
