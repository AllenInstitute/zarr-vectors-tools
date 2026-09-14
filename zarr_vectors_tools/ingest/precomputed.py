"""Neuroglancer precomputed skeleton layer → zarr-vectors store, in one call.

The two precomputed ingesters take a reader object and their own option
names, which is right for scripts that tune a FlyWire-scale run and wrong for
``zvtools convert``, whose registry calls every ingester the same way:
``func(source, out_store, chunk_shape, **options)``.  This module is that call.

Which ingester runs is decided by the layer's ``info``, not by the caller:

* a ``spatial_index`` block → :func:`.precomputed_skeletons.run_ingest`, one
  ``.frags`` chunk at a time.  Level-0 chunks are the spatial index's own, so
  ``chunk_shape`` is not needed.  Without ``anchor`` every ``.frags`` file in
  the layer is listed and ingested; with it, just the block it names.
* no spatial index → :func:`.precomputed_plain_skeletons.run_ingest_plain`,
  one read per segment.  There is no source grid to inherit, so
  ``chunk_shape`` (in the layer's units, nanometres) is required.

A layer is a directory or a bucket prefix, never a single file.  ``source``
may be a URL (``gs://``, ``s3://``, ``https://``, ``file://``, optionally
behind Neuroglancer's ``precomputed://``) or a local directory.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

_NEUROGLANCER_SCHEME = "precomputed://"
_SKELETON_TYPE = "neuroglancer_skeletons"


def is_url(source: str | Path) -> bool:
    """Does ``source`` name a remote (or ``file://``) location, not a local path?"""
    return "://" in str(source)


def layer_url(source: str | Path) -> str:
    """The CloudFiles / CloudVolume path for ``source``.

    Neuroglancer's ``precomputed://`` prefix is dropped (CloudFiles does not
    take it), and a local directory becomes ``file://`` plus its absolute
    POSIX path -- the spelling both libraries accept on Windows as well.
    """
    text = str(source)
    if text.startswith(_NEUROGLANCER_SCHEME):
        text = text[len(_NEUROGLANCER_SCHEME):]
    if is_url(text):
        return text.rstrip("/")
    return "file://" + Path(text).resolve().as_posix()


def read_layer_info(url: str) -> dict[str, Any]:
    """Read and check a layer's ``info``; refuse anything but a skeleton layer."""
    from cloudfiles import CloudFiles

    info = CloudFiles(url).get_json("info")
    if info is None:
        raise FileNotFoundError(f"no precomputed info file at {url}/info")

    kind = info.get("@type")
    is_volume = kind is None and "scales" in info
    if (kind is not None and kind != _SKELETON_TYPE) or is_volume:
        described = kind or "volume"
        if info.get("skeletons"):
            raise ValueError(
                f"{url} is a precomputed {described} layer, not a skeleton "
                f"layer; its skeletons are at {url}/{info['skeletons']}"
            )
        raise ValueError(
            f"{url} is a precomputed {described} layer; only skeleton layers "
            f"({_SKELETON_TYPE}) can be converted"
        )
    return info


def has_spatial_index(info: dict[str, Any]) -> bool:
    return bool(info.get("spatial_index"))


def list_frag_keys(url: str, frags_dir: str = "") -> list[str]:
    """Every ``.frags`` key in ``frags_dir`` (non-recursive), sorted."""
    from cloudfiles import CloudFiles

    prefix = f"{frags_dir.strip('/')}/" if frags_dir else ""
    names = set()
    for entry in CloudFiles(url).list(prefix=prefix, flat=True):
        # CloudFiles returns local listings with OS separators.
        name = entry.replace("\\", "/").rsplit("/", 1)[-1]
        if name.endswith(".frags"):
            names.add(name)
    return [prefix + name for name in sorted(names)]


def frag_bounds(info: Any, keys: Sequence[str]) -> tuple[list[float], list[float]]:
    """The nanometre box covering ``keys``, snapped to the ``.frags`` grid.

    ``run_ingest`` aligns the store's chunk grid to the box's minimum corner,
    so every key has to lie on the grid through it; one that does not would
    be written into the wrong chunk, and is refused.
    """
    from zarr_vectors_tools.ingest.precomputed_skeletons import parse_frag_key

    starts = np.array([parse_frag_key(k) for k in keys], dtype=np.int64)
    step = np.asarray(info.chunk_size_voxels, dtype=np.int64)
    res = np.asarray(info.resolution_nm, dtype=np.float64)
    lo = starts.min(axis=0)
    hi = starts.max(axis=0) + step
    off_grid = np.any((starts - lo) % step != 0, axis=1)
    if off_grid.any():
        bad = [k for k, b in zip(keys, off_grid) if b][:3]
        raise ValueError(
            f"these .frags files are not on the {tuple(step.tolist())}-voxel "
            f"chunk grid the others share: {', '.join(bad)}"
        )
    return (lo * res).tolist(), (hi * res).tolist()


def ingest_precomputed(
    source: str | Path,
    out_store: str | Path,
    chunk_shape: Sequence[float] | None = None,
    *,
    frags_dir: str = "",
    anchor: Sequence[int] | None = None,
    counts: Sequence[int] | None = None,
    segment_ids: Sequence[int] | None = None,
    strides: Sequence[int] = (),
    chunk_scale_factors: Sequence[int | tuple[int, ...]] | None = None,
    sparsity_factors: Sequence[float] | None = None,
    sparsity_strategy: str = "length",
    drop_interior_below: int = 0,
    workers: int | None = None,
    executor: Any = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Ingest a precomputed skeleton layer, choosing the path from its ``info``.

    Args:
        source: Layer URL or local directory.
        out_store: Output store path.
        chunk_shape: Level-0 chunk size in nm.  Required for a layer with no
            spatial index; for one with, it may only repeat the index's own.
        frags_dir: Subdirectory holding the ``.frags`` files (spatial index).
        anchor: Voxel corner of one ``.frags`` chunk; ingest the block of
            ``counts`` chunks from there instead of listing the layer.
        counts: Chunks per axis from ``anchor`` (default ``(1, 1, 1)``).
        segment_ids: Only these segments (no spatial index).
        strides: Per-level decimation strides; empty builds level 0 only.
        chunk_scale_factors / sparsity_factors / sparsity_strategy /
            drop_interior_below: As for the underlying ingester.
        workers: Dask worker processes, used when ``executor`` is ``None``.
        executor: A ``map``-like ``(func, items, shared)`` callable.
        progress: Print progress.

    Returns:
        The ingester's summary, plus ``layer`` (``"spatial_index"`` or
        ``"plain"``) and ``bounds_nm`` when known.
    """
    url = layer_url(source)
    info = read_layer_info(url)

    if has_spatial_index(info):
        if segment_ids:
            raise ValueError(
                f"{url} has a spatial index, so it is read by chunk, not by "
                f"segment; segment_ids selects from layers without one"
            )
        return _ingest_frags(
            url, out_store, chunk_shape,
            frags_dir=frags_dir, anchor=anchor, counts=counts,
            strides=strides, chunk_scale_factors=chunk_scale_factors,
            sparsity_factors=sparsity_factors,
            sparsity_strategy=sparsity_strategy,
            drop_interior_below=drop_interior_below,
            workers=workers, executor=executor, progress=progress,
        )

    given = [
        name for name, value in (
            ("frags_dir", frags_dir), ("anchor", anchor), ("counts", counts),
        ) if value
    ]
    if given:
        raise ValueError(
            f"{url} has no spatial index, so it has no .frags chunks for "
            f"{', '.join(given)} to select"
        )
    if chunk_shape is None:
        raise ValueError(
            f"{url} has no spatial index, so there is no source chunk grid to "
            f"inherit; pass chunk_shape (--chunk-shape), in nanometres"
        )
    from zarr_vectors_tools.ingest.precomputed_plain_skeletons import (
        PlainPrecomputedReader,
        run_ingest_plain,
    )

    summary = run_ingest_plain(
        PlainPrecomputedReader(url), out_store,
        chunk_shape_nm=tuple(float(c) for c in chunk_shape),
        seg_ids=[int(s) for s in segment_ids] if segment_ids else None,
        strides=list(strides),
        chunk_scale_factors=chunk_scale_factors,
        sparsity_factors=sparsity_factors,
        sparsity_strategy=sparsity_strategy,
        drop_interior_below=drop_interior_below,
        pyramid_workers=workers if executor is None else None,
        executor=executor,
        progress=progress,
    )
    summary["layer"] = "plain"
    return summary


def _ingest_frags(
    url: str,
    out_store: str | Path,
    chunk_shape: Sequence[float] | None,
    *,
    frags_dir: str,
    anchor: Sequence[int] | None,
    counts: Sequence[int] | None,
    progress: bool,
    **options: Any,
) -> dict[str, Any]:
    from zarr_vectors_tools.ingest.precomputed_skeletons import (
        PrecomputedFragsReader,
        enumerate_frag_keys,
        run_ingest,
    )

    reader = PrecomputedFragsReader(url, frags_dir=frags_dir)
    info = reader.info

    if chunk_shape is not None and not np.allclose(
        np.asarray(chunk_shape, dtype=np.float64), info.chunk_size_nm,
    ):
        raise ValueError(
            f"{url} has a spatial index, so level-0 chunks are its own "
            f"{tuple(info.chunk_size_nm)} nm, one per .frags file; leave "
            f"chunk_shape (--chunk-shape) out"
        )

    if anchor is None:
        if counts is not None:
            raise ValueError("counts needs an anchor to count from")
        keys = list_frag_keys(url, frags_dir)
        if not keys:
            where = f"{url}/{frags_dir}" if frags_dir else url
            raise FileNotFoundError(
                f"no .frags files in {where}; if they are in a subdirectory, "
                f"name it with frags_dir (--frags-dir)"
            )
        if progress:
            print(f"  [precomputed] listed {len(keys)} .frags chunks", flush=True)
        bounds = frag_bounds(info, keys)
    else:
        counts = tuple(counts) if counts is not None else (1, 1, 1)
        if len(anchor) != 3 or len(counts) != 3:
            raise ValueError("anchor and counts each take three values, x,y,z")
        keys = enumerate_frag_keys(info, tuple(anchor), counts)
        res = np.asarray(info.resolution_nm, dtype=np.float64)
        step = np.asarray(info.chunk_size_voxels, dtype=np.int64)
        lo = np.asarray(anchor, dtype=np.int64)
        hi = lo + np.asarray(counts, dtype=np.int64) * step
        bounds = ((lo * res).tolist(), (hi * res).tolist())

    summary = run_ingest(
        reader, out_store, keys, bounds_nm=bounds, progress=progress, **options,
    )
    summary["layer"] = "spatial_index"
    summary["bounds_nm"] = bounds
    return summary
