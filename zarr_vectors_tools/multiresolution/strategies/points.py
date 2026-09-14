"""Point cloud coarsening strategy for multi-resolution pyramids.

Point clouds are the simplest case: metanodes are centroids of vertices
within each spatial bin.  Attributes are aggregated (mean, sum, or first).
Object identity is preserved — if points carry object IDs, each metanode
inherits the majority object ID of its children.

This strategy also supports **density-preserving** mode: instead of
centroids, it selects the vertex closest to the centroid (medoid),
which preserves the original coordinate precision.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from zarr_vectors.building import (
    LevelMetadata,
    assign_chunks,
    create_attribute_array,
    create_resolution_level,
    create_vertices_array,
    get_resolution_level,
    open_store,
    read_root_metadata,
    write_chunk_attributes,
    write_chunk_vertices,
)
from zarr_vectors.constants import VERTICES
from zarr_vectors.exceptions import CoarseningError
from zarr_vectors.types.points import read_points

from zarr_vectors_tools.multiresolution.metanodes import generate_metanodes


def coarsen_points(
    positions: npt.NDArray[np.floating],
    bin_size: float | tuple[float, ...],
    *,
    attributes: dict[str, npt.NDArray] | None = None,
    object_ids: npt.NDArray[np.integer] | None = None,
    agg_mode: str = "mean",
    use_medoid: bool = False,
) -> dict[str, Any]:
    """Coarsen a point cloud to a lower resolution.

    Args:
        positions: ``(N, D)`` vertex positions.
        bin_size: Spatial bin edge length.
        attributes: Per-vertex attributes to aggregate.
        object_ids: ``(N,)`` per-vertex object IDs.  If provided,
            each metanode inherits the majority object ID.
        agg_mode: Attribute aggregation mode.
        use_medoid: If True, select the closest-to-centroid vertex
            instead of using the centroid itself.

    Returns:
        Dict with:
        - ``positions``: ``(M, D)`` coarsened positions
        - ``attributes``: ``{name: array}`` aggregated attributes
        - ``object_ids``: ``(M,)`` majority object IDs (if input had them)
        - ``children``: list of M arrays of parent vertex indices
        - ``vertex_count``: M
        - ``reduction_ratio``: N / M
    """
    n_input = len(positions)

    result = generate_metanodes(
        positions, bin_size,
        attributes=attributes,
        agg_mode=agg_mode,
    )

    meta_positions = result["metanode_positions"]
    children = result["children"]
    meta_attrs = result["metanode_attributes"]
    n_meta = len(meta_positions)

    # Medoid: replace centroids with closest original vertex
    if use_medoid and n_meta > 0:
        for i in range(n_meta):
            child_positions = positions[children[i]]
            centroid = meta_positions[i]
            dists = np.sum((child_positions - centroid) ** 2, axis=1)
            closest = np.argmin(dists)
            meta_positions[i] = child_positions[closest]

    # Majority object ID per metanode
    meta_object_ids: npt.NDArray | None = None
    if object_ids is not None:
        meta_object_ids = np.empty(n_meta, dtype=object_ids.dtype)
        for i in range(n_meta):
            child_oids = object_ids[children[i]]
            # Majority vote
            unique_oids, counts = np.unique(child_oids, return_counts=True)
            meta_object_ids[i] = unique_oids[np.argmax(counts)]

    out: dict[str, Any] = {
        "positions": meta_positions,
        "vertex_attributes": meta_attrs,
        "children": children,
        "vertex_count": n_meta,
        "reduction_ratio": n_input / max(n_meta, 1),
    }
    if meta_object_ids is not None:
        out["object_ids"] = meta_object_ids

    return out


def coarsen_points_store(
    store_path: str,
    target_level: int,
    bin_size: float | tuple[float, ...],
    *,
    source_level: int = 0,
    agg_mode: str = "mean",
    use_medoid: bool = False,
) -> dict[str, Any]:
    """Coarsen a point cloud store and write the result as a new level.

    Reads positions from ``source_level``, coarsens, and writes to
    ``target_level`` in the same store.

    Args:
        store_path: Path to the zarr vectors store.
        target_level: Resolution level to write (must not exist).
        bin_size: Spatial bin size for coarsening.
        source_level: Resolution level to read from.
        agg_mode: Attribute aggregation mode.
        use_medoid: Use medoid instead of centroid.

    Returns:
        Summary dict.
    """
    # Read source level
    source_data = read_points(str(store_path), level=source_level)
    positions = source_data["positions"]
    n_source = len(positions)

    if n_source == 0:
        return {"vertex_count": 0, "reduction_ratio": 0}

    # Coarsen
    coarsened = coarsen_points(
        positions, bin_size,
        agg_mode=agg_mode,
        use_medoid=use_medoid,
    )

    meta_positions = coarsened["positions"]
    n_meta = coarsened["vertex_count"]
    ndim = meta_positions.shape[1]

    # Open store and write new level
    root = open_store(store_path, mode="r+")
    root_meta = read_root_metadata(root)
    chunk_shape = root_meta.chunk_shape

    bin_tuple = (
        tuple(bin_size for _ in range(ndim))
        if isinstance(bin_size, (int, float))
        else tuple(bin_size)
    )
    level_meta = LevelMetadata(
        level=target_level,
        vertex_count=n_meta,
        arrays_present=[VERTICES],
        bin_shape=bin_tuple,
        coarsening_method="point_cloud_grid" + ("_medoid" if use_medoid else ""),
        parent_level=source_level,
    )
    level_group = create_resolution_level(root, target_level, level_meta)
    create_vertices_array(level_group, dtype="float32")

    # Assign to chunks and write
    chunk_assignments = assign_chunks(meta_positions, chunk_shape)
    for chunk_coords, global_indices in sorted(chunk_assignments.items()):
        chunk_verts = meta_positions[global_indices]
        write_chunk_vertices(
            level_group, chunk_coords, [chunk_verts], dtype=np.float32
        )

    return {
        "vertex_count": n_meta,
        "reduction_ratio": coarsened["reduction_ratio"],
        "source_count": n_source,
    }


# ===================================================================
# Random-subset pyramid (attribute-preserving)
# ===================================================================

def build_point_subset_pyramid(
    store_path: str,
    *,
    levels: int = 5,
    divisor: float = 8.0,
    seed: int = 0,
    source_level: int = 0,
    attributes: bool = True,
    attribute_names: list[str] | None = None,
    attribute_batch: int = 100,
    shard_shape: int | tuple[int, ...] | None = None,
    resume: bool = False,
    progress: bool = False,
) -> dict[str, Any]:
    """Build coarser levels, each a random subset of the level above.

    The general :func:`~zarr_vectors_tools.multiresolution.coarsen.build_pyramid`
    does not fit an object-less point cloud: its sparsity factor drops
    *objects*, so a store whose points carry no object IDs is treated as a
    single object and every level comes out identical. Its coarseners also
    write ``vertices`` (and object attributes) only — per-vertex attributes
    are dropped, which for a cell atlas means the coarse levels cannot be
    coloured by gene or metadata.

    This builder addresses both: each level keeps a uniformly random
    ``1/divisor`` of its parent's points, and carries every per-vertex
    attribute across, so any level can be rendered and coloured exactly
    like level 0.

    Sampling is drawn from a seeded generator, so the same ``seed`` gives
    the same pyramid.  Levels nest — level *n+1* is a subset of level
    *n* — which keeps a point's appearance/disappearance monotonic as a
    viewer changes zoom.

    Args:
        store_path: Store to add levels to (modified in place).
        levels: How many coarser levels to create.
        divisor: Keep ``1/divisor`` of the parent's points per level.
        seed: Seed for the subset draw.
        source_level: Level to coarsen from.
        attributes: Carry per-vertex attributes across. Turning this off
            makes the build dramatically faster but leaves the coarse
            levels geometry-only.
        attribute_names: Attributes to carry (default: all at
            ``source_level``).
        attribute_batch: Attributes read per pass. Bounds peak memory to
            roughly ``batch x source_points x 4`` bytes.
        shard_shape: Shard the new levels' arrays (chunks per axis per
            file). Strongly recommended — an unsharded level costs one
            file per chunk per attribute.
        resume: Keep levels that already exist and fill in only the
            attributes that are missing or partially written. The subsets
            are a pure function of ``seed``/``divisor``, so a resumed run
            reproduces them exactly; the level's point count is checked
            against that reproduction before anything is written. Without
            this, an existing level is an error.
        progress: Print per-level and per-batch progress.

    Returns:
        Summary dict with ``levels_created`` and a ``level_specs`` list of
        per-level ``{level, vertex_count, chunk_count}``.
    """
    import zarr_vectors.building as B
    from zarr_vectors.constants import VERTEX_ATTRIBUTES

    root = open_store(str(store_path), "r+")
    root_meta = read_root_metadata(root)
    src_group = get_resolution_level(root, source_level)
    src_meta = B.read_level_metadata(root, source_level)
    chunk_shape = B.get_level_chunk_shape(root_meta, src_meta)

    available = sorted(src_group.require_group(VERTEX_ATTRIBUTES).children())
    carried = (
        [a for a in (attribute_names if attribute_names is not None else available)]
        if attributes else []
    )
    unknown = [a for a in carried if a not in available]
    if unknown:
        raise CoarseningError(f"attributes not at level {source_level}: {unknown}")

    # Reference read: positions define the row order every later attribute
    # read must agree with.  A key attribute, when present, lets that
    # agreement be asserted rather than assumed.
    key_attr = "zv_join_key" if "zv_join_key" in available else None
    reference = read_points(
        str(store_path), level=source_level,
        attribute_names=[key_attr] if key_attr else None,
    )
    positions = np.asarray(reference["positions"])
    reference_keys = (
        np.asarray(reference["vertex_attributes"][key_attr]) if key_attr else None
    )
    n_source = len(positions)
    if progress:
        print(f"source level {source_level}: {n_source} points, "
              f"{len(carried)} attribute(s) to carry", flush=True)

    # ---- choose the nested subsets, then write geometry ------------------
    rng = np.random.default_rng(seed)
    kept = np.arange(n_source, dtype=np.int64)
    per_level: list[dict[str, Any]] = []
    existing_levels = set(B.list_resolution_levels(root))

    for step in range(1, levels + 1):
        target = source_level + step
        size = int(len(kept) // divisor)
        if size < 1:
            if progress:
                print(f"  level {target}: parent has {len(kept)} points; "
                      f"stopping (a 1/{divisor:g} subset would be empty)", flush=True)
            break
        kept = np.sort(rng.choice(kept, size=size, replace=False))

        level_positions = positions[kept]
        assignments = assign_chunks(level_positions, chunk_shape)

        if target in existing_levels:
            # The subsets are a pure function of (seed, divisor, source
            # order), so a resumed run recomputes exactly the same ``kept``
            # without touching geometry.  Cross-check the count anyway: a
            # mismatch means the level came from different parameters, and
            # writing attributes against it would misalign every value.
            if not resume:
                raise CoarseningError(
                    f"level {target} already exists; pass resume=True to keep "
                    f"the existing levels and fill in missing attributes"
                )
            existing_meta = B.read_level_metadata(root, target)
            if existing_meta.vertex_count != len(kept):
                raise CoarseningError(
                    f"level {target} holds {existing_meta.vertex_count} points "
                    f"but seed={seed}/divisor={divisor:g} reproduces "
                    f"{len(kept)}; refusing to resume against a level built "
                    f"with different parameters"
                )
            level_group = get_resolution_level(root, target)
            if progress:
                print(f"  level {target}: reusing existing "
                      f"{len(kept)} points", flush=True)
        else:
            level_meta = LevelMetadata(
                level=target,
                vertex_count=len(kept),
                arrays_present=[VERTICES] + ([VERTEX_ATTRIBUTES] if carried else []),
                # Subsetting removes points; it does not move or bin the ones
                # it keeps, so the level occupies level 0's coordinate space
                # unchanged.  bin_ratio becomes the NGFF scale transform, so it
                # stays 1 here — unlike a binning coarsener, where the ratio
                # records how far the grid was rescaled.
                bin_shape=tuple(root_meta.effective_bin_shape),
                bin_ratio=tuple(1 for _ in root_meta.effective_bin_shape),
                coarsening_method="random_subset",
                parent_level=target - 1,
                object_sparsity=float(1.0 / divisor),
            )
            level_group = create_resolution_level(root, target, level_meta)

            with _subset_session(B, level_group, root_meta, chunk_shape, shard_shape):
                create_vertices_array(level_group, dtype="float32")
                for chunk_coords, indices in sorted(assignments.items()):
                    write_chunk_vertices(
                        level_group, chunk_coords,
                        [level_positions[indices].astype(np.float32)],
                        dtype=np.float32,
                    )
            if progress:
                print(f"  level {target}: {len(kept)} points in "
                      f"{len(assignments)} chunks", flush=True)

        per_level.append({
            "level": target,
            "vertex_count": len(kept),
            "chunk_count": len(assignments),
            "kept": kept,
            "assignments": assignments,
            "group": level_group,
        })

    # A resumed run only redoes attributes that are missing or partial.
    # "Partial" matters as much as "missing": an interrupted write leaves an
    # array whose chunk set is a strict subset of the level's, and reads that
    # request it silently return fewer points rather than failing.
    if resume and carried:
        before = len(carried)
        carried = [
            name for name in carried
            if not all(_attribute_complete(B, spec["group"], name) for spec in per_level)
        ]
        if progress:
            print(f"  resume: {before - len(carried)} attribute(s) already "
                  f"complete at every level, {len(carried)} to carry", flush=True)

    # ---- carry attributes, one batch of names across all levels ----------
    for start in range(0, len(carried), attribute_batch):
        batch = carried[start:start + attribute_batch]
        # Subsets are index-based, so every batch must come back in the
        # same row order as the reference read. Carry the key attribute in
        # each request and check it, rather than assuming the order is
        # stable across a dozen independent reads.
        request = batch if key_attr in batch or key_attr is None else [*batch, key_attr]
        block = read_points(
            str(store_path), level=source_level, attribute_names=request,
        )
        values = block["vertex_attributes"]

        if reference_keys is not None:
            got = np.asarray(values.get(key_attr))
            if got is None or not np.array_equal(got, reference_keys):
                raise CoarseningError(
                    "attribute read returned a different row order than the "
                    "reference read; cannot align subsets safely"
                )

        for spec in per_level:
            with _subset_session(
                B, spec["group"], root_meta, chunk_shape, shard_shape
            ):
                for name in batch:
                    data = np.asarray(values[name])[spec["kept"]]
                    create_attribute_array(
                        spec["group"], name, dtype=str(data.dtype),
                        channel_names=(
                            [f"ch{i}" for i in range(data.shape[1])]
                            if data.ndim == 2 else None
                        ),
                        exist_ok=True,
                    )
                    for chunk_coords, indices in sorted(spec["assignments"].items()):
                        write_chunk_attributes(
                            spec["group"], name, chunk_coords,
                            [data[indices]], dtype=data.dtype,
                        )
        if progress:
            print(f"  carried attributes {min(start + attribute_batch, len(carried))}"
                  f"/{len(carried)}", flush=True)

    for spec in per_level:
        B.refresh_arrays_present(spec["group"])
        spec.pop("kept", None)
        spec.pop("assignments", None)
        spec.pop("group", None)

    return {
        "levels_created": len(per_level),
        "level_specs": per_level,
        "attributes_carried": len(carried),
        "divisor": divisor,
        "seed": seed,
    }


def _attribute_complete(B, level_group, name: str) -> bool:
    """True when ``name`` covers every chunk the level's vertices occupy.

    A partially written attribute is worse than a missing one: reads that
    request it come back short rather than raising, so completeness is
    checked against the vertices grid rather than mere existence.
    """
    try:
        expected = set(B.list_chunk_keys(level_group, VERTICES))
        return set(B.list_chunk_keys(level_group, f"vertex_attributes/{name}")) == expected
    except Exception:
        return False


def _subset_session(B, level_group, root_meta, chunk_shape, shard_shape):
    """Sharded write session for a new level, or a no-op when unsharded."""
    from contextlib import nullcontext

    if shard_shape is None:
        return nullcontext()
    return B.open_write_session(
        level_group,
        shard_shape=shard_shape,
        bounds=(list(root_meta.bounds[0]), list(root_meta.bounds[1])),
        chunk_shape=chunk_shape,
    )
