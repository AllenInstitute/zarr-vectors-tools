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

from zarr_vectors_tools.multiresolution.metanodes import generate_metanodes
from zarr_vectors.types.points import read_points, write_points
from zarr_vectors.core.store import (
    create_resolution_level,
    get_resolution_level,
    open_store,
    read_root_metadata,
)
from zarr_vectors.core.metadata import LevelMetadata
from zarr_vectors.core.arrays import (
    create_vertices_array,
    create_object_index_array,
    create_attribute_array,
    list_chunk_keys,
    read_chunk_vertices,
    write_chunk_vertices,
    write_chunk_attributes,
    write_object_index,
)
from zarr_vectors.spatial.chunking import assign_chunks
from zarr_vectors.constants import VERTICES


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
    ndim = positions.shape[1]

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
