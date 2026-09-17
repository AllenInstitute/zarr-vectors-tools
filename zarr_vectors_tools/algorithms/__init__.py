"""Chunked graph and mesh algorithms for zarr-vectors stores.

This sub-package implements algorithms that operate directly on the
chunked storage layout — streaming, halo, frontier, and pyramid patterns
— rather than materialising the full geometry into memory.
"""

from zarr_vectors_tools.algorithms.bundles import bundle_summary, read_bundle_summary
from zarr_vectors_tools.algorithms.graph_clustering import (
    compute_k_core,
    compute_label_propagation,
    compute_louvain,
)
from zarr_vectors_tools.algorithms.graph_components import (
    compute_connected_components,
)
from zarr_vectors_tools.algorithms.graph_search import bfs_distances, shortest_path
from zarr_vectors_tools.algorithms.mesh_attributes import (
    compute_mean_curvature,
    compute_vertex_normals,
)
from zarr_vectors_tools.algorithms.mesh_query import cast_ray, closest_point
from zarr_vectors_tools.algorithms.mesh_summary import compute_mesh_summary
from zarr_vectors_tools.algorithms.parcels import parcel_at, parcel_summary
from zarr_vectors_tools.algorithms.segment_link import SegmentLink
from zarr_vectors_tools.algorithms.skeleton_metrics import compute_skeleton_metrics
from zarr_vectors_tools.algorithms.streamline_select import select_streamlines
from zarr_vectors_tools.algorithms.surfaces import read_hemisphere

__all__ = [
    "SegmentLink",
    "bfs_distances",
    "bundle_summary",
    "cast_ray",
    "closest_point",
    "compute_connected_components",
    "compute_k_core",
    "compute_label_propagation",
    "compute_louvain",
    "compute_mean_curvature",
    "compute_mesh_summary",
    "compute_skeleton_metrics",
    "compute_vertex_normals",
    "parcel_at",
    "parcel_summary",
    "read_bundle_summary",
    "read_hemisphere",
    "select_streamlines",
    "shortest_path",
]
