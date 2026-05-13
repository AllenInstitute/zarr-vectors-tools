"""Chunked graph and mesh algorithms for zarr-vectors stores.

This sub-package implements algorithms that operate directly on the
chunked storage layout — streaming, halo, frontier, and pyramid patterns
— rather than materialising the full geometry into memory.

Modules prefixed with an underscore are private helpers that work around
core-package gaps; they will retire (or re-point at core) as the
corresponding additions land in zarr-vectors-py.
"""

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

__all__ = [
    "bfs_distances",
    "cast_ray",
    "closest_point",
    "compute_connected_components",
    "compute_k_core",
    "compute_label_propagation",
    "compute_louvain",
    "compute_mean_curvature",
    "compute_mesh_summary",
    "compute_vertex_normals",
    "shortest_path",
]
