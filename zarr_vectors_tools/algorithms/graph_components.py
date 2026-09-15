"""Connected components for graph stores.

The level's link family is read once as arrays, and the components are
found by a union-find over the whole edge list at a time (see
:func:`~zarr_vectors_tools.algorithms._graph_edges.component_roots`), so
the cost is a handful of numpy passes over the edges rather than a Python
call per edge.  Memory holds the edge list: two ``int64`` per edge plus the
reader's own arrays.

``write_back=True`` persists the component labels via
:func:`~zarr_vectors_tools._attributes.write_vertex_attribute` under
``attributes/component_label/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from zarr_vectors.building import get_resolution_level, open_store

from zarr_vectors_tools._attributes import write_vertex_attribute
from zarr_vectors_tools.algorithms._graph_edges import (
    compact_labels,
    component_roots,
    read_edges,
)


def compute_connected_components(
    store_path: str | Path,
    *,
    level: int = 0,
    write_back: bool = False,
) -> dict[str, Any]:
    """Compute 0-indexed connected-component labels for a graph store.

    Components are numbered in order of their lowest vertex, so vertex 0 is
    always in component 0 and the labels do not depend on the order edges
    are stored in.

    Args:
        store_path: Path to a zarr-vectors graph (or skeleton) store.
        level: Resolution level to operate on.
        write_back: When True, persist the labels under
            ``attributes/component_label/`` via
            :func:`~zarr_vectors_tools._attributes.write_vertex_attribute`.

    Returns:
        Dict with:
          - ``labels``: ``(N,) uint32``, component label per node in
            global ordering (matches ``read_graph``'s position order).
          - ``n_components`` (int)
          - ``largest_component_size`` (int)
          - ``component_sizes`` (np.ndarray): count of nodes per label,
            indexed by component id.
    """
    level_group = get_resolution_level(open_store(str(store_path)), level)
    a, b, _weights, n_vertices = read_edges(level_group)
    labels, sizes = compact_labels(component_roots(n_vertices, a, b))

    if write_back and n_vertices:
        # A second, writable handle: the read path above opens mode="r".
        write_vertex_attribute(
            get_resolution_level(open_store(str(store_path), mode="r+"), level),
            "component_label", labels, dtype=labels.dtype,
        )

    return {
        "labels": labels,
        "n_components": int(len(sizes)),
        "largest_component_size": int(sizes.max()) if len(sizes) else 0,
        "component_sizes": sizes,
    }
