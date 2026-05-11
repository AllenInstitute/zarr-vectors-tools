"""Ingest graphs from GraphML files into zarr vectors.

Requires ``networkx``: ``pip install networkx``.
Node positions must be stored as node attributes (e.g. ``x``, ``y``, ``z``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from zarr_vectors.exceptions import IngestError
from zarr_vectors.types.graphs import write_graph
from zarr_vectors.typing import ChunkShape


def ingest_graphml(
    input_path: str | Path,
    output_path: str | Path,
    chunk_shape: ChunkShape,
    *,
    position_attrs: tuple[str, ...] = ("x", "y", "z"),
    dtype: str = "float32",
) -> dict[str, Any]:
    """Ingest a GraphML file into a zarr vectors graph store.

    Args:
        input_path: Path to the input .graphml file.
        output_path: Path for the output zarr vectors store.
        chunk_shape: Spatial chunk size per dimension.
        position_attrs: Node attribute names for coordinates.
        dtype: Dtype for position data.

    Returns:
        Summary dict from :func:`write_graph`.
    """
    try:
        import networkx as nx
    except ImportError as e:
        raise IngestError(
            "networkx is required for GraphML ingest. "
            "Install with: pip install networkx"
        ) from e

    input_path = Path(input_path)
    if not input_path.exists():
        raise IngestError(f"Input file not found: {input_path}")

    try:
        G = nx.read_graphml(str(input_path))
    except Exception as e:
        raise IngestError(f"Failed to read GraphML '{input_path}': {e}") from e

    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)
    np_dtype = np.dtype(dtype)

    # Extract positions
    positions = np.zeros((n_nodes, len(position_attrs)), dtype=np_dtype)
    for i, node in enumerate(nodes):
        attrs = G.nodes[node]
        for d, attr_name in enumerate(position_attrs):
            if attr_name in attrs:
                positions[i, d] = float(attrs[attr_name])

    # Extract edges
    edge_list = list(G.edges())
    edges = np.array(
        [[node_to_idx[u], node_to_idx[v]] for u, v in edge_list],
        dtype=np.int64,
    ) if edge_list else np.zeros((0, 2), dtype=np.int64)

    # Extract node attributes (excluding position attrs)
    node_attributes: dict[str, np.ndarray] = {}
    if n_nodes > 0:
        sample_attrs = G.nodes[nodes[0]]
        for key in sample_attrs:
            if key not in position_attrs:
                try:
                    vals = [float(G.nodes[n].get(key, 0)) for n in nodes]
                    node_attributes[key] = np.array(vals, dtype=np.float32)
                except (ValueError, TypeError):
                    continue

    # Extract edge attributes
    edge_attributes: dict[str, np.ndarray] = {}
    if edge_list:
        sample_edge = G.edges[edge_list[0]]
        for key in sample_edge:
            try:
                vals = [float(G.edges[e].get(key, 0)) for e in edge_list]
                edge_attributes[key] = np.array(vals, dtype=np.float32)
            except (ValueError, TypeError):
                continue

    return write_graph(
        str(output_path),
        positions,
        edges,
        chunk_shape=chunk_shape,
        is_tree=False,
        node_attributes=node_attributes if node_attributes else None,
        edge_attributes=edge_attributes if edge_attributes else None,
        dtype=dtype,
    )
