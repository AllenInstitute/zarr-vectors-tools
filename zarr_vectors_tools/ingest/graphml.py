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
from zarr_vectors.typing import BinShape, ChunkShape


def ingest_graphml(
    input_path: str | Path,
    output_path: str | Path,
    chunk_shape: ChunkShape,
    *,
    bin_shape: BinShape | None = None,
    position_attrs: tuple[str, ...] = ("x", "y", "z"),
    dtype: str = "float32",
    compute_degree: bool = False,
    compute_component: bool = False,
    compute_clustering: bool = False,
    compute_summary: bool = False,
) -> dict[str, Any]:
    """Ingest a GraphML file into a zarr vectors graph store.

    Args:
        input_path: Path to the input .graphml file.
        output_path: Path for the output zarr vectors store.
        chunk_shape: Spatial chunk size per dimension.
        position_attrs: Node attribute names for coordinates.
        dtype: Dtype for position data.
        compute_degree: If True, write per-node degree (total in+out for
            directed graphs) to ``node_attributes["degree"]``.
        compute_component: If True, write 0-indexed connected-component
            label to ``node_attributes["component"]``. Uses weakly-
            connected components for directed graphs.
        compute_clustering: If True, write per-node clustering coefficient
            to ``node_attributes["clustering"]``.
        compute_summary: If True, write a :class:`GraphHeader` containing
            node/edge counts, mean degree, and connected-component stats.
            Blocked on a core API that supports per-graph attributes; the
            header is the workaround until that lands.

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

    if compute_degree:
        deg = np.zeros(n_nodes, dtype=np.float32)
        for node, d in G.degree():
            deg[node_to_idx[node]] = float(d)
        node_attributes["degree"] = deg

    if compute_component:
        comp_labels = np.zeros(n_nodes, dtype=np.float32)
        if G.is_directed():
            components = nx.weakly_connected_components(G)
        else:
            components = nx.connected_components(G)
        for ci, comp in enumerate(components):
            for node in comp:
                comp_labels[node_to_idx[node]] = float(ci)
        node_attributes["component"] = comp_labels

    if compute_clustering:
        # nx.clustering returns dict[node] -> float.
        clust = nx.clustering(G.to_undirected() if G.is_directed() else G)
        clust_arr = np.array(
            [float(clust.get(n, 0.0)) for n in nodes], dtype=np.float32
        )
        node_attributes["clustering"] = clust_arr

    result = write_graph(
        str(output_path),
        positions,
        edges,
        chunk_shape=chunk_shape,
        bin_shape=bin_shape,
        is_tree=False,
        node_attributes=node_attributes if node_attributes else None,
        edge_attributes=edge_attributes if edge_attributes else None,
        dtype=dtype,
    )

    if compute_summary:
        try:
            from zarr_vectors_tools.headers.formats import GraphHeader
            from zarr_vectors_tools.headers.registry import HeaderRegistry

            if G.is_directed():
                comps = list(nx.weakly_connected_components(G))
            else:
                comps = list(nx.connected_components(G))
            comp_sizes = [len(c) for c in comps] or [0]

            graph_header = GraphHeader(
                node_count=n_nodes,
                edge_count=len(edge_list),
                is_directed=bool(G.is_directed()),
                mean_degree=(2 * len(edge_list) / n_nodes) if n_nodes else 0.0,
                n_components=len(comps),
                largest_component_size=max(comp_sizes),
            )
            HeaderRegistry(str(output_path)).add("graph", graph_header)
        except Exception:
            pass

    return result
