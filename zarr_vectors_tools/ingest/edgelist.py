"""Ingest graphs from CSV edge-lists into ZVF.

Two input CSVs:
- ``edges_path``: rows are ``source,target`` plus optional weight/attr columns.
- ``nodes_path``: rows are ``node_id,x,y,z`` plus optional node-attr columns.

Optional ``use_cudf=True`` reads both CSVs on GPU via RAPIDS cuDF. Falls
back to pandas otherwise (``pandas`` is a top-level dependency).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from zarr_vectors.exceptions import IngestError
from zarr_vectors.types.graphs import write_graph
from zarr_vectors.typing import BinShape, ChunkShape


def _read_csv(path: Path, *, use_cudf: bool):
    """Read a CSV with cuDF (if requested) and return a pandas DataFrame."""
    if use_cudf:
        try:
            import cudf  # type: ignore
        except ImportError as e:
            raise IngestError(
                "use_cudf=True requires the cuDF package (RAPIDS). "
                "Install per https://docs.rapids.ai or pass use_cudf=False."
            ) from e
        try:
            df = cudf.read_csv(str(path)).to_pandas()
        except Exception as e:
            raise IngestError(f"cuDF failed to read {path}: {e}") from e
    else:
        import pandas as pd
        try:
            df = pd.read_csv(str(path))
        except Exception as e:
            raise IngestError(f"pandas failed to read {path}: {e}") from e
    return df


def ingest_edgelist(
    edges_path: str | Path,
    nodes_path: str | Path,
    output_path: str | Path,
    chunk_shape: ChunkShape,
    *,
    bin_shape: BinShape | None = None,
    use_cudf: bool = False,
    source_col: str = "source",
    target_col: str = "target",
    node_id_col: str = "node_id",
    position_columns: tuple[str, ...] = ("x", "y", "z"),
    edge_attribute_columns: list[str] | None = None,
    node_attribute_columns: list[str] | None = None,
    dtype: str = "float32",
    drop_na: bool = False,
    drop_duplicates: bool = False,
    compute_degree: bool = False,
    compute_component: bool = False,
    compute_clustering: bool = False,
    compute_summary: bool = False,
) -> dict[str, Any]:
    """Ingest a CSV edge-list (plus a node-position CSV) into a graph store.

    Args:
        edges_path: CSV with at least ``source_col`` and ``target_col``
            columns. Extra columns become edge attributes (filterable via
            ``edge_attribute_columns``).
        nodes_path: CSV with at least ``node_id_col`` and ``position_columns``.
            Extra columns become node attributes.
        output_path: Path for the output ZVF store.
        chunk_shape: Spatial chunk size per dimension.
        use_cudf: Read both CSVs via cuDF on GPU. Requires a RAPIDS install.
        source_col, target_col: Column names in the edge CSV.
        node_id_col: Column name in the node CSV linking to ``source_col``
            and ``target_col``.
        position_columns: Column names in the node CSV holding coordinates.
        edge_attribute_columns: Names of edge-CSV columns to retain.
            Default: every non-source/non-target column.
        node_attribute_columns: Names of node-CSV columns to retain.
            Default: every non-id/non-position column.
        dtype: Dtype for position data.
        drop_na: Drop edge rows with NaN source/target and node rows with
            NaN positions. Reports drop counts in summary.
        drop_duplicates: Drop duplicate edge rows (and duplicate node IDs,
            keeping the first).
        compute_degree, compute_component, compute_clustering, compute_summary:
            Same semantics as :func:`ingest_graphml`.

    Returns:
        Summary dict from :func:`write_graph`, plus enrichment counters.

    Raises:
        IngestError: For missing files, cuDF errors, or empty results.
    """
    edges_path = Path(edges_path)
    nodes_path = Path(nodes_path)
    if not edges_path.exists():
        raise IngestError(f"Edges file not found: {edges_path}")
    if not nodes_path.exists():
        raise IngestError(f"Nodes file not found: {nodes_path}")

    edges_df = _read_csv(edges_path, use_cudf=use_cudf)
    nodes_df = _read_csv(nodes_path, use_cudf=use_cudf)

    for col in (source_col, target_col):
        if col not in edges_df.columns:
            raise IngestError(f"Missing column '{col}' in edge CSV: {edges_path}")
    for col in (node_id_col, *position_columns):
        if col not in nodes_df.columns:
            raise IngestError(f"Missing column '{col}' in node CSV: {nodes_path}")

    enrichment_summary: dict[str, Any] = {}

    if drop_na:
        before_e, before_n = len(edges_df), len(nodes_df)
        edges_df = edges_df.dropna(subset=[source_col, target_col])
        nodes_df = nodes_df.dropna(subset=list(position_columns))
        enrichment_summary["dropped_na_edges"] = before_e - len(edges_df)
        enrichment_summary["dropped_na_nodes"] = before_n - len(nodes_df)

    if drop_duplicates:
        before_e, before_n = len(edges_df), len(nodes_df)
        edges_df = edges_df.drop_duplicates(subset=[source_col, target_col])
        nodes_df = nodes_df.drop_duplicates(subset=[node_id_col])
        enrichment_summary["dropped_duplicate_edges"] = before_e - len(edges_df)
        enrichment_summary["dropped_duplicate_nodes"] = before_n - len(nodes_df)

    if len(nodes_df) == 0:
        raise IngestError("Node table is empty after filtering.")

    node_ids = nodes_df[node_id_col].to_numpy()
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    positions = nodes_df[list(position_columns)].to_numpy(dtype=np.float64).astype(np.dtype(dtype))

    edges_arr = np.array(
        [
            [node_to_idx[s], node_to_idx[t]]
            for s, t in zip(edges_df[source_col].to_numpy(), edges_df[target_col].to_numpy())
            if s in node_to_idx and t in node_to_idx
        ],
        dtype=np.int64,
    ) if len(edges_df) else np.zeros((0, 2), dtype=np.int64)

    node_attributes: dict[str, np.ndarray] = {}
    keep_node_cols = (
        node_attribute_columns
        if node_attribute_columns is not None
        else [c for c in nodes_df.columns if c != node_id_col and c not in position_columns]
    )
    for col in keep_node_cols:
        if col in nodes_df.columns:
            try:
                node_attributes[col] = nodes_df[col].to_numpy(dtype=np.float32)
            except Exception:
                continue

    edge_attributes: dict[str, np.ndarray] = {}
    keep_edge_cols = (
        edge_attribute_columns
        if edge_attribute_columns is not None
        else [c for c in edges_df.columns if c not in (source_col, target_col)]
    )
    for col in keep_edge_cols:
        if col in edges_df.columns:
            try:
                edge_attributes[col] = edges_df[col].to_numpy(dtype=np.float32)[: len(edges_arr)]
            except Exception:
                continue

    # Optional networkx-based per-node enrichments.
    if compute_degree or compute_component or compute_clustering or compute_summary:
        try:
            import networkx as nx
        except ImportError as e:
            raise IngestError(
                "networkx is required for graph enrichments. "
                "Install with: pip install zarr-vectors-tools[graph]"
            ) from e

        G = nx.Graph()
        G.add_nodes_from(range(len(node_ids)))
        for u, v in edges_arr.tolist():
            G.add_edge(int(u), int(v))

        if compute_degree:
            deg = np.zeros(len(node_ids), dtype=np.float32)
            for node, d in G.degree():
                deg[node] = float(d)
            node_attributes["degree"] = deg

        if compute_component:
            comp = np.zeros(len(node_ids), dtype=np.float32)
            for ci, c in enumerate(nx.connected_components(G)):
                for n in c:
                    comp[n] = float(ci)
            node_attributes["component"] = comp

        if compute_clustering:
            clust = nx.clustering(G)
            node_attributes["clustering"] = np.array(
                [float(clust.get(i, 0.0)) for i in range(len(node_ids))],
                dtype=np.float32,
            )

    result = write_graph(
        str(output_path),
        positions,
        edges_arr,
        chunk_shape=chunk_shape,
        bin_shape=bin_shape,
        kind="graph",
        vertex_attributes=node_attributes if node_attributes else None,
        link_attributes=edge_attributes if edge_attributes else None,
        dtype=dtype,
    )
    result.update(enrichment_summary)

    if compute_summary:
        try:
            import networkx as nx  # noqa: F401  (already imported above for enrichments)
            from zarr_vectors_tools.headers.formats import GraphHeader
            from zarr_vectors_tools.headers.registry import HeaderRegistry

            n_nodes = len(node_ids)
            comps = list(nx.connected_components(G))  # type: ignore[name-defined]
            comp_sizes = [len(c) for c in comps] or [0]
            graph_header = GraphHeader(
                node_count=n_nodes,
                edge_count=int(edges_arr.shape[0]),
                is_directed=False,
                mean_degree=(2 * edges_arr.shape[0] / n_nodes) if n_nodes else 0.0,
                n_components=len(comps),
                largest_component_size=max(comp_sizes),
            )
            HeaderRegistry(str(output_path)).add("graph", graph_header)
        except Exception:
            pass

    return result
