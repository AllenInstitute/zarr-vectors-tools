"""Ingest neuronal morphology from SWC files into zarr vectors.

SWC is a 7-column text format: ID type X Y Z radius parent_ID.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from zarr_vectors.exceptions import IngestError
from zarr_vectors.types.graphs import write_graph
from zarr_vectors.typing import BinShape, ChunkShape

from zarr_vectors_tools.convert.ingest._object_columns import stamp_level_object_columns


def ingest_swc(
    input_path: str | Path,
    output_path: str | Path,
    chunk_shape: ChunkShape,
    *,
    bin_shape: BinShape | None = None,
    dtype: str = "float32",
    preserve_header: bool = True,
    compute_topological_depth: bool = False,
    compute_strahler: bool = False,
    compute_node_kind: bool = False,
) -> dict[str, Any]:
    """Ingest an SWC file into a zarr vectors skeleton store.

    Args:
        input_path: Path to the input .swc file.
        output_path: Path for the output zarr vectors store.
        chunk_shape: Spatial chunk size per dimension (3D).
        dtype: Dtype for position data.
        preserve_header: If True, store SWC comment lines in
            ``/headers/swc/`` for round-trip export.
        compute_topological_depth: If True, write per-node edge count
            from soma to ``node_attributes["topological_depth"]``.
        compute_strahler: If True, write per-node Strahler stream order
            to ``node_attributes["strahler"]``.
        compute_node_kind: If True, write per-node kind label to
            ``node_attributes["node_kind"]`` (0=soma, 1=branch,
            2=continuation, 3=terminal).  All three are stored as float32.

    Each tree is one object, and ``object_attributes/segment_id`` numbers
    them from 1 in root order, the id precomputed export and ``zvtools
    synapses`` use; each fragment carries its ``object_id`` and
    ``segment_id`` too, which the skeleton pyramid groups by.

    Returns:
        Summary dict from :func:`write_graph`.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise IngestError(f"Input file not found: {input_path}")

    try:
        rows: list[list[float]] = []
        comment_lines: list[str] = []
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    comment_lines.append(line)
                    continue
                parts = line.split()
                if len(parts) < 7:
                    continue
                rows.append([float(p) for p in parts[:7]])

        if not rows:
            raise IngestError(f"SWC file has no data rows: {input_path}")

        data = np.array(rows, dtype=np.float64)

    except IngestError:
        raise
    except Exception as e:
        raise IngestError(f"Failed to parse SWC '{input_path}': {e}") from e

    np_dtype = np.dtype(dtype)

    # Columns: ID(0) type(1) X(2) Y(3) Z(4) radius(5) parent_ID(6)
    swc_ids = data[:, 0].astype(np.int64)
    compartment = data[:, 1].astype(np.float32)
    positions = data[:, 2:5].astype(np_dtype)
    radius = data[:, 5].astype(np.float32)
    parent_ids = data[:, 6].astype(np.int64)

    # SWC IDs may not be contiguous 0-based — build remapping
    id_to_idx = {int(sid): i for i, sid in enumerate(swc_ids)}
    n_nodes = len(swc_ids)

    # Build edge list: [child_idx, parent_idx]
    edges: list[list[int]] = []
    for i in range(n_nodes):
        pid = int(parent_ids[i])
        if pid == -1 or pid not in id_to_idx:
            continue  # root or disconnected
        edges.append([i, id_to_idx[pid]])

    edges_arr = np.array(edges, dtype=np.int64) if edges else np.zeros((0, 2), dtype=np.int64)

    node_attributes: dict[str, np.ndarray] = {
        "radius": radius,
        "compartment": compartment,
    }

    # Parent index in SWC node order; a root (or a node whose parent is not
    # in the file) has -1.  A file with several roots is a forest, and a
    # skeleton store holds one rooted tree per object, so each tree becomes
    # its own object, numbered in the order its root appears.
    parent_idx = np.full(n_nodes, -1, dtype=np.int64)
    for i in range(n_nodes):
        pid = int(parent_ids[i])
        if pid != -1 and pid in id_to_idx:
            parent_idx[i] = id_to_idx[pid]
    roots = np.flatnonzero(parent_idx == -1)
    if roots.size == 0:
        raise IngestError(f"SWC file has no root (parent -1): {input_path}")
    object_ids = _tree_ids(parent_idx, roots) if roots.size > 1 else None

    if compute_topological_depth or compute_strahler or compute_node_kind:
        from zarr_vectors_tools.convert.ingest._tree_enrichments import compute_tree_metrics
        depth, strahler, node_kind = compute_tree_metrics(
            parent_idx, root_idx=roots.tolist(),
        )

        if compute_topological_depth:
            node_attributes["topological_depth"] = depth.astype(np.float32)
        if compute_strahler:
            node_attributes["strahler"] = strahler.astype(np.float32)
        if compute_node_kind:
            node_attributes["node_kind"] = node_kind.astype(np.float32)

    result = write_graph(
        str(output_path),
        positions,
        edges_arr,
        chunk_shape=chunk_shape,
        bin_shape=bin_shape,
        kind="skeleton",
        vertex_attributes=node_attributes,
        object_ids=object_ids,
        dtype=dtype,
    )
    _stamp_tree_ids(output_path)

    if preserve_header:
        try:
            from zarr_vectors_tools.headers.formats import SWCHeader
            from zarr_vectors_tools.headers.registry import HeaderRegistry

            swc_header = SWCHeader(
                comment_lines=comment_lines,
            )
            reg = HeaderRegistry(str(output_path))
            reg.add("swc", swc_header)
        except Exception:
            pass

    return result


def _stamp_tree_ids(store_path: str | Path) -> None:
    """Write each tree's segment id, then the per-object and per-fragment columns.

    ``write_graph`` writes neither.  Segment ids run 1..N in object order --
    0 is background to Neuroglancer -- and each fragment is stamped with its
    tree's segment id and dense object id (what the skeleton coarsener
    reads) from the object manifests the writer just produced, so this
    needs no second pass over the geometry.  The store is marked as the
    linked skeleton layout, so a reader holding these segment ids does not
    take it for a precomputed store.
    """
    from zarr_vectors.building import (
        create_object_attributes_array,
        get_resolution_level,
        open_store,
        read_all_object_manifests,
        write_object_attributes,
    )

    from zarr_vectors_tools.multiresolution.skeleton_layout import (
        LAYOUT_LINKED,
        mark_skeleton_layout,
    )

    root = open_store(str(store_path), mode="r+")
    level0 = get_resolution_level(root, 0)
    n_objects = len(read_all_object_manifests(level0))
    create_object_attributes_array(level0, "segment_id", dtype="uint64")
    write_object_attributes(
        level0, "segment_id", np.arange(1, n_objects + 1, dtype=np.uint64),
    )
    stamp_level_object_columns(level0, object_id=True)
    mark_skeleton_layout(root, LAYOUT_LINKED)


def _tree_ids(parent_idx: np.ndarray, roots: np.ndarray) -> np.ndarray:
    """Object id per node: the rank of its tree's root among ``roots``.

    Pointer jumping: each pass replaces a node's pointer with its pointer's
    pointer, so ``log2(depth)`` passes reach every root.  A cycle in the
    parent column never reaches one, which is reported rather than looped on.
    """
    up = np.where(parent_idx == -1, np.arange(parent_idx.size), parent_idx)
    for _ in range(64):
        nxt = up[up]
        if np.array_equal(nxt, up):
            break
        up = nxt
    else:
        raise IngestError("SWC parent column has a cycle; it is not a tree or forest")
    rank = np.full(parent_idx.size, -1, dtype=np.int64)
    rank[roots] = np.arange(roots.size)
    ids = rank[up]
    if (ids < 0).any():  # a node that is its own parent
        raise IngestError("SWC parent column has a cycle; it is not a tree or forest")
    return ids
