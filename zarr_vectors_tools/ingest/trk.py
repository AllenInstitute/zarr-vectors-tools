"""Ingest streamlines from TrackVis TRK files into zarr vectors.

Requires ``nibabel``: ``pip install nibabel``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from zarr_vectors.exceptions import IngestError
from zarr_vectors.types.polylines import write_polylines
from zarr_vectors.typing import ChunkShape


def ingest_trk(
    input_path: str | Path,
    output_path: str | Path,
    chunk_shape: ChunkShape,
    *,
    dtype: str = "float32",
    preserve_header: bool = True,
    compute_length: bool = False,
    compute_endpoints: bool = False,
    length_range: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Ingest a TRK file into a zarr vectors streamline store.

    Args:
        input_path: Path to the input .trk file.
        output_path: Path for the output zarr vectors store.
        chunk_shape: Spatial chunk size per dimension (3D).
        dtype: Dtype for position data.
        preserve_header: If True, store the TRK header in
            ``/headers/trk/`` for round-trip export.
        compute_length: If True, write per-streamline path length to
            ``object_attributes["length"]``.
        compute_endpoints: If True, write per-streamline start and end
            points to ``object_attributes["start"]`` and ``["end"]``.
        length_range: Optional ``(min, max)`` length bounds; streamlines
            outside the range are dropped before writing. The dropped
            count is reported in the summary dict as ``dropped_by_length``.

    Returns:
        Summary dict from :func:`write_polylines`, plus any enrichment
        counters such as ``dropped_by_length``.

    Raises:
        IngestError: If nibabel is not installed or the file is unreadable.
    """
    try:
        import nibabel as nib
    except ImportError as e:
        raise IngestError(
            "nibabel is required for TRK ingest. "
            "Install with: pip install nibabel"
        ) from e

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        trk = nib.streamlines.load(str(input_path))
    except Exception as e:
        raise IngestError(f"Failed to read TRK '{input_path}': {e}") from e

    streamlines = trk.streamlines
    np_dtype = np.dtype(dtype)
    polylines = [np.asarray(s, dtype=np_dtype) for s in streamlines]

    if len(polylines) == 0:
        raise IngestError(f"TRK file contains no streamlines: {input_path}")

    # Extract per-vertex scalars if present
    vertex_attributes: dict[str, list[np.ndarray]] | None = None
    if hasattr(trk, "tractogram") and trk.tractogram.data_per_point:
        vertex_attributes = {}
        for key, values in trk.tractogram.data_per_point.items():
            vertex_attributes[key] = [
                np.asarray(v, dtype=np.float32) for v in values
            ]

    # Extract per-streamline properties if present
    object_attributes: dict[str, np.ndarray] | None = None
    if hasattr(trk, "tractogram") and trk.tractogram.data_per_streamline:
        object_attributes = {}
        for key, values in trk.tractogram.data_per_streamline.items():
            object_attributes[key] = np.asarray(values, dtype=np.float32)

    # Optional enrichments
    from zarr_vectors_tools.ingest._polyline_enrichments import (
        compute_endpoints as _compute_endpoints,
        compute_lengths as _compute_lengths,
        filter_by_length as _filter_by_length,
    )

    enrichment_summary: dict[str, Any] = {}
    lengths: np.ndarray | None = None

    if length_range is not None:
        lengths = _compute_lengths(polylines)
        kept, kept_idx, dropped = _filter_by_length(polylines, length_range, lengths=lengths)
        if dropped:
            polylines = kept
            lengths = lengths[kept_idx]
            if vertex_attributes is not None:
                vertex_attributes = {
                    k: [v[i] for i in kept_idx] for k, v in vertex_attributes.items()
                }
            if object_attributes is not None:
                object_attributes = {
                    k: v[kept_idx] for k, v in object_attributes.items()
                }
        enrichment_summary["dropped_by_length"] = dropped

    if compute_length:
        if lengths is None:
            lengths = _compute_lengths(polylines)
        object_attributes = dict(object_attributes or {})
        object_attributes["length"] = lengths

    if compute_endpoints:
        start, end = _compute_endpoints(polylines)
        object_attributes = dict(object_attributes or {})
        object_attributes["start"] = start
        object_attributes["end"] = end

    if len(polylines) == 0:
        raise IngestError(
            f"No streamlines remain after enrichment filtering: {input_path}"
        )

    result = write_polylines(
        str(output_path),
        polylines,
        chunk_shape=chunk_shape,
        vertex_attributes=vertex_attributes,
        object_attributes=object_attributes,
        dtype=dtype,
        geometry_type="streamline",
    )
    result.update(enrichment_summary)

    # Preserve TRK header
    if preserve_header:
        try:
            from zarr_vectors_tools.headers.registry import HeaderRegistry
            from zarr_vectors_tools.headers.formats import TRKHeader

            hdr = trk.header
            vox_size = tuple(float(v) for v in hdr["voxel_size"])
            dims = tuple(int(d) for d in hdr["dim"])
            affine = hdr["vox_to_ras"].flatten().tolist() if "vox_to_ras" in hdr.dtype.names else None
            vox_order = hdr["voxel_order"].item().decode() if isinstance(hdr["voxel_order"].item(), bytes) else str(hdr["voxel_order"].item())

            scalar_names = []
            if "scalar_name" in hdr.dtype.names:
                for sn in hdr["scalar_name"]:
                    name = sn.item().decode().strip("\x00") if isinstance(sn.item(), bytes) else str(sn.item()).strip("\x00")
                    if name:
                        scalar_names.append(name)

            property_names = []
            if "property_name" in hdr.dtype.names:
                for pn in hdr["property_name"]:
                    name = pn.item().decode().strip("\x00") if isinstance(pn.item(), bytes) else str(pn.item()).strip("\x00")
                    if name:
                        property_names.append(name)

            trk_header = TRKHeader(
                voxel_size=vox_size,
                dimensions=dims,
                vox_to_ras=affine,
                voxel_order=vox_order,
                n_scalars=int(hdr.get("n_scalars", 0)) if hasattr(hdr, "get") else 0,
                scalar_names=scalar_names,
                n_properties=int(hdr.get("n_properties", 0)) if hasattr(hdr, "get") else 0,
                property_names=property_names,
                n_count=len(polylines),
            )

            reg = HeaderRegistry(str(output_path))
            reg.add("trk", trk_header)
        except Exception:
            pass  # header preservation is best-effort

    return result
