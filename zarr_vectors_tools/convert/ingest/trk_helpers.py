"""Shared helpers for TrackVis TRK ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from zarr_vectors.exceptions import IngestError


def _load_trk(input_path: str | Path) -> Any:
    """Load a TRK file with nibabel."""
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
        raise IngestError(
            f"Failed to read TRK '{input_path}': {e}"
        ) from e

    if not hasattr(trk, "tractogram"):
        raise IngestError(f"TRK has no tractogram: {input_path}")

    return trk


def _extract_trk_data(
    trk: Any,
    dtype: str,
) -> tuple[
    list[np.ndarray],
    dict[str, list[np.ndarray]] | None,
    dict[str, np.ndarray] | None,
]:
    """Extract geometry, vertex attributes, and object attributes.

    Args:
        trk: A nibabel TRK object.
        dtype: Dtype for streamline position data.
        integer_object_attributes: Object-attribute names that should
            retain their integer dtype instead of being converted to
            float32.

    Returns:
        ``(polylines, vertex_attributes, object_attributes)``.
    """
    np_dtype = np.dtype(dtype)

    streamlines = trk.streamlines
    polylines = [np.asarray(s, dtype=np_dtype) for s in streamlines]

    if not polylines:
        raise IngestError("TRK file contains no streamlines")

    vertex_attributes: dict[str, list[np.ndarray]] | None = None

    if trk.tractogram.data_per_point:
        vertex_attributes = {}

        for key, values in trk.tractogram.data_per_point.items():
            vertex_attributes[key] = [
                np.asarray(v, dtype=np.float32)
                for v in values
            ]

    object_attributes: dict[str, np.ndarray] | None = None

    if trk.tractogram.data_per_streamline:
        object_attributes = {}

        for key, values in trk.tractogram.data_per_streamline.items():
            object_attributes[key] = np.asarray(
                values,
                dtype=np.float32,
            )

    return polylines, vertex_attributes, object_attributes


def _apply_trk_enrichments(
    polylines: list[np.ndarray],
    vertex_attributes: dict[str, list[np.ndarray]] | None,
    object_attributes: dict[str, np.ndarray] | None,
    *,
    compute_length: bool = False,
    compute_endpoints: bool = False,
    length_range: tuple[float, float] | None = None,
) -> tuple[
    list[np.ndarray],
    dict[str, list[np.ndarray]] | None,
    dict[str, np.ndarray] | None,
    dict[str, Any],
]:
    """Apply the common TRK enrichments and length filtering.

    Length filtering is applied before the optional length/endpoints
    attributes are added.
    """
    from zarr_vectors_tools.convert.ingest._polyline_enrichments import (
        compute_endpoints as _compute_endpoints,
    )
    from zarr_vectors_tools.convert.ingest._polyline_enrichments import (
        compute_lengths as _compute_lengths,
    )
    from zarr_vectors_tools.convert.ingest._polyline_enrichments import (
        filter_by_length as _filter_by_length,
    )

    enrichment_summary: dict[str, Any] = {}
    lengths: np.ndarray | None = None

    if length_range is not None:
        lengths = _compute_lengths(polylines)

        kept, kept_idx, dropped = _filter_by_length(
            polylines,
            length_range,
            lengths=lengths,
        )

        if dropped:
            polylines = kept
            lengths = lengths[kept_idx]

            if vertex_attributes is not None:
                vertex_attributes = {
                    key: [values[i] for i in kept_idx]
                    for key, values in vertex_attributes.items()
                }

            if object_attributes is not None:
                object_attributes = {
                    key: values[kept_idx]
                    for key, values in object_attributes.items()
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

    if not polylines:
        raise IngestError(
            "No streamlines remain after enrichment filtering"
        )

    return (
        polylines,
        vertex_attributes,
        object_attributes,
        enrichment_summary,
    )


def _trk_header(trk: Any, n_streamlines: int) -> Any:
    """Create a :class:`TRKHeader` from a nibabel TRK object."""
    from nibabel.streamlines import Field

    from zarr_vectors_tools.headers.formats import TRKHeader

    hdr = trk.header

    def _text(value: Any) -> str:
        if isinstance(value, np.ndarray):
            value = value.item()

        if isinstance(value, bytes):
            value = value.decode("ascii", errors="replace")

        return str(value).split("\x00", 1)[0]

    origin = hdr.get(Field.ORIGIN)
    version = hdr.get("version")

    return TRKHeader(
        voxel_size=tuple(
            float(v) for v in hdr[Field.VOXEL_SIZES]
        ),
        dimensions=tuple(
            int(d) for d in hdr[Field.DIMENSIONS]
        ),
        vox_to_ras=np.asarray(
            hdr[Field.VOXEL_TO_RASMM],
            dtype=np.float64,
        ).reshape(-1).tolist(),
        voxel_order=_text(hdr[Field.VOXEL_ORDER]),
        n_scalars=int(
            hdr.get(Field.NB_SCALARS_PER_POINT, 0)
        ),
        scalar_names=list(
            trk.tractogram.data_per_point.keys()
        ),
        n_properties=int(
            hdr.get(Field.NB_PROPERTIES_PER_STREAMLINE, 0)
        ),
        property_names=list(
            trk.tractogram.data_per_streamline.keys()
        ),
        n_count=int(n_streamlines),
        origin=(
            None
            if origin is None
            else [float(v) for v in origin]
        ),
        version=(
            None
            if version is None
            else int(version)
        ),
        # Nibabel returns streamlines in RAS+ millimetres.
        space="rasmm",
    )
