"""Ingest streamlines from TrackVis TRK files into zarr vectors.

Requires ``nibabel``: ``pip install nibabel``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from zarr_vectors.types.polylines import write_polylines
from zarr_vectors.typing import BinShape, ChunkShape

from zarr_vectors_tools.convert.ingest._attribute_widths import (
    record_vertex_attribute_widths,
)
from zarr_vectors_tools.convert.ingest._segment_ids import (
    stamp_segment_ids,
)
from zarr_vectors_tools.convert.ingest.trk_helpers import (
    _apply_trk_enrichments,
    _extract_trk_data,
    _load_trk,
    _trk_header,
)


def ingest_trk(
    input_path: str | Path,
    output_path: str | Path,
    chunk_shape: ChunkShape,
    *,
    bin_shape: BinShape | None = None,
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
        bin_shape: Optional spatial bin size.
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
    input_path = Path(input_path)

    trk = _load_trk(input_path)

    (
        polylines,
        vertex_attributes,
        object_attributes,
    ) = _extract_trk_data(trk, dtype)

    (
        polylines,
        vertex_attributes,
        object_attributes,
        enrichment_summary,
    ) = _apply_trk_enrichments(
        polylines,
        vertex_attributes,
        object_attributes,
        compute_length=compute_length,
        compute_endpoints=compute_endpoints,
        length_range=length_range,
    )

    result = write_polylines(
        str(output_path),
        polylines,
        chunk_shape=chunk_shape,
        bin_shape=bin_shape,
        vertex_attributes=vertex_attributes,
        object_attributes=object_attributes,
        dtype=dtype,
        geometry_type="streamline",
    )

    # The pyramid needs a per-fragment segment id, and core's writer does
    # not produce one. Without this, the ingest succeeds but coarsening
    # refuses the store.
    stamp_segment_ids(output_path)

    record_vertex_attribute_widths(
        output_path,
        vertex_attributes,
    )

    result.update(enrichment_summary)

    if preserve_header:
        from zarr_vectors_tools.headers.registry import HeaderRegistry

        HeaderRegistry(str(output_path)).add(
            "trk",
            _trk_header(trk, len(polylines)),
        )

    return result
