"""Ingest LINC/atlas-labelled TRK files into zarr vectors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from zarr_vectors.exceptions import IngestError
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


def _read_lut(path: str | Path) -> dict[int, dict[str, Any]]:
    """Read a FreeSurfer-style ``ID Name R G B A`` LUT."""
    lut: dict[int, dict[str, Any]] = {}

    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            fields = line.split()
            if len(fields) < 6:
                raise IngestError(
                    f"Invalid LUT line {line_number} in '{path}': "
                    f"expected at least 6 fields"
                )

            try:
                label_id = int(fields[0])
                name = " ".join(fields[1:-4])
                rgba = tuple(int(x) for x in fields[-4:])
            except ValueError as e:
                raise IngestError(
                    f"Invalid LUT line {line_number} in '{path}'"
                ) from e

            if any(x < 0 or x > 255 for x in rgba):
                raise IngestError(
                    f"Invalid RGBA value on LUT line {line_number}: {rgba}"
                )

            lut[label_id] = {
                "name": name,
                "color": rgba,
            }

    return lut


def _read_mapping(path: str | Path) -> dict[str, int]:
    """Read the TRK filename -> numeric label mapping JSON."""
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise IngestError(
            f"Expected a JSON object in '{path}', got {type(raw).__name__}"
        )

    mapping: dict[str, int] = {}

    for filename, label_id in raw.items():
        if not isinstance(filename, str):
            raise IngestError(
                f"JSON mapping contains a non-string filename: {filename!r}"
            )

        if isinstance(label_id, bool) or not isinstance(label_id, int):
            raise IngestError(
                f"JSON mapping for {filename!r} is not an integer: {label_id!r}"
            )

        mapping[filename] = label_id

    return mapping


def _coerce_label_ids(values: Any) -> np.ndarray:
    """Validate and convert label IDs to int64.

    Integer arrays are accepted directly. Floating-point values are
    accepted when every value is finite, exactly integral, and within
    the int64 range.
    """
    label_ids = np.asarray(values)

    if label_ids.ndim > 1:
        label_ids = label_ids.reshape(len(label_ids), -1)

        if label_ids.shape[1] != 1:
            raise IngestError(
                "data_per_streamline['label_id'] must contain "
                "one integer per streamline"
            )

        label_ids = label_ids[:, 0]

    if np.issubdtype(label_ids.dtype, np.integer):
        info = np.iinfo(np.int64)

        if np.any(label_ids < info.min) or np.any(label_ids > info.max):
            raise IngestError(
                "data_per_streamline['label_id'] contains values "
                "outside the int64 range"
            )

        return label_ids.astype(np.int64, copy=False)

    if not np.issubdtype(label_ids.dtype, np.floating):
        raise IngestError(
            "data_per_streamline['label_id'] must contain integers"
        )

    if not np.all(np.isfinite(label_ids)):
        raise IngestError(
            "data_per_streamline['label_id'] must contain integers"
        )

    if not np.all(label_ids == np.floor(label_ids)):
        raise IngestError(
            "data_per_streamline['label_id'] must contain integers"
        )

    info = np.iinfo(np.int64)

    if (
        np.any(label_ids < info.min)
        or np.any(label_ids > info.max)
    ):
        raise IngestError(
            "data_per_streamline['label_id'] contains values outside the int64 range"
        )

    return label_ids.astype(np.int64)


def _make_groups(label_ids: np.ndarray) -> dict[int, list[int]]:
    """Return ``label_id -> polyline indices``."""
    groups: dict[int, list[int]] = {}

    for object_id, label_id in enumerate(label_ids.tolist()):
        label_id = int(label_id)
        groups.setdefault(label_id, []).append(object_id)

    return groups


def _make_group_attributes(
    groups: dict[int, list[int]],
    lut: dict[int, dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Create group-level name and RGBA arrays indexed by label ID."""
    if not groups:
        raise IngestError("No groups were found")

    max_label_id = max(groups)

    names = np.full(max_label_id + 1, "", dtype="U128")
    colors = np.zeros((max_label_id + 1, 4), dtype=np.uint8)

    for label_id in groups:
        try:
            metadata = lut[label_id]
        except KeyError as e:
            raise IngestError(
                f"Label ID {label_id} occurs in the TRK but is "
                f"missing from the LUT"
            ) from e

        names[label_id] = metadata["name"]
        colors[label_id] = metadata["color"]

    return {
        "name": names,
        "color": colors,
    }


def ingest_linc_trk(
    input_path: str | Path,
    output_path: str | Path,
    chunk_shape: ChunkShape,
    *,
    lut_path: str | Path,
    mapping_path: str | Path,
    bin_shape: BinShape | None = None,
    dtype: str = "float32",
    preserve_header: bool = True,
    compute_length: bool = False,
    compute_endpoints: bool = False,
    length_range: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Ingest a LINC-labelled TRK file into a zarr vectors store.

    The input TRK must contain
    ``data_per_streamline["label_id"]``. Each streamline's ``label_id``
    determines which group it belongs to. Integral floating-point label
    IDs are accepted and converted to ``int64``.

    The mapping JSON maps the input TRK filename to its expected atlas
    label ID. The LUT maps atlas label IDs to group metadata, including
    the group name and RGBA color. A group is created for each unique
    label ID present in the TRK.

    Args:
        input_path: Path to the input .trk file.
        output_path: Path for the output zarr vectors store.
        chunk_shape: Spatial chunk size per dimension (3D).
        lut_path: Path to the label lookup table containing label IDs,
            names, and RGBA colors.
        mapping_path: Path to the JSON file mapping TRK filenames to
            their expected atlas label IDs.
        bin_shape: Optional spatial bin size.
        dtype: Dtype for position data.
        preserve_header: If True, store the TRK header in
            ``/headers/trk/`` for round-trip export.
        compute_length: If True, write per-streamline path length to
            ``object_attributes["length"]``.
        compute_endpoints: If True, write per-streamline start and end
            points to ``object_attributes["start"]`` and
            ``object_attributes["end"]``.
        length_range: Optional ``(min, max)`` length bounds. Streamlines
            outside the range are dropped before writing. The dropped
            count is reported in the summary as
            ``dropped_by_length``.

    Returns:
        Summary dict from :func:`write_polylines`, plus any enrichment
        counters such as ``dropped_by_length``.

    Raises:
        IngestError: If the TRK does not contain ``label_id``, a label ID
            is invalid, the input filename is missing from the mapping,
            a mapped label does not occur in the TRK, or a TRK label is
            missing from the LUT.
        FileNotFoundError: If the input TRK file does not exist.
    """
    input_path = Path(input_path)

    # LINC-specific metadata files.
    lut = _read_lut(lut_path)
    mapping = _read_mapping(mapping_path)

    try:
        mapped_label_id = mapping[input_path.name]
    except KeyError as e:
        raise IngestError(
            f"TRK filename {input_path.name!r} is not present in "
            f"the mapping JSON"
        ) from e

    # Shared TRK loading.
    trk = _load_trk(input_path)

    dps = trk.tractogram.data_per_streamline

    if "label_id" not in dps:
        raise IngestError(
            "LINC TRK requires data_per_streamline['label_id']"
        )

    label_ids = _coerce_label_ids(dps["label_id"])

    if mapped_label_id not in set(label_ids.tolist()):
        raise IngestError(
            f"Mapping JSON assigns {input_path.name!r} to label "
            f"{mapped_label_id}, but that label does not occur in "
            f"the TRK label_id attribute"
        )

    # Shared TRK extraction. Tell the helper that label_id must remain
    # integer-valued rather than being converted to float32.
    (
        polylines,
        vertex_attributes,
        object_attributes,
    ) = _extract_trk_data(
        trk,
        dtype,
    )

    # Replace nibabel's representation with the validated int64 array.
    if object_attributes is None:
        object_attributes = {}

    object_attributes["label_id"] = label_ids.copy()

    # Shared length/endpoints/filtering.
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

    # Filtering can remove streamlines, so reconstruct label IDs from the
    # object attributes after enrichment.
    label_ids = np.asarray(
        object_attributes["label_id"],
        dtype=np.int64,
    )

    groups = _make_groups(label_ids)
    group_attributes = _make_group_attributes(
        groups,
        lut,
    )

    result = write_polylines(
        str(output_path),
        polylines,
        chunk_shape=chunk_shape,
        bin_shape=bin_shape,
        vertex_attributes=vertex_attributes,
        object_attributes=object_attributes,
        groups=groups,
        group_attributes=group_attributes,
        dtype=dtype,
        geometry_type="streamline",
    )

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
