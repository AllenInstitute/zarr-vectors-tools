"""Format-specific header dataclasses.

Each header captures the metadata that would be lost when converting
to the zarr vectors format.  Stored as JSON-serialisable dicts in
``/headers/<format>/.zattrs``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ===================================================================
# Base
# ===================================================================

@dataclass
class Header:
    """Base class for format-specific headers."""

    format_name: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Header:
        """Deserialise from a dict."""
        raise NotImplementedError


# ===================================================================
# TRK (TrackVis)
# ===================================================================

@dataclass
class TRKHeader(Header):
    """TrackVis .trk file header.

    Captures the vox_to_ras affine, voxel sizes, dimensions,
    voxel order, and scalar/property names needed for round-trip.
    """

    format_name: str = "trk"
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0)
    dimensions: tuple[int, int, int] = (1, 1, 1)
    vox_to_ras: list[float] | None = None  # flattened 4×4 affine (16 floats)
    voxel_order: str = "LAS"
    n_scalars: int = 0
    scalar_names: list[str] = field(default_factory=list)
    n_properties: int = 0
    property_names: list[str] = field(default_factory=list)
    n_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_name": self.format_name,
            "voxel_size": list(self.voxel_size),
            "dimensions": list(self.dimensions),
            "vox_to_ras": self.vox_to_ras,
            "voxel_order": self.voxel_order,
            "n_scalars": self.n_scalars,
            "scalar_names": self.scalar_names,
            "n_properties": self.n_properties,
            "property_names": self.property_names,
            "n_count": self.n_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TRKHeader:
        return cls(
            voxel_size=tuple(d.get("voxel_size", [1, 1, 1])),
            dimensions=tuple(d.get("dimensions", [1, 1, 1])),
            vox_to_ras=d.get("vox_to_ras"),
            voxel_order=d.get("voxel_order", "LAS"),
            n_scalars=d.get("n_scalars", 0),
            scalar_names=d.get("scalar_names", []),
            n_properties=d.get("n_properties", 0),
            property_names=d.get("property_names", []),
            n_count=d.get("n_count", 0),
        )

    @property
    def affine(self) -> np.ndarray | None:
        """Return the vox_to_ras affine as a 4×4 numpy array."""
        if self.vox_to_ras is None:
            return None
        return np.array(self.vox_to_ras, dtype=np.float64).reshape(4, 4)


# ===================================================================
# NIfTI (spatial reference)
# ===================================================================

@dataclass
class NIfTIHeader(Header):
    """NIfTI spatial reference header.

    Stores the affine transform, voxel sizes, and dimension info.
    Used by any geometry type with a known coordinate system.
    """

    format_name: str = "nifti"
    affine: list[float] | None = None  # flattened 4×4 (16 floats)
    dimensions: tuple[int, ...] = (1, 1, 1)
    voxel_sizes: tuple[float, ...] = (1.0, 1.0, 1.0)
    qform_code: int = 0
    sform_code: int = 0
    xyzt_units: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_name": self.format_name,
            "affine": self.affine,
            "dimensions": list(self.dimensions),
            "voxel_sizes": list(self.voxel_sizes),
            "qform_code": self.qform_code,
            "sform_code": self.sform_code,
            "xyzt_units": self.xyzt_units,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NIfTIHeader:
        return cls(
            affine=d.get("affine"),
            dimensions=tuple(d.get("dimensions", [1, 1, 1])),
            voxel_sizes=tuple(d.get("voxel_sizes", [1, 1, 1])),
            qform_code=d.get("qform_code", 0),
            sform_code=d.get("sform_code", 0),
            xyzt_units=d.get("xyzt_units", 0),
        )

    @property
    def affine_matrix(self) -> np.ndarray | None:
        if self.affine is None:
            return None
        return np.array(self.affine, dtype=np.float64).reshape(4, 4)


# ===================================================================
# SWC
# ===================================================================

@dataclass
class SWCHeader(Header):
    """SWC file header (comment lines and metadata)."""

    format_name: str = "swc"
    comment_lines: list[str] = field(default_factory=list)
    coordinate_space: str = ""
    scaling: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_name": self.format_name,
            "comment_lines": self.comment_lines,
            "coordinate_space": self.coordinate_space,
            "scaling": list(self.scaling),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SWCHeader:
        return cls(
            comment_lines=d.get("comment_lines", []),
            coordinate_space=d.get("coordinate_space", ""),
            scaling=tuple(d.get("scaling", [1, 1, 1])),
        )


# ===================================================================
# LAS
# ===================================================================

@dataclass
class LASHeader(Header):
    """LAS/LAZ point cloud file header."""

    format_name: str = "las"
    version: str = "1.4"
    point_format: int = 0
    point_count: int = 0
    scale: tuple[float, float, float] = (0.001, 0.001, 0.001)
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    min_bound: tuple[float, float, float] = (0.0, 0.0, 0.0)
    max_bound: tuple[float, float, float] = (0.0, 0.0, 0.0)
    crs_wkt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_name": self.format_name,
            "version": self.version,
            "point_format": self.point_format,
            "point_count": self.point_count,
            "scale": list(self.scale),
            "offset": list(self.offset),
            "min_bound": list(self.min_bound),
            "max_bound": list(self.max_bound),
            "crs_wkt": self.crs_wkt,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LASHeader:
        return cls(
            version=d.get("version", "1.4"),
            point_format=d.get("point_format", 0),
            point_count=d.get("point_count", 0),
            scale=tuple(d.get("scale", [0.001, 0.001, 0.001])),
            offset=tuple(d.get("offset", [0, 0, 0])),
            min_bound=tuple(d.get("min_bound", [0, 0, 0])),
            max_bound=tuple(d.get("max_bound", [0, 0, 0])),
            crs_wkt=d.get("crs_wkt", ""),
        )


# ===================================================================
# OBJ
# ===================================================================

@dataclass
class OBJHeader(Header):
    """Wavefront OBJ file header metadata."""

    format_name: str = "obj"
    mtllib: str = ""
    object_names: list[str] = field(default_factory=list)
    group_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_name": self.format_name,
            "mtllib": self.mtllib,
            "object_names": self.object_names,
            "group_names": self.group_names,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OBJHeader:
        return cls(
            mtllib=d.get("mtllib", ""),
            object_names=d.get("object_names", []),
            group_names=d.get("group_names", []),
        )


# ===================================================================
# CSV
# ===================================================================

@dataclass
class CSVHeader(Header):
    """CSV point cloud file header metadata.

    ``normalise_offset`` and ``normalise_scale`` are populated by the
    ``normalise=True`` ingest path so export can invert the transform.
    They are ``None`` for stores that were not normalised.
    """

    format_name: str = "csv"
    column_names: list[str] = field(default_factory=list)
    delimiter: str = ","
    position_columns: list[str] = field(default_factory=lambda: ["x", "y", "z"])
    attribute_columns: list[str] = field(default_factory=list)
    has_header_row: bool = True
    normalise_offset: list[float] | None = None
    normalise_scale: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_name": self.format_name,
            "column_names": self.column_names,
            "delimiter": self.delimiter,
            "position_columns": self.position_columns,
            "attribute_columns": self.attribute_columns,
            "has_header_row": self.has_header_row,
            "normalise_offset": self.normalise_offset,
            "normalise_scale": self.normalise_scale,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CSVHeader:
        return cls(
            column_names=d.get("column_names", []),
            delimiter=d.get("delimiter", ","),
            position_columns=d.get("position_columns", ["x", "y", "z"]),
            attribute_columns=d.get("attribute_columns", []),
            has_header_row=d.get("has_header_row", True),
            normalise_offset=d.get("normalise_offset"),
            normalise_scale=d.get("normalise_scale"),
        )


# ===================================================================
# Graph (summary metrics)
# ===================================================================

@dataclass
class GraphHeader(Header):
    """Graph-level summary metrics.

    Stored as a workaround for the lack of per-object attributes on the
    ``graphs`` geometry type in the core write API. When that gap is
    closed in core, this header can migrate into ``graph_attributes``.
    """

    format_name: str = "graph"
    node_count: int = 0
    edge_count: int = 0
    is_directed: bool = False
    mean_degree: float = 0.0
    n_components: int = 0
    largest_component_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_name": self.format_name,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "is_directed": self.is_directed,
            "mean_degree": self.mean_degree,
            "n_components": self.n_components,
            "largest_component_size": self.largest_component_size,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GraphHeader:
        return cls(
            node_count=d.get("node_count", 0),
            edge_count=d.get("edge_count", 0),
            is_directed=d.get("is_directed", False),
            mean_degree=d.get("mean_degree", 0.0),
            n_components=d.get("n_components", 0),
            largest_component_size=d.get("largest_component_size", 0),
        )


# ===================================================================
# Neuroglancer Precomputed
# ===================================================================

@dataclass
class NeuroglancerHeader(Header):
    """Neuroglancer Precomputed source metadata.

    Captures the parts of the precomputed ``info`` JSON needed to
    reconstruct an equivalent layer on export and to preserve the
    original segment-ID space across a round trip.

    ``segment_ids`` is stored as ``list[str]`` because precomputed uses
    uint64 IDs that exceed JSON's safe integer range. Cast back to
    ``int`` on read.
    """

    format_name: str = "neuroglancer"
    data_type: str = ""  # "mesh" | "skeleton" | "annotation"
    annotation_type: str = ""  # "POINT" | "LINE" (annotations only)
    source_url: str = ""
    resolution: tuple[float, float, float] = (1.0, 1.0, 1.0)
    transform: list[float] | None = None  # flattened 4×4 (12 or 16 floats)
    mesh_metadata: dict[str, Any] = field(default_factory=dict)
    skeleton_metadata: dict[str, Any] = field(default_factory=dict)
    annotation_properties: list[dict[str, Any]] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)
    segment_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_name": self.format_name,
            "data_type": self.data_type,
            "annotation_type": self.annotation_type,
            "source_url": self.source_url,
            "resolution": list(self.resolution),
            "transform": self.transform,
            "mesh_metadata": self.mesh_metadata,
            "skeleton_metadata": self.skeleton_metadata,
            "annotation_properties": self.annotation_properties,
            "relationships": self.relationships,
            "segment_ids": self.segment_ids,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NeuroglancerHeader:
        return cls(
            data_type=d.get("data_type", ""),
            annotation_type=d.get("annotation_type", ""),
            source_url=d.get("source_url", ""),
            resolution=tuple(d.get("resolution", [1.0, 1.0, 1.0])),
            transform=d.get("transform"),
            mesh_metadata=d.get("mesh_metadata", {}),
            skeleton_metadata=d.get("skeleton_metadata", {}),
            annotation_properties=d.get("annotation_properties", []),
            relationships=d.get("relationships", []),
            segment_ids=d.get("segment_ids", []),
        )

    @property
    def transform_matrix(self) -> np.ndarray | None:
        """Return the transform as a 4×4 numpy array (or None)."""
        if self.transform is None:
            return None
        arr = np.array(self.transform, dtype=np.float64)
        if arr.size == 12:
            mat = np.eye(4, dtype=np.float64)
            mat[:3, :] = arr.reshape(3, 4)
            return mat
        return arr.reshape(4, 4)


# ===================================================================
# Dispatch helper
# ===================================================================

HEADER_CLASSES: dict[str, type[Header]] = {
    "trk": TRKHeader,
    "nifti": NIfTIHeader,
    "swc": SWCHeader,
    "las": LASHeader,
    "obj": OBJHeader,
    "csv": CSVHeader,
    "graph": GraphHeader,
    "neuroglancer": NeuroglancerHeader,
}


def header_from_dict(d: dict[str, Any]) -> Header:
    """Deserialise a header dict, dispatching to the correct class."""
    fmt = d.get("format_name", "")
    cls = HEADER_CLASSES.get(fmt)
    if cls is None:
        raise ValueError(f"Unknown header format: '{fmt}'")
    return cls.from_dict(d)
