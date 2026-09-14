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
    origin: list[float] | None = None
    """TrackVis ``origin`` field, as written by the file."""
    version: int | None = None
    """TrackVis format version."""
    space: str | None = None
    """Which space the STORED coordinates are in -- ``"voxmm"`` (as the file
    had them) or ``"rasmm"`` (registered at read time).  Without it a later
    merge or export cannot tell whether the affine still has to be applied,
    which is the difference between two tractograms overlaying and one of
    them sitting in the wrong place entirely."""
    n_count_mismatch: list[int] | None = None
    """``[declared, actual]`` when the header's streamline count disagreed
    with what the file holds."""
    extra: dict[str, Any] = field(default_factory=dict)
    """Any key a producer wrote that this class does not name.  Carried so a
    newer writer's field survives a round-trip through an older reader
    instead of being dropped on the floor."""

    #: Keys ``to_dict`` writes itself; everything else lands in ``extra``.
    _KNOWN = (
        "format_name", "voxel_size", "dimensions", "vox_to_ras", "voxel_order",
        "n_scalars", "scalar_names", "n_properties", "property_names",
        "n_count", "origin", "version", "space", "n_count_mismatch",
    )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
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
        for name in ("origin", "version", "space", "n_count_mismatch"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        out.update(self.extra)
        return out

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
            origin=d.get("origin"),
            version=d.get("version"),
            space=d.get("space"),
            n_count_mismatch=d.get("n_count_mismatch"),
            extra={k: v for k, v in d.items() if k not in cls._KNOWN},
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
# H5AD (AnnData / single-cell + spatial omics)
# ===================================================================

@dataclass
class H5ADHeader(Header):
    """AnnData ``.h5ad`` header metadata.

    Zarr Vectors stores numeric per-vertex arrays, so three kinds of AnnData
    information cannot survive in the arrays alone and live here instead:

    - **Names.** ``obs`` columns and genes become attribute arrays whose
      names are sanitised for Zarr paths, so the original labels are kept
      in the ``*_names``/``*_attrs`` parallel lists.
    - **Categories.** Categorical/string ``obs`` columns are stored as
      integer codes; ``categories`` maps a column to its level labels so
      export can rebuild the ``pandas.Categorical``.
    - **Order and identity.** Zarr Vectors orders vertices by spatial chunk, not by
      original row, so ``row_attr`` names the attribute holding each
      cell's source row index and ``obs_index`` (when small enough to
      inline) holds the original barcodes.
    """

    format_name: str = "h5ad"
    spatial_key: str = "spatial"
    spatial_ndim: int = 3
    # Columns of obsm[spatial_key] that became positions, in order.
    spatial_columns: list[int] = field(default_factory=list)
    n_obs: int = 0
    n_vars: int = 0
    # obs columns: original label -> attribute array name (parallel lists).
    obs_names: list[str] = field(default_factory=list)
    obs_attrs: list[str] = field(default_factory=list)
    # Expression columns: original var name -> attribute array name.
    gene_names: list[str] = field(default_factory=list)
    gene_attrs: list[str] = field(default_factory=list)
    # Source of the expression values: None = adata.X, else layers[<name>].
    layer: str | None = None
    # obs column label -> ordered category labels (for code -> label decode).
    categories: dict[str, list[str]] = field(default_factory=dict)
    # obs column label -> source pandas dtype, for the types that do not
    # survive as numpy arrays (bool -> uint8, datetime64 -> int64).
    dtypes: dict[str, str] = field(default_factory=dict)
    # Attribute holding the source obs row index (restores original order).
    row_attr: str | None = None
    # Original obs index (barcodes); None when omitted for size.
    obs_index: list[str] | None = None
    # obs column that became Zarr Vectors object_ids, plus its category labels.
    object_id_column: str | None = None
    object_id_categories: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_name": self.format_name,
            "spatial_key": self.spatial_key,
            "spatial_ndim": self.spatial_ndim,
            "spatial_columns": list(self.spatial_columns),
            "n_obs": self.n_obs,
            "n_vars": self.n_vars,
            "obs_names": self.obs_names,
            "obs_attrs": self.obs_attrs,
            "gene_names": self.gene_names,
            "gene_attrs": self.gene_attrs,
            "layer": self.layer,
            "categories": self.categories,
            "dtypes": self.dtypes,
            "row_attr": self.row_attr,
            "obs_index": self.obs_index,
            "object_id_column": self.object_id_column,
            "object_id_categories": self.object_id_categories,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> H5ADHeader:
        return cls(
            spatial_key=d.get("spatial_key", "spatial"),
            spatial_ndim=d.get("spatial_ndim", 3),
            spatial_columns=d.get("spatial_columns", []),
            n_obs=d.get("n_obs", 0),
            n_vars=d.get("n_vars", 0),
            obs_names=d.get("obs_names", []),
            obs_attrs=d.get("obs_attrs", []),
            gene_names=d.get("gene_names", []),
            gene_attrs=d.get("gene_attrs", []),
            layer=d.get("layer"),
            categories=d.get("categories", {}),
            dtypes=d.get("dtypes", {}),
            row_attr=d.get("row_attr"),
            obs_index=d.get("obs_index"),
            object_id_column=d.get("object_id_column"),
            object_id_categories=d.get("object_id_categories"),
        )

    def attr_to_obs(self) -> dict[str, str]:
        """Map stored attribute name -> original ``obs`` column label."""
        return dict(zip(self.obs_attrs, self.obs_names))

    def attr_to_gene(self) -> dict[str, str]:
        """Map stored attribute name -> original ``var`` (gene) name."""
        return dict(zip(self.gene_attrs, self.gene_names))


# ===================================================================
# Cortical surfaces (GIFTI, FreeSurfer, CIFTI)
# ===================================================================

@dataclass
class SurfaceHeader(Header):
    """What a cortical surface store needs that its arrays cannot say.

    A surface store holds one mesh object per hemisphere.  Four things about
    it live here rather than in the arrays:

    - **Which hemisphere is which object**, and how many vertices the source
      mesh had.  CIFTI data is indexed by surface vertex, and a 32k map is
      meaningless on a 164k surface, so the count is what lets an attach
      refuse a mismatch instead of writing garbage.
    - **Which surface is the geometry** and which ride along as coordinate
      attributes (``coords_white``, ``coords_inflated``...).  All share one
      topology; only one can be chunked.
    - **The coordinate space** the geometry is in.  FreeSurfer surfaces are
      in surface RAS, offset from scanner RAS by ``c_ras``; whether that
      shift was applied is the difference between a surface that overlays a
      tractogram and one that sits beside it.
    - **Label tables.** Parcellations are stored as integer codes; the names
      and colours that make them a parcellation are kept per attribute.
    """

    format_name: str = "surface"
    #: Where the geometry came from: ``"gifti"`` or ``"freesurfer"``.
    source: str = ""
    #: ``"scanner"`` (world RAS mm), ``"surface"`` (FreeSurfer tkr RAS), or
    #: the GIFTI dataspace name when the file declares one.
    space: str = "unknown"
    #: The surface chunked as geometry, e.g. ``"midthickness"``.
    geometry: str = ""
    #: One entry per object: ``{object_id, hemisphere, structure, n_vertices,
    #: n_faces}``.  ``object_id`` is the store's; ``hemisphere`` is
    #: ``"left"``/``"right"``.
    hemispheres: list[dict[str, Any]] = field(default_factory=list)
    #: Other surfaces of the same topology, as ``{attribute: surface name}``.
    alternates: dict[str, str] = field(default_factory=dict)
    #: Continuous per-vertex maps, as ``{attribute: source description}``.
    scalars: dict[str, str] = field(default_factory=dict)
    #: ``{attribute: {code: {"name": str, "rgba": [r, g, b, a]}}}``.  Codes
    #: are strings because JSON object keys must be.
    label_tables: dict[str, dict[str, dict[str, Any]]] = field(
        default_factory=dict,
    )
    #: Per-vertex attribute holding ``hemisphere << 32 | source vertex``.
    key_attribute: str = "zv_join_key"
    #: The FreeSurfer ``c_ras`` offset, when the geometry carried one.
    c_ras: list[float] | None = None
    #: Anything a producer wrote that this class does not name.
    extra: dict[str, Any] = field(default_factory=dict)

    _KNOWN = (
        "format_name", "source", "space", "geometry", "hemispheres",
        "alternates", "scalars", "label_tables", "key_attribute", "c_ras",
    )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "format_name": self.format_name,
            "source": self.source,
            "space": self.space,
            "geometry": self.geometry,
            "hemispheres": [dict(h) for h in self.hemispheres],
            "alternates": dict(self.alternates),
            "scalars": dict(self.scalars),
            "label_tables": {
                name: {str(code): dict(entry) for code, entry in table.items()}
                for name, table in self.label_tables.items()
            },
            "key_attribute": self.key_attribute,
            "c_ras": None if self.c_ras is None else [float(v) for v in self.c_ras],
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SurfaceHeader:
        return cls(
            source=d.get("source", ""),
            space=d.get("space", "unknown"),
            geometry=d.get("geometry", ""),
            hemispheres=[dict(h) for h in d.get("hemispheres", [])],
            alternates=dict(d.get("alternates", {})),
            scalars=dict(d.get("scalars", {})),
            label_tables={
                name: {str(code): dict(entry) for code, entry in table.items()}
                for name, table in d.get("label_tables", {}).items()
            },
            key_attribute=d.get("key_attribute", "zv_join_key"),
            c_ras=d.get("c_ras"),
            extra={k: v for k, v in d.items() if k not in cls._KNOWN},
        )

    def object_for(self, hemisphere: str) -> int | None:
        """The store's object id for ``"left"`` or ``"right"``, if present."""
        for entry in self.hemispheres:
            if entry.get("hemisphere") == hemisphere:
                return int(entry["object_id"])
        return None


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
    "h5ad": H5ADHeader,
    "surface": SurfaceHeader,
}


def header_from_dict(d: dict[str, Any]) -> Header:
    """Deserialise a header dict, dispatching to the correct class."""
    fmt = d.get("format_name", "")
    cls = HEADER_CLASSES.get(fmt)
    if cls is None:
        raise ValueError(f"Unknown header format: '{fmt}'")
    return cls.from_dict(d)
