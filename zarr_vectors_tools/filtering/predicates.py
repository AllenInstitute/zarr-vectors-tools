"""Filter trees for streamlines: regions, leaves, operators, and their JSON.

A TrackVis-style filter is a tree.  Each leaf asks one question of every
streamline and answers yes or no:

- :class:`Passes` -- does the streamline meet a region (:class:`Sphere`,
  :class:`Box`, :class:`Slab`, :class:`Disk` or :class:`Mask`)?  ``mode``
  says which part of it counts: ``"path"`` (any segment), ``"endpoints"``
  (either end) or ``"both_endpoints"``.
- :class:`Column` -- does a per-streamline value pass a comparison?
- :class:`InGroup` -- is the streamline a member of any of these groups?
- :class:`All` -- yes, always (the default group before any filter).

:class:`And`, :class:`Or` and :class:`Not` combine them, and so do the
operators ``&``, ``|``, ``~`` and ``-`` (and-not) on any node.

The JSON form (:func:`to_json` / :func:`from_json`) is the contract between
engines: the brute-force oracle here, the indexed numpy engine, and the
browser engines that are tested against them all read the same document.
Coordinates are world millimetres, the same space
:func:`~zarr_vectors_tools.algorithms.select_streamlines` takes regions in.

What each region means, exactly
-------------------------------
Every set is closed, so a segment that just touches a region meets it.

- ``sphere``: some point of the segment is within ``radius`` of ``center``.
- ``box``: the segment meets the axis-aligned box ``min``..``max``.
- ``slab``: the segment's coordinate on ``axis`` takes some value in
  ``min``..``max`` (a box unbounded on the other two axes; TrackVis's slice
  filter).
- ``disk``: the segment crosses the disk's plane at a point within
  ``radius`` of ``center``, and the angle between the segment and the
  disk's normal axis lies within ``angle`` (degrees, 0..90; direction does
  not matter).  A segment lying in the plane does not cross it.
- ``mask``: a point sampled along the segment falls in a nonzero voxel.  The
  points are the segment's two vertices and ``ceil(length / spacing) - 1``
  evenly spaced between them; a point is in the voxel its coordinates round
  to.  ``spacing`` defaults to half the smallest voxel edge.  This is the
  scheme :func:`~zarr_vectors_tools.algorithms.select_streamlines` uses, so
  the two agree point for point.

A streamline with one vertex is that point; one with none meets nothing.
The endpoint modes test the first and last vertex as points, so they do not
apply to a disk, which has no interior.

Column comparisons
------------------
``cmp`` is one of ``<  <=  >  >=  ==  !=``, ``between`` (``value`` is
``[lo, hi]``, inclusive), ``in`` (``value`` is a list) or ``absent``.  An
absent value -- a row the store marks absent, or a NaN -- passes no
comparison, ``!=`` included, and passes only ``absent``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "All",
    "And",
    "Box",
    "COMPARISONS",
    "Column",
    "Disk",
    "FORMAT",
    "InGroup",
    "MODES",
    "Mask",
    "Node",
    "Not",
    "Or",
    "Passes",
    "Region",
    "Slab",
    "Sphere",
    "VERSION",
    "from_json",
    "to_json",
]

#: Name and version written into every JSON document.  A reader refuses a
#: version it does not know.
FORMAT = "zvf-predicate"
VERSION = 1

MODES = ("path", "endpoints", "both_endpoints")
COMPARISONS = ("<", "<=", ">", ">=", "==", "!=", "between", "in", "absent")


def _vec3(values: Iterable[float], what: str) -> tuple[float, float, float]:
    out = tuple(float(v) for v in values)
    if len(out) != 3 or not all(math.isfinite(v) for v in out):
        raise ValueError(f"{what} must be three finite numbers, got {list(out)}")
    return out  # type: ignore[return-value]


def _finite(value: float, what: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{what} must be finite, got {value}")
    return out


# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------


class Region:
    """Base class of the shapes a :class:`Passes` leaf tests against."""

    kind: str = ""

    def to_json(self) -> dict[str, Any]:  # pragma: no cover - every subclass overrides
        raise NotImplementedError


@dataclass(frozen=True)
class Sphere(Region):
    """A ball: ``center`` and ``radius`` in world millimetres."""

    center: tuple[float, float, float]
    radius: float
    kind = "sphere"

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _vec3(self.center, "sphere center"))
        radius = _finite(self.radius, "sphere radius")
        if radius < 0:
            raise ValueError(f"sphere radius must be 0 or more, got {radius}")
        object.__setattr__(self, "radius", radius)

    def to_json(self) -> dict[str, Any]:
        return {"type": self.kind, "center": list(self.center), "radius": self.radius}


@dataclass(frozen=True)
class Box(Region):
    """An axis-aligned box from corner ``min`` to corner ``max``."""

    min: tuple[float, float, float]
    max: tuple[float, float, float]
    kind = "box"

    def __post_init__(self) -> None:
        lo, hi = _vec3(self.min, "box min"), _vec3(self.max, "box max")
        if any(h < low for low, h in zip(lo, hi)):
            raise ValueError(f"box max corner {list(hi)} is below its min corner {list(lo)}")
        object.__setattr__(self, "min", lo)
        object.__setattr__(self, "max", hi)

    def to_json(self) -> dict[str, Any]:
        return {"type": self.kind, "min": list(self.min), "max": list(self.max)}


@dataclass(frozen=True)
class Slab(Region):
    """Everything whose coordinate on ``axis`` (0, 1 or 2) is in ``min``..``max``."""

    axis: int
    min: float
    max: float
    kind = "slab"

    def __post_init__(self) -> None:
        if self.axis not in (0, 1, 2):
            raise ValueError(f"slab axis must be 0, 1 or 2, got {self.axis}")
        lo, hi = _finite(self.min, "slab min"), _finite(self.max, "slab max")
        if hi < lo:
            raise ValueError(f"slab max {hi} is below its min {lo}")
        object.__setattr__(self, "axis", int(self.axis))
        object.__setattr__(self, "min", lo)
        object.__setattr__(self, "max", hi)

    def to_json(self) -> dict[str, Any]:
        return {"type": self.kind, "axis": self.axis, "min": self.min, "max": self.max}


@dataclass(frozen=True)
class Disk(Region):
    """A flat disk, crossed by segments whose angle to ``normal`` is in ``angle``.

    ``normal`` is normalised on construction.  ``angle`` is ``(lo, hi)`` in
    degrees between the segment and the normal axis: ``(0, 90)`` accepts any
    crossing, ``(0, 30)`` only segments within 30 degrees of the normal.
    """

    center: tuple[float, float, float]
    normal: tuple[float, float, float]
    radius: float
    angle: tuple[float, float] = (0.0, 90.0)
    kind = "disk"

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _vec3(self.center, "disk center"))
        normal = np.asarray(_vec3(self.normal, "disk normal"))
        norm = float(np.linalg.norm(normal))
        if norm == 0:
            raise ValueError("disk normal must not be the zero vector")
        object.__setattr__(self, "normal", tuple(float(v) for v in normal / norm))
        radius = _finite(self.radius, "disk radius")
        if radius < 0:
            raise ValueError(f"disk radius must be 0 or more, got {radius}")
        object.__setattr__(self, "radius", radius)
        lo, hi = (_finite(a, "disk angle") for a in self.angle)
        if not 0 <= lo <= hi <= 90:
            raise ValueError(f"disk angle must satisfy 0 <= lo <= hi <= 90, got {[lo, hi]}")
        object.__setattr__(self, "angle", (lo, hi))

    def to_json(self) -> dict[str, Any]:
        return {
            "type": self.kind, "center": list(self.center), "normal": list(self.normal),
            "radius": self.radius, "angle": list(self.angle),
        }


@dataclass(frozen=True, eq=False)
class Mask(Region):
    """A voxel region: nonzero ``voxels`` placed in the world by ``affine``.

    ``affine`` maps voxel indices to world millimetres (a NIfTI affine).
    Two masks are equal when their voxels, affine and spacing are.
    """

    voxels: npt.NDArray[np.bool_]
    affine: npt.NDArray[np.float64]
    spacing: float | None = None
    _to_voxel: npt.NDArray[np.float64] = field(init=False, repr=False)
    kind = "mask"

    def __post_init__(self) -> None:
        voxels = np.asarray(self.voxels)
        if voxels.ndim == 4 and voxels.shape[3] == 1:
            voxels = voxels[..., 0]
        if voxels.ndim != 3:
            raise ValueError(f"a mask must be 3-D, got shape {voxels.shape}")
        voxels = np.ascontiguousarray(voxels != 0)
        voxels.setflags(write=False)
        affine = np.asarray(self.affine, dtype=np.float64).reshape(4, 4).copy()
        affine.setflags(write=False)
        object.__setattr__(self, "voxels", voxels)
        object.__setattr__(self, "affine", affine)
        object.__setattr__(self, "_to_voxel", np.linalg.inv(affine))
        if self.spacing is not None:
            spacing = _finite(self.spacing, "mask spacing")
            if spacing <= 0:
                raise ValueError(f"mask spacing must be above 0, got {spacing}")
            object.__setattr__(self, "spacing", spacing)

    @property
    def sample_spacing(self) -> float:
        """Distance between points tested along a segment."""
        if self.spacing is not None:
            return self.spacing
        return float(np.linalg.norm(self.affine[:3, :3], axis=0).min()) / 2.0

    def contains(self, points: npt.NDArray) -> npt.NDArray[np.bool_]:
        """Whether each world point falls in a nonzero voxel."""
        to_voxel = self._to_voxel
        ijk = np.rint(points @ to_voxel[:3, :3].T + to_voxel[:3, 3]).astype(np.int64)
        shape = np.asarray(self.voxels.shape)
        valid = np.all((ijk >= 0) & (ijk < shape), axis=1)
        hit = np.zeros(len(points), dtype=bool)
        hit[valid] = self.voxels[tuple(ijk[valid].T)]
        return hit

    def world_bounds(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """World box around every nonzero voxel (each voxel +/- half a voxel)."""
        occupied = np.argwhere(self.voxels)
        if len(occupied) == 0:
            return np.full(3, np.inf), np.full(3, -np.inf)
        corners = np.array(np.meshgrid(*[
            [occupied[:, a].min() - 0.5, occupied[:, a].max() + 0.5] for a in range(3)
        ], indexing="ij")).reshape(3, -1).T
        world = corners @ self.affine[:3, :3].T + self.affine[:3, 3]
        return world.min(axis=0), world.max(axis=0)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "type": self.kind,
            "shape": list(self.voxels.shape),
            "affine": self.affine.tolist(),
            # C order, most significant bit first (numpy.packbits).
            "bits": base64.b64encode(np.packbits(self.voxels.ravel())).decode("ascii"),
        }
        if self.spacing is not None:
            out["spacing"] = self.spacing
        return out

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mask):
            return NotImplemented
        return (
            self.spacing == other.spacing
            and self.voxels.shape == other.voxels.shape
            and np.array_equal(self.affine, other.affine)
            and np.array_equal(self.voxels, other.voxels)
        )

    def __hash__(self) -> int:
        return hash(_digest(self.to_json()))


def _region_from_json(doc: dict[str, Any]) -> Region:
    kind = doc.get("type")
    if kind == "sphere":
        return Sphere(doc["center"], doc["radius"])
    if kind == "box":
        return Box(doc["min"], doc["max"])
    if kind == "slab":
        return Slab(int(doc["axis"]), doc["min"], doc["max"])
    if kind == "disk":
        return Disk(doc["center"], doc["normal"], doc["radius"], tuple(doc.get("angle", (0, 90))))
    if kind == "mask":
        shape = tuple(int(s) for s in doc["shape"])
        bits = np.frombuffer(base64.b64decode(doc["bits"]), dtype=np.uint8)
        voxels = np.unpackbits(bits, count=int(np.prod(shape))).astype(bool).reshape(shape)
        return Mask(voxels, np.asarray(doc["affine"], dtype=np.float64), doc.get("spacing"))
    raise ValueError(f"unknown region type {kind!r}")


# ---------------------------------------------------------------------------
# Tree nodes
# ---------------------------------------------------------------------------


class Node:
    """Base class of every filter tree node."""

    def __and__(self, other: Node) -> And:
        return And(self, other)

    def __or__(self, other: Node) -> Or:
        return Or(self, other)

    def __invert__(self) -> Not:
        return Not(self)

    def __sub__(self, other: Node) -> And:
        return And(self, Not(other))

    def to_json(self) -> dict[str, Any]:  # pragma: no cover - every subclass overrides
        raise NotImplementedError

    def key(self) -> str:
        """A stable digest of the node, for caching its result."""
        return _digest(self.to_json())

    def leaves(self) -> list[Node]:
        """The leaves under this node, left to right."""
        return [self]


@dataclass(frozen=True)
class All(Node):
    """Every streamline."""

    def to_json(self) -> dict[str, Any]:
        return {"op": "all"}


@dataclass(frozen=True)
class Passes(Node):
    """Streamlines that meet ``region``, by path or by their ends."""

    region: Region
    mode: str = "path"

    def __post_init__(self) -> None:
        if not isinstance(self.region, Region):
            raise TypeError(f"Passes needs a region, got {type(self.region).__name__}")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}; got {self.mode!r}")
        if self.mode != "path" and isinstance(self.region, Disk):
            raise ValueError("a disk has no inside, so it takes only mode='path'")

    def to_json(self) -> dict[str, Any]:
        return {"op": "passes", "mode": self.mode, "region": self.region.to_json()}


@dataclass(frozen=True)
class Column(Node):
    """Streamlines whose ``name`` column passes ``cmp value``.

    ``channel`` picks one channel of a multi-channel column.
    """

    name: str
    cmp: str
    value: Any = None
    channel: int | None = None

    def __post_init__(self) -> None:
        if self.cmp not in COMPARISONS:
            raise ValueError(f"cmp must be one of {', '.join(COMPARISONS)}; got {self.cmp!r}")
        value = self.value
        if self.cmp == "absent":
            value = None
        elif self.cmp == "between":
            lo, hi = (float(v) for v in value)
            if hi < lo:
                raise ValueError(f"between needs lo <= hi, got {[lo, hi]}")
            value = (lo, hi)
        elif self.cmp == "in":
            value = tuple(sorted({_number(v) for v in value}))
        else:
            value = _number(value)
        object.__setattr__(self, "value", value)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"op": "column", "name": self.name, "cmp": self.cmp}
        if self.value is not None:
            out["value"] = list(self.value) if isinstance(self.value, tuple) else self.value
        if self.channel is not None:
            out["channel"] = int(self.channel)
        return out


def _number(value: Any) -> int | float:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return float(value)


@dataclass(frozen=True)
class InGroup(Node):
    """Streamlines that belong to any of the groups ``ids``."""

    ids: tuple[int, ...]

    def __post_init__(self) -> None:
        ids = tuple(sorted({int(i) for i in self.ids}))
        if any(i < 0 for i in ids):
            raise ValueError(f"group ids are 0 or more, got {list(ids)}")
        object.__setattr__(self, "ids", ids)

    def to_json(self) -> dict[str, Any]:
        return {"op": "group", "ids": list(self.ids)}


class _Combine(Node):
    op = ""

    def __init__(self, *children: Node) -> None:
        if len(children) == 1 and not isinstance(children[0], Node):
            children = tuple(children[0])
        if not children:
            raise ValueError(f"{self.op} needs at least one operand")
        for child in children:
            if not isinstance(child, Node):
                raise TypeError(f"{self.op} operands are nodes, got {type(child).__name__}")
        self.children: tuple[Node, ...] = tuple(children)

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self.children == other.children  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        return hash((self.op, self.children))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(map(repr, self.children))})"

    def to_json(self) -> dict[str, Any]:
        return {"op": self.op, "args": [c.to_json() for c in self.children]}

    def leaves(self) -> list[Node]:
        return [leaf for child in self.children for leaf in child.leaves()]


class And(_Combine):
    """Streamlines every operand accepts."""

    op = "and"


class Or(_Combine):
    """Streamlines any operand accepts."""

    op = "or"


class Not(Node):
    """Streamlines the operand rejects."""

    def __init__(self, child: Node) -> None:
        if not isinstance(child, Node):
            raise TypeError(f"not takes a node, got {type(child).__name__}")
        self.child = child

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Not) and self.child == other.child

    def __hash__(self) -> int:
        return hash(("not", self.child))

    def __repr__(self) -> str:
        return f"Not({self.child!r})"

    def to_json(self) -> dict[str, Any]:
        return {"op": "not", "arg": self.child.to_json()}

    def leaves(self) -> list[Node]:
        return self.child.leaves()


def _node_from_json(doc: dict[str, Any]) -> Node:
    op = doc.get("op")
    if op == "all":
        return All()
    if op == "passes":
        return Passes(_region_from_json(doc["region"]), doc.get("mode", "path"))
    if op == "column":
        return Column(doc["name"], doc["cmp"], doc.get("value"), doc.get("channel"))
    if op == "group":
        return InGroup(tuple(doc["ids"]))
    if op == "and":
        return And(*[_node_from_json(d) for d in doc["args"]])
    if op == "or":
        return Or(*[_node_from_json(d) for d in doc["args"]])
    if op == "not":
        return Not(_node_from_json(doc["arg"]))
    raise ValueError(f"unknown node op {op!r}")


def to_json(tree: Node) -> dict[str, Any]:
    """The JSON document for ``tree``: ``{"format", "version", "tree"}``."""
    return {"format": FORMAT, "version": VERSION, "tree": tree.to_json()}


def from_json(doc: dict[str, Any] | str | bytes) -> Node:
    """The tree a :func:`to_json` document (or its text) describes.

    Raises:
        ValueError: If the document is not a filter tree, is a version this
            reader does not know, or names an unknown node or region.
    """
    if isinstance(doc, (str, bytes)):
        doc = json.loads(doc)
    if not isinstance(doc, dict) or doc.get("format") != FORMAT:
        raise ValueError(f"not a {FORMAT} document")
    if doc.get("version") != VERSION:
        raise ValueError(
            f"{FORMAT} version {doc.get('version')!r} is not supported (expected {VERSION})"
        )
    return _node_from_json(doc["tree"])


def _digest(doc: dict[str, Any]) -> str:
    text = json.dumps(doc, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

