"""Read one hemisphere back out of a cortical surface store.

A surface store chunks one surface per hemisphere as its geometry and keeps
the others -- white, pial, inflated, sphere -- as ``coords_<surface>``
vertex attributes sharing its topology.  The store orders vertices by chunk;
the tools that made the surfaces number them.  Every vertex carries the join
key ``hemisphere << 32 | source vertex``, so :func:`read_hemisphere` can put
them back in the source file's order, and asking for a different surface is
only a question of which column supplies the coordinates.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

_HEMISPHERE_ALIASES = {"lh": "left", "rh": "right", "l": "left", "r": "right"}
_KEY_SHIFT = 32
_VERTEX_MASK = (1 << _KEY_SHIFT) - 1


def read_hemisphere(
    store_path: str | Path,
    hemisphere: str,
    *,
    coords: str | None = None,
    attributes: Sequence[str] = (),
    level: int = 0,
) -> dict[str, Any]:
    """One hemisphere's mesh, indexed like the file it came from.

    Args:
        store_path: A store written by the GIFTI or FreeSurfer ingest.
        hemisphere: ``"left"`` / ``"right"`` (or ``"lh"`` / ``"rh"``).
        coords: The surface to return coordinates for: ``None`` or the
            geometry's own name for the chunked surface, or any alternate
            the store kept (``"inflated"``, or its attribute name
            ``"coords_inflated"``).  The faces are the same either way.
        attributes: Per-vertex attributes to return alongside, in the same
            vertex order (``"thickness"``, ``"aparc"``...).
        level: Resolution level.  At level 0 vertex ``i`` is source vertex
            ``i``.  A coarser level holds fewer vertices; they come back
            ordered by the source vertex each one stands for, and
            ``source_vertex`` says which.

    Returns:
        Dict with ``vertices`` ``(V, 3)``, ``faces`` ``(F, 3)`` indexing
        ``vertices``, ``source_vertex`` ``(V,)``, ``attributes``
        ``{name: (V,) or (V, C)}`` and ``surface``, the name of the surface
        the coordinates are.

    Raises:
        ValueError: If the store is not a surface store, lacks the
            hemisphere, or does not have the surface or an attribute asked
            for.  Each message lists what the store does have.
    """
    from zarr_vectors.building import (
        attribute_layout,
        chunk_local_to_global_offsets,
        get_resolution_level,
        open_store,
        read_chunk_attributes,
    )
    from zarr_vectors.types.meshes import read_mesh

    from zarr_vectors_tools.convert.ingest._surface_store import HEMISPHERE_CODES
    from zarr_vectors_tools.headers.registry import HeaderRegistry

    registry = HeaderRegistry(str(store_path))
    if not registry.has("surface"):
        raise ValueError(
            f"{store_path} has no surface header; read_hemisphere reads stores "
            f"written by the GIFTI or FreeSurfer ingest"
        )
    header = registry.get("surface")

    side = _HEMISPHERE_ALIASES.get(hemisphere.lower(), hemisphere.lower())
    present = [h["hemisphere"] for h in header.hemispheres]
    if side not in present:
        raise ValueError(f"{store_path} has no {hemisphere!r} hemisphere; it has {present}")

    surface, column = _coordinate_column(header, coords)

    level_group = get_resolution_level(open_store(str(store_path)), level)
    _offsets, chunk_keys, _total = chunk_local_to_global_offsets(level_group)

    def _level_order(name: str) -> npt.NDArray[Any]:
        # ``read_mesh`` returns vertices chunk by chunk in this order, each
        # chunk's fragments concatenated, and attribute cells are stored the
        # same way, so walking the same chunk list lines the rows up.
        dtype, ncols = attribute_layout(level_group, name)
        parts = []
        for cc in chunk_keys:
            for group in read_chunk_attributes(level_group, name, cc, dtype=dtype, ncols=ncols):
                parts.append(np.asarray(group).reshape(len(group), -1))
        values = np.concatenate(parts, axis=0) if parts else np.zeros((0, ncols), dtype)
        return values[:, 0] if ncols == 1 else values

    mesh = read_mesh(str(store_path), level=level)
    keys = _level_order(header.key_attribute).astype(np.int64)
    if len(keys) != len(mesh["vertices"]):
        raise ValueError(
            f"level {level} has {len(mesh['vertices'])} vertices but "
            f"{len(keys)} join keys; the store's key attribute does not line up"
        )

    code = HEMISPHERE_CODES[side]
    source = keys & _VERTEX_MASK
    mine = np.flatnonzero((keys >> _KEY_SHIFT) == code)
    # Stable, so vertices standing for the same source vertex at a coarse
    # level keep their store order.
    order = mine[np.argsort(source[mine], kind="stable")]
    new_index = np.full(len(keys), -1, dtype=np.int64)
    new_index[order] = np.arange(len(order))

    faces = np.asarray(mesh["faces"], dtype=np.int64)
    inside = np.all(new_index[faces] >= 0, axis=1) if len(faces) else np.zeros(0, bool)

    if column is None:
        vertices = np.asarray(mesh["vertices"])[order]
    else:
        vertices = _level_order(column)[order]

    available = _vertex_attribute_names(level_group)
    missing = [name for name in attributes if name not in available]
    if missing:
        raise ValueError(
            f"attribute(s) {missing} are not at level {level}; it has "
            f"{sorted(available)}"
        )
    return {
        "vertices": vertices,
        "faces": new_index[faces[inside]],
        "source_vertex": source[order],
        "attributes": {name: _level_order(name)[order] for name in attributes},
        "surface": surface,
    }


def _coordinate_column(header: Any, coords: str | None) -> tuple[str, str | None]:
    """``(surface name, attribute to read)``; ``None`` means the geometry."""
    if coords is None or coords == header.geometry:
        return header.geometry, None
    by_surface = {surface: attr for attr, surface in header.alternates.items()}
    if coords in by_surface:
        return coords, by_surface[coords]
    if coords in header.alternates:
        return header.alternates[coords], coords
    raise ValueError(
        f"the store has no {coords!r} surface; it has {header.geometry!r} "
        f"(the geometry) and {sorted(by_surface) or 'no alternates'}"
    )


def _vertex_attribute_names(level_group: Any) -> set[str]:
    from zarr_vectors.constants import VERTEX_ATTRIBUTES

    try:
        return set(level_group[VERTEX_ATTRIBUTES].children())
    except Exception:  # noqa: BLE001 - a level with no vertex attributes
        return set()
