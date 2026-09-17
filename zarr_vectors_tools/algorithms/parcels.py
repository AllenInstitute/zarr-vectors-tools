"""Per-parcel summaries of a cortical surface store, and the parcel at a point.

A parcellation is stored as a code per vertex with a label table in the
surface header, which is enough to answer the questions a surface analysis
starts with: how big is each region, and what is its mean thickness?
:func:`parcel_summary` computes them the way FreeSurfer's
``mris_anatomical_stats`` does, so a table built from a store matches the
``?h.aparc.stats`` of the same subject:

- ``vertex_count`` -- vertices carrying the code (``NumVert``);
- ``surface_area`` -- the sum of the vertices' areas on the white surface,
  each vertex taking a third of every triangle it belongs to (``SurfArea``);
- ``<map>_mean`` and ``<map>_std`` -- the plain mean and population standard
  deviation over those vertices (``ThickAvg``, ``ThickStd`` for thickness),
  plus ``<map>_area_weighted_mean``, which FreeSurfer does not report.

:func:`parcel_at` answers the other direction: which parcel is nearest to a
point, for example the hit of a ray cast or a streamline's endpoint.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

#: The store's fill for "no parcel" (a vertex no label covers).
_UNASSIGNED = -1


def vertex_areas(vertices: npt.NDArray, faces: npt.NDArray) -> npt.NDArray[np.float64]:
    """Each vertex's area: a third of every triangle that uses it.

    FreeSurfer's ``?h.area`` convention; the parcel sums match its stats.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    triangle = 0.5 * np.linalg.norm(
        np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1,
    )
    areas = np.zeros(len(v), dtype=np.float64)
    np.add.at(areas, f.ravel(), np.repeat(triangle / 3.0, 3))
    return areas


def parcel_summary(
    store_path: str | Path,
    parcellation: str,
    *,
    metrics: Sequence[str] | None = None,
    surface: str | None = "white",
    hemispheres: Iterable[str] | None = None,
    include_unassigned: bool = False,
    level: int = 0,
) -> Any:
    """One row per parcel per hemisphere.

    Args:
        store_path: A surface store (GIFTI or FreeSurfer ingest).
        parcellation: The parcellation attribute, e.g. ``"aparc"``.
        metrics: Continuous maps to summarise per parcel.  ``None`` means
            every scalar map the store's header lists; ``[]`` none.
        surface: The surface areas are measured on.  ``"white"`` matches
            FreeSurfer's stats; ``None`` uses the store's geometry.
        hemispheres: ``"left"``/``"right"`` (or ``"lh"``/``"rh"``); default all.
        include_unassigned: Also report vertices no parcel covers, as the
            row named ``"unassigned"``.
        level: Resolution level.  Areas at a coarser level are the coarse
            mesh's; counts are its vertices.

    Returns:
        A pandas DataFrame indexed by ``(hemisphere, parcel)`` with columns
        ``code``, ``vertex_count``, ``surface_area`` and, per metric,
        ``<metric>_mean``, ``<metric>_std`` and ``<metric>_area_weighted_mean``.
        Parcels in the label table that no vertex carries are left out, as
        FreeSurfer leaves them out of its stats.

    Raises:
        ValueError: If the store is not a surface store, or lacks the
            parcellation, a metric, the surface or a hemisphere -- each named.
    """
    import pandas as pd

    from zarr_vectors_tools.algorithms.surfaces import read_hemisphere

    header = _surface_header(store_path)
    if parcellation not in header.label_tables:
        raise ValueError(
            f"{parcellation!r} is not a parcellation in this store; it has "
            f"{sorted(header.label_tables) or 'none'}"
        )
    if metrics is None:
        metrics = sorted(header.scalars)
    names = {int(code): entry.get("name", str(code))
             for code, entry in header.label_tables[parcellation].items()}
    coords = None if surface is None or surface == header.geometry else surface

    rows: list[dict[str, Any]] = []
    for side in _hemispheres(header, hemispheres):
        mesh = read_hemisphere(
            store_path, side, coords=coords, level=level,
            attributes=[parcellation, *metrics],
        )
        areas = vertex_areas(mesh["vertices"], mesh["faces"])
        codes = np.asarray(mesh["attributes"][parcellation]).reshape(-1).astype(np.int64)
        values = {m: np.asarray(mesh["attributes"][m], dtype=np.float64) for m in metrics}
        for name, array in values.items():
            if array.ndim != 1:
                raise ValueError(
                    f"{name!r} has {array.shape[1]} columns per vertex; summarise "
                    f"single-column maps"
                )
        for code in np.unique(codes):
            if code == _UNASSIGNED and not include_unassigned:
                continue
            members = codes == code
            row: dict[str, Any] = {
                "hemisphere": side,
                "parcel": "unassigned" if code == _UNASSIGNED else names.get(int(code), str(code)),
                "code": int(code),
                "vertex_count": int(members.sum()),
                "surface_area": float(areas[members].sum()),
            }
            weights = areas[members]
            for name, array in values.items():
                inside = array[members]
                row[f"{name}_mean"] = float(inside.mean())
                row[f"{name}_std"] = float(inside.std())
                row[f"{name}_area_weighted_mean"] = (
                    float((inside * weights).sum() / weights.sum())
                    if weights.sum() > 0 else float("nan")
                )
            rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.set_index(["hemisphere", "parcel"])


def parcel_at(
    store_path: str | Path,
    points: npt.ArrayLike,
    parcellation: str,
    *,
    surface: str | None = "white",
    hemispheres: Iterable[str] | None = None,
    max_distance: float | None = None,
    level: int = 0,
) -> Any:
    """The parcel of the vertex nearest to each point.

    Args:
        store_path: A surface store.
        points: ``(N, 3)`` points in the store's coordinates, such as
            :func:`~zarr_vectors_tools.algorithms.mesh_query.closest_point`
            or ``cast_ray`` hits, or streamline endpoints.
        parcellation: The parcellation attribute.
        surface: The surface whose vertices are searched; ``None`` for the
            geometry.  Use the surface the points were measured against.
        hemispheres: Hemispheres to search; default all.
        max_distance: Points further than this from every vertex get no
            parcel.
        level: Resolution level.

    Returns:
        A pandas DataFrame with one row per point: ``hemisphere``, ``parcel``,
        ``code``, ``vertex`` (the source vertex number) and ``distance``.
    """
    import pandas as pd

    from zarr_vectors_tools.algorithms.surfaces import read_hemisphere

    header = _surface_header(store_path)
    if parcellation not in header.label_tables:
        raise ValueError(
            f"{parcellation!r} is not a parcellation in this store; it has "
            f"{sorted(header.label_tables) or 'none'}"
        )
    query = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    names = {int(code): entry.get("name", str(code))
             for code, entry in header.label_tables[parcellation].items()}
    coords = None if surface is None or surface == header.geometry else surface

    best = np.full(len(query), np.inf)
    rows: list[dict[str, Any]] = [
        {"hemisphere": None, "parcel": None, "code": _UNASSIGNED, "vertex": -1,
         "distance": np.inf}
        for _ in range(len(query))
    ]
    for side in _hemispheres(header, hemispheres):
        mesh = read_hemisphere(
            store_path, side, coords=coords, level=level, attributes=[parcellation],
        )
        codes = np.asarray(mesh["attributes"][parcellation]).reshape(-1).astype(np.int64)
        distance, nearest = _nearest(np.asarray(mesh["vertices"], dtype=np.float64), query)
        closer = np.flatnonzero(distance < best)
        best[closer] = distance[closer]
        for i in closer:
            code = int(codes[nearest[i]])
            rows[i] = {
                "hemisphere": side,
                "parcel": None if code == _UNASSIGNED else names.get(code, str(code)),
                "code": code,
                "vertex": int(mesh["source_vertex"][nearest[i]]),
                "distance": float(distance[i]),
            }
    table = pd.DataFrame(rows)
    if max_distance is not None:
        far = table["distance"] > float(max_distance)
        table.loc[far, ["hemisphere", "parcel"]] = None
        table.loc[far, "code"] = _UNASSIGNED
    return table


def _nearest(vertices: npt.NDArray, query: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """Distance to, and index of, the nearest vertex for each query point."""
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None
    if cKDTree is not None:
        distance, index = cKDTree(vertices).query(query)
        return np.asarray(distance), np.asarray(index, dtype=np.int64)
    # Without scipy, compare in blocks so memory stays at block x vertices.
    distance = np.empty(len(query))
    index = np.empty(len(query), dtype=np.int64)
    block = max(1, int(2e7 // max(1, len(vertices))))
    for start in range(0, len(query), block):
        part = query[start:start + block]
        squared = ((part[:, None, :] - vertices[None, :, :]) ** 2).sum(axis=2)
        index[start:start + block] = squared.argmin(axis=1)
        distance[start:start + block] = np.sqrt(squared.min(axis=1))
    return distance, index


def _surface_header(store_path: str | Path) -> Any:
    from zarr_vectors_tools.headers.registry import HeaderRegistry

    registry = HeaderRegistry(str(store_path))
    if not registry.has("surface"):
        raise ValueError(
            f"{store_path} has no surface header; parcel summaries read stores "
            f"written by the GIFTI or FreeSurfer ingest"
        )
    return registry.get("surface")


def _hemispheres(header: Any, requested: Iterable[str] | None) -> list[str]:
    present = [h["hemisphere"] for h in header.hemispheres]
    if requested is None:
        return present
    aliases = {"lh": "left", "rh": "right", "l": "left", "r": "right"}
    chosen = [aliases.get(str(h).lower(), str(h).lower()) for h in (
        [requested] if isinstance(requested, str) else requested
    )]
    missing = [h for h in chosen if h not in present]
    if missing:
        raise ValueError(f"the store has no {missing} hemisphere; it has {present}")
    return chosen
