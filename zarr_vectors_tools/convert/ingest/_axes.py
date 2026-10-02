"""Declare a store's coordinate unit after a core writer created the store.

``create_store(unit=...)`` puts a length unit on every space axis, but the
one-call writers (``write_polylines``, ``write_points``) create their store
themselves and take no unit.  A store whose axes carry none is read as
unitless: the viewer shows bare numbers, and a precomputed export has to
assume nanometres.  So the ingesters that know their unit stamp it here.

The axes live in two places, and both are rewritten: NGFF
``multiscales[0].axes`` (what readers, and every later
``write_multiscale_metadata``, take them from) and the ``world`` coordinate
system of the RFC 8 ``ome`` node.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any


def declare_axis_unit(store_path: str | Path, unit: str) -> None:
    """Put ``unit`` (a UDUNITS-2 name, e.g. ``"millimeter"``) on every space axis."""
    from zarr_vectors.building import open_store

    root = open_store(str(store_path), mode="r+")
    attrs = root.attrs.to_dict()

    def with_unit(axes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {**axis, "unit": unit} if axis.get("type", "space") == "space" else dict(axis)
            for axis in axes
        ]

    update: dict[str, Any] = {}
    multiscales = copy.deepcopy(attrs.get("multiscales"))
    if multiscales:
        multiscales[0]["axes"] = with_unit(multiscales[0].get("axes") or [])
        update["multiscales"] = multiscales
    ome = copy.deepcopy(attrs.get("ome"))
    systems = (((ome or {}).get("attributes") or {}).get("scene") or {}).get(
        "coordinateSystems"
    ) or []
    for system in systems:
        if system.get("axes"):
            system["axes"] = with_unit(system["axes"])
    if systems:
        update["ome"] = ome
    if update:
        root.attrs.update(update)
