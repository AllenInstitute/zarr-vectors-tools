"""Metadata of the virtual stores in the core format 0.10.0 layout.

The store classes describe each node in the pre-0.10 shape (flat array
fields tagged ``zv_array``, ``zarr_vectors_level``, a root ``zarr_vectors``
object with an NGFF ``multiscales`` list).  :func:`document` turns one
``zarr.json`` document into the 0.10.0 layout on the way out, with the same
rules core's ``migrate_metadata_0_10`` applies to a stored 0.9.x store, so a
virtual store reads and validates like one written by core.  No chunk
payload is affected.
"""

from __future__ import annotations

from typing import Any

from zarr_vectors.constants import FORMAT_VERSION
from zarr_vectors.core import convention as zvc

ZV_VERSION = FORMAT_VERSION


def attributes(attrs: dict[str, Any]) -> dict[str, Any]:
    """One node's attributes in the 0.10.0 layout."""
    if "zarr_vectors_level" in attrs:
        out = dict(attrs)
        block = dict(out.pop("zarr_vectors_level") or {})
        out[zvc.ZV_KEY] = {"kind": "level", **block}
        return zvc.with_registration(out)
    if "zv_array" in attrs:
        return zvc.wrap_array_meta(attrs)
    block = attrs.get(zvc.ZV_KEY)
    if isinstance(block, dict) and "kind" not in block and "zv_version" in block:
        return _root(attrs)
    return attrs


def _root(attrs: dict[str, Any]) -> dict[str, Any]:
    out = dict(attrs)
    zv = dict(out.pop(zvc.ZV_KEY))
    ms = out.pop("multiscales", None)
    axes: list[dict[str, Any]] = []
    levels: list[str] = []
    if isinstance(ms, list) and ms:
        axes = [dict(a) for a in ms[0].get("axes") or []]
        levels = [str(d["path"]) for d in ms[0].get("datasets") or [] if "path" in d]
    crs = zv.pop("crs", None)
    out[zvc.ZV_KEY] = {"kind": "root", **zv, "zv_version": FORMAT_VERSION, "axes": axes}
    out["multiscales"] = {"layout": [{"asset": lv} for lv in sorted(levels or ["0"], key=int)]}
    extra: tuple[dict[str, str], ...] = (zvc.MULTISCALES_CMO,)
    if crs is not None:
        out[zvc.PROJ_PROJJSON_KEY] = crs
        extra += (zvc.PROJ_CMO,)
    return zvc.with_registration(out, full=True, extra=extra)


def document(doc: dict[str, Any]) -> dict[str, Any]:
    """A ``zarr.json`` document with its attributes in the 0.10.0 layout."""
    if isinstance(doc, dict) and isinstance(doc.get("attributes"), dict) and doc["attributes"]:
        return {**doc, "attributes": attributes(doc["attributes"])}
    return doc
