"""Read the metadata layout of core format 0.10.0 in tests.

A store's levels are listed by the root ``multiscales.layout`` (one
``{"asset": <level>}`` per level, no transforms), its axes are
``zarr_vectors.axes``, and a level group's facts are its ``zarr_vectors``
object (``kind`` ``"level"``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def level_paths(root_attrs: Mapping[str, Any]) -> list[str]:
    """Level group names, as the root ``multiscales`` block lists them."""
    return [entry["asset"] for entry in root_attrs["multiscales"]["layout"]]


def root_axes(root_attrs: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The store's axes, from the root ``zarr_vectors`` object."""
    return list(root_attrs["zarr_vectors"]["axes"])


def level_block(level_attrs: Mapping[str, Any]) -> dict[str, Any]:
    """A level group's ``zarr_vectors`` object."""
    return dict(level_attrs["zarr_vectors"])


def array_block(array_attrs: Mapping[str, Any]) -> dict[str, Any]:
    """An array's ``zarr_vectors`` object: ``kind`` plus its own fields
    (``dtype``, ``encoding``, ``categories``, ...)."""
    return dict(array_attrs["zarr_vectors"])
