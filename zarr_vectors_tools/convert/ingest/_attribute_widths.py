"""Record the width of multi-column vertex attributes a core writer left out.

Core's ``write_polylines`` (like ``write_mesh``) writes a ``(N, C)`` vertex
attribute's values in full but stamps the array's ``row_shape`` as one
column.  A reader then sees ``C`` times as many rows as vertices, and
``read_polylines`` drops the column as misaligned -- so an RGB per-point
scalar from a TRK, or a multi-channel TRX ``dpv``, was written and then
unreadable.  The values are laid out row-major, so stamping the real width
is the whole repair.  Remove this once core records it itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def record_vertex_attribute_widths(
    store_path: str | Path,
    vertex_attributes: dict[str, Any] | None,
    *,
    level: int = 0,
) -> None:
    """Stamp ``row_shape`` for every attribute in ``vertex_attributes`` wider than one.

    Args:
        store_path: The store just written.
        vertex_attributes: What was handed to the writer: ``{name: [array
            per polyline]}`` or ``{name: array}``.
        level: The level written.
    """
    from zarr_vectors.building import get_resolution_level, open_store
    from zarr_vectors.constants import VERTEX_ATTRIBUTES

    widths: dict[str, int] = {}
    for name, values in (vertex_attributes or {}).items():
        sample = values[0] if isinstance(values, (list, tuple)) and values else values
        array = np.asarray(sample)
        if array.ndim == 2 and array.shape[1] > 1:
            widths[name] = int(array.shape[1])
    if not widths:
        return

    level_group = get_resolution_level(open_store(str(store_path), mode="r+"), level)
    for name, width in widths.items():
        path = f"{VERTEX_ATTRIBUTES}/{name}"
        meta = dict(level_group.read_array_meta(path) or {})
        if list(meta.get("row_shape") or []) == [width]:
            continue
        meta["row_shape"] = [width]
        level_group.write_array_meta(path, meta)
