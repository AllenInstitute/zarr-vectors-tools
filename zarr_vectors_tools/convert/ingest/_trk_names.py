"""The scalar and property name tables of a TrackVis header.

A TRK header has two tables of ten 20-byte names, one for the per-point
scalars and one for the per-streamline properties, next to a count of
COLUMNS (``n_scalars``, ``n_properties``).  Names and columns are not one to
one.  nibabel, which writes most TRK files in circulation, stores a
multi-column value in one slot as ``name\\x00<columns>`` -- an RGB colour is
``rgb\\x003`` -- so reading one name per column shifts every later column
onto the wrong name.  Files from TrackVis itself use one slot per column,
often unnamed.  :func:`decode_trk_names` reads both.
"""

from __future__ import annotations

TRK_N_SCALARS = 36
TRK_SCALAR_NAMES = 38
TRK_N_PROPERTIES = 238
TRK_PROPERTY_NAMES = 240

_SLOTS = 10
_SLOT_BYTES = 20


def decode_trk_names(
    raw: bytes, offset: int, columns: int, *, unnamed: str = "scalar",
) -> list[tuple[str, int]]:
    """``[(name, width), ...]`` covering exactly ``columns`` columns, in order.

    Args:
        raw: The 1000-byte header.
        offset: Start of the name table (:data:`TRK_SCALAR_NAMES` or
            :data:`TRK_PROPERTY_NAMES`).
        columns: The header's column count for that table.
        unnamed: Prefix for columns with no name (``"property"`` for the
            property table).

    A slot with no name, or columns beyond the last named slot, become
    ``<unnamed>_<i>``, one column each, where ``i`` is the column index, so
    every column is still addressable.  A width that would run past
    ``columns`` is clipped rather than trusted.
    """
    out: list[tuple[str, int]] = []
    used = 0
    for slot in range(_SLOTS):
        if used >= columns:
            break
        start = offset + slot * _SLOT_BYTES
        field = raw[start:start + _SLOT_BYTES]
        name_bytes, _, rest = field.partition(b"\x00")
        name = name_bytes.decode("ascii", errors="replace")
        width_text = rest.split(b"\x00", 1)[0]
        width = int(width_text) if width_text.isdigit() and int(width_text) > 0 else 1
        width = min(width, columns - used)
        out.append((name or f"{unnamed}_{used}", width))
        used += width
    while used < columns:
        out.append((f"{unnamed}_{used}", 1))
        used += 1
    return out
