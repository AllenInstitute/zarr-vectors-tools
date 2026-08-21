"""The 4 GiB per-chunk payload ceiling, and the pre-write check for it.

Every zarr-vectors per-chunk array (``vertices``, ``vertex_fragments``,
``links``, ...) is one Zarr v3 ``vlen-bytes`` array whose cells are the
per-chunk blobs.  That codec records each cell's byte length in a **uint32**
header, so a cell of 4 GiB or more wraps it modulo 2**32.  numcodecs does not
range-check the write: every byte lands on disk, but the recorded length is
``len(payload) % 2**32``, and every later read hands back a buffer truncated to
that wrapped length.  The write looks clean and the corruption surfaces
arbitrarily far downstream — on a real 10.7 GB tractogram it showed up hours
later as ``cannot reshape array of size 172225607 into shape (3)`` when the
pyramid tried ``reshape(-1, 3)`` on a truncated vertex buffer.

The ceiling applies to the *uncompressed* payload: ``vlen-bytes`` is the
array→bytes codec and writes its length header before any bytes→bytes
compressor runs, so a ``--compressor`` does not buy headroom.

A spatial chunk's payload is set entirely by how much geometry falls inside it,
so the only lever is a finer grid — hence :func:`check_vertex_cell_limit`,
which the ingest calls once binning has produced exact per-chunk vertex counts
and *before* the expensive write phase.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Sequence

# Exclusive: a cell payload must be strictly smaller than this to round-trip.
VLEN_CELL_LIMIT_BYTES = 2**32

# What the "re-run with a finer grid" suggestion aims the worst chunk at.  Well
# under the hard ceiling on purpose: subdividing a grid does not split a dense
# chunk evenly, so a suggestion that only just fits would often still overflow.
SUGGESTED_TARGET_BYTES = VLEN_CELL_LIMIT_BYTES // 4  # 1 GiB


def format_bytes(n: int) -> str:
    """Render a byte count as a human-readable binary-prefix string."""
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    raise AssertionError("unreachable")


def _round_up_2sf(n: float) -> int:
    """Round up to two significant figures, so suggestions read as advice."""
    if n <= 0:
        return 1
    from math import ceil, floor, log10

    mag = 10 ** max(0, floor(log10(n)) - 1)
    return int(ceil(n / mag) * mag)


def check_vertex_cell_limit(
    chunk_vertex_counts: Mapping[tuple[int, ...], int],
    *,
    ndim: int,
    itemsize: int,
    grid_shape: Sequence[int],
    chunk_shape: Sequence[float] | None = None,
    limit_bytes: int | None = None,
) -> None:
    """Raise if any spatial chunk's vertex payload would overflow a cell.

    Args:
        chunk_vertex_counts: Vertex count per occupied chunk coordinate — the
            count actually destined for that chunk's cell, boundary-split
            duplicates included.
        ndim: Coordinate components per vertex (3 for TRK).
        itemsize: Bytes per component (4 for float32).
        grid_shape: Chunk-grid extent, used to size the suggested refinement.
        chunk_shape: Optional chunk edge lengths, echoed in the message.
        limit_bytes: Ceiling to enforce.  ``None`` (default) reads
            :data:`VLEN_CELL_LIMIT_BYTES` at call time, so tests can lower it
            rather than materialise 4 GiB.

    Raises:
        ValueError: If any chunk is at or over the ceiling.  The message names
            the worst offenders and a ``num_chunks`` that should clear it.
    """
    if limit_bytes is None:
        limit_bytes = VLEN_CELL_LIMIT_BYTES
    row_bytes = ndim * itemsize
    over = sorted(
        (
            (count * row_bytes, coord, count)
            for coord, count in chunk_vertex_counts.items()
            if count * row_bytes >= limit_bytes
        ),
        reverse=True,
    )
    if not over:
        return

    grid_cells = 1
    for n in grid_shape:
        grid_cells *= int(n)

    worst_bytes = over[0][0]
    # Scaling the total cell count by how far the worst chunk overshoots the
    # target assumes refinement splits that chunk roughly evenly.  It is an
    # estimate, and the message says so.
    target = min(SUGGESTED_TARGET_BYTES, limit_bytes // 4) or 1
    suggested = _round_up_2sf(grid_cells * worst_bytes / target)

    shown = over[:3]
    detail = "; ".join(
        f"chunk {coord} holds {count:,} vertices = {format_bytes(nbytes)}"
        for nbytes, coord, count in shown
    )
    if len(over) > len(shown):
        detail += f"; and {len(over) - len(shown)} more over the limit"

    chunk_note = f", chunk_shape {tuple(chunk_shape)}" if chunk_shape else ""
    raise ValueError(
        f"spatial grid too coarse for this input: {detail}. A single chunk's "
        f"payload must stay under {format_bytes(limit_bytes)} — the Zarr "
        f"vlen-bytes codec records each cell's length as a uint32, so a larger "
        f"cell writes without error but reads back truncated (compression does "
        f"not help; the length is written before the compressor runs). Current "
        f"grid: {'x'.join(str(int(n)) for n in grid_shape)} = {grid_cells} "
        f"cells{chunk_note}. Re-run with a finer grid — num_chunks "
        f"(--num-chunks) of about {suggested:,} or more, or an explicit "
        f"per-axis triple. The estimate assumes the dense region subdivides "
        f"evenly, so go higher if it is a tight bundle."
    )
