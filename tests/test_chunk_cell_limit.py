"""A spatial chunk too big for a vlen-bytes cell is refused, not written.

Each per-chunk array is a Zarr v3 ``vlen-bytes`` array, and that codec records
a cell's byte length as a uint32.  A payload of 4 GiB or more therefore wraps
modulo 2**32: numcodecs writes every byte and range-checks nothing, so the
recorded length becomes ``len % 2**32`` and reads hand back a truncated buffer.
On a 10.7 GB tractogram (5.6M streamlines binned onto a 6x6x7 grid, one chunk
taking 415M vertices = 4.6 GiB) that surfaced only hours later, mid-pyramid, as
``cannot reshape array of size 172225607 into shape (3)`` — the truncated
vertex buffer failing ``reshape(-1, 3)``, its length no longer a multiple of 3.

The ingest catches it instead, right after Phase A binning has produced exact
per-chunk counts and before Phase B writes anything.  These tests lower the
ceiling rather than materialise 4 GiB.
"""

from __future__ import annotations

import numpy as np
import pytest

from zarr_vectors_tools.convert.ingest import _cell_limits
from zarr_vectors_tools.convert.ingest._cell_limits import (
    VLEN_CELL_LIMIT_BYTES,
    check_vertex_cell_limit,
)
from zarr_vectors_tools.convert.ingest.trk_parallel import ingest_trk_parallel

from .test_trk_registration import _streamlines_voxmm, _write_radiological_trk


def test_limit_matches_the_uint32_length_header():
    """The ceiling is the uint32 wrap point — not a tunable safety margin."""
    assert VLEN_CELL_LIMIT_BYTES == 2**32


def test_under_limit_passes():
    """A chunk one row short of the ceiling is fine; the check is exclusive."""
    check_vertex_cell_limit(
        {(0, 0, 0): (2**32 // 12) - 1}, ndim=3, itemsize=4, grid_shape=(4, 4, 4),
    )


def test_over_limit_raises_and_names_the_chunk():
    """The message carries the offender, the ceiling, and a finer grid to use."""
    with pytest.raises(ValueError) as exc:
        check_vertex_cell_limit(
            {(1, 2, 3): 40, (0, 0, 0): 5},
            ndim=3,
            itemsize=4,
            grid_shape=(2, 2, 2),
            chunk_shape=(10.0, 10.0, 10.0),
            limit_bytes=256,
        )
    msg = str(exc.value)
    assert "(1, 2, 3)" in msg
    assert "(0, 0, 0)" not in msg, "under-limit chunks must not be reported"
    assert "num_chunks" in msg


def test_suggestion_scales_past_the_current_grid():
    """The suggested grid is coarser-input-aware: bigger overshoot, more cells."""
    def suggest(count):
        with pytest.raises(ValueError) as exc:
            check_vertex_cell_limit(
                {(0, 0, 0): count}, ndim=3, itemsize=4,
                grid_shape=(4, 4, 4), limit_bytes=4096,
            )
        return int(str(exc.value).split("about ")[1].split()[0].replace(",", ""))

    small, large = suggest(1_000), suggest(10_000)
    assert large > small > 64, "suggestion must exceed the current 4x4x4 grid"


def test_the_boundary_itself_is_refused():
    """A payload landing exactly on the ceiling is refused, not allowed — the
    uint32 wraps to 0 there, so an off-by-one is a silently corrupt store."""
    with pytest.raises(ValueError):
        check_vertex_cell_limit(
            {(0, 0, 0): 100}, ndim=3, itemsize=4,
            grid_shape=(4, 4, 4), limit_bytes=1200,
        )
    check_vertex_cell_limit(
        {(0, 0, 0): 99}, ndim=3, itemsize=4,
        grid_shape=(4, 4, 4), limit_bytes=1200,
    )


def test_ingest_refuses_before_writing_level_zero(tmp_path, monkeypatch):
    """End-to-end: the ingest fails during binning, leaving no vertex cells."""
    trk = tmp_path / "dense.trk"
    _write_radiological_trk(trk, _streamlines_voxmm(n=40, npts=30))
    out = tmp_path / "dense.zv"

    # 512 B per cell = 42 vertices.  num_chunks=8 puts ~300 of the fixture's
    # 1200 vertices in each of 8 cells, so this stands in for the 4 GiB wrap.
    monkeypatch.setattr(_cell_limits, "VLEN_CELL_LIMIT_BYTES", 512)

    with pytest.raises(ValueError, match="spatial grid too coarse"):
        ingest_trk_parallel(
            str(trk), str(out), num_chunks=8, workers=1,
            build_multiscale=False, progress=False,
        )

    # Phase B never ran: the arrays exist (created up front) but hold no cells.
    cells = list((out / "0" / "vertices" / "c").rglob("*")) if out.exists() else []
    assert not [p for p in cells if p.is_file()]


# A full TRK ingest (the refusal above stops during binning, so it stays fast).
@pytest.mark.slow
def test_finer_grid_ingests_the_same_input(tmp_path, monkeypatch):
    """The advice works: the same input under the same lowered ceiling
    succeeds once the grid is fine enough."""
    trk = tmp_path / "dense.trk"
    _write_radiological_trk(trk, _streamlines_voxmm(n=40, npts=30))
    monkeypatch.setattr(_cell_limits, "VLEN_CELL_LIMIT_BYTES", 512)

    ingest_trk_parallel(
        str(trk), str(tmp_path / "fine.zv"), num_chunks=4096, workers=1,
        build_multiscale=False, progress=False,
    )
    cells = (tmp_path / "fine.zv" / "0" / "vertices" / "c").rglob("*")
    assert [p for p in cells if p.is_file()]


@pytest.mark.slow
def test_checked_counts_are_the_bytes_actually_written(tmp_path, monkeypatch):
    """The guard is only as good as its prediction: every count it inspects
    must equal the cell payload Phase B goes on to write, boundary-split
    duplicates included.  A refactor that drifts these apart would let a
    real overflow through."""
    import zarr_vectors_tools.convert.ingest.trk_parallel as tp

    trk = tmp_path / "src.trk"
    _write_radiological_trk(trk, _streamlines_voxmm(n=60, npts=25))
    out = tmp_path / "src.zv"

    seen: dict[tuple[int, int, int], int] = {}
    real = tp.check_vertex_cell_limit
    monkeypatch.setattr(
        tp, "check_vertex_cell_limit",
        lambda counts, **kw: (seen.update(counts), real(counts, **kw))[1],
    )

    ingest_trk_parallel(
        str(trk), str(out), num_chunks=64, workers=1,
        build_multiscale=False, progress=False,
    )

    assert seen, "the check never ran"
    for coord, count in seen.items():
        cell = out.joinpath("0", "vertices", "c", *(str(c) for c in coord))
        # vlen-bytes cell framing: uint32 item count + uint32 payload length.
        assert cell.stat().st_size - 8 == count * 3 * 4, f"chunk {coord}"


def test_wrap_is_real_not_theoretical():
    """Pin the arithmetic the guard exists for: 4,983,869,724 bytes of float32
    xyz wraps to 172,225,607 elements, which is not a multiple of 3."""
    payload = 415_322_477 * 3 * np.dtype(np.float32).itemsize
    assert payload >= VLEN_CELL_LIMIT_BYTES
    wrapped_elements = (payload % 2**32) // 4
    assert wrapped_elements == 172_225_607
    assert wrapped_elements % 3 == 2, "truncation is what breaks reshape(-1, 3)"
