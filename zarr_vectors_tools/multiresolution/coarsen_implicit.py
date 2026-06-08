"""Per-(object, coarsened-chunk) fragment construction for the
``implicit_sequential`` coarsening path.

The streamline / polyline pyramid keeps each surviving object's path
intact at the coarse level by walking the object's source-level vertex
sequence in order, splitting only when the coarsened-chunk-of-the-vertex
changes.  Within each per-chunk run, consecutive source vertices that
fall in the same target bin collapse to a single metavertex
("bin reduction").  The result is a sequence of runs — one per
contiguous coarsened-chunk visit by the object — that become the
fragments at the next level.

Source-level cross-chunk edges whose endpoints both fall in the same
coarsened chunk are absorbed into the merged run for that chunk; ones
that still cross are handled later by remapping endpoints to the new
chunk-local indices.

Pure-functions only — no zarr/store I/O — so the segmentation logic
can be unit-tested directly.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from zarr_vectors.typing import ChunkCoords


def coarse_chunks_of(
    positions: npt.NDArray[np.floating],
    target_chunk_shape: tuple[float, ...],
) -> npt.NDArray[np.int64]:
    """Return ``(N, D)`` int64 array of coarsened-chunk coords per vertex.

    Floor-divides ``positions`` by ``target_chunk_shape``.  Returns an
    empty ``(0, D)`` array when ``positions`` is empty.
    """
    cs = np.asarray(target_chunk_shape, dtype=np.float64)
    if positions.shape[0] == 0:
        return np.zeros((0, len(target_chunk_shape)), dtype=np.int64)
    return np.floor(np.asarray(positions, dtype=np.float64) / cs).astype(np.int64)


def segment_object_by_coarse_chunk(
    mv_per_vertex: npt.NDArray[np.int64],
    coarse_chunk_per_vertex: npt.NDArray[np.int64],
) -> list[tuple[ChunkCoords, list[int]]]:
    """Split one object's vertex walk into per-coarsened-chunk runs.

    Walks the parallel arrays ``mv_per_vertex`` (length N) and
    ``coarse_chunk_per_vertex`` (shape ``(N, D)``) in source-path order.
    Every change in the chunk coord starts a new run.  Within each run,
    consecutive duplicate metavertex ids are collapsed.

    A path that re-enters the same coarsened chunk (e.g. A → B → A)
    produces *two* runs in A — they become two separate fragments at the
    coarse level.

    Args:
        mv_per_vertex: ``(N,)`` int64 metavertex id per source vertex.
        coarse_chunk_per_vertex: ``(N, D)`` int64 target chunk coord per
            source vertex.

    Returns:
        List of ``(chunk_coords, [mv_id, ...])`` tuples in object-path
        order.  Each list of mv ids is non-empty and has no consecutive
        duplicates.  An empty input returns an empty list.
    """
    n = int(mv_per_vertex.shape[0])
    if n == 0:
        return []
    if coarse_chunk_per_vertex.shape[0] != n:
        raise ValueError(
            f"length mismatch: mv_per_vertex {n} != "
            f"coarse_chunk_per_vertex {coarse_chunk_per_vertex.shape[0]}",
        )

    runs: list[tuple[ChunkCoords, list[int]]] = []
    current_chunk: ChunkCoords | None = None
    current_run: list[int] = []

    for i in range(n):
        cc = tuple(int(x) for x in coarse_chunk_per_vertex[i])
        mv = int(mv_per_vertex[i])
        if cc != current_chunk:
            if current_run:
                runs.append((current_chunk, current_run))  # type: ignore[arg-type]
            current_chunk = cc
            current_run = [mv]
        else:
            if not current_run or current_run[-1] != mv:
                current_run.append(mv)

    if current_run:
        runs.append((current_chunk, current_run))  # type: ignore[arg-type]
    return runs


def positions_in_run(
    n: int,
    mv_per_vertex: npt.NDArray[np.int64],
    coarse_chunk_per_vertex: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Return ``(run_idx_per_vertex, position_in_run_per_vertex)``.

    For each source vertex, report:
    * the index of the run it falls in (0-based, within the object's
      ordered run list), and
    * its position inside that run after consecutive-duplicate
      deduplication (so duplicates in a row share the same position).

    The ``runs`` list returned by :func:`segment_object_by_coarse_chunk`
    is mirrored by these two arrays: ``runs[r][1][p]`` is the metavertex
    visited by every source vertex with ``run_idx == r`` and
    ``position_in_run == p``.

    Args:
        n: number of source vertices.
        mv_per_vertex: ``(n,)`` int64.
        coarse_chunk_per_vertex: ``(n, D)`` int64.

    Returns:
        Two ``(n,)`` int64 arrays.
    """
    run_idx = np.empty(n, dtype=np.int64)
    pos_in_run = np.empty(n, dtype=np.int64)

    current_chunk: ChunkCoords | None = None
    cur_run_idx = -1
    cur_pos = -1
    cur_last_mv = -(2**62)

    for i in range(n):
        cc = tuple(int(x) for x in coarse_chunk_per_vertex[i])
        mv = int(mv_per_vertex[i])
        if cc != current_chunk:
            cur_run_idx += 1
            current_chunk = cc
            cur_pos = 0
            cur_last_mv = mv
        else:
            if mv != cur_last_mv:
                cur_pos += 1
                cur_last_mv = mv
            # else: stay on the deduped position
        run_idx[i] = cur_run_idx
        pos_in_run[i] = cur_pos
    return run_idx, pos_in_run


def compute_chunk_fragment_starts(
    n_fragments_per_chunk: dict[ChunkCoords, int],
    fragment_lengths_per_chunk: dict[ChunkCoords, list[int]],
) -> dict[ChunkCoords, list[int]]:
    """Cumulative starts of each fragment in its chunk's flat vertex array.

    Given how many vertices each fragment contributes (in fragment-index
    order), return per-chunk lists of starting offsets so that
    ``chunk_local_vi(fragment_idx, position_in_fragment) =
    starts[chunk][fragment_idx] + position_in_fragment``.

    Args:
        n_fragments_per_chunk: count of fragments per chunk (used only
            to allocate the output list length).
        fragment_lengths_per_chunk: fragment vertex counts in
            fragment-index order, per chunk.

    Returns:
        ``{chunk_coords: [start_0, start_1, ...]}``.
    """
    out: dict[ChunkCoords, list[int]] = {}
    for cc, lengths in fragment_lengths_per_chunk.items():
        starts: list[int] = []
        cum = 0
        for n in lengths:
            starts.append(cum)
            cum += int(n)
        out[cc] = starts
        if cc in n_fragments_per_chunk and len(starts) != n_fragments_per_chunk[cc]:
            raise ValueError(
                f"fragment count mismatch in chunk {cc}: "
                f"{len(starts)} starts vs {n_fragments_per_chunk[cc]} fragments"
            )
    return out
