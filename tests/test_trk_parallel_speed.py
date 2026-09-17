"""The vectorised TRK ingest paths give the per-streamline answers exactly.

Phase A bins whole blocks of streamlines at once and Phase B writes batches
of chunks; neither may change a single stored byte.  These pin the two
places where that is not automatic: a streamline's length (numpy's pairwise
sum rounds differently over different runs) and the block / batch
boundaries themselves.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from zarr_vectors_tools.convert.ingest import trk_parallel


def test_line_lengths_match_a_per_streamline_sum() -> None:
    rng = np.random.default_rng(3)
    counts = np.concatenate([
        [1, 2, 3, 8, 9, 17, 129, 130, 257, 1025], rng.integers(1, 600, 300),
    ]).astype(np.int64)
    first = np.concatenate([[0], np.cumsum(counts)[:-1]])
    verts = rng.normal(0, 40, (int(counts.sum()), 3)).astype(np.float32)
    got = trk_parallel._line_lengths(verts, first, counts)
    for i, (a, n) in enumerate(zip(first.tolist(), counts.tolist())):
        line = verts[a:a + n]
        expected = (
            np.float32(np.sum(np.linalg.norm(np.diff(line, axis=0), axis=1)))
            if n >= 2 else np.float32(0)
        )
        assert got[i].tobytes() == expected.tobytes(), f"streamline {i} ({n} points)"


def _tree(store: Path) -> dict[str, str]:
    out = {}
    for root, _dirs, files in os.walk(store):
        for name in files:
            path = Path(root) / name
            out[str(path.relative_to(store))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


@pytest.mark.slow
def test_block_and_batch_boundaries_do_not_change_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("nibabel")
    from nibabel.streamlines import Field, Tractogram
    from nibabel.streamlines.trk import TrkFile

    rng = np.random.default_rng(11)
    streamlines = [
        (np.cumsum(rng.normal(0, 4, (int(rng.integers(1, 40)), 3)), axis=0)
         + rng.uniform(-30, 90, 3)).astype(np.float32)
        for _ in range(120)
    ]
    trk = tmp_path / "in.trk"
    header = {
        Field.VOXEL_TO_RASMM: np.eye(4, dtype=np.float32),
        Field.VOXEL_SIZES: (1.0, 1.0, 1.0),
        Field.DIMENSIONS: (100, 100, 100),
        Field.VOXEL_ORDER: "RAS",
    }
    TrkFile(Tractogram(streamlines, affine_to_rasmm=np.eye(4)), header=header).save(str(trk))

    def ingest(out: Path) -> dict[str, str]:
        trk_parallel.ingest_trk_parallel(
            str(trk), str(out), num_chunks=64, n_parts=4, workers=1,
            object_attrs={"length", "endpoints", "vertex_count"},
            vertex_attrs={"arc_length", "x", "random"},
            build_multiscale=False, progress=False,
        )
        return _tree(out)

    reference = ingest(tmp_path / "reference.zv")
    # Blocks of a few vertices (most streamlines their own block, several
    # split across none) and parts' worth of single-chunk Phase B tasks.
    monkeypatch.setattr(trk_parallel, "_PHASE_A_BLOCK_VERTICES", 7)
    small = ingest(tmp_path / "small_blocks.zv")
    assert small == reference
