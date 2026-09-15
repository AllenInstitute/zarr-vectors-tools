"""The plain precomputed ingest holds a batch, not the dataset.

``run_ingest_plain`` used to read every skeleton into memory and distribute
them all into a second in-memory copy before writing a chunk.  It now reads
``batch_size`` skeletons at a time and spills each stage to scratch disk.
What has to hold is that the store is the same whatever the batch size, that
no more than a batch is ever read ahead, and that the scratch is removed.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from zarr_vectors_tools.convert.ingest.precomputed_plain_skeletons import (
    PlainSkeletonInfo,
    run_ingest_plain,
)

CHUNK_NM = (1000.0, 1000.0, 1000.0)


class _Reader:
    """Offline stand-in for PlainPrecomputedReader, counting reads in flight."""

    def __init__(self, n_segments: int = 23, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.info = PlainSkeletonInfo(
            base_url="mem://plain",
            transform=np.eye(4),
            vertex_attributes=[{"id": "radius", "data_type": "float32", "num_components": 1}],
        )
        self.segment_ids = [720575940000000000 + 7 * i for i in range(n_segments)]
        self.segment_properties_raw = None
        self._skeletons = {}
        for sid in self.segment_ids:
            n = int(rng.integers(5, 40))
            start = rng.uniform(0, 4000, 3)
            vertices = (start + np.cumsum(rng.normal(0, 150, (n, 3)), axis=0)).astype(np.float32)
            self._skeletons[sid] = {
                "vertices": vertices,
                "edges": np.array([[i, i - 1] for i in range(1, n)], dtype=np.int64),
                "radius": rng.uniform(1, 5, n).astype(np.float32),
            }
        # One segment without geometry, which the ingest must skip.
        self.segment_ids.append(999)
        self.requested: list[int] = []

    def read_skeleton(self, seg_id: int):
        self.requested.append(int(seg_id))
        return self._skeletons.get(int(seg_id))


def _tree(store: Path) -> dict[str, str]:
    out = {}
    for root, _dirs, files in os.walk(store):
        for name in files:
            path = os.path.join(root, name)
            with open(path, "rb") as handle:
                out[os.path.relpath(path, store)] = hashlib.sha256(handle.read()).hexdigest()
    return out


@pytest.mark.parametrize("strides", [[], [2]], ids=["level0", "pyramid"])
def test_the_store_does_not_depend_on_the_batch_size(tmp_path: Path, strides) -> None:
    stores = {}
    for batch_size in (1, 5, 1000):
        store = tmp_path / f"b{batch_size}.zv"
        summary = run_ingest_plain(
            _Reader(), store, chunk_shape_nm=CHUNK_NM, strides=strides,
            batch_size=batch_size, progress=False,
        )
        assert summary["objects"] == 23
        stores[batch_size] = _tree(store)
    assert stores[1] == stores[5] == stores[1000]


def test_every_segment_reads_back(tmp_path: Path) -> None:
    from zarr_vectors.types import skeletons as sk

    reader = _Reader()
    store = tmp_path / "s.zv"
    run_ingest_plain(reader, store, chunk_shape_nm=CHUNK_NM, batch_size=4, progress=False)
    for sid in reader.segment_ids[:-1]:
        got = sk.read_skeleton_by_segment_id(str(store), sid)
        expected = reader._skeletons[sid]["vertices"]
        assert got is not None
        np.testing.assert_allclose(
            np.sort(got["positions"], axis=0), np.sort(expected, axis=0), atol=1e-2,
        )


def test_segments_are_read_in_order_one_batch_at_a_time(tmp_path: Path) -> None:
    reader = _Reader()
    run_ingest_plain(
        reader, tmp_path / "s.zv", chunk_shape_nm=CHUNK_NM, batch_size=4,
        read_workers=1, progress=False,
    )
    assert reader.requested == reader.segment_ids


def test_scratch_is_removed(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    run_ingest_plain(
        _Reader(), tmp_path / "s.zv", chunk_shape_nm=CHUNK_NM, batch_size=3,
        scratch_dir=scratch, progress=False,
    )
    assert list(scratch.iterdir()) == []


def test_batch_size_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        run_ingest_plain(
            _Reader(), tmp_path / "s.zv", chunk_shape_nm=CHUNK_NM, batch_size=0,
            progress=False,
        )
