"""The in-memory streamline readers must produce a store the pyramid accepts.

``zvtools convert tracts.tck out.zv --coarsen 8 --sparsity 1`` ingested
cleanly and then failed at the pyramid step with *"requires
fragment_attributes/segment_id on the source level"* -- after the
expensive half of the job had already succeeded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from zarr_vectors.building import (
    get_resolution_level,
    open_store,
    read_all_object_manifests,
    read_chunk_fragment_attributes,
)

from zarr_vectors_tools.multiresolution.coarsen import build_pyramid

CHUNK = (10.0, 10.0, 10.0)


def _write_tck(path: Path, streamlines: list[np.ndarray]) -> Path:
    """A minimal MRtrix .tck: text header, then float32 triples."""
    header = (
        "mrtrix tracks\n"
        "datatype: Float32LE\n"
        f"count: {len(streamlines)}\n"
    )
    offset_line = "file: . {}\n"
    # The offset has to point past the whole header, including itself.
    body_start = len(header) + len(offset_line.format(0)) + len("END\n")
    for _ in range(3):
        candidate = len(header) + len(offset_line.format(body_start)) + len("END\n")
        if candidate == body_start:
            break
        body_start = candidate
    with open(path, "wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write((offset_line.format(body_start)).encode("ascii"))
        handle.write(b"END\n")
        for line in streamlines:
            handle.write(np.asarray(line, dtype="<f4").tobytes())
            handle.write(np.array([np.nan] * 3, dtype="<f4").tobytes())
        handle.write(np.array([np.inf] * 3, dtype="<f4").tobytes())
    return path


def _assert_coarsenable(store: Path, n_objects: int) -> None:
    level = get_resolution_level(open_store(str(store)), 0)
    manifests = read_all_object_manifests(level)
    assert len(manifests) == n_objects
    for oid, manifest in enumerate(manifests):
        for chunk, fragment_index in manifest:
            column = read_chunk_fragment_attributes(
                level, "segment_id", tuple(int(c) for c in chunk),
                dtype=np.uint64,
            )
            assert column is not None, "no segment_id on the ingested level"
            assert int(np.asarray(column).ravel()[fragment_index]) == oid

    # The real check: the coarsener accepts the store the ingest wrote.
    build_pyramid(
        str(store), factors=[(2.0, 1.0)], coarsen_mode="decimate",
        cross_level_depth=0, cross_level_storage="none",
    )
    coarse = read_all_object_manifests(
        get_resolution_level(open_store(str(store)), 1),
    )
    assert sum(1 for m in coarse if m) == n_objects


def _streamlines() -> list[np.ndarray]:
    return [
        np.column_stack([
            np.linspace(1.0, 25.0, 12),
            np.full(12, 5.0),
            np.full(12, 5.0),
        ]).astype(np.float32),
        np.column_stack([
            np.linspace(2.0, 18.0, 9),
            np.full(9, 6.0),
            np.full(9, 6.0),
        ]).astype(np.float32),
    ]


def test_tck_ingest_produces_a_coarsenable_store(tmp_path: Path) -> None:
    pytest.importorskip("nibabel")
    from zarr_vectors_tools.ingest.tck import ingest_tck

    source = _write_tck(tmp_path / "t.tck", _streamlines())
    store = tmp_path / "tck.zv"
    ingest_tck(str(source), str(store), CHUNK)
    _assert_coarsenable(store, 2)


def test_trk_ingest_produces_a_coarsenable_store(tmp_path: Path) -> None:
    nib = pytest.importorskip("nibabel")
    from zarr_vectors_tools.ingest.trk import ingest_trk

    source = tmp_path / "t.trk"
    tractogram = nib.streamlines.Tractogram(
        _streamlines(), affine_to_rasmm=np.eye(4),
    )
    nib.streamlines.save(tractogram, str(source))
    store = tmp_path / "trk.zv"
    ingest_trk(str(source), str(store), CHUNK)
    _assert_coarsenable(store, 2)
