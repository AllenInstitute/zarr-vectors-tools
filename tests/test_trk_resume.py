"""A parallel TRK ingest that dies can be resumed, not restarted.

On a whole-brain tractogram the offset scan and Phase A take hours, and the
pyramid as long again; a run that died in either used to start over from the
file scan.  With ``intermediate_dir`` each stage records itself, and
``resume=True`` skips what is recorded and still on disk.  Each test here
kills a run at one stage by making that stage raise, resumes, and checks
both that the finished stages were not run again and that the store matches
an ingest that never stopped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("nibabel")

from nibabel.streamlines import Field, Tractogram  # noqa: E402
from nibabel.streamlines.trk import TrkFile  # noqa: E402
from zarr_vectors.building import list_resolution_levels, open_store  # noqa: E402
from zarr_vectors.exceptions import IngestError  # noqa: E402
from zarr_vectors.types.polylines import read_polylines  # noqa: E402

from zarr_vectors_tools.convert.ingest import trk_parallel  # noqa: E402
from zarr_vectors_tools.multiresolution import coarsen  # noqa: E402

FACTORS = [(2.0, 1.0), (2.0, 1.0)]


class _Killed(RuntimeError):
    pass


@pytest.fixture
def trk(tmp_path: Path) -> Path:
    rng = np.random.default_rng(5)
    streamlines = [
        np.cumsum(rng.normal(0, 3, (int(rng.integers(8, 30)), 3)), axis=0).astype(np.float32)
        + rng.uniform(20, 80, 3).astype(np.float32)
        for _ in range(60)
    ]
    path = tmp_path / "in.trk"
    header = {
        Field.VOXEL_TO_RASMM: np.eye(4, dtype=np.float32),
        Field.VOXEL_SIZES: (1.0, 1.0, 1.0),
        Field.DIMENSIONS: (100, 100, 100),
        Field.VOXEL_ORDER: "RAS",
    }
    TrkFile(Tractogram(streamlines, affine_to_rasmm=np.eye(4)), header=header).save(str(path))
    return path


def _ingest(trk: Path, out: Path, scratch: Path | None, **kw):
    return trk_parallel.ingest_trk_parallel(
        str(trk), str(out), num_chunks=27, n_parts=3, workers=1,
        pyramid_factors=FACTORS, intermediate_dir=None if scratch is None else str(scratch),
        progress=False, **kw,
    )


def _contents(store: Path, level: int = 0):
    result = read_polylines(str(store), level=level)
    return sorted(
        tuple(np.round(np.concatenate(p), 3).ravel().tolist()) for p in result["polylines"]
    )


def _refuse(name: str):
    def _called(*_a, **_k):
        raise AssertionError(f"{name} ran again on resume")
    return _called


@pytest.mark.slow
def test_a_run_killed_in_phase_b_resumes_without_phase_a(
    trk: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = tmp_path / "reference.zv"
    _ingest(trk, reference, None)

    out, scratch = tmp_path / "out.zv", tmp_path / "scratch"
    real_b = trk_parallel._phase_b_worker
    calls = {"n": 0}

    def _dies_midway(chunk, shared):
        calls["n"] += 1
        if calls["n"] > 3:
            raise _Killed("phase B")
        return real_b(chunk, shared)

    monkeypatch.setattr(trk_parallel, "_phase_b_worker", _dies_midway)
    with pytest.raises(_Killed):
        _ingest(trk, out, scratch)
    assert out.exists(), "the interrupted run should have left a partial store"

    monkeypatch.setattr(trk_parallel, "_phase_b_worker", real_b)
    monkeypatch.setattr(trk_parallel, "_phase_a_worker", _refuse("Phase A"))
    monkeypatch.setattr(trk_parallel, "build_offset_index", _refuse("the offset scan"))
    _ingest(trk, out, scratch, resume=True)

    for level in (0, 1, 2):
        assert _contents(out, level) == _contents(reference, level)


@pytest.mark.slow
def test_a_run_killed_in_the_pyramid_keeps_finished_levels(
    trk: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = tmp_path / "reference.zv"
    _ingest(trk, reference, None)

    out, scratch = tmp_path / "out.zv", tmp_path / "scratch"
    real_level = coarsen.coarsen_level
    built: list[int] = []

    def _dies_at_level_2(*args, target_level, **kw):
        if target_level == 2:
            real_level(*args, target_level=target_level, **kw)   # half-written
            raise _Killed("pyramid")
        built.append(target_level)
        return real_level(*args, target_level=target_level, **kw)

    monkeypatch.setattr(coarsen, "coarsen_level", _dies_at_level_2)
    with pytest.raises(_Killed):
        _ingest(trk, out, scratch)
    assert 2 in list_resolution_levels(open_store(str(out)))

    built.clear()
    monkeypatch.setattr(coarsen, "coarsen_level", lambda *a, **k: (
        built.append(k["target_level"]) or real_level(*a, **k)
    ))
    monkeypatch.setattr(trk_parallel, "_phase_a_worker", _refuse("Phase A"))
    monkeypatch.setattr(trk_parallel, "_phase_b_worker", _refuse("Phase B"))
    _ingest(trk, out, scratch, resume=True)

    assert built == [2], "level 1 was finished and should have been kept"
    for level in (0, 1, 2):
        assert _contents(out, level) == _contents(reference, level)


@pytest.mark.slow
def test_a_finished_run_resumes_to_the_same_store_doing_nothing(
    trk: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, scratch = tmp_path / "out.zv", tmp_path / "scratch"
    first = _ingest(trk, out, scratch)
    monkeypatch.setattr(trk_parallel, "_phase_a_worker", _refuse("Phase A"))
    monkeypatch.setattr(trk_parallel, "_phase_b_worker", _refuse("Phase B"))
    monkeypatch.setattr(coarsen, "coarsen_level", _refuse("the pyramid"))
    again = _ingest(trk, out, scratch, resume=True)
    assert again["streamline_count"] == first["streamline_count"]
    assert again["cross_chunk_link_count"] == first["cross_chunk_link_count"]


@pytest.mark.slow
def test_different_options_are_refused(trk: Path, tmp_path: Path) -> None:
    out, scratch = tmp_path / "out.zv", tmp_path / "scratch"
    _ingest(trk, out, scratch, build_multiscale=False)
    with pytest.raises(IngestError, match="dtype"):
        _ingest(trk, out, scratch, build_multiscale=False, resume=True, dtype="float64")


def test_resume_needs_an_intermediate_dir(trk: Path, tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="intermediate_dir"):
        _ingest(trk, tmp_path / "out.zv", None, resume=True)


@pytest.mark.slow
def test_a_store_that_is_not_ours_is_left_alone(trk: Path, tmp_path: Path) -> None:
    out, scratch = tmp_path / "out.zv", tmp_path / "fresh"
    _ingest(trk, out, None, build_multiscale=False)
    marker = out / "zarr.json"
    before = marker.read_bytes()
    with pytest.raises(Exception):  # noqa: B017 - the store writer's own refusal
        _ingest(trk, out, scratch, build_multiscale=False, resume=True)
    assert marker.read_bytes() == before
