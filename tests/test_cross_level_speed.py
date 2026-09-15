"""Cross-level link emission: array-form placement, batched writes, counts.

The ``±N`` families used to go through core's record-shaped writers --
``write_links`` (a Python object per record, a synchronous write plus a
presence read-modify-write per cell) followed by ``finalize_links`` (which
decodes every cell again to count rows).  They are now placed as arrays and
written in batched blocks, with the counts taken from what was written.
These pin that the families still describe the parent map exactly and stay
self-consistent, and that a rebuild over existing families (the path that
still recounts from the store) lands on the same bytes as a fresh build.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from zarr_vectors.building import (
    get_resolution_level,
    list_link_deltas,
    list_link_offsets,
    list_resolution_levels,
    open_store,
    read_level_metadata,
    read_links,
)
from zarr_vectors.types.points import write_points

from zarr_vectors_tools.multiresolution import coarsen as coarsen_module
from zarr_vectors_tools.multiresolution.coarsen import build_pyramid


def _points_store(path: Path, *, lo: float = 0.0, n: int = 3000) -> None:
    rng = np.random.default_rng(11)
    write_points(
        str(path),
        rng.uniform(lo, lo + 800.0, size=(n, 3)).astype(np.float32),
        chunk_shape=(100.0, 100.0, 100.0), bin_shape=(25.0, 25.0, 25.0),
    )


def _tree_sha(root: Path) -> dict[str, str]:
    out = {}
    for base, _dirs, files in os.walk(root):
        for name in files:
            fp = Path(base) / name
            out[str(fp.relative_to(root))] = hashlib.sha256(fp.read_bytes()).hexdigest()
    return out


def _families(store: Path):
    root = open_store(str(store), mode="r+")
    for level in sorted(list_resolution_levels(root)):
        lg = get_resolution_level(root, level)
        for delta in list_link_deltas(lg):
            if delta != 0:
                yield level, delta, lg


class TestCrossLevelFamilies:

    @pytest.mark.parametrize("chunk_scale", [1, 2])
    def test_counts_presence_and_mirror(self, tmp_path: Path, chunk_scale: int) -> None:
        store = tmp_path / "p.zv"
        _points_store(store, lo=-400.0)
        build_pyramid(
            str(store), factors=[(2.0, 1.0), (2.0, 1.0)],
            chunk_scale_factors=[chunk_scale, chunk_scale],
            cross_level_depth=2, cross_level_storage="explicit",
        )
        records: dict[tuple[int, int], list] = {}
        for level, delta, lg in _families(store):
            recs = read_links(lg, delta=delta)
            records[(level, delta)] = recs
            meta = lg.read_array_meta(f"links/{delta:+d}")
            assert meta["num_links"] == len(recs) == meta["num_physical_records"]
            for seg in list_link_offsets(lg, delta):
                name = f"links/{delta:+d}/{seg}"
                stamped = list(lg.list_chunks(name))
                assert stamped == lg.derive_nonempty_chunks(name)
        assert {(0, 1), (1, -1), (1, 1), (2, -1), (0, 2), (2, -2)} <= set(records)
        for (level, delta), recs in records.items():
            if delta < 0:
                continue
            mirror = records[(level + delta, -delta)]
            assert sorted((b, a) for a, b in recs) == sorted(mirror)

    def test_one_record_per_vertex_on_aligned_grids(self, tmp_path: Path) -> None:
        store = tmp_path / "p.zv"
        _points_store(store)
        build_pyramid(str(store), factors=[(4.0, 1.0)])
        root = open_store(str(store))
        n0 = read_level_metadata(root, 0).vertex_count
        plus = read_links(get_resolution_level(root, 0), delta=1)
        assert len(plus) == n0
        assert len({ep0 for ep0, _ in plus}) == n0

    def test_fresh_build_skips_the_recount(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Counts come from the rows written; nothing decodes them back."""
        calls: list[int] = []
        original = coarsen_module.finalize_links

        def spy(level_group, *, delta=0):
            if delta != 0:
                calls.append(delta)
            return original(level_group, delta=delta)

        monkeypatch.setattr(coarsen_module, "finalize_links", spy)
        store = tmp_path / "p.zv"
        _points_store(store)
        build_pyramid(
            str(store), factors=[(2.0, 1.0), (2.0, 1.0)], chunk_scale_factors=[2, 2],
            cross_level_depth=2,
        )
        assert calls == []

    def test_rebuild_over_existing_families_is_byte_identical(
        self, tmp_path: Path,
    ) -> None:
        fresh = tmp_path / "fresh.zv"
        _points_store(fresh)
        kwargs = dict(
            factors=[(2.0, 1.0), (2.0, 1.0)], chunk_scale_factors=[2, 2],
            cross_level_depth=2,
        )
        build_pyramid(str(fresh), **kwargs)
        again = tmp_path / "again.zv"
        shutil.copytree(fresh, again)
        build_pyramid(str(again), **kwargs)
        assert _tree_sha(again) == _tree_sha(fresh)
