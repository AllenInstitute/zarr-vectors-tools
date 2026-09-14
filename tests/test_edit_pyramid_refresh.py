"""Tests for tools' ``rebuild_pyramid_from_level`` (post-edit refresh).

After a vertex edit at level 0 the coarser pyramid levels are stale.
Calling ``zarr_vectors_tools.multiresolution.refresh.rebuild_pyramid_from_level``
re-coarsens every level above the edited one with the tools' rich
coarseners, leaving the post-refresh pyramid equivalent (within float
tolerance) to a from-scratch ``build_pyramid`` against the post-edit
level-0 data.

Note: core's ``EditSession(refresh_pyramid=...)`` runs core's *basic*
refresher, not these tools' strategies, so the rich refresh is invoked
explicitly here after the edit is committed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr_vectors as zv
from zarr_vectors.building import list_resolution_levels, open_store, read_chunk_vertices
from zarr_vectors.types.points import write_points

from zarr_vectors_tools.multiresolution.coarsen import build_pyramid
from zarr_vectors_tools.multiresolution.refresh import rebuild_pyramid_from_level


def _build_pyramid_store(tmp_path: Path) -> tuple[str, np.ndarray]:
    """Multi-chunk store with a 1-level coarsened pyramid built on
    top of 32 random points spanning a 4×4×4 chunk grid.
    """
    path = tmp_path / "store.zv"
    rng = np.random.default_rng(7)
    positions = rng.uniform(0, 200, size=(32, 3)).astype(np.float32)
    write_points(
        str(path),
        positions,
        chunk_shape=(50.0, 50.0, 50.0),
        bounds=([0.0, 0.0, 0.0], [200.0, 200.0, 200.0]),
        object_ids=np.arange(len(positions), dtype=np.int64),
    )
    build_pyramid(
        str(path),
        factors=[(2.0, 1.0)],     # one coarser level, 2× bin reduction
        cross_level_storage="none",
    )
    return str(path), positions


class TestRefreshDirty:

    def test_refresh_false_reports_dirty_levels(
        self, tmp_path: Path,
    ) -> None:
        path, _ = _build_pyramid_store(tmp_path)
        # Through the data-oriented API: an edit names an object and an
        # index within it, not a chunk and a fragment.
        with zv.open(path, mode="r+").editing(in_place=True) as edit:
            edit.move_vertex(object=0, index=0, to=[5.0, 5.0, 5.0])
        # Levels above 0 should be reported dirty.
        assert 1 in edit.report.dirty_pyramid_levels


class TestRefreshBatch:

    def test_tools_refresh_after_edit_rebuilds_level1(
        self, tmp_path: Path,
    ) -> None:
        path, _ = _build_pyramid_store(tmp_path)

        # Edit level 0 without core-driven refresh, then re-coarsen the
        # pyramid explicitly with the tools' rich refresher.  in_place=True
        # is EditPlan's spelling of the session's atomic=False;
        # refresh_pyramid is forwarded to the session unchanged.
        with zv.open(path, mode="r+").editing(
            in_place=True, refresh_pyramid=False,
        ) as edit:
            edit.move_vertex(object=0, index=0, to=[5.0, 5.0, 5.0])

        root = open_store(path, mode="r+")
        rebuild_pyramid_from_level(root, source_level=0)

        # Level 1 must still exist and have at least one vertex.
        root = open_store(path, mode="r+")
        assert 1 in list_resolution_levels(root)
        level1 = root["1"]
        total = 0
        from zarr_vectors.building import list_chunk_keys
        for cc in list_chunk_keys(level1):
            groups = read_chunk_vertices(
                level1, cc, dtype=np.float32, ndim=3,
            )
            for g in groups:
                total += g.shape[0]
        assert total > 0


class TestRefreshFunction:

    def test_rebuild_pyramid_from_level_matches_fresh_build(
        self, tmp_path: Path,
    ) -> None:
        """A refresh after no edits must leave level 1 identical to a
        from-scratch ``build_pyramid`` (within float tolerance).
        """
        path, _ = _build_pyramid_store(tmp_path)
        root = open_store(path, mode="r+")

        # Capture level-1 vertex count before refresh.
        before = _level_vertices(root, 1)

        rebuild_pyramid_from_level(root, source_level=0)
        # Re-open for fresh reads.
        root = open_store(path, mode="r+")
        after = _level_vertices(root, 1)
        assert before.shape == after.shape
        # Centroids are deterministic given a fixed assignment so the
        # post-refresh data should match the pre-refresh data exactly.
        np.testing.assert_allclose(
            np.sort(before, axis=0),
            np.sort(after, axis=0),
            atol=1e-4,
        )


def _level_vertices(root, level: int) -> np.ndarray:
    """Return all vertices at ``level`` concatenated into ``(N, D)``."""
    from zarr_vectors.building import list_chunk_keys
    lg = root[str(level)]
    out: list[np.ndarray] = []
    for cc in list_chunk_keys(lg):
        for g in read_chunk_vertices(lg, cc, dtype=np.float32, ndim=3):
            out.append(g)
    if not out:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(out, axis=0)
