"""Cross-level link families after a pyramid is rebuilt in part or in whole.

A family is stale once either level it joins is rebuilt.  These pin that
every rebuild path removes the stale ones, writes the new ones, and leaves
the root's cross-level stamp describing what is actually on disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from zarr_vectors.building import (
    get_resolution_level,
    list_link_deltas,
    list_link_offsets,
    list_resolution_levels,
    open_store,
    read_root_metadata,
)
from zarr_vectors.types.points import write_points

from zarr_vectors_tools.compose import merge_stores
from zarr_vectors_tools.multiresolution.coarsen import build_pyramid
from zarr_vectors_tools.multiresolution.refresh import rebuild_pyramid_from_level


def _points(path: Path, *, n: int = 60, seed: int = 7) -> Path:
    positions = np.random.default_rng(seed).uniform(0, 200, size=(n, 3))
    write_points(
        str(path), positions.astype(np.float32),
        chunk_shape=(100.0, 100.0, 100.0),
        bounds=([0.0, 0.0, 0.0], [200.0, 200.0, 200.0]),
        object_ids=np.arange(n, dtype=np.int64),
    )
    return path


def _pyramid(path: Path) -> Path:
    build_pyramid(
        str(_points(path)), factors=[(2, 1), (2, 1)],
        cross_level_storage="explicit", cross_level_depth=2,
    )
    return path


def _families(path: Path) -> dict[int, list[int]]:
    root = open_store(str(path), mode="r")
    out = {}
    for level in list_resolution_levels(root):
        group = get_resolution_level(root, level)
        out[level] = sorted(
            int(d) for d in list_link_deltas(group)
            if int(d) != 0 and list_link_offsets(group, int(d))
        )
    return out


def _stamp(path: Path) -> tuple[int, str]:
    meta = read_root_metadata(open_store(str(path), mode="r"))
    return int(meta.cross_level_depth), str(meta.cross_level_storage)


def _link_files(path: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(path)): p.read_bytes()
        for p in path.rglob("*")
        if p.is_file() and "links" in p.relative_to(path).parts
    }


def test_a_refresh_rewrites_the_links_it_invalidates(tmp_path: Path) -> None:
    # The refresh used to coarsen without cross-level links: the rebuilt
    # levels came back with none while level 0's still pointed into them.
    store = _pyramid(tmp_path / "p.zv")
    families, stamp, files = _families(store), _stamp(store), _link_files(store)
    assert families == {0: [1, 2], 1: [-1, 1], 2: [-2, -1]}

    rebuild_pyramid_from_level(open_store(str(store), mode="r+"), 0)

    assert _families(store) == families
    assert _stamp(store) == stamp
    assert _link_files(store) == files


def test_a_rebuild_above_level_1_drops_level_0_links_reaching_past_it(
    tmp_path: Path,
) -> None:
    # Level 0's +2 points into level 2; rebuilding from level 1 used to keep it.
    store = _pyramid(tmp_path / "p.zv")
    build_pyramid(
        str(store), factors=[(2, 1), (2, 1)],
        cross_level_storage="none", start_level=1,
    )
    assert _families(store) == {0: [1], 1: [-1], 2: []}
    assert _stamp(store) == (1, "explicit")


def test_a_merge_that_drops_the_pyramid_drops_its_links(tmp_path: Path) -> None:
    store = _pyramid(tmp_path / "p.zv")
    merge_stores(str(store), [str(_points(tmp_path / "s.zv", n=5, seed=3))],
                 pyramid="drop")
    assert _families(store) == {0: []}
    assert _stamp(store) == (0, "none")


def test_a_merge_that_rebuilds_the_pyramid_rebuilds_its_links(tmp_path: Path) -> None:
    store = _pyramid(tmp_path / "p.zv")
    merge_stores(str(store), [str(_points(tmp_path / "s.zv", n=5, seed=3))],
                 pyramid="rebuild")
    assert _families(store) == {0: [1, 2], 1: [-1, 1], 2: [-2, -1]}
    assert _stamp(store) == (2, "explicit")


def test_a_split_part_gets_the_links_its_parent_was_built_with(tmp_path: Path) -> None:
    # The part's own root held the format default, so its rebuild wrote
    # links the parent never had (or claimed ones it did not write).
    from zarr_vectors_tools.compose import split_store

    parent = tmp_path / "p.zv"
    build_pyramid(
        str(_points(parent)), factors=[(2, 1), (2, 1)],
        cross_level_storage="implicit", cross_level_depth=2,
    )
    assert _families(parent) == {0: [1, 2], 1: [1], 2: []}

    summary = split_store(
        parent, tmp_path / "parts", by="objects",
        parts={"a": list(range(30)), "b": list(range(30, 60))},
        pyramid="rebuild",
    )
    assert len(summary["written"]) == 2
    for entry in summary["written"]:
        part = Path(entry["path"])
        assert _families(part) == {0: [1, 2], 1: [1], 2: []}
        assert _stamp(part) == (2, "implicit")
