"""``zvtools convert --cross-level-storage/--cross-level-depth``.

The flags existed only on ``zvtools pyramid``, so a one-command convert of a
point cloud always paid for explicit cross-level links -- most of the build
on a large store.  These check the flags reach the pyramid, and are refused
where they would be ignored.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from zarr_vectors.building import get_resolution_level, list_link_deltas, open_store

from zarr_vectors_tools.cli import main


def _points_csv(path: Path, n: int = 400) -> Path:
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 100, size=(n, 3))
    path.write_text("x,y,z\n" + "\n".join(f"{a},{b},{c}" for a, b, c in pts) + "\n")
    return path


def _deltas(store: Path, level: int) -> list[int]:
    return list_link_deltas(get_resolution_level(open_store(str(store)), level))


def test_none_skips_the_cross_level_links(tmp_path: Path) -> None:
    src = _points_csv(tmp_path / "p.csv")
    linked, bare = tmp_path / "linked.zv", tmp_path / "bare.zv"
    common = ["--chunk-shape", "25,25,25", "--coarsen", "2", "--sparsity", "1"]
    assert main(["convert", str(src), str(linked), *common]) == 0
    assert main(["convert", str(src), str(bare), *common,
                 "--cross-level-storage", "none"]) == 0
    assert 1 in _deltas(linked, 0)
    assert 1 not in _deltas(bare, 0)


def test_implicit_writes_only_the_fine_side(tmp_path: Path) -> None:
    src = _points_csv(tmp_path / "p.csv")
    out = tmp_path / "implicit.zv"
    assert main(["convert", str(src), str(out), "--chunk-shape", "25,25,25",
                 "--coarsen", "2", "--sparsity", "1",
                 "--cross-level-storage", "implicit"]) == 0
    assert 1 in _deltas(out, 0)
    assert -1 not in _deltas(out, 1)


def test_the_flags_need_a_pyramid(tmp_path: Path, capsys) -> None:
    src = _points_csv(tmp_path / "p.csv")
    with pytest.raises(SystemExit, match="--coarsen and --sparsity"):
        main(["convert", str(src), str(tmp_path / "o.zv"), "--chunk-shape", "25,25,25",
              "--cross-level-storage", "none"])


def test_the_flags_are_refused_on_export(tmp_path: Path) -> None:
    src = _points_csv(tmp_path / "p.csv")
    store = tmp_path / "s.zv"
    assert main(["convert", str(src), str(store), "--chunk-shape", "25,25,25"]) == 0
    with pytest.raises(SystemExit, match="--cross-level"):
        main(["convert", str(store), str(tmp_path / "o.csv"), "--cross-level-depth", "2"])
