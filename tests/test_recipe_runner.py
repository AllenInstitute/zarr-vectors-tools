"""``zvtools run`` carries a store through several steps, and resumes.

A recipe step is a subcommand with its options written as a mapping; what
matters is that it becomes the argument list that subcommand accepts, that
an option it does not have is refused by name, and that a rerun skips what
already finished and redoes what changed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from zarr_vectors.building import list_resolution_levels, open_store

from zarr_vectors_tools.cli import build_parser, main
from zarr_vectors_tools.cli.pipeline import RecipeError, step_argv


def _write_csv(path: Path, n: int = 200) -> Path:
    rng = np.random.default_rng(0)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "z"])
        for row in rng.uniform(0, 400, size=(n, 3)):
            writer.writerow([f"{v:.3f}" for v in row])
    return path


def _recipe(tmp_path: Path, steps: list, name: str = "recipe.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps({"name": "points", "steps": steps}), encoding="utf-8")
    return path


def _points_steps(shard_shape: int = 2) -> list:
    return [
        {"convert": {
            "input": "cells.csv", "output": "cells.zarrvectors",
            "chunk_shape": [100, 100, 100], "coarsen": [2], "sparsity": [2],
        }},
        {"validate": {"store": "cells.zarrvectors"}},
        {"shard": {"store": "cells.zarrvectors", "shape": shard_shape}},
    ]


def test_a_recipe_builds_a_validated_sharded_pyramid(tmp_path: Path) -> None:
    _write_csv(tmp_path / "cells.csv")
    recipe = _recipe(tmp_path, _points_steps())

    assert main(["run", str(recipe)]) == 0

    store = tmp_path / "cells.zarrvectors"
    assert len(list_resolution_levels(open_store(str(store)))) == 2
    state = json.loads((tmp_path / "recipe.run.json").read_text())
    assert [s["status"] for s in state["steps"].values()] == ["done", "done", "done"]


def test_a_rerun_skips_finished_steps_and_redoes_changed_ones(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _write_csv(tmp_path / "cells.csv")
    recipe = _recipe(tmp_path, _points_steps())
    assert main(["run", str(recipe)]) == 0
    capsys.readouterr()

    assert main(["run", str(recipe)]) == 0
    assert capsys.readouterr().out.count("done earlier, skipped") == 3

    _recipe(tmp_path, _points_steps(shard_shape=4))
    assert main(["run", str(recipe)]) == 0
    out = capsys.readouterr().out
    assert out.count("done earlier, skipped") == 2
    assert "[3/3] shard: zvtools shard" in out


def test_a_missing_output_reruns_its_step_and_everything_after(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    import shutil

    _write_csv(tmp_path / "cells.csv")
    recipe = _recipe(tmp_path, _points_steps())
    assert main(["run", str(recipe)]) == 0
    shutil.rmtree(tmp_path / "cells.zarrvectors")
    capsys.readouterr()

    assert main(["run", str(recipe)]) == 0
    assert "skipped" not in capsys.readouterr().out


def test_a_failing_step_stops_the_run_and_is_retried(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    recipe = _recipe(tmp_path, _points_steps())
    assert main(["run", str(recipe)]) != 0          # cells.csv does not exist yet
    state = json.loads((tmp_path / "recipe.run.json").read_text())
    assert state["steps"]["1"]["status"] == "failed"
    assert "2" not in state["steps"]

    _write_csv(tmp_path / "cells.csv")
    capsys.readouterr()
    assert main(["run", str(recipe)]) == 0
    assert "skipped" not in capsys.readouterr().out


def test_force_and_from(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_csv(tmp_path / "cells.csv")
    recipe = _recipe(tmp_path, _points_steps())
    assert main(["run", str(recipe)]) == 0
    capsys.readouterr()

    assert main(["run", str(recipe), "--from", "2"]) == 0
    assert capsys.readouterr().out.count("done earlier, skipped") == 1
    assert main(["run", str(recipe), "--force"]) == 0
    assert "skipped" not in capsys.readouterr().out


class TestSteps:

    def test_lists_become_commas_or_repeated_flags(self, tmp_path: Path) -> None:
        argv = step_argv(build_parser(), "convert", {
            "input": "tracts.zarrvectors", "output": "out.trk",
            "attribute": ["fa", "md"], "object_id": [3, 5], "level": 1,
        }, tmp_path, 0)
        assert argv[:3] == [
            "convert", str((tmp_path / "tracts.zarrvectors").resolve()),
            str((tmp_path / "out.trk").resolve()),
        ]
        assert argv[3:] == [
            "--attribute", "fa", "--attribute", "md",
            "--object-id", "3", "--object-id", "5", "--level", "1",
        ]

        argv = step_argv(build_parser(), "pyramid", {
            "store": "gs://bucket/s.zarrvectors", "coarsen": [8, 8], "sparsity": [1, 4],
        }, tmp_path, 1)
        assert argv == [
            "pyramid", "gs://bucket/s.zarrvectors", "--coarsen", "8,8", "--sparsity", "1,4",
        ]

    def test_switches(self, tmp_path: Path) -> None:
        argv = step_argv(build_parser(), "convert", {
            "input": "a.csv", "output": "b.zv", "overwrite": True, "compute_length": False,
        }, tmp_path, 0)
        assert "--overwrite" in argv and "--compute-length" not in argv
        with pytest.raises(RecipeError, match="switch"):
            step_argv(build_parser(), "convert", {
                "input": "a.csv", "output": "b.zv", "overwrite": "yes",
            }, tmp_path, 0)

    @pytest.mark.parametrize("steps, message", [
        ([{"convert": {"input": "a.csv", "output": "b.zv", "colour": "red"}}],
         r"step 1 \(convert\): no option colour"),
        ([{"convert": {"input": "a.csv"}}], "'output' is required"),
        ([{"render": {"store": "a.zv"}}], "not a step a recipe can run"),
        ([{"convert": {}, "validate": {}}], "single-key mapping"),
    ])
    def test_refusals_name_the_step(self, tmp_path: Path, steps, message) -> None:
        recipe = _recipe(tmp_path, steps)
        with pytest.raises(SystemExit, match=message):
            main(["run", str(recipe)])

    def test_toml_and_yaml_recipes(self, tmp_path: Path, capsys) -> None:
        (tmp_path / "r.toml").write_text(
            '[[steps]]\n[steps.info]\nstore = "s.zarrvectors"\n', encoding="utf-8",
        )
        assert main(["run", str(tmp_path / "r.toml"), "--dry-run"]) == 0
        assert "zvtools info" in capsys.readouterr().out

        pytest.importorskip("yaml")
        (tmp_path / "r.yaml").write_text(
            "steps:\n  - validate: {store: s.zarrvectors, level: 2}\n", encoding="utf-8",
        )
        assert main(["run", str(tmp_path / "r.yaml"), "--dry-run"]) == 0
        assert "--level 2" in capsys.readouterr().out


@pytest.mark.parametrize(
    "recipe",
    sorted((Path(__file__).resolve().parents[1] / "examples" / "recipes").glob("*.*")),
    ids=lambda p: p.name,
)
def test_the_example_recipes_parse(recipe: Path) -> None:
    """Every example is valid against the current flags, so docs cannot rot."""
    if recipe.suffix in (".yaml", ".yml"):
        pytest.importorskip("yaml")
    assert main(["run", str(recipe), "--dry-run"]) == 0
