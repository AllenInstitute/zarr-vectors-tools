"""``zvtools run`` — carry a store from raw files to viewable in one command.

Every track ends with the same chain: convert, build the pyramid, attach or
merge, validate, shard.  Each step is already a subcommand; a recipe lists
them with their options, and this runs them in order::

    name: hcp-tracts
    steps:
      - convert: {input: tracts.trk, output: tracts.zarrvectors,
                  num_chunks: 5000, coarsen: [8, 8], sparsity: [1, 4]}
      - validate: {store: tracts.zarrvectors}
      - shard: {store: tracts.zarrvectors, shape: 8}

A step's keys are the subcommand's own options with dashes as underscores,
plus its positional arguments by name.  They are turned into the argument
list that subcommand's parser already accepts, so a recipe can say exactly
what the command line can, is checked by the same code, and cannot drift
from it.  Relative paths resolve against the recipe's directory.

Progress is kept in ``<recipe>.run.json`` next to the recipe.  A step that
finished, whose options have not changed and whose outputs still exist, is
skipped on the next run, so a pipeline that failed at step four resumes at
step four.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
import time
from pathlib import Path
from typing import Any

#: Positional arguments of each subcommand a recipe may run, in order, and
#: which of them are outputs a finished step must have left behind.
_POSITIONALS: dict[str, tuple[str, ...]] = {
    "convert": ("input", "output"),
    "pyramid": ("store",),
    "attach": ("store", "input"),
    "merge": ("target", "sources"),
    "split": ("store", "output"),
    "validate": ("store",),
    "info": ("store",),
    "shard": ("store",),
    "bundles": ("store",),
    "synapses": ("table", "skeletons", "output"),
}
_OUTPUTS: dict[str, tuple[str, ...]] = {
    "convert": ("output",),
    "pyramid": ("store",),
    "attach": ("store",),
    "merge": ("target",),
    "split": ("output",),
    "shard": ("store",),
    "bundles": ("store",),
    "synapses": ("output",),
}
#: Options, besides the positionals, that name files to resolve.
_PATH_OPTIONS = {"nodes", "lut", "transform", "scratch_dir"}


class RecipeError(ValueError):
    """A recipe that cannot run as written."""


def load_recipe(path: str | Path) -> dict[str, Any]:
    """Read a recipe from ``.yaml``/``.yml``, ``.toml`` or ``.json``."""
    path = Path(path)
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise RecipeError(
                "YAML recipes need PyYAML: pip install 'zarr-vectors-tools[recipes]', "
                "or write the recipe as .toml or .json"
            ) from exc
        recipe = yaml.safe_load(text)
    elif suffix == ".toml":
        import tomllib

        recipe = tomllib.loads(text)
    elif suffix == ".json":
        recipe = json.loads(text)
    else:
        raise RecipeError(
            f"cannot tell what format {path.name!r} is; name it .yaml, .toml or .json"
        )
    if not isinstance(recipe, dict) or not isinstance(recipe.get("steps"), list):
        raise RecipeError(f"{path.name}: a recipe is a mapping with a 'steps' list")
    return recipe


def _step(entry: Any, index: int) -> tuple[str, dict[str, Any]]:
    """``(command, options)`` from one ``{command: {options}}`` entry."""
    if not isinstance(entry, dict) or len(entry) != 1:
        raise RecipeError(
            f"step {index + 1}: write each step as a single-key mapping, "
            f"such as {{convert: {{input: ..., output: ...}}}}"
        )
    command, options = next(iter(entry.items()))
    if command not in _POSITIONALS:
        raise RecipeError(
            f"step {index + 1}: {command!r} is not a step a recipe can run; "
            f"it can run {', '.join(sorted(_POSITIONALS))}"
        )
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise RecipeError(f"step {index + 1} ({command}): its options must be a mapping")
    return command, dict(options)


def _subparser(parser: argparse.ArgumentParser, command: str) -> argparse.ArgumentParser:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices[command]
    raise RecipeError(f"zvtools has no {command!r} command")  # pragma: no cover


def _resolve(value: Any, base: Path) -> Any:
    text = str(value)
    if "://" in text or Path(text).is_absolute():
        return text
    return str((base / text).resolve())


def step_argv(
    parser: argparse.ArgumentParser,
    command: str,
    options: dict[str, Any],
    base: Path,
    index: int,
) -> list[str]:
    """The ``zvtools`` argument list one recipe step stands for.

    Lists become comma-separated values for options that take one
    (``coarsen: [8, 8]`` is ``--coarsen 8,8``) and repeated flags for options
    that repeat (``attribute: [fa, md]`` is ``--attribute fa --attribute md``);
    ``true`` switches a flag on and ``false`` leaves it off.  An option the
    subcommand does not have is refused here, by name and step, rather than
    by argparse's usage message.
    """
    sub = _subparser(parser, command)
    by_dest: dict[str, argparse.Action] = {}
    for action in sub._actions:
        if action.option_strings:
            long = max(action.option_strings, key=len).lstrip("-").replace("-", "_")
            by_dest[long] = action
            by_dest.setdefault(action.dest, action)

    options = dict(options)
    argv = [command]
    for name in _POSITIONALS[command]:
        if name not in options:
            raise RecipeError(f"step {index + 1} ({command}): '{name}' is required")
        value = options.pop(name)
        values = value if isinstance(value, list) else [value]
        argv += [_resolve(v, base) for v in values]

    unknown = sorted(k for k in options if k not in by_dest)
    if unknown:
        raise RecipeError(
            f"step {index + 1} ({command}): no option {', '.join(unknown)}; "
            f"`zvtools {command} --help` lists them"
        )
    for key, value in options.items():
        action = by_dest[key]
        flag = max(action.option_strings, key=len)
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            if not isinstance(value, bool):
                raise RecipeError(
                    f"step {index + 1} ({command}): '{key}' is a switch; "
                    f"give it true or false"
                )
            if value:
                argv.append(flag)
            continue
        if value is None:
            continue
        if key in _PATH_OPTIONS:
            value = _resolve(value, base)
        if isinstance(action, argparse._AppendAction):
            for item in value if isinstance(value, list) else [value]:
                argv += [flag, str(item)]
        elif isinstance(value, list):
            argv += [flag, ",".join(str(v) for v in value)]
        else:
            argv += [flag, str(value)]
    return argv


def _fingerprint(argv: list[str]) -> str:
    return hashlib.sha256(json.dumps(argv).encode("utf-8")).hexdigest()


def run_recipe(
    recipe_path: str | Path,
    *,
    force: bool = False,
    start: int | None = None,
    dry_run: bool = False,
) -> int:
    """Run a recipe; returns the exit code of the step that stopped it, or 0.

    Args:
        recipe_path: The recipe file.
        force: Rerun every step, finished or not.
        start: Rerun from this step (1-based) onward, whatever was recorded.
        dry_run: Print each step's command and stop.
    """
    from zarr_vectors_tools.cli import build_parser

    recipe_path = Path(recipe_path).resolve()
    recipe = load_recipe(recipe_path)
    base = recipe_path.parent
    if recipe.get("workdir"):
        base = Path(_resolve(recipe["workdir"], recipe_path.parent))
        base.mkdir(parents=True, exist_ok=True)
    parser = build_parser()

    plan = []
    for index, entry in enumerate(recipe["steps"]):
        command, options = _step(entry, index)
        argv = step_argv(parser, command, options, base, index)
        outputs = [
            _resolve(options[name], base) for name in _OUTPUTS.get(command, ())
        ]
        plan.append((command, argv, outputs))

    name = recipe.get("name", recipe_path.stem)
    if dry_run:
        for index, (_command, argv, _outputs) in enumerate(plan):
            print(f"[{index + 1}/{len(plan)}] zvtools {shlex.join(argv)}")
            # Parsed as well as printed, so a value the subcommand would
            # reject (a bad choice, a malformed list) fails the dry run too.
            parser.parse_args(argv)
        return 0

    state_path = recipe_path.with_name(recipe_path.stem + ".run.json")
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    done: dict[str, Any] = dict(state.get("steps", {}))

    print(f"recipe {name}: {len(plan)} step(s)")
    code = 0
    for index, (command, argv, outputs) in enumerate(plan):
        key = str(index + 1)
        label = f"[{key}/{len(plan)}] {command}"
        fingerprint = _fingerprint(argv)
        record = done.get(key, {})
        rerun = force or (start is not None and index + 1 >= start)
        if (
            not rerun
            and record.get("status") == "done"
            and record.get("fingerprint") == fingerprint
            and all(Path(p).exists() for p in outputs)
        ):
            print(f"{label}: done earlier, skipped")
            continue

        print(f"{label}: zvtools {shlex.join(argv)}")
        began = time.perf_counter()
        try:
            args = parser.parse_args(argv)
            code = int(args.func(args) or 0)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
            if not isinstance(exc.code, int) and exc.code:
                print(exc.code, file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 - reported like the CLI does
            print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
            code = 1
        elapsed = round(time.perf_counter() - began, 3)

        done[key] = {
            "command": command,
            "fingerprint": fingerprint,
            "status": "done" if code == 0 else "failed",
            "seconds": elapsed,
        }
        # A later step's result depends on the earlier ones, so once a step
        # reruns, what is recorded after it no longer describes the store.
        for later in [k for k in done if int(k) > index + 1]:
            del done[later]
        state_path.write_text(
            json.dumps({"recipe": str(recipe_path), "steps": done}, indent=2),
            encoding="utf-8",
        )
        if code != 0:
            print(f"{label}: failed (exit {code}) after {elapsed}s; fix it and run again")
            return code
        print(f"{label}: done in {elapsed}s")
    print(f"recipe {name}: finished")
    return code


def run(args) -> int:
    """``zvtools run RECIPE``."""
    try:
        return run_recipe(
            args.recipe, force=args.force, start=args.start, dry_run=args.dry_run,
        )
    except RecipeError as exc:
        raise SystemExit(f"error: {exc}") from None
