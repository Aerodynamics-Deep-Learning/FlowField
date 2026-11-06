"""
Runs the data-generation stages a YAML config enables, always in this order, each in a process of its own:

    machine_check -> geometry_calibrate -> geometry_ingest -> mesh_calibrate -> campaign -> remesh

    python -m scripts.datagen.mesh_master <config.yaml>

One section per stage with `enabled: true` or `false` (machine_check is on when its section is left out); its other
keys are the stage script's options, `-` written as `_` (`--settings-label` as `settings_label`; `--help` on
each script lists them). Relative paths are under data/, `shared` names a committed store, and an output left out
goes to data/scratchpad/<time>_<config name>/<stage>/. A stage reads what an earlier stage of the same run wrote
unless told otherwise. Everything is checked before anything runs; the first stage that fails stops the run.
A log of the run lands beside every output. See scripts/datagen/README.md and mesh_master.example.yaml.
"""

import argparse
import sys
from pathlib import Path

import yaml

from scripts.datagen import engine
from scripts.datagen import (
    geometry_calibrate, geometry_ingest, machine_check, mesh_calibrate, mesh_campaign, mesh_remesh,
)
from scripts.datagen.engine import SHARED, PlanError, Script_refuse, Script_run, Stage

STAGES = (machine_check.STAGE, geometry_calibrate.STAGE, geometry_ingest.STAGE, mesh_calibrate.STAGE,
          mesh_campaign.STAGE, mesh_remesh.STAGE)


def Script_load_config(path) -> tuple[list[tuple[Stage, list[str]]], list[str]]:
    """
    A config's enabled stages, in the fixed order, each with the arguments its options make.

    Args:
        path (str | Path): The YAML config

    Returns:
        tuple: (stage, arguments) per enabled stage, and the names of the stages it switched off

    Raises:
        PlanError: With every problem in the config
    """
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise PlanError([("config", f"not readable as YAML: {e}")]) from None
    data = {} if data is None else data
    if not isinstance(data, dict):
        raise PlanError([("config", "a config is one section per stage")])
    names = [stage.name for stage in STAGES]
    problems = [("config", f"no stage named {key!r}; the stages are {', '.join(names)}")
                for key in data if key not in names]
    requested, off = [], []
    for stage in STAGES:
        section = data.get(stage.name, {"enabled": True} if stage.name == "machine_check" else None)
        if section is None and stage.name not in data:
            continue
        if not isinstance(section, dict) or not isinstance(section.get("enabled"), bool):
            problems.append((stage.name, "needs `enabled: true` or `enabled: false`"))
            continue
        if not section["enabled"]:
            off.append(stage.name)
            continue
        try:
            requested.append((stage, Script_options_to_argv(stage, {k: v for k, v in section.items()
                                                                     if k != "enabled"})))
        except ValueError as e:
            problems.append((stage.name, str(e)))
    if problems:
        raise PlanError(problems)
    return requested, off


def Script_options_to_argv(stage: Stage, options: dict) -> list[str]:
    """
    A config section's options as the stage script's arguments: `true` a flag, a list comma-joined, a relative
    path under data/.

    Raises:
        ValueError: With every key the script has no option for, or a value of the wrong kind
    """
    actions = {a.dest: a for a in stage.build_parser()._actions if not isinstance(a, argparse._HelpAction)}
    positional, optional, problems = [], [], []
    for key, value in options.items():
        action = actions.get(key)
        if action is None:
            problems.append(f"no option {key!r} (its options: {', '.join(sorted(actions))})")
            continue
        if value is None:
            continue
        if action.nargs == 0:
            if not isinstance(value, bool):
                problems.append(f"{key} is true or false, not {value!r}")
            elif value:
                optional.append(action.option_strings[-1])
            continue
        if isinstance(value, bool):
            problems.append(f"{key} takes a value, not {value}")
            continue
        text = ",".join(str(v) for v in value) if isinstance(value, (list, tuple)) else str(value)
        if action.type is Path and text != SHARED and not Path(text).is_absolute():
            text = str(engine.DATA_ROOT / text)
        if action.option_strings:
            optional += [action.option_strings[-1], text]
        else:
            positional.append(text)
    if problems:
        raise ValueError("; ".join(problems))
    return positional + optional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="the run's YAML config, anywhere")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.config.is_file():
        parser.error(f"no config at {args.config}")
    name, title, config = args.config.stem, f"mesh_master {args.config.stem}", args.config.resolve()
    try:
        requested, off = Script_load_config(args.config)
    except PlanError as e:
        return Script_refuse(name, e.problems, title=title, config=config)
    return Script_run(name, requested, in_process=False, title=title, config=config, off=off)


if __name__ == "__main__":
    sys.exit(main())
