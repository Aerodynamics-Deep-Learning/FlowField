"""
Runs the data-generation stages a YAML config enables, always in this order, each in a process of its own:

    machine_check -> geometry_calibrate -> geometry_ingest -> mesh_calibrate -> campaign -> remesh

    python -m scripts.datagen.meshing.master <config.yaml>

One section per stage with `enabled: true` or `false` (machine_check is on when its section is left out); its other
keys are the stage script's options, `-` written as `_` (`--settings-label` as `settings_label`; `--help` on
each script lists them). `settings:` sets values of src/datagen/constants.py for the run, by their names in lower
case: at the top for every stage that reads one, inside a stage's section for that stage alone, over the top's.
Relative paths are under data/ and `shared` names a committed store. `folder:` names the
run's folder, which holds its log (log.log), a copy of the config and every output left out, under <stage>/; left
out, it is a new data/scratchpad/<time>_<config name>/. A stage reads what an earlier stage of the same run wrote
unless told otherwise. Everything is checked before anything runs; the first stage that fails stops the run.
See scripts/datagen/README.md and master.example.yaml beside this file.
"""

import argparse
import sys
from pathlib import Path
from typing import NamedTuple, Optional

import yaml

from scripts.datagen import engine
from scripts.datagen.engine import SHARED, PlanError, Script_refuse, Script_run, Stage
from scripts.datagen.geometry import calibrate as geometry_calibrate, ingest as geometry_ingest
from scripts.datagen.meshing import calibrate as mesh_calibrate, campaign as mesh_campaign, machine_check
from scripts.datagen.meshing import remesh as mesh_remesh

STAGES = (machine_check.STAGE, geometry_calibrate.STAGE, geometry_ingest.STAGE, mesh_calibrate.STAGE,
          mesh_campaign.STAGE, mesh_remesh.STAGE)
FOLDER_KEY = "folder"
SETTINGS_KEY = "settings"


class Config(NamedTuple):
    stages: list[tuple[Stage, list[str]]]          # (stage, arguments) per enabled stage, in the fixed order
    off: list[str]                                 # the stages it switched off
    folder: Optional[Path]                         # the run's folder; None: a new one in the scratchpad
    sources: dict = {}                             # (stage, setting) -> which block set it
    notes: list = []                               # (stage, level, text) for the plan


def Script_load_config(path) -> Config:
    """
    A config's enabled stages, in the fixed order, each with the arguments its options make, and its run folder.

    Args:
        path (str | Path): The YAML config

    Returns:
        Config: The stages, those switched off, the run folder

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
    problems = [("config", f"no stage named {key!r}; the stages are {', '.join(names)} (beside them: `folder`, "
                           f"`settings`)") for key in data if key not in (*names, FOLDER_KEY, SETTINGS_KEY)]
    folder = _folder(data.get(FOLDER_KEY), problems)
    top = _settings(data.get(SETTINGS_KEY), "settings", problems)
    problems += [("settings", f"no stage reads {key}") for key in top if not any(key in s.reads for s in STAGES)]
    requested, off, resolved, sources, options = [], [], {}, {}, {}
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
        own = _settings(section.get(SETTINGS_KEY), stage.name, problems)
        problems += [(stage.name, f"does not read {key}") for key in own if key not in stage.reads]
        settings = {k: v for k, v in {**top, **own}.items() if k in stage.reads}
        for key in settings:
            sources[(stage.name, key)] = "the settings block" if key not in own else (
                f"its own settings, over the settings block's {engine._text(top[key])}" if key in top else
                "its own settings")
        resolved[stage.name] = settings
        options[stage.name] = {key for key, value in section.items() if value is not None}
        try:
            argv = Script_options_to_argv(stage, {k: v for k, v in section.items() if k not in ("enabled",
                                                                                                SETTINGS_KEY)})
        except ValueError as e:
            problems.append((stage.name, str(e)))
            continue
        requested.append((stage, argv + [a for key, value in settings.items()
                                         for a in ("--set", f"{key}={engine._text(value)}")]))
    problems += _disagreements(resolved, options)
    notes = [("settings", "!", f"{key} is read by no stage this run") for key in top
             if any(key in s.reads for s in STAGES) and not any(key in r for r in resolved.values())]
    if problems:
        raise PlanError(problems)
    return Config(requested, off, folder, sources, notes)


def _settings(value, where: str, problems: list) -> dict:
    """A `settings:` block, each value typed as its constant: setting -> value."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        problems.append((where, "settings are `name: value` lines under `settings:`"))
        return {}
    out = {}
    for key, raw in value.items():
        text = (",".join(f"{k}:{v}" for k, v in raw.items()) if isinstance(raw, dict) else
                ",".join(str(v) for v in raw) if isinstance(raw, (list, tuple)) else str(raw))
        try:
            setting = engine.Script_setting(f"{key}={text}")
        except argparse.ArgumentTypeError as e:
            problems.append((where, str(e)))
            continue
        out[setting.key] = setting.value
    return out


def _disagreements(resolved: dict, options: dict) -> list[tuple[str, str]]:
    """
    A stage that reads what an earlier stage of the run writes must set alike every setting both read: the re-mesh
    finds the campaign's meshes by their settings, a mesh stage the ingest's shapes by their CST order and nose.
    """
    stages, out = {stage.name: stage for stage in STAGES}, []
    for name, settings in resolved.items():
        for inp in stages[name].inputs:
            maker = inp.made_by[0] if inp.made_by else None
            if maker not in resolved or inp.option in options[name]:  # it reads no output of this run
                continue
            for key in sorted(stages[name].reads & stages[maker].reads):
                if _value(settings, key) != _value(resolved[maker], key):
                    out.append((name, f"{key} is {_said(settings, key)} here but {_said(resolved[maker], key)} in "
                                      f"{maker}, whose {inp.what} it reads; both must set it alike"))
    return out


def _value(settings: dict, key: str):
    """A setting as the run uses it: given, else constants.py's; a dict's given names over constants.py's."""
    value, default = settings.get(key), engine.SETTABLE[key]
    if value is None:
        return default
    return {**default, **value} if isinstance(default, dict) else value


def _said(settings: dict, key: str) -> str:
    return engine._text(settings[key]) if key in settings else "left to constants.py"


def _folder(value, problems: list) -> Optional[Path]:
    """`folder:` as a path, a relative one under data/."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or value == SHARED:
        problems.append(("config", f"folder is the run's own folder, a path, not {value!r}"))
        return None
    path = Path(value)
    return path if path.is_absolute() else engine.DATA_ROOT / path


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
    parser = argparse.ArgumentParser(prog="python -m scripts.datagen.meshing.master", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="the run's YAML config, anywhere")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.config.is_file():
        parser.error(f"no config at {args.config}")
    name, title, config = args.config.stem, f"master {args.config.stem}", args.config.resolve()
    try:
        loaded = Script_load_config(args.config)
    except PlanError as e:
        return Script_refuse(name, e.problems, title=title, config=config)
    return Script_run(name, loaded.stages, in_process=False, title=title, config=config, off=loaded.off,
                      folder=loaded.folder, sources=loaded.sources, notes=loaded.notes)


if __name__ == "__main__":
    sys.exit(main())
