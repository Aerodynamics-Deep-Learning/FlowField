"""
What every datagen script shares, and the run it makes (scripts/datagen/README.md is the guide):

    plan     before anything runs: each stage's options parsed by its own parser, every output decided (left
             out: the scratchpad; given: new or existing; `shared`: the committed one), this run's outputs handed
             to later stages' inputs, each stage's own checks, the warnings; any problem refuses the whole run
    log      one log, a copy in every folder the run saves data into (the scratchpad when it saves none), every
             line `HH:MM:SS [run|cmd] [plan|<stage>|summary] [info|!|!!|error] text`
    run      a script's one stage in this process, so a debugger stops in it; the master's stages each as a
             child process, `python -m <script>` with FLOWFIELD_STAGE_CHILD naming the stage, which only runs
    summary  each stage's exit code and minutes, the warnings again

Exit codes: 0 done, 1 a stage stopped or failed, 2 refused (a usage error), 130 Ctrl-C.
"""

import argparse
import datetime
import enum
import io
import logging
import os
import re
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, NamedTuple, Optional, Sequence

from src.datagen.geometry.common.paths import DATA_ROOT, REPO_ROOT, Geom_paths

CHILD_ENV = "FLOWFIELD_STAGE_CHILD"
SHARED = "shared"
INTERRUPTED = 130
HELP_TAIL = ("Through this script: an output left out goes to a new folder in data/scratchpad/, `shared` names the "
             "committed store, and a log of the run lands beside every output.")

_LOGGED = re.compile(r"^\d\d:\d\d:\d\d (DEBUG|INFO|WARNING|ERROR|CRITICAL) (.*)$")
_LEVEL_OF = {"DEBUG": "info", "INFO": "info", "WARNING": "!", "ERROR": "error", "CRITICAL": "error"}


class UsageError(ValueError):
    """A mistake in what was asked, found after the options parsed: exit code 2."""


class PlanError(ValueError):
    """Every reason a run was refused before anything ran, as (stage, text) pairs."""

    def __init__(self, problems: list[tuple[str, str]]):
        super().__init__("; ".join(f"{stage}: {text}" for stage, text in problems))
        self.problems = problems


class _Refused(Exception):
    pass


#region What a stage declares

class OutputKind(enum.Enum):
    FILE = "file"
    FOLDER = "folder"
    DATA_ROOT = "data root"  # a data/ layout; its log copy goes beside its manifest


class Output(NamedTuple):
    option: str                                    # the parser's dest
    kind: OutputKind
    what: str                                      # what it is, for the messages
    scratch: str = ""                              # its name in the stage's scratchpad folder ({date} allowed)
    shared: Optional[Callable[[], Path]] = None    # what `shared` means; None: there is no shared one
    exists: str = "this run adds to it"            # what happens to one that exists
    dry_run: Optional[str] = None                  # a flag under which nothing is written


class Input(NamedTuple):
    option: str
    what: str
    folder: bool = False                           # a folder, else a file
    shared: Optional[Callable[[], Path]] = None    # what `shared`, and nothing given, mean
    made_by: Optional[tuple[str, str]] = None      # (stage, output option) whose output this run reads instead
    from_output: Callable[[Path], Path] = Path     # that output's path -> what this input reads


class Stage(NamedTuple):
    name: str
    module: str
    build_parser: Callable[[], argparse.ArgumentParser]
    run: Callable[[argparse.Namespace], int]
    outputs: tuple = ()
    inputs: tuple = ()
    slots: bool = False                            # runs worker slots: relaunched on Linux for their caps
    meshes: bool = False                           # what it makes depends on this machine meshing like the reference
    check: Optional[Callable] = None               # (args, inputs made by this run) -> problems, before anything runs
    after: Optional[Callable] = None               # (its plan, stages passed, the run) after it exits 0


def _manifest_of(data_root: Path) -> Path:
    return Geom_paths(data_root).db


GEOMETRY_DB = Input("geometry_db", "geometry manifest", shared=lambda: Geom_paths().db,
                    made_by=("geometry_ingest", "data_root"), from_output=_manifest_of)


def Script_names(text: str) -> list[str]:
    """Comma-separated airfoil names as a list, for an option's `type`."""
    return [name.strip() for name in text.split(",") if name.strip()]


def Script_check_names(load: Callable, args: argparse.Namespace, made_here: set) -> list[str]:
    """A stage's named airfoils checked against the manifest it reads, unless this run's ingest writes it."""
    if not getattr(args, "airfoils", None) or "geometry_db" in made_here:
        return []
    try:
        load(args.airfoils, args.geometry_db)
    except ValueError as e:
        return [str(e)]
    return []

#endregion


#region Plan

class Resolved(NamedTuple):
    option: str
    path: Path
    how: str                                       # outputs: scratchpad, new, exists, shared; inputs: given, shared,
                                                   # this run's <stage>
    log_dir: Optional[Path] = None                 # an output's: where its copy of the log goes


class StagePlan(NamedTuple):
    stage: Stage
    args: argparse.Namespace
    argv: list[str]                                # every path absolute: what a child is started with
    writes: list[Resolved]
    reads: list[Resolved]


class RunPlan(NamedTuple):
    title: str
    file_name: str                                 # the log's
    started: datetime.datetime
    scratch: Path                                  # data/scratchpad/<time>_<name>
    stages: list[StagePlan]
    off: list[str]                                 # stages a config switched off
    config: Optional[Path]
    notes: list[tuple[str, str, str]]              # (stage, level, text)
    log_dirs: list[Path]


def Script_plan(name: str, requested: Sequence[tuple[Stage, Sequence[str]]], *, title: Optional[str] = None,
                config: Optional[Path] = None, off: Sequence[str] = (),
                started: Optional[datetime.datetime] = None) -> RunPlan:
    """
    Plans a run without running or writing anything.

    Args:
        name (str): The run's name: the config's, or the one stage's
        requested (Sequence[tuple[Stage, Sequence[str]]]): The stages, in order, each with its arguments
        title (str | None): The log's first words; `name` if None
        config (Path | None): The master's config
        off (Sequence[str]): Stages the config switched off
        started (datetime | None): When the run started; now if None

    Returns:
        RunPlan: Everything the run will do and say

    Raises:
        PlanError: With every problem found, when there is any
    """
    started = started or datetime.datetime.now()
    scratch = DATA_ROOT / "scratchpad" / f"{started:%Y-%m-%d_%H%M%S}_{name}"
    problems, notes, plans, made = [], [], [], {}
    for stage, argv in requested:
        parser = stage.build_parser()
        parser.error = _refuse
        try:
            args = parser.parse_args(list(argv))
        except _Refused as e:
            problems.append((stage.name, str(e)))
            continue
        writes = [_output(stage, out, args, scratch, started, made, notes, problems) for out in stage.outputs]
        made_here = set()
        reads = [_input(stage, inp, args, made, made_here, notes, problems) for inp in stage.inputs]
        if stage.check is not None:
            problems += [(stage.name, text) for text in stage.check(args, made_here)]
        plans.append(StagePlan(stage, args, Script_argv(parser, args), [w for w in writes if w],
                               [r for r in reads if r]))

    names = [stage.name for stage, _ in requested]
    if "machine_check" in off:
        notes.append(("machine_check", "!", "machine_check is off: this run does not check that this machine "
                                            "reproduces the reference"))
    for maker, user in (("geometry_calibrate", "geometry_ingest"), ("mesh_calibrate", "campaign")):
        if maker in names and user in names:
            notes.append((user, "!", f"{maker} and {user} both run: {user} uses the constants as they are now, "
                                     f"not what this calibration recommends"))
    meshing = [stage.name for stage, _ in requested if stage.meshes]
    if meshing and "machine_check" not in names:
        from scripts.datagen.meshing.parity import Script_stamp_problem  # the stamp is the machine check's
        problem = Script_stamp_problem()
        if problem:
            notes.append((meshing[0], "!", problem))
    if problems:
        raise PlanError(problems)

    log_dirs = []
    for w in (w for sp in plans for w in sp.writes):
        if w.log_dir not in log_dirs:
            log_dirs.append(w.log_dir)
    return RunPlan(title or name, f"{started:%Y-%m-%d_%H%M%S}_{name}.log", started, scratch, plans, list(off), config,
                   notes, log_dirs)


def _refuse(message: str):
    raise _Refused(message)


def _output(stage, out, args, scratch, started, made, notes, problems) -> Optional[Resolved]:
    value = getattr(args, out.option)
    dry = bool(out.dry_run and getattr(args, out.dry_run))
    if value is None:
        folder = scratch / stage.name
        path = folder / out.scratch.format(date=started.date().isoformat()) if out.scratch else folder
        how, level, said = "scratchpad", "info", f"nothing given, writing to the scratchpad: {_show(path)}"
    elif str(value) == SHARED:
        if out.shared is None:
            problems.append((stage.name, f"{out.option}: there is no shared {out.what}; give a path"))
            return None
        path, how = out.shared(), "shared"
        level, said = ("info", f"the shared {out.what}, {_show(path)} (dry run: nothing written)") if dry else \
            ("!!", f"the SHARED {out.what}, {_show(path)} (committed)")
    else:
        path = Path(value).resolve()
        how = "exists" if _exists(out.kind, path) else "new"
        level, said = ("info", f"{_show(path)} (new)") if how == "new" else \
            ("info", f"{_show(path)} exists (dry run: nothing written)") if dry else \
            ("!", f"{_show(path)} exists; {out.exists}")
    setattr(args, out.option, path)
    made[(stage.name, out.option)] = path
    notes.append((stage.name, level, f"writes {out.option}: {said}"))
    log_dir = {OutputKind.FILE: path.parent, OutputKind.FOLDER: path,
               OutputKind.DATA_ROOT: Geom_paths(path).manifest_dir}[out.kind]
    return Resolved(out.option, path, how, log_dir)


def _exists(kind: OutputKind, path: Path) -> bool:
    return Geom_paths(path).db.exists() if kind is OutputKind.DATA_ROOT else path.exists()


def _input(stage, inp, args, made, made_here, notes, problems) -> Optional[Resolved]:
    value = getattr(args, inp.option)
    if value is None and inp.made_by in made:
        path, how = inp.from_output(made[inp.made_by]), f"this run's {inp.made_by[0]}"
        made_here.add(inp.option)
    elif value is None or str(value) == SHARED:
        if inp.shared is None:
            return None
        path, how = inp.shared(), "shared"
    else:
        path, how = Path(value).resolve(), "given"
        if any(inp.from_output(p) == path for p in made.values()):
            made_here.add(inp.option)
    if inp.option not in made_here and not (path.is_dir() if inp.folder else path.is_file()):
        problems.append((stage.name, f"{inp.option}: no {inp.what} at {_show(path)}"))
    setattr(args, inp.option, path)
    notes.append((stage.name, "info", f"reads {inp.option}: {how}, {_show(path)}"))
    return Resolved(inp.option, path, how)


def Script_argv(parser: argparse.ArgumentParser, args: argparse.Namespace) -> list[str]:
    """The arguments that parse back to `args`: every option set, positionals first."""
    positional, optional = [], []
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        value = getattr(args, action.dest, None)
        if value is None or value is False:
            continue
        if not action.option_strings:
            positional.append(_text(value))
        elif value is True:
            optional.append(action.option_strings[-1])
        else:
            optional += [action.option_strings[-1], _text(value)]
    return positional + optional


def _text(value) -> str:
    if isinstance(value, range):
        return f"{value.start}-{value.stop - 1}"
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def _show(path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)

#endregion


#region Log

class RunLog:
    """The run's log: each line to the screen and to every copy, written as it comes."""

    def __init__(self, folders: Sequence[Path], file_name: str):
        self.echo, self.lock, self.files = sys.stdout, threading.Lock(), []
        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)
            self.files.append(open(folder / file_name, "a", encoding="utf-8", newline="\n"))

    def line(self, who: str, stage: str, level: str, text) -> None:
        head = f"{time.strftime('%H:%M:%S')} [{who}] {'[' + stage + ']':<21}{'[' + level + ']':<8}"
        with self.lock:
            for part in str(text).splitlines() or [""]:
                row = head + part
                try:
                    self.echo.write(row + "\n")
                except UnicodeEncodeError:
                    self.echo.write(row.encode("ascii", "replace").decode() + "\n")
                self.echo.flush()
                for f in self.files:
                    f.write(row + "\n")
                    f.flush()

    def close(self) -> None:
        for f in self.files:
            f.close()


class _Lines(io.TextIOBase):
    """A text stream handing each complete line to `say`."""

    def __init__(self, say: Callable[[str], None]):
        self.say, self.rest = say, ""

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        *lines, self.rest = (self.rest + text).split("\n")
        for line in lines:
            self.say(line.rstrip("\r"))
        return len(text)

    def finish(self) -> None:
        if self.rest:
            self.say(self.rest)
            self.rest = ""


class _Handler(logging.Handler):
    def __init__(self, say: Callable[[str, str], None]):
        super().__init__(logging.INFO)
        self.say = say

    def emit(self, record: logging.LogRecord) -> None:
        level = "error" if record.levelno >= logging.ERROR else "!" if record.levelno >= logging.WARNING else "info"
        text = record.getMessage()
        if record.exc_info:
            text += "\n" + "".join(traceback.format_exception(*record.exc_info)).rstrip()
        self.say(level, text)

#endregion


#region Run

class Run:
    """A run under way: what a stage's `after` may use to say something or start another script."""

    def __init__(self, plan: RunPlan, log: RunLog):
        self.plan, self.log = plan, log

    def say(self, stage: str, level: str, text) -> None:
        self.log.line("run", stage, level, text)

    def relay(self, stage: str, module: str, argv: Sequence[str], child: str) -> int:
        """Runs `python -m module argv` as `child`'s stage, its output logged under `stage`; returns its exit code."""
        code, interrupted = self._child(stage, module, argv, child)
        return INTERRUPTED if interrupted else code

    def _child(self, stage: str, module: str, argv: Sequence[str], child: str) -> tuple[int, bool]:
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", CHILD_ENV: child}
        proc = subprocess.Popen([sys.executable, "-m", module, *argv], cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        pumps = [threading.Thread(target=self._pump, args=(proc.stdout, stage, False), daemon=True),
                 threading.Thread(target=self._pump, args=(proc.stderr, stage, True), daemon=True)]
        for pump in pumps:
            pump.start()
        interrupted = False
        while True:
            try:
                code = proc.wait()
                break
            except KeyboardInterrupt:  # the child has it too; its own summary still comes through
                interrupted = True
        for pump in pumps:
            pump.join(timeout=10)
        return code, interrupted

    def _pump(self, stream, stage: str, errors: bool) -> None:
        level = "error"
        for raw in iter(stream.readline, b""):
            text = raw.decode("utf-8", "replace").rstrip("\r\n")
            if not errors:
                self.log.line("cmd", stage, "info", text)
                continue
            logged = _LOGGED.match(text)
            if logged:
                level, text = _LEVEL_OF[logged.group(1)], logged.group(2)
            elif text.startswith("Traceback"):
                level = "error"
            self.log.line("cmd", stage, level, text)

    def here(self, sp: StagePlan) -> int:
        """`sp`'s stage in this process, its output and logging caught into the log."""
        name = sp.stage.name
        out = _Lines(lambda text: self.log.line("cmd", name, "info", text))
        err = _Lines(lambda text: self.log.line("cmd", name, "error", text))
        root = logging.getLogger()
        kept = root.handlers[:], root.level, sys.stdout, sys.stderr
        root.handlers = [_Handler(lambda level, text: self.log.line("cmd", name, level, text))]
        root.setLevel(logging.INFO)
        sys.stdout, sys.stderr = out, err
        try:
            return sp.stage.run(sp.args)
        except UsageError as e:
            self.say(name, "error", str(e))
            return 2
        except KeyboardInterrupt:
            return INTERRUPTED
        finally:
            out.finish()
            err.finish()
            root.handlers, sys.stdout, sys.stderr = kept[0], kept[2], kept[3]
            root.setLevel(kept[1])


def Script_execute(plan: RunPlan, *, in_process: bool) -> int:
    """
    Runs a plan: the log opened in every folder it saves into, the stages in order until one fails.

    Args:
        plan (RunPlan): `Script_plan`'s
        in_process (bool): Run the stages here (a script's one stage), else each as a child process (the master)

    Returns:
        int: 0, the failing stage's exit code, or 130 on Ctrl-C
    """
    log = RunLog(plan.log_dirs or [plan.scratch], plan.file_name)
    run = Run(plan, log)
    t_run, ran, code = time.monotonic(), [], 0
    try:
        _header(run, plan)
        for i, sp in enumerate(plan.stages):
            name, rest = sp.stage.name, [s.stage.name for s in plan.stages[i + 1:]]
            run.say(name, "info", " ".join(["start: python -m", sp.stage.module, _line(sp.argv)]).rstrip()
                    + (" (in this process)" if in_process else ""))
            t0 = time.monotonic()
            try:
                if in_process:
                    code = run.here(sp)
                else:
                    code, interrupted = run._child(name, sp.stage.module, sp.argv, name)
                    code = INTERRUPTED if interrupted else code
            except Exception:
                run.say(name, "error", traceback.format_exc().rstrip())
                ran.append((name, 1, time.monotonic() - t0))
                _end(run, name, 1, ran[-1][2], rest)
                _summary(run, plan, ran, 1, time.monotonic() - t_run)
                raise
            ran.append((name, code, time.monotonic() - t0))
            if code != 0:
                _end(run, name, code, ran[-1][2], rest)
                break
            run.say(name, "info", f"done: exit 0, {ran[-1][2] / 60:.1f} min")
            if sp.stage.after is not None:
                sp.stage.after(sp, {n for n, c, _ in ran if c == 0}, run)
        _summary(run, plan, ran, code, time.monotonic() - t_run)
        return code
    finally:
        log.close()


def _header(run: Run, plan: RunPlan) -> None:
    run.say("plan", "info", f"{plan.title}, started {plan.started:%Y-%m-%d %H:%M:%S} on {socket.gethostname()}, "
                            f"code {_code_version()}")
    if plan.config is not None:
        run.say("plan", "info", f"config {plan.config}")
    run.say("plan", "info", "stages " + " -> ".join(sp.stage.name for sp in plan.stages)
            + (f" (off: {', '.join(plan.off)})" if plan.off else ""))
    for stage, level, text in plan.notes:
        run.say(stage, level, text)
    run.say("plan", "info", "log copies in: " + ", ".join(_show(d) for d in plan.log_dirs or [plan.scratch]))


def _end(run: Run, name: str, code: int, seconds: float, rest: list[str]) -> None:
    after = f"; not run: {', '.join(rest)}" if rest else ""
    if code == INTERRUPTED:
        run.say(name, "!", f"interrupted (Ctrl-C) after {seconds / 60:.1f} min{after}")
    else:
        run.say(name, "error", f"exit {code} after {seconds / 60:.1f} min" + ("; the run stops here" if rest else "")
                + after)


def _summary(run: Run, plan: RunPlan, ran: list, code: int, seconds: float) -> None:
    if code == 0:
        run.say("summary", "info", f"finished in {seconds / 60:.1f} min")
    elif code == INTERRUPTED:
        run.say("summary", "!", f"interrupted at {ran[-1][0]}, exit {code}")
    else:
        run.say("summary", "error", f"stopped at {ran[-1][0]}, exit {code}")
    width = max(len(name) for name, _, _ in ran)
    for name, c, s in ran:
        run.say("summary", "info", f"{name:<{width}}  exit {c:<4} {s / 60:5.1f} min")
    for stage, level, text in plan.notes:
        if level in ("!", "!!"):
            run.say("summary", level, text if text.startswith(stage) else f"{stage}: {text}")


def _line(argv: Sequence[str]) -> str:
    return " ".join(f'"{a}"' if " " in a else a for a in argv)


def _code_version() -> str:
    """The short commit, '-dirty' with tracked changes; 'unknown' outside git."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + ("-dirty" if dirty else "")

#endregion


#region Entry points

def Script_run(name: str, requested: Sequence[tuple[Stage, Sequence[str]]], *, in_process: bool,
               title: Optional[str] = None, config: Optional[Path] = None, off: Sequence[str] = ()) -> int:
    """
    Plans and runs; a refused plan is logged into the scratchpad and exits 2.

    Args:
        name (str): The run's name, in its log's and scratchpad folder's names
        requested (Sequence[tuple[Stage, Sequence[str]]]): The stages, in order, each with its arguments
        in_process (bool): As `Script_execute`
        title (str | None): The log's first words; `name` if None
        config (Path | None): The master's config
        off (Sequence[str]): Stages the config switched off

    Returns:
        int: The exit code
    """
    started = datetime.datetime.now()
    try:
        plan = Script_plan(name, requested, title=title, config=config, off=off, started=started)
    except PlanError as e:
        return Script_refuse(name, e.problems, title=title, config=config, started=started)
    return Script_execute(plan, in_process=in_process)


def Script_refuse(name: str, problems: Sequence[tuple[str, str]], *, title: Optional[str] = None,
                  config: Optional[Path] = None, started: Optional[datetime.datetime] = None) -> int:
    """Logs a refused run into its scratchpad folder; returns 2."""
    started = started or datetime.datetime.now()
    scratch = DATA_ROOT / "scratchpad" / f"{started:%Y-%m-%d_%H%M%S}_{name}"
    plan = RunPlan(title or name, f"{started:%Y-%m-%d_%H%M%S}_{name}.log", started, scratch, [], [], config, [], [])
    log = RunLog([scratch], plan.file_name)
    run = Run(plan, log)
    try:
        run.say("plan", "info", f"{plan.title}, started {started:%Y-%m-%d %H:%M:%S} on {socket.gethostname()}, "
                                f"code {_code_version()}")
        if config is not None:
            run.say("plan", "info", f"config {config}")
        run.say("plan", "error", f"refused, nothing was run: {len(problems)} problem" + "s" * (len(problems) > 1))
        for stage, text in problems:
            run.say(stage, "error", text)
        run.say("plan", "info", f"log copies in: {_show(scratch)}")
    finally:
        log.close()
    return 2


def Script_main(stage: Stage, argv: Optional[Sequence[str]] = None) -> int:
    """
    A stage script's entry: planned, logged and run in this process; as a child of the master, only run.

    Args:
        stage (Stage): The script's stage
        argv (Sequence[str] | None): The arguments, the command line's if None

    Returns:
        int: The exit code
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    if stage.slots:
        from src.datagen.meshing.optimizer.batch.slots import Optim_relaunch_for_slots  # only slot stages load it
        Optim_relaunch_for_slots(stage.module, argv)
    if os.environ.get(CHILD_ENV) == stage.name:
        return _as_child(stage, argv)
    if {"-h", "--help"} & set(argv):
        stage.build_parser().print_help()
        print("\n" + HELP_TAIL)
        return 0
    return Script_run(stage.name, [(stage, argv)], in_process=True)


def _as_child(stage: Stage, argv: Sequence[str]) -> int:
    parser = stage.build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
                        force=True)
    try:
        return stage.run(args)
    except UsageError as e:
        parser.error(str(e))

#endregion
