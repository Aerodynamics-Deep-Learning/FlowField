"""
Tests for `scripts.datagen.engine`, the plan, logs and runs every datagen script shares, i.e.:

Step 1: Ensure every output is decided before anything runs, and a later stage reads what an earlier one wrote
    - test_an_output_left_out_goes_to_the_scratchpad
    - test_a_given_output_is_new_or_existing_and_shared_names_the_committed_one
    - test_a_later_stage_reads_what_an_earlier_one_wrote
    - test_the_arguments_parse_back_to_the_same_options
Step 2: Ensure a run with any problem is refused whole, before anything runs
    - test_every_problem_refuses_the_run
Step 3: Ensure a run logs every line, labelled, into every folder it writes, and stops at its first failure
    - test_a_run_logs_every_line_labelled_in_every_output_folder
    - test_a_failing_stage_stops_the_run_with_its_exit_code
    - test_a_usage_error_or_ctrl_c_inside_a_stage
    - test_a_crash_is_logged_then_raised
Step 4: Ensure the master's stages run as child processes, their lines labelled by level, Ctrl-C waited out
    - test_the_stages_run_as_child_processes
    - test_ctrl_c_waits_for_the_child_then_stops_the_run
Step 5: Ensure a script's own entry: its help, and only running when it is a child
    - test_a_scripts_help_and_its_run_as_a_child

The stages are `engine_stages.py`'s stand-ins; the scripts' data/ is a temporary folder (conftest).
"""

import logging
import os
import re
import subprocess

import pytest

from scripts.datagen import engine
from scripts.datagen.engine import CHILD_ENV, PlanError, Script_argv, Script_main, Script_plan, Script_run
from tests.unit.scripts.datagen.engine_stages import MAKER, READER

_LINE = re.compile(r"^\d\d:\d\d:\d\d \[(run|cmd)\] \[([a-z_]+)\] +\[(info|!|!!|error)\] +(.*)$")


def _lines(path):
    """A log's lines as (who, stage, level, text)."""
    rows = [_LINE.match(row) for row in path.read_text(encoding="utf-8").splitlines()]
    assert all(rows), "every line labelled"
    return [m.groups() for m in rows]


def _log(folder):
    [log] = folder.glob("*.log")
    return log


# region Step 1
def test_an_output_left_out_goes_to_the_scratchpad(data_root, capsys):
    assert Script_run("solo", [(MAKER, [])], in_process=True) == 0

    [run] = (data_root / "scratchpad").iterdir()
    assert re.fullmatch(r"\d{4}-\d\d-\d\d_\d{6}_solo", run.name)
    assert (run / "maker" / "made.txt").read_text() == "made"
    rows = _lines(_log(run / "maker"))
    assert ("run", "maker", "info", f"writes out: nothing given, writing to the scratchpad: {run / 'maker'}") in rows


def test_a_given_output_is_new_or_existing_and_shared_names_the_committed_one(tmp_path, data_root):
    (tmp_path / "old").mkdir()
    for given, level, said in ((tmp_path / "new", "info", "(new)"), (tmp_path / "old", "!", "exists; this run "
                                                                                             "writes into it"),
                               ("shared", "!!", "the SHARED made folder")):
        plan = Script_plan("p", [(MAKER, ["--out", str(given)])])
        [(stage, got_level, text)] = plan.notes
        assert (stage, got_level) == ("maker", level) and said in text
    assert plan.stages[0].args.out == data_root / "shared_made" == plan.log_dirs[0]

    with pytest.raises(PlanError, match="there is no shared copy"):
        Script_plan("p", [(READER, ["--src", str(tmp_path / "old"), "--out", "shared"])])


def test_a_later_stage_reads_what_an_earlier_one_wrote(tmp_path):
    out = tmp_path / "made"
    plan = Script_plan("p", [(MAKER, ["--out", str(out)]), (READER, ["--out", str(tmp_path / "copy.txt")])])

    reader = plan.stages[1]
    assert reader.args.src == out.resolve() / "made.txt" and reader.reads[0].how == "this run's maker"
    assert plan.log_dirs == [out.resolve(), tmp_path.resolve()]  # the folder, and the file's folder
    with pytest.raises(PlanError, match=r"reader: src: no made file at .*missing\.txt"):
        Script_plan("p", [(READER, ["--src", str(tmp_path / "missing.txt")])])  # not made here, not there


def test_the_arguments_parse_back_to_the_same_options(tmp_path):
    parser = MAKER.build_parser()
    args = parser.parse_args(["--out", str(tmp_path), "--exit", "0", "--raise", "boom"])

    assert parser.parse_args(Script_argv(parser, args)) == args
    assert Script_argv(parser, parser.parse_args([])) == ["--exit", "0"]  # defaults kept, nothing set left out
# endregion


# region Step 2
def test_every_problem_refuses_the_run(tmp_path, data_root, capsys):
    requested = [(MAKER, ["--exit", "x"]), (READER, ["--src", str(tmp_path / "missing.txt"), "--bogus"])]

    assert Script_run("bad", requested, in_process=True) == 2

    [run] = (data_root / "scratchpad").iterdir()
    rows = _lines(_log(run))
    assert ("run", "plan", "error", "refused, nothing was run: 2 problems") in rows
    assert [(s, lvl) for _, s, lvl, t in rows if "invalid int value" in t or "unrecognized" in t] == \
        [("maker", "error"), ("reader", "error")]
    assert [p.name for p in run.iterdir()] == [_log(run).name]  # nothing ran, nothing made
# endregion


# region Step 3
def test_a_run_logs_every_line_labelled_in_every_output_folder(tmp_path, capsys):
    made, copy = tmp_path / "made", tmp_path / "copies" / "copy.txt"

    assert Script_run("two", [(MAKER, ["--out", str(made)]), (READER, ["--out", str(copy)])], in_process=True) == 0

    first, second = _log(made), _log(copy.parent)
    assert first.read_text() == second.read_text() and first.name == second.name
    rows = _lines(first)
    assert ("cmd", "maker", "info", "made it") in rows and ("cmd", "maker", "info", "an info") in rows
    assert ("cmd", "maker", "!", "a warning") in rows and ("cmd", "reader", "info", "read made.txt") in rows
    assert rows[0][:3] == ("run", "plan", "info") and copy.read_text() == "made"
    assert [t for w, s, _, t in rows if s == "summary"][0].startswith("finished in")
    assert "made it" in capsys.readouterr().out  # the screen gets the same lines


def test_a_failing_stage_stops_the_run_with_its_exit_code(tmp_path, capsys):
    made = tmp_path / "made"

    assert Script_run("fail", [(MAKER, ["--out", str(made), "--exit", "3"]),
                               (READER, ["--out", str(tmp_path / "copy.txt")])], in_process=True) == 3

    rows = _lines(_log(made))
    assert any(text.startswith("exit 3 after") and text.endswith("the run stops here; not run: reader")
               for who, stage, level, text in rows if (who, stage, level) == ("run", "maker", "error"))
    assert ("run", "summary", "error", "stopped at maker, exit 3") in rows
    assert not (tmp_path / "copy.txt").exists()


@pytest.mark.parametrize("how, code, said", [("usage", 2, ("error", "not like that")),
                                             ("ctrl-c", 130, ("!", "interrupted (Ctrl-C)"))])
def test_a_usage_error_or_ctrl_c_inside_a_stage(tmp_path, capsys, how, code, said):
    assert Script_run("stop", [(MAKER, ["--out", str(tmp_path), "--raise", how])], in_process=True) == code

    assert any(level == said[0] and said[1] in text for _, _, level, text in _lines(_log(tmp_path)))


def test_a_crash_is_logged_then_raised(tmp_path, capsys):
    with pytest.raises(RuntimeError, match="boom"):
        Script_run("crash", [(MAKER, ["--out", str(tmp_path), "--raise", "boom"])], in_process=True)

    rows = _lines(_log(tmp_path))
    assert ("run", "maker", "error", "RuntimeError: boom") in rows
    assert ("run", "summary", "error", "stopped at maker, exit 1") in rows
# endregion


# region Step 4
def test_the_stages_run_as_child_processes(tmp_path, capsys):
    made, copy = tmp_path / "made", tmp_path / "copy.txt"

    assert Script_run("kids", [(MAKER, ["--out", str(made)]), (READER, ["--out", str(copy)])], in_process=False) == 0

    rows = _lines(_log(made))
    assert ("cmd", "maker", "info", "made it") in rows and ("cmd", "maker", "!", "a warning") in rows
    assert ("cmd", "reader", "info", "read made.txt") in rows and copy.read_text() == "made"
    assert not any("(in this process)" in text for *_, text in rows)
    crashed = tmp_path / "crashed"
    assert Script_run("kids", [(MAKER, ["--out", str(crashed), "--raise", "boom"])], in_process=False) == 1
    rows = _lines(_log(crashed))
    assert ("cmd", "maker", "error", "Traceback (most recent call last):") in rows
    assert ("cmd", "maker", "error", "RuntimeError: boom") in rows


def test_ctrl_c_waits_for_the_child_then_stops_the_run(tmp_path, monkeypatch, capsys):
    class _Interrupted(subprocess.Popen):
        """Ctrl-C pressed once while the run waits on a stage; any other process waits as usual."""
        def wait(self, timeout=None):
            if MAKER.module in self.args and not getattr(self, "pressed", False):
                self.pressed = True
                raise KeyboardInterrupt
            return super().wait(timeout)

    monkeypatch.setattr(engine.subprocess, "Popen", _Interrupted)
    made = tmp_path / "made"

    assert Script_run("ctrl", [(MAKER, ["--out", str(made)]), (READER, ["--out", str(tmp_path / "c.txt")])],
                      in_process=False) == 130

    rows = _lines(_log(made))
    assert ("cmd", "maker", "info", "made it") in rows  # the child's own output still came through
    assert any(level == "!" and text.startswith("interrupted (Ctrl-C)") and "not run: reader" in text
               for _, stage, level, text in rows if stage == "maker")
    assert ("run", "summary", "!", "interrupted at maker, exit 130") in rows
# endregion


# region Step 5
def test_a_scripts_help_and_its_run_as_a_child(tmp_path, data_root, monkeypatch, capsys):
    assert Script_main(MAKER, ["--help"]) == 0
    assert "Through this script" in capsys.readouterr().out and not (data_root / "scratchpad").exists()

    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", root.handlers[:])  # a child sets up its own logging
    monkeypatch.setitem(os.environ, CHILD_ENV, "maker")  # started by a run: only runs
    assert Script_main(MAKER, ["--out", str(tmp_path), "--exit", "4"]) == 4
    assert (tmp_path / "made.txt").exists() and not list(tmp_path.glob("*.log"))
# endregion
