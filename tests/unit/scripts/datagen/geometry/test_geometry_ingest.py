"""
Tests for `scripts.datagen.geometry.ingest`, the ingest's command line, i.e.:

Step 1: Ensure the options and the CST settings reach the ingest, its data folder the run folder's when left out, its
digest printed and its suspect points warned
    - test_the_options_reach_the_ingest
Step 2: Ensure a mistake in what was asked stops it before the ingest runs, and the shared manifest takes only
constants.py's CST values
    - test_a_mistake_is_a_usage_error
    - test_the_shared_manifest_takes_only_the_constants_cst
Step 3: Ensure an ingest into the shared manifest keeps the parity reference in step
    - test_after_a_shared_ingest_the_reference_is_rewritten_or_flagged

The ingest itself is a stand-in; it is tested in the geometry tiers.
"""

import re
from types import SimpleNamespace

import pytest

from scripts.datagen.geometry import ingest as geometry_ingest
from scripts.datagen.engine import Resolved
from src.datagen.constants import CST_FIT_TOL, CST_NOSE, CST_ORDER

_CONSTANTS = {"order": CST_ORDER, "nose": CST_NOSE, "fit_tol": CST_FIT_TOL}


@pytest.fixture
def ingest(monkeypatch):
    calls = []
    summary = SimpleNamespace(summary_path="raw/datagen/geometry/report/x/summary.txt")
    monkeypatch.setattr(geometry_ingest, "Geom_ingest_folder", lambda *args, **kw: calls.append((*args, kw)) or summary)
    monkeypatch.setattr(geometry_ingest, "Geom_format_digest", lambda s: "the digest")
    monkeypatch.setattr(geometry_ingest, "Geom_suspect_points", lambda s: ["a suspect point"])
    return calls


# region Step 1
def test_the_options_reach_the_ingest(ingest, tmp_path, data_root, capsys):
    here, data = tmp_path.resolve(), tmp_path.resolve() / "data_out"

    assert geometry_ingest.main([str(tmp_path), "--data-root", str(data), "--dry-run"]) == 0
    assert geometry_ingest.main(["--data-root", str(data), "--set", "cst_order=10", "--set", "cst_fit_tol=2e-3"]) == 0
    assert geometry_ingest.main([str(tmp_path)]) == 0
    left_out = ingest[2][1]
    assert left_out.name == "geometry_ingest" and left_out.parent.parent == data_root / "scratchpad"  # its run folder
    assert ingest == [(here, data, True, _CONSTANTS),
                      (None, data, False, {**_CONSTANTS, "order": 10, "fit_tol": 2e-3}),  # the stored airfoils only
                      (here, left_out, False, _CONSTANTS)]
    printed = capsys.readouterr().out
    assert "the digest" in printed and f"Every file's line: {data / 'raw'}" in printed
    assert re.search(r"\[cmd\] \[geometry_ingest\] +\[!\] +a suspect point", printed)  # a warning
# endregion


# region Step 2
@pytest.mark.parametrize("args, says", [
    (["nope"], "folder: no .dat folder at"),
    ([".", "--set", "cst_nose=pointy"], "cst_nose is one of separate, shared, lem, not 'pointy'"),
    ([".", "--set", "cst_order=0"], "cst_order is 1 or more, not 0"),
], ids=["no folder", "no such nose", "no such order"])
def test_a_mistake_is_a_usage_error(ingest, tmp_path, capsys, args, says):
    assert geometry_ingest.main([str(tmp_path / a) if a in (".", "nope") else a for a in args]) == 2

    assert says in capsys.readouterr().out and ingest == []


def test_the_shared_manifest_takes_only_the_constants_cst(ingest, tmp_path, data_root, capsys):
    assert geometry_ingest.main(["--data-root", "shared", "--set", "cst_order=10", "--set", "cst_nose=lem"]) == 2
    assert ("cst_order 10: the shared manifest is made with constants.py's CST values" in capsys.readouterr().out
            and ingest == [])  # constants.py's own nose given is no change

    assert geometry_ingest.main(["--data-root", "shared", "--dry-run", "--set", "cst_order=10"]) == 0  # writes nothing
    assert ingest[0][3]["order"] == 10
# endregion


# region Step 3
class _Run:
    def __init__(self, code=0):
        self.said, self.relayed, self.code = [], [], code

    def say(self, stage, level, text):
        self.said.append((stage, level, text))

    def relay(self, *args):
        self.relayed.append(args)
        return self.code


@pytest.mark.parametrize("how, problem, passed, said", [
    ("new", "stale", {"machine_check"}, None),
    ("shared", None, {"machine_check"}, ("info", "still lists")),
    ("shared", "the reference lists other airfoils: rewrite", set(), ("!", "the reference lists other airfoils")),
    ("shared", "the reference lists other airfoils: rewrite", {"machine_check"}, ("!", "commit tests/")),
], ids=["not shared", "still in step", "unchecked: flagged", "checked: rewritten"])
def test_after_a_shared_ingest_the_reference_is_rewritten_or_flagged(monkeypatch, how, problem, passed, said):
    monkeypatch.setattr(geometry_ingest, "Script_reference_problem", lambda: problem)
    plan = SimpleNamespace(writes=[Resolved("data_root", None, how)], args=SimpleNamespace(dry_run=False))
    run = _Run()

    geometry_ingest._after(plan, passed, run)

    if said is None:
        assert run.said == [] and run.relayed == []
        return
    stage, level, text = run.said[-1]
    assert (stage, level) == ("geometry_ingest", said[0]) and said[1] in text
    rewrote = [("geometry_ingest", "scripts.datagen.meshing.machine_check", ["--rewrite"], "machine_check")]
    assert run.relayed == (rewrote if problem and passed else [])
# endregion
