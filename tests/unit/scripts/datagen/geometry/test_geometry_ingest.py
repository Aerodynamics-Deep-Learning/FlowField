"""
Tests for `scripts.datagen.geometry.ingest`, the ingest's command line, i.e.:

Step 1: Ensure the options reach the ingest, its data folder the scratchpad's when left out, its summary printed
    - test_the_options_reach_the_ingest
Step 2: Ensure a mistake in what was asked stops it before the ingest runs
    - test_a_mistake_is_a_usage_error
Step 3: Ensure an ingest into the shared manifest keeps the parity reference in step
    - test_after_a_shared_ingest_the_reference_is_rewritten_or_flagged

The ingest itself is a stand-in; it is tested in the geometry tiers.
"""

from types import SimpleNamespace

import pytest

from scripts.datagen.geometry import ingest as geometry_ingest
from scripts.datagen.engine import Resolved


@pytest.fixture
def ingest(monkeypatch):
    calls = []
    summary = SimpleNamespace(summary_path="raw/datagen/geometry/report/x/summary.txt")
    monkeypatch.setattr(geometry_ingest, "Geom_ingest_folder", lambda *args: calls.append(args) or summary)
    monkeypatch.setattr(geometry_ingest, "Geom_format_summary", lambda s: "the summary")
    return calls


# region Step 1
def test_the_options_reach_the_ingest(ingest, tmp_path, data_root, capsys):
    here, data = tmp_path.resolve(), tmp_path.resolve() / "data_out"

    assert geometry_ingest.main([str(tmp_path), "--data-root", str(data), "--dry-run"]) == 0
    assert geometry_ingest.main(["--data-root", str(data)]) == 0  # no folder: the stored airfoils only
    assert geometry_ingest.main([str(tmp_path)]) == 0
    left_out = ingest[2][1]
    assert left_out.name == "geometry_ingest" and left_out.parent.parent == data_root / "scratchpad"  # its run folder
    assert ingest == [(here, data, True), (None, data, False), (here, left_out, False)]
    printed = capsys.readouterr().out
    assert "the summary" in printed and f"Summary written to {data / 'raw'}" in printed
# endregion


# region Step 2
def test_a_mistake_is_a_usage_error(ingest, tmp_path, capsys):
    assert geometry_ingest.main([str(tmp_path / "nope")]) == 2

    assert "folder: no .dat folder at" in capsys.readouterr().out and ingest == []
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
