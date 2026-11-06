"""
Tests for `scripts.datagen.parity`, the machine check, and `scripts.datagen.machine_check`, its script, i.e.:

Step 1: Ensure a check passes only when every test passed, a skip counting as a failure, and only a pass is stamped
    - test_a_check_passes_only_when_every_test_passed
    - test_the_machine_check_stamps_only_a_pass
Step 2: Ensure the stamp says what moved since the last pass
    - test_the_stamp_names_what_moved
Step 3: Ensure a reference that cannot check this code is refused first, and a rewrite keeps an unchanged file
    - test_a_reference_that_does_not_fit_is_refused
    - test_a_rewrite_keeps_the_old_file_when_only_the_writer_changed

The check runs pytest for real on stand-in test files; the stamp's view of this machine is a stand-in.
"""

import json
import subprocess

import pytest

from scripts.datagen import machine_check, parity
from scripts.datagen.parity import CheckResult, Script_machine_check, Script_stamp_problem
from src.datagen.geometry.common.constants import SAMPLER_VERSION, SHAPE_VERSION

_STEP1, _STEP2 = "test_this_machine_samples_like_the_reference", "test_this_machine_meshes_like_the_reference"


# region Step 1
@pytest.mark.parametrize("body, passed, steps", [
    (f"def {_STEP1}():\n    pass\ndef {_STEP2}():\n    pass\n", True,
     [("step 1, sampling", "passed", ""), ("step 2, c2d", "passed", "")]),
    (f"import pytest\ndef {_STEP1}():\n    pass\ndef {_STEP2}():\n    pytest.skip('no c2d here')\n", False,
     [("step 1, sampling", "passed", ""), ("step 2, c2d", "skipped", "no c2d here")]),
    (f"def {_STEP1}():\n    assert False, '3 of 158 airfoils sample differently here'\n", False,
     [("step 1, sampling", "failed", "AssertionError: 3 of 158 airfoils sample differently here\nassert False")]),
    ("x = 1\n", False, []),
], ids=["both pass", "a skip", "a failure", "no tests"])
def test_a_check_passes_only_when_every_test_passed(tmp_path, body, passed, steps):
    test = tmp_path / "test_standin.py"
    test.write_text(body, encoding="utf-8")

    result = Script_machine_check(test)

    assert result.passed is passed and result.steps == steps


@pytest.fixture
def here(monkeypatch):
    """This machine as the stamp sees it, without hashing a c2d."""
    state = {"host": "here", "mesher_sha256": "a" * 64, "reference_sha256": "r" * 64,
             "sampler_version": SAMPLER_VERSION, "shape_version": SHAPE_VERSION}
    monkeypatch.setattr(parity, "_state", lambda: dict(state))
    return state


def test_the_machine_check_stamps_only_a_pass(here, monkeypatch, capsys):
    results = iter([CheckResult(False, [("step 2, c2d", "skipped", "no c2d here")]),
                    CheckResult(True, [("step 1, sampling", "passed", ""), ("step 2, c2d", "passed", "")])])
    monkeypatch.setattr(machine_check, "Script_machine_check", lambda: next(results))
    monkeypatch.setattr(machine_check, "Script_reference_problem", lambda: None)

    assert machine_check.main([]) == 1
    assert "a skip counts as a failure" in capsys.readouterr().out and not parity.STAMP.exists()
    assert machine_check.main([]) == 0
    assert json.loads(parity.STAMP.read_text())["host"] == "here" and Script_stamp_problem() is None

    monkeypatch.setattr(machine_check, "Script_reference_problem", lambda: "the parity reference is for samp-r1")
    assert machine_check.main([]) == 2  # refused before the test runs
    assert "the parity reference is for samp-r1" in capsys.readouterr().out
    monkeypatch.setattr(machine_check, "Script_rewrite_reference", lambda: "machine_parity.json unchanged")
    assert machine_check.main(["--rewrite"]) == 0  # no reference check before a rewrite
    assert "machine_parity.json unchanged" in capsys.readouterr().out
# endregion


# region Step 2
def test_the_stamp_names_what_moved(here, monkeypatch):
    assert Script_stamp_problem().startswith("this machine has not passed machine_check: meshes made here")
    parity.Script_write_stamp()
    assert Script_stamp_problem() is None

    for key, value, says in (("host", "elsewhere", "since it passed on here, not on this computer"),
                             ("mesher_sha256", "b" * 64, "since its c2d changed (aaaaaaaa... -> bbbbbbbb...)"),
                             ("reference_sha256", "s" * 64, "since the parity reference changed"),
                             ("mesher_sha256", None, "no c2d executable is found here")):
        monkeypatch.setattr(parity, "_state", lambda: {**here, key: value})
        assert says in Script_stamp_problem()
    monkeypatch.setattr(parity, "_state", lambda: {**here, "sampler_version": "samp-r0"})
    parity.Script_write_stamp()  # stamped under an older sampler
    monkeypatch.setattr(parity, "_state", lambda: dict(here))
    assert f"the sampler or shape version changed (samp-r0/{SHAPE_VERSION}" in Script_stamp_problem()
# endregion


# region Step 3
def _reference(samples, written_on="there", **versions) -> dict:
    return {"sampler_version": versions.get("sampler", SAMPLER_VERSION), "shape_version": SHAPE_VERSION,
            "written_on": written_on, "nstar_grid": "g", "samples": samples, "c2d": {"meshes": {"a": "m"}}}


def test_a_reference_that_does_not_fit_is_refused(tmp_path, monkeypatch):
    reference = tmp_path / "machine_parity.json"
    monkeypatch.setattr(parity, "REFERENCE", reference)
    monkeypatch.setattr("src.datagen.meshing.optimizer.batch.runner.Optim_load_shapes",
                        lambda: [("a", None), ("b", None)])

    assert "no parity reference at" in parity.Script_reference_problem()
    reference.write_text(json.dumps(_reference({"a": "1"}, sampler="samp-r0")))
    assert "This is a release step" in parity.Script_reference_problem()
    reference.write_text(json.dumps(_reference({"a": "1"})))
    assert "lists other airfoils than the shared manifest (1 not in it, 0 gone)" in parity.Script_reference_problem()
    reference.write_text(json.dumps(_reference({"a": "1", "b": "2"})))
    assert parity.Script_reference_problem() is None


@pytest.mark.parametrize("samples, says", [({"a": "1"}, "unchanged"),
                                           ({"a": "1", "b": "2"}, "rewritten on here: 2 airfoils (1 new, 0 gone), "
                                                                  "1 meshes")], ids=["same", "an airfoil more"])
def test_a_rewrite_keeps_the_old_file_when_only_the_writer_changed(tmp_path, monkeypatch, samples, says):
    reference = tmp_path / "machine_parity.json"
    old_text = json.dumps(_reference({"a": "1"}), indent=1)
    reference.write_text(old_text, encoding="utf-8")
    monkeypatch.setattr(parity, "REFERENCE", reference)

    def writer(cmd, **kw):
        reference.write_text(json.dumps(_reference(samples, written_on="here")), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(parity.subprocess, "run", writer)

    assert says in parity.Script_rewrite_reference()
    assert (reference.read_text(encoding="utf-8") == old_text) is (says == "unchanged")
    monkeypatch.setattr(parity.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "boom"))
    with pytest.raises(RuntimeError, match="the reference writer failed"):
        parity.Script_rewrite_reference()
# endregion
