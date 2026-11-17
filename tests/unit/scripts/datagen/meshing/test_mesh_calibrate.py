"""
Tests for `scripts.datagen.meshing.calibrate`, the ladder calibration's command line, i.e.:

Step 1: Ensure the options and settings reach the calibration, its folder the run folder's when left out, its files
named, and the run's account of itself for its rungs' batch rows
    - test_the_options_reach_the_calibration
    - test_settings_reach_the_calibration
Step 2: Ensure a mistake in what was asked stops it before anything meshes
    - test_a_mistake_is_a_usage_error

The calibration and the manifest are stand-ins; the calibration is tested in the optimizer tiers.
"""

from pathlib import Path

import pytest

from scripts.datagen import engine
from scripts.datagen.meshing import calibrate as mesh_calibrate
from src.datagen.constants import CAL_RUNG2_CAP, CAL_RUNG3_CAP, SLOT_MEMORY_CAP_MB, WATCHDOG_S, WORKERS
from src.datagen.meshing.optimizer.batch.calibrate import CalibrationFiles, Method


def _known(names, db=None, shape_version=None):
    unknown = [n for n in names or () if n not in ("naca0012", "s1223")]
    if unknown:
        raise ValueError(f"no shape under the current shape version for {', '.join(unknown)}")
    return [(n, None) for n in names or ("naca0012", "s1223")]


@pytest.fixture
def calibrate(monkeypatch):
    calls = []

    def run(*args, **kw):
        calls.append((*args, kw))
        return CalibrationFiles(Path("r.txt"), Path("r.sqlite"), "the report")

    monkeypatch.setattr(mesh_calibrate, "Optim_run_calibration", run)
    monkeypatch.setattr(mesh_calibrate, "Optim_load_shapes", _known)
    return calls


# region Step 1
def test_the_options_reach_the_calibration(calibrate, tmp_path, data_root, capsys):
    here = tmp_path.resolve()
    out, manifest, work = here / "cal", here / "geometry.sqlite", tmp_path / "work"
    manifest.touch()

    assert mesh_calibrate.main(["--out", str(out), "--rungs", "3,1", "--airfoils", "naca0012, s1223",
                                "--golden-jmax", "150", "--rung3-cap", "30", "--rung2-cap", "10", "--workers", "2",
                                "--geometry-db", str(manifest), "--work-root", str(work)]) == 0
    assert mesh_calibrate.main([]) == 0
    left_out = calibrate[1][0]
    assert left_out.name == "mesh_calibrate" and left_out.parent.parent == data_root / "scratchpad"  # its run folder
    constants = {"overrides": {}, "method": Method(), "watchdog_s": WATCHDOG_S, "cap_mb": SLOT_MEMORY_CAP_MB}
    given, _ = (kw.pop("invocation") for *_, kw in calibrate)
    assert calibrate == [(out, [1, 3], ["naca0012", "s1223"], 150, 30, 10, 2, manifest, work, constants),
                         (left_out, [1, 2, 3], None, None, CAL_RUNG3_CAP, CAL_RUNG2_CAP, WORKERS,
                          engine.SHARED_GEOMETRY_DB, None, constants)]
    printed = capsys.readouterr().out
    assert "the report" not in printed and "r.txt; every run: r.sqlite" in printed  # the findings came per rung
    assert (given.run_name, given.config_yaml) == ("mesh_calibrate", None)
    assert given.command.startswith(f"python -m {mesh_calibrate.STAGE.module} --out ")
    assert given.machine_check["problem"].startswith("this machine has not passed machine_check")


def test_settings_reach_the_calibration(calibrate, tmp_path):
    assert mesh_calibrate.main(["--out", str(tmp_path / "cal"), "--set", "gate_skew_max=0.22", "--set",
                                "golden_jmax=150", "--set", "cal_golden_margin=3", "--set", "cal_windows=10,20",
                                "--set", "cal_rung2_cap=40", "--set", "slot_memory_cap_mb=1024"]) == 0

    [(_, _, _, golden, rung3_cap, rung2_cap, *_, kw)] = calibrate
    assert (golden, rung3_cap, rung2_cap) == (150, CAL_RUNG3_CAP, 40)  # the golden jmax and caps the calibration's
    assert kw["overrides"] == {"gate_skew_max": 0.22} and kw["cap_mb"] == 1024
    assert kw["method"] == Method(golden_margin=3, windows=(10, 20))
# endregion


# region Step 2
@pytest.mark.parametrize("args, says", [
    (["--rungs", "4"], "rungs are numbers from 1,2,3, not '4'"),
    (["--rungs", "1,x"], "rungs are numbers from 1,2,3, not '1,x'"),
    (["--airfoils", "naca0012,naca9999"], "naca9999"),
    (["--set", "rung2_budget=60"], "mesh_calibrate does not read rung2_budget"),  # it measures it
    (["--rung2-cap", "10", "--set", "cal_rung2_cap=20"], "cal_rung2_cap is given twice"),
], ids=["rung 4", "rung unreadable", "unknown airfoil", "a budget", "cap twice"])
def test_a_mistake_is_a_usage_error(calibrate, capsys, args, says):
    assert mesh_calibrate.main(args) == 2

    assert says in capsys.readouterr().out and calibrate == []
# endregion
