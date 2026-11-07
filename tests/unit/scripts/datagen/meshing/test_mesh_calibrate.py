"""
Tests for `scripts.datagen.meshing.calibrate`, the ladder calibration's command line, i.e.:

Step 1: Ensure the options reach the calibration, its folder the run folder's when left out, and its files named
    - test_the_options_reach_the_calibration
Step 2: Ensure a mistake in what was asked stops it before anything meshes
    - test_a_mistake_is_a_usage_error

The calibration and the manifest are stand-ins; the calibration is tested in the optimizer tiers.
"""

import datetime
from pathlib import Path

import pytest

from scripts.datagen.meshing import calibrate as mesh_calibrate
from src.datagen.constants import CAL_RUNG2_CAP, CAL_RUNG3_CAP, WORKERS
from src.datagen.geometry.common.paths import Geom_paths
from src.datagen.meshing.optimizer.batch.calibrate import CalibrationFiles


def _known(names, db=None):
    unknown = [n for n in names if n not in ("naca0012", "s1223")]
    if unknown:
        raise ValueError(f"no shape under the current shape version for {', '.join(unknown)}")
    return [(n, None) for n in names]


@pytest.fixture
def calibrate(monkeypatch):
    calls = []

    def run(*args):
        calls.append(args)
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
    assert calibrate == [(out, [1, 3], ["naca0012", "s1223"], 150, 30, 10, 2, manifest, work),
                         (left_out, [1, 2, 3], None, None, CAL_RUNG3_CAP, CAL_RUNG2_CAP, WORKERS, Geom_paths().db, None)]
    printed = capsys.readouterr().out
    assert "the report" not in printed and "r.txt; every run: r.sqlite" in printed  # the findings came per rung
# endregion


# region Step 2
@pytest.mark.parametrize("args, says", [
    (["--rungs", "4"], "rungs are numbers from 1,2,3, not '4'"),
    (["--rungs", "1,x"], "rungs are numbers from 1,2,3, not '1,x'"),
    (["--airfoils", "naca0012,naca9999"], "naca9999"),
    (["--out", "TAKEN"], "already holds today's calibration"),
], ids=["rung 4", "rung unreadable", "unknown airfoil", "today's calibration there"])
def test_a_mistake_is_a_usage_error(calibrate, tmp_path, capsys, args, says):
    taken = tmp_path / "taken"
    taken.mkdir()
    (taken / f"{datetime.date.today().isoformat()}.sqlite").touch()

    assert mesh_calibrate.main([str(taken) if a == "TAKEN" else a for a in args]) == 2

    assert says in capsys.readouterr().out and calibrate == []
# endregion
