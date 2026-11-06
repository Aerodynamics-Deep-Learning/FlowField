"""
Tests for `scripts.datagen.mesh_calibrate`, the ladder calibration's command line, i.e.:

Step 1: Ensure the options reach the calibration and its report is printed
    - test_the_options_reach_the_calibration
Step 2: Ensure a mistake in what was asked stops it before anything meshes
    - test_a_mistake_is_a_usage_error

The calibration and the manifest are stand-ins; the calibration is tested in the optimizer tiers.
"""

from pathlib import Path

import pytest

from scripts.datagen import mesh_calibrate
from src.datagen.meshing.optimizer.batch.calibrate import RUNG2_CAP, RUNG3_CAP, CalibrationFiles
from src.datagen.meshing.optimizer.common.constants import WORKERS


def _known(names, db=None):
    unknown = [n for n in names if n not in ("naca0012", "s1223")]
    if unknown:
        raise ValueError(f"no shape under the current shape version for {', '.join(unknown)}")
    return [(n, None) for n in names]


@pytest.fixture
def calibrate(monkeypatch):
    calls = []

    def run(*args):
        if Path(args[0]).name == "taken":
            raise FileExistsError(f"{args[0]} already holds a calibration; give another folder")
        calls.append(args)
        return CalibrationFiles(Path("r.txt"), Path("r.sqlite"), "the report")

    monkeypatch.setattr(mesh_calibrate, "Optim_run_calibration", run)
    monkeypatch.setattr(mesh_calibrate, "Optim_load_shapes", _known)
    return calls


# region Step 1
def test_the_options_reach_the_calibration(calibrate, tmp_path, capsys):
    out, manifest, work = tmp_path / "cal", tmp_path / "geometry.sqlite", tmp_path / "work"

    assert mesh_calibrate.main(["--out", str(out), "--rungs", "3,1", "--airfoils", "naca0012, s1223",
                                "--golden-jmax", "150", "--rung3-cap", "30", "--rung2-cap", "10", "--workers", "2",
                                "--geometry-db", str(manifest), "--work-root", str(work)]) == 0
    assert mesh_calibrate.main(["--out", str(out)]) == 0
    assert calibrate == [(out, [1, 3], ["naca0012", "s1223"], 150, 30, 10, 2, manifest, work),
                         (out, [1, 2, 3], None, None, RUNG3_CAP, RUNG2_CAP, WORKERS, None, None)]
    printed = capsys.readouterr().out
    assert "the report" in printed and "Written to r.txt and r.sqlite" in printed
# endregion


# region Step 2
@pytest.mark.parametrize("args, says", [
    (["--out", "o", "--rungs", "4"], "rungs are numbers from 1,2,3, not '4'"),
    (["--out", "o", "--rungs", "1,x"], "rungs are numbers from 1,2,3, not '1,x'"),
    (["--out", "o", "--airfoils", "naca0012,naca9999"], "naca9999"),
    (["--out", "taken"], "already holds a calibration"),
    ([], "required: --out"),
], ids=["rung 4", "rung unreadable", "unknown airfoil", "today's calibration there", "no out"])
def test_a_mistake_is_a_usage_error(calibrate, capsys, args, says):
    with pytest.raises(SystemExit) as stopped:
        mesh_calibrate.main(args)

    assert stopped.value.code == 2 and says in capsys.readouterr().err and calibrate == []
# endregion
