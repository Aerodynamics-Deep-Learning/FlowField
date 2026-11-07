"""
Tests for `scripts.datagen.geometry.calibrate`, the CST calibration's command line, i.e.:

Step 1: Ensure the options reach the calibration, its report the scratchpad's when left out, and it is printed
    - test_the_options_reach_the_calibration
Step 2: Ensure a mistake in what was asked stops it before the calibration runs
    - test_a_mistake_is_a_usage_error

The calibration itself is a stand-in; it is tested in the geometry tier.
"""

import datetime

import pytest

from scripts.datagen.geometry import calibrate as geometry_calibrate


@pytest.fixture
def calibrate(monkeypatch):
    calls = []
    monkeypatch.setattr(geometry_calibrate, "Geom_run_calibration", lambda *args: calls.append(args) or "the report")
    return calls


# region Step 1
def test_the_options_reach_the_calibration(calibrate, tmp_path, data_root, capsys):
    here, out = tmp_path.resolve(), tmp_path.resolve() / "cst.txt"

    assert geometry_calibrate.main([str(tmp_path), "--out", str(out), "--orders", "4-6", "--workers", "2"]) == 0
    assert geometry_calibrate.main([str(tmp_path)]) == 0
    [scratch] = [p.parent for p in (data_root / "scratchpad").glob("*/geometry_calibrate")]  # the run that left it out
    report = scratch / "geometry_calibrate" / f"{datetime.date.today().isoformat()}.txt"
    assert calibrate == [(here, out, range(4, 7), 2), (here, report, range(4, 13), None)]
    printed = capsys.readouterr().out
    assert "the report" in printed and f"Written to {out}" in printed
# endregion


# region Step 2
@pytest.mark.parametrize("args, says", [
    (["nope"], "folder: no .dat folder at"),
    ([".", "--orders", "6-4"], "orders are 'lo-hi' from 1 up, not '6-4'"),
    ([".", "--orders", "x"], "orders are 'lo-hi' from 1 up, not 'x'"),
], ids=["no folder", "orders reversed", "orders unreadable"])
def test_a_mistake_is_a_usage_error(calibrate, capsys, args, says):
    assert geometry_calibrate.main(args) == 2

    assert says in capsys.readouterr().out and calibrate == []
# endregion
