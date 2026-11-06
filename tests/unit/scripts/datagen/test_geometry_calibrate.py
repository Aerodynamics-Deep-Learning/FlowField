"""
Tests for `scripts.datagen.geometry_calibrate`, the CST calibration's command line, i.e.:

Step 1: Ensure the options reach the calibration and its report is printed
    - test_the_options_reach_the_calibration
Step 2: Ensure a mistake in what was asked stops it before the calibration runs
    - test_a_mistake_is_a_usage_error

The calibration itself is a stand-in; it is tested in the geometry tier.
"""

import pytest

from scripts.datagen import geometry_calibrate


@pytest.fixture
def calibrate(monkeypatch):
    calls = []
    monkeypatch.setattr(geometry_calibrate, "Geom_run_calibration", lambda *args: calls.append(args) or "the report")
    return calls


# region Step 1
def test_the_options_reach_the_calibration(calibrate, tmp_path, capsys):
    out = tmp_path / "cst.txt"

    assert geometry_calibrate.main([str(tmp_path), "--out", str(out), "--orders", "4-6", "--workers", "2"]) == 0
    assert geometry_calibrate.main([str(tmp_path), "--out", str(out)]) == 0
    assert calibrate == [(tmp_path, out, range(4, 7), 2), (tmp_path, out, range(4, 13), None)]
    printed = capsys.readouterr().out
    assert "the report" in printed and f"Written to {out}" in printed
# endregion


# region Step 2
@pytest.mark.parametrize("args, says", [
    (["nope", "--out", "r.txt"], "nope is not a folder"),
    ([".", "--out", "r.txt", "--orders", "6-4"], "orders are 'lo-hi' from 1 up, not '6-4'"),
    ([".", "--out", "r.txt", "--orders", "x"], "orders are 'lo-hi' from 1 up, not 'x'"),
    (["."], "required: --out"),
], ids=["no folder", "orders reversed", "orders unreadable", "no out"])
def test_a_mistake_is_a_usage_error(calibrate, capsys, args, says):
    with pytest.raises(SystemExit) as stopped:
        geometry_calibrate.main(args)

    assert stopped.value.code == 2 and says in capsys.readouterr().err and calibrate == []
# endregion
