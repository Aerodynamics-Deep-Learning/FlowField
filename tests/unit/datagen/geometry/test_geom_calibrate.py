"""
Tests for `geometry.calibrate`: the order/tolerance sweep and its report, i.e.:

Step 1: Ensure every file is fitted at every configuration, or its rejection kept
    - test_calibrate_rows
    - test_parallel_matches_sequential
    - test_strays
Step 2: Ensure the report and the command line
    - test_format_calibration
    - test_main_writes_the_report
"""

import pytest

from src.datagen.geometry.build.calibrate import CalibrationFit, Geom_calibrate, Geom_format_calibration, main
from src.datagen.geometry.build.families import _family_contour
from src.datagen.geometry.common.io import Geom_write_dat
from src.datagen.geometry.common.schemas import Family, Nose, RejectReason


@pytest.fixture
def folder(tmp_path):
    root = tmp_path / "roots"
    root.mkdir()
    Geom_write_dat(root / "naca2412.dat", "naca2412", _family_contour(Family.NACA4, {"m": 2, "p": 4, "t": 12}, 101, False))
    Geom_write_dat(root / "testfoil.dat", "testfoil", _family_contour(Family.NACA4, {"m": 0, "p": 0, "t": 12}, 101, False))
    (root / "broken.dat").write_text("broken\n1.0 0.0\n")
    return root


# region Step 1
def test_calibrate_rows(folder):
    rows = Geom_calibrate(folder, orders=[4, 6], noses=(Nose.SEPARATE, Nose.LEM), workers=1)

    assert [r.source for r in rows] == ["broken.dat", "naca2412.dat", "testfoil.dat"]
    assert rows[0].reject is RejectReason.FORMAT and rows[0].fits == []
    assert rows[1].family is Family.NACA4 and rows[2].family is None
    for row in rows[1:]:
        assert [(f.nose, f.order) for f in row.fits] == [(Nose.SEPARATE, 4), (Nose.SEPARATE, 6), (Nose.LEM, 4),
                                                         (Nose.LEM, 6)]
        lem = {f.order: f for f in row.fits if f.nose is Nose.LEM}
        assert lem[6].max_dev < lem[4].max_dev
        assert not any(f.crossed or f.strays for f in row.fits)


def test_parallel_matches_sequential(folder):
    assert Geom_calibrate(folder, orders=[5], workers=2) == Geom_calibrate(folder, orders=[5], workers=1)


def test_strays():
    assert CalibrationFit(order=8, nose=Nose.LEM, max_dev=1e-4, between_dev=1e-3).strays
    assert not CalibrationFit(order=8, nose=Nose.LEM, max_dev=1e-4, between_dev=4e-4).strays  # under the floor
    assert not CalibrationFit(order=8, nose=Nose.LEM, max_dev=1e-3, between_dev=1.5e-3).strays  # under twice
    assert not CalibrationFit(order=8, nose=Nose.LEM).strays
# endregion


# region Step 2
def test_format_calibration(folder):
    text = Geom_format_calibration(Geom_calibrate(folder, orders=[4, 6], workers=1), folder)

    assert text.startswith("CST calibration of roots\n")  # the folder's name only, never a local path
    assert "3 files: 2 fitted (2 unique geometries), 1 rejected before fitting" in text
    assert "separate    shared       lem" in text  # one column per nose in the family table
    assert "broken.dat" in text and "format" in text
    assert "Formula families only (1 shapes" in text
    assert any(line.startswith("naca2412.dat") for line in text.splitlines())


def test_main_writes_the_report(folder, tmp_path, capsys):
    out = tmp_path / "report" / "calibration.txt"
    main([str(folder), "--orders", "4-5", "--out", str(out), "--workers", "1"])

    assert out.read_text(encoding="utf-8").startswith("CST calibration of")
    assert str(out) in capsys.readouterr().out
# endregion
