"""
Tests for `geometry.calibrate`: the order/tolerance sweep and its report, i.e.:

Step 1: Ensure every file is fitted at every configuration, or its rejection kept
    - test_calibrate_rows
    - test_parallel_matches_sequential
    - test_strays
Step 2: Ensure the report, and that a calibration writes it
    - test_format_calibration
    - test_a_calibration_writes_its_report_and_returns_it_without_the_per_airfoil_table
"""

import pytest

from src.datagen.geometry.build.calibrate import (
    CalibrationFit, Geom_calibrate, Geom_format_calibration, Geom_run_calibration, _orders,
)
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
    assert "Surfaces cross at nose lem: 0" in text  # orders 4 and 6 hold no CST_ORDER: no strays line
    assert _orders([6, 10, 11, 12]) == "orders 6, 10-12" and _orders([8]) == "order 8"


def test_a_calibration_writes_its_report_and_returns_it_without_the_per_airfoil_table(folder, tmp_path):
    out = tmp_path / "report" / "calibration.txt"  # its folder made on the way
    shown = Geom_run_calibration(folder, out, orders=range(4, 6), workers=1)

    report = out.read_text(encoding="utf-8")
    assert report.startswith(shown) and shown.startswith("CST calibration of")  # the table is all it leaves out
    assert "Per airfoil, max_dev" in report and "Per airfoil" not in shown
    assert "Surfaces cross at nose lem: " in shown
# endregion
