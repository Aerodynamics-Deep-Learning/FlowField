"""
Tests for `geometry.calibrate`: the order/tolerance sweep and its report, i.e.:

Step 1: Ensure every file is fitted at every configuration, or its rejection kept
    - test_calibrate_rows
    - test_parallel_matches_sequential
    - test_strays
Step 2: Ensure the report, the digest a log shows, and that a calibration writes the one and returns the other
    - test_format_calibration
    - test_the_digest_shows_the_ingests_configuration_and_its_neighbours
    - test_a_calibration_writes_its_report_and_returns_its_digest
"""

import pytest

from src.datagen.geometry.build.calibrate import (
    CalibrationFit, Geom_calibrate, Geom_calibration_digest, Geom_format_calibration, Geom_run_calibration, _orders,
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


def _table(text):
    """The digest's rows as (order, nose, kept + over + cross, marked)."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.lstrip().startswith("order")) + 1
    rows = [line.split() for line in lines[start:lines.index("", start)]]
    return [(int(r[0]), r[1], int(r[2]) + int(r[3]) + int(r[4]), "ingest's" in r[-1]) for r in rows]


def test_the_digest_shows_the_ingests_configuration_and_its_neighbours(folder):
    text = Geom_calibration_digest(Geom_calibrate(folder, orders=[7, 8, 9], workers=1), folder)

    assert text.startswith("CST calibration of roots\n") and "broken.dat" in text  # header, rejections
    assert "Tolerance 1e-03c, over the 2 unique geometries" in text and "the 1 formula shapes' worst" in text
    assert _table(text) == [(7, "lem", 2, False), (8, "lem", 2, True), (9, "lem", 2, False),
                            (8, "separate", 2, False), (8, "shared", 2, False)]
    assert text.endswith("Strays at order 8, nose lem (the ingest's order): 0")
    # without the ingest's order: around the nearest calibrated one, nothing marked
    assert _table(Geom_calibration_digest(Geom_calibrate(folder, orders=[4, 5], workers=1), folder)) == [
        (4, "lem", 2, False), (5, "lem", 2, False), (5, "separate", 2, False), (5, "shared", 2, False)]


def test_a_calibration_writes_its_report_and_returns_its_digest(folder, tmp_path):
    out = tmp_path / "report" / "calibration.txt"  # its folder made on the way
    shown = Geom_run_calibration(folder, out, orders=range(4, 6), workers=1)

    report = out.read_text(encoding="utf-8")
    assert shown.startswith("CST calibration of") and "Tolerance 1e-03c" in shown
    assert "Per configuration" in report and "Per airfoil, max_dev" in report and "Formula families only" in report
    assert not any(table in shown for table in ("Per configuration", "Per airfoil", "Formula families only"))
    assert "Surfaces cross at nose lem: " in shown and "Surfaces cross at nose lem: " in report
# endregion
