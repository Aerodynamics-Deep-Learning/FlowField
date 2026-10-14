"""
Tests for `geometry.checks`: the gates before an airfoil is stored, i.e.:

Step 1: Ensure the fit gate reads its limit inclusively
    - test_check_fit
Step 2: Ensure the thickness gate finds both kinds of crossing, and lets a cusp through
    - test_check_thickness_passes_an_airfoil
    - test_check_thickness_passes_a_cusp
    - test_check_thickness_crossing_inside
    - test_check_thickness_crossed_te
Step 3: Ensure the family gate tells a name's shape from another
    - test_check_family_match
Step 4: Ensure a lone, zig-zagging worst point is flagged and nothing else is
    - test_check_suspect_point
"""

import numpy as np

from src.datagen.geometry.build.checks import (
    Geom_check_family_match, Geom_check_fit, Geom_check_suspect_point, Geom_check_thickness,
)
from src.datagen.geometry.common.constants import CST_N1, CST_N2, FAMILY_MATCH_TOL
from src.datagen.geometry.build.families import Geom_family_coords
from src.datagen.geometry.build.normalize import Geom_close_te, Geom_normalize
from src.datagen.geometry.common.schemas import CSTParams, Family, FitMetrics


def _cst(upper, lower) -> CSTParams:
    return CSTParams(n1=CST_N1, n2=CST_N2, upper=upper, lower=lower)


def _normalized(t: int):
    q, tf = Geom_normalize(Geom_family_coords(Family.NACA4, {"m": 0, "p": 0, "t": t}, 201))
    return Geom_close_te(q, tf.le_index)[0], tf.le_index


# region Step 1
def _metrics(max_dev=1e-3, isolation=1.0, zigzag=False) -> FitMetrics:
    return FitMetrics(max_dev=max_dev, rms_dev=1e-4, x_at_max=0.3, n_points=10, isolation=isolation, zigzag=zigzag)


def test_check_fit():
    at_limit, over = _metrics(1e-3), _metrics(1.1e-3)

    assert Geom_check_fit(at_limit, tol=1e-3) == (True, 1e-3, "")
    ok, value, reason = Geom_check_fit(over, tol=1e-3)
    assert not ok and value == 1.1e-3 and "1.10e-03" in reason and "x=0.3000" in reason
# endregion


# region Step 2
def test_check_thickness_passes_an_airfoil():
    ok, value, reason = Geom_check_thickness(_cst([0.17] * 5, [-0.12] * 5))

    assert ok and value > 0 and reason == ""


def test_check_thickness_passes_a_cusp():
    ok, value, reason = Geom_check_thickness(_cst([0.17, 0.2, 0.2, 0.2, 0.1], [-0.12, -0.1, -0.1, -0.1, 0.1]))

    assert ok and value == 0.0 and reason == ""  # zero wedge: the surfaces arrive at one slope


def test_check_thickness_crossing_inside():
    ok, value, reason = Geom_check_thickness(_cst([0.17, 0.1, -0.8, 0.1, 0.1], [-0.12] * 5))

    assert not ok and value < 0
    assert "cross at x=0.5" in reason


def test_check_thickness_crossed_te():
    ok, _, reason = Geom_check_thickness(_cst([0.17, 0.2, 0.2, 0.2, 0.05], [-0.12, -0.1, -0.1, -0.1, 0.1]))

    assert not ok
    assert "crossed" in reason and "wedge angle -" in reason
# endregion


# region Step 3
def test_check_family_match():
    pts12, le12 = _normalized(12)
    pts21, le21 = _normalized(21)

    ok, dev, _ = Geom_check_family_match(pts12, le12, pts12, le12)
    assert ok and dev < 1e-12
    ok, dev, reason = Geom_check_family_match(pts12, le12, pts21, le21)
    assert not ok and dev > FAMILY_MATCH_TOL and "miss the shape" in reason
# endregion


# region Step 4
def test_check_suspect_point():
    ok, value, note = Geom_check_suspect_point(_metrics(isolation=6.0, zigzag=True))
    assert not ok and value == 6.0 and "x=0.3000" in note and "6.0x" in note

    assert Geom_check_suspect_point(_metrics(isolation=6.0, zigzag=False))[0]  # a lone miss on a smooth run: a nose
    assert Geom_check_suspect_point(_metrics(isolation=2.0, zigzag=True))[0]   # its neighbours miss too: noise
# endregion
