"""
Tests for `geometry.families`: reading designations off names, and the closed-TE formulas, i.e.:

Step 1: Ensure names parse strictly
    - test_parse_family_valid
    - test_parse_family_invalid
    - test_family_key_and_id
Step 2: Ensure every family builds a closed Selig contour
    - test_contours_are_closed_selig
Step 3: Ensure each formula has the properties its definition fixes
    - test_naca4_thickness
    - test_mod4_thickness_conditions
    - test_mod4_63_is_close_to_four_digit
    - test_naca5_camber
    - test_a1_camber

Scope: the formulas' exact agreement with reference ordinates was checked once against the analytic
NACA files (4/5-digit, to 5e-11) and the tabulated naca001066/naca16018 (to 5e-5); those files are not
part of the repo.
"""

import numpy as np
import pytest

from src.datagen.geometry.build.families import (
    Geom_family_coords, Geom_family_id, Geom_family_key, Geom_parse_family,
    _a1_camber, _mod4_thickness, _naca4_thickness, _naca5_camber,
)
from src.datagen.geometry.common.schemas import Family


# region Step 1
@pytest.mark.parametrize("name, family, params", [
    ("naca0012", Family.NACA4, {"m": 0, "p": 0, "t": 12}),
    ("NACA2412", Family.NACA4, {"m": 2, "p": 4, "t": 12}),
    ("naca23012", Family.NACA5, {"l": 2, "p": 3, "s": 0, "t": 12}),
    ("naca23112", Family.NACA5, {"l": 2, "p": 3, "s": 1, "t": 12}),
    ("naca001066", Family.NACA4_MOD, {"m": 0, "p": 0, "t": 10, "i": 6, "pos": 6}),
    ("naca16018", Family.NACA16, {"cl": 0, "t": 18}),
    ("naca16212", Family.NACA16, {"cl": 2, "t": 12}),
])
def test_parse_family_valid(name, family, params):
    assert Geom_parse_family(name) == (family, params)


@pytest.mark.parametrize("name", [
    "naca64a010", "naca66_018", "naca0012-64", "n0012", "clarky",  # 6-series, separators, other names
    "naca0412", "naca2012", "naca0000",                            # 4-digit: camber without position, zero thickness
    "naca26012", "naca21112", "naca00121",                         # 5-digit: no such mean line
    "naca633418", "naca001096",                                    # modified 4-digit: position 8, LE index 9
])
def test_parse_family_invalid(name):
    assert Geom_parse_family(name) is None


def test_family_key_and_id():
    params = {"m": 2, "p": 4, "t": 12}

    assert Geom_family_key(Family.NACA4, params) == "naca4:m=2:p=4:t=12:te=closed"
    assert Geom_family_id(Family.NACA4, params) == Geom_family_id(Family.NACA4, dict(params))
    assert Geom_family_id(Family.NACA4, params) != Geom_family_id(Family.NACA4, {"m": 2, "p": 4, "t": 21})
# endregion


# region Step 2
@pytest.mark.parametrize("name", ["naca0012", "naca2412", "naca23012", "naca23112", "naca001066",
                                  "naca16018", "naca16212"])
def test_contours_are_closed_selig(name):
    coords = Geom_family_coords(*Geom_parse_family(name), n_per_surface=201)
    x, y = coords.T
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)

    assert coords.shape == (401, 2)
    assert np.isfinite(coords).all()
    np.testing.assert_allclose(coords[[0, -1]], [[1.0, 0.0], [1.0, 0.0]], atol=1e-12)
    assert area > 0  # upper surface first
# endregion


# region Step 3
def test_naca4_thickness():
    x = np.linspace(0.0, 1.0, 10001)

    assert _naca4_thickness(np.array([1.0]), 0.12, closed_te=True)[0] == pytest.approx(0.0, abs=1e-12)
    assert _naca4_thickness(np.array([1.0]), 0.12, closed_te=False)[0] == pytest.approx(0.00126, abs=1e-6)
    assert _naca4_thickness(x, 0.12, closed_te=True).max() == pytest.approx(0.06, abs=1e-4)
    assert x[np.argmax(_naca4_thickness(x, 0.12, closed_te=True))] == pytest.approx(0.30, abs=0.01)


@pytest.mark.parametrize("i, pos", [(6, 3), (4, 5), (2, 2), (8, 6)])
def test_mod4_thickness_conditions(i, pos):
    m, h, t = pos / 10, 1e-5, 0.12
    y = lambda x: _mod4_thickness(np.atleast_1d(x), t, i, pos, closed_te=True)[0]

    assert y(m) == pytest.approx(t / 2, abs=1e-12)                       # max thickness at the position
    assert y(m - 1e-12) == pytest.approx(t / 2, abs=1e-10)               # both pieces meet there
    assert (y(m + h) - y(m - h)) / (2 * h) == pytest.approx(0.0, abs=1e-6)  # with zero slope
    fwd = (y(m) - 2 * y(m - h) + y(m - 2 * h)) / h ** 2  # one-sided, so O(h) apart even when equal
    aft = (y(m + 2 * h) - 2 * y(m + h) + y(m)) / h ** 2
    assert fwd == pytest.approx(aft, rel=1e-2)                           # and matching curvature
    assert y(1.0) == pytest.approx(0.0, abs=1e-12)
    assert _mod4_thickness(np.array([1.0]), t, i, pos, closed_te=False)[0] == pytest.approx(0.002 * t / 0.2)


def test_mod4_63_is_close_to_four_digit():
    x = np.linspace(0.0, 1.0, 2001)
    diff = _mod4_thickness(x, 0.12, 6, 3, closed_te=False) - _naca4_thickness(x, 0.12, closed_te=False)

    assert np.abs(diff).max() < 2e-3  # 0012-63 is the modified form built to match 0012


@pytest.mark.parametrize("p, s", [(3, 0), (3, 1), (5, 0), (5, 1)])
def test_naca5_camber(p, s):
    x = np.linspace(0.0, 1.0, 20001)
    yc, _ = _naca5_camber(x, 2, p, s)
    r = {(3, 0): 0.2025, (3, 1): 0.2170, (5, 0): 0.3910, (5, 1): 0.4410}[(p, s)]
    near_r = _naca5_camber(np.array([r - 1e-9, r + 1e-9]), 2, p, s)[0]

    assert yc[0] == 0.0
    assert yc[-1] == pytest.approx(0.0, abs=1e-12)
    assert near_r[0] == pytest.approx(near_r[1], abs=1e-9)                      # continuous at r
    assert _naca5_camber(x, 4, p, s)[0] == pytest.approx(2 * yc)                 # scales with design CL
    if s == 0:
        assert x[np.argmax(yc)] == pytest.approx(0.05 * p, abs=2e-3)             # max camber at 0.05 * P


def test_a1_camber():
    x = np.array([0.0, 0.5, 1.0])
    yc, dyc = _a1_camber(x, 0.2)

    assert yc[1] == pytest.approx(0.2 * np.log(2) / (4 * np.pi), abs=1e-12)
    assert dyc[1] == pytest.approx(0.0, abs=1e-12)
    assert yc[0] == 0.0 and yc[2] == 0.0
    assert dyc[0] == np.inf and dyc[2] == -np.inf
# endregion
