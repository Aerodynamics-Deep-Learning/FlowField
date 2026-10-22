"""
Tests for `geometry.parametrization`: CST evaluation, fit, contour and distances, i.e.:

Step 1: Ensure the basis and evaluation
    - test_bernstein_partition_of_unity
    - test_eval_passes_through_le_and_te
    - test_le_term_changes_camber_only
    - test_te_wedge
    - test_slope
Step 2: Ensure the fit recovers the CST it was given, never crosses the TE, and why the nose is fitted as it is
    - test_fit_recovers_coefficients
    - test_fit_meets_a_crossed_te_as_a_cusp
    - test_lem_fits_a_cambered_nose_with_one_radius
Step 3: Ensure the contour and the distances measured against it
    - test_contour_is_closed_selig
    - test_curve_distance
    - test_fit_metrics
"""

import math

import numpy as np
import pytest

from src.datagen.geometry.build.families import Geom_family_coords
from src.datagen.geometry.build.normalize import Geom_close_te, Geom_normalize
from src.datagen.geometry.common.constants import CST_N1, CST_N2
from src.datagen.geometry.build.parametrization import (
    Geom_cst_contour, Geom_cst_eval, Geom_cst_fit, Geom_cst_fit_metrics, Geom_cst_slope, Geom_cst_te_wedge,
    Geom_curve_distance, _bernstein,
)
from src.datagen.geometry.common.schemas import CSTParams, Family, Nose

_UPPER = [0.17, 0.15, 0.20, 0.18, 0.16, 0.20, 0.19]


def _cst(nose: Nose = Nose.SEPARATE) -> CSTParams:
    lower = [-0.12 if nose is Nose.SEPARATE else -0.17, -0.10, -0.08, -0.05, -0.06, -0.02, 0.01]
    return CSTParams(n1=CST_N1, n2=CST_N2, upper=_UPPER, lower=lower, le_weight=0.05 if nose is Nose.LEM else 0.0)


# region Step 1
def test_bernstein_partition_of_unity():
    np.testing.assert_allclose(_bernstein(np.linspace(0, 1, 11), 8).sum(axis=1), 1.0, atol=1e-14)


@pytest.mark.parametrize("nose", list(Nose))
def test_eval_passes_through_le_and_te(nose):
    yu, yl = Geom_cst_eval(_cst(nose), np.array([0.0, 1.0]))

    assert yu.tolist() == [0.0, 0.0] and yl.tolist() == [0.0, 0.0]


def test_le_term_changes_camber_only():
    x = np.linspace(0.0, 1.0, 201)
    plain, lem = _cst(Nose.SHARED), _cst(Nose.LEM)
    (pu, pl), (lu, ll) = Geom_cst_eval(plain, x), Geom_cst_eval(lem, x)

    np.testing.assert_allclose(lu - ll, pu - pl, atol=1e-15)  # thickness unchanged
    assert np.abs(lu - pu).max() > 1e-3  # the camber did change
    assert Geom_cst_te_wedge(lem) == Geom_cst_te_wedge(plain)


def test_te_wedge():
    cst, h = _cst(Nose.LEM), 1e-7
    yu, yl = Geom_cst_eval(cst, np.array([1.0 - h]))  # both surfaces end at y = 0, so y / h is minus each slope

    assert Geom_cst_te_wedge(cst) == pytest.approx(math.degrees(math.atan(yu[0] / h) - math.atan(yl[0] / h)), abs=1e-4)
    assert Geom_cst_te_wedge(CSTParams(n1=CST_N1, n2=CST_N2, upper=[0.1, -0.1], lower=[-0.1, 0.1])) < 0  # crossed


@pytest.mark.parametrize("nose", list(Nose))
def test_slope(nose):
    cst, h = _cst(nose), 1e-6
    x = np.linspace(0.02, 0.98, 25)
    (su, sl), (pu, pl), (mu, ml) = Geom_cst_slope(cst, x), Geom_cst_eval(cst, x + h), Geom_cst_eval(cst, x - h)
    (su_te,), (sl_te,) = Geom_cst_slope(cst, np.array([1.0]))

    np.testing.assert_allclose(su, (pu - mu) / (2 * h), rtol=0, atol=1e-8)
    np.testing.assert_allclose(sl, (pl - ml) / (2 * h), rtol=0, atol=1e-8)
    assert (su_te, sl_te) == (-cst.upper[-1], -cst.lower[-1])  # each surface ends as ~A_N (1 - x)
# endregion


# region Step 2
@pytest.mark.parametrize("nose", list(Nose))
def test_fit_recovers_coefficients(nose):
    truth = _cst(nose)
    pts = Geom_cst_contour(truth, 301)
    fit = Geom_cst_fit(pts, 150, order=truth.order, nose=nose)

    np.testing.assert_allclose(fit.upper, truth.upper, atol=1e-9)
    np.testing.assert_allclose(fit.lower, truth.lower, atol=1e-9)
    assert fit.le_weight == pytest.approx(truth.le_weight, abs=1e-9)
    assert (fit.n1, fit.n2) == (CST_N1, CST_N2)


@pytest.mark.parametrize("nose", list(Nose))
def test_fit_meets_a_crossed_te_as_a_cusp(nose):
    truth = _cst(nose)
    crossed = truth.model_copy(update={"lower": truth.lower[:-1] + [0.25]})  # lower A_N above the upper's 0.19
    pts = Geom_cst_contour(crossed, 301)
    fit = Geom_cst_fit(pts, 150, order=crossed.order, nose=nose)
    tied = crossed.model_copy(update={"lower": crossed.lower[:-1] + [crossed.upper[-1]]})  # a cusp, by hand

    assert Geom_cst_te_wedge(crossed) < 0
    assert fit.lower[-1] == fit.upper[-1] and Geom_cst_te_wedge(fit) == 0.0
    assert Geom_cst_fit_metrics(fit, pts, 150).rms_dev < Geom_cst_fit_metrics(tied, pts, 150).rms_dev  # the best cusp


def test_lem_fits_a_cambered_nose_with_one_radius():
    # A cambered nose leans, which one A_0 alone cannot follow and separate ones fake with two radii
    pts, tf = Geom_normalize(Geom_family_coords(Family.NACA4, {"m": 6, "p": 4, "t": 12}))
    pts, _ = Geom_close_te(pts, tf.le_index)
    dev = {nose: Geom_cst_fit_metrics(Geom_cst_fit(pts, tf.le_index, 8, nose), pts, tf.le_index).max_dev
           for nose in Nose}
    separate, lem = (Geom_cst_fit(pts, tf.le_index, 8, n) for n in (Nose.SEPARATE, Nose.LEM))

    assert lem.lower[0] == -lem.upper[0]
    assert (separate.upper[0] / separate.lower[0]) ** 2 == pytest.approx(1.36, abs=0.02)  # nose radius ratio
    assert dev[Nose.LEM] < dev[Nose.SEPARATE] < dev[Nose.SHARED]
    assert dev[Nose.LEM] < 0.25 * dev[Nose.SHARED]
# endregion


# region Step 3
def test_contour_is_closed_selig():
    c = Geom_cst_contour(_cst(), 101)

    assert c.shape == (101, 2)
    assert c[0].tolist() == [1.0, 0.0] and c[-1].tolist() == [1.0, 0.0]
    assert c[50].tolist() == [0.0, 0.0]
    assert (c[1:50, 1] > c[51:-1, 1][::-1]).all()  # upper surface first, above the lower at each x
    with pytest.raises(ValueError):
        Geom_cst_contour(_cst(), 100)


def test_curve_distance():
    polyline = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    points = np.array([[0.3, 0.0], [0.5, 0.2], [2.0, 0.5], [1.5, 1.5], [-1.0, 0.0]])

    np.testing.assert_allclose(Geom_curve_distance(points, polyline, chunk=2),
                               [0.0, 0.2, 1.0, np.hypot(0.5, 0.5), 1.0], atol=1e-15)


def test_fit_metrics():
    truth = _cst()
    pts = Geom_cst_contour(truth, 201)
    exact = Geom_cst_fit_metrics(truth, pts, 100)
    bumped = pts.copy()
    bumped[50, 1] += 1e-3
    off = Geom_cst_fit_metrics(truth, bumped, 100)

    assert exact.max_dev < 1e-6 and exact.n_points == 201
    assert off.max_dev == pytest.approx(1e-3, rel=0.1)
    assert off.x_at_max == bumped[50, 0]
    assert off.isolation > 100 and off.zigzag  # a mistyped point: alone, and poking out of the curve
    assert not exact.zigzag
# endregion
