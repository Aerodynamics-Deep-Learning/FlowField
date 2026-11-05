"""
Tests for `geometry.sample.checks`: a placed sample's measures and the rules its construction guarantees, i.e.:

Step 1: Ensure the measures
    - test_dev_max_is_a_circles_sagitta
    - test_measures_match_a_direct_reading
    - test_closure_turn_skipped_only_for_a_sharp_te
Step 2: Ensure a broken sample raises
    - test_broken_sample_raises
"""

import math

import numpy as np
import pytest

from src.datagen.geometry.build.parametrization import Geom_curve_distance
from src.datagen.geometry.common.constants import DX_MAX, THETA_MAX_DEG
from src.datagen.geometry.common.schemas import TETreatment
from src.datagen.geometry.sample.checks import DEV_MAX, Geom_sample_checks, SamplerAssertionError
from src.datagen.geometry.sample.contour import Geom_build_contour
from src.datagen.geometry.sample.fields import SamplingFields
from src.datagen.geometry.sample.place import Geom_stations, Geom_stretch_panels, Geom_target_spacing
from tests.geometry_shapes import stored_shape

FIELDS = SamplingFields()


def _placed(name: str, te: TETreatment):
    c = Geom_build_contour(stored_shape(name), te, 0.004)
    sp = Geom_target_spacing(c, FIELDS)
    s = Geom_stations(c, sp, Geom_stretch_panels(c, sp))
    return c, s, c.evaluate(s)


def _turn_deg(u: np.ndarray, v: np.ndarray) -> float:
    return math.degrees(abs(math.atan2(u[0] * v[1] - u[1] * v[0], u @ v)))


# region Step 1
def test_dev_max_is_a_circles_sagitta():
    radius = DX_MAX / (2 * math.sin(math.radians(THETA_MAX_DEG) / 2))  # chords of DX_MAX turn THETA_MAX_DEG on it

    assert DEV_MAX == pytest.approx(radius * (1 - math.cos(math.radians(THETA_MAX_DEG) / 2)), rel=1e-12)
    assert DEV_MAX == pytest.approx(1.966e-4, abs=1e-7)


@pytest.mark.parametrize("name", ["naca0012", "naca4412"])
def test_measures_match_a_direct_reading(name):
    c, s, pts = _placed(name, TETreatment.ROUND)
    m = Geom_sample_checks(c, s, pts, FIELDS.dx_mid)
    panel = np.hypot(*np.diff(pts, axis=0).T)
    dense = c.evaluate(np.union1d(np.linspace(0.0, c.length, 30001), s))
    turns = [_turn_deg(pts[k] - pts[k - 1], pts[k + 1] - pts[k]) for k in range(1, len(pts) - 1)]

    assert (m["max_panel"], m["min_panel"]) == (panel.max(), panel.min())
    assert m["max_turn_deg"] == pytest.approx(max(turns + [_turn_deg(pts[-1] - pts[-2], pts[1] - pts[0])]), rel=1e-9)
    assert m["max_dev"] == pytest.approx(Geom_curve_distance(dense, pts).max(), rel=0.05)  # the curve to its panels
    assert m["max_panel_ratio"] == pytest.approx(max(np.max(panel[1:] / panel[:-1]), np.max(panel[:-1] / panel[1:])))


def test_closure_turn_skipped_only_for_a_sharp_te():
    c, s, pts = _placed("naca4412", TETreatment.SHARP)
    closure = _turn_deg(pts[-1] - pts[-2], pts[1] - pts[0])
    round_c, round_s, round_pts = _placed("naca4412", TETreatment.ROUND)
    round_closure = _turn_deg(round_pts[-1] - round_pts[-2], round_pts[1] - round_pts[0])

    assert closure > 150  # the sharp TE's corner, not a turn the spacing makes
    assert Geom_sample_checks(c, s, pts, FIELDS.dx_mid)["max_turn_deg"] < THETA_MAX_DEG
    assert Geom_sample_checks(round_c, round_s, round_pts, FIELDS.dx_mid)["max_turn_deg"] >= round_closure
# endregion


# region Step 2
def test_broken_sample_raises():
    c, s, pts = _placed("naca0012", TETreatment.ROUND)
    kinked = pts.copy()
    kinked[len(pts) // 4, 1] += 1e-3  # one point off the curve

    with pytest.raises(SamplerAssertionError, match="turn .*deviation"):
        Geom_sample_checks(c, s, kinked, FIELDS.dx_mid)
    with pytest.raises(SamplerAssertionError, match="panel"):
        Geom_sample_checks(c, s, pts, dx_mid=FIELDS.dx_mid / 2)  # longer panels than the config allowed
# endregion
