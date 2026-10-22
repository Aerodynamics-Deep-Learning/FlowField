"""
Tests for `geometry.sample.place`: the spacing wanted along a contour, the turn pre-check on it, and where
the points go, i.e.:

Step 1: Ensure the spacing
    - test_spacing_hits_its_targets
    - test_te_spacing_is_clamped_to_the_caps
    - test_sharp_and_blunt_dip_at_their_ends
Step 2: Ensure the turn pre-check
    - test_turn_precheck
Step 3: Ensure the panel counts and stations
    - test_stretches_are_rounded_up
    - test_no_panel_exceeds_its_target
    - test_uniform_when_the_ratios_are_one
"""

import math

import numpy as np
import pytest

from src.datagen.geometry.common.constants import PRECHECK_MARGIN, THETA_MAX_DEG
from src.datagen.geometry.common.schemas import TETreatment
from src.datagen.geometry.sample.contour import Geom_build_contour, PinKind
from src.datagen.geometry.sample.fields import SamplingFields
from src.datagen.geometry.sample.place import (
    Geom_stations, Geom_stretch_panels, Geom_target_spacing, Geom_turn_precheck,
)
from tests.geometry_shapes import SHAPE_NAMES, stored_shape

G = 0.004
FIELDS = SamplingFields(dx_mid=0.005, le_ratio=0.1, te_ratio=0.02, cluster_width=0.06)


def _contour(name: str, te: TETreatment = TETreatment.ROUND):
    return Geom_build_contour(stored_shape(name), te, G)


def _h(spacing, s) -> np.ndarray:
    return np.interp(s, spacing.s, spacing.h)


# region Step 1
@pytest.mark.parametrize("name", SHAPE_NAMES)
def test_spacing_hits_its_targets(name):
    c = _contour(name)
    sp = Geom_target_spacing(c, FIELDS)
    pins = {p.kind: p.s for p in c.pins}
    on_caps = (sp.s <= pins[PinKind.WELD_UPPER]) | (sp.s >= pins[PinKind.WELD_LOWER])
    mid_upper = 0.5 * (pins[PinKind.WELD_UPPER] + pins[PinKind.LE])  # half a chord from both dips

    assert (sp.le, sp.te, sp.te_clamped) == (0.1 * 0.005, 0.02 * 0.005, False)
    assert _h(sp, pins[PinKind.LE]) == pytest.approx(sp.le, rel=1e-12)
    np.testing.assert_allclose(sp.h[on_caps], sp.te, rtol=1e-12)  # flat across the caps
    assert _h(sp, mid_upper) == pytest.approx(FIELDS.dx_mid, rel=1e-12)
    assert sp.h.max() <= FIELDS.dx_mid and sp.h.min() == pytest.approx(sp.te, rel=1e-12)
    assert np.all(np.diff(sp.count) > 0) and sp.count[0] == 0.0


@pytest.mark.parametrize("name", SHAPE_NAMES)
def test_te_spacing_is_clamped_to_the_caps(name):
    coarse = SamplingFields(dx_mid=0.005, te_ratio=1.0)
    round_, sharp = _contour(name), _contour(name, TETreatment.SHARP)
    sp_round, sp_sharp = Geom_target_spacing(round_, coarse), Geom_target_spacing(sharp, coarse)

    assert sp_round.te_clamped and sp_round.te == pytest.approx(round_.te_cap_limit * PRECHECK_MARGIN, rel=1e-4)
    assert Geom_turn_precheck(round_, sp_round)[0] < 0  # the clamped caps pass the pre-check
    assert not sp_sharp.te_clamped and sp_sharp.te == coarse.dx_mid  # no caps, no limit


@pytest.mark.parametrize("te", [TETreatment.SHARP, TETreatment.BLUNT])
def test_sharp_and_blunt_dip_at_their_ends(te):
    c = _contour("naca4412", te)
    sp = Geom_target_spacing(c, FIELDS)

    assert _h(sp, 0.0) == pytest.approx(sp.te, rel=1e-12) and _h(sp, c.length) == pytest.approx(sp.te, rel=1e-12)
    assert _h(sp, 0.01) > 1.01 * sp.te  # rising straight away: nothing to hold it flat over
# endregion


# region Step 2
@pytest.mark.parametrize("name", SHAPE_NAMES)
def test_turn_precheck(name):
    c = _contour(name)
    limit = 2 * math.sin(math.radians(THETA_MAX_DEG) / 2) * PRECHECK_MARGIN
    fine, blunt_nose = Geom_target_spacing(c, FIELDS), Geom_target_spacing(c, SamplingFields(le_ratio=1.0))
    excess, _ = Geom_turn_precheck(c, fine)
    nose_excess, nose_x = Geom_turn_precheck(c, blunt_nose)

    assert excess == pytest.approx(np.max(c.table_kappa * fine.h_table) / limit - 1, rel=1e-12) and excess < 0
    assert nose_excess > 0 and nose_x < 0.01  # 0.006c panels round the nose turn ~22 deg
# endregion


# region Step 3
@pytest.mark.parametrize("name", SHAPE_NAMES)
def test_stretches_are_rounded_up(name):
    c = _contour(name)
    sp = Geom_target_spacing(c, FIELDS)
    panels = Geom_stretch_panels(c, sp)
    s = Geom_stations(c, sp, panels)
    called_for = np.diff(np.interp([p.s for p in c.pins], sp.s, sp.count))
    rows = np.cumsum([0] + panels)

    assert len(panels) == len(c.pins) - 1
    assert all(n - 1e-9 <= m < n + 1 for n, m in zip(called_for, panels))
    assert len(s) == 1 + sum(panels) and np.all(np.diff(s) > 0)
    assert s[rows].tolist() == [p.s for p in c.pins]  # every pin is a station, exactly


@pytest.mark.parametrize("te", list(TETreatment))
def test_no_panel_exceeds_its_target(te):
    c = _contour("naca4412", te)
    sp = Geom_target_spacing(c, FIELDS)
    s = Geom_stations(c, sp, Geom_stretch_panels(c, sp))
    inside = np.maximum.reduceat(sp.h, np.searchsorted(sp.s, s[:-1]))  # the target's largest within each panel
    target = np.maximum(inside, np.maximum(_h(sp, s[:-1]), _h(sp, s[1:])))

    assert np.all(np.diff(s) <= target * (1 + 1e-6))
    assert np.diff(s).max() <= FIELDS.dx_mid * (1 + 1e-6)


def test_uniform_when_the_ratios_are_one():
    c = _contour("naca0012", TETreatment.SHARP)
    flat = SamplingFields(dx_mid=0.004, le_ratio=1.0, te_ratio=1.0)
    sp = Geom_target_spacing(c, flat)
    panels = Geom_stretch_panels(c, sp)
    s = Geom_stations(c, sp, panels)
    upper = np.diff(s[:panels[0] + 1])

    np.testing.assert_allclose(sp.h, flat.dx_mid, rtol=1e-12)
    np.testing.assert_allclose(upper, upper.mean(), rtol=1e-6)  # equal panels, each just under dx_mid
    assert upper.max() <= flat.dx_mid and panels[0] == math.ceil(c.pins[1].s / flat.dx_mid)
# endregion
