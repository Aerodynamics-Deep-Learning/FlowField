"""
Tests for `geometry.sample.sample`: one sampling of a contour, from a config's knobs to the points, i.e.:

Step 1: Ensure an accepted sample's points
    - test_points_are_the_contour_at_the_stations
    - test_pins_closure_and_le_idx
    - test_deterministic
    - test_report
Step 2: Ensure the rejections, and that a clamped TE spacing is not one
    - test_turn_rejection
    - test_points_rejection
    - test_clamped_te_is_accepted
Step 3: Ensure the guarantees hold across the search space
    - test_random_configs_keep_the_rules
"""

import numpy as np
import pytest

from src.datagen.geometry.build.parametrization import Geom_cst_eval
from src.datagen.geometry.common.constants import SAMPLER_VERSION, SHAPE_VERSION, THETA_MAX_DEG
from src.datagen.geometry.common.schemas import Rejection, TETreatment
from src.datagen.geometry.sample.checks import DEV_MAX
from src.datagen.geometry.sample.contour import Geom_build_contour, PinKind
from src.datagen.geometry.sample.fields import SamplingFields
from src.datagen.geometry.sample.place import Geom_stations, Geom_stretch_panels, Geom_target_spacing
from src.datagen.geometry.sample.sample import Geom_sample
from tests.geometry_shapes import SHAPE_NAMES, search_space_configs, stored_shape

G = 0.004
FIELDS = SamplingFields()


def _contour(name: str, te: TETreatment = TETreatment.ROUND):
    return Geom_build_contour(stored_shape(name), te, G)


# region Step 1
@pytest.mark.parametrize("te", list(TETreatment))
def test_points_are_the_contour_at_the_stations(te):
    c = _contour("naca4412", te)
    smp = Geom_sample(c, FIELDS)
    sp = Geom_target_spacing(c, FIELDS)
    s = Geom_stations(c, sp, Geom_stretch_panels(c, sp))
    pins = {p.kind: p.s for p in c.pins}
    upper = (s > pins.get(PinKind.WELD_UPPER, 0.0)) & (s < pins[PinKind.LE])
    yu, _ = Geom_cst_eval(stored_shape("naca4412").cst, smp.points[upper, 0])

    assert smp.accepted and np.array_equal(smp.points, c.evaluate(s))
    np.testing.assert_allclose(smp.points[upper, 1], yu + (c.gap or 0.0) / 2 * smp.points[upper, 0], rtol=0, atol=1e-15)


@pytest.mark.parametrize("name", SHAPE_NAMES)
@pytest.mark.parametrize("te", list(TETreatment))
def test_pins_closure_and_le_idx(name, te):
    c = _contour(name, te)
    smp = Geom_sample(c, FIELDS)
    rows = np.cumsum([0] + smp.report.stretch_panels)

    assert [tuple(p) for p in smp.points[rows].tolist()] == [p.xy for p in c.pins]  # every pin exactly
    assert smp.points[smp.le_idx].tolist() == [0.0, 0.0]
    assert (smp.points[0].tolist() == smp.points[-1].tolist()) == c.closed  # closed TEs repeat the first point last


def test_deterministic():
    c = _contour("cusp")

    assert np.array_equal(Geom_sample(c, FIELDS).points, Geom_sample(c, FIELDS).points)


def test_report():
    c = _contour("naca4412")
    smp = Geom_sample(c, FIELDS)
    r = smp.report

    assert (r.sampler_version, r.shape_version, r.geometry_id, r.te, r.gap) == (
        SAMPLER_VERSION, SHAPE_VERSION, "naca4412", TETreatment.ROUND, G)
    assert (r.weld_x, r.te_cap_limit) == (c.weld_x, c.te_cap_limit)
    assert (r.dx_mid, r.le_spacing, r.te_spacing, r.te_clamped) == (
        FIELDS.dx_mid, FIELDS.le_ratio * FIELDS.dx_mid, FIELDS.te_ratio * FIELDS.dx_mid, False)
    assert r.rejection is None and r.precheck_turn < 0
    assert r.n_points == r.n_required == len(smp.points) == 1 + sum(r.stretch_panels)
    assert r.le_idx == smp.le_idx
    assert r.max_panel <= FIELDS.dx_mid and r.max_turn_deg <= THETA_MAX_DEG and r.max_dev <= DEV_MAX
    assert 1.0 <= r.max_panel_ratio < 1.2 and 0 < r.min_panel < r.max_panel
    assert r.model_dump_json()  # plain numbers: it stores as JSON
# endregion


# region Step 2
def test_turn_rejection():
    smp = Geom_sample(_contour("naca0012"), SamplingFields(le_ratio=1.0), max_points=10)  # coarse nose, and too many

    assert not smp.accepted and smp.report.rejection is Rejection.TURN  # the turn is reported first
    assert smp.points is None and smp.le_idx is None and smp.report.precheck_turn > 0
    assert smp.report.x_at_precheck_turn < 0.01  # at the nose
    assert smp.report.n_required > 10 and smp.report.max_panel is None


def test_points_rejection():
    c = _contour("naca0012")
    n = Geom_sample(c, FIELDS).report.n_required

    assert Geom_sample(c, FIELDS, max_points=n - 1).report.rejection is Rejection.POINTS
    assert Geom_sample(c, FIELDS, max_points=n).accepted


@pytest.mark.parametrize("name", SHAPE_NAMES)
def test_clamped_te_is_accepted(name):
    smp = Geom_sample(_contour(name), SamplingFields(te_ratio=1.0))  # asks for 0.006c on the caps

    assert smp.accepted and smp.report.te_clamped
    assert smp.report.max_turn_deg <= THETA_MAX_DEG
# endregion


# region Step 3
@pytest.mark.parametrize("name", SHAPE_NAMES)
def test_random_configs_keep_the_rules(name):
    c = _contour(name)
    samples = [Geom_sample(c, f) for f in search_space_configs(np.random.default_rng(7), 12)]  # raises on a broken rule
    accepted = [smp for smp in samples if smp.accepted]

    assert accepted, "no config accepted"
    assert all(smp.report.rejection is Rejection.TURN for smp in samples if not smp.accepted)
    assert all(smp.report.max_turn_deg <= THETA_MAX_DEG and smp.report.max_dev <= DEV_MAX for smp in accepted)
# endregion
