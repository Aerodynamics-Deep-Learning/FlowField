"""
Tests for `geometry.sample.contour`: a stored shape given the TE its mesher needs, i.e.:

Step 1: Ensure a round TE meets the weld conditions exactly
    - test_round_caps_continue_the_surfaces
    - test_round_tip_is_mid_gap_with_a_vertical_tangent
    - test_round_contour_spans_the_unit_chord
Step 2: Ensure the sharp and blunt TEs
    - test_sharp_is_the_stored_shape
    - test_blunt_is_opened_and_left_open
Step 3: Ensure the tables, the caps' spacing limit, and the refusals
    - test_arc_length_and_pins
    - test_curvature_table
    - test_te_cap_limit
    - test_refused
    - test_contour_pickles
"""

import math
import pickle
from functools import cache

import numpy as np
import pytest

from src.datagen.geometry.build.families import Geom_family_coords
from src.datagen.geometry.build.normalize import Geom_close_te, Geom_normalize
from src.datagen.geometry.build.parametrization import Geom_cst_eval, Geom_cst_fit, Geom_cst_slope, Geom_cst_te_wedge
from src.datagen.geometry.common.constants import CST_N1, CST_N2, SHAPE_VERSION, THETA_MAX_DEG
from src.datagen.geometry.common.schemas import Closure, CSTParams, Family, ShapeRecord, TETreatment
from src.datagen.geometry.sample.contour import ContourError, Geom_build_contour, PinKind

G = 0.004
_SHAPES = ["naca0012", "naca4412", "cusp"]


def _record(name: str, cst: CSTParams) -> ShapeRecord:
    return ShapeRecord(geometry_id=name, shape_version=SHAPE_VERSION, cst=cst, rotation_deg=0.0,
                       closure=Closure.FORMULA, root_te_gap=0.0, source_dev=0.0, fit_max=0.0,
                       te_wedge_deg=Geom_cst_te_wedge(cst), norm_path=f"normalised/{name}.dat", norm_hash="0" * 64)


@cache
def _shape(name: str) -> ShapeRecord:
    """Two NACA formula shapes through the ingest's own normalize/close/fit, and a cusp (wedge 0)."""
    if name == "cusp":
        return _record(name, CSTParams(n1=CST_N1, n2=CST_N2, le_weight=0.02,
                                       upper=[0.17, 0.15, 0.20, 0.18, 0.16, 0.20, 0.19, 0.12, 0.05],
                                       lower=[-0.17, -0.10, -0.08, -0.05, -0.06, -0.02, 0.01, 0.03, 0.05]))
    digits = {"naca0012": {"m": 0, "p": 0, "t": 12}, "naca4412": {"m": 4, "p": 4, "t": 12}}[name]
    pts, tf = Geom_normalize(Geom_family_coords(Family.NACA4, digits))
    pts, _ = Geom_close_te(pts, tf.le_index)
    return _record(name, Geom_cst_fit(pts, tf.le_index))


def _pins(contour) -> dict:
    return {p.kind: p for p in contour.pins}


def _bezier_curvature(cap: np.ndarray, t: np.ndarray) -> np.ndarray:
    a, b = cap[1] - cap[0], cap[2] - 2 * cap[1] + cap[0]
    d1 = 2 * (a + t[:, None] * b)
    return np.abs(d1[:, 0] * 2 * b[1] - d1[:, 1] * 2 * b[0]) / np.hypot(d1[:, 0], d1[:, 1]) ** 3


# region Step 1
@pytest.mark.parametrize("name", _SHAPES)
def test_round_caps_continue_the_surfaces(name):
    shape = _shape(name)
    c = Geom_build_contour(shape, TETreatment.ROUND, G)
    x_w = np.array([1 - G / 2])
    (yu,), (yl,) = Geom_cst_eval(shape.cst, x_w)
    (su,), (sl,) = Geom_cst_slope(shape.cst, x_w)
    pins = _pins(c)

    assert c.weld_x == x_w[0]
    for cap, pin, y, slope in ((c.caps[0], pins[PinKind.WELD_UPPER], yu + x_w[0] * G / 2, su + G / 2),
                               (c.caps[1], pins[PinKind.WELD_LOWER], yl - x_w[0] * G / 2, sl - G / 2)):
        assert pin.xy == tuple(cap[0]) == (x_w[0], y)  # 0: the cap starts exactly at the weld, on the surface
        d = cap[1] - cap[0]
        assert d[1] / d[0] == pytest.approx(slope, rel=1e-12)  # 1: and leaves it along the surface's slope
        before, w, after = c.evaluate(np.array([pin.s - 1e-6, pin.s, pin.s + 1e-6]))
        np.testing.assert_allclose([before, after], [w, w], atol=1.1e-6)  # the pieces meet...
        u, v = w - before, after - w
        assert abs(u[0] * v[1] - u[1] * v[0]) / (np.hypot(*u) * np.hypot(*v)) < 1e-3  # ...without a kink


@pytest.mark.parametrize("name", _SHAPES)
def test_round_tip_is_mid_gap_with_a_vertical_tangent(name):
    c = Geom_build_contour(_shape(name), TETreatment.ROUND, G)
    near = c.evaluate(np.array([1e-7, c.length - 1e-7]))

    for cap in c.caps:
        assert cap[2].tolist() == [1.0, 0.0]  # 2: the tip, mid-gap
        assert cap[1][0] == 1.0               # 3: P1 straight above or below it, so B'(1) = 2 (P2 - P1) is vertical
    assert c.closed and c.pins[0].xy == c.pins[-1].xy == (1.0, 0.0)
    assert c.evaluate(np.array([0.0, c.length])).tolist() == [[1.0, 0.0], [1.0, 0.0]]
    assert near[0, 1] > 0 > near[1, 1]  # leaves upward, returns from below
    assert np.all(np.abs(1.0 - near[:, 0]) < 1e-3 * np.abs(near[:, 1]))  # moving vertically on both sides


@pytest.mark.parametrize("name", _SHAPES)
def test_round_contour_spans_the_unit_chord(name):
    shape = _shape(name)
    c = Geom_build_contour(shape, TETreatment.ROUND, G)
    pins = _pins(c)
    s = np.union1d(np.linspace(0.0, c.length, 20001), [p.s for p in c.pins])
    pts = c.evaluate(s)
    upper = (s > pins[PinKind.WELD_UPPER].s) & (s < pins[PinKind.LE].s)
    lower = (s > pins[PinKind.LE].s) & (s < pins[PinKind.WELD_LOWER].s)
    yu, _ = Geom_cst_eval(shape.cst, pts[upper, 0])
    _, yl = Geom_cst_eval(shape.cst, pts[lower, 0])
    t = np.linspace(0.0, 1.0, 2001)[:, None]

    assert pts[:, 0].min() == 0.0 and pts[:, 0].max() == 1.0  # LE at x = 0, tip at x = 1: chord exactly 1
    np.testing.assert_allclose(pts[upper, 1], yu + pts[upper, 0] * G / 2, rtol=0, atol=1e-15)  # opened by x * g/2
    np.testing.assert_allclose(pts[lower, 1], yl - pts[lower, 0] * G / 2, rtol=0, atol=1e-15)
    for cap, side in zip(c.caps, (1.0, -1.0)):
        curve = (1 - t) ** 2 * cap[0] + 2 * t * (1 - t) * cap[1] + t ** 2 * cap[2]
        assert np.all(side * curve[:-1, 1] > 0) and np.all(np.diff(curve[:, 0]) >= 0)  # own side, x rising to 1
# endregion


# region Step 2
@pytest.mark.parametrize("name", _SHAPES)
def test_sharp_is_the_stored_shape(name):
    shape = _shape(name)
    c = Geom_build_contour(shape, TETreatment.SHARP, gap=G)  # a sharp TE takes no gap
    s = np.linspace(0.0, c.length, 4001)
    pts = c.evaluate(s)
    upper, lower = s <= c.pins[1].s, s >= c.pins[1].s
    yu, _ = Geom_cst_eval(shape.cst, pts[upper, 0])
    _, yl = Geom_cst_eval(shape.cst, pts[lower, 0])

    assert c.gap is None and c.closed and c.caps is None and c.weld_x is None and c.te_cap_limit is None
    assert [p.kind for p in c.pins] == [PinKind.TIP, PinKind.LE, PinKind.TIP]
    assert [p.xy for p in c.pins] == [(1.0, 0.0), (0.0, 0.0), (1.0, 0.0)]
    np.testing.assert_allclose(pts[upper, 1], yu, rtol=0, atol=1e-15)
    np.testing.assert_allclose(pts[lower, 1], yl, rtol=0, atol=1e-15)


@pytest.mark.parametrize("name", _SHAPES)
def test_blunt_is_opened_and_left_open(name):
    shape = _shape(name)
    c = Geom_build_contour(shape, TETreatment.BLUNT, G)
    s = np.linspace(0.0, c.length, 4001)
    pts = c.evaluate(s)
    upper, lower = s <= c.pins[1].s, s >= c.pins[1].s
    yu, _ = Geom_cst_eval(shape.cst, pts[upper, 0])
    _, yl = Geom_cst_eval(shape.cst, pts[lower, 0])

    assert not c.closed and c.gap == G and c.caps is None and c.te_cap_limit is None
    assert [p.kind for p in c.pins] == [PinKind.END_UPPER, PinKind.LE, PinKind.END_LOWER]
    assert [p.xy for p in c.pins] == [(1.0, G / 2), (0.0, 0.0), (1.0, -G / 2)]
    assert c.evaluate(np.array([0.0, c.length])).tolist() == [[1.0, G / 2], [1.0, -G / 2]]
    np.testing.assert_allclose(pts[upper, 1] - yu, pts[upper, 0] * G / 2, rtol=0, atol=1e-16)  # opened by x * g/2
    np.testing.assert_allclose(pts[lower, 1] - yl, -pts[lower, 0] * G / 2, rtol=0, atol=1e-16)
# endregion


# region Step 3
@pytest.mark.parametrize("name", _SHAPES)
def test_arc_length_and_pins(name):
    c = Geom_build_contour(_shape(name), TETreatment.ROUND, G)
    s = [p.s for p in c.pins]
    dense = c.evaluate(np.linspace(0.0, c.length, 200001))

    assert [p.kind for p in c.pins] == [PinKind.TIP, PinKind.WELD_UPPER, PinKind.LE, PinKind.WELD_LOWER, PinKind.TIP]
    assert s == sorted(s) and s[0] == 0.0 and s[-1] == c.length
    assert [tuple(p) for p in c.evaluate(np.array(s)).tolist()] == [p.xy for p in c.pins]  # pins exactly
    assert np.hypot(*np.diff(dense, axis=0).T).sum() == pytest.approx(c.length, rel=1e-6)
    assert np.all(np.diff(c.table_s) > 0) and c.table_s.shape == c.table_kappa.shape
    assert c.table_s[0] > 0.0 and c.table_s[-1] < c.length
    with pytest.raises(ValueError):
        c.evaluate(np.array([1.01 * c.length]))


def test_curvature_table():
    shape = _shape("naca0012")
    c = Geom_build_contour(shape, TETreatment.ROUND, G)
    pins = _pins(c)
    on_upper_cap = c.table_s < pins[PinKind.WELD_UPPER].s
    peak = _bezier_curvature(c.caps[0], np.linspace(0.0, 1.0, 200001)).max()
    nose = c.table_kappa[np.argmin(np.abs(c.table_s - pins[PinKind.LE].s))]

    assert c.table_kappa[on_upper_cap].max() == pytest.approx(peak, rel=1e-3)
    assert nose == pytest.approx(2 / shape.cst.upper[0] ** 2, rel=1e-2)  # nose radius A_0^2 / 2
    assert c.table_kappa[(c.table_s > 0.3) & (c.table_s < 0.7)].max() < 5.0  # mid-chord: nearly flat


@pytest.mark.parametrize("name", _SHAPES)
def test_te_cap_limit(name):
    c = Geom_build_contour(_shape(name), TETreatment.ROUND, G)
    t = np.linspace(0.0, 1.0, 200001)
    peak = max(_bezier_curvature(cap, t).max() for cap in c.caps)
    limit = 2 * math.sin(math.radians(THETA_MAX_DEG) / 2) / peak

    assert c.te_cap_limit == pytest.approx(limit, rel=1e-6)
    assert 1e-4 < c.te_cap_limit < 1e-3  # ~2.9/g peak at g = 0.4%: a few 1e-4 c
    # a chord of that length on a circle of the peak curvature turns exactly the limit
    assert 2 * math.asin(c.te_cap_limit * peak / 2) == pytest.approx(math.radians(THETA_MAX_DEG), rel=1e-8)


def test_refused():
    shape = _shape("naca0012")
    drooped = _record("drooped", CSTParams(n1=CST_N1, n2=CST_N2, upper=[0.17, 0.2, 0.2, -2.0],
                                           lower=[-0.17, -0.1, -0.1, -2.5]))  # TE dives ~63 deg below the chord

    for te in (TETreatment.BLUNT, TETreatment.ROUND):
        for gap in (None, 0.0, -1e-3):
            with pytest.raises(ContourError, match="positive gap"):
                Geom_build_contour(shape, te, gap)
    with pytest.raises(ContourError, match="upper cap"):
        Geom_build_contour(drooped, TETreatment.ROUND, G)
    assert Geom_build_contour(drooped, TETreatment.SHARP).closed  # a sharp TE needs no cap


def test_contour_pickles():
    c = Geom_build_contour(_shape("naca4412"), TETreatment.ROUND, G)
    again = pickle.loads(pickle.dumps(c))
    s = np.linspace(0.0, c.length, 101)

    assert np.array_equal(again.evaluate(s), c.evaluate(s)) and again.pins == c.pins
    assert again.te_cap_limit == c.te_cap_limit
# endregion
