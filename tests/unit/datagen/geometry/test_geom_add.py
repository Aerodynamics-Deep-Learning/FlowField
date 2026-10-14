"""
Tests for `geometry.add`: one root file through the whole per-airfoil pipeline, i.e.:

Step 1: Ensure identification is cheap and family-aware
    - test_identify_tabulated
    - test_family_id_ignores_the_point_set
Step 2: Ensure each kind of airfoil comes out normalized, closed and rendered
    - test_add_family
    - test_add_open_tabulated_is_pried
    - test_add_closed_tabulated
    - test_add_writes_nothing
Step 3: Ensure each gate rejects with its reason
    - test_reject_format
    - test_reject_family_mismatch
    - test_reject_too_few_points
    - test_reject_surfaces_cross
    - test_reject_fit_tolerance
"""

import numpy as np
import pytest

from src.datagen.geometry.build.add import Geom_add_airfoil, Geom_identify
from src.datagen.geometry.common.constants import CST_FIT_TOL, CST_ORDER, FAMILY_MATCH_TOL, NORM_N_POINTS
from src.datagen.geometry.build.families import Geom_family_id, _family_contour
from src.datagen.geometry.common.io import Geom_coords_hash, Geom_write_dat
from src.datagen.geometry.common.schemas import Closure, Family, RejectReason


def _naca(m, p, t, closed_te=False, n=201):
    return _family_contour(Family.NACA4, {"m": m, "p": p, "t": t}, n, closed_te)


def _dat(folder, name, coords):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.dat"
    Geom_write_dat(path, name, coords)
    return path


def _from_thickness(t):
    """A symmetric contour from a half-thickness function, cosine-spaced."""
    x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 201)))
    y = t(x) / 2
    return np.vstack([np.stack([x, y], 1)[::-1], np.stack([x, -y], 1)[1:]])


# region Step 1
def test_identify_tabulated(tmp_path):
    path = _dat(tmp_path, "testfoil", _naca(0, 0, 12))
    ident = Geom_identify(path)

    assert ident.family is None and ident.family_params is None
    assert ident.geometry_id == ident.raw_hash
    assert ident.raw_points == 401


def test_family_id_ignores_the_point_set(tmp_path):
    dense = Geom_identify(_dat(tmp_path / "a", "naca2412", _naca(2, 4, 12, n=201)))
    sparse = Geom_identify(_dat(tmp_path / "b", "naca2412", _naca(2, 4, 12, n=41)))

    assert dense.family is Family.NACA4
    assert dense.geometry_id == sparse.geometry_id == Geom_family_id(Family.NACA4, {"m": 2, "p": 4, "t": 12})
    assert dense.raw_hash != sparse.raw_hash
# endregion


# region Step 2
def test_add_family(tmp_path):
    res = Geom_add_airfoil(_dat(tmp_path, "naca2412", _naca(2, 4, 12)))

    assert res.accepted, res.reason
    assert res.closure is Closure.FORMULA
    assert res.root_te_gap == pytest.approx(0.00252, abs=2e-5)  # the file's gap, not the formula's
    assert res.family_dev < 1e-3
    assert res.cst.order == CST_ORDER and res.fit.max_dev < CST_FIT_TOL
    assert res.cst.lower[0] == -res.cst.upper[0]  # one nose radius
    assert res.source_dev == pytest.approx(res.root_te_gap / 2, rel=0.05)  # the open file's TE, against the closed shape
    assert res.te_wedge_deg == pytest.approx(16.5, abs=0.5)
    assert res.norm_coords.shape == (NORM_N_POINTS, 2)
    assert res.norm_coords[0].tolist() == [1.0, 0.0] and res.norm_coords[-1].tolist() == [1.0, 0.0]
    assert res.norm_coords[NORM_N_POINTS // 2].tolist() == [0.0, 0.0]
    assert res.norm_hash == Geom_coords_hash(res.norm_coords)


def test_add_open_tabulated_is_pried(tmp_path):
    res = Geom_add_airfoil(_dat(tmp_path, "testfoil", _naca(0, 0, 12)))

    assert res.accepted, res.reason
    assert res.identity.family is None
    assert res.closure is Closure.PRIED
    assert res.root_te_gap == pytest.approx(0.00252, abs=2e-5)
    assert res.source_dev == pytest.approx(res.root_te_gap / 2, rel=0.05)  # closing moved its TE by half the gap
    assert res.fit.max_dev < res.source_dev


def test_add_closed_tabulated(tmp_path):
    coords = _naca(2, 4, 12, closed_te=True)
    coords[-1] = coords[0]
    res = Geom_add_airfoil(_dat(tmp_path, "closedfoil", coords))

    assert res.accepted, res.reason
    assert res.closure is Closure.NONE and res.root_te_gap == 0.0
    assert res.source_dev == res.fit.max_dev  # nothing was closed: the source is what was fitted


def test_add_writes_nothing(tmp_path):
    path = _dat(tmp_path, "naca2412", _naca(2, 4, 12))
    before = sorted(tmp_path.rglob("*"))
    Geom_add_airfoil(path)

    assert sorted(tmp_path.rglob("*")) == before
# endregion


# region Step 3
def test_reject_format(tmp_path):
    path = tmp_path / "broken.dat"
    path.write_text("broken\n1.0 0.0\n")
    res = Geom_add_airfoil(path)

    assert res.reject is RejectReason.FORMAT and res.identity is None and "points" in res.reason


def test_reject_family_mismatch(tmp_path):
    res = Geom_add_airfoil(_dat(tmp_path, "naca0021", _naca(0, 0, 12)))

    assert res.reject is RejectReason.FAMILY_MISMATCH
    assert res.family_dev > FAMILY_MATCH_TOL


def test_reject_too_few_points(tmp_path):
    res = Geom_add_airfoil(_dat(tmp_path, "coarsefoil", _naca(0, 0, 12, n=7)))

    assert res.reject is RejectReason.TOO_FEW_POINTS
    assert f"order {CST_ORDER}" in res.reason


def test_reject_surfaces_cross(tmp_path):
    # Thin aft body ending in a 3% gap: closing it linearly pushes the surfaces through each other
    coords = _from_thickness(lambda x: 0.24 * (np.sqrt(x) - x) + 0.03 * x ** 30)
    res = Geom_add_airfoil(_dat(tmp_path, "crossfoil", coords))

    assert res.reject is RejectReason.SURFACES_CROSS
    assert res.root_te_gap == pytest.approx(0.03, abs=1e-3)


def test_reject_fit_tolerance(tmp_path):
    coords = _naca(0, 0, 12, closed_te=True)
    upper = slice(0, 201)
    coords[upper, 1] += 0.01 * np.exp(-((coords[upper, 0] - 0.5) / 0.02) ** 2)  # a bump no order-8 CST follows
    res = Geom_add_airfoil(_dat(tmp_path, "bumpfoil", coords))

    assert res.reject is RejectReason.FIT_TOLERANCE
    assert res.fit.max_dev > CST_FIT_TOL
# endregion
