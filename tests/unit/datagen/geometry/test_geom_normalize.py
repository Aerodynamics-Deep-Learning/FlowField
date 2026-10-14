"""
Tests for `geometry.normalize`: the LE, the rigid normalization and the linear TE close, i.e.:

Step 1: Ensure the LE is found where CST needs it, on a point or between two
    - test_le_of_a_symmetric_airfoil_is_its_nose_point
    - test_le_between_points_is_inserted
Step 2: Ensure normalization is rigid and lands exactly
    - test_normalize_lands_exactly
    - test_normalize_undoes_a_similarity_transform
    - test_slanted_te_is_stretched_to_one
Step 3: Ensure the close
    - test_close_open_te
    - test_close_closed_te_is_a_noop
"""

import numpy as np
import pytest

from src.datagen.geometry.build.families import _family_contour
from src.datagen.geometry.build.normalize import Geom_close_te, Geom_find_le, Geom_normalize
from src.datagen.geometry.common.schemas import Family

_N = 201  # points per surface


def _naca(m, p, t, closed_te=True):
    return _family_contour(Family.NACA4, {"m": m, "p": p, "t": t}, _N, closed_te)


# region Step 1
def test_le_of_a_symmetric_airfoil_is_its_nose_point():
    i, le, inserted = Geom_find_le(_naca(0, 0, 12))

    assert (i, inserted) == (_N - 1, False)
    np.testing.assert_array_equal(le, [0.0, 0.0])


def test_le_between_points_is_inserted():
    coords = np.delete(_naca(0, 0, 12), _N - 1, axis=0)  # drop the nose point
    i, le, inserted = Geom_find_le(coords)

    assert inserted and i == _N - 1
    np.testing.assert_allclose(le, [0.0, 0.0], atol=1e-6)
# endregion


# region Step 2
def test_normalize_lands_exactly():
    q, tf = Geom_normalize(_naca(4, 4, 12))

    assert q[tf.le_index].tolist() == [0.0, 0.0]
    assert q[0, 0] == 1.0 and q[-1, 0] == 1.0
    assert tf.le_inserted  # a cambered NACA's nose point is not its geometric LE


def test_normalize_undoes_a_similarity_transform():
    base = _naca(4, 4, 12)
    a = np.radians(7.0)
    moved = 2.5 * base @ np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]]) + [3.0, -1.0]
    q0, tf0 = Geom_normalize(base)
    q1, tf1 = Geom_normalize(moved)

    assert int(np.argmin(moved[:, 0])) != tf1.le_index  # the raw leftmost point is not the LE
    assert (tf1.le_index, tf1.le_inserted) == (tf0.le_index, tf0.le_inserted)
    np.testing.assert_allclose(q1, q0, atol=1e-12)
    assert tf1.scale == pytest.approx(tf0.scale / 2.5)
    assert tf1.rotation_deg == pytest.approx(tf0.rotation_deg - 7.0)


def test_slanted_te_is_stretched_to_one():
    coords = _naca(0, 0, 12, closed_te=False)
    coords[-1, 0] -= 0.003
    q, tf = Geom_normalize(coords)

    assert q[0, 0] == 1.0 and q[-1, 0] == 1.0
    # The TE midpoint moved by half the shift, so each surface stretches by about half of it
    assert tf.stretch_upper == pytest.approx(1 - 0.0015, abs=2e-4)
    assert tf.stretch_lower == pytest.approx(1 + 0.0015, abs=2e-4)
# endregion


# region Step 3
def test_close_open_te():
    q, tf = Geom_normalize(_naca(0, 0, 12, closed_te=False))
    closed, gap = Geom_close_te(q, tf.le_index)

    assert closed[0].tolist() == [1.0, 0.0] and closed[-1].tolist() == [1.0, 0.0]
    assert gap == pytest.approx(0.00252, abs=1e-5)
    assert closed[tf.le_index].tolist() == [0.0, 0.0]
    np.testing.assert_allclose(q[:, 1] - closed[:, 1], q[:, 0] * np.sign(q[:, 1]) * gap / 2, atol=1e-15)


def test_close_closed_te_is_a_noop():
    coords = _naca(2, 4, 12)
    coords[-1] = coords[0]
    q, tf = Geom_normalize(coords)
    closed, gap = Geom_close_te(q, tf.le_index)

    assert gap == 0.0
    np.testing.assert_allclose(closed, q, atol=1e-15)
# endregion
