"""
Tests for `geometry.common.portable`, the sampler's machine-independent arithmetic, i.e.:

Step 1: Ensure each function matches numpy's to a few ULP over the ranges it serves
    - test_exp_and_log_match_numpy
    - test_atan2_matches_numpy_in_every_quadrant
    - test_sin_and_tan_match_numpy
Step 2: Ensure the edges: exact angles on the axes, underflow, refusals, scalars back as floats
    - test_edges_and_refusals

The bounds leave a ULP of slack over what was measured (exp 1, log 3, atan2 5, sin 1, tan 3), since numpy's
own functions may round a last bit differently on another machine.
"""

import numpy as np
import pytest

from src.datagen.geometry.common.portable import PI, Geom_atan2, Geom_exp, Geom_log, Geom_sin, Geom_tan

rng = np.random.default_rng(0)


def _ulps(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b) / np.spacing(np.maximum(np.abs(a), np.abs(b)))))


# region Step 1
def test_exp_and_log_match_numpy():
    x = np.concatenate([-rng.uniform(0, 40, 100_000), rng.uniform(-700, 700, 100_000)])  # the spacing's, then all
    p = np.concatenate([rng.uniform(1e-4, 10, 100_000), np.geomspace(1e-300, 1e300, 1001)])

    assert _ulps(Geom_exp(x), np.exp(x)) <= 2
    assert _ulps(Geom_log(p), np.log(p)) <= 4


def test_atan2_matches_numpy_in_every_quadrant():
    y, x = rng.normal(size=200_000), rng.normal(size=200_000)

    assert _ulps(Geom_atan2(y, x), np.arctan2(y, x)) <= 6
    assert set(np.sign(Geom_atan2(y, x)).tolist()) == {-1.0, 1.0}


def test_sin_and_tan_match_numpy():
    x = rng.uniform(-PI / 4, PI / 4, 200_000)

    assert _ulps(Geom_sin(x), np.sin(x)) <= 2
    assert _ulps(Geom_tan(x), np.tan(x)) <= 4
# endregion


# region Step 2
def test_edges_and_refusals():
    assert [Geom_atan2(y, x) for y, x in ((0.0, 0.0), (1.0, 0.0), (0.0, -1.0), (-1.0, 0.0))] == \
        [0.0, PI / 2, PI, -PI / 2]  # on the axes the reduction is exact
    assert Geom_exp(-800.0) == 0.0 and Geom_exp(0.0) == 1.0 and Geom_log(1.0) == 0.0
    assert all(isinstance(f(0.5), float) for f in (Geom_exp, Geom_log, Geom_sin, Geom_tan))
    with pytest.raises(ValueError, match="positive"):
        Geom_log(np.array([1.0, 0.0]))
    for f in (Geom_sin, Geom_tan):
        with pytest.raises(ValueError, match="pi/4"):
            f(0.8)
# endregion
