"""
Arithmetic that gives the same bits on every machine, for everything that places sample points.

numpy's transcendental functions (exp, log, arctan2, power, ...) come from the platform's math library or from
SIMD routines picked per CPU, and matrix products from OpenBLAS kernels picked per CPU; any of them may round a
last bit differently on another machine, and the sampler grows such a bit to ~1e-9 in the points, enough to
change the .dat a mesher reads and so the mesh's hash. IEEE-754 rounds + - * / and sqrt the same everywhere, and
numpy applies each as an operation of its own, so these functions, built from those alone in a fixed order, do
not depend on the machine.

    Geom_exp    e^x
    Geom_log    ln x, for x > 0
    Geom_atan2  the angle of the point (x, y), in (-pi, pi]
    Geom_sin    sin x, for |x| <= pi/4
    Geom_tan    tan x, for |x| <= pi/4

Each is within a few ULP of numpy's. The rule for the sampling path (`geometry/sample`, the CST evaluation in
`build/parametrization.py`, n*'s grid): these, + - * / and sqrt, `** 2` on arrays, and nothing else; no numpy
transcendental, no other `**`, no `@`/dot/norm, no sum whose order is not written out.
"""

import math

import numpy as np

PI = 3.141592653589793

_LN2_HI = 6.93147180369123816490e-01  # ln 2 in two parts: k * _LN2_HI is exact for |k| < 2**20
_LN2_LO = 1.90821492927058770002e-10
_INV_LN2 = 1.44269504088896338700e+00
_SQRT_HALF = 0.7071067811865476
_EXP_TERMS = [1.0 / math.factorial(j) for j in range(14)]
_ATANH_TERMS = [1.0 / (2 * k + 1) for k in range(15)]
_ATAN_TERMS = [(1.0 if k % 2 == 0 else -1.0) / (2 * k + 1) for k in range(12)]
_SIN_TERMS = [(1.0 if k % 2 == 0 else -1.0) / math.factorial(2 * k + 1) for k in range(11)]
_COS_TERMS = [(1.0 if k % 2 == 0 else -1.0) / math.factorial(2 * k) for k in range(12)]


def _horner(z: np.ndarray, terms: list) -> np.ndarray:
    """terms[0] + terms[1] z + terms[2] z^2 + ..., innermost first."""
    p = np.full(z.shape, terms[-1])
    for c in terms[-2::-1]:
        p = p * z + c
    return p


def _out(a: np.ndarray):
    return a if a.ndim else float(a)


def Geom_exp(x):
    """
    e^x: x = k ln 2 + r with |r| <= ln 2 / 2, e^r by its series to r^13, scaled by 2^k (exact).

    Args:
        x (float | np.ndarray): Exponents, at most ~709; below -745 the result is 0

    Returns:
        float | np.ndarray: e^x, as `x` is shaped
    """
    x = np.asarray(x, dtype=float)
    k = np.rint(np.clip(x, -745.0, 709.0) * _INV_LN2)
    r = (x - k * _LN2_HI) - k * _LN2_LO
    out = np.ldexp(_horner(r, _EXP_TERMS), k.astype(np.int32))
    return _out(np.where(x < -745.0, 0.0, out))


def Geom_log(x):
    """
    ln x: x = m 2^e with m in [sqrt(1/2), sqrt(2)), ln m = 2 atanh((m - 1) / (m + 1)) by its series to s^29.

    Args:
        x (float | np.ndarray): Positive, normal numbers

    Returns:
        float | np.ndarray: ln x, as `x` is shaped

    Raises:
        ValueError: On a value not above zero
    """
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0.0):
        raise ValueError("Geom_log takes positive numbers only")
    m, e = np.frexp(x)
    low = m < _SQRT_HALF
    m, e = np.where(low, m * 2.0, m), np.where(low, e - 1, e).astype(float)
    s = (m - 1.0) / (m + 1.0)
    return _out((e * _LN2_LO + 2.0 * s * _horner(s * s, _ATANH_TERMS)) + e * _LN2_HI)


def _atan01(u: np.ndarray) -> np.ndarray:
    """atan on [0, 1]: the angle halved twice (to <= pi/16), then its series to v^23."""
    w = u / (1.0 + np.sqrt(1.0 + u * u))
    v = w / (1.0 + np.sqrt(1.0 + w * w))
    return 4.0 * (v * _horner(v * v, _ATAN_TERMS))


def Geom_atan2(y, x):
    """
    The angle of the point (x, y), as `np.arctan2` gives it.

    Args:
        y (float | np.ndarray): Ordinates
        x (float | np.ndarray): Abscissas, broadcast with `y`

    Returns:
        float | np.ndarray: Angles in (-pi, pi]; 0 at the origin
    """
    y, x = np.broadcast_arrays(np.asarray(y, dtype=float), np.asarray(x, dtype=float))
    ay, ax = np.abs(y), np.abs(x)
    steep = ay > ax
    num, den = np.where(steep, ax, ay), np.where(steep, ay, ax)
    a = _atan01(np.divide(num, den, out=np.zeros_like(num), where=den > 0.0))
    a = np.where(steep, PI / 2 - a, a)
    a = np.where(x < 0.0, PI - a, a)
    return _out(np.where(y < 0.0, -a, a))


def _small_angle(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if np.any(np.abs(x) > PI / 4):
        raise ValueError("Geom_sin and Geom_tan take |x| <= pi/4 only")
    return x


def Geom_sin(x):
    """sin x by its series to x^21, for |x| <= pi/4 (ValueError beyond)."""
    x = _small_angle(x)
    return _out(x * _horner(x * x, _SIN_TERMS))


def Geom_tan(x):
    """tan x as the two series' quotient, for |x| <= pi/4 (ValueError beyond)."""
    x = _small_angle(x)
    z = x * x
    return _out(x * _horner(z, _SIN_TERMS) / _horner(z, _COS_TERMS))
