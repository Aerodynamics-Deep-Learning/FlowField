"""
Rigid normalization and the linear TE close, the two steps between a contour and its CST fit.

After `Geom_normalize` the LE is exactly (0, 0) and both surfaces end exactly at x = 1. After
`Geom_close_te` both ends are exactly (1, 0).
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from src.datagen.geometry.common.schemas import Transform


def _seed_le(coords: np.ndarray) -> int:
    """The leftmost point in the frame where the line from it to the TE midpoint is horizontal, iterated."""
    te = 0.5 * (coords[0] + coords[-1])
    interior = coords[1:-1]
    i = int(np.argmin(interior[:, 0])) + 1
    seen = []
    while i not in seen:
        seen.append(i)
        j = int(np.argmin((interior - coords[i]) @ (te - coords[i]))) + 1
        if j == i:
            return i
        i = j
    cycle = seen[seen.index(i):]
    return max(cycle, key=lambda k: float(np.linalg.norm(te - coords[k])))


def Geom_find_le(coords: np.ndarray) -> tuple[int, np.ndarray, bool]:
    """
    The LE: where the tangent is perpendicular to the line to the TE midpoint (XFoil's definition), the
    leftmost point in the frame where that line is horizontal, so CST's vertical nose tangent sits there.
    Found on a cubic spline through the points: tabulated files often put a point on the mean line's LE
    instead, up to ~1e-2c from this one.

    Args:
        coords (np.ndarray): (n, 2) Selig-ordered coordinates

    Returns:
        tuple: (index, LE point, inserted). Not inserted: the LE is coords[index]. Inserted: it lies
            between coords[index - 1] and coords[index]
    """
    i = _seed_le(coords)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(coords, axis=0).T))])
    sx, sy = CubicSpline(s, coords[:, 0]), CubicSpline(s, coords[:, 1])
    te = 0.5 * (coords[0] + coords[-1])

    def f(t):  # positive on the upper surface, negative on the lower
        return (sx(t) - te[0]) * sx(t, 1) + (sy(t) - te[1]) * sy(t, 1)

    lo, hi = i - 1, i + 1
    while f(s[lo]) * f(s[hi]) > 0:
        if lo == 1 and hi == len(s) - 2:
            return i, coords[i].copy(), False  # no sign change near the seed: keep the seed
        lo, hi = max(lo - 1, 1), min(hi + 1, len(s) - 2)
    t = brentq(f, s[lo], s[hi], xtol=1e-14 * s[-1])

    k = int(np.argmin(np.abs(s - t)))
    if abs(s[k] - t) <= 1e-12 * s[-1]:
        return k, coords[k].copy(), False
    return int(np.searchsorted(s, t)), np.array([float(sx(t)), float(sy(t))]), True


def Geom_normalize(coords: np.ndarray) -> tuple[np.ndarray, Transform]:
    """
    Moves the LE to (0, 0), rotates the TE midpoint onto the x axis and scales it to x = 1, all rigid
    but the uniform scale. Then stretches each surface's x so it ends exactly at 1, which a slanted TE
    base (ends at different x) needs.

    Args:
        coords (np.ndarray): (n, 2) Selig-ordered coordinates

    Returns:
        tuple: (normalized coordinates, with the LE point inserted if it fell between points; the Transform)
    """
    i, le, inserted = Geom_find_le(coords)
    if inserted:
        coords = np.insert(coords, i, le, axis=0)
    d = 0.5 * (coords[0] + coords[-1]) - le
    chord = float(np.hypot(*d))
    if chord == 0.0:
        raise ValueError("the LE coincides with the TE midpoint")
    c, s = d / chord
    q = (coords - le) @ np.array([[c, -s], [s, c]]) / chord
    q[i] = 0.0  # exact, whatever the rotation's rounding

    x_end_up, x_end_lo = q[0, 0], q[-1, 0]
    q[:i + 1, 0] = q[:i + 1, 0] / x_end_up  # division, so the end lands on exactly 1
    q[i:, 0] = q[i:, 0] / x_end_lo
    transform = Transform(le_index=i, le_inserted=inserted, translate=(float(le[0]), float(le[1])),
                          rotation_deg=float(-np.degrees(np.arctan2(s, c))), scale=1.0 / chord,
                          stretch_upper=float(1.0 / x_end_up), stretch_lower=float(1.0 / x_end_lo))
    return q + 0.0, transform


def Geom_close_te(coords: np.ndarray, le_index: int) -> tuple[np.ndarray, float]:
    """
    Closes the TE with the inverse of Sequence 3's linear opening: each surface loses x times its own
    TE ordinate, so both ends land on (1, 0) and the LE stays put. A no-op on a closed contour.

    Args:
        coords (np.ndarray): (n, 2) normalized coordinates, both surfaces ending at x = 1
        le_index (int): Index of the LE point

    Returns:
        tuple: (closed coordinates, the TE gap before closing: upper minus lower TE ordinate)
    """
    q = coords.copy()
    y_up, y_lo = q[0, 1], q[-1, 1]
    q[:le_index + 1, 1] -= q[:le_index + 1, 0] * y_up
    q[le_index:, 1] -= q[le_index:, 0] * y_lo
    return q + 0.0, float(y_up - y_lo)
