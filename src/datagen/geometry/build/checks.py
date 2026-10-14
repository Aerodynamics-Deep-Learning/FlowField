"""
The gates an airfoil passes before it is stored. Each returns (ok, measured value, reason), the
reason empty when ok.
"""

import numpy as np

from src.datagen.geometry.common.constants import CST_FIT_TOL, CURVE_POINTS, FAMILY_MATCH_TOL, SUSPECT_ISOLATION
from .parametrization import Geom_cst_eval, Geom_cst_te_wedge, Geom_curve_distance, _bernstein
from src.datagen.geometry.common.schemas import CSTParams, FitMetrics


def Geom_check_fit(fit: FitMetrics, tol: float = CST_FIT_TOL) -> tuple[bool, float, str]:
    """
    The fit gate: the worst point-to-shape distance against the tolerance.

    Args:
        fit (FitMetrics): The fit's metrics
        tol (float): Largest allowed distance, chord units

    Returns:
        tuple: (ok, worst distance, reason)
    """
    if fit.max_dev <= tol:
        return True, fit.max_dev, ""
    return False, fit.max_dev, f"fit misses a point by {fit.max_dev:.2e}c at x={fit.x_at_max:.4f} (limit {tol:g})"


def Geom_check_suspect_point(fit: FitMetrics, ratio: float = SUSPECT_ISOLATION) -> tuple[bool, float, str]:
    """
    A hint for the report, not a gate: the worst-fit point is far worse than its neighbours and the
    points zig-zag there. A mistyped coordinate does both; a region the CST can't follow (a nose) does
    neither, since its neighbours miss too and the points stay smooth.

    Args:
        fit (FitMetrics): The fit's metrics
        ratio (float): How many times its neighbours' distance the worst must be

    Returns:
        tuple: (ok = nothing suspect, the worst point's isolation, what to check)
    """
    if fit.isolation >= ratio and fit.zigzag:
        return False, fit.isolation, (f"the point at x={fit.x_at_max:.4f} is {fit.isolation:.1f}x further off than "
                                      f"its neighbours and the points zig-zag there: check that coordinate in the file")
    return True, fit.isolation, ""


def Geom_check_thickness(cst: CSTParams) -> tuple[bool, float, str]:
    """
    The shape must have positive thickness everywhere inside and a TE wedge of at least zero (a cusp).
    Read off the shape functions, upper minus lower, since the class function only scales that and is
    zero at both ends, and the LEM term cancels; at x = 1 their difference gives the wedge.

    Args:
        cst (CSTParams): The fitted shape

    Returns:
        tuple: (ok, smallest upper-minus-lower shape function, reason)
    """
    x = np.linspace(0.0, 1.0, CURVE_POINTS)
    diff = _bernstein(x, cst.order) @ (np.asarray(cst.upper) - np.asarray(cst.lower))
    if diff[-1] < 0:  # the fit's bound prevents this; kept for shapes made elsewhere
        return False, float(diff[-1]), f"surfaces meet at the TE crossed (wedge angle {Geom_cst_te_wedge(cst):.2f} deg)"
    k = int(np.argmin(diff[:-1]))
    if diff[k] > 0:
        return True, float(min(diff[k], diff[-1])), ""
    yu, yl = Geom_cst_eval(cst, x[k:k + 1])
    return False, float(diff[k]), (f"surfaces cross at x={x[k]:.4f} (thickness {yu[0] - yl[0]:.2e}c), "
                                   f"usually a TE too open to close linearly")


def Geom_check_family_match(family_coords: np.ndarray, family_le: int, file_coords: np.ndarray,
                            file_le: int, tol: float = FAMILY_MATCH_TOL) -> tuple[bool, float, str]:
    """
    A family's formula shape against its own file's points, both normalized and closed, each surface
    against its own.

    Args:
        family_coords (np.ndarray): The formula contour
        family_le (int): Its LE index
        file_coords (np.ndarray): The file's points
        file_le (int): Their LE index
        tol (float): Largest allowed distance, chord units

    Returns:
        tuple: (ok, worst distance, reason)
    """
    dev = max(Geom_curve_distance(file_coords[:file_le + 1], family_coords[:family_le + 1]).max(),
              Geom_curve_distance(file_coords[file_le:], family_coords[family_le:]).max())
    if dev <= tol:
        return True, float(dev), ""
    return False, float(dev), f"the file's points miss the shape its name describes by {dev:.2e}c (limit {tol:g})"
