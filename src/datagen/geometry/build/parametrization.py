"""
CST parametrization (Kulfan). Each surface is y = C(x) * S(x) + L * x (1 - x)^(N + 0.5): class function
C(x) = x^N1 (1 - x)^N2, shape function S(x) = sum_i A_i B_i(x) over the Bernstein basis of order N, and
Kulfan's leading-edge modification (LEM) term with one weight L for both surfaces (upper/lower). Kulfan's 
TE-thickness term is always zero: every stored shape is closed (`normalize.Geom_close_te` removes it).

With N1 = 0.5 and N2 = 1 every surface leaves (0, 0) with a vertical tangent and closes sharp at (1, 0),
which is the frame `normalize.py` puts every airfoil in. Near the LE a surface is ~A_0 sqrt(x), a nose of
radius A_0^2 / 2, so the fit gives both surfaces one A_0. A cambered nose also leans, y ~ ±sqrt(2Rx) + cx,
and the half powers of x cannot make that cx: the LEM term does, cancels in the thickness and has zero
value and slope at the TE.

Near the TE a surface is ~A_N (1 - x), so the TE wedge is positive exactly when the upper A_N exceeds the
lower's. The fit solves for that difference and bounds it at zero: a cusp at worst, never crossed.
"""

import math

import numpy as np
from scipy.optimize import lsq_linear

from src.datagen.geometry.common.constants import CST_N1, CST_N2, CST_NOSE, CST_ORDER, CURVE_POINTS
from src.datagen.geometry.common.schemas import CSTParams, FitMetrics, Nose


def _bernstein(x: np.ndarray, order: int) -> np.ndarray:
    """(len(x), order + 1) Bernstein basis."""
    i = np.arange(order + 1)
    binom = np.array([math.comb(order, k) for k in i], dtype=float)
    return binom * x[:, None] ** i * (1 - x[:, None]) ** (order - i)


def _class(x: np.ndarray, n1: float, n2: float) -> np.ndarray:
    return x ** n1 * (1 - x) ** n2


def _lem(x: np.ndarray, order: int) -> np.ndarray:
    """The LEM term's shape: ~x at the nose, peaks at x = 1 / (N + 1.5), 3% of that by mid-chord at N = 8."""
    return x * (1 - x) ** (order + 0.5)


def Geom_cst_eval(cst: CSTParams, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluates both surfaces.

    Args:
        cst (CSTParams): The shape
        x (np.ndarray): Stations in [0, 1]

    Returns:
        tuple: (upper y, lower y) at `x`
    """
    x = np.asarray(x, dtype=float)
    basis = _class(x, cst.n1, cst.n2)[:, None] * _bernstein(x, cst.order)
    lem = cst.le_weight * _lem(x, cst.order)
    return basis @ np.asarray(cst.upper) + lem, basis @ np.asarray(cst.lower) + lem


def Geom_cst_slope(cst: CSTParams, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    The analytic slope dy/dx of both surfaces. Infinite at the LE, so x > 0; at x = 1 each surface's is -A_N.

    Args:
        cst (CSTParams): The shape
        x (np.ndarray): Stations in (0, 1]

    Returns:
        tuple: (upper dy/dx, lower dy/dx) at `x`
    """
    x = np.asarray(x, dtype=float)
    n, n1, n2 = cst.order, cst.n1, cst.n2
    basis = _bernstein(x, n)
    if n > 0:
        lower_basis = _bernstein(x, n - 1)
        d_basis = n * (np.pad(lower_basis, ((0, 0), (1, 0))) - np.pad(lower_basis, ((0, 0), (0, 1))))
    else:
        d_basis = np.zeros_like(basis)
    c = _class(x, n1, n2)
    d_c = n1 * x ** (n1 - 1) * (1 - x) ** n2 - n2 * x ** n1 * (1 - x) ** (n2 - 1)
    d_lem = cst.le_weight * ((1 - x) ** (n + 0.5) - (n + 0.5) * x * (1 - x) ** (n - 0.5))
    m = d_c[:, None] * basis + c[:, None] * d_basis
    return m @ np.asarray(cst.upper) + d_lem, m @ np.asarray(cst.lower) + d_lem


def Geom_cst_te_wedge(cst: CSTParams) -> float:
    """
    The angle between the surfaces where they meet at the TE: each arrives as y ~ A_N (1 - x).

    Args:
        cst (CSTParams): The shape

    Returns:
        float: The wedge angle [deg], negative when the surfaces cross there
    """
    return math.degrees(math.atan(cst.upper[-1]) - math.atan(cst.lower[-1]))


def Geom_cst_fit(coords: np.ndarray, le_index: int, order: int = CST_ORDER, nose: Nose = CST_NOSE) -> CSTParams:
    """
    Fits a CST to normalized, closed coordinates: one bounded linear least-squares solve on y at the
    points' x. The lower A_N is the upper's minus d, d >= 0, so the TE wedge is never negative. Where the
    unbounded best fit already has d >= 0 the bound is inactive and the result is that fit exactly; else
    it is the best fit with d = 0, a cusp (BVLS: Stark & Parker's active-set method).

    Args:
        coords (np.ndarray): (n, 2) Selig-ordered coordinates, LE at (0, 0), both ends at (1, 0)
        le_index (int): Index of the LE point
        order (int): Bernstein order of each surface
        nose (Nose): How the nose is represented

    Returns:
        CSTParams: The fitted shape
    """
    nose = Nose(nose)
    xu, yu = coords[:le_index + 1][::-1].T
    xl, yl = coords[le_index:].T
    au = _class(xu, CST_N1, CST_N2)[:, None] * _bernstein(xu, order)
    al = _class(xl, CST_N1, CST_N2)[:, None] * _bernstein(xl, order)
    zu, zl = np.zeros(len(xu)), np.zeros(len(xl))

    # Unknowns, each an (upper rows, lower rows) column: upper A_0..A_N, the lower's own coefficients
    # (A_1..A_N-1 when it shares A_0), d, then the LEM weight
    shared = nose is not Nose.SEPARATE
    cols = [(au[:, i], zl) for i in range(order + 1)]
    cols[0] = (au[:, 0], -al[:, 0] if shared else zl)
    cols[order] = (au[:, order], al[:, order])  # the lower A_N follows the upper's...
    cols += [(zu, al[:, i]) for i in range(int(shared), order)]
    cols.append((zu, -al[:, order]))            # ...minus d
    if nose is Nose.LEM:
        cols.append((_lem(xu, order), _lem(xl, order)))
    m = np.column_stack([np.concatenate(c) for c in cols])
    d = order + 1 + order - int(shared)
    floor = np.full(m.shape[1], -np.inf)
    floor[d] = 0.0
    sol = lsq_linear(m, np.concatenate([yu, yl]), bounds=(floor, np.inf), method="bvls").x

    upper = sol[:order + 1]
    lower = ([-upper[0]] if shared else []) + list(sol[order + 1:d]) + [upper[order] - sol[d]]
    le_weight = float(sol[d + 1]) if nose is Nose.LEM else 0.0
    return CSTParams(n1=CST_N1, n2=CST_N2, upper=upper.tolist(), lower=[float(v) for v in lower], le_weight=le_weight)


def Geom_cst_contour(cst: CSTParams, n_points: int) -> np.ndarray:
    """
    The shape as a cosine-spaced closed Selig contour.

    Args:
        cst (CSTParams): The shape
        n_points (int): Total points, odd, so the LE is the middle one

    Returns:
        np.ndarray: (n_points, 2), first and last point exactly (1, 0), the middle one exactly (0, 0)
    """
    if n_points < 3 or n_points % 2 == 0:
        raise ValueError(f"n_points must be odd and at least 3, got {n_points}")
    x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, (n_points + 1) // 2)))
    yu, yl = Geom_cst_eval(cst, x)
    upper = np.stack([x, yu], axis=1)[::-1]
    lower = np.stack([x, yl], axis=1)[1:]
    return np.vstack([upper, lower]) + 0.0


def Geom_curve_distance(points: np.ndarray, polyline: np.ndarray, chunk: int = 256) -> np.ndarray:
    """
    For each point, the distance to the nearest segment of a polyline.

    Args:
        points (np.ndarray): (n, 2) points
        polyline (np.ndarray): (m, 2) vertices, consecutive ones joined
        chunk (int): Points handled per batch, to bound memory

    Returns:
        np.ndarray: (n,) distances
    """
    a = polyline[:-1]
    dx, dy = (polyline[1:] - a).T
    l2 = np.maximum(dx * dx + dy * dy, 1e-30)
    out = np.empty(len(points))
    for k in range(0, len(points), chunk):  # (chunk, segments) arrays, one sqrt per point after the min
        p = points[k:k + chunk]
        ex, ey = p[:, :1] - a[:, 0], p[:, 1:] - a[:, 1]
        t = np.clip((ex * dx + ey * dy) / l2, 0.0, 1.0)
        ex -= t * dx
        ey -= t * dy
        out[k:k + chunk] = np.sqrt((ex * ex + ey * ey).min(axis=1))
    return out


def Geom_cst_fit_metrics(cst: CSTParams, coords: np.ndarray, le_index: int) -> FitMetrics:
    """
    How far the fitted points lie from the shape. Each surface's points are measured against that
    surface only: near a sharp TE the other one is closer than the fit error.

    Args:
        cst (CSTParams): The fitted shape
        coords (np.ndarray): (n, 2) the points it was fitted to
        le_index (int): Index of their LE point

    Returns:
        FitMetrics: Worst and RMS distance, the x of the worst, and how far the worst stands out from
            its neighbours
    """
    dense = Geom_cst_contour(cst, CURVE_POINTS)
    mid = CURVE_POINTS // 2
    dist = np.concatenate([Geom_curve_distance(coords[:le_index + 1], dense[:mid + 1]),
                           Geom_curve_distance(coords[le_index + 1:], dense[mid:])])
    worst = int(np.argmax(dist))
    near = np.concatenate([dist[max(worst - 3, 0):worst], dist[worst + 1:worst + 4]])
    return FitMetrics(max_dev=float(dist[worst]), rms_dev=float(np.sqrt(np.mean(dist ** 2))),
                      x_at_max=float(coords[worst, 0]), n_points=len(coords),
                      isolation=float(dist[worst] / max(np.median(near), 1e-15)), zigzag=_zigzags(coords, worst))


def _zigzags(coords: np.ndarray, i: int) -> bool:
    """Whether the turn at point i has the opposite sign to the turns at both its neighbours."""
    if i < 2 or i > len(coords) - 3:
        return False
    d = np.diff(coords[i - 2:i + 3], axis=0)
    turn = np.diff(np.arctan2(d[:, 1], d[:, 0]))
    turn = (turn + np.pi) % (2 * np.pi) - np.pi  # at i - 1, i, i + 1
    return bool(turn[0] * turn[1] < 0 and turn[1] * turn[2] < 0)
