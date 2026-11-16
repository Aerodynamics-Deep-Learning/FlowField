"""
Airfoil families built from their name, always in closed-TE form: NACA 4-digit, 5-digit, modified
4-digit and 16-series.

Names match strictly (`naca` + digits, nothing else), so 6/6A-series and anything with a separator take
the tabulated path. A name is still no guarantee of its shape; `add.py` checks the formula against the
file's own points.

Formulas and constants: Abbott & von Doenhoff, and Ladson et al. NASA TM-4741 for the modified 4-digit.
"""

import hashlib
import re
from typing import Optional

import numpy as np

from src.datagen.geometry.common.constants import FAMILY_POINTS_PER_SURFACE
from src.datagen.geometry.common.schemas import Family

# Lengths differ except 16-series vs 5-digit, which the 5-digit table settles (no camber position 6)
_PATTERNS = [
    (Family.NACA4_MOD, re.compile(r"naca(\d)(\d)(\d\d)(\d)(\d)", re.IGNORECASE), ("m", "p", "t", "i", "pos")),
    (Family.NACA16, re.compile(r"naca16(\d)(\d\d)", re.IGNORECASE), ("cl", "t")),
    (Family.NACA5, re.compile(r"naca(\d)(\d)(\d)(\d\d)", re.IGNORECASE), ("l", "p", "s", "t")),
    (Family.NACA4, re.compile(r"naca(\d)(\d)(\d\d)", re.IGNORECASE), ("m", "p", "t")),
]

# 5-digit mean lines, (camber position digit, reflex digit) -> (r, k1, k2/k1), for design CL 0.3
_NACA5_MEAN_LINES = {
    (1, 0): (0.0580, 361.400, 0.0),
    (2, 0): (0.1260, 51.640, 0.0),
    (3, 0): (0.2025, 15.957, 0.0),
    (4, 0): (0.2900, 6.643, 0.0),
    (5, 0): (0.3910, 3.230, 0.0),
    (2, 1): (0.1300, 51.990, 0.000764),
    (3, 1): (0.2170, 15.793, 0.00677),
    (4, 1): (0.3180, 6.520, 0.0303),
    (5, 1): (0.4410, 3.191, 0.1355),
}

# Modified 4-digit TE slope d1, per max-thickness position digit
_MOD4_TE_SLOPE = {2: 0.200, 3: 0.234, 4: 0.315, 5: 0.465, 6: 0.700}


def Geom_parse_family(name: str) -> Optional[tuple[Family, dict[str, int]]]:
    """
    Reads a family designation off an airfoil name.

    Args:
        name (str): The airfoil name, e.g. "naca2412"

    Returns:
        tuple | None: (Family, its digits by name), or None for a name that is not a valid designation
    """
    for family, pattern, keys in _PATTERNS:
        match = pattern.fullmatch(name)
        if match:
            params = dict(zip(keys, map(int, match.groups())))
            if _valid(family, params):
                return family, params
    return None


def _valid(family: Family, q: dict[str, int]) -> bool:
    """Whether the digits describe a section the formulas define."""
    if q["t"] == 0:
        return False
    if family in (Family.NACA4, Family.NACA4_MOD) and (q["m"] == 0) != (q["p"] == 0):
        return False
    if family is Family.NACA4_MOD and not (1 <= q["i"] <= 8 and q["pos"] in _MOD4_TE_SLOPE):
        return False
    if family is Family.NACA5 and (q["l"] == 0 or (q["p"], q["s"]) not in _NACA5_MEAN_LINES):
        return False
    return True


def Geom_family_key(family: Family, params: dict[str, int]) -> str:
    """
    The canonical string a family airfoil's geometry ID is the hash of.

    Args:
        family (Family): The family
        params (dict): Its digits, as `Geom_parse_family` returns them

    Returns:
        str: e.g. "naca4:m=2:p=4:t=12:te=closed"
    """
    return ":".join([family.value, *(f"{k}={v}" for k, v in params.items()), "te=closed"])


def Geom_family_id(family: Family, params: dict[str, int]) -> str:
    """SHA-256 of `Geom_family_key`."""
    return hashlib.sha256(Geom_family_key(family, params).encode()).hexdigest()


def Geom_family_coords(family: Family, params: dict[str, int],
                       n_per_surface: int = FAMILY_POINTS_PER_SURFACE) -> np.ndarray:
    """
    A family airfoil's closed-TE contour, thickness applied perpendicular to the mean line.

    Args:
        family (Family): The family
        params (dict): Its digits, as `Geom_parse_family` returns them
        n_per_surface (int): Cosine-spaced stations per surface, LE shared

    Returns:
        np.ndarray: (2 * n_per_surface - 1, 2) Selig-ordered coordinates
    """
    return _family_contour(family, params, n_per_surface, closed_te=True)


def _family_contour(family: Family, q: dict[str, int], n: int, closed_te: bool) -> np.ndarray:
    """`Geom_family_coords` with the TE choice open, so the open forms can be checked against references."""
    x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))
    if family is Family.NACA4:
        yt = _naca4_thickness(x, q["t"] / 100, closed_te)
        yc, dyc = _naca4_camber(x, q["m"] / 100, q["p"] / 10)
    elif family is Family.NACA4_MOD:
        yt = _mod4_thickness(x, q["t"] / 100, q["i"], q["pos"], closed_te)
        yc, dyc = _naca4_camber(x, q["m"] / 100, q["p"] / 10)
    elif family is Family.NACA5:
        yt = _naca4_thickness(x, q["t"] / 100, closed_te)
        yc, dyc = _naca5_camber(x, q["l"], q["p"], q["s"])
    else:
        yt = _mod4_thickness(x, q["t"] / 100, 4, 5, closed_te)
        yc, dyc = _a1_camber(x, q["cl"] / 10)

    theta = np.arctan(dyc)
    upper = np.stack([x - yt * np.sin(theta), yc + yt * np.cos(theta)], axis=1)
    lower = np.stack([x + yt * np.sin(theta), yc - yt * np.cos(theta)], axis=1)
    return np.vstack([upper[::-1], lower[1:]]) + 0.0


def _naca4_thickness(x: np.ndarray, t: float, closed_te: bool) -> np.ndarray:
    """4-digit half-thickness; -0.1036 instead of -0.1015 closes the TE."""
    a4 = -0.1036 if closed_te else -0.1015
    return 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 + a4 * x ** 4)


def _mod4_thickness(x: np.ndarray, t: float, i: int, pos: int, closed_te: bool) -> np.ndarray:
    """
    Modified 4-digit half-thickness, LE radius index `i`, max thickness at `pos`/10. Aft of the max a
    cubic in (1 - x) with TE ordinate d0 (0.002, or 0 closed) and slope d1, d2/d3 from the max; forward
    a0*sqrt(x) + cubic, matching the aft part's value, slope and curvature at the max.
    """
    m, u = pos / 10, 1 - pos / 10
    a0 = 0.296904 * i / 6
    d0 = 0.0 if closed_te else 0.002
    d1 = _MOD4_TE_SLOPE[pos]
    d2, d3 = np.linalg.solve([[u ** 2, u ** 3], [2 * u, 3 * u ** 2]], [0.1 - d0 - d1 * u, -d1])
    a1, a2, a3 = np.linalg.solve(
        [[m, m ** 2, m ** 3], [1, 2 * m, 3 * m ** 2], [0, 2, 6 * m]],
        [0.1 - a0 * np.sqrt(m), -a0 / (2 * np.sqrt(m)), 2 * d2 + 6 * d3 * u + a0 / (4 * m ** 1.5)])
    v = 1 - x
    fwd = a0 * np.sqrt(x) + a1 * x + a2 * x ** 2 + a3 * x ** 3
    aft = d0 + d1 * v + d2 * v ** 2 + d3 * v ** 3
    return t / 0.2 * np.where(x < m, fwd, aft)


def _naca4_camber(x: np.ndarray, m: float, p: float) -> tuple[np.ndarray, np.ndarray]:
    """4-digit mean line and its slope: max camber `m` at `p`."""
    if m == 0:
        return np.zeros_like(x), np.zeros_like(x)
    fwd = x < p
    yc = np.where(fwd, m / p ** 2 * (2 * p * x - x ** 2), m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))
    dyc = np.where(fwd, 2 * m / p ** 2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    return yc, dyc


def _naca5_camber(x: np.ndarray, l: int, p: int, s: int) -> tuple[np.ndarray, np.ndarray]:
    """5-digit mean line and its slope, standard (s=0) or reflex (s=1), scaled to design CL 0.15*l."""
    r, k1, k21 = _NACA5_MEAN_LINES[(p, s)]
    f = k1 / 6 * l / 2
    fwd = x < r
    if s == 0:
        yc = np.where(fwd, x ** 3 - 3 * r * x ** 2 + r ** 2 * (3 - r) * x, r ** 3 * (1 - x))
        dyc = np.where(fwd, 3 * x ** 2 - 6 * r * x + r ** 2 * (3 - r), -r ** 3)
    else:
        lin = -k21 * (1 - r) ** 3 * x - r ** 3 * x + r ** 3
        yc = np.where(fwd, (x - r) ** 3, k21 * (x - r) ** 3) + lin
        dyc = np.where(fwd, 3 * (x - r) ** 2, 3 * k21 * (x - r) ** 2) - k21 * (1 - r) ** 3 - r ** 3
    return f * yc, f * dyc


def _a1_camber(x: np.ndarray, cl: float) -> tuple[np.ndarray, np.ndarray]:
    """The a=1.0 (uniform load) mean line of the 16-series, design CL `cl`; its slope is infinite at both ends."""
    if cl == 0:
        return np.zeros_like(x), np.zeros_like(x)
    k = cl / (4 * np.pi)
    with np.errstate(divide="ignore", invalid="ignore"):  # 0 * log(0) at the ends, replaced by 0
        log_x, log_v = np.log(x), np.log(1 - x)
        yc = -k * (np.where(x < 1, (1 - x) * log_v, 0.0) + np.where(x > 0, x * log_x, 0.0))
    return yc, k * (log_v - log_x)
