"""
Sequence 3, steps 3-5: the spacing wanted along a contour, the turn pre-check on it, and where the points go.

Spacing h(s): dx_mid away from the LE and TE, dipping to le_spacing at the LE and te_spacing at the TE over
about cluster_width (Gaussian dips). Where both dips reach, the smaller wins, so h is exactly le_spacing at the
LE and te_spacing at the TE, and never above dx_mid. On a round TE the TE spacing holds across the caps, the
dip starting at the welds; te_spacing is clamped to what the caps' curvature lets pass the pre-check, so the
caps never fail it.

Points: N(s), the integral of ds/h, counts the panels the spacing calls for up to s. Each stretch between
pins gets ceil(its N) panels, evenly in N: no panel is longer than its target and the pins land exactly.
"""

import math
from typing import NamedTuple, Optional

import numpy as np

from src.datagen.geometry.common.constants import PRECHECK_MARGIN, THETA_MAX_DEG
from src.datagen.geometry.common.schemas import TETreatment
from src.datagen.geometry.sample.contour import Contour, PinKind
from src.datagen.geometry.sample.fields import SamplingFields


class Spacing(NamedTuple):
    """The target spacing over a contour. Lengths in chords."""
    s: np.ndarray        # arc lengths: the contour's table points and its pins
    h: np.ndarray        # target spacing there
    count: np.ndarray    # N(s): panels called for from the start up to s
    h_table: np.ndarray  # target spacing at the curvature table's points
    le: float
    te: float
    te_clamped: bool     # te_ratio * dx_mid was more than the caps let pass the pre-check


def Geom_target_spacing(contour: Contour, fields: SamplingFields) -> Spacing:
    """
    The spacing a config's knobs call for along a contour, and the panel count it implies.

    Args:
        contour (Contour): The study's contour
        fields (SamplingFields): The config's sampling knobs

    Returns:
        Spacing: h and N(s) on the contour's arc-length grid
    """
    le, te = fields.le_ratio * fields.dx_mid, fields.te_ratio * fields.dx_mid
    te_max = _te_max(contour)
    te_clamped = te_max is not None and te > te_max
    if te_clamped:
        te = te_max
    s = np.union1d(contour.table_s, [p.s for p in contour.pins])
    h = _spacing(contour, s, fields.dx_mid, le, te, fields.cluster_width)
    inv = 1.0 / h
    count = np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(s))])
    h_table = _spacing(contour, contour.table_s, fields.dx_mid, le, te, fields.cluster_width)
    return Spacing(s=s, h=h, count=count, h_table=h_table, le=le, te=te, te_clamped=bool(te_clamped))


def Geom_turn_precheck(contour: Contour, spacing: Spacing) -> tuple[float, float]:
    """
    How far the spacing comes to turning more than THETA_MAX_DEG at a point: a chord h on curvature k turns
    2 asin(k h / 2), so k h must stay under 2 sin(THETA_MAX_DEG / 2), here with PRECHECK_MARGIN to spare.
    Needs no points, so a config failing it is rejected before any are placed.

    Args:
        contour (Contour): The study's contour
        spacing (Spacing): Its target spacing

    Returns:
        tuple: (largest curvature x spacing over that limit, minus 1, above 0 rejecting; x where it is)
    """
    product = contour.table_kappa * spacing.h_table
    worst = int(np.argmax(product))
    x = float(contour.evaluate(contour.table_s[worst:worst + 1])[0, 0])
    return float(product[worst] / _turn_limit() - 1.0), x


def Geom_stretch_panels(contour: Contour, spacing: Spacing) -> list[int]:
    """
    Panels between each pair of consecutive pins: the spacing's count over the stretch, rounded up.

    Args:
        contour (Contour): The study's contour
        spacing (Spacing): Its target spacing

    Returns:
        list: Panels per stretch; the sample has 1 + their sum points
    """
    at = np.interp([p.s for p in contour.pins], spacing.s, spacing.count)
    return [max(1, math.ceil(n - 1e-9)) for n in np.diff(at)]  # 1e-9: a count landing on a whole number stays one


def Geom_stations(contour: Contour, spacing: Spacing, panels: list[int]) -> np.ndarray:
    """
    The arc length of every point: each stretch's panels evenly in N(s), its pins exactly.

    Args:
        contour (Contour): The study's contour
        spacing (Spacing): Its target spacing
        panels (list): Panels per stretch, from `Geom_stretch_panels`

    Returns:
        np.ndarray: (1 + sum(panels),) increasing arc lengths, from the first pin's to the last's
    """
    pins = contour.pins
    at = np.interp([p.s for p in pins], spacing.s, spacing.count)
    parts = []
    for j, m in enumerate(panels):
        stretch = np.interp(np.linspace(at[j], at[j + 1], m + 1)[:-1], spacing.count, spacing.s)
        stretch[0] = pins[j].s
        parts.append(stretch)
    return np.concatenate(parts + [[pins[-1].s]])


def _turn_limit() -> float:
    """The most curvature x spacing the pre-check passes."""
    return 2 * math.sin(math.radians(THETA_MAX_DEG) / 2) * PRECHECK_MARGIN


def _te_max(contour: Contour) -> Optional[float]:
    """A round TE's longest cap spacing that passes the pre-check, read off the same curvature table."""
    if contour.te is not TETreatment.ROUND:
        return None
    pins = {p.kind: p.s for p in contour.pins}
    caps = (contour.table_s < pins[PinKind.WELD_UPPER]) | (contour.table_s > pins[PinKind.WELD_LOWER])
    return _turn_limit() / float(contour.table_kappa[caps].max()) * (1 - 1e-9)


def _spacing(contour: Contour, s: np.ndarray, dx: float, le: float, te: float, width: float) -> np.ndarray:
    pins = {p.kind: p.s for p in contour.pins}
    if contour.te is TETreatment.ROUND:  # flat across the caps, rising from the welds
        start, end = pins[PinKind.WELD_UPPER], pins[PinKind.WELD_LOWER]
    else:
        start, end = 0.0, contour.length
    d_le = np.abs(s - pins[PinKind.LE])
    d_te = np.maximum(0.0, np.minimum(s - start, end - s))
    return np.minimum(dx - (dx - le) * np.exp(-(d_le / width) ** 2), dx - (dx - te) * np.exp(-(d_te / width) ** 2))
