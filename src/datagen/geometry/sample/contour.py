"""
Sequence 3, step 2: a stored shape given the TE its mesher needs, as one continuous curve in Selig order.
Built once per study; the sampler then places points on it by arc length.

    sharp   the stored CST as is, closed at (1, 0)
    blunt   each surface opened by x * g/2 (Kulfan's TE-thickness term, the inverse of the ingest's
            closing); open, ending at (1, +-g/2)
    round   the blunt shape cut at the welds x_w = 1 - WELD_OFFSET * g, each cut piece replaced by a
            quadratic Bezier cap W -> P1 -> (1, 0), P1 where the surface's tangent at W reaches x = 1

A cap starts at W with the surface's own slope and reaches (1, 0), the middle of the gap, with a vertical
tangent: the round contour has no kink and its chord stays exactly 1. A Bezier lies in the triangle of its
control points, so with all three on its own side of y = 0 a cap cannot cross the other.

The surfaces are parametrized by u = sqrt(x), which stays regular at the LE where dy/dx does not. Each
piece is tabulated densely; the tables only say where along the curve a point sits, `evaluate` computes
it on the piece itself.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple, Optional

import numpy as np

from src.datagen.geometry.build.parametrization import Geom_cst_eval, Geom_cst_slope
from src.datagen.geometry.common.constants import ARC_TABLE_POINTS, THETA_MAX_DEG, WELD_OFFSET
from src.datagen.geometry.common.schemas import CSTParams, ShapeRecord, TETreatment


class ContourError(ValueError):
    """The shape cannot take the requested TE: a study setup failure, not a failed evaluation."""


class PinKind(str, Enum):
    """Points the sampler places exactly."""
    TIP = "tip"                # a closed contour's closure point (1, 0), first and last
    END_UPPER = "end_upper"    # a blunt contour's open ends
    END_LOWER = "end_lower"
    WELD_UPPER = "weld_upper"  # where a round TE's caps meet the surfaces
    WELD_LOWER = "weld_lower"
    LE = "le"


class Pin(NamedTuple):
    kind: PinKind
    s: float                   # arc length from the contour's start
    xy: tuple[float, float]    # exact coordinates


class _Surface:
    """One surface, opened by x * gap/2, as a function of u = sqrt(x)."""

    def __init__(self, cst: CSTParams, upper: bool, gap: float):
        self.cst, self.upper, self.gap = cst, upper, gap

    def points(self, u: np.ndarray) -> np.ndarray:
        x = np.asarray(u, dtype=float) ** 2
        yu, yl = Geom_cst_eval(self.cst, x)
        y = yu + x * self.gap / 2 if self.upper else yl - x * self.gap / 2
        return np.stack([x, y], axis=1)


class _Cap:
    """A quadratic Bezier from p0 to p2, pulled toward p1."""

    def __init__(self, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray):
        self.p0, self.p1, self.p2 = p0, p1, p2

    def points(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)[:, None]
        return (1 - t) ** 2 * self.p0 + 2 * t * (1 - t) * self.p1 + t ** 2 * self.p2

    def peak_curvature(self) -> float:
        """The largest curvature, at the vertex: B' x B'' is constant, so where |B'| is smallest."""
        a, b = self.p1 - self.p0, self.p2 - 2 * self.p1 + self.p0
        t = float(np.clip(-np.dot(a, b) / np.dot(b, b), 0.0, 1.0))
        return float(abs(a[0] * b[1] - a[1] * b[0]) / (2 * np.linalg.norm(a + t * b) ** 3))


@dataclass(frozen=True, eq=False)
class Contour:
    """
    A stored shape with the TE its mesher needs: one curve from the first pin to the last, in Selig order.
    `evaluate` gives exact points by arc length; the curvature table is for spacing them.
    """
    geometry_id: str
    shape_version: str
    te: TETreatment
    gap: Optional[float]                            # None for a sharp TE
    closed: bool
    length: float
    pins: tuple[Pin, ...]                           # in order, first and last included
    weld_x: Optional[float]                         # round only
    caps: Optional[tuple[np.ndarray, np.ndarray]]   # round only: (3, 2) control points, upper then lower
    te_cap_limit: Optional[float]                   # round only: the longest panel on the caps that turns <= THETA_MAX_DEG
    table_s: np.ndarray                             # arc length of the curvature table's points
    table_kappa: np.ndarray                         # |curvature| there, each piece on its own (none across a weld)
    _pieces: tuple                                  # (piece, its parameters, arc length within it) per piece
    _starts: np.ndarray                             # arc length where each piece starts

    def evaluate(self, s: np.ndarray) -> np.ndarray:
        """
        Points on the contour, each computed on its own piece; at a pin's s, exactly the pin.

        Args:
            s (np.ndarray): Arc lengths in [0, length]

        Returns:
            np.ndarray: (len(s), 2) points
        """
        s = np.asarray(s, dtype=float)
        if s.size and (s.min() < 0.0 or s.max() > self.length):
            raise ValueError(f"arc length outside [0, {self.length}]")
        piece = np.clip(np.searchsorted(self._starts, s, side="right") - 1, 0, len(self._pieces) - 1)
        out = np.empty((len(s), 2))
        for i, (shape, params, arc) in enumerate(self._pieces):
            on = piece == i
            if on.any():
                out[on] = shape.points(np.interp(s[on] - self._starts[i], arc, params))
        for pin in self.pins:
            out[s == pin.s] = pin.xy
        return out


def Geom_build_contour(shape: ShapeRecord, te: TETreatment, gap: Optional[float] = None) -> Contour:
    """
    Gives a stored shape the TE its mesher needs, as one tabulated curve.

    Args:
        shape (ShapeRecord): The stored shape
        te (TETreatment): The TE the mesher needs
        gap (float): The study's TE gap [c], for a blunt or round TE; ignored for a sharp one

    Returns:
        Contour: The curve, its pins, and for a round TE its caps and their spacing limit

    Raises:
        ContourError: If a blunt or round TE gets no positive gap, or a round TE's cap would leave its own
            side of y = 0
    """
    te = TETreatment(te)
    if te is TETreatment.SHARP:
        gap = None
    elif gap is None or not gap > 0.0:
        raise ContourError(f"a {te.value} TE needs a positive gap, got {gap}")
    upper, lower = _Surface(shape.cst, True, gap or 0.0), _Surface(shape.cst, False, gap or 0.0)

    weld_x = caps = te_cap_limit = None
    if te is TETreatment.ROUND:
        weld_x = 1.0 - WELD_OFFSET * gap
        cap_u, cap_l = _caps(shape.cst, gap, weld_x)
        u_w = math.sqrt(weld_x)
        spec = [(cap_u, 1.0, 0.0), (upper, u_w, 0.0), (lower, 0.0, u_w), (cap_l, 0.0, 1.0)]
        kinds = [PinKind.TIP, PinKind.WELD_UPPER, PinKind.LE, PinKind.WELD_LOWER, PinKind.TIP]
        xys = [(1.0, 0.0), tuple(cap_u.p0), (0.0, 0.0), tuple(cap_l.p0), (1.0, 0.0)]
        caps = tuple(np.array([c.p0, c.p1, c.p2]) for c in (cap_u, cap_l))
        te_cap_limit = 2 * math.sin(math.radians(THETA_MAX_DEG) / 2) / max(cap_u.peak_curvature(),
                                                                        cap_l.peak_curvature())
    elif te is TETreatment.SHARP:
        spec = [(upper, 1.0, 0.0), (lower, 0.0, 1.0)]
        kinds = [PinKind.TIP, PinKind.LE, PinKind.TIP]
        xys = [(1.0, 0.0), (0.0, 0.0), (1.0, 0.0)]
    else:
        spec = [(upper, 1.0, 0.0), (lower, 0.0, 1.0)]
        kinds = [PinKind.END_UPPER, PinKind.LE, PinKind.END_LOWER]
        xys = [(1.0, gap / 2), (0.0, 0.0), (1.0, -gap / 2)]

    pieces, starts, table_s, table_kappa, offset = [], [], [], [], 0.0
    for piece, start, end in spec:
        params = np.linspace(start, end, ARC_TABLE_POINTS)
        d = np.diff(piece.points(params), axis=0)
        seg = np.hypot(d[:, 0], d[:, 1])
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        turn = np.abs(np.diff(np.unwrap(np.arctan2(d[:, 1], d[:, 0]))))
        table_s.append(offset + arc[1:-1])
        table_kappa.append(turn / (0.5 * (seg[1:] + seg[:-1])))
        pieces.append((piece, _frozen(params), _frozen(arc)))
        starts.append(offset)
        offset += float(arc[-1])

    pins = tuple(Pin(k, float(s), (float(x), float(y))) for k, s, (x, y) in zip(kinds, starts + [offset], xys))
    return Contour(geometry_id=shape.geometry_id, shape_version=shape.shape_version, te=te, gap=gap,
                   closed=te is not TETreatment.BLUNT, length=offset, pins=pins, weld_x=weld_x, caps=caps,
                   te_cap_limit=te_cap_limit, table_s=_frozen(np.concatenate(table_s)),
                   table_kappa=_frozen(np.concatenate(table_kappa)), _pieces=tuple(pieces),
                   _starts=_frozen(np.array(starts)))


def _caps(cst: CSTParams, gap: float, weld_x: float) -> tuple[_Cap, _Cap]:
    """The upper and lower caps: W on the opened surface, P1 on its tangent at x = 1, and the tip."""
    x = np.array([weld_x])
    (yu,), (yl,) = Geom_cst_eval(cst, x)
    (su,), (sl,) = Geom_cst_slope(cst, x)
    tip = np.array([1.0, 0.0])
    caps = []
    for side, y, slope in ((1.0, yu + weld_x * gap / 2, su + gap / 2), (-1.0, yl - weld_x * gap / 2, sl - gap / 2)):
        w = np.array([weld_x, float(y)])
        p1 = np.array([1.0, float(y + slope * (1.0 - weld_x))])
        if not (side * w[1] > 0.0 and side * p1[1] > 0.0):
            surface = "upper" if side > 0 else "lower"
            raise ContourError(f"the {surface} cap would leave its side of y = 0 (W {w.tolist()}, P1 {p1.tolist()}): "
                               f"the opened TE turns too far off the chord line for a round cap")
        caps.append(_Cap(w, p1, tip))
    return caps[0], caps[1]


def _frozen(a: np.ndarray) -> np.ndarray:
    a.setflags(write=False)
    return a
