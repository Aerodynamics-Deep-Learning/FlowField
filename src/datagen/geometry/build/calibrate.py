"""
Calibration of the CST order, nose and fit tolerance: fits every airfoil in a root folder at a range of
orders, with each way of representing the nose, through the same code the ingest uses, and writes the
table the constants are chosen from. Touches no database. Run from the command line by
`scripts/datagen/geometry/calibrate.py`.

Files are independent, so they are fitted in parallel processes, one per core by default.
"""

import datetime
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
from pydantic import BaseModel
from scipy.interpolate import CubicSpline

from .add import _fit_and_gate, _prepare
from src.datagen.geometry.common.constants import CST_FIT_TOL, CST_NOSE, CST_ORDER, CURVE_POINTS
from .parametrization import Geom_cst_contour, Geom_curve_distance
from src.datagen.geometry.common.paths import Geom_git_version, Geom_source_label
from src.datagen.geometry.common.schemas import CSTParams, Family, Nose, RejectReason

_TOLS = (5e-4, 1e-3, 2e-3, 3e-3)
_STRAY_MIN = 5e-4  # a fit strays when it leaves the data's spline by this much and twice its at-point error


class CalibrationFit(BaseModel):
    """One airfoil at one configuration. `max_dev` None: too few points for the order."""
    order: int
    nose: Nose
    max_dev: Optional[float] = None
    between_dev: Optional[float] = None  # midway between points, against a spline through them
    crossed: bool = False

    @property
    def strays(self) -> bool:
        return self.max_dev is not None and self.between_dev > max(2 * self.max_dev, _STRAY_MIN)


class CalibrationRow(BaseModel):
    """One file: its rejection before fitting, or its fits."""
    name: str
    source: str
    geometry_id: Optional[str] = None
    family: Optional[Family] = None
    n_fitted: int = 0
    reject: Optional[RejectReason] = None
    reason: str = ""
    fits: list[CalibrationFit] = []


def Geom_calibrate(folder, orders=range(4, 13), noses=tuple(Nose), workers: Optional[int] = None) -> list[CalibrationRow]:
    """
    Fits every .dat under `folder` at every (nose, order) configuration.

    Args:
        folder (str | Path): A root folder
        orders (Iterable[int]): CST orders to fit
        noses (Iterable[Nose]): Nose representations to fit
        workers (int | None): Parallel processes, one per core by default; 1 runs in this process

    Returns:
        list[CalibrationRow]: One per file, in sorted path order
    """
    folder = Path(folder)
    paths = sorted(folder.rglob("*.dat"))
    one = partial(_calibrate_file, folder=folder, orders=list(orders), noses=list(noses))
    workers = min(workers or os.cpu_count() or 1, max(len(paths), 1))
    if workers == 1:
        return [one(path) for path in paths]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, paths))


def _calibrate_file(path: Path, folder: Path, orders: list[int], noses: list[Nose]) -> CalibrationRow:
    """One file at every configuration; module-level so a worker process can run it."""
    res, pts = _prepare(path)
    row = CalibrationRow(name=res.identity.name if res.identity else path.stem,
                         source=path.relative_to(folder).as_posix(), reject=res.reject, reason=res.reason)
    if not res.accepted:
        return row
    row.geometry_id, row.family, row.n_fitted = res.identity.geometry_id, res.identity.family, len(pts.fit)
    midpoints = _midpoints(pts.fit)
    le = res.transform.le_index
    for nose in noses:
        for order in orders:
            out = _fit_and_gate(res.model_copy(), pts, order, nose, np.inf, measure_source=False)
            if out.reject is RejectReason.TOO_FEW_POINTS:
                row.fits.append(CalibrationFit(order=order, nose=nose))
            else:
                row.fits.append(CalibrationFit(order=order, nose=nose, max_dev=out.fit.max_dev,
                                               between_dev=_between_dev(out.cst, midpoints, le),
                                               crossed=out.reject is RejectReason.SURFACES_CROSS))
    return row


def _midpoints(pts: np.ndarray) -> np.ndarray:
    """Points on a cubic spline through `pts`, midway (by chord length) between each consecutive pair."""
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
    mid = 0.5 * (s[1:] + s[:-1])
    return np.stack([CubicSpline(s, pts[:, 0])(mid), CubicSpline(s, pts[:, 1])(mid)], axis=1)


def _between_dev(cst: CSTParams, midpoints: np.ndarray, le: int) -> float:
    """Worst distance from the midpoints to the fitted shape, each surface against its own: where a fit overshoots."""
    dense = Geom_cst_contour(cst, CURVE_POINTS)
    m = CURVE_POINTS // 2
    return float(max(Geom_curve_distance(midpoints[:le], dense[:m + 1]).max(),
                     Geom_curve_distance(midpoints[le:], dense[m:]).max()))


class _Sweep(NamedTuple):
    """A calibration's rows, sorted out: the fitted files, one per geometry, and the configurations fitted."""
    fitted: list[CalibrationRow]
    unique: list[CalibrationRow]
    families: list[CalibrationRow]
    configs: list[tuple[Nose, int]]
    orders: list[int]
    noses: list[Nose]


def _sweep(rows: list[CalibrationRow]) -> _Sweep:
    fitted = [r for r in rows if r.fits]
    unique = list({r.geometry_id: r for r in fitted}.values())
    configs = sorted({(f.nose, f.order) for r in fitted for f in r.fits}, key=lambda c: (list(Nose).index(c[0]), c[1]))
    return _Sweep(fitted, unique, [r for r in unique if r.family is not None], configs,
                  sorted({o for _, o in configs}), list(dict.fromkeys(n for n, _ in configs)))


class _Config(NamedTuple):
    """One configuration over the unique geometries: every fit's max_dev, those whose surfaces don't cross, the rest."""
    dev: np.ndarray
    clean: np.ndarray
    crossed: int
    strays: int


def _config(unique: list[CalibrationRow], nose: Nose, order: int) -> _Config:
    fits = [f for r in unique for f in r.fits if (f.nose, f.order) == (nose, order) and f.max_dev is not None]
    return _Config(np.array([f.max_dev for f in fits]), np.array([f.max_dev for f in fits if not f.crossed]),
                   sum(f.crossed for f in fits), sum(f.strays for f in fits))


def _spread(dev: np.ndarray) -> str:
    """Median, p90 and max, or dashes for none."""
    if not len(dev):
        return f"{'-':>8} {'-':>8} {'-':>8}"
    return f"{np.median(dev):8.1e} {np.percentile(dev, 90):8.1e} {dev.max():8.1e}"


def _family_worst(families: list[CalibrationRow], nose: Nose, order: int) -> Optional[float]:
    devs = [f.max_dev for r in families for f in r.fits if (f.nose, f.order) == (nose, order) and f.max_dev is not None]
    return max(devs) if devs else None


def _head(rows: list[CalibrationRow], folder, sweep: _Sweep) -> list[str]:
    """The header, then a line per file rejected before fitting."""
    out = [f"CST calibration of {Geom_source_label(folder)}",
           f"{datetime.date.today().isoformat()}, code {Geom_git_version()}",
           f"{len(rows)} files: {len(sweep.fitted)} fitted ({len(sweep.unique)} unique geometries), "
           f"{len(rows) - len(sweep.fitted)} rejected before fitting", ""]
    if len(rows) > len(sweep.fitted):
        out.append("Rejected before fitting:")
        out += [f"  {r.source:40s} {r.reject.value:16s} {r.reason}" for r in rows if not r.fits]
        out.append("")
    return out


def _shown(sweep: _Sweep, default_nose: Nose) -> Nose:
    return Nose(default_nose) if Nose(default_nose) in sweep.noses else sweep.noses[0]


def _edge_cases(sweep: _Sweep, shown: Nose) -> list[str]:
    """The airfoils whose surfaces cross at the nose shown, and those whose fit strays at the ingest's order."""
    crossing = [f"{r.name} ({_orders([f.order for f in r.fits if f.nose == shown and f.crossed])})"
                for r in sweep.unique if any(f.nose == shown and f.crossed for f in r.fits)]
    out = [f"Surfaces cross at nose {shown.value}: {len(crossing)}" + (f": {', '.join(crossing)}" if crossing else "")]
    if CST_ORDER in sweep.orders:
        strays = [r.name for r in sweep.unique for f in r.fits if (f.nose, f.order) == (shown, CST_ORDER) and f.strays]
        out.append(f"Strays at order {CST_ORDER}, nose {shown.value} (the ingest's order): {len(strays)}"
                   + (f": {', '.join(strays)}" if strays else ""))
    return out + [""]


def Geom_calibration_digest(rows: list[CalibrationRow], folder, default_nose: Nose = CST_NOSE) -> str:
    """
    The calibration as a log shows it: the ingest's configuration and its neighbours (an order either side
    at its nose, its order at the other noses) at the fit tolerance, then the edge cases. Around the
    calibrated order nearest the ingest's if its own was not calibrated.

    Args:
        rows (list[CalibrationRow]): `Geom_calibrate`'s output
        folder (str | Path): The folder it ran on, for the header
        default_nose (Nose): The ingest's nose

    Returns:
        str: The digest's lines
    """
    sweep = _sweep(rows)
    out = _head(rows, folder, sweep)
    shown = _shown(sweep, default_nose)
    centre = min(sweep.orders, key=lambda o: (abs(o - CST_ORDER), o))
    picked = [(shown, centre + d) for d in (-1, 0, 1)] + [(n, centre) for n in sweep.noses if n is not shown]
    out.append(f"Tolerance {CST_FIT_TOL:.0e}c, over the {len(sweep.unique)} unique geometries; max_dev [c]; "
               f"formulas = the {len(sweep.families)} formula shapes' worst")
    out.append(f"{'order':>5}  {'nose':<8} {'kept':>5} {'over':>5} {'cross':>5} {'strays':>6} "
               f"{'median':>8} {'p90':>8} {'worst':>8} {'formulas':>9}")
    for nose, order in (c for c in picked if c in sweep.configs):
        c = _config(sweep.unique, nose, order)
        kept = int((c.clean <= CST_FIT_TOL).sum())
        worst = _family_worst(sweep.families, nose, order)
        mark = "  <- the ingest's" if (nose, order) == (Nose(default_nose), CST_ORDER) else ""
        out.append(f"{order:>5}  {nose.value:<8} {kept:>5} {len(c.clean) - kept:>5} {c.crossed:>5} {c.strays:>6} "
                   f"{_spread(c.dev)} {'-' if worst is None else format(worst, '.1e'):>9}{mark}")
    out.append("")
    return "\n".join(out + _edge_cases(sweep, shown)).rstrip("\n")


def Geom_format_calibration(rows: list[CalibrationRow], folder, default_nose: Nose = CST_NOSE) -> str:
    """
    The calibration report: rejections before fitting, a summary per configuration (over unique
    geometries), the formula families' pure representation error, the airfoils whose surfaces cross or whose
    fit strays at the ingest's nose, and every airfoil's error by order.

    Args:
        rows (list[CalibrationRow]): `Geom_calibrate`'s output
        folder (str | Path): The folder it ran on, for the header
        default_nose (Nose): The nose the edge cases and the per-airfoil table show

    Returns:
        str: The report text
    """
    sweep = _sweep(rows)
    out = _head(rows, folder, sweep)
    out.append("Per configuration, unique geometries. max_dev = worst point-to-shape distance [c]; "
               "crossed = surfaces cross; strays = leaves a spline through the data, midway between points, "
               f"by more than twice its max_dev and {_STRAY_MIN:g}c. Noses: separate = one A_0 per surface "
               "(two radii), shared = one A_0, lem = one A_0 plus Kulfan's LE term.")
    out.append(f"{'order':>5} {'nose':>8} {'fitted':>6} " + " ".join(f"{'<=' + format(t, 'g'):>7}" for t in _TOLS)
               + f" {'median':>8} {'p90':>8} {'max':>8} {'crossed':>7} {'strays':>6}")
    for nose, order in sweep.configs:
        c = _config(sweep.unique, nose, order)
        passes = " ".join(f"{int((c.clean <= t).sum()):>7d}" for t in _TOLS)
        out.append(f"{order:>5} {nose.value:>8} {len(c.dev):>6} {passes} {_spread(c.dev)} {c.crossed:>7} {c.strays:>6}")
    out.append("")

    if sweep.families:
        out.append(f"Formula families only ({len(sweep.families)} shapes, pure representation error): "
                   "worst max_dev [c]")
        out.append(f"{'order':>5} " + " ".join(f"{n.value:>9}" for n in sweep.noses))
        for order in sweep.orders:
            out.append(f"{order:>5} " + " ".join(f"{_family_worst(sweep.families, n, order):9.1e}" for n in sweep.noses))
        out.append("")

    shown = _shown(sweep, default_nose)
    out += _edge_cases(sweep, shown)
    out.append(f"Per airfoil, max_dev [1e-4 c] by order, nose {shown.value} "
               "(x: surfaces cross, s: strays, -: too few points):")
    out.append(f"{'source':40s} {'pts':>5} " + " ".join(f"{o:>7}" for o in sweep.orders))
    for r in sweep.fitted:
        cells = []
        for o in sweep.orders:
            f = next(f for f in r.fits if (f.nose, f.order) == (shown, o))
            mark = ("x" if f.crossed else "") + ("s" if f.strays else "")
            cells.append(f"{'-':>7}" if f.max_dev is None else f"{f.max_dev * 1e4:5.1f}{mark:<2s}")
        out.append(f"{r.source:40s} {r.n_fitted:>5} " + " ".join(cells))
    return "\n".join(out) + "\n"


def _orders(orders: list[int]) -> str:
    """[6, 10, 11, 12] -> 'orders 6, 10-12'."""
    runs, start = [], orders[0]
    for prev, cur in zip(orders, orders[1:] + [None]):
        if cur is None or cur != prev + 1:
            runs.append(str(start) if start == prev else f"{start}-{prev}")
            start = cur
    return ("order " if len(orders) == 1 else "orders ") + ", ".join(runs)


def Geom_run_calibration(folder, out, orders=range(4, 13), workers: Optional[int] = None) -> str:
    """
    The calibration of `folder` at `orders`, its whole report written to `out`.

    Args:
        folder (str | Path): A root folder of .dat files
        out (str | Path): The report file; its folder is made if missing
        orders (Iterable[int]): CST orders to fit
        workers (int | None): Parallel processes, one per core by default

    Returns:
        str: Its digest, as a log shows it (`Geom_calibration_digest`)
    """
    rows = Geom_calibrate(folder, orders, workers=workers)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(Geom_format_calibration(rows, folder), encoding="utf-8", newline="\n")
    return Geom_calibration_digest(rows, folder)
