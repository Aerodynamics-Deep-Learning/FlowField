"""
Calibration of the CST order, nose and fit tolerance: fits every airfoil in a root folder at a range of
orders, with each way of representing the nose, through the same code the ingest uses, and writes the
table the constants are chosen from. Touches no database.

To run calibration:
    `python -m src.datagen.geometry.build.calibrate <folder> [--orders 4-12] [--out <file>] [--workers <n>]`

Files are independent, so they are fitted in parallel processes, one per core by default.
"""

import argparse
import datetime
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel
from scipy.interpolate import CubicSpline

from .add import _fit_and_gate, _prepare
from src.datagen.geometry.common.constants import CST_NOSE, CURVE_POINTS
from .parametrization import Geom_cst_contour, Geom_curve_distance
from src.datagen.geometry.common.paths import Geom_git_version, Geom_paths, Geom_source_label
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


def Geom_format_calibration(rows: list[CalibrationRow], folder, default_nose: Nose = CST_NOSE) -> str:
    """
    The calibration report: rejections before fitting, a summary per configuration (over unique
    geometries), the formula families' pure representation error, and every airfoil's error by order.

    Args:
        rows (list[CalibrationRow]): `Geom_calibrate`'s output
        folder (str | Path): The folder it ran on, for the header
        default_nose (Nose): The nose the per-airfoil table shows

    Returns:
        str: The report text
    """
    fitted = [r for r in rows if r.fits]
    unique = list({r.geometry_id: r for r in fitted}.values())
    configs = sorted({(f.nose, f.order) for r in fitted for f in r.fits}, key=lambda c: (list(Nose).index(c[0]), c[1]))
    orders = sorted({o for _, o in configs})
    noses = list(dict.fromkeys(n for n, _ in configs))
    out = [f"CST calibration of {Geom_source_label(folder)}",
           f"{datetime.date.today().isoformat()}, code {Geom_git_version()}",
           f"{len(rows)} files: {len(fitted)} fitted ({len(unique)} unique geometries), "
           f"{len(rows) - len(fitted)} rejected before fitting", ""]

    if len(rows) > len(fitted):
        out.append("Rejected before fitting:")
        out += [f"  {r.source:40s} {r.reject.value:16s} {r.reason}" for r in rows if not r.fits]
        out.append("")

    out.append("Per configuration, unique geometries. max_dev = worst point-to-shape distance [c]; "
               "crossed = surfaces cross; strays = leaves a spline through the data, midway between points, "
               f"by more than twice its max_dev and {_STRAY_MIN:g}c. Noses: separate = one A_0 per surface "
               "(two radii), shared = one A_0, lem = one A_0 plus Kulfan's LE term.")
    out.append(f"{'order':>5} {'nose':>8} {'fitted':>6} " + " ".join(f"{'<=' + format(t, 'g'):>7}" for t in _TOLS)
               + f" {'median':>8} {'p90':>8} {'max':>8} {'crossed':>7} {'strays':>6}")
    for nose, order in configs:
        fits = [f for r in unique for f in r.fits if (f.nose, f.order) == (nose, order)]
        dev = np.array([f.max_dev for f in fits if f.max_dev is not None])
        ok = [f for f in fits if f.max_dev is not None and not f.crossed]
        passes = " ".join(f"{sum(f.max_dev <= t for f in ok):>7d}" for t in _TOLS)
        spread = (f"{np.median(dev):8.1e} {np.percentile(dev, 90):8.1e} {dev.max():8.1e}" if len(dev)
                  else f"{'-':>8} {'-':>8} {'-':>8}")
        out.append(f"{order:>5} {nose.value:>8} {len(dev):>6} {passes} {spread} "
                   f"{sum(f.crossed for f in fits):>7} {sum(f.strays for f in fits):>6}")
    out.append("")

    families = [r for r in unique if r.family is not None]
    if families:
        out.append(f"Formula families only ({len(families)} shapes, pure representation error): worst max_dev [c]")
        out.append(f"{'order':>5} " + " ".join(f"{n.value:>9}" for n in noses))
        for order in orders:
            worst = [max(f.max_dev for r in families for f in r.fits if (f.nose, f.order) == (n, order))
                     for n in noses]
            out.append(f"{order:>5} " + " ".join(f"{w:9.1e}" for w in worst))
        out.append("")

    shown = Nose(default_nose) if Nose(default_nose) in noses else noses[0]
    out.append(f"Per airfoil, max_dev [1e-4 c] by order, nose {shown.value} "
               "(x: surfaces cross, s: strays, -: too few points):")
    out.append(f"{'source':40s} {'pts':>5} " + " ".join(f"{o:>7}" for o in orders))
    for r in fitted:
        cells = []
        for o in orders:
            f = next(f for f in r.fits if (f.nose, f.order) == (shown, o))
            mark = ("x" if f.crossed else "") + ("s" if f.strays else "")
            cells.append(f"{'-':>7}" if f.max_dev is None else f"{f.max_dev * 1e4:5.1f}{mark:<2s}")
        out.append(f"{r.source:40s} {r.n_fitted:>5} " + " ".join(cells))
    return "\n".join(out) + "\n"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="root folder of .dat files")
    parser.add_argument("--orders", default="4-12", help="CST orders, 'lo-hi' (default 4-12)")
    parser.add_argument("--out", type=Path, help="report file (default: the calibration folder under data/)")
    parser.add_argument("--workers", type=int, help="parallel processes (default: one per core)")
    args = parser.parse_args(argv)

    lo, hi = (int(v) for v in args.orders.split("-"))
    out = args.out or Geom_paths().calibration_dir / f"{datetime.date.today().isoformat()}.txt"
    text = Geom_format_calibration(Geom_calibrate(args.folder, range(lo, hi + 1), workers=args.workers), args.folder)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    print(text)
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
