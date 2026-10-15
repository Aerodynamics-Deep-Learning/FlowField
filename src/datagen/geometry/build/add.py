"""
One root .dat file -> one result: Is per-airfoil work. Pure, writes nothing; deduplication
and storage are the ingest's.

Steps: read, identify, take the points (the formula for a family, else the file's), normalize, close,
check a family against its file, fit, measure the fit against the source as it was before closing,
gate, render the normalised contour. A fit the gates reject is rendered too, so it can be looked at.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .checks import Geom_check_family_match, Geom_check_fit, Geom_check_thickness
from src.datagen.geometry.common.constants import CST_FIT_TOL, CST_NOSE, CST_ORDER, NORM_N_POINTS
from .families import Geom_family_coords, Geom_family_id, Geom_parse_family
from src.datagen.geometry.common.io import GeometryFormatError, Geom_as_written, Geom_coords_hash, Geom_read_dat
from .normalize import Geom_close_te, Geom_normalize
from .parametrization import Geom_cst_contour, Geom_cst_fit, Geom_cst_fit_metrics, Geom_cst_te_wedge
from src.datagen.geometry.common.schemas import AddResult, Closure, Identity, Nose, RejectReason, RootAirfoil


@dataclass(frozen=True)
class _Points:
    """
    What the fit is made to (normalized and closed, LE at the result's `transform.le_index`), and the
    file's own points normalized but not closed, which the fit is measured against too.
    """
    fit: np.ndarray
    source: np.ndarray
    source_le: int


def Geom_identify(path) -> Identity:
    """
    The IDs of a root file, without fitting anything: what the ingest deduplicates by.

    Args:
        path (str | Path): The .dat file

    Returns:
        Identity: Name, raw hash, family and geometry ID

    Raises:
        GeometryFormatError: If the file is not in the root-folder format
    """
    return _identify(Geom_read_dat(path))


def _identify(root: RootAirfoil) -> Identity:
    raw_hash, n = Geom_coords_hash(root.coords), len(root.coords)
    parsed = Geom_parse_family(root.name)
    if parsed is None:
        return Identity(name=root.name, raw_hash=raw_hash, raw_points=n, geometry_id=raw_hash)
    family, params = parsed
    return Identity(name=root.name, raw_hash=raw_hash, raw_points=n, family=family, family_params=params,
                    geometry_id=Geom_family_id(family, params))


def Geom_add_airfoil(path, order: int = CST_ORDER, nose: Nose = CST_NOSE, fit_tol: float = CST_FIT_TOL) -> AddResult:
    """
    Runs one root file through the geometry pipeline.

    Args:
        path (str | Path): The .dat file
        order (int): CST Bernstein order
        nose (Nose): How the CST represents the nose
        fit_tol (float): The fit gate's limit

    Returns:
        AddResult: Accepted (`reject` None) with the shape and its rendering, or rejected with the
            reason and whatever was measured before it (a fit and its rendering, if it got that far)
    """
    res, pts = _prepare(path)
    if not res.accepted:
        return res
    res = _fit_and_gate(res, pts, order, nose, fit_tol)
    if res.cst is not None:
        res.norm_coords = Geom_as_written(Geom_cst_contour(res.cst, NORM_N_POINTS))
        res.norm_hash = Geom_coords_hash(res.norm_coords)
    return res


def _prepare(path) -> tuple[AddResult, Optional[_Points]]:
    """Everything before the fit: read, identify, normalize, close, check a family against its file."""
    res = AddResult(path=str(Path(path)))
    try:
        root = Geom_read_dat(path)
    except GeometryFormatError as e:
        return _reject(res, RejectReason.FORMAT, str(e)), None
    res.identity = _identify(root)

    source, file_tf = Geom_normalize(root.coords)
    file_pts, res.root_te_gap = Geom_close_te(source, file_tf.le_index)

    if res.identity.family is None:
        pts, res.transform = file_pts, file_tf
        res.closure = Closure.NONE if res.root_te_gap == 0.0 else Closure.PRIED
    else:
        pts, res.transform = Geom_normalize(Geom_family_coords(res.identity.family, res.identity.family_params))
        pts, _ = Geom_close_te(pts, res.transform.le_index)
        res.closure = Closure.FORMULA
        ok, res.family_dev, reason = Geom_check_family_match(pts, res.transform.le_index, file_pts, file_tf.le_index)
        if not ok:
            return _reject(res, RejectReason.FAMILY_MISMATCH, reason), None
    return res, _Points(fit=pts, source=source, source_le=file_tf.le_index)


def _fit_and_gate(res: AddResult, pts: _Points, order: int, nose: Nose, fit_tol: float,
                  measure_source: bool = True) -> AddResult:
    """Everything from the fit on, filling `res` in: fit, measure, gate. The calibration skips `source_dev`."""
    le = res.transform.le_index
    fewest = min(le + 1, len(pts.fit) - le)
    if fewest < order + 4:  # end points carry no information; keep one point spare per surface
        return _reject(res, RejectReason.TOO_FEW_POINTS,
                       f"a surface has {fewest} points, order {order} needs at least {order + 4}")

    res.cst = Geom_cst_fit(pts.fit, le, order, nose)
    res.fit = Geom_cst_fit_metrics(res.cst, pts.fit, le)
    if measure_source:
        res.source_dev = Geom_cst_fit_metrics(res.cst, pts.source, pts.source_le).max_dev
    res.te_wedge_deg = Geom_cst_te_wedge(res.cst)
    ok, _, reason = Geom_check_thickness(res.cst)
    if not ok:
        return _reject(res, RejectReason.SURFACES_CROSS, reason)
    ok, _, reason = Geom_check_fit(res.fit, fit_tol)
    if not ok:
        return _reject(res, RejectReason.FIT_TOLERANCE, reason)
    return res


def _reject(res: AddResult, reason: RejectReason, text: str) -> AddResult:
    res.reject, res.reason = reason, text
    return res
