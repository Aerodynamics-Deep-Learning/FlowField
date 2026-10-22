"""
Inputs for the sampling tests: stored shapes standing in for the geometry manifest (two NACA formula
sections run through the ingest's own normalize/close/fit, and a cusp with TE wedge 0), and random
sampling configs inside the optimizer's search space.
"""

import math
from functools import cache

import numpy as np

from src.datagen.geometry.build.families import Geom_family_coords
from src.datagen.geometry.build.normalize import Geom_close_te, Geom_normalize
from src.datagen.geometry.build.parametrization import Geom_cst_fit, Geom_cst_te_wedge
from src.datagen.geometry.common.constants import CST_N1, CST_N2, DX_MAX, SHAPE_VERSION
from src.datagen.geometry.common.schemas import Closure, CSTParams, Family, ShapeRecord
from src.datagen.geometry.sample.fields import SamplingFields

SHAPE_NAMES = ("naca0012", "naca4412", "cusp")

CUSP = CSTParams(n1=CST_N1, n2=CST_N2, le_weight=0.02,
                 upper=[0.17, 0.15, 0.20, 0.18, 0.16, 0.20, 0.19, 0.12, 0.05],
                 lower=[-0.17, -0.10, -0.08, -0.05, -0.06, -0.02, 0.01, 0.03, 0.05])


def as_record(name: str, cst: CSTParams) -> ShapeRecord:
    """A manifest row for a CST, under its name as geometry_id."""
    return ShapeRecord(geometry_id=name, shape_version=SHAPE_VERSION, cst=cst, rotation_deg=0.0,
                       closure=Closure.FORMULA, root_te_gap=0.0, source_dev=0.0, fit_max=0.0,
                       te_wedge_deg=Geom_cst_te_wedge(cst), norm_path=f"normalised/{name}.dat", norm_hash="0" * 64)


@cache
def stored_shape(name: str) -> ShapeRecord:
    """One of `SHAPE_NAMES`."""
    if name == "cusp":
        return as_record(name, CUSP)
    digits = {"naca0012": {"m": 0, "p": 0, "t": 12}, "naca4412": {"m": 4, "p": 4, "t": 12}}[name]
    pts, tf = Geom_normalize(Geom_family_coords(Family.NACA4, digits))
    pts, _ = Geom_close_te(pts, tf.le_index)
    return as_record(name, Geom_cst_fit(pts, tf.le_index))


def search_space_configs(rng: np.random.Generator, k: int) -> list[SamplingFields]:
    """Random configs inside the optimizer's bounds, the ratios log-uniform as in its search space."""
    log_ratio = (math.log(0.002), 0.0)
    return [SamplingFields(dx_mid=rng.uniform(1e-3, DX_MAX), le_ratio=math.exp(rng.uniform(*log_ratio)),
                           te_ratio=math.exp(rng.uniform(*log_ratio)), cluster_width=rng.uniform(0.026, 0.25))
            for _ in range(k)]
