"""
The mesh study's n*: the fewest surface points the sampler accepts for an airfoil, found by sampling
alone (nothing is meshed). The ladder's golden run meshes at n*'s knobs.

    n_star_search  the grid search
    NStarResult    n*, its knobs, the samples taken and the time

The grid: le_ratio log-spaced over [RATIO_MIN, 1] x cluster_width over its range, at dx_mid = dx_max and
te_ratio 1 (the coarsest a round TE's caps allow; the sampler clamps it), each sampled under the golden
run's point ceiling. The fewest points wins; among equal counts the finest nose, then the narrowest
cluster. Nothing finer follows: n* sits on the turn check's edge, so the turn limit, not a fine-tuned
le_ratio, sets how closely the points follow the shape.
"""

import json
import math
import time
from typing import NamedTuple, Optional

from src.datagen.geometry.sample.contour import Contour
from src.datagen.geometry.sample.fields import SamplingFields
from src.datagen.geometry.sample.sample import Geom_sample
from src.datagen.meshing.optimizer.constants import CLUSTER_WIDTH_MAX, CLUSTER_WIDTH_MIN, RATIO_MIN


class NStarResult(NamedTuple):
    """The search's outcome; `n` and `knobs` are None when the sampler accepted no grid point."""
    n: Optional[int]        # surface points at the knobs
    knobs: Optional[dict]   # dx_mid, le_ratio, te_ratio, cluster_width
    evals: int              # samples taken
    seconds: float


def n_star_search(contour: Contour, settings: dict) -> NStarResult:
    """
    Samples the grid and keeps the accepted point with the fewest surface points.

    Args:
        contour (Contour): The study's contour, from `sampling.build_study_contour`
        settings (dict): The settings row: the grid sizes (`nstar_json`), `dx_max`, `cell_ceiling` and
            `golden_jmax`

    Returns:
        NStarResult: n* and its knobs, or None for both if no grid point was accepted

    Raises:
        SamplerAssertionError: If the sampler broke a rule it guarantees (a bug)
    """
    grid = json.loads(settings["nstar_json"])
    max_points = settings["cell_ceiling"] // settings["golden_jmax"]
    t0 = time.perf_counter()
    best, evals = None, 0
    for le_ratio in _grid(RATIO_MIN, 1.0, grid["le_grid"], log=True):
        for cluster_width in _grid(CLUSTER_WIDTH_MIN, CLUSTER_WIDTH_MAX, grid["cw_grid"]):
            knobs = dict(dx_mid=settings["dx_max"], le_ratio=le_ratio, te_ratio=1.0, cluster_width=cluster_width)
            report = Geom_sample(contour, SamplingFields(**knobs), max_points=max_points).report
            evals += 1
            if report.rejection is None and (best is None or report.n_required < best[0]):
                best = (report.n_required, knobs)
    n, knobs = best or (None, None)
    return NStarResult(n=n, knobs=knobs, evals=evals, seconds=time.perf_counter() - t0)


def _grid(lo: float, hi: float, count: int, log: bool = False) -> list[float]:
    """`count` values from lo to hi, evenly spaced (in log10 if `log`); the ends are the bounds exactly."""
    if count == 1:
        return [lo]
    a, b = (math.log10(lo), math.log10(hi)) if log else (lo, hi)
    inner = (a + i * (b - a) / (count - 1) for i in range(1, count - 1))
    return [lo, *(10.0 ** v if log else v for v in inner), hi]
