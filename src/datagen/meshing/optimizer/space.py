"""
The mesh study's search space: the config fields NOMAD moves, their bounds and scales, and the rungs that
move them. NOMAD sees log10 of a log knob, so a ratio spanning 500x is searched evenly across it.

    knobs_for     the knobs a rung searches
    to_x, from_x  a knob dict to NOMAD's point and back (undoing the log, rounding integers, clamping)
    nomad_bounds  NOMAD's lower and upper bounds
    input_types   NOMAD's BB_INPUT_TYPE line
    space_json    the whole table, for the settings row
"""

import json
import math
from typing import Literal, NamedTuple, Sequence

from src.datagen.geometry.common.constants import DX_MAX
from src.datagen.meshing.optimizer.constants import (
    CLUSTER_WIDTH_MAX, CLUSTER_WIDTH_MIN, DX_MID_MIN, JMAX_MAX, JMAX_MIN, RATIO_MIN,
)


class Knob(NamedTuple):
    """One searched config field: NOMAD's type ('R' real, 'I' integer), bounds, scale, and the rungs moving it."""
    name: str
    type: Literal["R", "I"]
    lo: float
    hi: float
    scale: Literal["lin", "log"]
    rungs: tuple[int, ...]


# The sampling knobs and jmax move only in rung 3; the smoothing knobs, bounded as C2D_MeshingConfig bounds
# them, in rungs 2 and 3. epse is pinned at 0, so absent
KNOBS = (
    Knob("dx_mid", "R", DX_MID_MIN, DX_MAX, "lin", (3,)),
    Knob("le_ratio", "R", RATIO_MIN, 1.0, "log", (3,)),
    Knob("te_ratio", "R", RATIO_MIN, 1.0, "log", (3,)),  # above a round TE's cap limit it is clamped: a plateau
    Knob("cluster_width", "R", CLUSTER_WIDTH_MIN, CLUSTER_WIDTH_MAX, "lin", (3,)),
    Knob("jmax", "I", JMAX_MIN, JMAX_MAX, "lin", (3,)),
    Knob("asmt", "I", 5, 80, "lin", (2, 3)),
    Knob("epsi", "R", 2.0, 40.0, "log", (2, 3)),
    Knob("funi", "R", 0.005, 0.3, "log", (2, 3)),
    Knob("alfa", "R", 0.5, 2.0, "lin", (2, 3)),
)


def knobs_for(rung: int) -> tuple[Knob, ...]:
    """
    The knobs a NOMAD rung searches, in table order.

    Args:
        rung (int): 2 (smoothing only) or 3 (every knob)

    Returns:
        tuple[Knob, ...]: Its knobs

    Raises:
        ValueError: For any other rung; rung 1 steps jmax without NOMAD
    """
    if rung not in (2, 3):
        raise ValueError(f"rung {rung} is not a NOMAD search; only 2 and 3 are")
    return tuple(k for k in KNOBS if rung in k.rungs)


def to_x(values: dict, knobs: Sequence[Knob]) -> list[float]:
    """
    A knob dict as NOMAD's point.

    Args:
        values (dict): A value per knob, by name; other keys are ignored
        knobs (Sequence[Knob]): The knobs, in the point's order

    Returns:
        list[float]: The point, log10 for log knobs
    """
    return [math.log10(values[k.name]) if k.scale == "log" else float(values[k.name]) for k in knobs]


def from_x(x: Sequence[float], knobs: Sequence[Knob]) -> dict:
    """
    NOMAD's point back as config values: the log undone, integers rounded, then clamped into [lo, hi],
    since the log's round trip can step just past a bound.

    Args:
        x (Sequence[float]): The point, in the knobs' order
        knobs (Sequence[Knob]): The knobs

    Returns:
        dict: A value per knob, by name
    """
    values = {}
    for v, k in zip(x, knobs, strict=True):
        v = 10.0 ** v if k.scale == "log" else float(v)
        v = int(round(v)) if k.type == "I" else v
        values[k.name] = min(max(v, k.lo), k.hi)
    return values


def nomad_bounds(knobs: Sequence[Knob]) -> tuple[list[float], list[float]]:
    """NOMAD's lower and upper bounds for `knobs`, log10 for log knobs."""
    lb = to_x({k.name: k.lo for k in knobs}, knobs)
    ub = to_x({k.name: k.hi for k in knobs}, knobs)
    return lb, ub


def input_types(knobs: Sequence[Knob]) -> str:
    """NOMAD's parameter line typing each coordinate, e.g. "BB_INPUT_TYPE ( R R I )"."""
    return f"BB_INPUT_TYPE ( {' '.join(k.type for k in knobs)} )"


def space_json() -> str:
    """The whole knob table as sorted JSON, recorded in the settings row."""
    return json.dumps([k._asdict() for k in KNOBS], sort_keys=True)
