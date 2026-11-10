"""
The mesh study's search space: the config fields NOMAD moves, their bounds and scales, and the rungs that
move them. NOMAD sees log10 of a log knob, so a ratio spanning 500x is searched evenly across it.

    Optim_space               the knob table from its bounds (src/datagen/constants.py's, unless a run sets them)
    Optim_knobs_of            the table a settings row recorded
    Optim_knobs_for           the knobs a rung searches
    Optim_to_x, Optim_from_x  a knob dict to NOMAD's point and back (undoing the log, rounding integers,
                              clamping)
    Optim_nomad_bounds        NOMAD's lower and upper bounds
    Optim_input_types         NOMAD's BB_INPUT_TYPE line
    Optim_space_json          the whole table, for the settings row
"""

import json
import math
from typing import Literal, NamedTuple, Optional, Sequence

from src.datagen.constants import (
    CLUSTER_WIDTH_MAX, CLUSTER_WIDTH_MIN, DX_MAX, DX_MID_MIN, JMAX_MAX, JMAX_MIN, RATIO_MIN,
)


class Knob(NamedTuple):
    """One searched config field: NOMAD's type ('R' real, 'I' integer), bounds, scale, and the rungs moving it."""
    name: str
    type: Literal["R", "I"]
    lo: float
    hi: float
    scale: Literal["lin", "log"]
    rungs: tuple[int, ...]


def Optim_space(dx_mid_min: float = DX_MID_MIN, dx_max: float = DX_MAX, ratio_min: float = RATIO_MIN,
                cluster_width_min: float = CLUSTER_WIDTH_MIN, cluster_width_max: float = CLUSTER_WIDTH_MAX,
                jmax_min: int = JMAX_MIN, jmax_max: int = JMAX_MAX) -> tuple[Knob, ...]:
    """
    The knob table. The sampling knobs and jmax move only in rung 3; the smoothing knobs, bounded as
    C2D_MeshingConfig bounds them, in rungs 2 and 3. epse is pinned at 0, so absent.

    Raises:
        ValueError: If a range is empty
    """
    knobs = (
        Knob("dx_mid", "R", dx_mid_min, dx_max, "lin", (3,)),
        Knob("le_ratio", "R", ratio_min, 1.0, "log", (3,)),
        Knob("te_ratio", "R", ratio_min, 1.0, "log", (3,)),  # above a round TE's cap limit it is clamped: a plateau
        Knob("cluster_width", "R", cluster_width_min, cluster_width_max, "lin", (3,)),
        Knob("jmax", "I", jmax_min, jmax_max, "lin", (3,)),
        Knob("asmt", "I", 5, 80, "lin", (2, 3)),
        Knob("epsi", "R", 2.0, 40.0, "log", (2, 3)),
        Knob("funi", "R", 0.005, 0.3, "log", (2, 3)),
        Knob("alfa", "R", 0.5, 2.0, "lin", (2, 3)),
    )
    empty = [k.name for k in knobs if not k.lo < k.hi]
    if empty:
        raise ValueError(f"empty search range for {', '.join(empty)}")
    return knobs


# The constants' table
KNOBS = Optim_space()


def Optim_knobs_of(settings: dict) -> tuple[Knob, ...]:
    """The knob table a settings row recorded (`search_space_json`), in table order."""
    rows = {row["name"]: row for row in json.loads(settings["search_space_json"])}
    return tuple(Knob(**{**rows[k.name], "rungs": tuple(rows[k.name]["rungs"])}) for k in KNOBS)


def Optim_knobs_for(rung: int, settings: Optional[dict] = None) -> tuple[Knob, ...]:
    """
    The knobs a NOMAD rung searches, in table order.

    Args:
        rung (int): 2 (smoothing only) or 3 (every knob)
        settings (dict | None): The settings row whose table to take; the constants' if None

    Returns:
        tuple[Knob, ...]: Its knobs

    Raises:
        ValueError: For any other rung; rung 1 steps jmax without NOMAD
    """
    if rung not in (2, 3):
        raise ValueError(f"rung {rung} is not a NOMAD search; only 2 and 3 are")
    return tuple(k for k in (KNOBS if settings is None else Optim_knobs_of(settings)) if rung in k.rungs)


def Optim_to_x(values: dict, knobs: Sequence[Knob]) -> list[float]:
    """
    A knob dict as NOMAD's point.

    Args:
        values (dict): A value per knob, by name; other keys are ignored
        knobs (Sequence[Knob]): The knobs, in the point's order

    Returns:
        list[float]: The point, log10 for log knobs
    """
    return [math.log10(values[k.name]) if k.scale == "log" else float(values[k.name]) for k in knobs]


def Optim_from_x(x: Sequence[float], knobs: Sequence[Knob]) -> dict:
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


def Optim_nomad_bounds(knobs: Sequence[Knob]) -> tuple[list[float], list[float]]:
    """NOMAD's lower and upper bounds for `knobs`, log10 for log knobs."""
    lb = Optim_to_x({k.name: k.lo for k in knobs}, knobs)
    ub = Optim_to_x({k.name: k.hi for k in knobs}, knobs)
    return lb, ub


def Optim_input_types(knobs: Sequence[Knob]) -> str:
    """NOMAD's parameter line typing each coordinate, e.g. "BB_INPUT_TYPE ( R R I )"."""
    return f"BB_INPUT_TYPE ( {' '.join(k.type for k in knobs)} )"


def Optim_space_json(knobs: Sequence[Knob] = KNOBS) -> str:
    """The whole knob table as sorted JSON, recorded in the settings row."""
    return json.dumps([k._asdict() for k in knobs], sort_keys=True)
