"""
Tests for `meshing.optimizer.space`, the knobs NOMAD searches, i.e.:

Step 1: Ensure the table sits inside the config's own bounds, and each rung moves the right knobs
    - test_every_bound_is_a_valid_config
    - test_smoothing_knobs_are_bounded_as_the_config_bounds_them
    - test_rungs_move_the_agreed_knobs
Step 2: Ensure a point survives the trip to NOMAD and back
    - test_values_round_trip
    - test_from_x_rounds_integers_and_clamps_into_bounds
    - test_nomad_bounds_are_log10_for_log_knobs
    - test_space_json_is_the_whole_table
"""

import json
import math

import pytest

from src.datagen.geometry.common.constants import DX_MAX
from src.datagen.meshing.c2d.schemas import C2D_MeshingConfig
from src.datagen.meshing.optimizer.constants import GOLDEN_JMAX, RATIO_MIN, SMOOTHING_DEFAULTS
from src.datagen.meshing.optimizer.space import (
    KNOBS, from_x, input_types, knobs_for, nomad_bounds, space_json, to_x,
)

_SMOOTHING = ["asmt", "epsi", "funi", "alfa"]


def _bound(name, which):
    """The `ge`/`le` pydantic puts on a C2D_MeshingConfig field."""
    return next(getattr(m, which) for m in C2D_MeshingConfig.model_fields[name].metadata if hasattr(m, which))


# region Step 1
def test_every_bound_is_a_valid_config():
    # Both corners of the box validate, so NOMAD can never propose a config pydantic rejects
    C2D_MeshingConfig(**{k.name: k.lo for k in KNOBS})
    C2D_MeshingConfig(**{k.name: k.hi for k in KNOBS})
    assert all(k.lo < k.hi for k in KNOBS)
    assert all(isinstance(k.lo, int) and isinstance(k.hi, int) for k in KNOBS if k.type == "I")


def test_smoothing_knobs_are_bounded_as_the_config_bounds_them():
    knobs = {k.name: k for k in KNOBS}

    for name in _SMOOTHING:
        assert (knobs[name].lo, knobs[name].hi) == (_bound(name, "ge"), _bound(name, "le"))


def test_rungs_move_the_agreed_knobs():
    assert [k.name for k in knobs_for(2)] == _SMOOTHING
    assert [k.name for k in knobs_for(3)] == [k.name for k in KNOBS]
    assert len(KNOBS) == 9 and "epse" not in {k.name for k in KNOBS}  # pinned off
    assert input_types(knobs_for(2)) == "BB_INPUT_TYPE ( I R R R )"
    with pytest.raises(ValueError, match="not a NOMAD search"):
        knobs_for(1)  # rung 1 steps jmax without NOMAD
# endregion


# region Step 2
def test_values_round_trip():
    # The golden point: n*'s usual landing plus the smoothing defaults
    values = {"dx_mid": DX_MAX, "le_ratio": 0.025, "te_ratio": 1.0, "cluster_width": 0.026,
              "jmax": GOLDEN_JMAX, **SMOOTHING_DEFAULTS}
    back = from_x(to_x(values, KNOBS), KNOBS)

    assert back == pytest.approx(values, rel=1e-12)
    assert isinstance(back["jmax"], int) and isinstance(back["asmt"], int)


def test_from_x_rounds_integers_and_clamps_into_bounds():
    knobs = knobs_for(3)
    at = {k.name: i for i, k in enumerate(knobs)}
    x = nomad_bounds(knobs)[0]
    x[at["jmax"]] = 146.6
    x[at["asmt"]] = 4.4                            # rounds to 4, below its floor
    x[at["le_ratio"]] = 1e-13                      # 10**x is just past 1
    x[at["epsi"]] = math.log10(40.0) + 1e-13

    values = from_x(x, knobs)

    assert (values["jmax"], values["asmt"]) == (147, 5)
    assert (values["le_ratio"], values["epsi"]) == (1.0, 40.0)
    C2D_MeshingConfig(**values)  # still a valid config


def test_nomad_bounds_are_log10_for_log_knobs():
    knobs = {k.name: i for i, k in enumerate(knobs_for(3))}
    lb, ub = nomad_bounds(knobs_for(3))

    assert (lb[knobs["le_ratio"]], ub[knobs["le_ratio"]]) == pytest.approx((math.log10(RATIO_MIN), 0.0))
    assert (lb[knobs["dx_mid"]], ub[knobs["dx_mid"]]) == (1e-3, DX_MAX)  # linear, as given
    assert (lb[knobs["jmax"]], ub[knobs["jmax"]]) == (120.0, 200.0)


def test_space_json_is_the_whole_table():
    table = json.loads(space_json())

    assert [row["name"] for row in table] == [k.name for k in KNOBS]
    assert table[0] == {"name": "dx_mid", "type": "R", "lo": 1e-3, "hi": DX_MAX, "scale": "lin", "rungs": [3]}
    assert space_json() == space_json()
# endregion
