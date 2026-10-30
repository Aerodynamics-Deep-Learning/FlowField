"""
The study's settings row: every value a run is meshed and judged by, plus the mesher's own hash, under one
id. Two runs share a settings_id exactly when none of them differs.

    build_settings    the row, with the tunable columns optionally overridden
    mesher_sha256     the hash of the c2d executable in use
    pinned            every config field the search leaves alone, with its value
    config_schema_fp  a fingerprint of C2D_MeshingConfig, recorded on every run
"""

import hashlib
import json

from src.datagen.geometry.common.constants import (
    DX_MAX, PRECHECK_MARGIN, SAMPLER_VERSION, SHAPE_VERSION, THETA_MAX_DEG,
)
from src.datagen.meshing.c2d.schemas import C2D_MeshingConfig
from src.datagen.meshing.optimizer.constants import (
    BACKEND, CELL_CEILING, DESIGN_RE, GATE_AR_MAX, GATE_GEO_DEV_MAX, GATE_JAC_MIN, GATE_ORTHO_MIN,
    GATE_SIZE_RATIO_MAX, GATE_SKEW_MAX, GOLDEN_JMAX, JMAX_BUMP_MAX, JMAX_BUMP_STEP, LADDER_VERSION,
    MAX_BB_EVAL_FACTOR, NOMAD_SEED, NSTAR_CW_GRID, NSTAR_LE_GRID, PINNED, RUNG2_BUDGET, RUNG3_BUDGET,
    TE_GAP_DEFAULT, TOPOLOGY,
)
from src.datagen.meshing.optimizer.space import KNOBS, space_json
from src.datagen.utils.tools import find_tool

# What the run code reads from the row; the rest describes the code or the executable, so overriding it
# would record settings the run did not use. Checked per (backend, topology) once other meshers join
OVERRIDABLE = (
    "gate_skew_max", "gate_ortho_min", "gate_jac_min", "gate_ar_max", "gate_size_ratio_max",
    "gate_geo_dev_max", "golden_jmax", "jmax_bump_step", "jmax_bump_max", "rung2_budget", "rung3_budget",
    "max_bb_eval_factor", "nomad_seed",
)


def build_settings(**overrides) -> dict:
    """
    The settings row, every column but `created_at` and `label` (neither is hashed), `settings_id` last:
    the first 16 hex of the SHA-256 over the rest.

    Args:
        **overrides: New values for columns in `OVERRIDABLE`; a float column takes an int, an int column
            only an int, so one setting cannot hash two ways

    Returns:
        dict: Column name to value

    Raises:
        ValueError: If an override names a column outside `OVERRIDABLE`
        TypeError: If an int column's override is not an int
        FileNotFoundError: If no c2d executable is configured
    """
    unknown = sorted(set(overrides) - set(OVERRIDABLE))
    if unknown:
        raise ValueError(f"cannot override {', '.join(unknown)}; only {', '.join(OVERRIDABLE)}")

    values = {
        "ladder_version": LADDER_VERSION,
        "backend": BACKEND,
        "topology": TOPOLOGY,
        "design_re": DESIGN_RE,
        "shape_version": SHAPE_VERSION,
        "sampler_version": SAMPLER_VERSION,
        "mesher_sha256": mesher_sha256(),
        "dx_max": DX_MAX,
        "theta_max_deg": THETA_MAX_DEG,
        "precheck_margin": PRECHECK_MARGIN,
        "te_gap": TE_GAP_DEFAULT,
        "cell_ceiling": CELL_CEILING,
        "gate_skew_max": GATE_SKEW_MAX,
        "gate_ortho_min": GATE_ORTHO_MIN,
        "gate_jac_min": GATE_JAC_MIN,
        "gate_ar_max": GATE_AR_MAX,
        "gate_size_ratio_max": GATE_SIZE_RATIO_MAX,
        "gate_geo_dev_max": GATE_GEO_DEV_MAX,
        "golden_jmax": GOLDEN_JMAX,
        "jmax_bump_step": JMAX_BUMP_STEP,
        "jmax_bump_max": JMAX_BUMP_MAX,
        "rung2_budget": RUNG2_BUDGET,
        "rung3_budget": RUNG3_BUDGET,
        "max_bb_eval_factor": MAX_BB_EVAL_FACTOR,
        "nomad_seed": NOMAD_SEED,
        "nstar_json": json.dumps({"le_grid": NSTAR_LE_GRID, "cw_grid": NSTAR_CW_GRID}, sort_keys=True),
        "search_space_json": space_json(),
        "pinned_json": json.dumps(pinned(), sort_keys=True),
    }
    for key, value in overrides.items():
        if isinstance(values[key], int):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{key} takes an int, got {value!r}")
        else:
            value = float(value)
        values[key] = value
    values["settings_id"] = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:16]
    return values


def mesher_sha256() -> str:
    """
    The SHA-256 of the c2d executable `find_tool` resolves, so a rebuilt exe makes new settings.

    Raises:
        FileNotFoundError: If none is configured
    """
    exe = find_tool("c2d")
    if exe is None:
        raise FileNotFoundError("no c2d executable found: set `c2d` in tools.toml, or FLOWFIELD_TOOL_C2D")
    with open(exe, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def pinned() -> dict:
    """Every C2D_MeshingConfig field the search leaves alone: `PINNED`'s value, else the config's default."""
    searched = {k.name for k in KNOBS}
    defaults = C2D_MeshingConfig()
    return {name: PINNED.get(name, getattr(defaults, name))
            for name in C2D_MeshingConfig.model_fields if name not in searched}


def config_schema_fp() -> str:
    """The first 16 hex of the SHA-256 of C2D_MeshingConfig's JSON schema: moves with any field, bound or default."""
    schema = json.dumps(C2D_MeshingConfig.model_json_schema(), sort_keys=True)
    return hashlib.sha256(schema.encode()).hexdigest()[:16]
