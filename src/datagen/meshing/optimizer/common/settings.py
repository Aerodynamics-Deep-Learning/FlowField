"""
The study's settings row: every value a run is meshed and judged by, plus the mesher's own hash, under one
id. Two runs share a settings_id exactly when none of them differs.

    Optim_build_settings    the row, with the tunable columns optionally overridden
    Optim_shape_version     the stored shapes a run's settings study
    Optim_mesher_sha256     the hash of the c2d executable in use
    Optim_pinned            every config field the search leaves alone, with its value
    Optim_config_schema_fp  a fingerprint of C2D_MeshingConfig, recorded on every run
"""

import hashlib
import json

from src.datagen.constants import (
    CELL_CEILING, CLUSTER_WIDTH_MAX, CLUSTER_WIDTH_MIN, CST_NOSE, CST_ORDER, DESIGN_RE, DX_MAX, DX_MID_MIN,
    FARFIELD_RADIUS, GATE_AR_MAX, GATE_GEO_DEV_MAX, GATE_JAC_MIN, GATE_ORTHO_MIN, GATE_SIZE_RATIO_MAX, GATE_SKEW_MAX,
    GOLDEN_JMAX, JMAX_BUMP_MAX, JMAX_BUMP_STEP, JMAX_MAX, JMAX_MIN, NOMAD_SEED, NSTAR_CW_GRID, NSTAR_LE_GRID,
    RATIO_MIN, RUNG2_BUDGET, RUNG3_BUDGET, RUNG3_MIN_GAIN, RUNG3_PATIENCE, SMOOTHING_DEFAULTS, TE_GAP, THETA_MAX_DEG,
    Y_PLUS,
)
from src.datagen.geometry.common.constants import PRECHECK_MARGIN, SAMPLER_VERSION, Geom_shape_version
from src.datagen.geometry.common.schemas import Nose
from src.datagen.meshing.c2d.schemas import C2D_MeshingConfig
from src.datagen.meshing.optimizer.common.constants import BACKEND, LADDER_VERSION, MAX_BB_EVAL_FACTOR, PINNED, TOPOLOGY
from src.datagen.meshing.optimizer.common.space import KNOBS, Optim_space, Optim_space_json
from src.datagen.utils.tools import find_tool

# The search space's bounds, folding into search_space_json
_SPACE = {"dx_mid_min": DX_MID_MIN, "ratio_min": RATIO_MIN, "cluster_width_min": CLUSTER_WIDTH_MIN,
          "cluster_width_max": CLUSTER_WIDTH_MAX, "jmax_min": JMAX_MIN, "jmax_max": JMAX_MAX}

# What the run code reads from the row, each by its constant's name: the CST order and nose fold into shape_version,
# the n* grid's two into nstar_json, the space's bounds into search_space_json, y+ and the farfield radius into
# pinned_json, the smoothing defaults into smoothing_json. The rest describes the code or the executable, so
# overriding it would record settings the run did not use. Checked per (backend, topology) once other meshers join
OVERRIDABLE = (
    "cst_order", "cst_nose", "design_re", "te_gap", "cell_ceiling", "dx_max", "theta_max_deg", "gate_skew_max",
    "gate_ortho_min", "gate_jac_min", "gate_ar_max", "gate_size_ratio_max", "gate_geo_dev_max", "golden_jmax",
    "jmax_bump_step", "jmax_bump_max", "rung2_budget", "rung3_budget", "rung3_patience", "rung3_min_gain",
    "max_bb_eval_factor", "nomad_seed", "nstar_le_grid", "nstar_cw_grid", *_SPACE, "y_plus", "farfield_radius",
    "smoothing_defaults",
)

# The ones in src/datagen/constants.py, which a run's settings can set
STUDY_KEYS = frozenset(OVERRIDABLE) - {"max_bb_eval_factor"}


def Optim_build_settings(**overrides) -> dict:
    """
    The settings row, every column but `created_at` and `label` (neither is hashed), `settings_id` last:
    the first 16 hex of the SHA-256 over the rest.

    Args:
        **overrides: New values for columns in `OVERRIDABLE`; a float column takes an int, an int column
            only an int, so one setting cannot hash two ways

    Returns:
        dict: Column name to value

    Raises:
        ValueError: If an override names a column outside `OVERRIDABLE`, leaves a search range empty, sets a
            smoothing default outside its knob's bounds, or names no nose
        TypeError: If an int column's override is not an int
        FileNotFoundError: If no c2d executable is configured
    """
    unknown = sorted(set(overrides) - set(OVERRIDABLE))
    if unknown:
        raise ValueError(f"cannot override {', '.join(unknown)}; only {', '.join(OVERRIDABLE)}")
    overrides = dict(overrides)

    def take(key, default):
        return _checked(key, default, overrides.pop(key)) if key in overrides else default

    shape_version = Optim_shape_version({"cst_order": take("cst_order", CST_ORDER),
                                         "cst_nose": overrides.pop("cst_nose", CST_NOSE)})
    grid = {"le_grid": take("nstar_le_grid", NSTAR_LE_GRID), "cw_grid": take("nstar_cw_grid", NSTAR_CW_GRID)}
    dx_max = take("dx_max", DX_MAX)
    space = Optim_space(dx_max=dx_max, **{key: take(key, default) for key, default in _SPACE.items()})
    pinned = Optim_pinned(take("y_plus", Y_PLUS), take("farfield_radius", FARFIELD_RADIUS))
    smoothing = _smoothing(overrides.pop("smoothing_defaults", {}))

    values = {
        "ladder_version": LADDER_VERSION,
        "backend": BACKEND,
        "topology": TOPOLOGY,
        "design_re": DESIGN_RE,
        "shape_version": shape_version,
        "sampler_version": SAMPLER_VERSION,
        "mesher_sha256": Optim_mesher_sha256(),
        "dx_max": dx_max,
        "theta_max_deg": THETA_MAX_DEG,
        "precheck_margin": PRECHECK_MARGIN,
        "te_gap": TE_GAP,
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
        "rung3_patience": RUNG3_PATIENCE,
        "rung3_min_gain": RUNG3_MIN_GAIN,
        "max_bb_eval_factor": MAX_BB_EVAL_FACTOR,
        "nomad_seed": NOMAD_SEED,
        "nstar_json": json.dumps(grid, sort_keys=True),
        "search_space_json": Optim_space_json(space),
        "pinned_json": json.dumps(pinned, sort_keys=True),
        "smoothing_json": json.dumps(smoothing, sort_keys=True),
    }
    for key, value in overrides.items():
        values[key] = _checked(key, values[key], value)
    values["settings_id"] = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:16]
    return values


def Optim_shape_version(settings: dict) -> str:
    """
    The shape version a run's settings study: their `cst_order` and `cst_nose`, else the constants'.

    Raises:
        ValueError: If `cst_nose` names no nose
    """
    nose = getattr(settings.get("cst_nose", CST_NOSE), "value", settings.get("cst_nose", CST_NOSE))
    if nose not in {n.value for n in Nose}:
        raise ValueError(f"cst_nose is one of {', '.join(n.value for n in Nose)}, not {nose!r}")
    return Geom_shape_version(settings.get("cst_order", CST_ORDER), nose)


def _checked(key: str, default, value):
    """An override in its column's type: an int column takes only an int, a float one an int as well."""
    if isinstance(default, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{key} takes an int, got {value!r}")
        return value
    return float(value)


def _smoothing(given: dict) -> dict:
    """The smoothing defaults with a run's over them, each typed and inside its knob's bounds."""
    unknown = sorted(set(given) - set(SMOOTHING_DEFAULTS))
    if unknown:
        raise ValueError(f"smoothing_defaults has no {', '.join(unknown)}; only {', '.join(SMOOTHING_DEFAULTS)}")
    out = {**SMOOTHING_DEFAULTS, **{k: _checked(f"smoothing_defaults' {k}", SMOOTHING_DEFAULTS[k], v)
                                    for k, v in given.items()}}
    outside = [f"{k.name} {out[k.name]:g} not in [{k.lo:g}, {k.hi:g}]" for k in KNOBS
               if k.name in out and not k.lo <= out[k.name] <= k.hi]
    if outside:
        raise ValueError(f"smoothing_defaults outside the search's bounds: {'; '.join(outside)}")
    return out


def Optim_mesher_sha256() -> str:
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


def Optim_pinned(y_plus: float = Y_PLUS, farfield_radius: float = FARFIELD_RADIUS) -> dict:
    """
    Every C2D_MeshingConfig field the search leaves alone: `PINNED`'s value with a run's y+ and farfield radius,
    else the config's default.
    """
    searched = {k.name for k in KNOBS}
    defaults, pinned = C2D_MeshingConfig(), {**PINNED, "ypls": y_plus, "radi": farfield_radius}
    return {name: pinned.get(name, getattr(defaults, name))
            for name in C2D_MeshingConfig.model_fields if name not in searched}


def Optim_config_schema_fp() -> str:
    """The first 16 hex of the SHA-256 of C2D_MeshingConfig's JSON schema: moves with any field, bound or default."""
    schema = json.dumps(C2D_MeshingConfig.model_json_schema(), sort_keys=True)
    return hashlib.sha256(schema.encode()).hexdigest()[:16]
