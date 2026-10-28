"""
The mesh study's fixed choices in one place: what it meshes, the gates a mesh must pass, the ladder's
steps and budgets, and the search space's bounds. `settings.build_settings` records every one that shapes
or judges a mesh, so a run can always say what it was held to.

Imports only `meshing.common.constants`, which imports nothing.
"""

from src.datagen.meshing.common.constants import ASPECT_RATIO_LIMIT, GEO_DEV_LIMIT, SIZE_RATIO_LIMIT

#region The study

# MeshBackend.C2D's and MeshTopology.OGRD's values
BACKEND = "c2d"
TOPOLOGY = "OGRID"

# The top of Freestream.Re, so one mesh over-resolves every case; the only freestream value c2d reads
DESIGN_RE = 1.5e7

# The pipeline's normalized chord; c2d meshes in unit chord whatever it is told
CHORD = 1.0

# The study's TE gap [c] for a round or blunt TE: a study hyperparameter, recorded with the study
TE_GAP_DEFAULT = 0.004

# Most surface points x jmax a config may ask for; more is rejected before meshing
CELL_CEILING = 300_000

#endregion

#region Gates, on the worst cell

# The study's shape gates, stricter than the entry flag's for now, hence provisionary thresholds specified
GATE_SKEW_MAX = 0.4
GATE_ORTHO_MIN = 0.75
GATE_JAC_MIN = 0.75

# The entry flag's own
GATE_AR_MAX = ASPECT_RATIO_LIMIT
GATE_SIZE_RATIO_MAX = SIZE_RATIO_LIMIT
GATE_GEO_DEV_MAX = GEO_DEV_LIMIT

#endregion

#region The ladder

# naca0012's smallest passing jmax (142) + 5. Provisional: the calibration re-derives it
GOLDEN_JMAX = 147

# Rung 1 raises jmax by this, at most this many times
JMAX_BUMP_STEP = 1
JMAX_BUMP_MAX = 10

# Finished meshes per NOMAD rung: 2 (smoothing only), 3 (all knobs). Provisional: the calibration sets them
RUNG2_BUDGET = 30
RUNG3_BUDGET = 100

# NOMAD's own evaluation cap, as a multiple of the budget; only guards a runaway, since rejections are free
MAX_BB_EVAL_FACTOR = 10

NOMAD_SEED = 1

# Bump by hand on any change to the ladder's flow logic
LADDER_VERSION = "ladder-r1"

#endregion

#region The search space's bounds, see space.py

# Floor on dx_mid. Provisional, pending Emre's
DX_MID_MIN = 1e-3

# Floor on le_ratio and te_ratio
RATIO_MIN = 0.002

# cluster_width's range; 0.026 keeps the neighbour panel ratio <= 1.2. The floor is provisional, pending Emre's
CLUSTER_WIDTH_MIN = 0.026
CLUSTER_WIDTH_MAX = 0.25

# jmax's range; the size-ratio gate needs ~139-150 at DESIGN_RE
JMAX_MIN = 120
JMAX_MAX = 200

#endregion

#region n*, the sampler-only search for the fewest surface points

# le_ratio log-spaced over [RATIO_MIN, 1] x cluster_width over its range, then a NOMAD polish from the best
NSTAR_LE_GRID = 40
NSTAR_CW_GRID = 30
NSTAR_POLISH_EVALS = 600

#endregion

#region Config values

# c2d's smoothing defaults, kept by the golden run and rung 1 ("if something works, don't break it")
SMOOTHING_DEFAULTS = {"asmt": 40, "epsi": 20.0, "funi": 0.02, "alfa": 1.0}

# Never searched; epse (explicit smoothing) stays off. The config's other fields keep their defaults
PINNED = {"slvr": "HYPR", "topo": "OGRD", "radi": 20.0, "ypls": 0.7, "cfrc": 0.5, "epse": 0.0}

#endregion

#region Execution

# Pool size on the 6900HS: one serial ladder per worker
WORKERS = 6

#endregion
