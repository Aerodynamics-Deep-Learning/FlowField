"""
The datagen values that can reasonably change, in one place: the CST fit and the sampler's limits, the mesh
study's gates, ladder and search space, the calibration's method, and how batches run. What stays fixed (the
pipeline's acceptance limits, protocol names, implementation details, version strings) lives in its domain's own
constants module.

Each one can be set for a run (a config's `settings:`, a script's `--set`) by its name in lower case. Imports only
`meshing.common.constants`, which imports nothing, as a module: every upper-case name here is one of these values.
"""

from src.datagen.meshing.common import constants as _pipeline

#region Geometry: the CST fit and the sampler

# Bernstein order of every airfoil's CST fit; the shape version names it (cst8-r1)
CST_ORDER = 8

# How the CST fit represents the nose: "lem" (one radius plus Kulfan's LE term), "separate" (a radius per surface)
# or "shared" (one radius); the shape version names any but lem
CST_NOSE = "lem"

# Largest distance [c] from a root point to its fitted CST; a worse fit is rejected
CST_FIT_TOL = 1e-3

# Longest panel [c] allowed anywhere on a sampled surface, so the top of dx_mid
DX_MAX = 0.006

# Largest turn [deg] a sampled point may make; sets how finely a round TE's caps are spaced
THETA_MAX_DEG = 15.0

#endregion

#region The mesh study

# The Reynolds number c2d sizes the first cell for, with Y_PLUS
DESIGN_RE = 1.5e7

# The gap [c] a round or blunt TE is opened to
TE_GAP = 0.004

# Most surface points x jmax a config may ask for; more is rejected before meshing
CELL_CEILING = 300_000

# c2d's first-cell y+ and farfield radius [c], pinned in every config
Y_PLUS = 0.7
FARFIELD_RADIUS = 20.0

# c2d's smoothing knobs at the golden run and in rung 1
SMOOTHING_DEFAULTS = {"asmt": 40, "epsi": 20.0, "funi": 0.02, "alfa": 1.0}

#endregion

#region Gates, on the worst cell

# The study's shape gates: skewness at most, orthogonality and Jacobian at least
GATE_SKEW_MAX = 0.25
GATE_ORTHO_MIN = 0.9
GATE_JAC_MIN = 0.9

# Aspect ratio, size ratio and geometric deviation at most: the pipeline's own limits
GATE_AR_MAX = _pipeline.ASPECT_RATIO_LIMIT
GATE_SIZE_RATIO_MAX = _pipeline.SIZE_RATIO_LIMIT
GATE_GEO_DEV_MAX = _pipeline.GEO_DEV_LIMIT

#endregion

#region The ladder

# jmax of the golden run, every airfoil's first mesh
GOLDEN_JMAX = 147

# Rung 1 raises jmax by this, at most this many times
JMAX_BUMP_STEP = 1
JMAX_BUMP_MAX = 10

# Finished meshes per NOMAD rung: 2 (smoothing only), 3 (all knobs)
RUNG2_BUDGET = 50
RUNG3_BUDGET = 100

# Rung 3 ends once its last RUNG3_PATIENCE c2d runs cut the best's nodes by less than this share of them
RUNG3_PATIENCE = 20
RUNG3_MIN_GAIN = 0.005

# NOMAD's random seed
NOMAD_SEED = 1

#endregion

#region The search space's bounds (meshing/optimizer/common/space.py)

# Floor on dx_mid [c]
DX_MID_MIN = 1e-3

# Floor on le_ratio and te_ratio
RATIO_MIN = 0.002

# cluster_width's range
CLUSTER_WIDTH_MIN = 0.026
CLUSTER_WIDTH_MAX = 0.25

# jmax's range
JMAX_MIN = 120
JMAX_MAX = 200

# n*'s grid, n* being its best point: le_ratio points log-spaced over [RATIO_MIN, 1] x cluster_width points over its
# range
NSTAR_LE_GRID = 40
NSTAR_CW_GRID = 1

#endregion

#region The calibration (meshing/optimizer/batch/calibrate.py)

# The calibration's fixed airfoils (rung 2 when rung 1 does not run, rung 3), also meshed by the machine check
CAL_SET = ("s1223", "s1210", "mid012r", "e376", "goe531", "naca0040", "e423", "fx69pr281", "naca0006", "naca4412")

# Rung 3's other airfoils, drawn from the rest
CAL_ORDINARY_COUNT = 5

# Meshes rung 3 and rung 2 may spend while calibrated
CAL_RUNG3_CAP = 300
CAL_RUNG2_CAP = 100

# Golden jmax = median jmax* + this
CAL_GOLDEN_MARGIN = 5

# Rung 3's replay: every (window, min_gain) pair, and the most a pair may give up on any airfoil against its
# cap's best
CAL_WINDOWS = (10, 15, 20, 30, 50)
CAL_GAINS = (0.0025, 0.005, 0.01)
CAL_MAX_LOSS = 0.005

# How close to the final best counts as reaching it, for the rung-3 cap
CAL_WITHIN = 0.01

#endregion

#region Execution

# Worker processes, one serial ladder each
WORKERS = 6

# A worker silent this long [s] is killed as hung
WATCHDOG_S = 600

# Each slot's memory cap [MB], worker and c2d together
SLOT_MEMORY_CAP_MB = 2048

#endregion
