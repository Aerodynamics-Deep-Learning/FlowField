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

# Bernstein order, one for every airfoil so the ML input length is fixed. From the calibration of
# 2026-10-14 (report/calibration/): higher orders fit the points better but overshoot between them on
# sparse files (3 files at 8, 13 at 10, 38 at 12). SHAPE_VERSION (geometry/common/constants.py) names it
CST_ORDER = 8

# How the nose is fitted, a `schemas.Nose` value: one radius plus Kulfan's LE term passes 158 of 180
# within 1e-3 at order 8, two radii (a curvature jump) 150, one radius alone 87. A change needs a new SHAPE_VERSION
CST_NOSE = "lem"

# Worst perpendicular distance from a root point to its fitted CST, chord units. Set above where good fits
# land at order 8 (formula NACAs <= 1.2e-4, median 4.6e-4); passes 158 of 180
CST_FIT_TOL = 1e-3

# Longest panel allowed anywhere, so the bound on the mid-chord spacing.
DX_MAX = 0.006

# Largest turn a sampled point may make; sets how fine the spacing on a round TE's caps must be
THETA_MAX_DEG = 15.0

#endregion

#region The mesh study

# The top of Freestream.Re, so one mesh over-resolves every case; the only freestream value c2d reads
DESIGN_RE = 1.5e7

# The study's TE gap [c] for a round or blunt TE: a study hyperparameter, recorded with the study
TE_GAP = 0.004

# Most surface points x jmax a config may ask for; more is rejected before meshing
CELL_CEILING = 300_000

# c2d's first-cell y+ and farfield radius [c], pinned in every config
Y_PLUS = 0.7
FARFIELD_RADIUS = 20.0

# c2d's smoothing defaults, kept by the golden run and rung 1 ("if something works, don't break it")
SMOOTHING_DEFAULTS = {"asmt": 40, "epsi": 20.0, "funi": 0.02, "alfa": 1.0}

#endregion

#region Gates, on the worst cell

# The study's shape gates, what a campaign's mesh must reach; far stricter than the entry flag's (0.85, 0.10, 0.10).
GATE_SKEW_MAX = 0.25
GATE_ORTHO_MIN = 0.9
GATE_JAC_MIN = 0.9

# The entry flag's own
GATE_AR_MAX = _pipeline.ASPECT_RATIO_LIMIT
GATE_SIZE_RATIO_MAX = _pipeline.SIZE_RATIO_LIMIT
GATE_GEO_DEV_MAX = _pipeline.GEO_DEV_LIMIT

#endregion

#region The ladder

# naca0012's smallest passing jmax (142) + 5. Provisional: the calibration re-derives it
GOLDEN_JMAX = 147

# Rung 1 raises jmax by this, at most this many times
JMAX_BUMP_STEP = 1
JMAX_BUMP_MAX = 10

# Finished meshes per NOMAD rung: 2 (smoothing only), 3 (all knobs). 
RUNG2_BUDGET = 50
RUNG3_BUDGET = 100

# Rung 3 ends once its last RUNG3_PATIENCE c2d runs cut the best's nodes by less than this share of them.
RUNG3_PATIENCE = 20
RUNG3_MIN_GAIN = 0.005

NOMAD_SEED = 1

#endregion

#region The search space's bounds (meshing/optimizer/common/space.py)

# Floor on dx_mid.
DX_MID_MIN = 1e-3

# Floor on le_ratio and te_ratio
RATIO_MIN = 0.002

# cluster_width's range; 0.026 keeps the neighbour panel ratio <= 1.2.
CLUSTER_WIDTH_MIN = 0.026
CLUSTER_WIDTH_MAX = 0.25

# jmax's range; the size-ratio gate needs ~139-150 at DESIGN_RE
JMAX_MIN = 120
JMAX_MAX = 200

# n*'s grid: le_ratio log-spaced over [RATIO_MIN, 1] x cluster_width over its range; no finer search past it.
NSTAR_LE_GRID = 40
NSTAR_CW_GRID = 1

#endregion

#region The calibration (meshing/optimizer/batch/calibrate.py)

# The hard cases: the most surface points (s1223, s1210, mid012r), the highest jmax* (e376, goe531) and the
# lowest (naca0040), worst lift losses (e423, fx69pr281), a thin and a typical section
CAL_SET = ("s1223", "s1210", "mid012r", "e376", "goe531", "naca0040", "e423", "fx69pr281", "naca0006", "naca4412")

# Rung 3's ordinary airfoils, drawn from the rest
CAL_ORDINARY_COUNT = 5

# Meshes rung 3 and rung 2 may spend
CAL_RUNG3_CAP = 300
CAL_RUNG2_CAP = 100

# Golden jmax = median jmax* + this
CAL_GOLDEN_MARGIN = 5

# Rung 3's replay: every (window, min_gain) pair, and the most a pair may give up on any airfoil against its
# cap's best
CAL_WINDOWS = (10, 15, 20, 30, 50)
CAL_GAINS = (0.0025, 0.005, 0.01)
CAL_MAX_LOSS = 0.005

# Close to the final best, for the rung-3 cap
CAL_WITHIN = 0.01

#endregion

#region Execution

# Pool size on the 6900HS: one serial ladder per worker
WORKERS = 6

# A worker that sends nothing for this long is killed as hung. Above c2d's own 300 s timeout, with room for
# the processing after it
WATCHDOG_S = 600

# Each slot's memory cap [MB], worker and c2d together: measured worst case at CELL_CEILING ~1350 (0.5 GB
# imports, 0.97 KB per cell, c2d's 66 B per surface point squared)
SLOT_MEMORY_CAP_MB = 2048

#endregion
