"""
The mesh study's fixed choices in one place: what it meshes, the gates a mesh must pass, the ladder's
steps and budgets, and the search space's bounds. `settings.Optim_build_settings` records every one that shapes
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

# The study's shape gates, what a campaign's mesh must reach; far stricter than the entry flag's (0.85, 0.10, 0.10).
# Skew 0.25: 128 of 158 golden-like meshes under it, 2 under 0.22 (2026-11-07 calibration)
GATE_SKEW_MAX = 0.25
GATE_ORTHO_MIN = 0.9
GATE_JAC_MIN = 0.9

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

# Finished meshes per NOMAD rung: 2 (smoothing only), 3 (all knobs). Rung 2's from the 2026-11-07 calibration
# (41 meshes to s1223's first pass at skew 0.22); rung 3's provisional
RUNG2_BUDGET = 50
RUNG3_BUDGET = 100

# Rung 3 ends once its last RUNG3_PATIENCE c2d runs cut the best's nodes by less than this share of them.
# Provisional: the calibration sets them
RUNG3_PATIENCE = 20
RUNG3_MIN_GAIN = 0.005

# NOMAD's own evaluation cap, as a multiple of the budget; only guards a runaway, since rejections are free
MAX_BB_EVAL_FACTOR = 10

NOMAD_SEED = 1

# Bump by hand on any change to the ladder's flow logic
LADDER_VERSION = "ladder-r3"

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

# le_ratio log-spaced over [RATIO_MIN, 1] x cluster_width over its range; no finer search past the grid.
# cluster_width's grid is its narrowest alone: n* landed there on all 158 airfoils (2026-11-07 calibration)
NSTAR_LE_GRID = 40
NSTAR_CW_GRID = 1

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

# A worker that sends nothing for this long is killed as hung. Above c2d's own 300 s timeout, with room for
# the processing after it
WATCHDOG_S = 600

# Each slot's memory cap [MB], worker and c2d together: measured worst case at CELL_CEILING ~1350 (0.5 GB
# imports, 0.97 KB per cell, c2d's 66 B per surface point squared)
SLOT_MEMORY_CAP_MB = 2048

# Aborts in a row that stop a batch: the pool's size, but never fewer than this
ABORT_STREAK_MIN = 3

# How long a write waits out another program holding the run store (a viewer mid-read)
DB_LOCK_WAIT_S = 60

#endregion
