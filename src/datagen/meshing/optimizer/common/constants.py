"""
The mesh study's fixed choices: what it meshes, the config fields it pins, and the limits that guard a
batch. The values that can change (gates, ladder, budgets, the search space, Re, the TE gap, execution) are in
`src/datagen/constants.py`; `settings.Optim_build_settings` records every one that shapes or judges a mesh.

Imports only `src/datagen/constants.py`.
"""

from src.datagen.constants import FARFIELD_RADIUS, Y_PLUS

#region The study

# MeshBackend.C2D's and MeshTopology.OGRD's values
BACKEND = "c2d"
TOPOLOGY = "OGRID"

# The pipeline's normalized chord; c2d meshes in unit chord whatever it is told
CHORD = 1.0

#endregion

#region The ladder

# NOMAD's own evaluation cap, as a multiple of the budget; only guards a runaway, since rejections are free
MAX_BB_EVAL_FACTOR = 10

# Bump by hand on any change to the ladder's flow logic
LADDER_VERSION = "ladder-r3"

#endregion

#region Config values

# Never searched; epse (explicit smoothing) stays off. The config's other fields keep their defaults
PINNED = {"slvr": "HYPR", "topo": "OGRD", "radi": FARFIELD_RADIUS, "ypls": Y_PLUS, "cfrc": 0.5, "epse": 0.0}

#endregion

#region Execution

# Aborts in a row that stop a batch: the pool's size, but never fewer than this
ABORT_STREAK_MIN = 3

# How long a write waits out another program holding the run store (a viewer mid-read)
DB_LOCK_WAIT_S = 60

#endregion
