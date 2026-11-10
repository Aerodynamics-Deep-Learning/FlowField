"""
Fixed values the geometry pipeline builds and gates airfoils with; the ones that can change (the CST order,
nose and fit tolerance, the sampler's limits) are in `src/datagen/constants.py`.

Imports only that. A change to anything in the shape block changes the stored shapes, so it must come
with a new `SHAPE_REVISION`.
"""

from src.datagen.constants import CST_NOSE, CST_ORDER

#region CST shape

# Class-function exponents: round nose, sharp TE (every airfoil is closed to this)
CST_N1 = 0.5
CST_N2 = 1.0

# The revision of the normalize -> close -> fit -> render pipeline behind a stored shape; bump it on any change
# to it (tests/unit/datagen/geometry/test_geom_shape_version.py fails until you do)
SHAPE_REVISION = "r1"


def Geom_shape_version(order: int, nose) -> str:
    """The name shapes fitted at a CST order and nose are stored under: `cst8-r1` (lem), `cst10separate-r1`."""
    nose = getattr(nose, "value", nose)
    return f"cst{order}{'' if nose == 'lem' else nose}-{SHAPE_REVISION}"


# The constants' own, which the shared manifest holds
SHAPE_VERSION = Geom_shape_version(CST_ORDER, CST_NOSE)


#endregion

#region Gates

# Worst distance from a family file's points to its formula shape: catches a misread name, not digitization noise
FAMILY_MATCH_TOL = 1e-2

# A report hint, not a gate: the worst-fit point is this many times further off than its neighbours and the
# points zig-zag there, as a mistyped coordinate does (3 of 180 on for_cst: bambino6, hs1430, ah79100a)
SUSPECT_ISOLATION = 4.0

#endregion

#region Sampling

# Points per surface when a family is built from its formula, as in the analytic NACA files
FAMILY_POINTS_PER_SURFACE = 501

# Points in the dense CST contour that distances are measured against
CURVE_POINTS = 4001

# The normalised .dat rendering of a stored shape
NORM_N_POINTS = 1001
DAT_DECIMALS = 10

# Fewest points a .dat can hold and still describe two surfaces
MIN_DAT_POINTS = 5

#endregion

#region Sequence 3: the TE a mesher needs, and sampling for it

# Names the contour -> spacing -> checks behind a sample, the constants below included (dx_max and theta_max_deg
# are each run's, recorded with its settings); bump the revision on any change to it
# (tests/unit/datagen/geometry/test_geom_sampler_version.py fails until you do)
SAMPLER_VERSION = "samp-r2"

# The turn pre-check passes below this share of its limit, so rounding cannot trip the post-check
PRECHECK_MARGIN = 0.995

# A round TE's welds sit at x = 1 - WELD_OFFSET * g, the gap g being the study's
WELD_OFFSET = 0.5

# Points per piece of a contour's dense arc-length and curvature tables
ARC_TABLE_POINTS = 20001

#endregion