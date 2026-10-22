"""
Fixed values the geometry pipeline builds and gates airfoils with.

Imports nothing. A change to anything in the shape block changes the stored shapes, so it must come
with a new `SHAPE_VERSION`.
"""

#region CST shape

# Class-function exponents: round nose, sharp TE (every airfoil is closed to this)
CST_N1 = 0.5
CST_N2 = 1.0

# Bernstein order, one for every airfoil so the ML input length is fixed. From the calibration of
# 2026-10-14 (report/calibration/): higher orders fit the points better but overshoot between them on
# sparse files (3 files at 8, 13 at 10, 38 at 12)
CST_ORDER = 8

# How the nose is fitted, a `schemas.Nose` value: one radius plus Kulfan's LE term passes 158 of 180
# within 1e-3 at order 8, two radii (a curvature jump) 150, one radius alone 87
CST_NOSE = "lem"

# Names the normalize -> close -> fit -> render pipeline behind a stored shape; bump the revision on any
# change to it (tests/unit/datagen/geometry/test_geom_shape_version.py fails until you do)
SHAPE_VERSION = f"cst{CST_ORDER}-r1"


#endregion

#region Gates

# Worst perpendicular distance from a root point to its fitted CST, chord units. Set above where good fits
# land at order 8 (formula NACAs <= 1.2e-4, median 4.6e-4); passes 158 of 180
CST_FIT_TOL = 1e-3

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

# Largest turn a sampled point may make; sets how fine the spacing on a round TE's caps must be
THETA_MAX_DEG = 15.0

# A round TE's welds sit at x = 1 - WELD_OFFSET * g, the gap g being the study's
WELD_OFFSET = 0.5

# Points per piece of a contour's dense arc-length and curvature tables
ARC_TABLE_POINTS = 20001

#endregion