"""
Every fixed name and acceptance threshold the meshing pipeline gates on, in one place.

Nothing here is a tunable. Meshing parameters live on the per-backend `MeshingConfig` classes and
callers change them freely; these are the fixed criteria the result is judged against afterwards,
plus the boundary names the solver step reads back out of the written file. Changing a value here
changes what the pipeline considers a usable mesh at all, so each one carries what it was set from.

This module imports nothing, deliberately. It is the one thing in `common/` the backends are allowed
to reach for; adding an import here would put the rest of `common` on the backends' import path for real.
So, please don't do that.
"""

# ----------------------------------------------------------------- SU2 boundary markers

# The no-slip wall and outer-boundary tags, written into every `.su2` by both backends and read back
# downstream. Single-sourced because four places have to agree on the exact string: `c2d.convert`
# and `gmsh.run` write them, `common.geo_dev` looks MARKER_AIRFOIL up in the result, and
# `solvers.su2.schemas` defaults its marker fields to them, which is what lets one SU2 config drive
# meshes from either backend. Matched exactly, never case-folded. A case mismatch is a real defect
# and `test_cross_backend.py` pins it.
MARKER_AIRFOIL = "MARKER_AIRFOIL"
MARKER_FARFIELD = "MARKER_FARFIELD"


# ------------------------------------------------------------- Geometric-deviation limit

# Worst allowed boundary deviation, in chord-normalized units. Measured against real meshes from
# both backends rather than picked a priori: the worst observed is 7.7e-4 (gmsh O-mesh at
# nx_afoil=100, where the boundary is coarser than the input) and 5.1e-4 (c2d), against a floor of
# ~3.0e-5 that does not move with resolution; that floor is the sagitta (look word up) between the input
# polyline and the spline drawn through it, so it is set by the *input's* point density, not the
# mesh's. This limit sits ~3x above the worst observed and well over an order of magnitude below a
# real geometry error (a wrongly-scaled boundary measured 5.3e-2). See the integration tier.
GEO_DEV_LIMIT = 2.5e-3


# ------------------------------------------------------------------- Mesh-quality gates
#
# All five are measured by `common.quality` (four per cell by `_cell_metrics`/`_orthogonal_quality`,
# the size ratio per interior face) and applied together in `_analyze_su2`. The limits come from
# real meshes at the configs the integration tier uses (`tests/integration/datagen/mesh/conftest.py`),
# plus Emre's reference `example_naca0012.vtk`:
#
#                             max_skew   min_ortho   min_jac    max_ar   max_size_ratio
#     reference .vtk            0.166      0.967      0.966     2.9e3        1.14
#     gmsh C-mesh               0.446      0.765      0.765     5.7e5        242
#     gmsh O-mesh               0.909      0.143      0.143     2.1e3        1.21
#     c2d  O-grid               0.231      0.937      0.935     6.7e3        1.18
#     c2d  C-grid               0.367      0.839      0.838     8.8e5        1.18
#
# Both gmsh rows are rejected: the O-mesh on skew at its sharp TE, the C-mesh on size ratio where its
# wake meets the TE. The per-cell gates below pass that C-mesh, so their margins are read against it.

# Equiangle skewness: 0 perfect, 1 degenerate. Every mesh here but the O-mesh sits at 0.45 or below,
# so this leaves a wide margin before a real defect.
SKEWNESS_LIMIT = 0.85

# Orthogonal quality: measured off face normals rather than inferred from skew: per face, the cosine
# between its outward normal and the centroid->face-centroid vector, and across a shared face the
# cosine against the vector joining the two cell centroids, worst face wins. That neighbour term is
# what makes this its own axis and not skewness read backwards; it measures how squarely a face
# sits between the two cells sharing it, which no single cell's corner angles can show. It binds on
# ~2% of cells in the O-mesh above. Every other mesh sits at 0.765 and up while the known-bad O-mesh
# reads 0.143, so this floor currently only catches near-degenerate cells; tightening it toward 0.2 and
# more would make orthogonality reject that O-mesh on its own rather than leaving it to the skew gate.
ORTHO_MIN = 0.10

# Scaled Jacobian: the minimum over a cell's corners of the normalized cross product of the two edges
# meeting there, signed by winding; 1 is a right angle, 0 a collapsed corner, below 0 a corner that
# has folded back on itself. This catches what skewness structurally cannot: skew comes from
# `arccos`, which folds a reflex corner back into [0, 180] and so reads a tangled cell as
# well-shaped, and the shoelace `neg` check only sees a cell's net area, which a partly folded quad
# keeps positive. <= 0 is therefore UNACCEPTABLE_QUALITY rather than LOW_QUALITY, matching the
# "minSICN <= 0" wording already in `MeshExitFlag` (gmsh's SICN is this same quantity). The worst
# mesh here but the O-mesh measures 0.765, so this floor only ever fires on folding. May consider increasing.
JACOBIAN_LIMIT = 0.10

# Longest edge / shortest edge, and deliberately far looser than the rest. A y+=1 boundary layer and
# a 20-chord wake stretch cells enormously on purpose: the meshes above reach 8.8e5 (c2d C-grid)
# and 5.7e5 (gmsh C-mesh), and high aspect ratio near the wall is the design "must have".
# Set at an order of magnitude above the worst of those so it fires only on a genuinely collapsed edge,
# which `_cell_metrics`' 1e-30 divisor floor sends to ~1e30, rather than policing stretching.
ASPECT_RATIO_LIMIT = 1.0e7

# Larger over smaller cell area across an interior face, worst face wins: how abruptly cell size
# changes between neighbours, which the four gates above cannot see because they score each cell on
# its own. In a structured grid it is the growth rate along whichever direction crosses the face.
# It is what stops a cell-count minimizer: coarsening c2d's O-grid passes every other gate
# (jmax=30 scores skew 0.35) while nearly every radial neighbour pair grows 2.7x. Set against the
# good meshes above (1.14-1.21) and the usual 1.2-1.25 wall-normal growth rule: it admits c2d down to
# jmax=130 at nsrf=300, or nsrf=250 at jmax=200 (measured on NACA0012, one knob at a time).
# Area ratio reads a triangle next to an equal-edged quad as 2 by shape alone, so a hybrid tri/quad
# mesh would need this normalized first; every mesh both backends write is all-quad.
SIZE_RATIO_LIMIT = 1.25
