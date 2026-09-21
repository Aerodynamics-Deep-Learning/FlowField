"""
Tests for `gmsh.utils`' meshing-param helpers, i.e.:

Step 1: Infer the within boundary layer params
    - test_firstcellheight_calculator_accept
Step 2: Scale chord-normalized config lengths to the airfoil's actual chord
    - test_scale_by_chord

The common counterpart is test_common_utils.py; gmsh's flagging lives in test_gmsh_runner.py.
"""

import pytest

from src.datagen.meshing.gmsh.utils import GMSH_get_mesh_height, GMSH_scale_by_chord


# region Step 1
# Checks hand calculated first mesh cell height values (using schlichting 1/5th law, to match assumptions)
@pytest.mark.parametrize("Re, chord, target_yplus, analytical_val, tol", [
    (1.5e6, 1.0, 1.0, 1.623001e-5, 1e-4),
    (3e6, 1.0, 1.0, 8.697415e-6, 1e-4),
    (4.5e6, 1.0, 1.0, 6.038166e-6, 1e-4),
    (6e6, 1.0, 1.0, 4.661005e-6, 1e-4),
    (7.5e6, 1.0, 1.0, 3.813083e-6, 1e-4),
    (9e6, 1.0, 1.0, 3.235808e-6, 1e-4),
    (1.05e7, 1.0, 1.0, 2.816635e-6, 1e-4),
    (1.2e7, 1.0, 1.0, 2.497686e-6, 1e-4),
    (1.35e7, 1.0, 1.0, 2.246470e-6, 1e-4),
    (1.5e7, 1.0, 1.0, 2.043238e-6, 1e-4),
])
def test_firstcellheight_calculator_accept(Re, chord, target_yplus, analytical_val, tol):
    y_height = GMSH_get_mesh_height(Re=Re, chord=chord, target_yplus=target_yplus)

    assert y_height == pytest.approx(analytical_val, rel=tol)
# endregion


# region Step 2
# chord=1.0 is the pipeline's normal invariant, so scaling should be a no-op there; other chords
# exercise the mitigation path (see GMSH_scale_by_chord's docstring)
@pytest.mark.parametrize("value, chord, expected", [
    (15.0, 1.0, 15.0),
    (0.4, 1.0, 0.4),
    (15.0, 2.0, 30.0),
    (0.4, 0.5, 0.2),
])
def test_scale_by_chord(value, chord, expected):
    assert GMSH_scale_by_chord(value, chord) == pytest.approx(expected)
# endregion
