"""
Conducts test on the step where we infer meshing params in Gmsh mesh pipeline, i.e.:

Step 1: Infer the within boundary layer params
    - test_firstcellheight_calculator_accept
"""

import pytest

from src.datagen.meshing.gmsh.utils import GMSH_get_mesh_height

#region Step 1

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
#endregion