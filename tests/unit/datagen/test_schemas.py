"""
Tests for the shared `src.datagen.schemas` contract, i.e.:

Step 1: Ensure `Airfoil.airfoil_name` always yields a usable filename stem
    - test_airfoil_name_defaults_when_omitted
    - test_airfoil_name_unusable_values_rejected

Both meshing backends build every output path from the name (gmsh `{name}_mesh.su2`, c2d
`{name}.dat`/`.p3d`), so a missing one used to surface as `None_mesh.su2` or a c2d FATAL_ERROR.
"""

import pytest
import torch
from pydantic import ValidationError

from src.datagen.schemas import Airfoil


def _coords() -> torch.Tensor:
    return torch.tensor([[1.0, 0.0], [0.5, 0.05], [0.0, 0.0], [0.5, -0.05], [1.0, 0.0]], dtype=torch.float32)


# region Step 1
def test_airfoil_name_defaults_when_omitted():
    assert Airfoil(coords_tensor=_coords(), le_idx=2).airfoil_name == "not_given"


@pytest.mark.parametrize("name", [None, ""])
def test_airfoil_name_unusable_values_rejected(name):
    with pytest.raises(ValidationError, match="airfoil_name"):
        Airfoil(airfoil_name=name, coords_tensor=_coords(), le_idx=2)
# endregion
