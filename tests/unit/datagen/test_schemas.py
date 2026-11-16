"""
Tests for the shared `src.datagen.schemas` contract, i.e.:

Step 1: Ensure `Airfoil.airfoil_name` always yields a usable filename stem
    - test_airfoil_name_defaults_when_omitted
    - test_airfoil_name_unusable_values_rejected
Step 2: Ensure `Airfoil.coords` is always a float64 array, and survives JSON
    - test_coords_become_float64_arrays
    - test_coords_that_are_not_numbers_rejected
    - test_airfoil_round_trips_through_json

Both meshing backends build every output path from the name (gmsh `{name}_mesh.su2`, c2d
`{name}.dat`/`.p3d`), so a missing one used to surface as `None_mesh.su2` or a c2d FATAL_ERROR.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from src.datagen.schemas import Airfoil

_COORDS = [[1.0, 0.0], [0.5, 0.05], [0.0, 0.0], [0.5, -0.05], [1.0, 0.0]]


# region Step 1
def test_airfoil_name_defaults_when_omitted():
    assert Airfoil(coords=_COORDS, le_idx=2).airfoil_name == "not_given"


@pytest.mark.parametrize("name", [None, ""])
def test_airfoil_name_unusable_values_rejected(name):
    with pytest.raises(ValidationError, match="airfoil_name"):
        Airfoil(airfoil_name=name, coords=_COORDS, le_idx=2)
# endregion


# region Step 2
@pytest.mark.parametrize("given", [_COORDS, np.array(_COORDS, dtype=np.float32), np.array(_COORDS)])
def test_coords_become_float64_arrays(given):
    # gmsh's O-mesh needs float64 (float32 circle points miss the radius by enough for OCC to reject the arcs)
    coords = Airfoil(coords=given, le_idx=2).coords

    assert isinstance(coords, np.ndarray) and coords.dtype == np.float64
    np.testing.assert_array_equal(coords, np.asarray(given, dtype=np.float64))  # values kept, only widened


@pytest.mark.parametrize("given", ["not coords", [[1.0, "x"]], object()])
def test_coords_that_are_not_numbers_rejected(given):
    with pytest.raises(ValidationError, match="coords"):
        Airfoil(coords=given, le_idx=2)


def test_airfoil_round_trips_through_json():
    airfoil = Airfoil(airfoil_name="naca0012", coords=np.array(_COORDS) / 3.0, chord=2.0, le_idx=2)
    restored = Airfoil.model_validate_json(airfoil.model_dump_json())

    np.testing.assert_array_equal(restored.coords, airfoil.coords)  # bit-exact: JSON floats round-trip
    assert restored.model_dump(exclude={"coords"}) == airfoil.model_dump(exclude={"coords"})
# endregion
