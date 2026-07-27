"""
Conducts tests on the input checks of Gmsh meshing pipeline, i.e.:

Step 1: Ensure numerically stable input coordinate tensor
    - test_not_tensor_reject
    - test_not_2d_reject
    - test_nan_reject
    - test_inf_reject
    - test_valid_tensor_accept
Step 2: Ensure requested bluntness
    - test_bluntness_zerogap_reject
    - test_bluntness_smallgap_reject
    - test_bluntness_goodgap_accept
Step 3: Ensure physical freestream values
    - test_negative_mach_reject
    - test_negative_reynolds_reject
    - test_negative_altitude_reject
    - test_valid_freestream_accept
"""

import numpy as np
import torch
import pytest
from pydantic import ValidationError

from src.datagen.schemas import Freestream
from src.datagen.meshing.gmsh.utils import GMSH_validate_tensor_numeric, GMSH_validate_te_bluntness

#region Step 1
@pytest.mark.parametrize("invalid_coords, checklist", [
    # 1. Not a torch.Tensor (NumPy array)
    (np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 0.0], [0.5, -0.5], [1.0, 0.0]]), 
     ["not", "torch.tensor"]),
    
    # 2. Not 2D (1D column tensor)
    (torch.tensor([[1.0], [0.5], [0.0], [0.5], [1.0]], dtype=torch.float32), 
     ["unexpected", "number", "dims"]),
    
    # 3. Contains NaN (Note: checklist strictly lowercase)
    (torch.tensor([[1.0, torch.nan], [0.5, 0.5], [0.0, 0.0], [0.5, -0.5], [1.0, 0.0]], dtype=torch.float32), 
     ["has", "nan"]),
    
    # 4. Contains Inf (Note: checklist strictly lowercase)
    (torch.tensor([[1.0, torch.inf], [0.5, 0.5], [0.0, 0.0], [0.5, -0.5], [1.0, 0.0]], dtype=torch.float32), 
     ["has", "inf"])
])
def test_tensor_numeric_reject(invalid_coords, checklist):
    is_valid_tensor, msg = GMSH_validate_tensor_numeric(coords_tensor=invalid_coords)
    
    assert is_valid_tensor is False
    assert all(cond in msg.lower() for cond in checklist)


def test_valid_tensor_accept():
    coords_tensor = torch.tensor([
        [1.0, 0.0],
        [0.5, 0.5],
        [0.0, 0.0],
        [0.5, -0.5],
        [1.0, 0.0]
    ], dtype=torch.float32)

    is_valid_tensor, msg = GMSH_validate_tensor_numeric(coords_tensor=coords_tensor)
        
    assert is_valid_tensor is True
    assert msg == ""
#endregion

#region Step 2
# Define base coordinate structures for the failure modes
coords_zero_gap = torch.tensor([
    [1.0, 0.0], [0.5, 0.5], [0.0, 0.0], [0.5, -0.5], [1.0, 0.0]
], dtype=torch.float32)

diff_micro = 1e-6
coords_micro_gap = torch.tensor([
    [1.0, diff_micro], [0.5, 0.5], [0.0, 0.0], [0.5, -0.5], [1.0, 0.0]
], dtype=torch.float32)


@pytest.mark.parametrize("coords_tensor, tolerance, checklist", [
    # 1. Zero gap (coincident trailing edge points)
    (coords_zero_gap, 1e-5, ["not blunted", "coincident"]),
    
    # 2. Micro gap (smaller than provided tolerance)
    (coords_micro_gap, diff_micro * 10, ["gap", "small"])
])
def test_bluntness_reject(coords_tensor, tolerance, checklist):
    is_valid_bluntness, msg = GMSH_validate_te_bluntness(coords_tensor=coords_tensor, micro_tol=tolerance)

    assert is_valid_bluntness is False
    assert all(cond in msg.lower() for cond in checklist)


def test_bluntness_goodgap_accept():
    diff_valid = 1e-6
    tolerance = diff_valid * 1e-1
    coords_tensor = torch.tensor([
        [1.0, diff_valid], [0.5, 0.5], [0.0, 0.0], [0.5, -0.5], [1.0, 0.0]
    ], dtype=torch.float32)

    is_valid_bluntness, msg = GMSH_validate_te_bluntness(coords_tensor=coords_tensor, micro_tol=tolerance)

    assert is_valid_bluntness is True
    assert msg == ""
#endregion

#region Step 3
@pytest.mark.parametrize("invalid_kwargs, expected_err_keyword", [
    ({"mach": -0.5}, "mach"),
    ({"Re": -1e6}, "re"),
    ({"altitude_m": -100.0}, "altitude")
])
def test_freestream_invalid_physics_reject(invalid_kwargs, expected_err_keyword):
    # Base dictionary of valid physical parameters
    params = {
        "alpha": 1.0,
        "Re": 2e6,
        "mach": 0.5,
        "altitude_m": 100.0,
        "temp": 27.4
    }
    params.update(invalid_kwargs)
    # Catch the expected Pydantic ValidationError
    with pytest.raises(ValidationError) as exc_info:
        freestream = Freestream(**params)
    # Verify the specific keyword is in the error traceback 
    assert expected_err_keyword.lower() in str(exc_info.value).lower()
#endregion