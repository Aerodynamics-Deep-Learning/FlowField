"""
Tests for `common.utils`' shared input validators -- backend-agnostic, run once by
`common.entry.Common_GenerateMesh` before dispatch, i.e.:

Step 1: Ensure a numerically stable, chord-normalized input coordinate tensor
    - test_tensor_numeric_reject
    - test_valid_tensor_accept
    - test_tensor_chord_normalization
    - test_tensor_chord_tolerance_admits_an_unsampled_leading_edge
Step 2: Ensure the requested bluntness
    - test_bluntness_reject
    - test_bluntness_goodgap_accept
Step 3: Ensure physical freestream values
    - test_freestream_invalid_physics_reject
Step 4: Ensure the TE shape is reconciled against the requested (backend, topology) pairing
    - test_te_topology_pairing_accept
    - test_te_topology_pairing_reject

Scope: the validators themselves. Their routing onto a MeshExitFlag is covered in
test_common_entry.py's Step 1.
"""

import numpy as np
import torch
import pytest
from pydantic import ValidationError

from src.datagen.schemas import Freestream
from src.datagen.meshing.common.schemas import MeshBackend, MeshTopology
from src.datagen.meshing.common.utils import (
    Common_validate_tensor_numeric, Common_validate_te_bluntness, Common_validate_te_for_topology,
)


# region Step 1
@pytest.mark.parametrize("invalid_coords, checklist", [
    # 1. Not a torch.Tensor (NumPy array)
    (np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 0.0], [0.5, -0.5], [1.0, 0.0]]), 
     ["not", "torch.tensor"]),
    
    # 2. 2-D, but one column instead of (x, y)
    (torch.tensor([[1.0], [0.5], [0.0], [0.5], [1.0]], dtype=torch.float32),
     ["unexpected", "shape", "(5, 1)"]),

    # 2b. Not 2-D: 1-D and 0-D, which have no shape[1] for the message to read
    (torch.tensor([1.0, 0.5, 0.0, 0.5, 1.0], dtype=torch.float32),
     ["unexpected", "shape", "(5,)"]),
    (torch.tensor(1.0), ["unexpected", "shape", "()"]),

    # 2c. Empty, which min()/max() raise on
    (torch.empty((0, 2)), ["unexpected", "shape", "(0, 2)"]),

    # 3. Contains NaN (Note: checklist strictly lowercase)
    (torch.tensor([[1.0, torch.nan], [0.5, 0.5], [0.0, 0.0], [0.5, -0.5], [1.0, 0.0]], dtype=torch.float32), 
     ["has", "nan"]),
    
    # 4. Contains Inf (Note: checklist strictly lowercase)
    (torch.tensor([[1.0, torch.inf], [0.5, 0.5], [0.0, 0.0], [0.5, -0.5], [1.0, 0.0]], dtype=torch.float32), 
     ["has", "inf"])
])
def test_tensor_numeric_reject(invalid_coords, checklist):
    is_valid_tensor, msg = Common_validate_tensor_numeric(coords_tensor=invalid_coords)
    
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

    is_valid_tensor, msg = Common_validate_tensor_numeric(coords_tensor=coords_tensor)

    assert is_valid_tensor is True
    assert msg == ""


# The chord-normalization half of the same validator. `Airfoil.chord` carries physical size and
# scales mesh lengths by it, so a tensor that is not unit-chord gets scaled twice (this check is
# the only thing holding that split).
@pytest.mark.parametrize("scale, offset, expect_valid, why", [
    (1.0, 0.0, True, "unit chord at the origin"),
    (2.0, 0.0, False, "physical size baked into the tensor instead of Airfoil.chord"),
    (0.001, 0.0, False, "chord in the wrong unit"),
    (1.0, 0.5, False, "unit chord but translated off the origin"),
])
def test_tensor_chord_normalization(scale, offset, expect_valid, why):
    base = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 0.0], [0.5, -0.5], [1.0, 0.0]],
                        dtype=torch.float32)
    coords = base * scale
    coords[:, 0] += offset

    is_valid, msg = Common_validate_tensor_numeric(coords_tensor=coords)

    assert is_valid is expect_valid, why
    if not expect_valid:
        assert "chord-normalized" in msg


def test_tensor_chord_tolerance_admits_an_unsampled_leading_edge():
    # Real sections need not place a point exactly at x=0 (the LE is a discretization choice while
    # the TE is pinned by Selig ordering). The integration fixture's nose sits at x=1.7e-5, so a
    # float-precision bound here would reject valid geometry.
    coords = torch.tensor([[1.0, 0.0], [0.5, 0.5], [1.7e-5, 0.0], [0.5, -0.5], [1.0, 0.0]],
                          dtype=torch.float32)

    is_valid, msg = Common_validate_tensor_numeric(coords_tensor=coords)

    assert is_valid is True, msg
# endregion


# region Step 2
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
    is_valid_bluntness, msg = Common_validate_te_bluntness(coords_tensor=coords_tensor, micro_tol=tolerance)

    assert is_valid_bluntness is False
    assert all(cond in msg.lower() for cond in checklist)


def test_bluntness_goodgap_accept():
    diff_valid = 1e-6
    tolerance = diff_valid * 1e-1
    coords_tensor = torch.tensor([
        [1.0, diff_valid], [0.5, 0.5], [0.0, 0.0], [0.5, -0.5], [1.0, 0.0]
    ], dtype=torch.float32)

    is_valid_bluntness, msg = Common_validate_te_bluntness(coords_tensor=coords_tensor, micro_tol=tolerance)

    assert is_valid_bluntness is True
    assert msg == ""
# endregion


# region Step 3
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
        Freestream(**params)
    # Verify the specific keyword is in the error traceback
    assert expected_err_keyword.lower() in str(exc_info.value).lower()
# endregion


# region Step 4
# The two backends' topology names are inverted relative to each other, so this matrix is spelled
# out per (backend, topology) rather than per topology. gmsh's C-mesh closes on a finite TE line
# and its O-mesh on a single TE point; c2d is the other way round (verified against the
# binary, which prints "Sharp trailing edge: C-grid topology is recommended." and "Blunt trailing
# edge: O-grid topology is recommended."). Getting either row backwards silently produces a
# distorted mesh rather than an error, which is what these pin down.
coords_sharp_te = coords_zero_gap
coords_blunt_te = torch.tensor([
    [1.0, 0.0063], [0.5, 0.5], [0.0, 0.0], [0.5, -0.5], [1.0, -0.0063]
], dtype=torch.float32)


@pytest.mark.parametrize("backend, topology, coords_tensor", [
    (MeshBackend.GMSH, MeshTopology.CGRD, coords_blunt_te),
    (MeshBackend.GMSH, MeshTopology.OGRD, coords_sharp_te),
    (MeshBackend.C2D, MeshTopology.CGRD, coords_sharp_te),
    (MeshBackend.C2D, MeshTopology.OGRD, coords_blunt_te),
])
def test_te_topology_pairing_accept(backend, topology, coords_tensor):
    is_valid, msg = Common_validate_te_for_topology(
        coords_tensor=coords_tensor, backend=backend, topology=topology)

    assert is_valid is True
    assert msg == ""


@pytest.mark.parametrize("backend, topology, coords_tensor, checklist", [
    (MeshBackend.GMSH, MeshTopology.CGRD, coords_sharp_te, ["not blunted", "coincident"]),
    (MeshBackend.GMSH, MeshTopology.OGRD, coords_blunt_te, ["sharp", "blunt"]),
    (MeshBackend.C2D, MeshTopology.CGRD, coords_blunt_te, ["sharp", "blunt"]),
    (MeshBackend.C2D, MeshTopology.OGRD, coords_sharp_te, ["not blunted", "coincident"]),
])
def test_te_topology_pairing_reject(backend, topology, coords_tensor, checklist):
    is_valid, msg = Common_validate_te_for_topology(
        coords_tensor=coords_tensor, backend=backend, topology=topology)

    assert is_valid is False
    assert all(cond in msg.lower() for cond in checklist)
# endregion