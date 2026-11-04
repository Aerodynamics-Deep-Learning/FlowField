"""
Tests for `common.utils`' shared input validators -- backend-agnostic, run once by
`common.entry.Common_GenerateMesh` before dispatch, i.e.:

Step 1: Ensure a numerically stable, chord-normalized input coordinate tensor
    - test_tensor_numeric_reject
    - test_valid_tensor_accept
    - test_tensor_chord_normalization
    - test_tensor_chord_tolerance_admits_an_unsampled_leading_edge
Step 2: Ensure the TE closure and shape are measured as the gates read them
    - test_te_closure_classification
    - test_te_closure_angle
    - test_sharpest_corner_finds_a_flat_base_the_closure_angle_cannot
Step 3: Ensure physical freestream values
    - test_freestream_invalid_physics_reject
Step 4: Ensure the TE closure gate and shape warnings follow the (backend, topology) matrix
    - test_te_closure_pairing_accept
    - test_te_closure_pairing_reject
    - test_te_shape_warnings

Scope: the validators themselves. Their routing onto a MeshExitFlag is covered in
test_common_entry.py's Step 1.
"""

import numpy as np
import torch
import pytest
from pydantic import ValidationError

from src.datagen.schemas import Freestream
from src.datagen.meshing.common.schemas import MeshBackend, MeshTopology
from src.datagen.meshing.common.constants import TE_GAP_MIN, CORNER_TURN_LIMIT, SHARP_TE_ANGLE
from src.datagen.meshing.common.utils import (
    Common_validate_tensor_numeric, Common_classify_te_closure, Common_te_closure_angle,
    Common_sharpest_corner, Common_validate_te_closure, Common_te_shape_warnings, TEClosure,
)
from tests.closed_airfoils import closed_naca0012


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
def _coarse(te_upper, te_lower):
    """A 5-point contour whose only variable is its two TE points. float64, so TE_GAP_MIN is exact."""
    return torch.tensor([te_upper, [0.5, 0.06], [0.0, 0.0], [0.5, -0.06], te_lower], dtype=torch.float64)


def _flat_base_naca0012() -> torch.Tensor:
    """Open NACA0012 closed by a flat base, closure point mid-base: straight through the closure,
    ~80 deg corners where the base meets each surface."""
    x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, 81)))
    y = 0.6 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1015 * x ** 4)
    pts = np.vstack([[1.0, 0.0], np.stack([x[::-1], y[::-1]], 1), np.stack([x[1:], -y[1:]], 1), [1.0, 0.0]])
    return torch.tensor(pts, dtype=torch.float64)


@pytest.mark.parametrize("coords, expected", [
    (_coarse([1.0, 0.0], [1.0, 0.0]), TEClosure.CLOSED),
    (_coarse([1.0, 0.0], [1.0, -0.0]), TEClosure.CLOSED),  # equal values, as c2d compares them
    # The NACA formula's own residue at x=1: a "closed" TE that isn't, which BUFF would refuse
    (_coarse([1.0, -1.67e-17], [1.0, 1.67e-17]), TEClosure.NEAR_CLOSED),
    (_coarse([1.0, TE_GAP_MIN / 2], [1.0, 0.0]), TEClosure.NEAR_CLOSED),
    (_coarse([1.0, TE_GAP_MIN], [1.0, 0.0]), TEClosure.OPEN),
    (_coarse([1.0, 0.0063], [1.0, -0.0063]), TEClosure.OPEN),
])
def test_te_closure_classification(coords, expected):
    assert Common_classify_te_closure(coords) == expected


@pytest.mark.parametrize("coords, lo, hi", [
    (closed_naca0012("sharp", te="sharp").coords_tensor, 10.0, 25.0),  # the NACA0012 TE wedge
    (closed_naca0012("round", te="round").coords_tensor, 150.0, 180.0),
    (_flat_base_naca0012(), 179.9, 180.0),
])
def test_te_closure_angle(coords, lo, hi):
    assert lo <= Common_te_closure_angle(coords) <= hi


def test_sharpest_corner_finds_a_flat_base_the_closure_angle_cannot():
    # Why the corner scan exists: a flat base closed mid-base reads ~180 at the closure, like a round
    # tip, but turns ~80 deg at each end of the base
    coords = _flat_base_naca0012()
    turn, idx = Common_sharpest_corner(coords)

    assert Common_te_closure_angle(coords) > SHARP_TE_ANGLE
    assert turn > CORNER_TURN_LIMIT
    assert idx in (1, len(coords) - 2)  # one of the two base corners
    assert Common_sharpest_corner(closed_naca0012("round", te="round").coords_tensor)[0] < CORNER_TURN_LIMIT
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
# Closure is the hard requirement: c2d's BUFF mode refuses an open contour on either grid, gmsh's
# C-mesh builds its TE line between the two end points, and gmsh's O-mesh drops the last point as a
# duplicate of the first. Near-closed is rounding noise and fails everywhere.
coords_closed = _coarse([1.0, 0.0], [1.0, 0.0])
coords_near_closed = _coarse([1.0, 1e-7], [1.0, 0.0])
coords_open = _coarse([1.0, 0.0063], [1.0, -0.0063])


@pytest.mark.parametrize("backend, topology, coords_tensor", [
    (MeshBackend.GMSH, MeshTopology.CGRD, coords_open),
    (MeshBackend.GMSH, MeshTopology.OGRD, coords_closed),
    (MeshBackend.C2D, MeshTopology.CGRD, coords_closed),
    (MeshBackend.C2D, MeshTopology.OGRD, coords_closed),
])
def test_te_closure_pairing_accept(backend, topology, coords_tensor):
    is_valid, msg = Common_validate_te_closure(coords_tensor=coords_tensor, backend=backend, topology=topology)

    assert is_valid is True
    assert msg == ""


@pytest.mark.parametrize("backend, topology, coords_tensor, checklist", [
    (MeshBackend.GMSH, MeshTopology.CGRD, coords_closed, ["requires an open te", "closed"]),
    (MeshBackend.GMSH, MeshTopology.CGRD, coords_near_closed, ["requires an open te", "too short"]),
    (MeshBackend.GMSH, MeshTopology.OGRD, coords_open, ["requires a closed te", "open"]),
    (MeshBackend.GMSH, MeshTopology.OGRD, coords_near_closed, ["requires a closed te", "rounding noise"]),
    (MeshBackend.C2D, MeshTopology.CGRD, coords_open, ["requires a closed te", "open"]),
    (MeshBackend.C2D, MeshTopology.CGRD, coords_near_closed, ["requires a closed te", "rounding noise"]),
    (MeshBackend.C2D, MeshTopology.OGRD, coords_open, ["requires a closed te", "open"]),
    (MeshBackend.C2D, MeshTopology.OGRD, coords_near_closed, ["requires a closed te", "rounding noise"]),
    (MeshBackend.UNKNOWN, MeshTopology.OGRD, coords_closed, ["no trailing-edge requirement"]),
])
def test_te_closure_pairing_reject(backend, topology, coords_tensor, checklist):
    is_valid, msg = Common_validate_te_closure(coords_tensor=coords_tensor, backend=backend, topology=topology)

    assert is_valid is False
    assert all(cond in msg.lower() for cond in checklist)


_ROUND = closed_naca0012("round", te="round").coords_tensor
_SHARP = closed_naca0012("sharp", te="sharp").coords_tensor


# TE shape is advisory: each closed pairing warns on the shape it meshes poorly, and on corners
@pytest.mark.parametrize("backend, topology, coords_tensor, expected", [
    (MeshBackend.C2D, MeshTopology.OGRD, _SHARP, ["sharp te"]),
    (MeshBackend.C2D, MeshTopology.OGRD, _ROUND, []),
    (MeshBackend.C2D, MeshTopology.CGRD, _ROUND, ["blunt te"]),
    (MeshBackend.C2D, MeshTopology.CGRD, _SHARP, []),
    (MeshBackend.GMSH, MeshTopology.OGRD, _SHARP, ["sharp te"]),
    (MeshBackend.GMSH, MeshTopology.OGRD, _ROUND, []),
    (MeshBackend.GMSH, MeshTopology.CGRD, coords_open, []),  # an open TE has no closure to read
    (MeshBackend.C2D, MeshTopology.OGRD, _flat_base_naca0012(), ["corners mesh poorly"]),
])
def test_te_shape_warnings(backend, topology, coords_tensor, expected):
    warnings = Common_te_shape_warnings(coords_tensor=coords_tensor, backend=backend, topology=topology)

    assert len(warnings) == len(expected), warnings
    assert all(e in w.lower() for e, w in zip(expected, warnings))
# endregion