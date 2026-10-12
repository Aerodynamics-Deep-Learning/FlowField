from enum import Enum
from typing import Tuple

import numpy as np
import torch

from src.datagen.schemas import Freestream
from .schemas import MeshBackend, MeshTopology
from .constants import TE_GAP_MIN, SHARP_TE_ANGLE, CORNER_TURN_LIMIT


class TEClosure(str, Enum):
    """How a Selig contour's first and last points relate."""
    CLOSED = "closed"            # identical points
    NEAR_CLOSED = "near_closed"  # 0 < gap < TE_GAP_MIN: rounding noise, not geometry
    OPEN = "open"                # gap >= TE_GAP_MIN


class TEShape(str, Enum):
    """A closed contour's TE, read off its closure angle."""
    SHARP = "sharp"
    BLUNT = "blunt"  # rounded or flat-based


"""
Per (backend, topology): the closure it requires (hard gate) and the TE shape it warns on (advisory,
None for no shape check). The backends' topology names don't line up, hence one row per pairing.
    - c2d meshes in BUFF mode, which refuses an open contour on either grid
    - gmsh's C-mesh draws its TE line from the first point to the last, so they must differ
    - gmsh's O-mesh treats the TE as one point and drops the last point, which must duplicate the first
"""
_TE_REQUIREMENTS = {
    (MeshBackend.C2D, MeshTopology.OGRD): (TEClosure.CLOSED, TEShape.SHARP),
    (MeshBackend.C2D, MeshTopology.CGRD): (TEClosure.CLOSED, TEShape.BLUNT),
    (MeshBackend.GMSH, MeshTopology.CGRD): (TEClosure.OPEN, None),
    (MeshBackend.GMSH, MeshTopology.OGRD): (TEClosure.CLOSED, TEShape.SHARP),
}

def Common_validate_freestream_physicality(freestream: Freestream) -> Tuple[bool, str]:
    """
    Validates the physicality of the input freestream, to prevent non-sensical first mesh cell height calcs

    Args:
        freestream (Freestream): The freestream class

    Returns:
        tuple: (is_valid_bluntness, warning_message)
    """
    if freestream.mach < 0:
        return False, f"Freestream has unexpected mach number: {freestream.mach}"
    elif freestream.Re < 0:
        return False, f"Freestream has unexpected Reynolds number: {freestream.Re}"
    elif freestream.altitude_m < 0:
        return False, f"Freestream has unexpected altitude: {freestream.altitude_m}"
    else:
        return True, ""

def Common_measure_te_gap(coords_tensor: torch.Tensor) -> float:
    """
    Measures the trailing edge gap of an airfoil coordinate tensor, i.e. the distance between the
    first and last points of the selig-ordered contour.

    Args:
        coords_tensor (torch.Tensor): The tensor of the airfoil coords, in selig format

    Returns:
        float: The trailing edge gap
    """
    return torch.linalg.norm(coords_tensor[0, :2] - coords_tensor[-1, :2]).item()


def Common_classify_te_closure(coords_tensor: torch.Tensor) -> TEClosure:
    """
    Classifies the contour's closure. Closed means bit-identical end points, since c2d compares them
    exactly; anything short of that below `TE_GAP_MIN` is rounding noise.

    Args:
        coords_tensor (torch.Tensor): The tensor of the airfoil coords, in selig format

    Returns:
        TEClosure: CLOSED, NEAR_CLOSED or OPEN
    """
    if torch.equal(coords_tensor[0, :2], coords_tensor[-1, :2]):
        return TEClosure.CLOSED
    return TEClosure.NEAR_CLOSED if Common_measure_te_gap(coords_tensor) < TE_GAP_MIN else TEClosure.OPEN


def Common_te_closure_angle(coords_tensor: torch.Tensor) -> float:
    """
    Interior angle at a closed contour's closure point, between its first and last panels: ~180 through
    a round tip or a flat base, the wedge angle at a sharp TE.

    Args:
        coords_tensor (torch.Tensor): The tensor of a closed airfoil contour, in selig format

    Returns:
        float: The angle in degrees, NaN if a panel at the closure point has zero length
    """
    pts = coords_tensor[:, :2].detach().cpu().double().numpy()
    u, v = pts[1] - pts[0], pts[-2] - pts[0]
    norms = np.linalg.norm(u) * np.linalg.norm(v)
    if norms == 0.0:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / norms, -1.0, 1.0))))


def Common_sharpest_corner(coords_tensor: torch.Tensor) -> Tuple[float, int]:
    """
    The largest direction change between consecutive panels, over interior vertices only (the closure
    point is `Common_te_closure_angle`'s). Zero-length panels read as straight.

    Args:
        coords_tensor (torch.Tensor): The tensor of the airfoil coords, in selig format

    Returns:
        tuple: (turn in degrees, index of the vertex it is at)
    """
    panels = np.diff(coords_tensor[:, :2].detach().cpu().double().numpy(), axis=0)
    a, b = panels[:-1], panels[1:]
    turn = np.degrees(np.abs(np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], np.sum(a * b, axis=1))))
    i = int(np.argmax(turn))
    return float(turn[i]), i + 1


def Common_validate_te_closure(coords_tensor: torch.Tensor, backend: MeshBackend,
                               topology: MeshTopology) -> Tuple[bool, str]:
    """
    The hard TE gate: the contour's closure against what the (backend, topology) pairing requires
    (see `_TE_REQUIREMENTS`). Near-closed fails every pairing.

    Args:
        coords_tensor (torch.Tensor): The tensor of the airfoil coords, in selig format
        backend (MeshBackend): The backend the mesh is being dispatched to
        topology (MeshTopology): The grid topology requested for that backend

    Returns:
        tuple: (is_valid_closure, warning_message)
    """
    requirement = _TE_REQUIREMENTS.get((backend, topology))
    if requirement is None:
        return False, f"No trailing-edge requirement is known for backend={backend}/topology={topology}."

    required = requirement[0]
    closure = Common_classify_te_closure(coords_tensor)
    if closure == required:
        return True, ""

    pairing = f"backend={backend.value}/topology={topology.value}"
    gap = Common_measure_te_gap(coords_tensor)
    if required == TEClosure.CLOSED and closure == TEClosure.NEAR_CLOSED:
        return False, (f"{pairing} requires a closed TE, but the first and last points differ by "
                       f"{gap:.3e} (rounding noise); set the last point equal to the first upstream.")
    if required == TEClosure.CLOSED:
        return False, f"{pairing} requires a closed TE, but the contour is open (gap {gap:.3e})."
    if closure == TEClosure.NEAR_CLOSED:
        return False, (f"{pairing} requires an open TE, but its gap {gap:.3e} is below {TE_GAP_MIN:g}, "
                       f"too short to build a TE line on.")
    return False, f"{pairing} requires an open TE to build its TE line on, but the contour is closed."


def Common_te_shape_warnings(coords_tensor: torch.Tensor, backend: MeshBackend,
                             topology: MeshTopology) -> list[str]:
    """
    The advisory TE checks, for pairings that require a closed contour: a TE shape the pairing meshes
    poorly (`_TE_REQUIREMENTS`), and corners anywhere on the contour, which the closure angle cannot
    see (a flat base closed mid-base reads ~180 there, like a round tip).

    Args:
        coords_tensor (torch.Tensor): The tensor of a closed airfoil contour, in selig format
        backend (MeshBackend): The backend the mesh is being dispatched to
        topology (MeshTopology): The grid topology requested for that backend

    Returns:
        list[str]: One message per finding, empty if none
    """
    requirement = _TE_REQUIREMENTS.get((backend, topology))
    # Under 3 points there is no panel pair to read a shape off
    if requirement is None or requirement[0] != TEClosure.CLOSED or coords_tensor.shape[0] < 3:
        return []

    pairing = f"backend={backend.value}/topology={topology.value}"
    warnings = []
    angle = Common_te_closure_angle(coords_tensor)
    if np.isnan(angle):
        warnings.append(f"{pairing}: cannot read the TE angle, a panel at the closure point has zero length.")
    else:
        shape = TEShape.SHARP if angle < SHARP_TE_ANGLE else TEShape.BLUNT
        if shape == requirement[1]:
            warnings.append(f"{pairing} is not built for a {shape.value} TE (closure angle {angle:.1f} deg); "
                            f"expect a lower-quality mesh.")

    turn, i = Common_sharpest_corner(coords_tensor)
    if turn > CORNER_TURN_LIMIT:
        x, y = coords_tensor[i, 0].item(), coords_tensor[i, 1].item()
        warnings.append(f"Contour turns {turn:.0f} deg at point {i} ({x:.5f}, {y:.5f}); corners mesh poorly.")
    return warnings


def Common_validate_tensor_numeric(coords_tensor: torch.Tensor, chord_tol: float = 1e-3) -> Tuple[bool, str]:
    """
    Validates the numeric validity of the airfoil coordinate tensor, including the pipeline-wide
    invariant that it is chord-normalized: x runs from 0 (LE) to 1 (TE).

    `Airfoil.chord` carries the real physical size separately and scales mesh lengths by it, so a
    tensor that is not unit-chord would be scaled twice. TThe tolerance is sized for scale errors,
    which are order-of-magnitude, not for float precision.

    Args:
        coords_tensor (torch.Tensor): The tensor of the airfoil coords, in selig format
        chord_tol (float): Allowed deviation of min(x) from 0 and max(x) from 1

    Returns:
        tuple: (is_valid_tensor, warning_message)
    """
    if not isinstance(coords_tensor, torch.Tensor):
        return False, f"Airfoil input is not a torch.Tensor, it is: {type(coords_tensor)}"
    # Empty is rejected here too, before min/max below raise on it
    elif coords_tensor.ndim != 2 or coords_tensor.shape[0] == 0 or coords_tensor.shape[1] not in (2, 3):
        return False, (f"Airfoil input tensor has unexpected shape: {tuple(coords_tensor.shape)}, "
                       f"expected (N, 2) or (N, 3) with N > 0")
    elif torch.isnan(coords_tensor).any().item():
        return False, f"Airfoil input tensor has NaN values"
    elif torch.isinf(coords_tensor).any().item():
        return False, f"Airfoil input tensor has Inf values"

    # Checked after NaN/Inf: min/max are meaningless until those are ruled out
    x_min = coords_tensor[:, 0].min().item()
    x_max = coords_tensor[:, 0].max().item()
    if abs(x_min) > chord_tol or abs(x_max - 1.0) > chord_tol:
        return False, (f"Airfoil input tensor is not chord-normalized: x spans [{x_min}, {x_max}], "
                       f"expected [0, 1] within {chord_tol}. Physical size belongs in Airfoil.chord.")

    return True, ""

