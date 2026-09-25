from typing import Tuple

import torch

from src.datagen.schemas import Freestream
from .schemas import MeshBackend, MeshTopology

"""
Which TE shape each (backend, topology) pairing is built for: True = needs a blunt
(finite-gap) TE, False = needs a sharp (coincident-point) one.

The two backends' topology names deliberately do NOT line up, so this table can't be collapsed
into a single per-topology rule:
    - gmsh's C-mesh closes the domain on a finite trailing-edge line 
    - c2d is the other way round: it cuts its C-grid's wake from a sharp TE point and
      closes its O-grid on a blunt TE face. Confirmed against the binary itself, which prints
      "Sharp trailing edge: C-grid topology is recommended." / "Blunt trailing edge: O-grid
      topology is recommended." for it these are recommendations, not requirements; it meshes
      either pairing after a y/n override (see `c2d/run.py`).
"""
_TOPOLOGY_WANTS_BLUNT_TE = {
    (MeshBackend.GMSH, MeshTopology.CGRD): True,
    (MeshBackend.GMSH, MeshTopology.OGRD): False,
    (MeshBackend.C2D, MeshTopology.CGRD): False,
    (MeshBackend.C2D, MeshTopology.OGRD): True,
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
    return torch.linalg.norm(coords_tensor[0] - coords_tensor[-1]).item()


def Common_te_is_blunt(coords_tensor: torch.Tensor, micro_tol: float) -> bool:
    """
    The single blunt-vs-sharp predicate the topology mapping is built on.

    Args:
        coords_tensor (torch.Tensor): The tensor of the airfoil coords, in selig format
        micro_tol (float): The micro tolerance below which a gap counts as sharp

    Returns:
        bool: True if the trailing edge has a resolvable finite gap (blunt), False if its first
            and last points coincide or sit closer than `micro_tol` (sharp)
    """
    return Common_measure_te_gap(coords_tensor) >= micro_tol


def Common_validate_te_bluntness(coords_tensor: torch.Tensor, micro_tol: float) -> Tuple[bool, str]:
    """
    Validates that the trailing edge gap of an airfoil coordinate tensor is blunt.

    Note this only answers one direction: a sharp TE is a valid input to the pipeline, it just
    isn't meshable by every (backend, topology) pairing. Use `Common_validate_te_for_topology` for
    the pairing-aware check; this stays as the plain "is it blunt?" primitive.

    Args:
        coords_tensor (torch.Tensor): The tensor of the airfoil coords, in selig format
        micro_tol (float): The micro tolerance for bluntness

    Returns:
        tuple: (is_valid_bluntness, warning_message)
    """
    te_gap = Common_measure_te_gap(coords_tensor)

    if te_gap == 0.0:
        return False, f"Airfoil not blunted. Coincident points at {coords_tensor[0]}."
    elif te_gap < micro_tol:
        return False, f"Trailing edge gap too small: {te_gap}."
    else:
        return True, ""


def Common_validate_te_for_topology(coords_tensor: torch.Tensor, backend: MeshBackend,
                                    topology: MeshTopology, micro_tol: float = 1e-5) -> Tuple[bool, str]:
    """
    Validates the airfoil's trailing edge shape against what the requested (backend, topology)
    pairing is built for. The shared bluntness -> topology reconciliation both backends go through,
    rather than each reimplementing it (see `_TOPOLOGY_WANTS_BLUNT_TE` above).

    Args:
        coords_tensor (torch.Tensor): The tensor of the airfoil coords, in selig format
        backend (MeshBackend): The backend the mesh is being dispatched to
        topology (MeshTopology): The grid topology requested for that backend
        micro_tol (float): The micro tolerance below which a gap counts as sharp

    Returns:
        tuple: (is_valid_pairing, warning_message)
    """
    wants_blunt = _TOPOLOGY_WANTS_BLUNT_TE.get((backend, topology))
    if wants_blunt is None:
        return False, f"No trailing-edge requirement is known for backend={backend}/topology={topology}."

    pairing = f"backend={backend.value}/topology={topology.value}"

    if wants_blunt:
        is_valid, msg = Common_validate_te_bluntness(coords_tensor=coords_tensor, micro_tol=micro_tol)
        return (True, "") if is_valid else (False, f"{pairing} expects a blunt trailing edge. {msg}")

    if Common_te_is_blunt(coords_tensor=coords_tensor, micro_tol=micro_tol):
        te_gap = Common_measure_te_gap(coords_tensor)
        return False, (f"{pairing} expects a sharp trailing edge, but the airfoil's is blunt "
                       f"(gap: {te_gap}).")
    return True, ""

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

