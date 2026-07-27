import matplotlib.pyplot as plt
import torch
from typing import Tuple

from src.datagen.schemas import Freestream

def GMSH_get_mesh_height(Re: float, chord: float, target_yplus: float=1.0) -> float:
    """
    Gets the target height of the first layer of meshing given Re and chord length, and the target yplus

    Args:
        Re (float): The reynolds number
        chord (float): The chord length of the parameterized airfoil
        target_yplus (float): Our target yplus value, given the relative ease of compute (2D) and Spalart-Allmaras / k-omega SST, it is the main choice
    
    Returns:
        float: The height of the first mesh cell (radial out from the airfoil)
    """
    Cf = 0.058 * (Re ** -0.2)
    y_height = target_yplus * (chord / (Re * ((Cf / 2) ** 0.5)))
    return y_height

def GMSH_validate_tensor_numeric(coords_tensor: torch.Tensor) -> Tuple[bool, str]:
    """
    Validates the numeric validity of the airfoil coordinate tensor
    
    Args:
        coords_tensor (torch.Tensor): The tensor of the airfoil coords, in selig format
    
    Returns:
        tuple: (is_valid_tensor, warning_message)
    """
    if not isinstance(coords_tensor, torch.Tensor):
        return False, f"Airfoil input is not a torch.Tensor, it is: {type(coords_tensor)}"
    elif coords_tensor.ndim != 2 or coords_tensor.shape[1] not in (2, 3):
        return False, f"Airfoil input tensor has unexpected number of dims: {coords_tensor.ndim}, {coords_tensor.shape[1]}"
    elif torch.isnan(coords_tensor).any().item():
        return False, f"Airfoil input tensor has NaN values"
    elif torch.isinf(coords_tensor).any().item():
        return False, f"Airfoil input tensor has Inf values"
    else:
        return True, ""

def GMSH_validate_te_bluntness(coords_tensor: torch.Tensor, micro_tol: float = 1e-5) -> Tuple[bool, str]:
    """
    Validates the trailing edge gap of an airfoil coordinate tensor
    
    Args: 
        coords_tensor (torch.Tensor): The tensor of the airfoil coords, in selig format
        micro_tol (float): The micro tolerance for bluntness

    Returns:
        tuple: (is_valid_bluntness, warning_message)
    """
    te_gap = torch.linalg.norm(coords_tensor[0] - coords_tensor[-1]).item()

    if te_gap == 0.0:
        return False, f"Airfoil not blunted. Coincident points at {coords_tensor[0]}."
    elif te_gap < micro_tol:
        return False, f"Trailing edge gap too small: {te_gap}."
    else:
        return True, ""

def GMSH_validate_freestream_physicality(freestream: Freestream) -> Tuple[bool, str]:
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

def GMSH_export_sicn_histogram(qualities: list[float], save_path: str) -> None:
    """
    Generates and saves a histogram of mesh element qualities.
    """
    plt.figure(figsize=(8, 6))
    
    # Bins are set to capture the standard SICN domain [-1.0, 1.0] or [0.0, 1.0]
    plt.hist(qualities, bins=250, color='steelblue', edgecolor='black', alpha=0.8)
    
    plt.title("Mesh Element Quality Distribution (minSICN)")
    plt.xlabel("Signed Inverse Condition Number")
    plt.ylabel("Element Count")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()