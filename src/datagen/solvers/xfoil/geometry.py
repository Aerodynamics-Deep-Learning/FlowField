"""
Code to prepare the airfoil geometry for XFoil
"""

import os

from torch import Tensor
import numpy as np

import logging
logger = logging.getLogger(__name__)

def XFoil_Geometry_Write(coords_tensor: Tensor, working_dir: str, filename: str="geometry.dat", airfoil_name: str="Unnamed Airfoil") -> str:
    """
    Writes the airfoil geometry coords in the Selig format given the tensor to a .dat file format readable by XFoil, and returns the path.

    Args:
        coords_tensor (Tensor): Tensor of shape (N, 2) contains x,y tuples for airfoil surface points. Is in the Selig format.
        working_dir (str): Directory where the geometry file will be saved.
        filename (str, optional): Name of the geometry file. Defaults to "geometry.dat".
        airfoil_name (str, optional): The name of the airfoil. Defaults to "Unnamed Airfoil".
    
    Returns:
        str: Path to the saved geometry .dat file.
    """

    # I/O checks
    #if not os.path.exists(working_dir):
    #    os.makedirs(working_dir)
    #    logger.info(f"Created working directory at {working_dir} for XFoil geometry.")
    if not filename.endswith(".dat"):
        logger.warning(f"Filename {filename} does not have a .dat extension, it will be converted.")
        filename = ".".join(filename.split(".")[:-1]) + ".dat"
    #if os.path.exists(os.path.join(working_dir, filename)):
    #    logger.warning(f"Geometry file {filename} already exists in {working_dir}. It will be overwritten.")

    # Convert tensor to numpy
    if hasattr(coords_tensor, 'detach'):
        coords = coords_tensor.detach().cpu().numpy().astype(np.float64)
    else:
        coords = np.array(coords_tensor, dtype=np.float64)

    coords_path = os.path.join(working_dir, filename)

    np.savetxt(
        coords_path,
        coords,
        fmt="%10.6f",
        delimiter=' ',
        header=airfoil_name,
        comments=''
    )

    return coords_path
    