from pydantic import BaseModel, Field, ConfigDict
from enum import IntEnum
import numpy as np
import torch
from typing import Optional, Tuple

class Airfoil(BaseModel):
    coords_tensor: torch.Tensor = Field(..., description="The geometry tensor of the airfoil in Selig format")
    alpha: float = Field(..., ge=-30.0, le=30.0, description="Angle of attack in degrees")
    Re: float = Field(..., gt=0.0, description="Reynolds number")
    mach: float = Field(..., ge=0.0, le=0.75, description="Mach number")
    altitude_m : float = Field(..., ge=0.0, description="Assumed altitude")

class XFoilConvergenceFlag(IntEnum):
    """
    Strict convergence flag, to identify CFD solver convergence
    """
    FAILED = -1
    DIVERGED = 0
    OSCILLATORY = 1
    STAGNATED = 2
    ITER_LIMITED = 3
    CONVERGED = 4

class XFoilSolverConfig(BaseModel):
    """
    Contract for the configuration of the XFoil solver and the convergence criteria.
    """
    n_panels: int = Field(..., gt=0, description="Number of panels to be used for the panel method in XFoil")
    tolerance: float = Field(..., description="The convergence tolerance for XFoil")
    max_iterations: int = Field(..., gt=0, description="Maximum number of iterations for XFoil to run")
    tail_length: int = Field(15, gt=0, description="The window to be used when checking resids for convergence flag assignment")
    var_thresh: float = Field(1e-5, gt=0.0, description="The threshold for the variance at tail when checking for oscillations for convergence flag assignment")
    slope_thresh: float = Field(-0.01, description="The threshold for the slope at tail when checking for stagnation for convergence flag assignment")

class XFoil_WarmStartIn(BaseModel):
    """
    Contract for the input given to XFoil, to get the warmstart values for SU2
    """
    airfoil: Airfoil
    work_dir: str = Field(..., description="Directory where the XFoil input script and Cp output will be saved")
    solver_config: XFoilSolverConfig

class XFoil_WarmStartOut(BaseModel):
    """
    Contract for the output given by XFoil, to conduct warmstart in SU2
    """
    
    flag: XFoilConvergenceFlag


