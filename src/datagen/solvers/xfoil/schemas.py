from pydantic import BaseModel, Field
from enum import IntEnum
import torch
from typing import Optional

from ...schemas import Airfoil, Freestream

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

class XFoilConvergenceConfig(BaseModel):
    """
    Contract for the convergence criteria.
    """
    tail_length: int = Field(15, gt=0, description="The window to be used when checking resids for convergence flag assignment")
    var_thresh: float = Field(1e-5, gt=0.0, description="The threshold for the variance at tail when checking for oscillations for convergence flag assignment")
    slope_thresh: float = Field(-0.01, description="The threshold for the slope at tail when checking for stagnation for convergence flag assignment")
    tolerance: float = Field(1e-4, description="The convergence tolerance for XFoil")

class XFoilSolverConfig(BaseModel):
    """
    Contract for the configuration of the XFoil solver and the convergence criteria.
    """
    n_panels: int = Field(..., gt=0, description="Number of panels to be used for the panel method in XFoil")
    max_iterations: int = Field(..., gt=0, description="Maximum number of iterations for XFoil to run")
    timeout_sec: int = Field(60, ge=0, le=120, description="The max allowed timeout limit")

class XFoil_WarmStartIn(BaseModel):
    """
    Contract for the input given to XFoil, to get the warmstart values for SU2
    """
    airfoil: Airfoil
    freestream: Freestream
    xfoil_exe: str = Field(..., description="Path to the XFoil executable")
    working_dir: str = Field(..., description="Directory where the XFoil input script and Cp output will be saved")
    solver_config: XFoilSolverConfig = Field(..., description="Specifics of the solver")
    conv_config: XFoilConvergenceConfig = Field(..., description="Specifics of convergence criteria")

class XFoil_WarmStartOut(BaseModel):
    """
    Contract for the output given by XFoil, to conduct warmstart in SU2
    """
    model_config = {"arbitrary_types_allowed": True}

    airfoil: Airfoil
    freestream: Freestream
    flag: XFoilConvergenceFlag
    Cp_tensor: Optional[torch.Tensor] = Field(None, description="The dimensionless Cp distribution over the airfoil, with form (N, [x, Cp])")
    conservative_tensor: Optional[torch.Tensor] = Field(None, description="The conservative tensor of form [rho, rho*u, rh*v, rho*E] to be used in the warmstart of SU2")
    verbose_list: list[str] = Field(..., description="A list to produce the verbose output. [input_script_path (.txt), cp_file_path (.dat), stdout_path (.txt)]")

