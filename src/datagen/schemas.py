from pydantic import BaseModel, Field, ConfigDict
from enum import IntEnum
import numpy as np
import torch
from typing import Optional, Tuple

class ConvergenceFlag(IntEnum):
    """
    Strict convergence flag, to identify CFD solver convergence
    """
    FAILED = -1
    DIVERGED = 0
    OSCILLATORY = 1
    CONVERGED = 2

class SolverConfig(BaseModel):
    """
    Contract to define the configuration of the physics solver, such as XFoil or SU2
    """
    solver_name: str = Field(..., description="Name of the solver (e.g., 'XFoil', 'SU2')")
    max_iterations: int = Field(..., gt=0, description="Maximum number of iterations for the solver")
    convergence_criterion: float = Field(..., gt=0.0, description="Convergence criterion for the solver (e.g., residual threshold)")

class AirfoilParameters(BaseModel):
    """
    Contract to define the parameters of the airfoil, and its operational conditions
    """
    parameters: Tuple[float, ...] = Field(..., description="Vector of geometric parameters")
    mach_number: float = Field(..., ge=0.0, le=5.0)
    angle_of_attack: float = Field(..., ge=-30.0, le=30.0)
    altitude_m : float = Field(..., ge=0.0)
    reynolds_number: float = Field(..., gt=0.0)

class FlowTensor(BaseModel):
    """
    Contract for the multi-fidelity flowfield tensors obtained after solving
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    coords = torch.Tensor = Field(..., description="Tensor of shape (N, 2) contains x,y tuples for airfoil surface points. Is in the Selig format.")
    pressure_field: torch.Tensor
    velocity_field: torch.Tensor
    turbulent_viscosity: Optional[torch.Tensor] = None

    def validate_dims(self, expected_shape: tuple):
        if self.pressure_field.shape != expected_shape:
            raise ValueError(f"Tensor mismatch. Expected shape: {expected_shape}, but got: {self.pressure_field.shape}")
        
class SolverResult(BaseModel):
    """
    Contract for any physics solver's (XFoil, SU2, etc.) output
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    flag: ConvergenceFlag
    solver_config: SolverConfig
    history: np.ndarray = Field(..., description="Final 200 iterations of the force coeffs")
    flow_data: Optional[FlowTensor] = Field(None, description="None if flag=DIVERGED")
    compute_time_sec: float
    solver: str = Field(..., description="Name of the solver used to generate this result")
