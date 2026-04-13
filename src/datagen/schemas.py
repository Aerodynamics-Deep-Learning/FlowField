from pydantic import BaseModel, Field
import torch
from typing import Optional

class Airfoil(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    airfoil_name: Optional[str] = Field(None, description="Name of the airfoil used")
    coords_tensor: torch.Tensor = Field(..., description="The geometry tensor of the airfoil in Selig format")
    coords_path: Optional[str] = Field(None, description="The path to the geometry file in Selig format, ending with .dat")
    chord: float = Field(..., description="The chord length of the parameterized airfoil")
    le_idx: int = Field(..., description="The index within the tensor that defines the leading edge")

class Freestream(BaseModel):

    alpha: float = Field(..., ge=-30.0, le=30.0, description="Angle of attack in degrees")
    Re: float = Field(..., gt=0.0, description="Reynolds number")
    mach: float = Field(..., ge=0.0, le=0.75, description="Mach number")
    altitude_m : float = Field(..., ge=0.0, description="Assumed altitude")