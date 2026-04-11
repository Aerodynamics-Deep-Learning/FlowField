from pydantic import BaseModel, Field
import torch
from typing import Optional

class Airfoil(BaseModel):
    airfoil_name: Optional[str] = Field(None, description="Name of the airfoil used")
    coords_tensor: torch.Tensor = Field(..., description="The geometry tensor of the airfoil in Selig format")
    coords_path: Optional[str] = Field(None, description="The path to the geometry file in Selig format, ending with .dat")
    alpha: float = Field(..., ge=-30.0, le=30.0, description="Angle of attack in degrees")
    Re: float = Field(..., gt=0.0, description="Reynolds number")
    mach: float = Field(..., ge=0.0, le=0.75, description="Mach number")
    altitude_m : float = Field(..., ge=0.0, description="Assumed altitude")