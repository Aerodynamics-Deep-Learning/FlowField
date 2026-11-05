from pydantic import BaseModel, Field, field_serializer, field_validator
import numpy as np
from typing import Optional

class Airfoil(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    airfoil_name: str = Field("not_given", min_length=1, description="Name of the airfoil used, both meshing backends build their output filenames from it")
    coords: np.ndarray = Field(..., description="The airfoil's coordinates in Selig format, (N, 2), float64")
    coords_path: Optional[str] = Field(None, description="The path to the geometry file in Selig format, ending with .dat")
    chord: float = Field(1, gt=0.0, description="The chord length of the parameterized airfoil")
    le_idx: int = Field(..., description="The index within the coords that defines the leading edge")

    @field_validator("coords", mode="before")
    @classmethod
    def _as_float64(cls, v):
        # Lists and float32 arrays alike; shape and values are checked by the consumers (meshing's entry)
        try:
            return np.asarray(v, dtype=np.float64)
        except (TypeError, ValueError) as e:  # pydantic reports a ValueError, a TypeError would escape it
            raise ValueError(f"coords must be numeric: {e}") from e

    @field_serializer("coords", when_used="json")
    def _as_list(self, v):
        return v.tolist()

class Freestream(BaseModel):

    alpha: float = Field(..., ge=-5.0, le=15.0, description="Angle of attack in degrees")
    Re: float = Field(..., gt=1.5e6, le=1.5e7, description="Reynolds number")
    mach: float = Field(..., ge=0.15, le=0.75, description="Mach number") # Considering to change
    # ISA
    altitude_m : float = Field(0.0, ge=0.0, description="Assumed altitude")
    temp: float = Field(288.15, ge=0.0, le=400.0, description="Assumed freestream temp")
