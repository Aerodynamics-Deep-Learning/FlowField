from pydantic import BaseModel, Field
from typing import Optional
from enum import IntEnum

from ...schemas import Airfoil, Freestream

class GMSH_ExitFlag(IntEnum):
    """
    Categorical flag for what happened in GMSH during/after mesh generation
    """
    BLUNTING_FAIL = -2
    FATAL_ERROR = -1 
    NEGATIVE_JACOBIAN = 0 # During flagging, we actually set the threshold for min mesh quality to be 0.0, this is because for example, a mesh quality of 0.05 is already bad enough for SU2 to flag it as "negative Jacobian"
    EXTRUSION_FAIL = 1
    SUCCESS = 2

class GMSH_MeshingConfig(BaseModel):
    """
    Contract for the configs of GMSH
    """
    # Airfoil specific params
    upper_anchor_idx: int = Field(65, ge=0, description="The index of the upper anchor point on the airfoil, which is a point of interest for meshing, typically where the curvature changes significantly")
    lower_anchor_idx: int = Field(95, ge=0, description="The index of the lower anchor point on the airfoil, which is a point of interest for meshing, typically where the curvature changes significantly")

    # Overall params
    wake_length: float = Field(20.0, description="Wake length")
    farfield_radius: float = Field(15.0, ge=15.0, description="The farfield radius around the airfoil, defines the outer boundary of the mesh")
    bl_thickness: float = Field(0.4, gt=0.0, description="The total thickness of the BL, where meshing is finer, given as the absolute spatial distance")
    target_yplus: float = Field(1.0, gt=0.0, description="The target yplus value for the first layer of meshing, to resolve the boundary layer with Spalart-Allmaras / k-omega SST in 2D. Assumed as 1.0 for the pipeline")

    # BL meshing configs
    nx_le1: int = Field(50, gt=5, description="Number of points in the leading edge region, where curvature is high, to ensure good resolution of the geometry")
    nx_le2: int = Field(50, gt=5, description="Number of points in the second region after the leading edge, where curvature is still relatively high, to ensure good resolution of the geometry")
    nx_upper: int = Field(200, gt=10, description="Number of points along the upper surface of the airfoil, excluding the leading edge region")
    nx_lower: int = Field(200, gt=10, description="Number of points along the lower surface of the airfoil, excluding the leading edge region")
    nx_wake: int = Field(200, gt=10, description="Number of points along the wake region, starting from the trailing edge and extending downstream")
    bl_growth_ratio: float = Field(1.075, gt=1.01, le=1.2, description="The growth ratio within the boundary layer meshing, to ensure smooth growth of mesh cells")
    wake_progression: float = Field(1.001, description="Wake progression")
    te_coarsen_factor: float = Field(600.0, gt=1.0, description="The coarsening factor for the trailing edge region, to allow for smoother transition from the fine mesh near the trailing edge to the coarser mesh in the wake, while avoiding abrupt changes in cell sizes that can lead to poor mesh quality")
    chord_bump: float = Field(50.0, gt=0.0, description="The chord bump is a small extension added to the chord length of the airfoil for meshing purposes, to ensure that the trailing edge region is properly resolved and to avoid issues with mesh generation at the trailing edge where the upper and lower surfaces meet. This helps in creating a more robust mesh around the trailing edge, which is critical for accurately capturing the flow features in that region")

    # Farfield meshing configs
    ff_growth_ratio: float = Field(1.1, gt=1.01, le=1.3, description="The growth ratio for the farfield meshing, to ensure smooth growth of mesh cells in the farfield region")

class GMSH_In(BaseModel):
    """
    Contract for the input given to GMSH for mesh generation
    """
    airfoil: Airfoil
    freestream: Freestream
    meshing_config: GMSH_MeshingConfig
    working_dir: str = Field(..., description="Directory where the generated mesh will be saved")

class GMSH_Out(BaseModel):
    """
    Contract for the output from GMSH post-mesh generation
    """
    airfoil: Airfoil
    freestream: Freestream
    flag: GMSH_ExitFlag
    mesh_path: Optional[str] = Field(None, description="The path of the generated mesh, if successful")
    min_mesh_quality: Optional[float] = Field(None, description="Minimuum mesh element quality")
    num_nodes: Optional[int] = Field(None, description="The number of nodes in the generated mesh")
    verbose_list: list[str | None] = Field(..., description="A list of paths for verbose output [gmsh_log_path (.txt), geometry_dump (.brep)]")

    

