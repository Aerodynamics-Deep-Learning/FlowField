from pydantic import BaseModel, Field
from typing import Optional
from enum import IntEnum

from ...schemas import Airfoil, Freestream

class GMSH_ExitFlag(IntEnum):
    """
    Categorical flag for what happened in GMSH during/after mesh generation
    """
    FATAL_ERROR = -1 
    NEGATIVE_JACOBIAN = 0 # During flagging, we actually set the threshold for min mesh quality to be 0.1, this is because for example, a mesh quality of 0.05 is already bad enough for SU2 to flag it as "negative Jacobian"
    EXTRUSION_FAIL = 1
    SUCCESS = 2

class GMSH_MeshingConfig(BaseModel):
    """
    Contract for the configs of GMSH
    """

    farfield_radius: float = Field(40.0, ge=15.0, description="The farfield radius around the airfoil, defines the outer boundary of the mesh")
    algorithm: int = Field(6, description="The integer representing the meshing algo to be used")

    # Unstructured res parms
    lc_leadingedge: float = Field(0.002, gt=0.0, description="Characteristic length (CL <-> LC) for the meshes at the leading edge, finer resolution needed to resolve stagnation point")
    lc_trailingedge: float = Field(0.004, gt=0.0, description="Characteristic length for the meshes at the trailing edge, finer resolution needed to enforce discrete Kutta condition")
    lc_farfield: float = Field(5.0, gt=0.0, description="Characteristic length for the meshes at the farfield, where coarser resolution is acceptable" )

    # Boundary layer params
    target_yplus: float = Field(1.0, gt=0.0, description="The target yplus value for the first layer of meshing, to resolve the boundary layer with Spalart-Allmaras / k-omega SST in 2D. Assumed as 1.0 for the pipeline")
    bl_growth_ratio: float = Field(1.15, gt=1.01, le=1.2, description="The growth ratio within the boundary layer meshing, to ensure smooth growth of mesh cells")
    bl_total_thickness: float = Field(0.05, gt=0.0, description="The total thickness of the BL, where meshing is finer, given as the absolute spatial distance")

    # Additions for robustness imprvement
    bl_fan_elements: int = Field(7, gt=0, description="The number of fan elements to be added at the sharp angles of the airfoil, to have a well defined mesh")
    smoothing_steps: int = Field(10, ge=0, description="The number of Laplacian smoothing steps to be applied, to improve mesh quality")

    # Markers
    marker_airfoil: str = Field("MARKER_AIRFOIL", description="The marker name for the airfoil boundary, i.e. physical group tag")
    marker_farfield: str = Field("MARKER_FARFIELD", description="The marker name for the farfield boundary, i.e. freestream boundary")
    marker_fluid: str = Field("FLUID_DOMAIN", description="The marker name for the fluid domain, i.e. physical group tag")

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

    

