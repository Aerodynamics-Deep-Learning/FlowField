from pydantic import BaseModel, Field, model_validator
from typing import Optional, Union
from enum import IntEnum, Enum

from src.datagen.schemas import Airfoil, Freestream

class GMSH_ExitFlag(IntEnum):
    """
    Categorical flag for what happened in GMSH during/after mesh generation
    """

    FATAL_ERROR = -2 # An undocumented/classified error
    EXTRUSION_FAIL = -1 # Fail in creating a grid
    SUCCESS = 0 # Successful exit, not necessarily a good grid

class GMSH_Topology(str, Enum):
    """
    Which grid topology GMSH should generate. C-mesh (C-mesh) handles blunt (finite-thickness) trailing
    edges; O-mesh (O-grid) requires a sharp/curved trailing edge (upper/lower surfaces meeting at one point).
    """
    CGRD = "CGRID"
    OGRD = "OGRID"

class GMSH_CMeshingConfig(BaseModel):
    """
    Contract for the configs of GMSH's C-mesh topology (blunt trailing edge)
    """
    model_config = {"extra": "forbid"}

    # Airfoil specific params
    upper_anchor_idx: int = Field(65, ge=0, description="The index of the upper anchor point on the airfoil, which is a point of interest for meshing, typically where the curvature changes significantly")
    lower_anchor_idx: int = Field(95, ge=0, description="The index of the lower anchor point on the airfoil, which is a point of interest for meshing, typically where the curvature changes significantly")

    # Overall params
    wake_length: float = Field(20.0, description="Wake length")
    farfield_radius: float = Field(15.0, ge=15.0, description="The farfield radius around the airfoil, defines the outer boundary of the mesh")
    bl_thickness: float = Field(0.4, gt=0.0, description="The total thickness of the BL, where meshing is finer, given as the absolute spatial distance")
    target_yplus: float = Field(0.75, gt=0.0, description="The target yplus value for the first layer of meshing, to resolve the boundary layer with Spalart-Allmaras / k-omega SST in 2D. Below 1 because the flat-plate correlation behind it underestimates wall shear near the LE")

    # BL meshing configs
    nx_le1: int = Field(50, gt=5, description="Number of points in the leading edge region, where curvature is high, to ensure good resolution of the geometry")
    nx_le2: int = Field(50, gt=5, description="Number of points in the second region after the leading edge, where curvature is still relatively high, to ensure good resolution of the geometry")
    nx_upper: int = Field(200, gt=10, description="Number of points along the upper surface of the airfoil, excluding the leading edge region")
    nx_lower: int = Field(200, gt=10, description="Number of points along the lower surface of the airfoil, excluding the leading edge region")
    nx_wake: int = Field(200, gt=10, description="Number of points along the wake region, starting from the trailing edge and extending downstream")
    bl_growth_ratio: float = Field(1.075, gt=1.01, le=1.2, description="The growth ratio within the boundary layer meshing, to ensure smooth growth of mesh cells")
    wake_progression: float = Field(1.001, description="Wake progression")
    te_coarsen_factor: float = Field(600.0, gt=1.0, description="The coarsening factor for the trailing edge region, to allow for smoother transition from the fine mesh near the trailing edge to the coarser mesh in the wake, while avoiding abrupt changes in cell sizes that can lead to poor mesh quality")
    chord_bump: float = Field(0.75, gt=0.0, description="Transfinite 'Bump' clustering coefficient along the chordwise block edges; dimensionless, not a length, so it is never scaled by chord")

    # Farfield meshing configs
    ff_growth_ratio: float = Field(1.1, gt=1.01, le=1.3, description="The growth ratio for the farfield meshing, to ensure smooth growth of mesh cells in the farfield region")

class GMSH_OMeshingConfig(BaseModel):
    """
    Contract for the configs of GMSH's O-mesh topology (sharp trailing edge)
    """
    model_config = {"extra": "forbid"}

    # Airfoil surface discretization
    nx_afoil: int = Field(120, gt=10, description="Number of mesh points along each of the upper and lower airfoil splines (LE to TE)")

    # Radial discretization, cell counts derived from these like the C-mesh's
    target_yplus: float = Field(0.75, gt=0.0, description="The target yplus value for the first layer of meshing, same as GMSH_CMeshingConfig's")
    bl_thickness: float = Field(0.4, gt=0.0, description="Radial distance from the wall over which bl_growth_ratio applies, before switching to ff_growth_ratio")
    bl_growth_ratio: float = Field(1.075, gt=1.01, le=1.2, description="The radial growth ratio within the boundary layer, starting from the y+-derived first cell height")
    ff_growth_ratio: float = Field(1.1, gt=1.01, le=1.3, description="The radial growth ratio from the boundary layer edge out to the farfield boundary")

    # Farfield circle placement, in chord-normalized units
    farfield_radius: float = Field(12.0, gt=0.0, description="The radius of the outer O-grid circle, in chord-normalized units")
    farfield_center_offset: float = Field(0.4, description="The x-offset of the outer O-grid circle's center from the airfoil's leading edge, in chord-normalized units")

# Which config class each topology meshes with, the two build different geometries
_CONFIG_FOR_TOPOLOGY = {
    GMSH_Topology.CGRD: GMSH_CMeshingConfig,
    GMSH_Topology.OGRD: GMSH_OMeshingConfig,
}

GMSH_MeshingConfig = Union[GMSH_CMeshingConfig, GMSH_OMeshingConfig]


class GMSH_In(BaseModel):
    """
    Contract for the input given to GMSH for mesh generation
    """
    # extra="forbid": see MeshIn for why a mistyped config keyword must not silently fall back to defaults
    model_config = {"extra": "forbid"}

    airfoil: Airfoil
    freestream: Freestream
    topology: GMSH_Topology
    meshing_config: Optional[GMSH_MeshingConfig] = Field(
        None, description="The meshing config for this topology. Left unset, the topology's own "
        "config class is instantiated with its defaults; a plain dict is coerced to that class."
    )
    working_dir: str = Field(..., description="Directory where the generated mesh will be saved")

    @model_validator(mode="before")
    @classmethod
    def _resolve_meshing_config(cls, raw):
        """
        Resolves `meshing_config` against the requested topology.

        `topology` determines the config class outright, so one field suffices, but that makes
        `GMSH_MeshingConfig` a bare union, and pydantic resolves a dict against a union by first
        match. Both config classes are all-defaulted, so `meshing_config={}` would silently
        validate as the C-mesh one. Running before field validation, where the topology is known,
        keeps it unambiguous. Mirrors `common.schemas.MeshIn._resolve_mesh_config`.
        """
        if not isinstance(raw, dict):
            return raw
        try:
            topology = GMSH_Topology(raw["topology"])
        except (KeyError, ValueError):
            return raw  # Missing/unparseable: let normal field validation report it properly

        expected = _CONFIG_FOR_TOPOLOGY[topology]
        config = raw.get("meshing_config")
        if config is None:
            config = expected()
        elif isinstance(config, dict):
            config = expected(**config)
        elif not isinstance(config, expected):
            raise ValueError(
                f"topology={topology.name} requires {expected.__name__}, got {type(config).__name__}"
            )
        return {**raw, "meshing_config": config}  # A copy: model_validate hands over the caller's own dict

class GMSH_Out(BaseModel):
    """
    Contract for the output from GMSH post-mesh generation
    """
    # extra="forbid": see MeshIn, keeps the output contract self-enforcing for out-of-tree callers
    model_config = {"extra": "forbid"}

    airfoil: Airfoil
    freestream: Freestream
    flag: GMSH_ExitFlag
    mesh_path: Optional[str] = Field(None, description="The path of the generated mesh, if successful")
    mesh_path_vtk: Optional[str] = Field(None, description="The path of the generated mesh in .vtk format, if successful")
    num_nodes: Optional[int] = Field(None, description="The number of nodes in the generated mesh")
    verbose_list: list[str | None] = Field(..., description="A list of paths for verbose output [gmsh_log_path (.txt), geometry_dump (.brep), exception (.txt)]")

    

