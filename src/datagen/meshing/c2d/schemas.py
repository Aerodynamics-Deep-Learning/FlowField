from pydantic import BaseModel, Field, model_validator
from typing import Optional, Literal
from enum import IntEnum, Enum

from src.datagen.geometry.sample.fields import SamplingFields
from src.datagen.schemas import Airfoil, Freestream

class C2D_ExitFlag(IntEnum):
    """
    Categorical flag for what happened in C2D during/after mesh generation.
    """

    EXECUTABLE_NOT_FOUND = -4   # .exe not found
    SUBPROCESS_FAIL = -3        # process crashed, timed out, or never wrote a .p3d
    FATAL_ERROR = -2            # unhandled exception elsewhere in the pipeline
    CONVERSION_FAIL = -1        # .p3d produced, but .p3d -> .su2 conversion never yielded a file
    SUCCESS = 0

class C2D_Topology(str, Enum):
    """
    Which grid topology c2d should generate; its own native OGRD (O-grid)/CGRD (C-grid)
    choice, independent of gmsh's cmesh/omesh (see `gmsh.schemas.GMSH_Topology`'s docstring and
    `tests/integration/datagen/mesh/conftest.py`'s `sterile_c2d_input`/`sterile_c2d_cgrd_input`
    for why those two backends' topology names don't line up letter-for-letter).
    """
    OGRD = "OGRD"
    CGRD = "CGRD"

class C2D_MeshingConfig(SamplingFields):
    """
    Contract for the configs of C2D. Surface points come from the input as given (c2d runs in BUFF
    mode); the sampling knobs inherited from `SamplingFields` set them, read by the sampler before
    meshing, never by c2d.
    """
    model_config = {"extra": "forbid"}

    topo: Optional[Literal["OGRD", "CGRD"]] = Field(
        None, description="The topology option. Left unset, it's filled in automatically from "
        "C2D_In.topology; set explicitly only to additionally assert it matches that selection."
    )
    slvr: Literal["HYPR", "ELLP"] = Field("HYPR", description="The solver option")

    # Normal/wake resolution (fixed)
    jmax: int = Field(200, gt=10, description="Number of points normal to the surface (wall to farfield)")
    nwke: int = Field(100, gt=1, description="Number of points along the wake (C-grid only)")

    # Farfield/domain sizing (fixed); fdst, fwkl and fwki reach the elliptic solver only
    radi: float = Field(20.0, gt=0.0, description="Farfield radius, in chords")
    fdst: float = Field(1.0, gt=0.0, description="O-grid farfield point spacing: 1 uniform, > 1 clustered toward the TE")
    fwkl: float = Field(1.0, gt=0.0, description="C-grid farfield wake length, as a ratio of the wake's own")
    fwki: float = Field(2.0, gt=0.0, description="C-grid farfield initial wake spacing, as a ratio of the wake's own")

    # Boundary-layer physics (fixed); the runner writes the freestream's Re as c2d's `recd`
    ypls: float = Field(0.7, gt=0.0, description="Target y+ for the first cell")
    cfrc: float = Field(0.5, ge=0.0, description="Reference length for the first cell's y+, as a fraction of the chord")

    # Elliptic solver controls (fixed; HYPR ignores them)
    stp1: int = Field(1000, gt=0, description="Initial grid smoothing iterations")
    stp2: int = Field(20, gt=0, description="Final grid smoothing iterations")
    nrmt: int = Field(1, ge=0, description="Top-surface point, counted from the TE, where the surface-normal condition starts")
    nrmb: int = Field(1, ge=0, description="Bottom-surface point, counted from the TE, where the surface-normal condition starts")

    # Recommended smoothing knobs (hyperbolic solver)
    epse: float = Field(0.0, ge=0.0, description="Explicit smoothing coefficient")
    asmt: int = Field(40, ge=5, le=80, description="Area-smoothing sweeps")
    epsi: float = Field(20.0, ge=2.0, le=40.0, description="Implicit smoothing")
    funi: float = Field(0.02, ge=0.005, le=0.3, description="Farfield uniformity")
    alfa: float = Field(1.0, ge=0.5, le=2.0, description="Marching implicitness")

    def to_params(self) -> dict:
        """Flatten to the plain dict `run._write_grid_options`/`C2D_generate_mesh` expect."""
        return self.model_dump()

class C2D_In(BaseModel):
    """
    Contract for the input given to c2d for mesh generation
    """
    # extra="forbid": see MeshIn, keeps all three input contracts rejecting mistyped keywords
    model_config = {"extra": "forbid"}

    airfoil: Airfoil
    freestream: Freestream
    topology: C2D_Topology
    meshing_config: C2D_MeshingConfig
    working_dir: str = Field(..., description="Directory where the generated mesh will be saved")
    write_vtk: bool = Field(True, description="Also convert the .su2 to the diagnostic .vtk")

    @model_validator(mode="after")
    def _sync_topo_with_topology(self) -> "C2D_In":
        if self.meshing_config.topo is None:
            # A copy: pydantic keeps the caller's instance, which may be reused for the other topology
            self.meshing_config = self.meshing_config.model_copy(update={"topo": self.topology.value})
        elif self.meshing_config.topo != self.topology:
            raise ValueError(
                f"topology={self.topology!r} does not match meshing_config.topo="
                f"{self.meshing_config.topo!r}"
            )
        return self

class C2D_Out(BaseModel):
    """
    Contract for the output from c2d post-mesh generation
    """
    # extra="forbid": see MeshIn, keeps the output contract self-enforcing for out-of-tree callers
    model_config = {"extra": "forbid"}

    airfoil: Airfoil
    freestream: Freestream
    flag: C2D_ExitFlag
    mesh_path: Optional[str] = Field(None, description="The path of the generated mesh, if successful")
    mesh_path_vtk: Optional[str] = Field(None, description="The path of the generated mesh in .vtk format, if successful")
    num_nodes: Optional[int] = Field(None, description="The number of nodes in the generated mesh")
    verbose_list: list[str | None] = Field(..., description="A list of paths for verbose output [c2d_log_path (.txt), airfoil_input (.dat), nmf (.nmf) on success or exception (.txt) on failure]")

    
