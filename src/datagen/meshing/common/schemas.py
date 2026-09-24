"""
All the high level schemas used for the mesh generation procedure.

Summary:
    - MeshIn: The input schema to the main handler `entry::Common_GenerateMesh`
    - MeshOut: The output schema to the main handle function `entry::Common_GenerateMesh`
    - MeshExitFlag: The exit flag to keep track of meshing-success
    - MeshQualitySummary: Data schema that governs the quality summary no fail meshes
    - MeshGeoDeviationSummary: Geometric deviation summary that quantifies diff between geometry and mesh
    - MeshBackend: Flag on which backend to use
    - MeshTopology: Flag on which meshing topology/grid structure to use
"""

from enum import IntEnum, Enum
from typing import Optional, Union

from pydantic import BaseModel, Field, model_validator

from src.datagen.schemas import Airfoil, Freestream
from src.datagen.meshing.gmsh.schemas import GMSH_CMeshingConfig, GMSH_OMeshingConfig
from src.datagen.meshing.c2d.schemas import C2D_MeshingConfig

class MeshBackend(str, Enum):
    """Which meshing implementation to dispatch to."""
    GMSH = "gmsh"
    C2D = "c2d"
    UNKNOWN = "UNKNOWN"

class MeshTopology(str, Enum):
    """Which meshing topology to dispatch to."""
    CGRD = "CGRID"
    OGRD = "OGRID"

class MeshExitFlag(IntEnum):
    """
    Backend-agnostic categorical flag for what happened during/after mesh generation.

    A superset of `gmsh.schemas.GMSH_ExitFlag` and c2d's plain `ok`/`acceptable` bools,
    ordered most-negative = most severe (mirrors gmsh's existing convention). Through the input and
    backend stages that also reads as earliest-first; the three post-mesh verdicts break that
    reading, since they are all evaluated last and rank by how unusable the mesh is instead.
    """
    INPUT_TENSOR_FAIL = -11      # malformed/NaN/Inf/non-chord-normalized airfoil geometry
    INPUT_BLUNTING_FAIL = -10   # trailing-edge gap check
    INPUT_FREESTREAM_FAIL = -9  # alpha/Re/mach/altitude physicality
    INPUT_UNKNOWN_SOLVER = -8   # unknown solver
    EXECUTABLE_NOT_FOUND = -7   # c2d only: the C2D exe isn't built/found
    SUBPROCESS_FAIL = -6        # c2d only: the exe crashed or timed out
    FATAL_ERROR = -5            # unhandled exception in either backend
    CONVERSION_FAIL = -4        # gmsh's extrusion produced 0 elements / c2d's p3d->su2 conversion failed / a boundary edge is in no marker
    UNACCEPTABLE_QUALITY = -3   # mesh produced but has inverted/zero-area cells (or minSICN <= 0) or quality that is unacceptable
    GEOMETRY_DEVIATION_FAIL = -2 # well-formed mesh, but its MARKER_AIRFOIL boundary does not reproduce the input geometry
    LOW_QUALITY = -1             # valid mesh, but exceeds the skew/growth/aspect-ratio acceptance thresholds
    SUCCESS = 0

# Which config class each supported (backend, topology) pairing is configured by. gmsh's two
# topologies mesh structurally different geometry and so take different configs; c2d uses
# one for both, since its OGRD/CGRD choice is a single field inside `C2D_MeshingConfig` (filled in
# from `topology` by `C2D_In`'s own validator). A pairing absent from this table is unsupported.
_CONFIG_FOR_PAIRING = {
    (MeshBackend.GMSH, MeshTopology.CGRD): GMSH_CMeshingConfig,
    (MeshBackend.GMSH, MeshTopology.OGRD): GMSH_OMeshingConfig,
    (MeshBackend.C2D, MeshTopology.CGRD): C2D_MeshingConfig,
    (MeshBackend.C2D, MeshTopology.OGRD): C2D_MeshingConfig,
}

MeshingConfig = Union[GMSH_CMeshingConfig, GMSH_OMeshingConfig, C2D_MeshingConfig]


class MeshIn(BaseModel):
    """Backend-agnostic contract for the input to mesh generation."""
    # extra="forbid": mesh_config defaults when omitted, so a mistyped keyword would otherwise be
    # dropped silently and mesh with defaults instead of the config the caller meant to pass
    model_config = {"arbitrary_types_allowed": True, "extra": "forbid"}

    airfoil: Airfoil
    freestream: Freestream
    working_dir: str = Field(..., description="Directory where the generated mesh will be saved")
    backend: MeshBackend
    topology: MeshTopology
    mesh_config: Optional[MeshingConfig] = Field(
        None, description="The meshing config for this (backend, topology) pairing. Left unset, "
        "the pairing's own config class is instantiated with its defaults; a plain dict is "
        "coerced to that class. See `_resolve_mesh_config`."
    )

    @model_validator(mode="before")
    @classmethod
    def _resolve_mesh_config(cls, raw):
        """
        Resolves `mesh_config` against the requested (backend, topology) pairing.
        """
        if not isinstance(raw, dict):
            return raw
        try:
            pairing = (MeshBackend(raw["backend"]), MeshTopology(raw["topology"]))
        except (KeyError, ValueError):
            return raw  # Missing/unparseable: let normal field validation report it properly
        expected = _CONFIG_FOR_PAIRING.get(pairing)
        if expected is None:
            return raw  # Unsupported pairing: `Common_GenerateMesh` flags it, not the schema

        config = raw.get("mesh_config")
        if config is None:
            raw["mesh_config"] = expected()
        elif isinstance(config, dict):
            raw["mesh_config"] = expected(**config)
        elif not isinstance(config, expected):
            raise ValueError(
                f"backend={pairing[0].value} with topology={pairing[1].value} requires "
                f"{expected.__name__}, got {type(config).__name__}"
            )
        return raw


class MeshQualitySummary(BaseModel):
    """Backend-agnostic `.su2` quality summary, from `common.quality.analyze_su2`."""
    ncell: int
    max_skew: float # Maximum skewness, self explanatory, not as hard as ortho or jac
    min_ortho: float = Field(..., description="Worst cell's orthogonal quality, measured off face normals against the centroid-to-face and centroid-to-centroid vectors. 1 is perfectly orthogonal, 0 degenerate, negative folded. Reads a cell's neighbours, so it is not derivable from that cell's own corner angles the way skewness is")
    min_jac: float = Field(..., description="Worst cell's scaled Jacobian: 1 is a right-angled corner, 0 a collapsed one, below 0 a corner folded back on itself. The one metric here that detects a tangled cell, which skewness reads as well-shaped")
    max_ar: float # Maximum aspect ratio, self explanatory, not as hard as ortho or jac
    acceptable: bool

class MeshGeoDeviationSummary(BaseModel):
    """
    How far the meshed `MARKER_AIRFOIL` boundary (boundary of the airfoil created by the meshing
    strays from the input geometry, from `common.geo_dev.Common_evaluate_mesh_geo_dev`. Distances 
    are in chord-normalized units.
    """
    n_boundary_nodes: int # Number of created boundary nodes in total that define the airfoil
    max_dev_mesh_to_input: float = Field(..., description="Worst distance from a meshed boundary node to the input polyline: the mesher leaving the curve")
    max_dev_input_to_mesh: float = Field(..., description="Worst distance from an input point to the meshed boundary: the mesher skipping detail")
    rms_dev_mesh_to_input: float # RMS of the dev_mesh_to_input
    rms_dev_input_to_mesh: float # RMS of the dev_input_to_mesh
    chord: float = Field(..., description="Chordwise extent of the meshed boundary in the mesh file's own units, which deviations are normalized by. Reads as which coordinate space the backend chose: gmsh meshes at Airfoil.chord, c2d re-normalizes to 1.0")
    acceptable: bool # Bool flag on acceptability

class MeshOut(BaseModel):
    """Backend-agnostic contract for the output from mesh generation."""
    model_config = {"arbitrary_types_allowed": True, "extra": "forbid"}

    airfoil: Airfoil
    freestream: Freestream
    backend: MeshBackend
    topology: MeshTopology
    flag: MeshExitFlag
    mesh_path: Optional[str] = Field(None, description="Path of the generated .su2 mesh, if produced")
    mesh_path_vtk: Optional[str] = Field(None, description="Path of the generated .vtk mesh, if produced")
    quality: Optional[MeshQualitySummary] = Field(None, description="Shared .su2 quality summary")
    geo_dev: Optional[MeshGeoDeviationSummary] = Field(None, description="Shared meshed-boundary vs input-geometry summary")
    num_nodes: Optional[int] = Field(None, description="Number of nodes in the generated mesh")
    verbose_list: list = Field(default_factory=list, description="Backend-specific verbose/debug paths (logs, exceptions, etc.)")
