import logging

logger = logging.getLogger(__name__)

from .schemas import (
    GMSH_In,
    GMSH_Out,
    GMSH_ExitFlag,
    GMSH_Topology,
    GMSH_CMeshingConfig,
    GMSH_OMeshingConfig,
)

# `run` is imported lazily: it is the only module here that needs the gmsh SDK, so importing the
# schemas must not drag the SDK in. `common/` imports these schemas, so an eager import would make
# all of common, and the whole unit test tier, unusable without gmsh installed.
def __getattr__(name):
    if name == "GMSH_MeshGenerator":
        from .run import GMSH_MeshGenerator

        return GMSH_MeshGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GMSH_MeshGenerator",
    "GMSH_In",
    "GMSH_Out",
    "GMSH_ExitFlag",
    "GMSH_Topology",
    "GMSH_CMeshingConfig",
    "GMSH_OMeshingConfig",
]
