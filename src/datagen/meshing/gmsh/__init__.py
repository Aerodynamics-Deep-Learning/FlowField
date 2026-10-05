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

# `run` is imported lazily, mirroring `c2d/__init__.py`: reaching for a schema must not drag the
# runner (geo writer, subprocess, numpy) in behind it, and `common.schemas` imports these schemas.
# gmsh itself is never imported, it runs as the executable set in tools.toml.
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
