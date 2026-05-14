import logging

logger = logging.getLogger(__name__)

def _validate_gmsh_existance():
    """
    Validates if the Gmsh Python SDK is available
    """
    try:
        import gmsh
    except ImportError:
        logger.error("GMSH Python API not found, will not be able to generate meshes.")
        raise EnvironmentError("Cannot initialize meshing modules, please 'pip install gmsh' in your venv. Terminating.")
    
_validate_gmsh_existance()

from .run import GMSH_MeshGenerator

from .schemas import (
    GMSH_In, 
    GMSH_Out, 
    GMSH_ExitFlag, 
    GMSH_MeshingConfig
)

__all__ = [
    "GMSH_MeshGenerator",
    "GMSH_In",
    "GMSH_Out",
    "GMSH_ExitFlag",
    "GMSH_MeshingConfig"
]