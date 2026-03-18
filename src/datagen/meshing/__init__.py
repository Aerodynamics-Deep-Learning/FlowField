import logging

logger = logging.getLogger(__name__)

def _validate_gmsh_existance():
    """
    Validates if the Gmsh Python SDK is available
    """
    try:
        import gmsh
    except ImportError:
        logger.error("Gmsh Python API not found, will not be able to generate meshes.")
        raise EnvironmentError("Cannot initialize meshing modules, please 'pip install gmsh' in your venv. Terminating.")
    
_validate_gmsh_existance()

from .gmsh_generator import MeshGenerator
from .mesh_io import export_to_su2

__all__ = [
    "MeshGenerator",
    "export_to_su2"
]