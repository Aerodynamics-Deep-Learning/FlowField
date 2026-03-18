import os
import logging

logger = logging.getLogger(__name__)

def _validate_su2_existance():
    """
    Validates if the necessary SU2 environment variables exist, upon import
    """
    if "SU2_RUN" not in os.environ:
        logger.error("'SU2_RUN' environment variable is missing. SU2 execution will fail.")
        raise EnvironmentError("'SU2_RUN' not found, thus cannot initialize SU2 bingings. Terminating.")
    if "SU2_HOME" not in os.environ:
        logger.warning("'SU2_HOME' not found, some Python-based SU2 utilities/APIs might/will fail.")

_validate_su2_existance()

from .runner import SU2Runner
from .config_builder import CFLScheduler
from .parser import parse_history
from .interpolator import map_solution

__all__ = [
    "SU2Runner", # Handles the execution of SU2, and the management of its input/output files
    "CFLScheduler", # Utility to build the CFL schedule for SU2, which is important when changing solver and mesh res
    "parse_history", # Parses the SU2 history file, to extract the force coefficients and convergence information
    "map_solution" # Utility to map the SU2 solution (which is on an unstructured mesh) to a structured grid, for easier ML training
]
