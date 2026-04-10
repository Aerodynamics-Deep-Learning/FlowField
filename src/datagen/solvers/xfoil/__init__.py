import os 
import shutil
import logging

logger = logging.getLogger(__name__)

def _validate_xfoil_existance():
    """
    Validates if XFoil exists in the system Path variable
    """
    if shutil.which("xfoil") is None:
        logger.error("XFoil binary not found in system Path variable, warmstart will not work")
        raise EnvironmentError("Cannot initialize the XFoil binding, ensure 'XFoil' is installed and in system Path. Terminating.")
    
_validate_xfoil_existance()

from .run import XFoilRunner

__all__ = [
    "XFoilRunner",
]