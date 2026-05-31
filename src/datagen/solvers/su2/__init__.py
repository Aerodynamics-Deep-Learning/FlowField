import os
import logging
import shutil

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
    if shutil.which("SU2_CFD") is None:
        logger.error("'SU2_CFD' binary is not found in the system PATH.")
        raise EnvironmentError(
            "'SU2_CFD' is missing from PATH. The subprocess runner will crash. "
            "Ensure $SU2_RUN is appended to your system $PATH. Terminating."
        )

_validate_su2_existance()

from .run import SU2_Runner

__all__ = [
    "SU2_Runner", # Handles the execution of SU2, and the management of its input/output files
]
