import os
import logging
import shutil

logger = logging.getLogger(__name__)

def _validate_su2_existance():
    """
    Validates that the necessary SU2 environment variables exist. Called when a run is actually
    requested, not at import. `schemas.py` is pure pydantic and consumers that only need the
    marker-name contract must not require an SU2 install.
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


def __getattr__(name):
    if name == "SU2_Runner":
        from .run import SU2_Runner

        return SU2_Runner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SU2_Runner", # Handles the execution of SU2, and the management of its input/output files
]
