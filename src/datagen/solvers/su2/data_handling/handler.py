from .maps import COMPRESS_WIPE_MAP
from ..schemas import SU2_ConvergenceFlag, SU2_SolutionStrategy
from .compress import SU2_CompressData
from .clean import SU2_WipeData

from typing import Union
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

def SU2_ComprWipeData(convergence: SU2_ConvergenceFlag, strategy: SU2_SolutionStrategy, working_dir: Union[str, Path], clean_paraview: bool, filename_dict: dict = None) -> None:
    """
    Data lifecycle router, to determine whether to compress the produced data or delete whatever is available

    Args:
        convergence (SU2_ConvergenceFlag): The exact convergence flag used to either decide on compression or wiping of data
        strategy (SU2_SolutionStrategy): The exact strategy used to determine which files to try and wipe out
        working_dir (str | Path): The working directory of interest
        clean_paraview (bool): Cleans the already decided upon files, goes for a leaner storage if chosen
        filename_dict (dict): If being cringe and giving a custom dict, this is where it is given
    
    Returns:
        None, will either heavily compress files or delete whatever it can
    """ 
    decision = COMPRESS_WIPE_MAP.get(convergence)

    if decision is None:
        logger.warning(f"Unknown/not mapped convergence flag to a decision: {convergence}. Doing nothing.")
        return None

    if decision == "compress":
        SU2_CompressData(strategy=strategy, working_dir=working_dir, clean_paraview=clean_paraview, filename_dict=filename_dict)
    elif decision == "wipe":
        SU2_WipeData(strategy=strategy, working_dir=working_dir, filename_dict=filename_dict)
    else:
        logger.warning(f"Unknown decision not mapped to either compress or wiping data: {decision}, not doing anything")

