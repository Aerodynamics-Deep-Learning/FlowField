"""
Code to handle the partial wipeout of the generated data, post proof that the solver failed in some capacity.
"""

from ..schemas import SU2_SolutionStrategy
from .maps import COLD_MAP, WARM_MAP, MACHSEQ_MAP

from pathlib import Path
from typing import Union

def SU2_WipeData(strategy: SU2_SolutionStrategy, working_dir: Union[str, Path], filename_dict: dict = None) -> None:
    """
    Completely wipes all the possibly existing files that may be produced after a crashed/diverged solver

    Args:
        strategy (SU2_SolutionStrategy): The exact strategy used to determine which files to try and wipe out
        working_dir (str | Path): The working directory of interest
        filename_dict (dict): If being cringe and giving a custom dict, this is where it is given
    
    Returns:
        None, will clean all the files it can
    """ 
    working_dir = Path(working_dir)

    if strategy == SU2_SolutionStrategy.COLD:

        if filename_dict is None:
            filename_dict = COLD_MAP
        else:
            if not set(COLD_MAP.keys()).issubset(filename_dict.keys()):
                raise ValueError(f"filename_dict is missing required keys for COLD start. Required: {list(COLD_MAP.keys())}")
            
        for filename in filename_dict.values():
            target_path = working_dir / filename
            target_path.unlink(missing_ok=True)

    elif strategy == SU2_SolutionStrategy.WARM_EULER:
        pass # not implemented yet

    elif strategy == SU2_SolutionStrategy.MACH_SEQ:
        pass # not implemented yet

    else: 
        raise NotImplementedError(f"Data wipeout not implemented yet for strategy {strategy}")


