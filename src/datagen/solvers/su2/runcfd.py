from .schemas import SU2_SolutionStrategy, SU2_ConvergenceFlag

import subprocess

import logging
logger = logging.getLogger(__name__)

def SU2_RunCFD(config_path_list: list[str], timeout_sec: int, strategy: SU2_SolutionStrategy) -> tuple[SU2_ConvergenceFlag, str, str]:
    """
    Handler to handle different strategy scenarios of running the SU2 solver.

    Args:

    """

    if strategy == SU2_SolutionStrategy.COLD:
        convergence_flag_temp, stdout, stderr = SU2_RunCFD_Cold(config_path_list=config_path_list, timeout_sec=timeout_sec)
    
    elif strategy == SU2_SolutionStrategy.WARM_EULER:
        convergence_flag_temp, stdout, stderr = SU2_RunCFD_Warm(config_path_list=config_path_list, timeout_sec=timeout_sec)

    elif strategy == SU2_SolutionStrategy.MACH_SEQ:
        convergence_flag_temp, stdout, stderr = SU2_RunCFD_MachSeq(config_path_list=config_path_list, timeout_sec=timeout_sec)
    
    else:
        logger.warning(f"Identified but not implemented route: {strategy}, defaulting to cold start")
        convergence_flag_temp, stdout, stderr = SU2_RunCFD_Cold(config_path_list=config_path_list, timeout_sec=timeout_sec)
        
    return convergence_flag_temp, stdout, stderr

def SU2_RunCFD_Cold(config_path_list: list[str], timeout_sec: int) -> tuple[SU2_ConvergenceFlag, str, str]:
    config_path = config_path_list[0] # It is the only confid path in the list
    try:
        result = subprocess.run(
            ["SU2_CFD", config_path],
            capture_output=True, 
            text=True, 
            timeout=timeout_sec
        )
        # Catch segfaults and OS-level crashes
        if result.returncode != 0:
            logger.error(f"SU2 exited with non-zero code: {result.returncode}")
            return SU2_ConvergenceFlag.FATAL, result.stdout, result.stderr
        return SU2_ConvergenceFlag.TEMP, result.stdout, result.stderr

    # Timeout handling
    except subprocess.TimeoutExpired as e:
        return SU2_ConvergenceFlag.TIMEOUT, (e.stdout if e.stdout else "Timed out"), (e.stderr if e.stderr else "Timed out")
    
    # Filenotfound handling
    except FileNotFoundError:
        logger.error("FATAL: SU2_CFD binary not found in system PATH.")
        # We can immediately flag this as FATAL because the solver never even launched.
        return SU2_ConvergenceFlag.FATAL, "No stderr, filenotfound error", "FileNotFoundError: Binary missing."
        
    # Unhandled handling
    except Exception as e:
        logger.error(f"FATAL: Unhandled OS error executing SU2: {e}")
        return SU2_ConvergenceFlag.FATAL, "No stderr, unhandles os error", str(e)

def SU2_RunCFD_Warm(config_path_list: list[str], timeout_sec: int) -> tuple[SU2_ConvergenceFlag, str, str]:
    raise NotImplementedError("WARM_EULER strategy execution is not yet implemented.")

def SU2_RunCFD_MachSeq(config_path_list: list[str], timeout_sec: int) -> tuple[SU2_ConvergenceFlag, str, str]:
    raise NotImplementedError("MACHSEQ strategy execution is not yet implemented.")
