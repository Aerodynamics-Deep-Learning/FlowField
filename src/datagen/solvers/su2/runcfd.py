from .schemas import SU2_SolutionStrategy, SU2_ConvergenceFlag

from .convergence import SU2_CheckConvergence

import subprocess
import os

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
    run_dir = os.path.dirname(config_path)
    try:
        result = subprocess.run(
            ["SU2_CFD", config_path],
            cwd=run_dir,
            capture_output=True, 
            text=True, 
            timeout=timeout_sec
        )
        # Catch segfaults and OS-level crashes
        if result.returncode != 0:
            logger.error(f"SU2 exited with non-zero code for Cold start: {result.returncode}")
            return SU2_ConvergenceFlag.FATAL, result.stdout, result.stderr

    # Timeout handling
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else str(e.stdout or "Timed out")
        err = e.stderr.decode() if isinstance(e.stderr, bytes) else str(e.stderr or "Timed out")
        return SU2_ConvergenceFlag.TIMEOUT, out, err
    
    # Filenotfound handling
    except FileNotFoundError:
        logger.error("FATAL: SU2_CFD binary not found in system PATH.")
        # We can immediately flag this as FATAL because the solver never even launched.
        return SU2_ConvergenceFlag.FATAL, "No stderr, filenotfound error", "FileNotFoundError: Binary missing."
        
    # Unhandled handling
    except Exception as e:
        logger.error(f"FATAL: Unhandled OS error executing SU2: {e}")
        return SU2_ConvergenceFlag.FATAL, "No stderr, unhandled os error", str(e)

def SU2_RunCFD_Warm(config_path_list: list[str], timeout_sec: int) -> tuple[SU2_ConvergenceFlag, str, str]:
    euler_config_path, restart_config_path = config_path_list # The first config path is for the euler phase, the second is for the restart phase
    run_dir = os.path.dirname(euler_config_path)
    # Running the Euler phase first, if it fails we can immediately return with the flag and not run the restart phase, if it succeeds we move on
    # to the restart phase and return its results regardless of its convergence since the euler phase convergence is a prerequisite for the restart 
    # phase to even be valid
    try:
        result = subprocess.run(
            ["SU2_CFD", euler_config_path],
            cwd=run_dir,
            capture_output=True, 
            text=True, 
            timeout=timeout_sec
        )
         # Catch segfaults and OS-level crashes
        if result.returncode != 0:
            logger.error(f"SU2 exited with non-zero code in the Euler phase: {result.returncode}")
            return SU2_ConvergenceFlag.FATAL, result.stdout, result.stderr
        euler_stdout = result.stdout
        euler_stderr = result.stderr
    # Timeout handling
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else str(e.stdout or "Timed out")
        err = e.stderr.decode() if isinstance(e.stderr, bytes) else str(e.stderr or "Timed out")
        return SU2_ConvergenceFlag.TIMEOUT, out, err
    # Filenotfound handling
    except FileNotFoundError:
        logger.error("FATAL: SU2_CFD binary not found in system PATH in the Euler phase.")
        # We can immediately flag this as FATAL because the solver never even launched.
        return SU2_ConvergenceFlag.FATAL, "No stderr, filenotfound error", "FileNotFoundError: Binary missing." 
    # Unhandled handling
    except Exception as e:
        logger.error(f"FATAL: Unhandled OS error executing SU2 in the Euler phase: {e}")
        return SU2_ConvergenceFlag.FATAL, "No stderr, unhandles os error", str(e)
    
    # Need to check convergence internally for the euler phase to decide whether to proceed to the restart or not
    euler_convergence_flag = SU2_CheckConvergence(
        convergence_flag_temp=SU2_ConvergenceFlag.TEMP,
        stdout=euler_stdout,
        stderr=euler_stderr,
        working_dir=run_dir
    )
    if euler_convergence_flag != SU2_ConvergenceFlag.CONVERGED:
        logger.warning(f"Euler phase did not converge, skipping Restart phase. Euler phase flag: {euler_convergence_flag}")
        # Get back to default convergence flag, for the upstream checks to happen normally
        return euler_convergence_flag, euler_stdout, euler_stderr
    
    # Identical to cold start
    try:
        result = subprocess.run(
            ["SU2_CFD", restart_config_path],
            cwd=run_dir,
            capture_output=True,
            text=True, 
            timeout=timeout_sec
        )
         # Catch segfaults and OS-level crashes
        if result.returncode != 0:
            logger.error(f"SU2 exited with non-zero code in the Restart phase: {result.returncode}")
            return SU2_ConvergenceFlag.FATAL, result.stdout, result.stderr
        return SU2_ConvergenceFlag.TEMP, result.stdout, result.stderr
    # Timeout handling
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else str(e.stdout or "Timed out")
        err = e.stderr.decode() if isinstance(e.stderr, bytes) else str(e.stderr or "Timed out")
        return SU2_ConvergenceFlag.TIMEOUT, out, err
    # Filenotfound handling
    except FileNotFoundError:
        logger.error("FATAL: SU2_CFD binary not found in system PATH in the Restart phase.")
        # We can immediately flag this as FATAL because the solver never even launched.
        return SU2_ConvergenceFlag.FATAL, "No stderr, filenotfound error", "FileNotFoundError: Binary missing." 
    # Unhandled handling
    except Exception as e:
        logger.error(f"FATAL: Unhandled OS error executing SU2 in the Restart phase: {e}")
        return SU2_ConvergenceFlag.FATAL, "No stderr, unhandled os error", str(e)


def SU2_RunCFD_MachSeq(config_path_list: list[str], timeout_sec: int) -> tuple[SU2_ConvergenceFlag, str, str]:
    raise NotImplementedError("MACHSEQ strategy execution is not yet implemented.")
