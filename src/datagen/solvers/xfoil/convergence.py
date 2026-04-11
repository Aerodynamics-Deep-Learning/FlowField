import os
import re

import numpy as np
import torch

from .schemas import XFoilConvergenceFlag

def XFoil_Check_ConvergenceCp(stdout: str, cp_file_path: str, tolerance: float, tail_length: int, var_thresh: float, slope_thresh: float) -> int:

    convergence_flag = _Check_Absolute(stdout, cp_file_path)

    # Check Absolute returns CONVERGED by default if XFoil did not fail catastrophically, recheck here to redefine specifically
    # divergence or convergence
    if convergence_flag == XFoilConvergenceFlag.CONVERGED.value:
        convergence_flag = _Check_Residuals(stdout, tolerance, tail_length, var_thresh, slope_thresh)
        return convergence_flag
    else:
        return convergence_flag # If it failed catastrophically, just return that

def _Check_Residuals(stdout: str, tolerance: float, tail_length: int, var_thresh: float, slope_thresh: float) -> int:
    """
    Checks the state of convergence by checking the behavior of RMS viscous residual history. Checks of limit-cycle oscillations
    in the Newton solver.

    Args:
        stdout (str): The stdout output of the runned subprocess
        tolerance (float): Tolerance used in XFoil
        tail_length (int): The window to be used when checking resids
        var_thresh (float): The threshold for the variance at tail
        slope_thresh (float): The threshold for the slope at tail
    """

    rms_pattern = re.compile(r"rms:\s+([0-9]*\.[0-9]+E[-+][0-9]+)") # Regex bs

    rms_history = []
    for line in stdout.splitlines():
        match = rms_pattern.search(line)
        if match:
            rms_history.append(float(match.group(1)))
    rms_history = np.array(rms_history)

    if len(rms_history) == 0:
        return XFoilConvergenceFlag.FAILED.value

    tailvar_bool = check_tail_variance(rms_history, tolerance, tail_length, var_thresh)
    loggrad_bool = check_log_grad(rms_history, tolerance, tail_length, slope_thresh)

    if not tailvar_bool:
        return XFoilConvergenceFlag.OSCILLATORY.value
    
    elif not loggrad_bool:
        return XFoilConvergenceFlag.STAGNATED.value

    elif rms_history[-1] > tolerance:
        return XFoilConvergenceFlag.ITER_LIMITED.value
    
    else:
        return XFoilConvergenceFlag.CONVERGED.value

def check_tail_variance(rms_array: np.ndarray, tolerance: float, tail_length: int, var_thresh: float) -> bool:
    """
    Checks for the variance in tail_length, if it is more than var_thresh, the sim is defined as diverged.
    """

    # Fast convergence case
    if rms_array[-1] < tolerance and len(rms_array) < tail_length:
        return True
    
    # Checks if a result did not meet tolerance requirement and ended up with a limit cycle
    tail = rms_array[-tail_length:]
    if tail[-1] > tolerance and np.var(tail) > var_thresh:
        return False

    return True

def check_log_grad(rms_array: np.ndarray, tolerance: float, tail_length: int, slope_thresh: float) -> bool:

    if rms_array[-1] < tolerance and len(rms_array) < tail_length:
        return True

    tail = rms_array[-tail_length:]
    if tail[-1] > tolerance:

        log_tail = np.log10(tail)
        x_steps = np.arange(len(log_tail))

        slope, _ = np.polyfit(x_steps, log_tail, 1)

        if slope > slope_thresh:
            return False

    return True 

def _Check_Absolute(stdout: str, cp_file_path: str) -> int:
    """
    Checks the state of convergence for an XFoil run, given its outputs. It is more of an absolute check, and checks for simple
    stuff such as output files having NaN, Inf, or 0 size output files.

    Args:
        stdout (str): The stdout output of the runned subprocess
        cp_file_path (str): The file path of the saved Cp file
        
    Returns:
        int: An integer, meaning if the solver failed catastrophically, diverged, reached max iter, or at least converged (or may have oscillated) to something
    """

    failure_keywords = [
        "n2 convergence failed",
        "VISCAL:  Convergence failed"
    ]

    # A hard check on the stdout to find specific divergences
    if any(keyword in stdout for keyword in failure_keywords):
        return XFoilConvergenceFlag.DIVERGED.value
    
    # A hard check on the file's existence and size
    if not os.path.exists(cp_file_path) or os.path.getsize(cp_file_path) == 0:
        return XFoilConvergenceFlag.FAILED.value
    
    # Check on numerical integrity
    try:
        data = np.loadtxt(cp_file_path, skiprows=2)

        # Ensuring it has data entries
        if data.size == 0:
            return XFoilConvergenceFlag.FAILED.value
        
        # Ensuring no NaNs or Infs
        if not np.all(np.isfinite(data)):
            return XFoilConvergenceFlag.DIVERGED.value

    # Catch all the parsing errors, in case the file is deformed
    except (ValueError, IndexError, OSError):
        return XFoilConvergenceFlag.FAILED.value
        
    return XFoilConvergenceFlag.CONVERGED.value # Temporarily classify it as converged, it will be further checked by the residuals to define if stagnated or oscillatory

