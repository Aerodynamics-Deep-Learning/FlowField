from .schemas import SU2_ConvergenceFlag

import pandas as pd
import numpy as np

from typing import Union
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

CONVERGENCE_STR_LIST = [
    "All convergence criteria satisfied"
]

ITERLIMITED_STR_LIST = [
    "Maximum number of iterations reached"
]

DIVERGENCE_STR_LIST = [
    "SU2 has diverged", 
    "Matrix inversion failed",
    "Non-physical state", 
    "NaN"
]

def SU2_CheckConvergence(convergence_flag_temp: SU2_ConvergenceFlag, stdout: str, stderr: str, working_dir: Union[str, Path]) -> SU2_ConvergenceFlag:
    
    working_dir = Path(working_dir) # Get the working directory to get the history and the output path

    # Save these in advance, there we'll save them either way
    stdout_path = working_dir / "stdout.txt"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path = working_dir / "stderr.txt"
    stderr_path.write_text(stderr, encoding="utf-8")

    # Combine logs for application-level checks to catch C++/MPI errors
    full_log = stdout + "\n" + stderr

    # Convergence checking branch
    convergence_flag = convergence_flag_temp # Create the temporary convergence flag
    if convergence_flag_temp == SU2_ConvergenceFlag.TIMEOUT:
        logger.warning(f"SU2 timed out. Partial logs saved to {working_dir}")
        convergence_flag = SU2_ConvergenceFlag.TIMEOUT
    
    elif convergence_flag_temp == SU2_ConvergenceFlag.FATAL:
        logger.warning(f"SU2 fatally failed. Partial logs saved to {working_dir}")
        convergence_flag = SU2_ConvergenceFlag.FATAL
    
    else: 
        history_path = working_dir / "history.csv" # Load the histories, important to check for stag or osc or div behavior
        history = _get_su2_history(history_path)

        if _check_if_convergent(stdout): # Convergence check is th emost definitive
            convergence_flag = SU2_ConvergenceFlag.CONVERGED

        elif _check_if_divergent(full_log) or _check_if_nonphysical(history): # Followed by divergence as 2nd most definitive
            convergence_flag = SU2_ConvergenceFlag.DIVERGED

        elif _check_if_iterlimited(stdout): # Iter limited case branch
            
            if _check_if_stagnating(history): # Stagnating run, may benefit from changing params
                convergence_flag = SU2_ConvergenceFlag.STAGNATED

            elif _check_if_oscillatory(history): # Limit cycle oscillation, oscillatory physics
                convergence_flag = SU2_ConvergenceFlag.OSCILLATORY

            else: # If neither oscillatory nor stagnating, then just iter limited
                convergence_flag = SU2_ConvergenceFlag.ITER_LIMITED
        
        else: # If nothing, then we have absolutely no idea what it is
            convergence_flag = SU2_ConvergenceFlag.UNKNOWN
    
    return convergence_flag

def _get_su2_history(history_path: Path) -> pd.DataFrame:
    """
    Gets and aggressively cleans SU2 history formatting, stripping whitespace/quotes 
    from headers and coercing padded string entries into float64 arrays.
    """
    if not history_path.exists():
        logger.warning(f"No history.csv found at {history_path}")
        return pd.DataFrame() # Return empty to gracefully fail downstream checks
    # Get the history
    history = pd.read_csv(history_path, skipinitialspace=True)
    # Clean columns
    history.columns = history.columns.str.strip().str.replace('"', '', regex=False)
    # Numeric type forcing
    history = history.apply(pd.to_numeric, errors='coerce')
    # Drop corrupted lines
    history = history.dropna(how='all')
    return history

def _check_if_convergent(stdout: str) -> bool:
    return any(s in stdout for s in CONVERGENCE_STR_LIST)

def _check_if_divergent(full_log: str) -> bool:
    return any(s in full_log for s in DIVERGENCE_STR_LIST)

def _check_if_nonphysical(history: pd.DataFrame, window: int = 10):
    if history.empty or 'Nonphysical_Points' not in history.columns:
        return False
    if len(history) < window: window = len(history) # Almost impossible, but good to have
    recent_nonphysical = history['Nonphysical_Points'].tail(window)
    return (recent_nonphysical > 0).any() # Return true if there are any non-physical points at the tail of history

def _check_if_iterlimited(stdout: str) -> bool:
    return any(s in stdout for s in ITERLIMITED_STR_LIST)

def _check_if_stagnating(history: pd.DataFrame, window: int = 30, slope_thresh: float = 1e-6, stdev_thresh: float = 1e-3):
    if len(history) < window: window = len(history) # Good to have

    # Done to dynamically identify the primary continuity residual
    if 'rms[Rho]' in history.columns: res_col = 'rms[Rho]'
    elif 'rms[P]' in history.columns: res_col = 'rms[P]' 
    else:
        # Failsafe: Find the first column that is structurally a residual
        res_candidates = [col for col in history.columns if 'rms[' in col or 'res[' in col]
        if not res_candidates:
            logger.error("Could not locate any residual columns in history.csv for stagnation check.")
            return False
        res_col = res_candidates[0]

    recent_res = history[res_col].tail(window).to_numpy()

    # Get the grad of the residual over the window
    x = np.arange(len(recent_res))
    slope, _ = np.polyfit(x, recent_res, 1)
    # Get the stdev to ensure it isn't wildly oscillating
    std_dev = np.std(recent_res)
    # Evaluate stagnation bounds
    if abs(slope) < slope_thresh and std_dev < stdev_thresh: return True
    return False

def _check_if_oscillatory(history: pd.DataFrame, window: int = 30):
    """
    Detects limit-cycle oscillations by counting the zero-crossings of the 
    first derivative (peaks and valleys) of the target aerodynamic functional.
    """
    if history.empty or 'CD' not in history.columns: return False # Safeguard
    if len(history) < window: window = len(history) # Good to have

    recent_drag = history['CD'].tail(window).to_numpy()
    delta_cd = np.diff(recent_drag)

    zero_crossings = np.where(np.diff(np.sign(delta_cd)))[0]
    # If there are multiple peak/valleys within the window, then it must a limit-cycle oscillation
    # 4 zero crossings mean at least 2 full periods of oscillation
    if len(zero_crossings) >= 4: return True
    return False
