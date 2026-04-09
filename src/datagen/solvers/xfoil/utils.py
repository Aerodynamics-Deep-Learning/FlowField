import os
import numpy as np

def XFoil_Check_ConvergenceCp(stdout: str, cp_file_path: str, convergence_limit: float=1.0e2) -> int:
    """
    Checks the state of convergence for an XFoil run, given its outputs.

    Args:
        stdout (str): The stdout output of the runned subprocess
        cp_file_path (str): The file path of the saved Cp file
        convergence_limit (float): The absolute limit of Cp below which a datapoint is assumed to be converged. Defaults at 1e3
    """
    failure_keywords = [
        "Convergence failed",
        "Convergence not reached",
        "n2 convergence failed"
    ]

    # A hard check on the stdout to find specific divergences
    if any(keyword in stdout for keyword in failure_keywords):
        return 0
    
    # A hard check on the file's existence and size
    if not os.path.exists(cp_file_path) or os.path.getsize(cp_file_path) == 0:
        return 0
    
    # Check on numerical integrity
    try:
        data = np.loadtxt(cp_file_path, skiprows=2)

        # Ensuring it has data entries
        if data.size == 0:
            return 0
        
        # Ensuring no NaNs or Infs
        if not np.all(np.isfinite(data)):
            return 0
        
        # Ensuring some sort of numerical convergence, no diverged specific values
        cp_values = data[:, 1]
        if np.max(np.abs(cp_values)) > convergence_limit:
            return 0

    # Catch all the parsing errors, in case the file is deformed
    except (ValueError, IndexError, OSError):
        return 0
        
    return 2