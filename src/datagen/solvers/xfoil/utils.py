import os
import numpy as np

import torch

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

def XFoil_Get_Freestream_Conditions(mach: float, altitude_m: float):
    """
    Gets the freestream conditions (P_inf, T_inf, rho_inf), given specific operational condition assumptions using
    International Standard Atmosphere (ISA) model (altitude_m < 10k) and the mach number.

    Args:
        mach (float): The mach number used in runs
        altitufe_m (float): The assumed altitude of the run/setup
    """

    # ISA constants
    R = 287.05
    GAMMA = 1.4
    T_SL = 288.15
    P_SL = 101325.0
    L = 0.0065
    G = 9.80665

    # Freestream temperature
    if altitude_m < 11000.0:
        T_inf = T_SL - L *altitude_m
    else: 
        T_inf = 216.5 # Assuming some constant temperature in lower stratosphere

    # Freestream pressure
    P_inf = P_SL * (T_inf / T_SL)**(G / (R * L))
    
    # Freestream density
    rho_inf = P_inf / (R * T_inf)

    # Freestream velocity magnitude
    a_inf = np.sqrt(GAMMA * R * T_inf)
    V_inf = mach * a_inf

    return P_inf, T_inf, rho_inf, V_inf

def XFoil_To_Conservative_State(cp_tensor: torch.Tensor,
                                coords_tensor: torch.Tensor,
                                M_inf: float,
                                P_inf: float,
                                T_inf: float,
                                alpha):

    """
    Conducts the relevant conversion from XFoil variables (dimensionless Cp) into isentropic flow assuming local conservative variables.
    These variables are used for the warm-start
    """

