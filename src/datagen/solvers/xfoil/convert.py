import numpy as np

from typing import Tuple

def XFoil_Get_Freestream_Conditions(mach: float, altitude_m: float) -> Tuple[float, float, float, float]:
    """
    Gets the freestream conditions (P_inf, T_inf, rho_inf), given specific operational condition assumptions using
    International Standard Atmosphere (ISA) model (altitude_m < 11k + minor corrections for lower stratosphere) and the mach number.

    Args:
        mach (float): The mach number used in runs
        altitude_m (float): The assumed altitude of the run/setup

    Returns:
        Tuple[float, float, float, float]: The derived freestream conditions (P_inf, T_inf, rho_inf, V_inf)
    """

    # ISA constants
    R = 287.052874247
    GAMMA = 1.4
    L = 0.0065
    G = 9.80665

    # when alt<11k
    T_SL = 288.15
    P_SL = 101325.0

    # when alt in lower stratosphere
    T_11 = 216.65
    P_11 = 22632.1

    # Freestream temperature and pressure given different altitudes
    if altitude_m < 11000.0:
        T_inf = T_SL - L *altitude_m
        P_inf = P_SL * (T_inf / T_SL)**(G / (R * L))
    else: 
        T_inf = T_11
        # Isotherm exp decay
        P_inf = P_11 * np.exp((-G / (R * T_11)) * (altitude_m - 11000.0))

    # Freestream density
    rho_inf = P_inf / (R * T_inf)

    # Freestream velocity magnitude
    a_inf = np.sqrt(GAMMA * R * T_inf)
    V_inf = mach * a_inf

    return P_inf, T_inf, rho_inf, V_inf

def XFoil_Cp_To_Conservative_State(Cp: np.ndarray, coords: np.ndarray, M_inf: float, altitude_m: float, alpha: float) -> np.ndarray:

    """
    Conducts the relevant conversion from XFoil variables (dimensionless Cp) into isentropic flow assuming local conservative variables.
    These variables are used for the warm-start within SU2. Conversions were done given the isentropic assumption in this regime.

    Args:
        Cp (np.ndarray): The Cp values, of shape (N, [x, Cp])
        coords (np.ndarray): The coordinates of the points defining the airfoil in shape (N, 2)
        M_inf (float): The derived freestream Mach number, same variable as the mach number defined earlier
        altitude_m (float): The assumed altitude of the run/setup
        alpha (float): Angle of attack of the given airfoil, in degrees

    Returns:
        np.ndarray: The conservative state of shape (N, 4), with the columns being [rho, rho*u, rho*v, rho*E]
    """

    GAMMA = 1.4
    R = 287.052874247

    alpha_rad = np.radians(alpha)

    P_inf, T_inf, rho_inf, V_inf = XFoil_Get_Freestream_Conditions(mach=M_inf, altitude_m=altitude_m)

    # Stagnation point (leading edge) temperature
    T_0 = T_inf * (1.0 + ((GAMMA - 1) * M_inf**2) / 2)

    # Infer the dimensionless Cps (N, [x, Cp])
    cp = Cp[:, 1]

    # Local static pressure, clamp for numeric safety on/around LE
    P_local = P_inf * (1.0 + (GAMMA/2) * M_inf**2 * cp)
    P_local = np.maximum(P_local, P_inf*0.001)
    # Local density
    rho_local = rho_inf * (P_local / P_inf)**(1.0 / GAMMA)
    # Local temp, set max to T_0 given entropy constraint
    T_local = T_inf * (P_local / P_inf)**(1.0 - (1.0 / GAMMA))
    T_local = np.minimum(T_local, T_0)

    # Velocity magnitude, via Enthalpy conservation
    V_mag = np.sqrt((2.0 * R * GAMMA / (GAMMA - 1.0)) * (T_0 - T_local))

    # Surface tangent vectors, using central difference, normalized to become unit
    tangents = np.zeros_like(coords, dtype=np.float64)
    tangents[1:-1] = coords[2:] - coords[:-2]
    tangents[0] = coords[1] - coords[0]
    tangents[-1] = coords[-1] - coords[-2]
    t_norm = np.linalg.norm(tangents, axis=1, keepdims=True)
    unit_tangents = tangents / (t_norm + 1e-12)

    # Given that the coords are in Selig format, the upper airfoil's tangent vectors will point forward
    # the logic presented here is used to flip them all such that all tangent vectors representing velocity point backward
    v_inf_dir = np.array([np.cos(alpha_rad), np.sin(alpha_rad)]) # Will be used to determine which normals to flip
    dot_prod = np.sum(unit_tangents * v_inf_dir, axis=1, keepdims=True) # Becomes the dot product
    unit_tangents = np.where(dot_prod < 0, -unit_tangents, unit_tangents) # We flip the tangents that point backward

    # Scale the velocity unit tangents
    u_local = V_mag * unit_tangents[:,0] # Scaling the x axis of velocity unit tangents
    v_local = V_mag * unit_tangents[:,1] # Scaling the y axis of velocity unit tangents

    # Convert everything into conservative variables
    rho_u = rho_local * u_local
    rho_v = rho_local * v_local
    rho_E = (P_local / (GAMMA - 1.0)) + (rho_local / 2) * V_mag**2

    conservative_state = np.stack((rho_local, rho_u, rho_v, rho_E), axis=1)

    return conservative_state



