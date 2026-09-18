def GMSH_scale_by_chord(value: float, chord: float) -> float:
    """
    Scales a chord-normalized length by the airfoil's actual chord.

    The pipeline normalizes chord to 1.0 upstream, so this is a no-op multiply by 1 in the
    normal case -- kept as a safety net in case that invariant is ever violated, so meshing
    config lengths (farfield radius, BL thickness, etc.) still scale correctly.

    Args:
        value (float): A length expressed in chord-normalized units
        chord (float): The airfoil's actual chord length

    Returns:
        float: The length scaled to the airfoil's actual chord
    """
    return value * chord

def GMSH_get_mesh_height(Re: float, chord: float, target_yplus: float=1.0) -> float:
    """
    Gets the target height of the first layer of meshing given Re and chord length, and the target yplus

    Args:
        Re (float): The reynolds number
        chord (float): The chord length of the parameterized airfoil
        target_yplus (float): Our target yplus value, given the relative ease of compute (2D) and Spalart-Allmaras / k-omega SST, it is the main choice
    
    Returns:
        float: The height of the first mesh cell (radial out from the airfoil)
    """
    Cf = 0.058 * (Re ** -0.2)
    y_height = target_yplus * (chord / (Re * ((Cf / 2) ** 0.5)))
    return y_height

