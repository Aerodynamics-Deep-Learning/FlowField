import matplotlib.pyplot as plt

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

def GMSH_export_sicn_histogram(qualities: list[float], save_path: str) -> None:
    """
    Generates and saves a histogram of mesh element qualities.
    """
    plt.figure(figsize=(8, 6))
    
    # Bins are set to capture the standard SICN domain [-1.0, 1.0] or [0.0, 1.0]
    plt.hist(qualities, bins=250, color='steelblue', edgecolor='black', alpha=0.8)
    
    plt.title("Mesh Element Quality Distribution (minSICN)")
    plt.xlabel("Signed Inverse Condition Number")
    plt.ylabel("Element Count")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()