import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
from typing import Tuple

from src.datagen.schemas import Freestream

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

def GMSH_validate_tensor_numeric(coords_tensor: torch.Tensor) -> Tuple[bool, str]:
    """
    Validates the numeric validity of the airfoil coordinate tensor
    
    Args:
        coords_tensor (torch.Tensor): The tensor of the airfoil coords, in selig format
    
    Returns:
        tuple: (is_valid_tensor, warning_message)
    """
    if not isinstance(coords_tensor, torch.Tensor):
        return False, f"Airfoil input is not a torch.Tensor, it is: {type(coords_tensor)}"
    elif coords_tensor.ndim != 2 or coords_tensor.shape[1] not in (2, 3):
        return False, f"Airfoil input tensor has unexpected number of dims: {coords_tensor.ndim}, {coords_tensor.shape[1]}"
    elif torch.isnan(coords_tensor).any().item():
        return False, f"Airfoil input tensor has NaN values"
    elif torch.isinf(coords_tensor).any().item():
        return False, f"Airfoil input tensor has Inf values"
    else:
        return True, ""

def GMSH_validate_te_bluntness(coords_tensor: torch.Tensor, micro_tol: float = 1e-5) -> Tuple[bool, str]:
    """
    Validates the trailing edge gap of an airfoil coordinate tensor
    
    Args: 
        coords_tensor (torch.Tensor): The tensor of the airfoil coords, in selig format
        micro_tol (float): The micro tolerance for bluntness

    Returns:
        tuple: (is_valid_bluntness, warning_message)
    """
    te_gap = torch.linalg.norm(coords_tensor[0] - coords_tensor[-1]).item()

    if te_gap == 0.0:
        return False, f"Airfoil not blunted. Coincident points at {coords_tensor[0]}."
    elif te_gap < micro_tol:
        return False, f"Trailing edge gap too small: {te_gap}."
    else:
        return True, ""

def GMSH_validate_freestream_physicality(freestream: Freestream) -> Tuple[bool, str]:
    """
    Validates the physicality of the input freestream, to prevent non-sensical first mesh cell height calcs

    Args:
        freestream (Freestream): The freestream class

    Returns:
        tuple: (is_valid_bluntness, warning_message)
    """
    if freestream.mach < 0:
        return False, f"Freestream has unexpected mach number: {freestream.mach}"
    elif freestream.Re < 0:
        return False, f"Freestream has unexpected Reynolds number: {freestream.Re}"
    elif freestream.altitude_m < 0:
        return False, f"Freestream has unexpected altitude: {freestream.altitude_m}"
    else:
        return True, ""

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

def GMSH_plot_graph(save_path: str) -> None:
    """
    Just loads and shows a plot
    """
    img = mpimg.imread(save_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show(block=True)

def GMSH_plot_mesh(mesh_path_vtk: str) -> None:
    import pyvista as pv
    import vtk
    import numpy as np

    grid = pv.read(mesh_path_vtk)
    interior_indices = np.where(
            (grid.celltypes == vtk.VTK_TRIANGLE) | 
            (grid.celltypes == vtk.VTK_QUAD)
        )[0]
    fluid_domain = grid.extract_cells(interior_indices)

    metrics = ['scaled_jacobian', 'skew', 'aspect_ratio', 'area']
    
    for metric in metrics:
        original_arrays = set(fluid_domain.cell_data.keys())
        temp_mesh = fluid_domain.cell_quality(quality_measure=metric)
        new_array_name = list(set(temp_mesh.cell_data.keys()) - original_arrays)[0]
        fluid_domain.cell_data[metric] = temp_mesh.cell_data[new_array_name]

    plotter = pv.Plotter()

    modes = [None] + metrics
    state = {"index": 0}

    def render_main_actor():
        current_mode = modes[state["index"]]
        
        if current_mode is None:
            for bar_title in list(plotter.scalar_bars.keys()):
                plotter.remove_scalar_bar(bar_title)
                
            plotter.add_mesh(
                fluid_domain,
                color="purple",
                show_edges=False,
                name="main_fluid_surface",
                reset_camera=False
            )
        else:
            plotter.add_mesh(
                fluid_domain,
                scalars=current_mode,
                cmap="jet",
                show_edges=False,
                name="main_fluid_surface",
                reset_camera=False,
                scalar_bar_args={"title": current_mode.replace("_", " ").title()}
            )

    # Initialize the base faces
    render_main_actor()

    # Create an independent copy for the wireframe and physically translate it minimally to prevent shift artifacts
    wireframe_mesh = fluid_domain.copy()
    wireframe_mesh.translate((0.0, 0.0, 0.000001), inplace=True)
    
    # Use native wireframe styling instead of edge extraction
    edge_actor = plotter.add_mesh(
        wireframe_mesh, 
        style="wireframe", 
        color="black", 
        line_width=0.5,
        name="elevated_edges",
        reset_camera=False
    )
    edge_actor.SetVisibility(False)

    def toggle_edges():
        current_state = edge_actor.GetVisibility()
        edge_actor.SetVisibility(not current_state)
        plotter.render()

    def toggle_quality():
        state["index"] = (state["index"] + 1) % len(modes)
        render_main_actor()
        
        active = modes[state["index"]] if modes[state["index"]] else "Solid Purple Baseline"
        print(f"Active Render Mode: {active}")

    plotter.add_key_event('t', toggle_edges)
    plotter.add_key_event('c', toggle_quality)

    plotter.view_xy()
    plotter.enable_2d_style()
    
    print("\n[PyVista] Viewer launched.")
    print("Left-click to pan. Right-click to zoom.")
    print("Press 't' to toggle mesh edges.")
    print("Press 'c' to cycle through quality metrics.")
    plotter.show()
