import pyvista as pv
from pathlib import Path

import numpy as np

def test_viz():
    working_dir = Path(__file__).parent
    
    # Load the VTU directly
    mesh_path = working_dir / "test_mesh.vtk"
    print(str(mesh_path))
    grid = pv.read(mesh_path)

    # In a VTU, cell types are natively preserved, extract 2D elements
    import vtk
    interior_indices = np.where(
        (grid.celltypes == vtk.VTK_TRIANGLE) | 
        (grid.celltypes == vtk.VTK_QUAD)
    )[0]
    
    fluid_domain = grid.extract_cells(interior_indices)

    plotter = pv.Plotter()
    plotter.add_mesh(
        fluid_domain,
        show_edges=True,
        edge_color="black",
        line_width=0.2,
    )
    plotter.view_xy()
    plotter.enable_parallel_projection()
    plotter.show()

if __name__ == "__main__":
    test_viz()