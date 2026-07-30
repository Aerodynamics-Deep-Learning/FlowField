from pathlib import Path

from src.datagen.meshing.gmsh.run import GMSH_MeshGenerator
from src.datagen.meshing.gmsh.schemas import GMSH_ExitFlag
from src.datagen.meshing.gmsh.utils import GMSH_plot_graph, GMSH_plot_mesh

def test_gmsh_integration(sterile_gmsh_input):

    out = GMSH_MeshGenerator(sterile_gmsh_input)

    error_msg = "No exception file generated."

    if out.flag == GMSH_ExitFlag.FATAL_ERROR and out.verbose_list[2] is not None:
        with open(out.verbose_list[2], "r") as f:
            error_msg = f.read()

    assert out.flag == GMSH_ExitFlag.SUCCESS, f"GMSH failed with flag {out.flag}.\nUnderlying Exception:\n{error_msg}"
    assert out.mesh_path is not None
    assert out.min_mesh_quality > 0.0, f"Mesh generated but quality is strictly negative/poor: {out.min_mesh_quality}"

    mesh_file_su2 = Path(out.mesh_path)
    mesh_file_vtk = Path(out.mesh_path.replace(".su2", ".vtk"))
    hist_path = Path(out.hist_path)

    assert mesh_file_su2.exists(), f"SU2 mesh file was not written to disk at {mesh_file_su2}"
    assert mesh_file_su2.stat().st_size > 0, "SU2 mesh file is empty"
    assert mesh_file_vtk.exists(), f"VTK mesh file was not written to disk at {mesh_file_vtk}"
    assert mesh_file_vtk.stat().st_size > 0, "VTK mesh file is empty"

    GMSH_plot_graph(hist_path)
    GMSH_plot_mesh(mesh_file_vtk)


