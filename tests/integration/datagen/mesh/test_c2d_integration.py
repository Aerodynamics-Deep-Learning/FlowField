import pytest
from pathlib import Path

from src.datagen.meshing.c2d.run import C2D_MeshGenerator, C2D_find_exe
from src.datagen.meshing.c2d.schemas import C2D_ExitFlag

pytestmark = pytest.mark.skipif(
    C2D_find_exe() is None, reason="c2d executable not built in this environment"
)


def test_c2d_integration(sterile_c2d_input, mesh_quality_baseline):

    out = C2D_MeshGenerator(sterile_c2d_input)

    error_msg = "No exception file generated."
    if out.flag == C2D_ExitFlag.FATAL_ERROR and len(out.verbose_list) > 2 and out.verbose_list[2]:
        with open(out.verbose_list[2], "r") as f:
            error_msg = f.read()

    assert out.flag == C2D_ExitFlag.SUCCESS, (
        f"c2d failed with flag {out.flag}.\nUnderlying Exception:\n{error_msg}"
    )
    assert out.mesh_path is not None

    mesh_file_su2 = Path(out.mesh_path)
    mesh_file_vtk = Path(out.mesh_path_vtk)

    assert mesh_file_su2.exists(), f"SU2 mesh file was not written to disk at {mesh_file_su2}"
    assert mesh_file_su2.stat().st_size > 0, "SU2 mesh file is empty"
    assert mesh_file_vtk.exists(), f"VTK mesh file was not written to disk at {mesh_file_vtk}"
    assert mesh_file_vtk.stat().st_size > 0, "VTK mesh file is empty"

    # Compare the generated mesh's quality against Emre's baseline
    # (example_naca0012.vtk), the same way the gmsh integration test does.
    mesh_quality_baseline.assert_not_worse_than_baseline(mesh_file_vtk)
