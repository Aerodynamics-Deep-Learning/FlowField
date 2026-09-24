import pytest
from pathlib import Path

from src.datagen.meshing.c2d.run import C2D_MeshGenerator, C2D_find_exe
from src.datagen.meshing.c2d.schemas import C2D_ExitFlag
from src.datagen.meshing.common.quality import Common_evaluate_mesh_quality
from src.datagen.meshing.common.schemas import MeshExitFlag

pytestmark = pytest.mark.skipif(
    C2D_find_exe() is None, reason="c2d executable not built in this environment"
)


def _generate_and_score(c2d_in) -> Path:
    """
    Meshes `c2d_in`, checks both files were written, and holds the `.su2` to the shared verdict at
    strict SUCCESS, which includes every boundary edge sitting in a marker. Both c2d grids clear
    every gate with room, so anything less is a regression. Returns the `.vtk` path.
    """
    out = C2D_MeshGenerator(c2d_in)

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

    flag, quality, _ = Common_evaluate_mesh_quality(mesh_file_su2)
    assert flag == MeshExitFlag.SUCCESS, (
        f"Mesh generated but the shared quality pass rejected it: {flag.name}, {quality}"
    )
    return mesh_file_vtk


def test_c2d_integration(sterile_c2d_input, mesh_quality_baseline):

    mesh_file_vtk = _generate_and_score(sterile_c2d_input)

    # Compare the generated mesh's quality against Emre's baseline
    # (example_naca0012.vtk), the same way the gmsh integration test does.
    mesh_quality_baseline.assert_not_worse_than_baseline(mesh_file_vtk)


def test_c2d_cgrd_integration(sterile_c2d_cgrd_input):
    """C-grid counterpart on the sharp-TE fixture, the only test that meshes this pairing."""

    _generate_and_score(sterile_c2d_cgrd_input)
