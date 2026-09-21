from pathlib import Path

import pytest

pytest.importorskip("gmsh", reason="gmsh Python SDK not installed")

from src.datagen.meshing.gmsh.run import GMSH_MeshGenerator
from src.datagen.meshing.gmsh.schemas import GMSH_ExitFlag
from src.datagen.meshing.common.quality import Common_evaluate_mesh_quality
from src.datagen.meshing.common.schemas import MeshExitFlag


def assert_no_inverted_cells(mesh_path):
    """
    Replaces the old `out.min_mesh_quality > 0.0` check, which read gmsh's native minSICN off
    `GMSH_Out`. That field is gone: quality is scored centrally from the written `.su2`, so the
    equivalent here is the shared verdict reporting no inverted or zero-area cells. LOW_QUALITY is
    deliberately tolerated; minSICN > 0 never implied the thresholds were met, and measured
    2026-09-16 the C-mesh scores SUCCESS (max_skew 0.446) while the O-mesh scores LOW_QUALITY
    (max_skew 0.909), so requiring SUCCESS would fail the O-mesh on an unrelated axis. The graded
    comparison is `assert_not_worse_than_baseline`.
    """
    flag, quality, _ = Common_evaluate_mesh_quality(mesh_path)
    assert flag not in (MeshExitFlag.CONVERSION_FAIL, MeshExitFlag.UNACCEPTABLE_QUALITY), (
        f"Mesh generated but the shared quality pass rejected it: {flag.name}, {quality}"
    )


def test_gmsh_integration(sterile_gmsh_input, mesh_quality_baseline):

    out = GMSH_MeshGenerator(sterile_gmsh_input)

    error_msg = "No exception file generated."

    if out.flag == GMSH_ExitFlag.FATAL_ERROR and out.verbose_list[2] is not None:
        with open(out.verbose_list[2], "r") as f:
            error_msg = f.read()

    assert out.flag == GMSH_ExitFlag.SUCCESS, f"GMSH failed with flag {out.flag}.\nUnderlying Exception:\n{error_msg}"
    assert out.mesh_path is not None

    mesh_file_su2 = Path(out.mesh_path)
    mesh_file_vtk = Path(out.mesh_path_vtk)

    assert mesh_file_su2.exists(), f"SU2 mesh file was not written to disk at {mesh_file_su2}"
    assert mesh_file_su2.stat().st_size > 0, "SU2 mesh file is empty"
    assert mesh_file_vtk.exists(), f"VTK mesh file was not written to disk at {mesh_file_vtk}"
    assert mesh_file_vtk.stat().st_size > 0, "VTK mesh file is empty"

    assert_no_inverted_cells(mesh_file_su2)

    # Compare the generated mesh's quality against Emre's baseline
    # (example_naca0012.vtk), rather than just eyeballing a side-by-side plot.
    mesh_quality_baseline.assert_not_worse_than_baseline(mesh_file_vtk)


def test_gmsh_omesh_integration(sterile_gmsh_omesh_input):
    """O-mesh counterpart to `test_gmsh_integration`; no baseline comparison, since Emre's
    reference mesh is a C-mesh, not an O-mesh; this just checks the O-mesh path (real, sharp-TE
    airfoil geometry, no mocking) structurally succeeds end to end."""

    out = GMSH_MeshGenerator(sterile_gmsh_omesh_input)

    error_msg = "No exception file generated."

    if out.flag == GMSH_ExitFlag.FATAL_ERROR and out.verbose_list[2] is not None:
        with open(out.verbose_list[2], "r") as f:
            error_msg = f.read()

    assert out.flag == GMSH_ExitFlag.SUCCESS, f"GMSH O-mesh failed with flag {out.flag}.\nUnderlying Exception:\n{error_msg}"
    assert out.mesh_path is not None

    mesh_file_su2 = Path(out.mesh_path)
    mesh_file_vtk = Path(out.mesh_path_vtk)

    assert mesh_file_su2.exists(), f"SU2 mesh file was not written to disk at {mesh_file_su2}"
    assert mesh_file_su2.stat().st_size > 0, "SU2 mesh file is empty"
    assert mesh_file_vtk.exists(), f"VTK mesh file was not written to disk at {mesh_file_vtk}"
    assert mesh_file_vtk.stat().st_size > 0, "VTK mesh file is empty"

    assert_no_inverted_cells(mesh_file_su2)

