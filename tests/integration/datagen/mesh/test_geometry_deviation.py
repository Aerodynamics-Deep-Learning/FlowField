"""
Real-mesher geometric deviation: does the meshed MARKER_AIRFOIL boundary reproduce the airfoil it
was built from, when a real gmsh kernel / real c2d.exe does the meshing? This is the tier that matters 
for `common.geo_dev`. Its unit tests pin the arithmetic against synthetic boundaries, but the thing the 
check exists to catch (a spline resampled at stations the input never specified) only happens when a 
real mesher runs.

Deviation is measured directly on each backend's output rather than through `Common_GenerateMesh`,
so a quality verdict cannot mask it: the O-mesh below scores LOW_QUALITY, which would short-circuit
the deviation pass in the real dispatcher.

`GEO_DEV_LIMIT` was calibrated from these runs, not guessed. Measured maxima at the time it
was set (chord-normalized, worst of the two directions):

    gmsh C-mesh, nx_upper=nx_lower=60/100/150/250/400 : 1.03e-4 -> 3.07e-5
    gmsh O-mesh, nx_afoil=100/200/400                 : 7.74e-4 -> 4.74e-5
    c2d O-grid                                        : 5.13e-4

The ~3.0e-5 floor does not move with mesh resolution because it is not a meshing error: it is the
sagitta between the input polyline and the spline interpolated through it, so it is set by the
*input's* point density. A deviation that actually indicates wrong geometry is orders larger;
a boundary meshed at the wrong scale measured 5.3e-2.
"""

from pathlib import Path

import pytest

pytest.importorskip("gmsh", reason="gmsh Python SDK not installed")

from src.datagen.meshing.c2d.run import C2D_MeshGenerator, C2D_find_exe
from src.datagen.meshing.gmsh.run import GMSH_MeshGenerator
from src.datagen.meshing.gmsh.schemas import GMSH_ExitFlag
from src.datagen.meshing.common.geo_dev import Common_evaluate_mesh_geo_dev, GEO_DEV_LIMIT
from src.datagen.meshing.common.schemas import MeshExitFlag


def _assert_within_limit(mesh_path, airfoil, label):
    """Scores one produced mesh and asserts both directions sit inside the limit."""
    flag, s = Common_evaluate_mesh_geo_dev(mesh_path, airfoil)

    assert s is not None, f"{label}: deviation could not be measured at all from {mesh_path}"
    detail = (f"{label}: mesh->input max {s.max_dev_mesh_to_input:.3e} (rms "
              f"{s.rms_dev_mesh_to_input:.3e}), input->mesh max {s.max_dev_input_to_mesh:.3e} (rms "
              f"{s.rms_dev_input_to_mesh:.3e}), {s.n_boundary_nodes} boundary nodes, chord "
              f"{s.chord:.6f}, limit {GEO_DEV_LIMIT:.3e}")
    assert flag == MeshExitFlag.SUCCESS, detail
    return s


def test_gmsh_cmesh_boundary_reproduces_the_airfoil(sterile_gmsh_input):
    out = GMSH_MeshGenerator(sterile_gmsh_input)
    assert out.flag == GMSH_ExitFlag.SUCCESS, f"gmsh C-mesh failed with flag {out.flag}"

    s = _assert_within_limit(out.mesh_path, sterile_gmsh_input.airfoil, "gmsh C-mesh")

    # gmsh builds its CAD at Airfoil.chord, so the written boundary carries physical size
    assert s.chord == pytest.approx(sterile_gmsh_input.airfoil.chord, rel=1e-4)


def test_gmsh_omesh_boundary_reproduces_the_airfoil(sterile_gmsh_omesh_input):
    out = GMSH_MeshGenerator(sterile_gmsh_omesh_input)
    assert out.flag == GMSH_ExitFlag.SUCCESS, f"gmsh O-mesh failed with flag {out.flag}"

    _assert_within_limit(out.mesh_path, sterile_gmsh_omesh_input.airfoil, "gmsh O-mesh")


@pytest.mark.skipif(C2D_find_exe() is None, reason="c2d executable not built in this environment")
def test_c2d_boundary_reproduces_the_airfoil(sterile_c2d_input):
    out = C2D_MeshGenerator(sterile_c2d_input)
    assert out.mesh_path is not None and Path(out.mesh_path).exists(), (
        f"c2d produced no mesh to score, flag {out.flag}")

    s = _assert_within_limit(out.mesh_path, sterile_c2d_input.airfoil, "c2d O-grid")

    # c2d re-normalizes whatever geometry it reads back to unit chord, so unlike gmsh its boundary
    # is chord-1 regardless of Airfoil.chord. Pinned here because it is the reason `geo_dev`
    # normalizes by the measured span rather than by Airfoil.chord.
    assert s.chord == pytest.approx(1.0, rel=1e-4)


def test_input_to_mesh_deviation_falls_with_surface_resolution(sterile_gmsh_omesh_input):
    """
    The two deviation directions measure different things, and this is the evidence.

    Coarsening the boundary leaves every mesh node sitting on the spline, so mesh->input barely
    moves (but the boundary starts chording across curvature the input specified, which only
    input->mesh sees). If this ever stops holding, the second direction has become decoration.
    """
    measured = {}
    for nx in (100, 400):
        gmsh_in = sterile_gmsh_omesh_input.model_copy(deep=True)
        gmsh_in.meshing_config = gmsh_in.meshing_config.model_copy(update={"nx_afoil": nx})
        out = GMSH_MeshGenerator(gmsh_in)
        assert out.flag == GMSH_ExitFlag.SUCCESS, f"gmsh O-mesh nx_afoil={nx} failed: {out.flag}"

        _flag, s = Common_evaluate_mesh_geo_dev(out.mesh_path, gmsh_in.airfoil)
        assert s is not None, f"nx_afoil={nx}: deviation could not be measured"
        measured[nx] = s

    coarse, fine = measured[100], measured[400]
    assert fine.max_dev_input_to_mesh < coarse.max_dev_input_to_mesh / 4.0, (
        f"input->mesh barely responded to resolution: {coarse.max_dev_input_to_mesh:.3e} at "
        f"nx_afoil=100 vs {fine.max_dev_input_to_mesh:.3e} at nx_afoil=400")
    assert fine.max_dev_mesh_to_input == pytest.approx(coarse.max_dev_mesh_to_input, rel=0.5), (
        f"mesh->input should be near-constant (it is the input's own sagitta floor): "
        f"{coarse.max_dev_mesh_to_input:.3e} vs {fine.max_dev_mesh_to_input:.3e}")
