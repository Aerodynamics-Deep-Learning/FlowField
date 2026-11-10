"""Manual/visual counterpart to `test_gmsh_integration.py` / `test_c2d_integration.py`.

Runs 5 meshes: Emre's baseline, gmsh's two topologies (cmesh/omesh), and c2d's two topologies (OGRD/CGRD), and
shows them in the shared mesh viewer (`tests/mesh_viewer.py`), one folder each, so a person can eyeball the
differences directly: keys 1-5 pick a mesh (1=baseline, 2=gmsh cmesh, 3=gmsh omesh, 4=c2d OGRD, 5=c2d CGRD),
't' toggles edges, 'c' cycles the quality metric and 'n' the colour scale; the viewer's docstring has the rest.
The "gate" scale here is `meshing/common/constants.py`'s, the pipeline's own verdict.

Note: gmsh's cmesh/omesh and c2d's OGRD/CGRD are each backend's own independent topology
choice, not a matched pair across backends, and they are not fed the same TE. gmsh's C-mesh gets an
open TE (it builds its TE line between the end points); the rest get closed contours: both
O-grids the same rounded NACA0012 (`tests/closed_airfoils.py`), c2d's C-grid a sharp one. c2d
meshes those points as given, so the rounded TE in "c2d OGRD" is the input's, not c2d's. Here the
gmsh O-mesh is fed that rounded airfoil rather than its sharp `sterile_gmsh_omesh_input` default,
so the two O-grids mesh the same geometry. Don't read "gmsh omesh" and "c2d OGRD" as the same
case; see `conftest.py`'s fixture docstrings.

Nothing to select, so no .yaml: the five meshes are the fixtures'.

    python -m pytest tests/integration/datagen/mesh/test_manual_visual_comparison.py -m interactive -n 0 -s
"""

import pytest

from src.datagen.utils.tools import find_tool
from src.datagen.meshing.gmsh.run import GMSH_MeshGenerator
from src.datagen.meshing.gmsh.schemas import GMSH_ExitFlag
from src.datagen.meshing.c2d.run import C2D_MeshGenerator, C2D_find_exe
from src.datagen.meshing.c2d.schemas import C2D_ExitFlag
from tests.closed_airfoils import closed_naca0012
from tests.mesh_viewer import Entry, Folder, MeshViewer

pytestmark = [
    # Out of default runs; opt in with `-m interactive`.
    pytest.mark.interactive,
    pytest.mark.skipif(
        C2D_find_exe() is None, reason="c2d executable not built in this environment"
    ),
    pytest.mark.skipif(find_tool("gmsh") is None, reason="gmsh executable not found (set it in tools.toml)"),
]


def test_visual_comparison_gmsh_vs_c2d_vs_baseline(
    sterile_gmsh_input, sterile_gmsh_omesh_input,
    sterile_c2d_input, sterile_c2d_cgrd_input,
    baseline_mesh_path,
):
    gmsh_cmesh_out = GMSH_MeshGenerator(sterile_gmsh_input)
    assert gmsh_cmesh_out.flag == GMSH_ExitFlag.SUCCESS, f"gmsh cmesh failed with flag {gmsh_cmesh_out.flag}"

    # The same rounded TE c2d's O-grid gets, so the two O-grids mesh one geometry
    gmsh_omesh_in = sterile_gmsh_omesh_input.model_copy(
        update={"airfoil": closed_naca0012("naca0012_omesh_round", te="round")})
    gmsh_omesh_out = GMSH_MeshGenerator(gmsh_omesh_in)
    assert gmsh_omesh_out.flag == GMSH_ExitFlag.SUCCESS, f"gmsh omesh failed with flag {gmsh_omesh_out.flag}"

    c2d_ogrd_out = C2D_MeshGenerator(sterile_c2d_input)
    assert c2d_ogrd_out.flag == C2D_ExitFlag.SUCCESS, (
        f"c2d OGRD failed with flag {c2d_ogrd_out.flag}"
    )

    c2d_cgrd_out = C2D_MeshGenerator(sterile_c2d_cgrd_input)
    assert c2d_cgrd_out.flag == C2D_ExitFlag.SUCCESS, (
        f"c2d CGRD failed with flag {c2d_cgrd_out.flag}"
    )

    # The backends' .su2, scored as the pipeline scores it; the baseline exists only as a .vtk
    meshes = [
        ("baseline (example_naca0012.vtk)", baseline_mesh_path),
        ("gmsh cmesh", gmsh_cmesh_out.mesh_path),
        ("gmsh omesh (rounded TE)", gmsh_omesh_out.mesh_path),
        ("c2d OGRD (rounded TE)", c2d_ogrd_out.mesh_path),
        ("c2d CGRD", c2d_cgrd_out.mesh_path),
    ]
    MeshViewer([Folder(name, [Entry(mesh_path=str(path))]) for name, path in meshes]).show()
