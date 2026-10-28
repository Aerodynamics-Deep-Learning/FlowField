"""
Real meshers on sampled input: a stored shape's contour, sampled at the default knobs by
`meshing.optimizer.sampling`, through `Common_GenerateMesh`, i.e.:

Step 1: Ensure c2d meshes a sample as it is, and well
    - test_c2d_ogrid_meshes_the_sample
    - test_c2d_cgrid_meshes_the_sample
Step 2: Ensure the gmsh O-mesh takes a sample's points as its wall
    - test_gmsh_omesh_takes_the_sample
Step 3: Ensure a study-style run writes no .vtk and reports where its worst cells sit
    - test_c2d_ogrid_reports_distributions_without_a_vtk

c2d in BUFF mode and the gmsh O-mesh keep the input points as their wall nodes, so the wall must be the
sample: to the 8 decimals c2d's `.dat` carries, and exactly for gmsh.
"""

import numpy as np
import pytest

from src.datagen.meshing.c2d.run import C2D_find_exe
from src.datagen.meshing.c2d.schemas import C2D_MeshingConfig
from src.datagen.meshing.common.entry import Common_GenerateMesh
from src.datagen.meshing.common.geo_dev import _boundary_from_su2
from src.datagen.meshing.common.quality import _cell_metrics, _read_su2
from src.datagen.meshing.common.schemas import MeshBackend, MeshExitFlag, MeshTopology, QualityLimits
from src.datagen.meshing.gmsh.schemas import GMSH_OMeshingConfig
from src.datagen.meshing.optimizer.sampling import build_study_contour, sample_for_mesher
from src.datagen.schemas import Freestream
from src.datagen.utils.tools import find_tool
from tests.geometry_shapes import stored_shape

_needs_c2d = pytest.mark.skipif(C2D_find_exe() is None, reason="c2d executable not built in this environment")
_needs_gmsh = pytest.mark.skipif(find_tool("gmsh") is None, reason="gmsh executable not found (set it in tools.toml)")


def _mesh(name, backend, topology, config, workspace, **kwargs):
    contour = build_study_contour(stored_shape(name), backend, topology)
    sampled = sample_for_mesher(config, contour, airfoil_name=name, chord=1.0,
                                freestream=Freestream(alpha=0.0, Re=5e6, mach=0.5), backend=backend,
                                topology=topology, working_dir=str(workspace), **kwargs)
    assert sampled.mesh_in is not None, sampled.report.rejection
    return sampled, Common_GenerateMesh(sampled.mesh_in)


def _worst_sample_point_off_the_wall(out, sampled) -> float:
    wall, _ = _boundary_from_su2(out.mesh_path)
    pts = sampled.mesh_in.airfoil.coords_tensor.numpy()
    return float(np.linalg.norm(pts[:, None, :] - wall[None, :, :], axis=-1).min(axis=1).max())


# region Step 1
@_needs_c2d
@pytest.mark.parametrize("name", ["naca4412", "cusp"])
def test_c2d_ogrid_meshes_the_sample(name, integration_mesh_workspace):
    config = C2D_MeshingConfig()
    sampled, out = _mesh(name, MeshBackend.C2D, MeshTopology.OGRD, config, integration_mesh_workspace)

    assert out.flag == MeshExitFlag.SUCCESS, f"{out.flag}: {out.quality}"
    assert out.num_nodes == (sampled.report.n_points - 1) * config.jmax  # its wall is the sample's points
    assert _worst_sample_point_off_the_wall(out, sampled) < 1e-7


@_needs_c2d
def test_c2d_cgrid_meshes_the_sample(integration_mesh_workspace):
    sampled, out = _mesh("naca0012", MeshBackend.C2D, MeshTopology.CGRD, C2D_MeshingConfig(),
                         integration_mesh_workspace)

    assert out.flag == MeshExitFlag.SUCCESS, f"{out.flag}: {out.quality}"
    assert _worst_sample_point_off_the_wall(out, sampled) < 1e-7
# endregion


# region Step 2
@_needs_gmsh
def test_gmsh_omesh_takes_the_sample(integration_mesh_workspace):
    # Known LOW_QUALITY at the TE whatever its input; what matters here is that the sample got through
    sampled, out = _mesh("naca4412", MeshBackend.GMSH, MeshTopology.OGRD, GMSH_OMeshingConfig(),
                         integration_mesh_workspace)

    assert out.mesh_path is not None and out.flag in (MeshExitFlag.SUCCESS, MeshExitFlag.LOW_QUALITY), out.flag
    assert _worst_sample_point_off_the_wall(out, sampled) < 1e-12
# endregion


# region Step 3
@_needs_c2d
def test_c2d_ogrid_reports_distributions_without_a_vtk(integration_mesh_workspace):
    # The mesh study's own gates, stricter than constants.py; they count, the verdict stays put
    study = QualityLimits(skew_max=0.4, ortho_min=0.75, jac_min=0.75)
    _, out = _mesh("naca0012", MeshBackend.C2D, MeshTopology.OGRD, C2D_MeshingConfig(),
                   integration_mesh_workspace, count_limits=study, write_vtk=False)
    q = out.quality

    assert out.flag == MeshExitFlag.SUCCESS, f"{out.flag}: {q}"
    assert out.mesh_path_vtk is None and not list(integration_mesh_workspace.glob("*.vtk"))
    assert q.neg == 0
    assert q.skew_dist.p50 <= q.skew_dist.t99 <= q.skew_dist.t999 <= q.max_skew
    assert q.ortho_dist.p50 >= q.ortho_dist.t99 >= q.ortho_dist.t999 >= q.min_ortho
    assert q.skew_dist.n_bad == q.ortho_dist.n_bad == q.jac_dist.n_bad == q.size_ratio_dist.n_bad == 0

    # The reported location is the worst cell's centroid, on a real mesh
    nodes, quads, _, _ = _read_su2(out.mesh_path)
    worst = int(np.argmax(_cell_metrics(nodes[quads])[1]))
    assert (q.skew_dist.x, q.skew_dist.y) == pytest.approx(tuple(nodes[quads[worst]].mean(axis=0)))
    assert 0.0 < q.skew_dist.dwall < 0.1  # near the wall, not out in the far field
# endregion
