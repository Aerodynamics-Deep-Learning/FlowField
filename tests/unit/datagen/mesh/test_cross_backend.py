"""
Tests for `common.entry.Common_GenerateMesh` across both backends; the same Airfoil/Freestream
through each, landing on a `MeshOut` the shared `.su2` quality path can score, i.e.:

Step 1: Ensure the gmsh backend lands on a scorable `.su2` carrying the SU2 boundary markers
    - test_gmsh_backend_produces_scorable_su2
Step 2: Ensure the c2d backend lands on the same, under the topology *it* builds for a blunt TE,
        and that this pairing (the production one) scores clean on *both* verdict passes
    - test_c2d_backend_produces_scorable_su2

Scope: both backends mesh for real here, so this is the one place the two are held against each
other rather than against mocks. The dispatcher's own routing is covered in test_common_entry.py.
The c2d half is skipped when its executable isn't built in the environment.
"""

from pathlib import Path

import pytest
import torch

# This module meshes for real through both backends, so the SDK is a hard requirement here.
pytest.importorskip("gmsh", reason="gmsh Python SDK not installed")

from src.datagen.schemas import Airfoil, Freestream
from src.datagen.meshing.gmsh.schemas import GMSH_CMeshingConfig
from src.datagen.meshing.c2d.run import C2D_find_exe
from src.datagen.meshing.c2d.schemas import C2D_MeshingConfig
from src.datagen.meshing.common.schemas import MeshIn, MeshBackend, MeshTopology, MeshExitFlag
from src.datagen.meshing.common.entry import Common_GenerateMesh
from src.datagen.solvers.su2.schemas import SU2_SolverConfig


# Taken from the SOLVER's schema, not hardcoded: these are the names the SU2 config step will look
# for, so both meshers must emit exactly them or the solver can't find its boundaries.
_EXPECTED_MARKERS = {SU2_SolverConfig().marker_airfoil, SU2_SolverConfig().marker_farfield}


def _su2_marker_tags(su2_path) -> set:
    """The MARKER_TAG names a written `.su2` declares."""
    tags = set()
    with open(su2_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("MARKER_TAG="):
                tags.add(line.split("=", 1)[1].strip())
    return tags

_NACA0012_UPPER = [
    [1.0, 0.00126], [0.9, 0.01055], [0.8, 0.01816], [0.7, 0.02412], [0.6, 0.02824],
    [0.5, 0.03038], [0.4, 0.03039], [0.3, 0.02797], [0.2, 0.02285], [0.1, 0.01448],
    [0.05, 0.00908], [0.0125, 0.00443], [0.0, 0.0],
]
_NACA0012_LOWER = [
    [0.0125, -0.00443], [0.05, -0.00908], [0.1, -0.01448], [0.2, -0.02285],
    [0.3, -0.02797], [0.4, -0.03039], [0.5, -0.03038], [0.6, -0.02824],
    [0.7, -0.02412], [0.8, -0.01816], [0.9, -0.01055], [1.0, -0.00126],
]


def _airfoil():
    coords = torch.tensor(_NACA0012_UPPER + _NACA0012_LOWER, dtype=torch.float32)
    le_idx = int(torch.argmin(coords[:, 0]).item())
    return Airfoil(airfoil_name="naca0012_xcheck", coords_tensor=coords, chord=1.0, le_idx=le_idx)


def _freestream():
    return Freestream(alpha=0.0, Re=5e6, mach=0.5, altitude_m=1.0)


# region Step 1
def test_gmsh_backend_produces_scorable_su2(tmp_path):
    data = MeshIn(
        airfoil=_airfoil(), freestream=_freestream(), working_dir=str(tmp_path),
        backend=MeshBackend.GMSH,
        # The fixture airfoil is blunt-TE, so each backend gets the topology *it* builds for a
        # blunt TE -- gmsh's C-mesh here, c2d's O-grid below. The two don't line up.
        topology=MeshTopology.CGRD,
        mesh_config=GMSH_CMeshingConfig(
            upper_anchor_idx=6, lower_anchor_idx=18,
            nx_le1=20, nx_le2=20, nx_upper=60, nx_lower=60, nx_wake=30,
        ),
    )
    out = Common_GenerateMesh(data)

    assert out.flag in (MeshExitFlag.SUCCESS, MeshExitFlag.LOW_QUALITY), out.flag
    assert out.mesh_path is not None
    assert out.quality is not None
    assert _su2_marker_tags(out.mesh_path) == _EXPECTED_MARKERS
    # `_score_mesh` measures deviation only on a mesh the quality pass accepted, so the two are
    # tied. 
    assert (out.geo_dev is not None) == (out.flag == MeshExitFlag.SUCCESS)
# endregion


# region Step 2
@pytest.mark.skipif(C2D_find_exe() is None, reason="c2d executable not built in this environment")
def test_c2d_backend_produces_scorable_su2(tmp_path):
    data = MeshIn(
        airfoil=_airfoil(), freestream=_freestream(), working_dir=str(tmp_path),
        backend=MeshBackend.C2D,
        topology=MeshTopology.OGRD,
        mesh_config=C2D_MeshingConfig(),
    )
    out = Common_GenerateMesh(data)

    # Strict SUCCESS, unlike the gmsh half above: this is the pairing the pipeline is meant to run
    # in production, and it clears the gates with room, so anything less is a regression rather than a tight bar.
    assert out.flag == MeshExitFlag.SUCCESS, out.flag
    assert out.mesh_path is not None
    assert out.quality is not None and out.quality.acceptable
    assert Path(out.mesh_path).is_file() and Path(out.mesh_path_vtk).is_file()
    # Same marker names as gmsh writes (one SU2 config has to drive either backend's mesh)
    assert _su2_marker_tags(out.mesh_path) == _EXPECTED_MARKERS
    # The second verdict pass, which nothing else observes arriving on a MeshOut: the integration
    # tier scores deviation by calling `Common_evaluate_mesh_geo_dev` directly, never through the
    # dispatcher, so without this `MeshOut.geo_dev` could stay None on a clean run undetected.
    assert out.geo_dev is not None, "quality passed but deviation was never measured"
    assert out.geo_dev.acceptable, out.geo_dev
# endregion
