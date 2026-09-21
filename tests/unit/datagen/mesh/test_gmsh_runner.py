"""
Tests for `GMSH_MeshGenerator`, i.e.:

Step 1: Ensure gmsh-specific structural failure modes route to the expected GMSH_ExitFlag
    (mesh-quality acceptability is decided centrally by common.entry.Common_evaluate_mesh_quality,
    not here -- see test_common_entry.py / test_cross_backend.py)
    - test_runner_extrusion_fail
    - test_runner_fatal_error
    - test_runner_success
Step 2: Ensure the two regressions found in the gmsh audit stay fixed
    - test_mesh_path_vtk_is_full_path_not_bare_filename
    - test_empty_extrusion_does_not_raise_nameerror

The c2d counterpart is test_c2d_runner.py. Input-validation short-circuit routing
(tensor/TE-shape/freestream) is NOT here: GMSH_MeshGenerator used to run those checks itself, but
they moved to common/entry.py (via common.utils.Common_validate_*; see test_common_entry.py's
Step 1, and test_common_utils.py for the validators themselves).
"""

import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# gmsh/run.py imports the SDK at module scope, so it is needed even though the API is patched.
pytest.importorskip("gmsh", reason="gmsh Python SDK not installed")

from src.datagen.schemas import Airfoil, Freestream
from src.datagen.meshing.gmsh.schemas import GMSH_ExitFlag, GMSH_CMeshingConfig, GMSH_Topology
from src.datagen.meshing.gmsh.run import GMSH_MeshGenerator


def _mock_gmsh_input():
    """Sterile `GMSH_In` stand-in for the tests below."""
    data = MagicMock()

    # tempfile.gettempdir() keeps the tests from writing into the repo (and avoids Windows path errs)
    data.working_dir = tempfile.gettempdir()

    # .model_construct() bypasses pydantic's validation, so the mocks don't need real geometry
    data.airfoil = Airfoil.model_construct(airfoil_name="naca0012", coords_tensor=MagicMock(), chord=1.0)
    data.freestream = Freestream.model_construct(Re=1e6, mach=0.1, alpha=0.0)
    data.topology = GMSH_Topology.CGRD
    data.meshing_config = GMSH_CMeshingConfig.model_construct(target_yplus=1.0)

    return data


# The gmsh API, the CAD/mesh builder and the BL-height helper are all stubbed: these tests are
# about GMSH_MeshGenerator's own flagging, not about gmsh itself.
api_patches = (
    patch("src.datagen.meshing.gmsh.run.gmsh"),
    patch("src.datagen.meshing.gmsh.run.GMSH_generate_cmesh", return_value="/tmp/mock.brep"),
    patch("src.datagen.meshing.gmsh.run.GMSH_get_mesh_height", return_value=1e-5),
)


def apply_api_patches(func):
    for api in api_patches:
        func = api(func)
    return func


# region Step 1
@apply_api_patches
def test_runner_extrusion_fail(mock_gmsh, *args):
    mock_input = _mock_gmsh_input()

    # Simulating a mesh with no quads, i.e. the BL extrusion produced nothing
    mock_gmsh.model.mesh.getElementsByType.return_value = (np.array([]), None)
    mock_gmsh.model.mesh.getNodes.return_value = ([1, 2], None, None)

    out = GMSH_MeshGenerator(mock_input)

    assert out.flag == GMSH_ExitFlag.EXTRUSION_FAIL
    mock_gmsh.finalize.assert_called_once()


@apply_api_patches
def test_runner_fatal_error(mock_gmsh, *args):
    mock_input = _mock_gmsh_input()

    # args[0] maps to generate_cmesh based on the tuple order (bottom-up mapping)
    mock_generate_cmesh = args[0]
    mock_generate_cmesh.side_effect = RuntimeError("Mesher segfaulted")

    with patch("src.datagen.meshing.gmsh.run.GMSH_Write_Exception") as mock_writer:
        mock_writer.return_value = "/tmp/exception.txt"

        out = GMSH_MeshGenerator(mock_input)

        assert out.flag == GMSH_ExitFlag.FATAL_ERROR
        assert out.verbose_list[2] == "/tmp/exception.txt"
        mock_gmsh.finalize.assert_called_once()


@apply_api_patches
def test_runner_success(mock_gmsh, *args):
    mock_input = _mock_gmsh_input()

    # Inject a non-empty quad layer
    mock_gmsh.model.mesh.getElementsByType.return_value = (np.array([2]), None)
    mock_gmsh.model.mesh.getNodes.return_value = ([1, 2], None, None)

    out = GMSH_MeshGenerator(mock_input)

    assert out.flag == GMSH_ExitFlag.SUCCESS
    assert mock_gmsh.write.call_count == 2  # Assert it attempted to save .su2 and .vtk
    mock_gmsh.finalize.assert_called_once()
# endregion


# region Step 2 (regression coverage for the two bugs found during the audit)
@apply_api_patches
def test_mesh_path_vtk_is_full_path_not_bare_filename(mock_gmsh, *args):
    mock_input = _mock_gmsh_input()
    mock_gmsh.model.mesh.getElementsByType.return_value = (np.array([2]), None)
    mock_gmsh.model.mesh.getNodes.return_value = ([1, 2], None, None)

    out = GMSH_MeshGenerator(mock_input)

    assert out.flag == GMSH_ExitFlag.SUCCESS
    assert out.mesh_path_vtk == out.mesh_path.replace("_mesh.su2", "_mesh.vtk")
    assert out.mesh_path_vtk.startswith(mock_input.working_dir)


@apply_api_patches
def test_empty_extrusion_does_not_raise_nameerror(mock_gmsh, *args):
    mock_input = _mock_gmsh_input()
    # No 2D elements at all.
    mock_gmsh.model.mesh.getElementsByType.return_value = (np.array([]), None)
    mock_gmsh.model.mesh.getNodes.return_value = ([], None, None)

    out = GMSH_MeshGenerator(mock_input)

    assert out.flag == GMSH_ExitFlag.EXTRUSION_FAIL
# endregion
