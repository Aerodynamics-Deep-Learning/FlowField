"""
Conduct tests on the step where we test the output flagging procedure of the Gmsh pipeline, i.e.:

Step 1: Ensure numeric/bluntness/freestream errors return expected
    - test_runner_tensor_fail_routing
    - test_runner_trailingedge_fail_routing
    - test_runner_freestream_fail_routing
Step 2: Ensure mesh pass/fail conditions return as expected
    - test_runner_extrusion_fail
    - test_runner_jacobian_fail
    - test_runner_fatal_error
    - test_runner_success
"""

import pytest
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock

from src.datagen.schemas import Airfoil, Freestream
from src.datagen.meshing.gmsh.schemas import GMSH_In, GMSH_Out, GMSH_ExitFlag, GMSH_MeshingConfig
from src.datagen.meshing.gmsh.run import GMSH_MeshGenerator


def create_mock_gmsh_input():
    """Sterile input schema factory for the following tests"""
    data = MagicMock()
    
    # Use tempfile.gettempfir() to prevent an additional file being written during tests and iwndows import errs
    data.working_dir = tempfile.gettempdir()
    
    # Use .model_construct() to bypass Pydantic's strict type validation during tests
    data.airfoil = Airfoil.model_construct(
        airfoil_name="naca0012", 
        coords_tensor=MagicMock(), 
        chord=1.0
    )
    data.freestream = Freestream.model_construct(
        Re=1e6, 
        mach=0.1, 
        alpha=0.0
    )
    data.meshing_config = GMSH_MeshingConfig.model_construct(target_yplus=1.0)
    
    return data

#region Step 1
# Test the orchestrator fail condition due to the airfoil tensor
@patch("src.datagen.meshing.gmsh.run.GMSH_validate_tensor_numeric")
def test_runner_tensor_fail_routing(mock_numeric):
    # Force the numeric validator to fail
    mock_numeric.return_value = (False, "Mock numeric error")

    mock_gmsh_input = create_mock_gmsh_input()
    out = GMSH_MeshGenerator(mock_gmsh_input)
    
    assert out.flag == GMSH_ExitFlag.TENSOR_FAIL
    assert out.verbose_list == [None, None, None]

# Test the orchestrator fail condition due to the trailing edge
@patch("src.datagen.meshing.gmsh.run.GMSH_validate_tensor_numeric", return_value=(True, ""))
@patch("src.datagen.meshing.gmsh.run.GMSH_validate_te_bluntness")
def test_runner_trailingedge_fail_routing(mock_bluntness, mock_numeric):
    # Force bluntness to fail (numeric passes)
    mock_bluntness.return_value = (False, "Mock bluntness error")

    mock_gmsh_input = create_mock_gmsh_input()
    out = GMSH_MeshGenerator(mock_gmsh_input)
    
    assert out.flag == GMSH_ExitFlag.BLUNTING_FAIL
    assert out.verbose_list == [None, None, None]

# Test the orchestrator fail condition due to the non-physical mach
@patch("src.datagen.meshing.gmsh.run.GMSH_validate_tensor_numeric", return_value=(True,""))
@patch("src.datagen.meshing.gmsh.run.GMSH_validate_te_bluntness", return_value=(True,""))
@patch("src.datagen.meshing.gmsh.run.GMSH_validate_freestream_physicality")
def test_runner_freestream_fail_routing(mock_freestream, mock_bluntness, mock_numeric):
    # Force freestream to fail (numeric and bluntness passes)
    mock_freestream.return_value = (False, "Mock freestream error")

    mock_gmsh_input = create_mock_gmsh_input()
    out = GMSH_MeshGenerator(mock_gmsh_input)

    assert out.flag == GMSH_ExitFlag.FREESTREAM_FAIL
    assert out.verbose_list == [None, None, None]
#endregion

#region Step 2

# First, we'll get a tuple of all the upstream API changes, to make things easier
api_patches = (
    patch("src.datagen.meshing.gmsh.run.gmsh"),
    patch("src.datagen.meshing.gmsh.run.GMSH_export_sicn_histogram"),
    patch("src.datagen.meshing.gmsh.run.generate_cmesh", return_value="/tmp/mock.brep"),
    patch("src.datagen.meshing.gmsh.run.GMSH_get_mesh_height", return_value=1e-5),
    patch("src.datagen.meshing.gmsh.run.GMSH_validate_freestream_physicality", return_value=(True,"")),
    patch("src.datagen.meshing.gmsh.run.GMSH_validate_te_bluntness", return_value=(True,"")),
    patch("src.datagen.meshing.gmsh.run.GMSH_validate_tensor_numeric", return_value=(True,""))
)

# Function to mass apply patches
def apply_api_patches(func):
    for api in api_patches:
        func=api(func)
    return func

@apply_api_patches
def test_runner_extrusion_fail(mock_gmsh, *args):
    mock_input = create_mock_gmsh_input()

    # Simulating a mesh of triangle but no quad
    mock_gmsh.model.mesh.getElementsByType.side_effect = [
        (np.array([1,2]), None), # Some trigs
        (np.array([]), None)     # No quads
    ]
    mock_gmsh.model.mesh.getNodes.return_value = ([1,2], None, None)
    mock_gmsh.model.mesh.getElementQualities.return_value = [1.0, 1.0]

    out = GMSH_MeshGenerator(mock_input)
    assert out.flag == GMSH_ExitFlag.EXTRUSION_FAIL
    mock_gmsh.finalize.assert_called_once()

@apply_api_patches
def test_runner_jacobian_fail(mock_gmsh, *args):
    mock_input = create_mock_gmsh_input()

    # simulating a valid topology of trigs and quads
    mock_gmsh.model.mesh.getElementsByType.side_effect = [
        (np.array([1]), None), 
        (np.array([2]), None)
    ]
    mock_gmsh.model.mesh.getNodes.return_value = ([1, 2], None, None)

    # Inject a negative mesh quality (SICN < 0 in second element)
    mock_gmsh.model.mesh.getElementQualities.return_value = [0.8, -0.1]

    out = GMSH_MeshGenerator(mock_input)
    
    assert out.flag == GMSH_ExitFlag.NEGATIVE_JACOBIAN
    mock_gmsh.finalize.assert_called_once()

@apply_api_patches
def test_runner_fatal_error(mock_gmsh, *args):
    mock_input = create_mock_gmsh_input()
    
    # args[4] maps to generate_cmesh based on the tuple order (bottom-up mapping)
    mock_generate_cmesh = args[1]
    mock_generate_cmesh.side_effect = RuntimeError("Mesher segfaulted")
    
    with patch("src.datagen.meshing.gmsh.run.GMSH_Write_Exception") as mock_writer:
        mock_writer.return_value = "/tmp/exception.txt"
        
        out = GMSH_MeshGenerator(mock_input)
        
        assert out.flag == GMSH_ExitFlag.FATAL_ERROR
        assert out.verbose_list[2] == "/tmp/exception.txt"
        mock_gmsh.finalize.assert_called_once()

@apply_api_patches
def test_runner_success(mock_gmsh, *args):
    mock_input = create_mock_gmsh_input()
    
    # Inject valid topology and strictly positive qualities
    mock_gmsh.model.mesh.getElementsByType.side_effect = [
        (np.array([1]), None), 
        (np.array([2]), None)
    ]
    mock_gmsh.model.mesh.getNodes.return_value = ([1, 2], None, None)
    mock_gmsh.model.mesh.getElementQualities.return_value = [0.8, 0.9]
    
    out = GMSH_MeshGenerator(mock_input)
    
    assert out.flag == GMSH_ExitFlag.SUCCESS
    assert mock_gmsh.write.call_count == 2  # Assert it attempted to save .su2 and .vtk
    mock_gmsh.finalize.assert_called_once()
#endregion

