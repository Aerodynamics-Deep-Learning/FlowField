"""
Tests for the shared `common.entry.Common_GenerateMesh` dispatcher, i.e.:

Step 1: Ensure the shared input-validation short-circuits route to the unified MeshExitFlag
    - test_entry_tensor_fail_routing
    - test_entry_te_topology_fail_routing
    - test_entry_freestream_fail_routing
Step 2: Ensure a gmsh GMSH_ExitFlag maps onto the equivalent MeshExitFlag
    - test_entry_gmsh_flag_mapping
Step 3: Ensure a TE-shape/topology mismatch costs what that backend says it costs
    - test_entry_gmsh_te_mismatch_is_rejected
    - test_entry_c2d_te_mismatch_warns_and_proceeds
Step 4: Ensure an unrecognised backend is flagged rather than dispatched anywhere
    - test_entry_unknown_backend_dispatches_to_neither
    - test_entry_backend_missing_from_dispatch_table_is_flagged
Step 5: Ensure nothing escapes as an exception -- every failure leaves as a MeshExitFlag
    - test_entry_gmsh_input_validation_error_becomes_flag
    - test_entry_c2d_input_validation_error_becomes_flag
    - test_entry_backend_exception_becomes_flag
    - test_entry_unparseable_mesh_keeps_its_paths
    - test_entry_quality_exception_becomes_flag

Scope: this file covers the dispatcher only. Each backend's own runner is tested in
test_gmsh_runner.py / test_c2d_runner.py, the shared validators in test_common_utils.py, and the
MeshIn contract in test_common_schemas.py.
"""

import logging
import tempfile
from unittest.mock import patch, MagicMock

import pytest
import torch
from pydantic import BaseModel, ValidationError

# The dispatcher imports gmsh/run.py, so the SDK must be present even though it is patched here.
pytest.importorskip("gmsh", reason="gmsh Python SDK not installed")

from src.datagen.schemas import Airfoil, Freestream
from src.datagen.meshing.gmsh.schemas import (
    GMSH_CMeshingConfig, GMSH_OMeshingConfig, GMSH_ExitFlag,
)
from src.datagen.meshing.c2d.schemas import C2D_ExitFlag, C2D_MeshingConfig
from src.datagen.meshing.common.schemas import MeshIn, MeshBackend, MeshTopology, MeshExitFlag
from src.datagen.meshing.common.entry import Common_GenerateMesh


def _mock_mesh_in(backend=MeshBackend.GMSH, topology=MeshTopology.CGRD):
    return MeshIn.model_construct(
        airfoil=Airfoil.model_construct(airfoil_name="naca0012", coords_tensor=MagicMock(), chord=1.0),
        freestream=Freestream.model_construct(Re=1e6, mach=0.1, alpha=0.0),
        working_dir=tempfile.gettempdir(),
        backend=backend,
        topology=topology,
        mesh_config=GMSH_CMeshingConfig.model_construct(target_yplus=1.0),
    )


# region Step 1
@patch("src.datagen.meshing.common.entry.Common_validate_tensor_numeric")
def test_entry_tensor_fail_routing(mock_numeric):
    mock_numeric.return_value = (False, "Mock numeric error")
    out = Common_GenerateMesh(_mock_mesh_in())
    assert out.flag == MeshExitFlag.INPUT_TENSOR_FAIL


@patch("src.datagen.meshing.common.entry.Common_validate_tensor_numeric", return_value=(True, ""))
@patch("src.datagen.meshing.common.entry.Common_validate_freestream_physicality", return_value=(True, ""))
@patch("src.datagen.meshing.common.entry.Common_validate_te_for_topology")
def test_entry_te_topology_fail_routing(mock_te, *args):
    mock_te.return_value = (False, "Mock TE/topology mismatch")
    out = Common_GenerateMesh(_mock_mesh_in(backend=MeshBackend.GMSH))
    assert out.flag == MeshExitFlag.INPUT_BLUNTING_FAIL


@patch("src.datagen.meshing.common.entry.Common_validate_tensor_numeric", return_value=(True, ""))
@patch("src.datagen.meshing.common.entry.Common_validate_te_for_topology", return_value=(True, ""))
@patch("src.datagen.meshing.common.entry.Common_validate_freestream_physicality")
def test_entry_freestream_fail_routing(mock_freestream, mock_bluntness, mock_numeric):
    mock_freestream.return_value = (False, "Mock freestream error")
    out = Common_GenerateMesh(_mock_mesh_in())
    assert out.flag == MeshExitFlag.INPUT_FREESTREAM_FAIL
# endregion


# region Step 2
@patch("src.datagen.meshing.common.entry.Common_validate_tensor_numeric", return_value=(True, ""))
@patch("src.datagen.meshing.common.entry.Common_validate_te_for_topology", return_value=(True, ""))
@patch("src.datagen.meshing.common.entry.Common_validate_freestream_physicality", return_value=(True, ""))
@patch("src.datagen.meshing.common.entry.GMSH_MeshGenerator")
def test_entry_gmsh_flag_mapping(mock_generator, *args):
    mock_out = MagicMock()
    mock_out.flag = GMSH_ExitFlag.EXTRUSION_FAIL
    mock_out.mesh_path = None
    mock_out.mesh_path_vtk = None
    mock_out.num_nodes = None
    mock_out.verbose_list = [None, None, None]
    mock_generator.return_value = mock_out

    out = Common_GenerateMesh(_mock_mesh_in())
    assert out.flag == MeshExitFlag.CONVERSION_FAIL
    assert out.backend == MeshBackend.GMSH
# endregion


# region Step 3
_BLUNT_TE_COORDS = [[1.0, 0.0063], [0.5, 0.06], [0.0, 0.0], [0.5, -0.06], [1.0, -0.0063]]


def _blunt_mesh_in(backend, topology, **configs):
    """A real (validated) MeshIn on a blunt-TE airfoil, so the TE/topology gate sees real coords."""
    coords = torch.tensor(_BLUNT_TE_COORDS, dtype=torch.float32)
    return MeshIn(
        airfoil=Airfoil(airfoil_name="naca0012_blunt", coords_tensor=coords, chord=1.0, le_idx=2),
        freestream=Freestream(alpha=0.0, Re=5e6, mach=0.3, altitude_m=1.0),
        working_dir=tempfile.gettempdir(),
        backend=backend,
        topology=topology,
        **configs,
    )


@patch("src.datagen.meshing.common.entry.GMSH_MeshGenerator")
def test_entry_gmsh_te_mismatch_is_rejected(mock_generator):
    # gmsh's O-mesh treats the TE as one point; handed a blunt airfoil it would silently snap the
    # lower TE onto the upper one, so the pairing has to be refused before the builder ever runs.
    data = _blunt_mesh_in(MeshBackend.GMSH, MeshTopology.OGRD,
                          mesh_config=GMSH_OMeshingConfig())
    out = Common_GenerateMesh(data)

    assert out.flag == MeshExitFlag.INPUT_BLUNTING_FAIL
    assert out.topology == MeshTopology.OGRD
    assert not mock_generator.called


@patch("src.datagen.meshing.common.entry.C2D_MeshGenerator")
def test_entry_c2d_te_mismatch_warns_and_proceeds(mock_generator, caplog):
    # c2d only recommends OGRD for a blunt TE; CGRD still meshes it after the y/n
    # override, so a mismatch must warn rather than short-circuit.
    mock_out = MagicMock()
    mock_out.flag = C2D_ExitFlag.SUBPROCESS_FAIL
    mock_out.mesh_path = None
    mock_out.mesh_path_vtk = None
    mock_out.num_nodes = None
    mock_out.verbose_list = []
    mock_generator.return_value = mock_out

    data = _blunt_mesh_in(MeshBackend.C2D, MeshTopology.CGRD, mesh_config=C2D_MeshingConfig())
    with caplog.at_level(logging.WARNING):
        out = Common_GenerateMesh(data)

    assert mock_generator.called
    assert out.flag == MeshExitFlag.SUBPROCESS_FAIL
    assert any("sharp trailing edge" in r.message.lower() for r in caplog.records)
# endregion


# region Step 4
@patch("src.datagen.meshing.common.entry.C2D_MeshGenerator")
@patch("src.datagen.meshing.common.entry.GMSH_MeshGenerator")
def test_entry_unknown_backend_dispatches_to_neither(mock_gmsh, mock_c2d):
    # Dispatch is `if GMSH ... else c2d`, so this guard is the only thing keeping an unrecognised
    # backend out of c2d. MeshIn accepts the pairing on purpose (flagging it is the entry's job).
    data = _blunt_mesh_in(MeshBackend.UNKNOWN, MeshTopology.CGRD)
    assert data.mesh_config is None  # no config class exists for an unsupported pairing

    out = Common_GenerateMesh(data)

    assert out.flag == MeshExitFlag.INPUT_UNKNOWN_SOLVER
    assert out.backend == MeshBackend.UNKNOWN
    assert out.topology == MeshTopology.CGRD
    assert not mock_gmsh.called and not mock_c2d.called
    assert out.mesh_path is None and out.quality is None


@patch("src.datagen.meshing.common.entry.C2D_MeshGenerator")
@patch("src.datagen.meshing.common.entry.GMSH_MeshGenerator")
def test_entry_backend_missing_from_dispatch_table_is_flagged(mock_gmsh, mock_c2d):
    # The guard tests membership of _TE_MISMATCH_IS_FATAL rather than equality with UNKNOWN, so a
    # backend added to the enum but never wired into the maps fails safe. Without it the TE gate
    # would return False and `_TE_MISMATCH_IS_FATAL[backend]` would raise KeyError straight out of
    # the dispatcher, breaking its every-failure-is-a-flag contract.
    data = _blunt_mesh_in(MeshBackend.GMSH, MeshTopology.CGRD, mesh_config=GMSH_CMeshingConfig())
    unwired = data.model_copy(update={"backend": "fenics"})  # bypasses validation, as a new enum member would

    out = Common_GenerateMesh(unwired)

    assert out.flag == MeshExitFlag.INPUT_UNKNOWN_SOLVER
    assert out.backend == MeshBackend.UNKNOWN
    assert not mock_gmsh.called and not mock_c2d.called
# endregion


# region Step 5
def _a_validation_error() -> ValidationError:
    """A genuine pydantic ValidationError, as GMSH_In/C2D_In would raise on a bad config."""
    class _M(BaseModel):
        x: int

    with pytest.raises(ValidationError) as excinfo:
        _M(x="not-an-int")
    return excinfo.value


@patch("src.datagen.meshing.common.entry.GMSH_MeshGenerator")
@patch("src.datagen.meshing.common.entry.GMSH_In")
def test_entry_gmsh_input_validation_error_becomes_flag(mock_in, mock_generator):
    # Unreachable while MeshIn's resolver and _CONFIG_FOR_PAIRING agree; this pins what happens if
    # they ever drift, since the caller contract is a flag rather than a raise.
    mock_in.side_effect = _a_validation_error()

    out = Common_GenerateMesh(_blunt_mesh_in(MeshBackend.GMSH, MeshTopology.CGRD))

    assert out.flag == MeshExitFlag.FATAL_ERROR
    assert out.backend == MeshBackend.GMSH
    assert not mock_generator.called  # never reached the builder


@patch("src.datagen.meshing.common.entry.C2D_MeshGenerator")
@patch("src.datagen.meshing.common.entry.C2D_In")
def test_entry_c2d_input_validation_error_becomes_flag(mock_in, mock_generator):
    mock_in.side_effect = _a_validation_error()

    out = Common_GenerateMesh(_blunt_mesh_in(MeshBackend.C2D, MeshTopology.OGRD))

    assert out.flag == MeshExitFlag.FATAL_ERROR
    assert out.backend == MeshBackend.C2D
    assert not mock_generator.called


@patch("src.datagen.meshing.common.entry.GMSH_MeshGenerator")
def test_entry_backend_exception_becomes_flag(mock_generator):
    # The backends flag their own internal failures, but their `except` blocks call the exception
    # writers (GMSH_Write_Exception / C2D_Write_Exception), which can raise on their own.
    mock_generator.side_effect = OSError("exception writer blew up on the working dir")

    out = Common_GenerateMesh(_blunt_mesh_in(MeshBackend.GMSH, MeshTopology.CGRD))

    assert out.flag == MeshExitFlag.FATAL_ERROR
    assert out.backend == MeshBackend.GMSH


@patch("src.datagen.meshing.common.entry.GMSH_MeshGenerator")
def test_entry_unparseable_mesh_keeps_its_paths(mock_generator, tmp_path):
    # The real quality pass, against a mesh that generated fine but cannot be read back. Because
    # `quality.py` handles that itself and returns CONVERSION_FAIL, the run stays on the normal
    # return path and the artifacts stay referenced (the entry backstop would have dropped them).
    broken = tmp_path / "broken.su2"
    broken.write_text("NDIME= 2\nNPOIN= 9\n0.0 0.0 0\n")
    mock_out = MagicMock()
    mock_out.flag = GMSH_ExitFlag.SUCCESS
    mock_out.mesh_path = str(broken)
    mock_out.mesh_path_vtk = str(tmp_path / "broken.vtk")
    mock_out.num_nodes = 9
    mock_out.verbose_list = ["gmsh.log"]
    mock_generator.return_value = mock_out

    out = Common_GenerateMesh(_blunt_mesh_in(MeshBackend.GMSH, MeshTopology.CGRD))

    assert out.flag == MeshExitFlag.CONVERSION_FAIL
    assert out.mesh_path == str(broken)
    assert out.mesh_path_vtk == str(tmp_path / "broken.vtk")
    assert out.verbose_list == ["gmsh.log"]
    assert out.num_nodes == 9  # falls back to the backend's count when the quality pass has none
    assert out.quality is None


@patch("src.datagen.meshing.common.entry.Common_evaluate_mesh_quality")
@patch("src.datagen.meshing.common.entry.GMSH_MeshGenerator")
def test_entry_quality_exception_becomes_flag(mock_generator, mock_quality):
    # Common_evaluate_mesh_quality has no handler of its own, so a malformed .su2 raises out of it
    # (after the mesh was generated, which is the reachable half of this contract).
    mock_out = MagicMock()
    mock_out.flag = GMSH_ExitFlag.SUCCESS
    mock_out.mesh_path = "mesh.su2"
    mock_out.mesh_path_vtk = None
    mock_out.num_nodes = 10
    mock_out.verbose_list = []
    mock_generator.return_value = mock_out
    mock_quality.side_effect = ValueError("malformed NPOIN block")

    out = Common_GenerateMesh(_blunt_mesh_in(MeshBackend.GMSH, MeshTopology.CGRD))

    assert out.flag == MeshExitFlag.FATAL_ERROR
    assert mock_quality.called
# endregion
