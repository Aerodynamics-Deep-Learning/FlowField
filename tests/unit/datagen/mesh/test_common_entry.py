"""
Tests for the shared `common.entry.Common_GenerateMesh` dispatcher, i.e.:

Step 1: Ensure the shared input-validation short-circuits route to the unified MeshExitFlag
    - test_entry_tensor_fail_routing
    - test_entry_te_closure_fail_routing
    - test_entry_freestream_fail_routing
Step 2: Ensure each backend's own exit flag maps onto the equivalent MeshExitFlag
    - test_entry_gmsh_flag_mapping
    - test_entry_c2d_flag_mapping
Step 3: Ensure the TE closure gate refuses, and the TE shape checks only warn
    - test_entry_te_closure_mismatch_is_refused
    - test_entry_te_shape_mismatch_warns_and_proceeds
Step 4: Ensure an unrecognised backend is flagged rather than dispatched anywhere
    - test_entry_unknown_backend_dispatches_to_neither
    - test_entry_backend_missing_from_dispatch_table_is_flagged
Step 5: Ensure nothing escapes as an exception -- every failure leaves as a MeshExitFlag
    - test_entry_gmsh_input_validation_error_becomes_flag
    - test_entry_c2d_input_validation_error_becomes_flag
    - test_entry_backend_exception_becomes_flag
    - test_entry_malformed_tensor_becomes_flag
    - test_entry_unparseable_mesh_keeps_its_paths
    - test_entry_quality_exception_becomes_flag
    - test_entry_geo_dev_exception_becomes_flag
Step 6: Ensure the config MeshIn resolved reaches the backend unmodified
    - test_entry_forwards_the_resolved_config_to_gmsh
    - test_entry_forwards_the_resolved_config_to_c2d
    - test_entry_one_c2d_config_sweeps_both_topologies
    - test_entry_forwards_the_reporting_options
Step 7: Ensure deviation is measured on every readable mesh, without overriding the quality verdict
    - test_entry_quality_verdict_outranks_deviation
    - test_entry_deviation_crash_keeps_a_failed_quality_verdict

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

from src.datagen.schemas import Airfoil, Freestream
from src.datagen.meshing.gmsh.schemas import (
    GMSH_CMeshingConfig, GMSH_ExitFlag, GMSH_Topology,
)
from src.datagen.meshing.c2d.schemas import C2D_ExitFlag, C2D_MeshingConfig, C2D_Topology
from src.datagen.meshing.common.schemas import (
    MeshIn, MeshBackend, MeshTopology, MeshExitFlag, MeshQualitySummary, MeshGeoDeviationSummary,
    QualityLimits,
)
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
@patch("src.datagen.meshing.common.entry.Common_validate_te_closure")
def test_entry_te_closure_fail_routing(mock_te, *args):
    mock_te.return_value = (False, "Mock TE closure mismatch")
    out = Common_GenerateMesh(_mock_mesh_in(backend=MeshBackend.GMSH))
    assert out.flag == MeshExitFlag.INPUT_CLOSURE_FAIL


@patch("src.datagen.meshing.common.entry.Common_validate_tensor_numeric", return_value=(True, ""))
@patch("src.datagen.meshing.common.entry.Common_validate_te_closure", return_value=(True, ""))
@patch("src.datagen.meshing.common.entry.Common_validate_freestream_physicality")
def test_entry_freestream_fail_routing(mock_freestream, mock_closure, mock_numeric):
    mock_freestream.return_value = (False, "Mock freestream error")
    out = Common_GenerateMesh(_mock_mesh_in())
    assert out.flag == MeshExitFlag.INPUT_FREESTREAM_FAIL
# endregion


# region Step 2
@pytest.mark.parametrize("gmsh_flag, expected", [
    (GMSH_ExitFlag.EXECUTABLE_NOT_FOUND, MeshExitFlag.EXECUTABLE_NOT_FOUND),
    (GMSH_ExitFlag.FATAL_ERROR, MeshExitFlag.FATAL_ERROR),
    (GMSH_ExitFlag.EXTRUSION_FAIL, MeshExitFlag.CONVERSION_FAIL),
])
@patch("src.datagen.meshing.common.entry.Common_validate_tensor_numeric", return_value=(True, ""))
@patch("src.datagen.meshing.common.entry.Common_validate_te_closure", return_value=(True, ""))
@patch("src.datagen.meshing.common.entry.Common_validate_freestream_physicality", return_value=(True, ""))
@patch("src.datagen.meshing.common.entry.GMSH_MeshGenerator")
def test_entry_gmsh_flag_mapping(mock_generator, mock_freestream, mock_te, mock_numeric, gmsh_flag, expected):
    mock_out = MagicMock()
    mock_out.flag = gmsh_flag
    mock_out.mesh_path = None
    mock_out.mesh_path_vtk = None
    mock_out.num_nodes = None
    mock_out.verbose_list = [None, None, None]
    mock_generator.return_value = mock_out

    out = Common_GenerateMesh(_mock_mesh_in())
    assert out.flag == expected
    assert out.backend == MeshBackend.GMSH


def _mock_backend_out(flag):
    """A minimal stand-in for a backend's own `*_Out`, carrying nothing but the flag."""
    out = MagicMock()
    out.flag = flag
    out.mesh_path = None
    out.mesh_path_vtk = None
    out.num_nodes = None
    out.verbose_list = []
    return out


@pytest.mark.parametrize("c2d_flag, expected", [
    (C2D_ExitFlag.EXECUTABLE_NOT_FOUND, MeshExitFlag.EXECUTABLE_NOT_FOUND),
    (C2D_ExitFlag.SUBPROCESS_FAIL, MeshExitFlag.SUBPROCESS_FAIL),
    (C2D_ExitFlag.FATAL_ERROR, MeshExitFlag.FATAL_ERROR),
    (C2D_ExitFlag.CONVERSION_FAIL, MeshExitFlag.CONVERSION_FAIL),
    (-99, MeshExitFlag.FATAL_ERROR),  # unmapped: _C2D_FLAG_MAP.get's default, as a new flag would be
])
@patch("src.datagen.meshing.common.entry.C2D_MeshGenerator")
def test_entry_c2d_flag_mapping(mock_generator, c2d_flag, expected):
    # gmsh's map has `test_entry_gmsh_flag_mapping`; c2d's five-entry one had only
    # SUBPROCESS_FAIL, and only incidentally (via the TE-mismatch test). A mistranslated
    # EXECUTABLE_NOT_FOUND/SUBPROCESS_FAIL reads as a mesher fault rather than a missing exe. SUCCESS is excluded: it continues into `_score_mesh`, which needs
    # a real mesh, and is covered by test_cross_backend.py.
    mock_generator.return_value = _mock_backend_out(c2d_flag)

    out = Common_GenerateMesh(_mesh_in(MeshBackend.C2D, MeshTopology.OGRD,
                                       mesh_config=C2D_MeshingConfig()))

    assert out.flag == expected
    assert out.backend == MeshBackend.C2D
    assert out.quality is None and out.geo_dev is None  # never scored, there is no mesh
# endregion


# region Step 3
_OPEN_TE_COORDS = [[1.0, 0.0063], [0.5, 0.06], [0.0, 0.0], [0.5, -0.06], [1.0, -0.0063]]
_CLOSED_TE_COORDS = [[1.0, 0.0], [0.5, 0.06], [0.0, 0.0], [0.5, -0.06], [1.0, 0.0]]  # sharp, ~14 deg
_NEAR_CLOSED_TE_COORDS = [[1.0, 1e-7], [0.5, 0.06], [0.0, 0.0], [0.5, -0.06], [1.0, 0.0]]


def _mesh_in(backend, topology, coords=None, **configs):
    """A real (validated) MeshIn, so the TE gates see real coords. By default the closure the pairing
    requires: open for gmsh's C-mesh, closed for everything else."""
    if coords is None:
        is_cmesh = (backend, topology) == (MeshBackend.GMSH, MeshTopology.CGRD)
        coords = _OPEN_TE_COORDS if is_cmesh else _CLOSED_TE_COORDS
    return MeshIn(
        airfoil=Airfoil(airfoil_name="naca0012_coarse", coords_tensor=torch.tensor(coords, dtype=torch.float32),
                        chord=1.0, le_idx=2),
        freestream=Freestream(alpha=0.0, Re=5e6, mach=0.3, altitude_m=1.0),
        working_dir=tempfile.gettempdir(),
        backend=backend,
        topology=topology,
        **configs,
    )


@pytest.mark.parametrize("backend, topology, coords, checklist", [
    (MeshBackend.C2D, MeshTopology.OGRD, _OPEN_TE_COORDS, ["requires a closed te", "open"]),
    (MeshBackend.C2D, MeshTopology.CGRD, _NEAR_CLOSED_TE_COORDS, ["requires a closed te", "rounding noise"]),
    # The O-mesh drops the last point as a duplicate of the first, so an open TE would silently lose it
    (MeshBackend.GMSH, MeshTopology.OGRD, _OPEN_TE_COORDS, ["requires a closed te", "open"]),
    (MeshBackend.GMSH, MeshTopology.CGRD, _CLOSED_TE_COORDS, ["requires an open te", "closed"]),
    (MeshBackend.GMSH, MeshTopology.CGRD, _NEAR_CLOSED_TE_COORDS, ["requires an open te", "too short"]),
])
@patch("src.datagen.meshing.common.entry.C2D_MeshGenerator")
@patch("src.datagen.meshing.common.entry.GMSH_MeshGenerator")
def test_entry_te_closure_mismatch_is_refused(mock_gmsh, mock_c2d, backend, topology, coords, checklist, caplog):
    with caplog.at_level(logging.WARNING):
        out = Common_GenerateMesh(_mesh_in(backend, topology, coords=coords))

    assert out.flag == MeshExitFlag.INPUT_CLOSURE_FAIL
    assert out.topology == topology
    assert not mock_gmsh.called and not mock_c2d.called
    assert any(all(c in r.message.lower() for c in checklist) for r in caplog.records)


@patch("src.datagen.meshing.common.entry.C2D_MeshGenerator")
def test_entry_te_shape_mismatch_warns_and_proceeds(mock_generator, caplog):
    # TE shape is advisory: c2d's O-grid handed a sharp closed TE warns, then meshes anyway
    mock_generator.return_value = _mock_backend_out(C2D_ExitFlag.SUBPROCESS_FAIL)

    with caplog.at_level(logging.WARNING):
        out = Common_GenerateMesh(_mesh_in(MeshBackend.C2D, MeshTopology.OGRD))

    assert mock_generator.called
    assert out.flag == MeshExitFlag.SUBPROCESS_FAIL
    assert any("sharp te" in r.message.lower() for r in caplog.records)
# endregion


# region Step 4
@patch("src.datagen.meshing.common.entry.C2D_MeshGenerator")
@patch("src.datagen.meshing.common.entry.GMSH_MeshGenerator")
def test_entry_unknown_backend_dispatches_to_neither(mock_gmsh, mock_c2d):
    # Dispatch is a lookup in `_RUNNERS`, and this guard is what turns a miss into the right flag.
    # MeshIn accepts the pairing on purpose (flagging it is the entry's job).
    data = _mesh_in(MeshBackend.UNKNOWN, MeshTopology.CGRD)
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
    # The guard tests membership of `_RUNNERS` rather than equality with UNKNOWN, so a backend added
    # to the enum but never wired in fails safe. Without it the closure gate would find no
    # requirement for the pairing and misreport it as INPUT_CLOSURE_FAIL.
    data = _mesh_in(MeshBackend.GMSH, MeshTopology.CGRD, mesh_config=GMSH_CMeshingConfig())
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

    out = Common_GenerateMesh(_mesh_in(MeshBackend.GMSH, MeshTopology.CGRD))

    assert out.flag == MeshExitFlag.FATAL_ERROR
    assert out.backend == MeshBackend.GMSH
    assert not mock_generator.called  # never reached the builder


@patch("src.datagen.meshing.common.entry.C2D_MeshGenerator")
@patch("src.datagen.meshing.common.entry.C2D_In")
def test_entry_c2d_input_validation_error_becomes_flag(mock_in, mock_generator):
    mock_in.side_effect = _a_validation_error()

    out = Common_GenerateMesh(_mesh_in(MeshBackend.C2D, MeshTopology.OGRD))

    assert out.flag == MeshExitFlag.FATAL_ERROR
    assert out.backend == MeshBackend.C2D
    assert not mock_generator.called


@patch("src.datagen.meshing.common.entry.GMSH_MeshGenerator")
def test_entry_backend_exception_becomes_flag(mock_generator):
    # The backends flag their own internal failures, but their `except` blocks call the exception
    # writers (GMSH_Write_Exception / C2D_Write_Exception), which can raise on their own.
    mock_generator.side_effect = OSError("exception writer blew up on the working dir")

    out = Common_GenerateMesh(_mesh_in(MeshBackend.GMSH, MeshTopology.CGRD))

    assert out.flag == MeshExitFlag.FATAL_ERROR
    assert out.backend == MeshBackend.GMSH


@pytest.mark.parametrize("bad_coords", [
    torch.tensor([1.0, 0.5, 0.0, 0.5, 1.0]),
    torch.tensor(1.0),
    torch.empty((0, 2)),
], ids=["1d", "0d", "empty"])
@patch("src.datagen.meshing.common.entry.GMSH_MeshGenerator")
def test_entry_malformed_tensor_becomes_flag(mock_generator, bad_coords):
    # The real validator, unpatched: it runs ahead of the dispatch `try`, so a raise from it escapes
    data = _mesh_in(MeshBackend.GMSH, MeshTopology.CGRD)
    data.airfoil.coords_tensor = bad_coords

    out = Common_GenerateMesh(data)

    assert out.flag == MeshExitFlag.INPUT_TENSOR_FAIL
    assert not mock_generator.called


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

    out = Common_GenerateMesh(_mesh_in(MeshBackend.GMSH, MeshTopology.CGRD))

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

    out = Common_GenerateMesh(_mesh_in(MeshBackend.GMSH, MeshTopology.CGRD))

    assert out.flag == MeshExitFlag.FATAL_ERROR
    assert mock_quality.called


@patch("src.datagen.meshing.common.entry.Common_evaluate_mesh_geo_dev")
@patch("src.datagen.meshing.common.entry.Common_evaluate_mesh_quality")
@patch("src.datagen.meshing.common.entry.GMSH_MeshGenerator")
def test_entry_geo_dev_exception_becomes_flag(mock_generator, mock_quality, mock_geo_dev):
    # The counterpart to the test above: `Common_evaluate_mesh_geo_dev` has no handler of its own
    # either, and runs second, so it is the one scoring pass that raises with a mesh already on
    # disk and a quality verdict already passed.
    mock_generator.return_value = _mock_backend_out(GMSH_ExitFlag.SUCCESS)
    mock_quality.return_value = (MeshExitFlag.SUCCESS, None, 1234)
    mock_geo_dev.side_effect = ValueError("unreadable MARKER_AIRFOIL block")

    out = Common_GenerateMesh(_mesh_in(MeshBackend.GMSH, MeshTopology.CGRD))

    assert out.flag == MeshExitFlag.FATAL_ERROR
    assert mock_geo_dev.called  # the quality pass did hand over
# endregion


# region Step 6
@patch("src.datagen.meshing.common.entry.GMSH_MeshGenerator")
def test_entry_forwards_the_resolved_config_to_gmsh(mock_generator):
    # Every other dispatcher test meshes with a default config, so a `_run_gmsh` that dropped
    # `mesh_config` and let `GMSH_In`'s resolver default it would pass all of them while silently
    # meshing at the wrong resolution. This is the knobs-actually-arrive test.
    mock_generator.return_value = _mock_backend_out(GMSH_ExitFlag.EXTRUSION_FAIL)

    Common_GenerateMesh(_mesh_in(
        MeshBackend.GMSH, MeshTopology.CGRD,
        mesh_config=GMSH_CMeshingConfig(nx_upper=321, nx_wake=77)))

    gmsh_in = mock_generator.call_args.args[0]
    assert gmsh_in.meshing_config.nx_upper == 321
    assert gmsh_in.meshing_config.nx_wake == 77
    assert gmsh_in.topology == GMSH_Topology.CGRD


@patch("src.datagen.meshing.common.entry.C2D_MeshGenerator")
def test_entry_forwards_the_resolved_config_to_c2d(mock_generator):
    # c2d's half of the above. `topo` is additionally asserted because it is the one field the
    # dispatcher does not pass: `C2D_In`'s validator fills it from `topology` on the way through.
    mock_generator.return_value = _mock_backend_out(C2D_ExitFlag.SUBPROCESS_FAIL)

    Common_GenerateMesh(_mesh_in(
        MeshBackend.C2D, MeshTopology.OGRD,
        mesh_config=C2D_MeshingConfig(nwke=77, jmax=321)))

    c2d_in = mock_generator.call_args.args[0]
    assert c2d_in.meshing_config.nwke == 77
    assert c2d_in.meshing_config.jmax == 321
    assert c2d_in.topology == C2D_Topology.OGRD
    assert c2d_in.meshing_config.topo == "OGRD"


@patch("src.datagen.meshing.common.entry.C2D_MeshGenerator")
def test_entry_one_c2d_config_sweeps_both_topologies(mock_generator):
    # A topology sweep reusing one config: the first run used to pin `topo` onto it, turning the
    # second into a conflict reported as FATAL_ERROR
    mock_generator.return_value = _mock_backend_out(C2D_ExitFlag.SUBPROCESS_FAIL)
    config = C2D_MeshingConfig()

    for topology in (MeshTopology.OGRD, MeshTopology.CGRD):
        out = Common_GenerateMesh(_mesh_in(MeshBackend.C2D, topology, mesh_config=config))

        assert out.flag == MeshExitFlag.SUBPROCESS_FAIL
        assert mock_generator.call_args.args[0].meshing_config.topo == C2D_Topology[topology.name].value
    assert config.topo is None


@pytest.mark.parametrize("backend, topology, generator, flag", [
    (MeshBackend.C2D, MeshTopology.OGRD, "C2D_MeshGenerator", C2D_ExitFlag.SUCCESS),
    (MeshBackend.GMSH, MeshTopology.CGRD, "GMSH_MeshGenerator", GMSH_ExitFlag.SUCCESS),
])
@patch("src.datagen.meshing.common.entry.Common_evaluate_mesh_quality",
       return_value=(MeshExitFlag.CONVERSION_FAIL, None, None))
def test_entry_forwards_the_reporting_options(mock_quality, backend, topology, generator, flag):
    # Both default to the old behaviour, so dropping either would pass every other test
    limits = QualityLimits(skew_max=0.4, ortho_min=0.75, jac_min=0.75)
    with patch(f"src.datagen.meshing.common.entry.{generator}",
               return_value=_mock_backend_out(flag)) as mock_generator:
        Common_GenerateMesh(_mesh_in(backend, topology, count_limits=limits, write_vtk=False))

    assert mock_generator.call_args.args[0].write_vtk is False
    assert mock_quality.call_args.kwargs["count_limits"] == limits
# endregion


# region Step 7
_QUALITY = MeshQualitySummary(ncell=10, max_skew=0.2, min_ortho=0.9, min_jac=0.9, max_ar=10.0,
                              max_size_ratio=1.3, acceptable=False)
_GEO_DEV = MeshGeoDeviationSummary(n_boundary_nodes=5, max_dev_mesh_to_input=1e-5, max_dev_input_to_mesh=1e-5,
                                   rms_dev_mesh_to_input=1e-6, rms_dev_input_to_mesh=1e-6, chord=1.0, acceptable=True)


@pytest.mark.parametrize("quality_flag, geo_flag, expected", [
    (MeshExitFlag.LOW_QUALITY, MeshExitFlag.SUCCESS, MeshExitFlag.LOW_QUALITY),
    (MeshExitFlag.LOW_QUALITY, MeshExitFlag.GEOMETRY_DEVIATION_FAIL, MeshExitFlag.LOW_QUALITY),
    (MeshExitFlag.UNACCEPTABLE_QUALITY, MeshExitFlag.SUCCESS, MeshExitFlag.UNACCEPTABLE_QUALITY),
    (MeshExitFlag.SUCCESS, MeshExitFlag.GEOMETRY_DEVIATION_FAIL, MeshExitFlag.GEOMETRY_DEVIATION_FAIL),
    (MeshExitFlag.SUCCESS, MeshExitFlag.SUCCESS, MeshExitFlag.SUCCESS),
])
@patch("src.datagen.meshing.common.entry.Common_evaluate_mesh_geo_dev")
@patch("src.datagen.meshing.common.entry.Common_evaluate_mesh_quality")
@patch("src.datagen.meshing.common.entry.GMSH_MeshGenerator")
def test_entry_quality_verdict_outranks_deviation(mock_generator, mock_quality, mock_geo_dev,
                                                  quality_flag, geo_flag, expected):
    # Every readable mesh records its deviation; deviation only decides the flag when quality passed
    mock_generator.return_value = _mock_backend_out(GMSH_ExitFlag.SUCCESS)
    mock_quality.return_value = (quality_flag, _QUALITY, 1234)
    mock_geo_dev.return_value = (geo_flag, _GEO_DEV)

    out = Common_GenerateMesh(_mesh_in(MeshBackend.GMSH, MeshTopology.CGRD))

    assert out.flag == expected
    assert out.quality == _QUALITY and out.geo_dev == _GEO_DEV


@patch("src.datagen.meshing.common.entry.Common_evaluate_mesh_geo_dev")
@patch("src.datagen.meshing.common.entry.Common_evaluate_mesh_quality")
@patch("src.datagen.meshing.common.entry.GMSH_MeshGenerator")
def test_entry_deviation_crash_keeps_a_failed_quality_verdict(mock_generator, mock_quality, mock_geo_dev):
    # Unlike on a passing mesh (test_entry_geo_dev_exception_becomes_flag), the verdict is already
    # decided here, so losing the deviation must not lose the quality result with it
    mock_generator.return_value = _mock_backend_out(GMSH_ExitFlag.SUCCESS)
    mock_quality.return_value = (MeshExitFlag.LOW_QUALITY, _QUALITY, 1234)
    mock_geo_dev.side_effect = ValueError("unreadable MARKER_AIRFOIL block")

    out = Common_GenerateMesh(_mesh_in(MeshBackend.GMSH, MeshTopology.CGRD))

    assert out.flag == MeshExitFlag.LOW_QUALITY
    assert out.quality == _QUALITY and out.geo_dev is None
    assert out.num_nodes == 1234
# endregion
