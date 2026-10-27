"""
Tests for `C2D_MeshGenerator`, i.e.:

Step 1: Ensure c2d-specific structural failure modes route to the expected C2D_ExitFlag
    (mesh-quality acceptability is decided centrally by common.entry._evaluate_mesh_quality, not
    here (see test_common_entry.py / test_cross_backend.py)
    - test_executable_not_found_routing
    - test_subprocess_fail_routing
    - test_conversion_fail_routing
    - test_success_routing
Step 2: Ensure C2D_MeshingConfig rejects an out-of-range smoothing knob
    - test_c2d_config_rejects_out_of_bounds_smoothing_knob
Step 3: Ensure C2D_count_su2_nodes reads the cheap node count the runner flags on
    - test_count_su2_nodes_reads_npoin_header
    - test_count_su2_nodes_returns_zero_when_unavailable
    - test_count_su2_nodes_returns_zero_for_missing_path
Step 4: Ensure the exe is driven in BUFF mode, answering its topology prompt only when it asks
    - test_c2d_driven_in_buff_mode
Step 5: Ensure c2d sizes the first cell from the freestream's Re, not a config value
    - test_first_cell_sized_from_the_freestream_re

Scope: the c2d runner only. The gmsh counterpart is test_gmsh_runner.py, and the dispatcher that
maps a C2D_ExitFlag onto a MeshExitFlag is test_common_entry.py.
"""

import tempfile
from unittest.mock import patch, MagicMock

import torch
import pytest
from pydantic import ValidationError

from src.datagen.schemas import Airfoil, Freestream
from src.datagen.meshing.c2d.schemas import C2D_In, C2D_MeshingConfig, C2D_ExitFlag, C2D_Topology
from src.datagen.meshing.c2d.run import C2D_MeshGenerator, C2D_generate_mesh
from src.datagen.meshing.c2d.utils import C2D_count_su2_nodes


def _airfoil():
    coords = torch.tensor([
        [1.0, 0.001], [0.5, 0.05], [0.0, 0.0], [0.5, -0.05], [1.0, -0.001],
    ])
    return Airfoil(airfoil_name="naca_test", coords_tensor=coords, chord=1.0, le_idx=2)


def _closed_airfoil():
    coords = torch.tensor([
        [1.0, 0.0], [0.5, 0.05], [0.0, 0.0], [0.5, -0.05], [1.0, 0.0],
    ])
    return Airfoil(airfoil_name="naca_closed", coords_tensor=coords, chord=1.0, le_idx=2)


def _freestream():
    return Freestream(alpha=0.0, Re=2e6, mach=0.3)


def _c2d_in(working_dir: str) -> C2D_In:
    return C2D_In(
        airfoil=_airfoil(), freestream=_freestream(), topology=C2D_Topology.OGRD,
        meshing_config=C2D_MeshingConfig(), working_dir=working_dir,
    )


RUNNER = "src.datagen.meshing.c2d.run"


# region Step 1
def test_executable_not_found_routing():
    with tempfile.TemporaryDirectory() as wd, patch(f"{RUNNER}.C2D_find_exe", return_value=None):
        out = C2D_MeshGenerator(_c2d_in(wd))
        assert out.flag == C2D_ExitFlag.EXECUTABLE_NOT_FOUND


def test_subprocess_fail_routing():
    with tempfile.TemporaryDirectory() as wd, \
         patch(f"{RUNNER}.C2D_find_exe", return_value="/fake/c2d"), \
         patch(f"{RUNNER}.C2D_generate_mesh", return_value=(None, None, None, None, "log")):
        out = C2D_MeshGenerator(_c2d_in(wd))
        assert out.flag == C2D_ExitFlag.SUBPROCESS_FAIL


def test_conversion_fail_routing():
    with tempfile.TemporaryDirectory() as wd, \
         patch(f"{RUNNER}.C2D_find_exe", return_value="/fake/c2d"), \
         patch(f"{RUNNER}.C2D_generate_mesh",
               return_value=("/tmp/mock.p3d", None, None, None, "log")):
        out = C2D_MeshGenerator(_c2d_in(wd))
        assert out.flag == C2D_ExitFlag.CONVERSION_FAIL


def test_success_routing(tmp_path):
    su2_path = tmp_path / "mock.su2"
    # Needs at least one node: C2D_MeshGenerator flags a nnode==0 `.su2` as CONVERSION_FAIL,
    # mirroring GMSH's num_quads==0 structural check.
    su2_path.write_text("NPOIN= 3\n0.0 0.0 0\n1.0 0.0 1\n0.5 1.0 2\n")
    vtk_path = tmp_path / "mock.vtk"
    with patch(f"{RUNNER}.C2D_find_exe", return_value="/fake/c2d"), \
         patch(f"{RUNNER}.C2D_generate_mesh", return_value=(
             "/tmp/mock.p3d", str(su2_path), str(vtk_path), "/tmp/mock.nmf", "log")):
        out = C2D_MeshGenerator(_c2d_in(str(tmp_path)))
        assert out.flag == C2D_ExitFlag.SUCCESS
        assert out.mesh_path == str(su2_path)
        assert out.mesh_path_vtk == str(vtk_path)
# endregion


# region Step 2
@pytest.mark.parametrize("field, value", [("asmt", 500), ("epsi", 1000.0), ("funi", -1.0), ("alfa", 10.0)])
def test_c2d_config_rejects_out_of_bounds_smoothing_knob(field, value):
    with pytest.raises(ValidationError):
        C2D_MeshingConfig(**{field: value})
# endregion


# region Step 3
def test_count_su2_nodes_reads_npoin_header(tmp_path):
    # Deliberately header-only, no node rows: the count must come from the NPOIN= declaration
    # without parsing the body, which is the whole point of not using _analyze_su2 here.
    su2 = tmp_path / "header_only.su2"
    su2.write_text("%\n% SU2 mesh\n%\nNDIME= 2\nNELEM= 0\nNPOIN= 59800\n")

    assert C2D_count_su2_nodes(str(su2)) == 59800


@pytest.mark.parametrize("content", [
    "NDIME= 2\nNELEM= 0\n",     # no NPOIN= at all
    "",                          # empty file
    "NPOIN= notanumber\n",       # malformed count
])
def test_count_su2_nodes_returns_zero_when_unavailable(tmp_path, content):
    su2 = tmp_path / "bad.su2"
    su2.write_text(content)

    assert C2D_count_su2_nodes(str(su2)) == 0


def test_count_su2_nodes_returns_zero_for_missing_path(tmp_path):
    assert C2D_count_su2_nodes(None) == 0
    assert C2D_count_su2_nodes(str(tmp_path / "does_not_exist.su2")) == 0
# endregion


# region Step 4
@pytest.mark.parametrize("airfoil, topo, confirms", [
    # c2d recommends CGRD for any closed contour and prompts y/n when `topo` disagrees; an unanswered
    # prompt would eat the next command
    (_closed_airfoil(), "OGRD", True),
    (_closed_airfoil(), "CGRD", False),
    (_airfoil(), "OGRD", False),  # open, so c2d recommends OGRD (and BUFF then refuses the contour)
])
def test_c2d_driven_in_buff_mode(tmp_path, airfoil, topo, confirms):
    with patch(f"{RUNNER}.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        C2D_generate_mesh(airfoil, "/fake/c2d", str(tmp_path), C2D_MeshingConfig(topo=topo), Re=2e6)

    stdin = mock_run.call_args.kwargs["input"].splitlines()
    assert stdin == [f"{airfoil.airfoil_name}.dat"] + (["y"] if confirms else []) + ["grid", "buff", "quit"]
    options = (tmp_path / "grid_options.in").read_text()
    assert f"topo = '{topo}'" in options
    assert not any(k in options for k in ("nsrf", "lesp", "tesp"))  # BUFF takes the surface from the input
# endregion


# region Step 5
def test_first_cell_sized_from_the_freestream_re(tmp_path):
    # The generator hands the freestream's Re down; no c2d exe runs, the options file is written first
    with patch(f"{RUNNER}.C2D_find_exe", return_value="/fake/c2d"), \
         patch(f"{RUNNER}.subprocess.run", return_value=MagicMock(returncode=0, stdout="", stderr="")):
        C2D_MeshGenerator(_c2d_in(str(tmp_path)))

    options = (tmp_path / "grid_options.in").read_text().splitlines()
    assert [line.strip() for line in options if line.strip().startswith("recd")] == [f"recd = {_freestream().Re!r}"]
    assert "recd" not in C2D_MeshingConfig.model_fields
# endregion
