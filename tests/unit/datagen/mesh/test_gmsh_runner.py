"""
Tests for `GMSH_MeshGenerator`, i.e.:

Step 1: Ensure gmsh-specific structural failure modes route to the expected GMSH_ExitFlag
    (mesh-quality acceptability is decided centrally by common.entry.Common_evaluate_mesh_quality,
    not here -- see test_common_entry.py / test_cross_backend.py)
    - test_executable_not_found_routing
    - test_runner_fatal_error
    - test_clean_exit_without_a_mesh_is_fatal
    - test_runner_extrusion_fail
    - test_runner_success
Step 2: Ensure a gmsh failure is caught from its output, since its exit code alone is not reliable
    - test_run_script_raises_on_gmsh_failure
    - test_run_script_returns_the_log_on_a_clean_run

The c2d counterpart is test_c2d_runner.py. The builders run for real here, as they only write script
text; the gmsh process is stubbed, and runs for real in the integration tier.
"""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from src.datagen.schemas import Airfoil, Freestream
from src.datagen.meshing.gmsh.geo import GeoScript
from src.datagen.meshing.gmsh.schemas import GMSH_In, GMSH_ExitFlag, GMSH_CMeshingConfig, GMSH_Topology
from src.datagen.meshing.gmsh.run import GMSH_MeshGenerator, GMSH_run_script

RUNNER = "src.datagen.meshing.gmsh.run"

_NACA0012 = [
    [1.0, 0.00126], [0.9, 0.01055], [0.8, 0.01816], [0.7, 0.02412], [0.6, 0.02824],
    [0.5, 0.03038], [0.4, 0.03039], [0.3, 0.02797], [0.2, 0.02285], [0.1, 0.01448],
    [0.05, 0.00908], [0.0125, 0.00443], [0.0, 0.0],
    [0.0125, -0.00443], [0.05, -0.00908], [0.1, -0.01448], [0.2, -0.02285],
    [0.3, -0.02797], [0.4, -0.03039], [0.5, -0.03038], [0.6, -0.02824],
    [0.7, -0.02412], [0.8, -0.01816], [0.9, -0.01055], [1.0, -0.00126],
]


def _gmsh_in(working_dir: str) -> GMSH_In:
    coords = torch.tensor(_NACA0012, dtype=torch.float32)
    return GMSH_In(
        airfoil=Airfoil(airfoil_name="naca_test", coords_tensor=coords, chord=1.0, le_idx=12),
        freestream=Freestream(alpha=0.0, Re=2e6, mach=0.3), topology=GMSH_Topology.CGRD,
        meshing_config=GMSH_CMeshingConfig(upper_anchor_idx=6, lower_anchor_idx=18), working_dir=working_dir,
    )


def _writes_su2(body: str):
    """A `GMSH_run_script` stand-in that writes `body` where the runner expects the mesh."""
    def run(exe, geo, geo_path, working_dir, timeout):
        Path(geo_path).with_suffix(".su2").write_text(body)
        return "log"
    return run


def _run(working_dir, **run_script):
    with patch(f"{RUNNER}.find_tool", return_value="/fake/gmsh"), \
         patch(f"{RUNNER}.GMSH_run_script", **run_script):
        return GMSH_MeshGenerator(_gmsh_in(working_dir))


# region Step 1
def test_executable_not_found_routing(tmp_path):
    with patch(f"{RUNNER}.find_tool", return_value=None):
        out = GMSH_MeshGenerator(_gmsh_in(str(tmp_path)))
    assert out.flag == GMSH_ExitFlag.EXECUTABLE_NOT_FOUND
    assert list(tmp_path.iterdir()) == []


def test_runner_fatal_error(tmp_path):
    out = _run(str(tmp_path), side_effect=RuntimeError("gmsh reported a failure"))
    assert out.flag == GMSH_ExitFlag.FATAL_ERROR
    assert "gmsh reported a failure" in Path(out.verbose_list[2]).read_text()


def test_clean_exit_without_a_mesh_is_fatal(tmp_path):
    out = _run(str(tmp_path), return_value="log")
    assert out.flag == GMSH_ExitFlag.FATAL_ERROR
    assert "wrote no mesh" in Path(out.verbose_list[2]).read_text()


def test_runner_extrusion_fail(tmp_path):
    # No elements at all, which used to raise a NameError rather than flag
    out = _run(str(tmp_path), side_effect=_writes_su2("NDIME= 2\nNELEM= 0\nNPOIN= 0\nNMARK= 0\n"))
    assert out.flag == GMSH_ExitFlag.EXTRUSION_FAIL


def test_runner_success(tmp_path):
    su2 = "NDIME= 2\nNELEM= 1\n9 0 1 2 3 0\nNPOIN= 4\n0 0 0\n1 0 1\n1 1 2\n0 1 3\nNMARK= 0\n"
    out = _run(str(tmp_path), side_effect=_writes_su2(su2))
    assert out.flag == GMSH_ExitFlag.SUCCESS
    assert out.num_nodes == 4
    # Full paths, not bare filenames (a regression the gmsh audit found)
    assert out.mesh_path_vtk == out.mesh_path.replace("_mesh.su2", "_mesh.vtk")
    assert out.mesh_path_vtk.startswith(str(tmp_path))
    assert Path(out.verbose_list[0]).read_text() == "log"
# endregion


# region Step 2
@pytest.mark.parametrize("result, message", [
    (subprocess.CompletedProcess([], 0, "Info    : Reading\nError   : Unknown curve 7\n", ""), "reported a failure"),
    (subprocess.CompletedProcess([], 1, "", ""), "reported a failure"),
    (subprocess.TimeoutExpired("gmsh", 5), "timed out"),
    (FileNotFoundError("no gmsh here"), "failed to launch"),
])
def test_run_script_raises_on_gmsh_failure(tmp_path, result, message):
    stub = {"side_effect": result} if isinstance(result, BaseException) else {"return_value": result}
    with patch(f"{RUNNER}.subprocess.run", **stub), pytest.raises(RuntimeError, match=message):
        GMSH_run_script(exe="gmsh", geo=GeoScript(), geo_path=str(tmp_path / "m.geo"),
                        working_dir=str(tmp_path), timeout=5)


def test_run_script_returns_the_log_on_a_clean_run(tmp_path):
    geo = GeoScript()
    geo.addPoint(0.0, 0.0, 0.0)
    done = subprocess.CompletedProcess([], 0, "Info    : Done\nWarning : a warning is not a failure\n", "")
    with patch(f"{RUNNER}.subprocess.run", return_value=done):
        log = GMSH_run_script(exe="gmsh", geo=geo, geo_path=str(tmp_path / "m.geo"),
                              working_dir=str(tmp_path), timeout=5)
    assert "a warning is not a failure" in log
    assert (tmp_path / "m.geo").read_text() == geo.text()
# endregion
