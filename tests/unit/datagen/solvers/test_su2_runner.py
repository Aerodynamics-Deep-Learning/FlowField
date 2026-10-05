"""
Tests for `SU2_Runner`, i.e.:

Step 1: Ensure a missing SU2_CFD is a flag returned before anything is written, never raised
    - test_executable_not_found_routing

Scope: the not-found path only. The lookup itself is test_tools.py.
"""

import os
from unittest.mock import patch

import torch

from src.datagen.schemas import Airfoil, Freestream
from src.datagen.solvers.su2.schemas import SU2_In, SU2_SolverConfig, SU2_ConvergenceFlag
from src.datagen.solvers.su2.run import SU2_Runner

RUNNER = "src.datagen.solvers.su2.run"


def _su2_in(working_dir: str) -> SU2_In:
    coords = torch.tensor([
        [1.0, 0.001], [0.5, 0.05], [0.0, 0.0], [0.5, -0.05], [1.0, -0.001],
    ])
    return SU2_In(
        airfoil=Airfoil(airfoil_name="naca_test", coords_tensor=coords, chord=1.0, le_idx=2),
        freestream=Freestream(alpha=0.0, Re=2e6, mach=0.3),
        manifest_path=os.path.join(working_dir, "manifest.db"),
        working_dir=working_dir, solver_cfg=SU2_SolverConfig(), sim_id=0,
    )


# region Step 1
def test_executable_not_found_routing(tmp_path):
    with patch(f"{RUNNER}.find_tool", return_value=None):
        out = SU2_Runner(_su2_in(str(tmp_path)))
    assert out.convergence == SU2_ConvergenceFlag.EXECUTABLE_NOT_FOUND
    assert out.config_path_list == []
    assert list(tmp_path.iterdir()) == []
# endregion
