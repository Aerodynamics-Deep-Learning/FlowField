"""
Tests for `XFoil_Runner`, i.e.:

Step 1: Ensure a missing XFoil is a flag at run time, never raised, and never checked at import
    - test_package_imports_without_xfoil_on_path
    - test_executable_not_found_routing

Scope: the not-found path only. The lookup itself is test_tools.py.
"""

import importlib
from unittest.mock import patch

import torch

from src.datagen.schemas import Airfoil, Freestream
from src.datagen.solvers.xfoil.schemas import (
    XFoil_WarmStartIn, XFoil_SolverConfig, XFoil_ConvergenceConfig, XFoil_ConvergenceFlag,
)
from src.datagen.solvers.xfoil.run import XFoil_Runner

RUNNER = "src.datagen.solvers.xfoil.run"


def _xfoil_in(working_dir: str) -> XFoil_WarmStartIn:
    coords = torch.tensor([
        [1.0, 0.001], [0.5, 0.05], [0.0, 0.0], [0.5, -0.05], [1.0, -0.001],
    ])
    return XFoil_WarmStartIn(
        airfoil=Airfoil(airfoil_name="naca_test", coords_tensor=coords, chord=1.0, le_idx=2),
        freestream=Freestream(alpha=0.0, Re=2e6, mach=0.3),
        working_dir=working_dir,
        solver_config=XFoil_SolverConfig(), conv_config=XFoil_ConvergenceConfig(),
    )


# region Step 1
def test_package_imports_without_xfoil_on_path(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    importlib.reload(importlib.import_module("src.datagen.solvers.xfoil"))


def test_executable_not_found_routing(tmp_path):
    xfoil_in = _xfoil_in(str(tmp_path))
    with patch(f"{RUNNER}.find_tool", return_value=None):
        out = XFoil_Runner(xfoil_in)
    assert out.flag == XFoil_ConvergenceFlag.EXECUTABLE_NOT_FOUND
    assert out.Cp_tensor is None and out.conservative_tensor is None
    assert xfoil_in.airfoil.coords_path is None  # Geometry was not written
    assert list(tmp_path.iterdir()) == []
# endregion
