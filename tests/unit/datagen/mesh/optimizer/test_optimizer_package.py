"""
Tests for `meshing.optimizer`'s front door, i.e.:

Step 1: Ensure every name it offers is its module's own, and nothing heavy loads before it is asked for
    - test_every_export_is_its_modules_own
    - test_an_unknown_name_is_refused
    - test_a_constant_loads_neither_nomad_nor_the_meshers
    - test_a_slot_worker_loads_neither_torch_nor_nomad
"""

import importlib
import subprocess
import sys

import pytest

import src.datagen.meshing.optimizer as optimizer


# region Step 1
def test_every_export_is_its_modules_own():
    assert optimizer.__all__ == list(optimizer._EXPORTS)
    for name, module in optimizer._EXPORTS.items():
        assert getattr(optimizer, name) is getattr(importlib.import_module(f"{optimizer.__name__}.{module}"), name)
    from src.datagen.meshing.optimizer import Optim_run_ladder
    assert Optim_run_ladder is optimizer.search.ladder.Optim_run_ladder


def test_an_unknown_name_is_refused():
    with pytest.raises(AttributeError, match="has no attribute 'run_campaign'"):
        optimizer.run_campaign


def test_a_constant_loads_neither_nomad_nor_the_meshers():
    # A fresh interpreter: this one already holds everything other tests imported
    code = ("import sys; import src.datagen.meshing.optimizer.common.constants; "
            "print(sorted(m for m in ('PyNomad', 'src.datagen.meshing.common.entry', "
            "'src.datagen.meshing.optimizer.search.mads') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


def test_a_slot_worker_loads_neither_torch_nor_nomad():
    # What every slot's worker imports before its task: the runner, a context and a ladder's outcome
    code = ("import sys; import src.datagen.meshing.optimizer.batch.runner, "
            "src.datagen.meshing.optimizer.evaluate.context, src.datagen.meshing.optimizer.search.events; "
            "print(sorted(m for m in ('torch', 'scipy', 'PyNomad', 'src.datagen.meshing.common.entry') "
            "if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"
# endregion
