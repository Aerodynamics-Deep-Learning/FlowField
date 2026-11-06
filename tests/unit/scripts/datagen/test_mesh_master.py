"""
Tests for `scripts.datagen.mesh_master`, the run a YAML config makes, i.e.:

Step 1: Ensure a config's sections become the enabled stages' arguments, in the fixed order
    - test_options_become_arguments
    - test_stages_run_in_the_fixed_order_and_machine_check_is_on_unless_switched_off
Step 2: Ensure a mistake in the config refuses the run, and the committed example plans
    - test_a_config_mistake_refuses_the_run
    - test_the_example_config_plans

The stages' runs are the engine's tests'; these stop at the plan.
"""

from pathlib import Path

import pytest

from scripts.datagen import geometry_ingest, mesh_campaign, mesh_master, mesh_remesh
from scripts.datagen.engine import PlanError, Script_plan
from scripts.datagen.mesh_master import Script_load_config, Script_options_to_argv
from src.datagen.geometry.common.paths import Geom_paths

EXAMPLE = Path(mesh_master.__file__).with_name("mesh_master.example.yaml")


def _config(tmp_path, text: str) -> Path:
    path = tmp_path / "run.yaml"
    path.write_text(text, encoding="utf-8")
    return path


# region Step 1
def test_options_become_arguments(tmp_path, data_root):
    work = tmp_path / "work"

    assert Script_options_to_argv(mesh_campaign.STAGE, {
        "airfoils": ["naca0012", "s1223"], "workers": 2, "settings_label": "trial 3",
        "db": "experiments/x/meshing.sqlite", "geometry_db": "shared", "work_root": str(work)}) == [
        "--airfoils", "naca0012,s1223", "--workers", "2", "--settings-label", "trial 3",
        "--db", str(data_root / "experiments/x/meshing.sqlite"), "--geometry-db", "shared", "--work-root", str(work)]
    assert Script_options_to_argv(geometry_ingest.STAGE, {"folder": "roots", "dry_run": True}) == \
        [str(data_root / "roots"), "--dry-run"]  # a positional first, a flag on
    assert Script_options_to_argv(mesh_remesh.STAGE, {"all": True, "vtk": False, "db": None}) == ["--all"]


def test_stages_run_in_the_fixed_order_and_machine_check_is_on_unless_switched_off(tmp_path):
    requested, off = Script_load_config(_config(tmp_path, "remesh:\n  enabled: true\n  all: true\n"
                                                          "campaign:\n  enabled: true\n"
                                                          "geometry_ingest:\n  enabled: false\n"))

    assert [(stage.name, argv) for stage, argv in requested] == [("machine_check", []), ("campaign", []),
                                                                 ("remesh", ["--all"])]
    assert off == ["geometry_ingest"]
    requested, off = Script_load_config(_config(tmp_path, "machine_check:\n  enabled: false\n"))
    assert requested == [] and off == ["machine_check"]
# endregion


# region Step 2
def test_a_config_mistake_refuses_the_run(tmp_path, data_root, capsys):
    config = _config(tmp_path, "campain:\n  enabled: true\n"
                               "campaign:\n  workers: 2\n"
                               "mesh_calibrate:\n  enabled: true\n  workers: true\n"
                               "remesh:\n  enabled: true\n  al: true\n")

    with pytest.raises(PlanError) as refused:
        Script_load_config(config)
    assert [(stage, text.split(" (")[0]) for stage, text in refused.value.problems] == [
        ("config", "no stage named 'campain'; the stages are machine_check, geometry_calibrate, geometry_ingest, "
                   "mesh_calibrate, campaign, remesh"),
        ("mesh_calibrate", "workers takes a value, not True"),
        ("campaign", "needs `enabled: true` or `enabled: false`"),
        ("remesh", "no option 'al'")]
    assert mesh_master.main([str(config)]) == 2
    [run] = (data_root / "scratchpad").iterdir()
    log = next(run.glob("*_run.log")).read_text(encoding="utf-8")
    assert "refused, nothing was run: 4 problems" in log and "[remesh]" in log and "no option 'al'" in log


def test_the_example_config_plans(data_root):
    if not Geom_paths().db.exists():
        pytest.skip(f"no geometry manifest at {Geom_paths().db}")
    requested, off = Script_load_config(EXAMPLE)

    plan = Script_plan("example", requested, off=off)
    assert [sp.stage.name for sp in plan.stages] == ["machine_check", "campaign", "remesh"]
    assert plan.stages[2].reads[0].how == "this run's campaign" and off == ["geometry_calibrate", "geometry_ingest",
                                                                           "mesh_calibrate"]
# endregion
