"""
Tests for `scripts.datagen.meshing.master`, the run a YAML config makes, i.e.:

Step 1: Ensure a config's sections become the enabled stages' arguments, in the fixed order, and `folder` the run's
folder
    - test_options_become_arguments
    - test_stages_run_in_the_fixed_order_and_machine_check_is_on_unless_switched_off
    - test_folder_names_the_run_folder
Step 2: Ensure a mistake in the config refuses the run, and the committed example plans
    - test_a_config_mistake_refuses_the_run
    - test_the_example_config_plans

The stages' runs are the engine's tests'; these stop at the plan.
"""

from pathlib import Path

import pytest

from scripts.datagen.geometry import ingest as geometry_ingest
from scripts.datagen.meshing import campaign as mesh_campaign, master as mesh_master, remesh as mesh_remesh
from scripts.datagen.engine import CONFIG_NAME, LOG_NAME, PlanError, Script_plan
from scripts.datagen.meshing.master import Script_load_config, Script_options_to_argv
from src.datagen.geometry.common.paths import Geom_paths

EXAMPLE = Path(mesh_master.__file__).with_name("master.example.yaml")


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
    requested, off, folder = Script_load_config(_config(tmp_path, "remesh:\n  enabled: true\n  all: true\n"
                                                                  "campaign:\n  enabled: true\n"
                                                                  "geometry_ingest:\n  enabled: false\n"))

    assert [(stage.name, argv) for stage, argv in requested] == [("machine_check", []), ("campaign", []),
                                                                 ("remesh", ["--all"])]
    assert off == ["geometry_ingest"] and folder is None
    requested, off, _ = Script_load_config(_config(tmp_path, "machine_check:\n  enabled: false\n"))
    assert requested == [] and off == ["machine_check"]


def test_folder_names_the_run_folder(tmp_path, data_root, capsys):
    elsewhere = (tmp_path / "elsewhere").resolve()

    assert Script_load_config(_config(tmp_path, "folder: experiments/x\n")).folder == data_root / "experiments/x"
    assert Script_load_config(_config(tmp_path, f"folder: {elsewhere.as_posix()}\n")).folder == elsewhere
    config = _config(tmp_path, f"folder: {elsewhere.as_posix()}\nmachine_check:\n  enabled: false\n")
    assert mesh_master.main([str(config)]) == 0
    assert sorted(p.name for p in elsewhere.iterdir()) == [CONFIG_NAME, LOG_NAME]
# endregion


# region Step 2
def test_a_config_mistake_refuses_the_run(tmp_path, data_root, capsys):
    config = _config(tmp_path, "campain:\n  enabled: true\n"
                               "folder: shared\n"
                               "campaign:\n  workers: 2\n"
                               "mesh_calibrate:\n  enabled: true\n  workers: true\n"
                               "remesh:\n  enabled: true\n  al: true\n")

    with pytest.raises(PlanError) as refused:
        Script_load_config(config)
    assert [(stage, text.split(" (")[0]) for stage, text in refused.value.problems] == [
        ("config", "no stage named 'campain'; the stages are machine_check, geometry_calibrate, geometry_ingest, "
                   "mesh_calibrate, campaign, remesh"),
        ("config", "folder is the run's own folder, a path, not 'shared'"),
        ("mesh_calibrate", "workers takes a value, not True"),
        ("campaign", "needs `enabled: true` or `enabled: false`"),
        ("remesh", "no option 'al'")]
    assert mesh_master.main([str(config)]) == 2
    [run] = (data_root / "scratchpad").iterdir()  # a refused run logs into the scratchpad, never a named folder
    log = (run / LOG_NAME).read_text(encoding="utf-8")
    assert "refused, nothing was run: 5 problems" in log and "[remesh]" in log and "no option 'al'" in log
    assert (run / CONFIG_NAME).read_text(encoding="utf-8") == config.read_text(encoding="utf-8")


def test_the_example_config_plans(data_root):
    if not Geom_paths().db.exists():
        pytest.skip(f"no geometry manifest at {Geom_paths().db}")
    requested, off, _ = Script_load_config(EXAMPLE)

    plan = Script_plan("example", requested, off=off)
    assert [sp.stage.name for sp in plan.stages] == ["machine_check", "campaign", "remesh"]
    assert plan.stages[2].reads[0].how == "this run's campaign" and off == ["geometry_calibrate", "geometry_ingest",
                                                                           "mesh_calibrate"]
# endregion
