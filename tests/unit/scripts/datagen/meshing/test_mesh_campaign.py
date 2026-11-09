"""
Tests for `scripts.datagen.meshing.campaign`, the campaign's command line, i.e.:

Step 1: Ensure the options reach the campaign, its store the scratchpad's when left out, its summary printed and
how it ended the exit code
    - test_the_options_reach_the_campaign
    - test_how_the_campaign_ended_is_the_exit_code
Step 2: Ensure settings reach it through `--set` alone, each one it reads, and a mistake in what was asked stops it
before anything is recorded
    - test_settings_reach_the_campaign
    - test_a_setting_it_cannot_take_is_a_usage_error
    - test_an_unknown_airfoil_is_a_usage_error

The campaign and the manifest are stand-ins; the campaign is tested in the optimizer tiers.
"""

import pytest

from scripts.datagen.meshing import campaign as mesh_campaign
from src.datagen.constants import SLOT_MEMORY_CAP_MB, WATCHDOG_S, WORKERS
from src.datagen.geometry.common.constants import SHAPE_VERSION
from src.datagen.geometry.common.paths import Geom_paths
from src.datagen.meshing.optimizer.batch.campaign import CampaignResult
from src.datagen.meshing.optimizer.batch.runner import BatchStopped


def _known(names, db=None, shape_version=SHAPE_VERSION):
    if shape_version != SHAPE_VERSION:
        raise ValueError(f"{db} holds no shapes under {shape_version}")
    unknown = [n for n in names or () if n not in ("naca0012", "naca4412")]
    if unknown:
        raise ValueError(f"no shape under {shape_version} for {', '.join(unknown)}")
    return [(n, None) for n in names or ("naca0012", "naca4412")]


@pytest.fixture
def campaign(monkeypatch):
    calls, ending = [], {"stopped": None}

    def run(store, *args, **kw):
        calls.append((store.path, *args, kw))
        return CampaignResult("batch", "settings", [], ["naca0012"], 1.0, ending["stopped"])

    monkeypatch.setattr(mesh_campaign, "Optim_run_campaign", run)
    monkeypatch.setattr(mesh_campaign, "Optim_campaign_summary", lambda store, result: "the summary")
    monkeypatch.setattr(mesh_campaign, "Optim_load_shapes", _known)
    return calls, ending


# region Step 1
def test_the_options_reach_the_campaign(campaign, tmp_path, data_root, capsys):
    calls, _ = campaign
    here = tmp_path.resolve()
    db, manifest, work = here / "meshing.sqlite", here / "geometry.sqlite", tmp_path / "work"
    manifest.touch()

    assert mesh_campaign.main(["--db", str(db), "--airfoils", "naca0012,naca4412", "--workers", "2",
                               "--settings-label", "trial 3", "--geometry-db", str(manifest),
                               "--work-root", str(work)]) == 0
    assert mesh_campaign.main([]) == 0
    [scratch] = [p.parent for p in (data_root / "scratchpad").glob("*/campaign")]  # the run that left it out
    constants = {"overrides": {}, "watchdog_s": WATCHDOG_S, "cap_mb": SLOT_MEMORY_CAP_MB}
    assert calls == [(db, ["naca0012", "naca4412"], 2, work, "trial 3", {"geometry_db": manifest, **constants}),
                     (scratch / "campaign" / "meshing.sqlite", None, WORKERS, None, None,
                      {"geometry_db": Geom_paths().db, **constants})]
    assert "the summary" in capsys.readouterr().out


@pytest.mark.parametrize("stopped, code", [(None, 0), (BatchStopped("3 airfoils aborted in a row", []), 1),
                                           (KeyboardInterrupt(), 130)], ids=["ran", "stopped", "ctrl-c"])
def test_how_the_campaign_ended_is_the_exit_code(campaign, tmp_path, stopped, code):
    campaign[1]["stopped"] = stopped

    assert mesh_campaign.main(["--db", str(tmp_path / "meshing.sqlite")]) == code
# endregion


# region Step 2
def test_settings_reach_the_campaign(campaign, tmp_path, capsys):
    assert mesh_campaign.main(["--db", str(tmp_path / "meshing.sqlite"), "--set", "gate_skew_max=0.22",
                               "--set", "RUNG2_BUDGET=60", "--set", "watchdog_s=900", "--workers", "3"]) == 0

    [(_, _, workers, _, _, kw)] = campaign[0]
    assert workers == 3 and kw["overrides"] == {"gate_skew_max": 0.22, "rung2_budget": 60}  # typed as the constants
    assert (kw["watchdog_s"], kw["cap_mb"]) == (900, SLOT_MEMORY_CAP_MB)
    printed = capsys.readouterr().out
    assert "settings: gate_skew_max 0.22 (given; constants.py 0.25)" in printed  # the plan shows what moved
    assert "settings: workers 3 (given; constants.py 6)" in printed


@pytest.mark.parametrize("args, says", [
    (["--golden-jmax", "150"], "unrecognized arguments"),
    (["--set", "cal_set=a,b"], "campaign does not read cal_set"),
    (["--set", "cst_order=10"], "holds no shapes under cst10-r1"),
    (["--set", "cst_nose=pointy"], "cst_nose is one of separate, shared, lem, not 'pointy'"),
    (["--set", "nope=1"], "no setting 'nope'"),
    (["--set", "rung2_budget=5.5"], "rung2_budget takes an int, not '5.5'"),
    (["--set", "gate_skew_max"], "a setting is key=value"),
    (["--set", "gate_skew_max=0.2", "--set", "gate_skew_max=0.3"], "gate_skew_max is set twice"),
    (["--workers", "4", "--set", "workers=5"], "workers is given twice: --workers 4 and --set workers=5"),
], ids=["not an option", "not read here", "no shapes at that order", "no such nose", "unknown", "wrong type",
        "no value", "twice", "flag too"])
def test_a_setting_it_cannot_take_is_a_usage_error(campaign, tmp_path, capsys, args, says):
    assert mesh_campaign.main(["--db", str(tmp_path / "meshing.sqlite"), *args]) == 2

    assert says in capsys.readouterr().out and campaign[0] == []


def test_an_unknown_airfoil_is_a_usage_error(campaign, tmp_path, capsys):
    db = tmp_path / "meshing.sqlite"

    assert mesh_campaign.main(["--db", str(db), "--airfoils", "naca9999"]) == 2

    assert "naca9999" in capsys.readouterr().out
    assert campaign[0] == [] and not db.exists()  # the store not even made
# endregion
