"""
Tests for `scripts.datagen.mesh_campaign`, the campaign's command line, i.e.:

Step 1: Ensure the options reach the campaign, its summary is printed and how it ended is the exit code
    - test_the_options_reach_the_campaign
    - test_how_the_campaign_ended_is_the_exit_code
Step 2: Ensure it takes no setting, and a mistake in what was asked stops it before anything is recorded
    - test_the_command_line_takes_no_setting
    - test_a_mistake_is_a_usage_error

The campaign and the manifest are stand-ins; the campaign is tested in the optimizer tiers.
"""

import pytest

from scripts.datagen import mesh_campaign
from src.datagen.meshing.optimizer.batch.campaign import CampaignResult
from src.datagen.meshing.optimizer.batch.runner import BatchStopped
from src.datagen.meshing.optimizer.common.constants import WORKERS


def _known(names, db=None):
    unknown = [n for n in names if n not in ("naca0012", "naca4412")]
    if unknown:
        raise ValueError(f"no shape under the current shape version for {', '.join(unknown)}")
    return [(n, None) for n in names]


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
def test_the_options_reach_the_campaign(campaign, tmp_path, capsys):
    calls, _ = campaign
    db, manifest, work = tmp_path / "meshing.sqlite", tmp_path / "geometry.sqlite", tmp_path / "work"

    assert mesh_campaign.main(["--db", str(db), "--airfoils", "naca0012,naca4412", "--workers", "2",
                               "--settings-label", "trial 3", "--geometry-db", str(manifest),
                               "--work-root", str(work)]) == 0
    assert mesh_campaign.main(["--db", str(db)]) == 0
    assert calls == [(db, ["naca0012", "naca4412"], 2, work, "trial 3", {"geometry_db": manifest}),
                     (db, None, WORKERS, None, None, {"geometry_db": None})]
    assert "the summary" in capsys.readouterr().out


@pytest.mark.parametrize("stopped, code", [(None, 0), (BatchStopped("3 airfoils aborted in a row", []), 1),
                                           (KeyboardInterrupt(), 130)], ids=["ran", "stopped", "ctrl-c"])
def test_how_the_campaign_ended_is_the_exit_code(campaign, tmp_path, stopped, code):
    campaign[1]["stopped"] = stopped

    assert mesh_campaign.main(["--db", str(tmp_path / "meshing.sqlite")]) == code
# endregion


# region Step 2
@pytest.mark.parametrize("setting", ["--golden-jmax", "--rung3-budget", "--gate-skew-max"])
def test_the_command_line_takes_no_setting(campaign, tmp_path, capsys, setting):
    with pytest.raises(SystemExit) as stopped:
        mesh_campaign.main(["--db", str(tmp_path / "meshing.sqlite"), setting, "150"])

    assert stopped.value.code == 2 and "unrecognized arguments" in capsys.readouterr().err


@pytest.mark.parametrize("args, says", [(["--airfoils", "naca9999"], "naca9999"), ([], "required: --db")],
                         ids=["unknown airfoil", "no db"])
def test_a_mistake_is_a_usage_error(campaign, tmp_path, capsys, args, says):
    db = tmp_path / "meshing.sqlite"
    with pytest.raises(SystemExit) as stopped:
        mesh_campaign.main(["--db", str(db), *args] if args else args)

    assert stopped.value.code == 2 and says in capsys.readouterr().err
    assert campaign[0] == [] and not db.exists()  # the store not even made
# endregion
