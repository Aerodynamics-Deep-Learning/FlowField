"""
Tests for `meshing.optimizer.batch.campaign`, the ladder over every airfoil, i.e.:

Step 1: Ensure a campaign runs its airfoils under the constants' own settings row, and takes no setting
    - test_a_campaign_runs_its_airfoils_under_the_constants
    - test_the_command_line_takes_no_setting
Step 2: Ensure a campaign resumes: the finished are skipped, the aborted and the ones left running run again
    - test_a_rerun_skips_the_finished_and_reruns_the_aborted
    - test_a_ladder_left_running_is_aborted_at_the_start
    - test_nothing_left_to_run
Step 3: Ensure the summary reads the batch back from the store
    - test_the_summary_counts_the_batch
    - test_a_stopped_campaign_says_so

Runs here with one worker and a stand-in ladder over the repo's geometry manifest; the runner and the store are
real. The stand-in passes at its golden run unless its airfoil is named in `_ABORTING` or `_SETUP`.
"""

import uuid

import pytest

from src.datagen.geometry.common.paths import Geom_paths
from src.datagen.meshing.optimizer.batch.campaign import Optim_campaign_summary, Optim_run_campaign, main
from src.datagen.meshing.optimizer.batch.runner import BatchStopped
from src.datagen.meshing.optimizer.common.settings import Optim_build_settings
from src.datagen.meshing.optimizer.evaluate.context import StudySetupError
from src.datagen.meshing.optimizer.manifest.store import Optim_utc_now, RunStore
from src.datagen.meshing.optimizer.search.events import LadderOutcome

_NODES = {"naca0012": 58_000, "naca4412": 61_000, "e376": 66_000}
_ABORTING: set = set()
_ABORTING_ONCE: set = set()
_SETUP: set = set()


def _ladder(name, shape, ctx, emit):
    s, ladder_id = ctx.settings, str(uuid.uuid4())
    shared = dict(settings_id=s["settings_id"], batch_id=ctx.batch_id, geometry_id=shape.geometry_id, airfoil_name=name)
    emit(("ladder_start", dict(ladder_id=ladder_id, status="running", started_at=Optim_utc_now(), **shared)))
    once = name in _ABORTING_ONCE
    _ABORTING_ONCE.discard(name)
    if once or name in _ABORTING or name in _SETUP:
        error = StudySetupError("EXECUTABLE_NOT_FOUND") if name in _SETUP else RuntimeError("stand-in failure")
        emit(("ladder_end", ladder_id, dict(status="aborted", error_text=f"{type(error).__name__}: {error}",
                                            finished_at=Optim_utc_now())))
        return LadderOutcome(ladder_id, "aborted", ["golden"], None, error)
    run = dict(run_id=str(uuid.uuid4()), ladder_id=ladder_id, rung="golden", seq=0, status="meshed", passed=1,
               num_nodes=_NODES[name], config_schema_fp="fp", entry_flag="SUCCESS", started_at=Optim_utc_now(), **shared)
    emit(("run", run))
    emit(("ladder_end", ladder_id, dict(status="passed", final_rung="golden", route=["golden"], meshes_golden=1,
                                        finished_at=Optim_utc_now())))
    return LadderOutcome(ladder_id, "passed", ["golden"], run, None)


@pytest.fixture(autouse=True)
def stand_ins(tmp_path, monkeypatch):
    if not Geom_paths().db.exists():
        pytest.skip(f"no geometry manifest at {Geom_paths().db}")
    exe = tmp_path / "c2d.exe"
    exe.write_bytes(b"not really c2d")
    monkeypatch.setattr("src.datagen.meshing.optimizer.common.settings.find_tool", lambda key: str(exe))
    _ABORTING.clear()
    _ABORTING_ONCE.clear()
    _SETUP.clear()


@pytest.fixture
def store(tmp_path):
    with RunStore.open(tmp_path / "meshing.sqlite") as store:
        yield store


def _campaign(store, *names):
    return Optim_run_campaign(store, list(names), workers=1, task=_ladder)


def _statuses(store, name):
    return [r[0] for r in store._conn.execute("SELECT status FROM ladders WHERE airfoil_name = ? ORDER BY rowid",
                                              (name,))]


# region Step 1
def test_a_campaign_runs_its_airfoils_under_the_constants(store):
    result = _campaign(store, "naca0012", "naca4412")
    constants = Optim_build_settings()

    assert result.settings_id == constants["settings_id"] and result.stopped is None
    assert [r[0] for r in store._conn.execute("SELECT settings_id FROM settings")] == [constants["settings_id"]]
    assert (result.names, result.skipped) == (["naca0012", "naca4412"], [])
    for ladder in store.ladders(result.batch_id):
        best = store.best(ladder["geometry_id"], result.settings_id)
        assert ladder["status"] == "passed" and store.run(best)["num_nodes"] == _NODES[ladder["airfoil_name"]]


@pytest.mark.parametrize("setting", ["--golden-jmax", "--rung3-budget", "--gate-skew-max"])
def test_the_command_line_takes_no_setting(setting, capsys):
    with pytest.raises(SystemExit) as stopped:
        main([setting, "150"])

    assert stopped.value.code == 2 and "unrecognized arguments" in capsys.readouterr().err
# endregion


# region Step 2
def test_a_rerun_skips_the_finished_and_reruns_the_aborted(store):
    _ABORTING.add("naca4412")
    first = _campaign(store, "naca0012", "naca4412", "e376")
    _ABORTING.clear()
    second = _campaign(store, "naca0012", "naca4412", "e376")

    assert _statuses(store, "naca4412") == ["aborted", "aborted", "passed"]  # retried once in the first
    assert (second.names, second.skipped) == (["naca4412"], ["naca0012", "e376"])
    assert second.batch_id != first.batch_id and len(store.ladders(second.batch_id)) == 1


def test_a_ladder_left_running_is_aborted_at_the_start(store):
    _campaign(store, "naca0012")
    shape_id = store._conn.execute("SELECT geometry_id FROM ladders").fetchone()[0]
    store._conn.execute("DELETE FROM best")
    store._conn.execute("DELETE FROM runs")
    store._conn.execute("UPDATE ladders SET status = 'running', finished_at = NULL")  # a campaign cut short

    result = _campaign(store, "naca0012")

    assert result.names == ["naca0012"]  # not finished: it runs again
    left, rerun = store._conn.execute("SELECT status, error_text FROM ladders ORDER BY rowid").fetchall()
    assert left == ("aborted", "still running when a campaign started") and rerun[0] == "passed"
    assert store.best(shape_id, result.settings_id) is not None


def test_nothing_left_to_run(store):
    _campaign(store, "naca0012")
    result = _campaign(store, "naca0012")

    assert (result.batch_id, result.names, result.skipped) == (None, [], ["naca0012"])
    assert Optim_campaign_summary(store, result).endswith("Nothing left to run.")
# endregion


# region Step 3
def test_the_summary_counts_the_batch(store):
    _ABORTING.add("e376")
    _ABORTING_ONCE.add("naca4412")  # its retry passes: counted by its last ladder
    text = Optim_campaign_summary(store, _campaign(store, "naca0012", "naca4412", "e376"))
    lines = text.splitlines()

    assert "3 airfoils run, 0 finished before" in lines[0]
    assert lines[1:] == ["Airfoils: 2 passed, 1 aborted",
                         "Best meshes: median 59,500 nodes, 58,000-61,000",
                         "Passed at: golden 2",
                         "c2d runs: golden 2, jmax_bump 0, smoothing 0, mads 0 (2 in all, retries included)",
                         "Retried: 2",
                         "  aborted e376: RuntimeError: stand-in failure",
                         "Run the campaign again to retry what is left."]


def test_a_stopped_campaign_says_so(store):
    _SETUP.add("naca4412")
    result = _campaign(store, "naca0012", "naca4412", "e376")
    text = Optim_campaign_summary(store, result)

    assert isinstance(result.stopped, BatchStopped) and not _statuses(store, "e376")
    assert "Stopped early: setup error on naca4412: EXECUTABLE_NOT_FOUND" in text
    assert text.endswith("Run the campaign again to retry what is left.")
# endregion
