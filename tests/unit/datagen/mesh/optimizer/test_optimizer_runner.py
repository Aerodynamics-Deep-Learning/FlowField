"""
Tests for `meshing.optimizer.batch.runner`, one task per airfoil over isolated worker slots, i.e.:

Step 1: Ensure the airfoils come from the geometry manifest
    - test_load_shapes_reads_the_manifest
    - test_an_unknown_airfoil_is_refused
Step 2: Ensure every event reaches the store from the main process as it arrives, outcomes keep their order, and
a slot's worker runs single-threaded in a folder of its own
    - test_every_event_is_recorded_and_outcomes_keep_their_order
    - test_a_slot_runs_single_threaded_in_a_folder_of_its_own
Step 3: Ensure an aborted airfoil runs once more after the rest, and a second abort stands
    - test_an_aborted_airfoil_runs_once_more_after_the_rest
    - test_an_airfoil_aborting_twice_stays_aborted
    - test_a_raising_task_aborts_only_its_own_ladder
Step 4: Ensure a worker that dies, goes silent or outgrows its cap loses its own airfoil and nothing else
    - test_a_dying_worker_loses_only_its_airfoil
    - test_a_worker_dying_twice_stays_aborted
    - test_a_silent_worker_is_killed_by_the_watchdog
    - test_a_worker_over_its_cap_loses_only_its_airfoil
Step 5: Ensure a setup error, aborts in a row and Ctrl-C stop the batch, leaving no ladder running
    - test_a_setup_error_stops_the_batch
    - test_aborts_in_a_row_stop_the_batch
    - test_aborts_with_passes_between_them_do_not_stop_the_batch
    - test_ctrl_c_stops_the_slots_and_aborts_their_ladders
Step 6: Ensure the limits hold together: the watchdog outlasts c2d's own timeout, a tight machine is warned of
    - test_the_watchdog_outlasts_c2ds_own_timeout
    - test_a_pool_too_big_for_memory_is_warned
Step 7: Ensure the log says how far a batch is and only what needs a look, by the caller's rule if it has one
    - test_the_log_has_progress_and_only_the_edge_cases

The task is `runner_tasks.task`, a ladder of two runs whose airfoil's name says how it goes, kept in a module of its
own so a worker starts as light as the runner. The slots, the pool and the store are real; only the watchdog's
wait is not sat through, the runner's clock moved past it instead.
"""

import inspect
import logging
import os
import shutil
import time
import types
import uuid
from pathlib import Path

import pytest

import src.datagen.meshing.optimizer.batch.runner as runner
from src.datagen.geometry.common.paths import Geom_paths
from src.datagen.meshing.c2d.run import C2D_generate_mesh
from src.datagen.meshing.optimizer.batch.runner import BatchStopped, Optim_load_shapes, Optim_run_airfoils, SlotLost
from src.datagen.meshing.optimizer.batch.slots import MemoryState, Optim_slot_platform
from src.datagen.meshing.optimizer.common.constants import WATCHDOG_S
from src.datagen.meshing.optimizer.common.settings import Optim_build_settings
from src.datagen.meshing.optimizer.evaluate.context import RunContext, StudySetupError
from src.datagen.meshing.optimizer.manifest.store import RunStore
from tests.geometry_shapes import stored_shape
from tests.unit.datagen.mesh.optimizer.runner_tasks import RUNS, task, tries_dir


@pytest.fixture
def settings(tmp_path, monkeypatch):
    exe = tmp_path / "c2d.exe"
    exe.write_bytes(b"not really c2d")
    monkeypatch.setattr("src.datagen.meshing.optimizer.common.settings.find_tool", lambda key: str(exe))
    return Optim_build_settings()


@pytest.fixture
def ctx(settings, tmp_path):
    batch_id = str(uuid.uuid4())
    yield RunContext(settings=settings, batch_id=batch_id, host="h", git_commit="c" * 40, git_dirty=0,
                     config_schema_fp="fp", work_root=str(tmp_path / "work"))
    shutil.rmtree(tries_dir(batch_id), ignore_errors=True)


@pytest.fixture
def store(tmp_path):
    with RunStore.open(tmp_path / "m.sqlite") as store:
        yield store


def _airfoils(*names):
    """Stand-in airfoils, each borrowing a test shape's record under its own name."""
    shapes = ("naca0012", "naca4412", "cusp")
    return [(name, stored_shape(shapes[i % 3])) for i, name in enumerate(names)]


def _ladders(store, name):
    """An airfoil's ladders as (status, error_text), in the order they started."""
    return [(row[0], row[1]) for row in store._conn.execute(
        "SELECT status, error_text FROM ladders WHERE airfoil_name = ? ORDER BY rowid", (name,))]


def _running(store):
    return store._conn.execute("SELECT COUNT(*) FROM ladders WHERE status = 'running'").fetchone()[0]


# region Step 1
@pytest.fixture
def manifest():
    db = Geom_paths().db
    if not db.exists():
        pytest.skip(f"no geometry manifest at {db}")
    return db


def test_load_shapes_reads_the_manifest(manifest):
    every = Optim_load_shapes(db=manifest)
    names = [name for name, _ in every]

    assert names == sorted(names) and {"naca0012", "naca4412"} <= set(names)
    picked = Optim_load_shapes(["naca4412", "naca0012"], db=manifest)
    assert [name for name, _ in picked] == ["naca4412", "naca0012"]  # the order asked for
    assert picked[0][1] == dict(every)["naca4412"]


def test_an_unknown_airfoil_is_refused(manifest):
    with pytest.raises(ValueError, match="no shape under .* for naca9999, not_a_foil"):
        Optim_load_shapes(["naca0012", "naca9999", "not_a_foil"], db=manifest)
# endregion


# region Step 2
@pytest.mark.parametrize("workers", [1, 2])
def test_every_event_is_recorded_and_outcomes_keep_their_order(ctx, store, workers):
    outcomes = Optim_run_airfoils(task, _airfoils("slow_a", "b", "c", "d"), ctx, store, workers)

    assert [o.route for o in outcomes] == [["slow_a"], ["b"], ["c"], ["d"]]  # the first finishes last
    assert store.settings(ctx.settings["settings_id"]) is not None  # recorded before any ladder
    for outcome in outcomes:
        assert store.ladder(outcome.ladder_id)["status"] == "failed"  # its end, after its start
        assert [r["seq"] for r in store.runs(outcome.ladder_id)] == list(range(RUNS))


def test_a_slot_runs_single_threaded_in_a_folder_of_its_own(ctx, store):
    ours = os.environ.get("OPENBLAS_NUM_THREADS")
    outcomes = Optim_run_airfoils(task, _airfoils("env_a", "env_b"), ctx, store, 2)
    folders = {Path(o.notes["work_root"]) for o in outcomes}

    assert [o.notes["blas"] for o in outcomes] == ["1", "1"]
    assert len(folders) == 2 and all(f.parent == Path(ctx.work_root) and f.name.startswith("slot") for f in folders)
    assert os.environ.get("OPENBLAS_NUM_THREADS") == ours  # the main process's own, as it was
# endregion


# region Step 3
@pytest.mark.parametrize("workers", [1, 2])
def test_an_aborted_airfoil_runs_once_more_after_the_rest(ctx, store, workers):
    outcomes = Optim_run_airfoils(task, _airfoils("a", "broken_once", "c", "d"), ctx, store, workers)

    assert [o.status for o in outcomes] == ["failed"] * 4  # the retry's outcome
    assert [s for s, _ in _ladders(store, "broken_once")] == ["aborted", "failed"]
    if workers == 1:
        order = [row[0] for row in store._conn.execute("SELECT airfoil_name FROM ladders ORDER BY rowid")]
        assert order == ["a", "broken_once", "c", "d", "broken_once"]


@pytest.mark.parametrize("workers", [1, 2])
def test_an_airfoil_aborting_twice_stays_aborted(ctx, store, workers):
    outcomes = Optim_run_airfoils(task, _airfoils("a", "broken", "c"), ctx, store, workers)

    assert [o.status for o in outcomes] == ["failed", "aborted", "failed"]
    assert str(outcomes[1].error) == "reported, not raised"  # handed back, not raised
    assert [s for s, _ in _ladders(store, "broken")] == ["aborted", "aborted"]


@pytest.mark.parametrize("workers", [1, 2])
def test_a_raising_task_aborts_only_its_own_ladder(ctx, store, workers):
    outcomes = Optim_run_airfoils(task, _airfoils("a", "boom_once", "c"), ctx, store, workers)

    assert [o.status for o in outcomes] == ["failed"] * 3
    (first, why), (second, _) = _ladders(store, "boom_once")
    assert (first, second) == ("aborted", "failed") and "a bug in the task" in why  # its traceback
    first_id = store._conn.execute("SELECT ladder_id FROM ladders WHERE airfoil_name = 'boom_once' "
                                   "ORDER BY rowid").fetchone()[0]
    assert len(store.runs(first_id)) == 1  # what it reported before raising stays
# endregion


# region Step 4
def test_a_dying_worker_loses_only_its_airfoil(ctx, store):
    outcomes = Optim_run_airfoils(task, _airfoils("slow_a", "crash_once", "slow_b", "c"), ctx, store, 2)

    assert [o.status for o in outcomes] == ["failed"] * 4  # the ones running beside it finished
    (first, why), (second, _) = _ladders(store, "crash_once")
    assert (first, second) == ("aborted", "failed") and why == "its worker died (exit code 3)"
    first_id = store._conn.execute("SELECT ladder_id FROM ladders WHERE airfoil_name = 'crash_once' "
                                   "ORDER BY rowid").fetchone()[0]
    assert len(store.runs(first_id)) == RUNS  # recorded as they came, before it died
    assert _running(store) == 0


def test_a_worker_dying_twice_stays_aborted(ctx, store):
    outcomes = Optim_run_airfoils(task, _airfoils("a", "crash", "b"), ctx, store, 2)

    assert [o.status for o in outcomes] == ["failed", "aborted", "failed"]
    assert isinstance(outcomes[1].error, SlotLost)
    assert [s for s, _ in _ladders(store, "crash")] == ["aborted", "aborted"]


def test_a_silent_worker_is_killed_by_the_watchdog(ctx, store, monkeypatch):
    # The runner's clock jumps the whole watchdog once the worker has gone silent after its runs, and "a" is done
    skipped = [0.0]
    monkeypatch.setattr(runner, "time", types.SimpleNamespace(monotonic=lambda: time.monotonic() + skipped[0],
                                                              perf_counter=time.perf_counter))
    real_wait = runner.wait

    def silence_passes(objects, timeout=None):
        hung = store._conn.execute("SELECT COUNT(*) FROM runs JOIN ladders USING (ladder_id) "
                                   "WHERE ladders.airfoil_name = 'hang_once'").fetchone()[0]
        if not skipped[0] and hung == RUNS and _ladders(store, "a") == [("failed", None)]:
            skipped[0] = WATCHDOG_S
        return real_wait(objects, timeout)

    monkeypatch.setattr(runner, "wait", silence_passes)
    t0 = time.monotonic()
    outcomes = Optim_run_airfoils(task, _airfoils("a", "hang_once"), ctx, store, 2)

    assert time.monotonic() - t0 < 60  # not the watchdog's own wait
    assert [o.status for o in outcomes] == ["failed", "failed"]
    assert _ladders(store, "hang_once")[0] == ("aborted", f"hung: nothing from its worker for {WATCHDOG_S} s")


def test_a_worker_over_its_cap_loses_only_its_airfoil(ctx, store):
    if not Optim_slot_platform().caps_memory:
        pytest.skip("no memory cap on this platform: a hog would take the machine")
    outcomes = Optim_run_airfoils(task, _airfoils("a", "hog_once", "b"), ctx, store, 2, cap_mb=1200)

    assert [o.status for o in outcomes] == ["failed"] * 3
    assert _ladders(store, "hog_once")[0] == ("aborted", "hit its 1200 MB memory cap")
# endregion


# region Step 5
@pytest.mark.parametrize("workers", [1, 2])
def test_a_setup_error_stops_the_batch(ctx, store, workers):
    with pytest.raises(BatchStopped, match="setup error on setup: EXECUTABLE_NOT_FOUND") as stopped:
        Optim_run_airfoils(task, _airfoils("slow_a", "setup", "slow_b", "slow_c", "d"), ctx, store, workers)

    assert isinstance(stopped.value.__cause__, StudySetupError)
    assert _ladders(store, "setup") == [("aborted", None)]  # never retried: a setup to fix
    assert _running(store) == 0 and not _ladders(store, "d")


def test_aborts_in_a_row_stop_the_batch(ctx, store):
    with pytest.raises(BatchStopped, match="3 airfoils aborted in a row, the last broken3"):
        Optim_run_airfoils(task, _airfoils("broken1", "broken2", "broken3", "d"), ctx, store, 1)

    assert not _ladders(store, "d")
    assert len(_ladders(store, "broken1")) == 1  # its retry would have come after d


def test_aborts_with_passes_between_them_do_not_stop_the_batch(ctx, store):
    names = ("broken1_once", "a", "broken2_once", "b", "broken3_once")  # three aborts, never two in a row
    outcomes = Optim_run_airfoils(task, _airfoils(*names), ctx, store, 1)

    assert [o.status for o in outcomes] == ["failed"] * 5  # every retry passed
    assert [len(_ladders(store, n)) for n in names] == [2, 1, 2, 1, 2]


def test_ctrl_c_stops_the_slots_and_aborts_their_ladders(ctx, store, monkeypatch):
    real_wait = runner.wait

    def interrupted(objects, timeout=None):
        if _running(store) == 2:
            raise KeyboardInterrupt
        return real_wait(objects, timeout)

    monkeypatch.setattr(runner, "wait", interrupted)
    t0 = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        Optim_run_airfoils(task, _airfoils("hang1", "hang2", "c"), ctx, store, 2)

    assert time.monotonic() - t0 < 60
    assert _ladders(store, "hang1") == _ladders(store, "hang2") == [("aborted", "interrupted (Ctrl-C)")]
    assert _running(store) == 0 and not _ladders(store, "c")
# endregion


# region Step 6
def test_the_watchdog_outlasts_c2ds_own_timeout():
    c2d_timeout = inspect.signature(C2D_generate_mesh).parameters["timeout"].default

    assert WATCHDOG_S >= c2d_timeout + 120  # c2d is killed at its timeout, then the run is scored


class _TightMachine:
    name, caps_memory = "tight", False

    def __init__(self, commit_available_mb=500):
        self.commit = commit_available_mb

    def memory(self):
        return MemoryState(16000, 1000, 20000, self.commit)


def test_a_pool_too_big_for_memory_is_warned(caplog):
    runner._warn_memory(_TightMachine(), 6, 2048)
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]

    assert len(warnings) == 2 and "not capped in memory" in warnings[0]
    assert "6 slots may use up to 12288 MB together at their caps" in warnings[1]  # one line for every limit short
    assert "1000 MB of RAM" in warnings[1] and "500 MB to commit" in warnings[1]
    caplog.clear()
    runner._warn_memory(_TightMachine(commit_available_mb=20000), 6, 2048)
    [_, memory] = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert "1000 MB of RAM" in memory and "to commit" not in memory
# endregion


# region Step 7
def test_the_log_has_progress_and_only_the_edge_cases(ctx, store, caplog, monkeypatch):
    names = [f"golden{i}" for i in range(8)] + ["mads", "plain"]  # 'plain' fails, as the task's other names do

    with caplog.at_level(logging.INFO, logger=runner.__name__):
        Optim_run_airfoils(task, _airfoils(*names), ctx, store, 1)

    said = [(r.levelname, r.getMessage()) for r in caplog.records]
    assert [m.split(" (")[0] for _, m in said if "/min, ends ~" in m] == [f"{k}/10" for k in range(1, 10)]
    lines = [(level, m.rsplit(" in ", 1)[0]) for level, m in said if m.startswith(("golden", "mads", "plain"))]
    assert lines == [  # a golden pass says nothing; the climb and the failure their golden runs' numbers
        ("INFO", f"mads: passed via golden > mads, 50,000 nodes, 2 c2d runs; golden failed: skew 0.500 > "
                 f"{ctx.settings['gate_skew_max']:g}"),
        ("WARNING", "plain: failed via plain, 0 c2d runs")]
    warned = []
    monkeypatch.setattr(runner, "_warn_memory", lambda *args: warned.append(args))
    Optim_run_airfoils(task, _airfoils("x", "y"), ctx, store, 2, warn_memory=False)
    assert warned == []  # a caller running several batches warns once

    caplog.clear()  # a caller's own rule: here only the climb to rung 3, in its words
    with caplog.at_level(logging.INFO, logger=runner.__name__):
        Optim_run_airfoils(task, _airfoils("golden0", "mads", "plain"), ctx, store, 1,
                           edge=lambda o: "climbed" if "mads" in o.route else None)
    assert [(r.levelname, r.getMessage().rsplit(" in ", 1)[0]) for r in caplog.records
            if r.getMessage().startswith(("golden", "mads", "plain"))] == [("WARNING", "mads: climbed")]
# endregion
