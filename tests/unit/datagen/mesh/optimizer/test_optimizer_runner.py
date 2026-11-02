"""
Tests for `meshing.optimizer.batch.runner`, one task per airfoil over a pool of processes, i.e.:

Step 1: Ensure the airfoils come from the geometry manifest
    - test_load_shapes_reads_the_manifest
    - test_an_unknown_airfoil_is_refused
Step 2: Ensure every airfoil's events reach the store from the main process, and outcomes keep their order
    - test_every_event_is_recorded_and_outcomes_keep_their_order
Step 3: Ensure a task that raises is raised once the airfoils already finished are recorded
    - test_a_raising_task_is_raised_with_the_finished_airfoils_kept

The tasks are module-level stand-ins, so a worker process can take them, each reporting a ladder of its own
in the ladder's events; the pool, the store and its writes are real.
"""

import time
import uuid

import pytest

from src.datagen.geometry.common.paths import Geom_paths
from src.datagen.meshing.optimizer.batch.runner import Optim_load_shapes, Optim_run_airfoils
from src.datagen.meshing.optimizer.common.settings import Optim_build_settings
from src.datagen.meshing.optimizer.evaluate.executor import RunContext
from src.datagen.meshing.optimizer.manifest.store import Optim_utc_now, RunStore
from src.datagen.meshing.optimizer.search.ladder import LadderOutcome
from tests.geometry_shapes import stored_shape

_RUNS = 3


def _task(name, shape, ctx, emit):
    """A ladder of three runs; 'boom' raises before reporting anything, 'broken' reports an error, 'slow…' waits."""
    if name == "boom":
        raise RuntimeError("a bug in the task")
    if name.startswith("slow"):
        time.sleep(0.5)
    s, ladder_id = ctx.settings, str(uuid.uuid4())
    shared = dict(settings_id=s["settings_id"], batch_id=ctx.batch_id, geometry_id=shape.geometry_id, airfoil_name=name)
    emit(("ladder_start", dict(ladder_id=ladder_id, status="running", started_at=Optim_utc_now(), **shared)))
    for seq in range(_RUNS):
        emit(("run", dict(run_id=str(uuid.uuid4()), ladder_id=ladder_id, rung="manual", seq=seq, status="rejected",
                          config_schema_fp="fp", started_at=Optim_utc_now(), **shared)))
    emit(("ladder_end", ladder_id, dict(status="failed", finished_at=Optim_utc_now())))
    error = RuntimeError("reported, not raised") if name == "broken" else None
    return LadderOutcome(ladder_id, "aborted" if error else "failed", [name], None, error)


@pytest.fixture
def settings(tmp_path, monkeypatch):
    exe = tmp_path / "c2d.exe"
    exe.write_bytes(b"not really c2d")
    monkeypatch.setattr("src.datagen.meshing.optimizer.common.settings.find_tool", lambda key: str(exe))
    return Optim_build_settings()


@pytest.fixture
def ctx(settings, tmp_path):
    return RunContext(settings=settings, batch_id="b1", host="h", git_commit="c" * 40, git_dirty=0,
                      config_schema_fp="fp", work_root=str(tmp_path / "work"))


def _airfoils(*names):
    """Stand-in airfoils, each borrowing a test shape's record under its own name."""
    shapes = ("naca0012", "naca4412", "cusp")
    return [(name, stored_shape(shapes[i % 3])) for i, name in enumerate(names)]


def _ladders(store):
    return {row[0]: row[1] for row in store._conn.execute("SELECT airfoil_name, status FROM ladders")}


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
def test_every_event_is_recorded_and_outcomes_keep_their_order(ctx, tmp_path, workers):
    airfoils = _airfoils("slow_a", "b", "broken", "d")  # the first finishes last
    with RunStore.open(tmp_path / "m.sqlite") as store:
        outcomes = Optim_run_airfoils(_task, airfoils, ctx, store, workers)

        assert [o.route for o in outcomes] == [["slow_a"], ["b"], ["broken"], ["d"]]
        assert str(outcomes[2].error) == "reported, not raised"  # the caller's to act on, not raised here
        assert store.settings(ctx.settings["settings_id"]) is not None  # recorded before any ladder
        for outcome in outcomes:
            assert store.ladder(outcome.ladder_id)["status"] == "failed"  # its end, after its start
            assert [r["seq"] for r in store.runs(outcome.ladder_id)] == list(range(_RUNS))
# endregion


# region Step 3
@pytest.mark.parametrize("workers", [1, 2])
def test_a_raising_task_is_raised_with_the_finished_airfoils_kept(ctx, tmp_path, workers):
    slow = [f"slow{i}" for i in range(2, 9)]
    with RunStore.open(tmp_path / "m.sqlite") as store:
        with pytest.raises(RuntimeError, match="a bug in the task"):
            Optim_run_airfoils(_task, _airfoils("slow1", "boom", *slow), ctx, store, workers)
        recorded = _ladders(store)

        assert "slow1" in recorded and "boom" not in recorded  # running when it raised: finished and kept
        assert set(recorded.values()) == {"failed"}  # every ladder recorded is whole: none left running
        if workers == 1:
            assert recorded == {"slow1": "failed"}  # nothing after it starts
        else:
            assert len(recorded) < 1 + len(slow)  # the pool stopped starting airfoils
# endregion
