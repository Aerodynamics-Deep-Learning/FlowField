"""
Tests for `meshing.optimizer.ladder`, one airfoil's study, i.e.:

Step 1: Ensure the golden run, and each rung its verdict routes the ladder to
    - test_a_golden_pass_ends_the_ladder
    - test_the_golden_run_is_n_star_at_the_golden_jmax
    - test_size_ratio_alone_bumps_jmax_to_a_pass
    - test_bumps_follow_the_settings_then_go_to_rung_3
    - test_shape_gates_alone_go_to_rung_2
    - test_a_bump_failing_on_shape_goes_to_rung_2_from_that_bump
    - test_no_smoothing_pass_goes_to_rung_3
    - test_anything_else_goes_to_rung_3
Step 2: Ensure the ladder row: the route, counts, NOMAD searches and best, and the events
    - test_the_ladder_row_records_the_climb
    - test_the_events_write_the_store
    - test_each_ladder_has_its_own_cache
Step 3: Ensure what ends a ladder early is recorded, then handed up
    - test_no_n_star_fails_the_ladder
    - test_an_error_aborts_the_ladder_and_is_handed_up
    - test_a_rejected_or_invalid_config_is_a_ladder_bug
    - test_ctrl_c_is_recorded_then_raised
    - test_the_outcome_and_events_cross_a_process_boundary
    - test_an_unknown_event_is_refused

n* is stubbed and the executor is a stand-in whose rows come from a formula of the knobs, judged by the
real verdict, with the real one's cache and dominance; NOMAD, the contour and the store are real. The
real sampler and c2d are in the integration tier.
"""

import itertools
import json
import pickle
import uuid

import pytest

from src.datagen.geometry.common.constants import DX_MAX
from src.datagen.geometry.sample.contour import ContourError
from src.datagen.meshing.optimizer import ladder
from src.datagen.meshing.optimizer.constants import SMOOTHING_DEFAULTS
from src.datagen.meshing.optimizer.executor import RunContext, RunResult, StudySetupError, failed_gates
from src.datagen.meshing.optimizer.ladder import LadderOutcome, record, run_ladder
from src.datagen.meshing.optimizer.nstar import NStarResult
from src.datagen.meshing.optimizer.settings import build_settings
from src.datagen.meshing.optimizer.store import RunStore
from tests.geometry_shapes import stored_shape

NSTAR = NStarResult(n=400, knobs=dict(dx_mid=DX_MAX, le_ratio=0.62, te_ratio=1.0, cluster_width=0.026), evals=1200,
                    seconds=0.5)
GOLDEN = {**NSTAR.knobs, "jmax": 147, **SMOOTHING_DEFAULTS}
SAMPLED = ("dx_mid", "le_ratio", "te_ratio", "cluster_width", "jmax")
# C2D_MeshingConfig's defaults: rung 3's second start
DEFAULTS = {"dx_mid": DX_MAX, "le_ratio": 0.025, "te_ratio": 0.05, "cluster_width": 0.05, "jmax": 200,
            **SMOOTHING_DEFAULTS}
_CLEAN = dict(entry_flag="SUCCESS", q_neg_cells=0, q_skew_max=0.25, q_ortho_min=0.91, q_jac_min=0.91, q_ar_max=9e3,
              q_size_ratio_max=1.2, g_max_mesh_to_input=1e-8, g_max_input_to_mesh=2e-8)


class _Executor:
    """
    A stand-in `RunExecutor`: `world(knobs)` gives a run's status, surface points and quality, which become
    a row the real verdict judges; the real one's cache, dominance and emit. "raise" is raised before the
    run, "setup" after emitting it, as a setup flag's StudySetupError is.
    """

    def __init__(self, ctx, emit, world):
        self.ctx, self.emit, self.world = ctx, emit, world
        self.calls, self.cache = [], {}

    def run(self, *, knobs, skip_if_nodes_ge=None, contour, **shared):
        self.calls.append(dict(knobs=knobs, skip_if_nodes_ge=skip_if_nodes_ge, contour=contour, **shared))
        key = json.dumps(knobs, sort_keys=True)
        if key in self.cache:
            return RunResult(self.cache[key], cached=True)
        outcome = {"n": 400, **self.world(knobs)}
        if "raise" in outcome:
            raise outcome["raise"]
        setup = outcome.pop("setup", None)
        row = self._row(knobs, skip_if_nodes_ge, shared, outcome)
        if row["status"] in ("meshed", "rejected", "mesher_failed"):
            self.cache[key] = row
        self.emit(row)
        if setup is not None:
            raise setup
        return RunResult(row, cached=False)

    def _row(self, knobs, skip, shared, outcome):
        s = self.ctx.settings
        status, n = outcome.pop("status", "meshed"), outcome.pop("n")
        row = dict(run_id=str(uuid.uuid4()), settings_id=s["settings_id"], batch_id=self.ctx.batch_id,
                   config_schema_fp=self.ctx.config_schema_fp, **shared, **knobs, s_n_required=n, s_n_points=n,
                   s_precheck_turn=-0.1)
        nodes = (n - 1) * knobs["jmax"]
        if status == "meshed" and skip is not None and nodes >= skip:
            status = "dominated"
        row["status"] = status
        if status == "rejected":
            row.update(s_n_points=None, s_precheck_turn=0.2, s_rejection="turn")
        elif status == "mesher_failed":
            row["entry_flag"] = "SUBPROCESS_FAIL"
        elif status in ("crashed", "config_invalid"):
            row.update(entry_flag=outcome.get("entry_flag"), error_text=f"{status}: here is why")
        elif status == "meshed":
            row.update(num_nodes=nodes, **{**_CLEAN, **outcome})
            failed = failed_gates(row, s)
            row.update(passed=int(not failed), failed_gates=failed)
        return row


def _at_golden_sampling(knobs):
    return all(knobs[k] == GOLDEN[k] for k in SAMPLED if k != "jmax")


def _at_default_smoothing(knobs):
    return all(knobs[k] == v for k, v in SMOOTHING_DEFAULTS.items())


def _sr_until(jmax):
    """Size ratio over its gate below `jmax`, at the gate from it on."""
    return lambda k: {"q_size_ratio_max": 1.25 + 0.01 * max(0, jmax - k["jmax"])}


def _only_golden(fault):
    """`fault` at the golden config's sampling knobs, whatever the smoothing; a clean pass elsewhere."""
    return lambda k: fault if _at_golden_sampling(k) else {}


def _calls(errors):
    """errors[i] raised at the i-th new run (from 1); otherwise skew fails at the golden sampling, passes elsewhere."""
    count = itertools.count(1)

    def world(k):
        i = next(count)
        return {"raise": errors[i]} if i in errors else _only_golden({"q_skew_max": 0.5})(k)
    return world


def _raises(error):
    def raiser(*args, **kw):
        raise error
    return raiser


@pytest.fixture
def settings(tmp_path, monkeypatch):
    exe = tmp_path / "c2d.exe"
    exe.write_bytes(b"not really c2d")
    monkeypatch.setattr("src.datagen.meshing.optimizer.settings.find_tool", lambda key: str(exe))
    return build_settings(rung2_budget=4, rung3_budget=5)  # apart, so neither can stand in for the other


@pytest.fixture
def climb(settings, tmp_path, monkeypatch):
    """Runs one ladder on naca0012 over `world`; returns what happened."""
    def go(world, s=None, nstar=NSTAR):
        s = s or settings
        executors, searches, events = [], [], []

        def executor(ctx, emit):
            executors.append(_Executor(ctx, emit, world))
            return executors[-1]

        def spy(*args, **kw):
            searches.append((args, kw))
            return real_run_mads(*args, **kw)

        monkeypatch.setattr(ladder, "RunExecutor", executor)
        monkeypatch.setattr(ladder, "run_mads", spy)
        monkeypatch.setattr(ladder, "n_star_search", lambda contour, settings: nstar)
        ctx = RunContext(settings=s, batch_id="b1", host="h", git_commit="c" * 40, git_dirty=0,
                         config_schema_fp="fp", work_root=str(tmp_path / "work"))
        outcome = run_ladder("naca0012", stored_shape("naca0012"), ctx, events.append)
        runs = [e[1] for e in events if e[0] == "run"]
        return _Climb(outcome, events, runs, events[-1][2], executors[0].calls if executors else [], searches)

    real_run_mads = ladder.run_mads
    return go


class _Climb:
    def __init__(self, outcome, events, runs, end, calls, searches):
        self.outcome, self.events, self.runs, self.end, self.calls, self.searches = \
            outcome, events, runs, end, calls, searches

    def rung(self, name):
        return [r for r in self.runs if r["rung"] == name]


# region Step 1
def test_a_golden_pass_ends_the_ladder(climb):
    c = climb(lambda k: {})
    golden = c.runs[0]

    assert c.outcome.status == "passed" and c.outcome.route == ["golden"] and c.outcome.error is None
    assert c.outcome.best_row is golden and len(c.runs) == len(c.calls) == 1 and c.searches == []
    assert (golden["rung"], golden["seq"], golden["parent_run_id"]) == ("golden", 0, None)


def test_the_golden_run_is_n_star_at_the_golden_jmax(climb, settings):
    c = climb(lambda k: {}, s=build_settings(golden_jmax=139))
    call = c.calls[0]

    assert call["knobs"] == {**NSTAR.knobs, "jmax": 139, **SMOOTHING_DEFAULTS}
    assert call["contour"].geometry_id == "naca0012" and call["contour"].te.value == "round"
    assert call["contour"].gap == settings["te_gap"]
    assert {k: call[k] for k in ("ladder_id", "geometry_id", "airfoil_name")} == \
        {"ladder_id": c.outcome.ladder_id, "geometry_id": "naca0012", "airfoil_name": "naca0012"}


def test_size_ratio_alone_bumps_jmax_to_a_pass(climb):
    c = climb(_sr_until(150))
    bumps = c.rung("jmax_bump")

    assert c.outcome.status == "passed" and c.outcome.route == ["golden", "jmax_bump"]
    assert [r["jmax"] for r in bumps] == [148, 149, 150] and [r["seq"] for r in bumps] == [0, 1, 2]
    assert [r["parent_run_id"] for r in bumps] == [r["run_id"] for r in c.runs[:3]]  # each follows the last
    assert all(r["failed_gates"] == ["size_ratio"] for r in c.runs[:3])
    assert c.outcome.best_row is bumps[-1] and c.searches == []


def test_bumps_follow_the_settings_then_go_to_rung_3(climb):
    c = climb(_sr_until(170), s=build_settings(jmax_bump_step=2, jmax_bump_max=3, rung3_budget=4))

    assert [r["jmax"] for r in c.rung("jmax_bump")] == [149, 151, 153]
    assert c.outcome.route == ["golden", "jmax_bump", "mads"]  # smoothing can't move the radial spacing
    (args, kw), = c.searches
    assert [k.name for k in args[3]] == list(GOLDEN) and args[2] == {} and args[4] == [GOLDEN]
    assert kw == dict(budget=4, objective="nodes", dominance=True, best_so_far=None)
    assert c.end["nomad_json"]["mads"]["budget"] == 4
    assert args[1]["parent_run_id"] == c.rung("jmax_bump")[-1]["run_id"]  # the run whose verdict sent it there
    assert c.calls[4]["knobs"] == GOLDEN and len(c.rung("golden")) == 1  # its start, the golden run, is cached


@pytest.mark.parametrize("fault, gates", [
    ({"q_skew_max": 0.5}, ["skew"]),
    ({"q_ortho_min": 0.5, "q_jac_min": 0.5}, ["ortho", "jac"]),
    ({"q_neg_cells": 2}, ["inverted"]),
], ids=["skew", "ortho_jac", "inverted"])
def test_shape_gates_alone_go_to_rung_2(climb, fault, gates):
    # Any move off c2d's smoothing defaults fixes the fault
    c = climb(lambda k: fault if _at_default_smoothing(k) else {})
    smoothing = c.rung("smoothing")

    assert c.runs[0]["failed_gates"] == gates
    assert c.outcome.status == "passed" and c.outcome.route == ["golden", "smoothing"]
    (args, kw), = c.searches
    assert [k.name for k in args[3]] == list(SMOOTHING_DEFAULTS) and args[4] == [SMOOTHING_DEFAULTS]
    assert args[2] == {k: GOLDEN[k] for k in SAMPLED}  # the routed config's rest, fixed
    assert kw == dict(budget=4, objective="violation", stop_at_first_pass=True)
    assert c.outcome.best_row is smoothing[-1] and smoothing[-1]["passed"] == 1
    assert all(r["parent_run_id"] == c.runs[0]["run_id"] for r in smoothing)
    assert [r["seq"] for r in smoothing] == list(range(len(smoothing)))  # the cached start took no number


def test_a_bump_failing_on_shape_goes_to_rung_2_from_that_bump(climb):
    def world(k):
        if k["jmax"] == 147:
            return {"q_size_ratio_max": 1.3}
        return {"q_skew_max": 0.5} if _at_default_smoothing(k) else {}

    c = climb(world)
    bump, = c.rung("jmax_bump")

    assert bump["failed_gates"] == ["skew"] and c.outcome.route == ["golden", "jmax_bump", "smoothing"]
    assert c.outcome.status == "passed"
    assert all(r["jmax"] == 148 and r["parent_run_id"] == bump["run_id"] for r in c.rung("smoothing"))


def test_no_smoothing_pass_goes_to_rung_3(climb):
    c = climb(_only_golden({"q_skew_max": 0.5}))

    assert c.outcome.route == ["golden", "smoothing", "mads"] and c.outcome.status == "passed"
    assert len(c.rung("smoothing")) == 4 and not any(r.get("passed") for r in c.rung("smoothing"))
    (_, rung2), (args, rung3) = c.searches
    assert args[4] == [GOLDEN] and rung3["budget"] == 5
    assert all(r["parent_run_id"] == c.runs[0]["run_id"] for r in c.rung("mads"))
    assert c.outcome.best_row["rung"] == "mads"


@pytest.mark.parametrize("fault, gates, second_start", [
    ({"q_skew_max": 0.5, "q_size_ratio_max": 1.3}, ["skew", "size_ratio"], False),
    ({"g_max_mesh_to_input": 0.01}, ["geo_dev"], False),
    ({"q_ar_max": 2e7}, ["aspect_ratio"], False),
    ({"entry_flag": "CONVERSION_FAIL"}, ["conversion"], True),
    ({"status": "mesher_failed"}, ["mesher_failed"], True),
    ({"status": "crashed"}, ["crashed"], True),
], ids=["shape_and_size_ratio", "geo_dev", "aspect_ratio", "conversion", "mesher_failed", "crashed"])
def test_anything_else_goes_to_rung_3(climb, fault, gates, second_start):
    c = climb(_only_golden(fault))
    (args, kw), = c.searches

    assert c.end["golden_failed_gates"] == gates
    assert c.end["meshes_golden"] == (0 if gates == ["crashed"] else 1)  # a failed c2d run is still a c2d run
    assert c.outcome.route == ["golden", "mads"] and c.outcome.status == "passed"
    # NOMAD can't evaluate a golden run without a usable mesh, so it starts again from the config defaults
    assert args[4] == ([GOLDEN, DEFAULTS] if second_start else [GOLDEN])
    assert c.end["nomad_json"]["mads"]["start"] == (1 if second_start else 0)
# endregion


# region Step 2
def test_the_ladder_row_records_the_climb(climb):
    settings = build_settings(rung2_budget=4, rung3_budget=5, nomad_seed=3)
    resampled = itertools.count(1)

    def world(k):
        if k == GOLDEN:
            return {"q_size_ratio_max": 1.3}
        if _at_golden_sampling(k):
            return {"q_skew_max": 0.5}  # no smoothing fixes it
        return {"status": "rejected"} if next(resampled) % 3 == 0 else {}  # every third resampled config

    c = climb(world, s=settings)
    end, runs = c.end, c.runs
    best = min((r for r in runs if r.get("passed")), key=lambda r: r["num_nodes"])

    assert [e[0] for e in c.events] == ["ladder_start"] + ["run"] * len(runs) + ["ladder_end"]
    start = c.events[0][1]
    assert start["status"] == "running" and start["ladder_id"] == c.outcome.ladder_id == c.events[-1][1]
    assert {r["ladder_id"] for r in runs} == {c.outcome.ladder_id}
    assert end["status"] == "passed" and end["route"] == ["golden", "jmax_bump", "smoothing", "mads"]
    assert end["final_rung"] == "mads" and end["golden_failed_gates"] == ["size_ratio"]
    assert (end["n_star"], end["n_star_knobs_json"], end["n_star_grid_evals"], end["n_star_s"]) == \
        (400, NSTAR.knobs, 1200, 0.5)
    for column, rung in (("meshes_golden", "golden"), ("meshes_bump", "jmax_bump"),
                         ("meshes_smoothing", "smoothing"), ("meshes_mads", "mads")):
        assert end[column] == sum(r.get("entry_flag") is not None for r in c.rung(rung))
    assert (end["meshes_golden"], end["meshes_bump"], end["meshes_smoothing"]) == (1, 1, 4)
    assert 0 < end["meshes_mads"] <= 5
    assert end["n_rejected"] == sum(r["status"] == "rejected" for r in runs) > 0
    assert end["n_dominated"] == sum(r["status"] == "dominated" for r in runs)
    for rung, budget in (("smoothing", 4), ("mads", 5)):
        search = end["nomad_json"][rung]
        assert (search["budget"], search["seed"], search["start"]) == (budget, settings["nomad_seed"], 0)
        assert search["mesher_runs"] == end[f"meshes_{rung}"] and search["n_calls"] >= search["mesher_runs"]
    assert end["best_run_id"] == best["run_id"] and c.outcome.best_row is best
    assert end["error_text"] is None and end["wall_s"] > 0 and end["finished_at"] > start["started_at"]


def test_the_events_write_the_store(climb, settings, tmp_path):
    c = climb(_only_golden({"q_skew_max": 0.5}))
    with RunStore.open(tmp_path / "meshing.sqlite") as store:
        store.ensure_settings(settings)
        for event in c.events:
            record(store, event)
        row = store.ladder(c.outcome.ladder_id)
        best = store.best("naca0012", settings["settings_id"])

    assert row["status"] == "passed" and json.loads(row["route"]) == ["golden", "smoothing", "mads"]
    assert json.loads(row["nomad_json"]) == json.loads(json.dumps(c.end["nomad_json"]))
    assert row["best_run_id"] == best == c.outcome.best_row["run_id"]
    assert row["started_at"] == c.events[0][1]["started_at"] and row["wall_s"] == c.end["wall_s"]


def test_each_ladder_has_its_own_cache(climb):
    first, second = climb(lambda k: {}), climb(lambda k: {})

    assert len(first.runs) == len(second.runs) == 1  # the second golden run meshed again, not cached
    assert first.outcome.ladder_id != second.outcome.ladder_id
# endregion


# region Step 3
def test_no_n_star_fails_the_ladder(climb):
    c = climb(lambda k: {}, nstar=NStarResult(n=None, knobs=None, evals=1200, seconds=0.4))

    assert c.outcome.status == "failed" and c.outcome.route == ["nstar"] and c.outcome.error is None
    assert c.runs == [] and c.end["final_rung"] == "nstar"
    assert (c.end["n_star"], c.end["n_star_grid_evals"], c.end["n_star_s"]) == (None, 1200, 0.4)
    assert (c.end["golden_failed_gates"], c.end["meshes_golden"], c.end["best_run_id"]) == (None, 0, None)


@pytest.mark.parametrize("how", ["contour", "setup", "bug_in_rung_3"])
def test_an_error_aborts_the_ladder_and_is_handed_up(climb, monkeypatch, how):
    error = {"contour": ContourError("no room for the round TE"), "setup": StudySetupError("EXECUTABLE_NOT_FOUND"),
             "bug_in_rung_3": RuntimeError("the writer's queue is gone")}[how]
    if how == "contour":
        monkeypatch.setattr(ladder, "build_study_contour", _raises(error))
        world = lambda k: {}
    elif how == "setup":
        world = lambda k: {"status": "crashed", "setup": error}
    else:
        world = _calls({7: error})  # the golden, four smoothing runs, then rung 3
    c = climb(world)

    assert c.outcome.status == c.end["status"] == "aborted" and c.outcome.error is error
    assert type(error).__name__ in c.end["error_text"] and "Traceback" in c.end["error_text"]
    assert c.events[-1][0] == "ladder_end"  # recorded, not raised
    if how == "contour":
        assert c.runs == [] and c.end["route"] == [] and c.end["n_star"] is None
    elif how == "setup":
        assert [r["status"] for r in c.runs] == ["crashed"]  # the run that hit it is kept
        assert c.end["route"] == ["golden"]
    else:
        assert c.end["route"] == ["golden", "smoothing", "mads"] and "mads" not in c.end["nomad_json"]
        assert c.end["meshes_smoothing"] == 4


@pytest.mark.parametrize("status, rung", [("rejected", "golden"), ("config_invalid", "golden"),
                                          ("rejected", "jmax_bump")])
def test_a_rejected_or_invalid_config_is_a_ladder_bug(climb, status, rung):
    def world(k):
        if rung == "golden" or k["jmax"] > 147:
            return {"status": status}
        return {"q_size_ratio_max": 1.3}

    c = climb(world)

    assert c.outcome.status == "aborted" and isinstance(c.outcome.error, RuntimeError)
    assert f"the {rung} run was {status}" in str(c.outcome.error)
    assert ("here is why" in c.end["error_text"]) == (status == "config_invalid")  # the run's own reason, if any


def test_ctrl_c_is_recorded_then_raised(climb, settings, tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(ladder, "n_star_search", lambda contour, s: NSTAR)
    monkeypatch.setattr(ladder, "RunExecutor", lambda ctx, emit: _Executor(ctx, emit, _calls({3: KeyboardInterrupt()})))
    ctx = RunContext(settings=settings, batch_id="b1", host="h", git_commit="c" * 40, git_dirty=0,
                     config_schema_fp="fp", work_root=str(tmp_path))
    with pytest.raises(KeyboardInterrupt):
        run_ladder("naca0012", stored_shape("naca0012"), ctx, events.append)

    kind, _, end = events[-1]
    assert kind == "ladder_end" and end["status"] == "aborted" and "KeyboardInterrupt" in end["error_text"]


@pytest.mark.parametrize("how", ["passed", "contour", "setup"])
def test_the_outcome_and_events_cross_a_process_boundary(climb, monkeypatch, how):
    if how == "contour":
        monkeypatch.setattr(ladder, "build_study_contour", _raises(ContourError("no room for the round TE")))
    if how == "setup":
        world = lambda k: {"status": "crashed", "setup": StudySetupError("EXECUTABLE_NOT_FOUND")}
    else:
        world = _only_golden({"q_skew_max": 0.5})
    c = climb(world)
    outcome, events = pickle.loads(pickle.dumps((c.outcome, c.events)))

    assert isinstance(outcome, LadderOutcome) and outcome.status == c.outcome.status
    assert outcome.route == c.outcome.route and len(events) == len(c.events)
    if c.outcome.error is None:
        assert outcome.error is None and outcome.best_row["run_id"] == c.outcome.best_row["run_id"]
    else:  # the type survives, so the campaign can tell a missing c2d from a bug
        assert type(outcome.error) is type(c.outcome.error) and str(outcome.error) == str(c.outcome.error)


def test_an_unknown_event_is_refused(settings, tmp_path):
    with RunStore.open(tmp_path / "meshing.sqlite") as store:
        with pytest.raises(ValueError, match="unknown ladder event 'ladder_pause'"):
            record(store, ("ladder_pause", {}))
# endregion
