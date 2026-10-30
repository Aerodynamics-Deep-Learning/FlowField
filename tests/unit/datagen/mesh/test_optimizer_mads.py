"""
Tests for `meshing.optimizer.mads`, the NOMAD driver, i.e.:

Step 1: Ensure the budget counts new c2d runs only, and every point is a whole config of its rung
    - test_the_budget_counts_c2d_runs_only
    - test_a_cached_start_is_free_and_makes_no_row
    - test_every_point_is_a_whole_config_of_its_rung
Step 2: Ensure the best pass, the first-pass stop and the dominance threshold
    - test_the_search_stops_at_its_first_pass
    - test_the_dominance_threshold_follows_the_best
    - test_without_dominance_nothing_is_skipped
    - test_a_pass_from_before_is_the_first_threshold
Step 3: Ensure NOMAD's mechanics: start points, the stop callback's reset, the settings it reads
    - test_a_failed_start_falls_back_to_the_next
    - test_no_start_point_evaluated_ends_the_search
    - test_a_second_search_is_not_stopped_early
    - test_nomad_is_left_without_a_stop_callback
    - test_the_same_seed_repeats_the_search
    - test_the_evaluation_cap_comes_from_the_settings
    - test_the_objective_reaches_nomad
    - test_what_nomad_cannot_take_is_refused_before_it_runs
Step 4: Ensure an error while a point is evaluated stops the search and is raised
    - test_an_error_stops_the_search_and_is_raised
    - test_a_bug_in_the_outputs_stops_the_search

The executor is a stand-in whose rows come from a formula of the knobs; NOMAD, the verdict and
`nomad_outputs` are real. The real executor and c2d are in the integration tier.
"""

import itertools
import json
import math
import uuid
from types import SimpleNamespace

import PyNomad
import pytest

from src.datagen.geometry.common.constants import DX_MAX
from src.datagen.meshing.optimizer.constants import SMOOTHING_DEFAULTS
from src.datagen.meshing.optimizer.executor import RunResult, StudySetupError, failed_gates
from src.datagen.meshing.optimizer.mads import run_mads
from src.datagen.meshing.optimizer.settings import build_settings
from src.datagen.meshing.optimizer.space import knobs_for

GOLDEN = {"dx_mid": DX_MAX, "le_ratio": 0.025, "te_ratio": 1.0, "cluster_width": 0.026, "jmax": 147,
          **SMOOTHING_DEFAULTS}
BASE = {k: v for k, v in GOLDEN.items() if k not in SMOOTHING_DEFAULTS}  # rung 2's fixed knobs
SHARED = dict(ladder_id="l1", geometry_id="g1", airfoil_name="naca0012", contour=None, rung="smoothing",
              parent_run_id="p1")
_CLEAN = dict(entry_flag="SUCCESS", q_neg_cells=0, q_skew_max=0.25, q_ortho_min=0.91, q_jac_min=0.91, q_ar_max=9e3,
              q_size_ratio_max=1.2, g_max_mesh_to_input=1e-8, g_max_input_to_mesh=2e-8)


class _Executor:
    """
    A stand-in `RunExecutor`: `world(knobs)` gives a run's status, surface points and quality, which become
    a row the real verdict judges; the real one's cache and dominance; every call and its row kept.
    `raises` maps a call's number (from 1) to what it raises.
    """

    def __init__(self, settings, world, raises=None):
        self.ctx = SimpleNamespace(settings=settings)
        self.world, self.raises = world, raises or {}
        self.calls, self.results, self.cache = [], [], {}
        self.mesher_runs = 0

    def run(self, *, knobs, seq, skip_if_nodes_ge=None, **shared):
        self.calls.append(dict(knobs=knobs, seq=seq, skip_if_nodes_ge=skip_if_nodes_ge, **shared))
        if len(self.calls) in self.raises:
            raise self.raises[len(self.calls)]
        key = json.dumps(knobs, sort_keys=True)
        cached = key in self.cache
        row = self.cache[key] if cached else self._row(knobs, seq, skip_if_nodes_ge, shared)
        if row["status"] in ("meshed", "rejected", "mesher_failed"):
            self.cache[key] = row
        self.mesher_runs += not cached and row.get("entry_flag") is not None
        self.results.append(row)
        return RunResult(row, cached)

    def _row(self, knobs, seq, skip, shared):
        outcome = {"status": "meshed", "n": 400, **self.world(knobs)}
        status, n = outcome.pop("status"), outcome.pop("n")
        row = dict(run_id=str(uuid.uuid4()), seq=seq, **shared, **knobs, s_n_points=n, s_n_required=n,
                   s_precheck_turn=-0.1)
        nodes = (n - 1) * knobs["jmax"]
        if status == "meshed" and skip is not None and nodes >= skip:
            status = "dominated"
        row["status"] = status
        if status == "rejected":
            row.update(s_n_points=None, s_precheck_turn=0.2)
        elif status == "mesher_failed":
            row["entry_flag"] = "SUBPROCESS_FAIL"
        elif status == "crashed":
            row.update(outcome)  # an entry flag if it crashed after c2d returned
        elif status == "meshed":
            row.update(num_nodes=nodes, **{**_CLEAN, **outcome})
            failed = failed_gates(row, self.ctx.settings)
            row.update(passed=int(not failed), failed_gates=failed)
        return row


def _never_passes(knobs):
    """Skew over the gate everywhere, falling with epsi so NOMAD has somewhere to go."""
    return {"q_skew_max": 0.5 + 0.1 * math.log10(knobs["epsi"])}


def _odd_asmt_rejected(knobs):
    return {"status": "rejected"} if knobs["asmt"] % 2 else _never_passes(knobs)


def _every_outcome():
    """In turn: a mesh, a rejection, a c2d failure, a crash before c2d, a crash after it."""
    turns = itertools.cycle([_never_passes, lambda k: {"status": "rejected"}, lambda k: {"status": "mesher_failed"},
                             lambda k: {"status": "crashed"}, lambda k: {"status": "crashed", "entry_flag": "SUCCESS"}])
    return lambda knobs: next(turns)(knobs)


def _smoothing_skew(knobs):
    """Skew grows with epsi: c2d's default 20 fails the 0.4 gate, below ~12.6 passes."""
    return {"q_skew_max": 0.3 + 0.2 * math.log10(knobs["epsi"] / 4)}


def _nodes(knobs):
    """Surface points from dx_mid; the size ratio fails below jmax 140; a high le_ratio is rejected."""
    if knobs["le_ratio"] > 0.3:
        return {"status": "rejected"}
    return {"n": round(2.4 / knobs["dx_mid"]), "q_size_ratio_max": 1.2 + 0.01 * max(0, 145 - knobs["jmax"])}


@pytest.fixture
def settings(tmp_path, monkeypatch):
    exe = tmp_path / "c2d.exe"
    exe.write_bytes(b"not really c2d")
    monkeypatch.setattr("src.datagen.meshing.optimizer.settings.find_tool", lambda key: str(exe))
    return build_settings()


def _smoothing(executor, x0s=(SMOOTHING_DEFAULTS,), budget=5, **kw):
    """A rung-2 search from c2d's smoothing defaults, the rest of the golden config fixed."""
    return run_mads(executor, SHARED, BASE, knobs_for(2), list(x0s), budget=budget, objective="violation", **kw)


def _full(executor, budget=15, dominance=True, **kw):
    """A rung-3 search over every knob from the golden config, fewest nodes first, with dominance."""
    return run_mads(executor, {**SHARED, "rung": "mads"}, {}, knobs_for(3), [GOLDEN], budget=budget,
                    objective="nodes", dominance=dominance, **kw)


# region Step 1
def test_the_budget_counts_c2d_runs_only(settings):
    executor = _Executor(settings, _every_outcome())
    result = _smoothing(executor, budget=8)
    ran = [r for r in result.rows if r.get("entry_flag") is not None]

    assert result.mesher_runs == len(ran) == executor.mesher_runs == 8
    assert {r["status"] for r in ran} == {"meshed", "mesher_failed", "crashed"}  # a failed c2d run spends it too
    assert {r["status"] for r in result.rows if r not in ran} == {"rejected", "crashed"}  # c2d never ran: free
    assert result.best_row is None
    # Nothing passed, yet the callback ended it within NOMAD's iteration, not at its cap of 80 calls: NOMAD only
    # calls it once it holds a feasible point, which an over search hands it
    assert result.run_flag == -5 and result.n_calls - len(executor.calls) <= 10


def test_a_cached_start_is_free_and_makes_no_row(settings):
    executor = _Executor(settings, _never_passes)
    golden = executor.run(knobs=GOLDEN, seq=0, **{**SHARED, "rung": "golden"}).row  # the ladder's golden run
    result = _smoothing(executor, budget=3)

    assert executor.calls[1]["knobs"] == GOLDEN  # the start is the golden config exactly, despite the log scale
    assert executor.results[1] is golden and golden not in result.rows
    assert executor.calls[1]["seq"] == executor.calls[2]["seq"] == 0  # the hit took no number
    assert [r["seq"] for r in result.rows] == list(range(len(result.rows)))
    assert result.mesher_runs == executor.mesher_runs - 1 == 3


def test_every_point_is_a_whole_config_of_its_rung(settings):
    executor = _Executor(settings, _odd_asmt_rejected)
    _smoothing(executor, budget=5)

    for call in executor.calls:
        assert {k: call[k] for k in SHARED} == SHARED
        assert {k: call["knobs"][k] for k in BASE} == BASE
        for k in knobs_for(2):
            value = call["knobs"][k.name]
            assert k.lo <= value <= k.hi and isinstance(value, int) == (k.type == "I")
# endregion


# region Step 2
def test_the_search_stops_at_its_first_pass(settings):
    executor = _Executor(settings, _smoothing_skew)
    result = _smoothing(executor, budget=30, stop_at_first_pass=True)

    assert result.best_row is not None and result.best_row["passed"] == 1
    assert executor.results[-1] is result.best_row  # nothing ran after it
    assert sum(r.get("passed", 0) for r in result.rows) == 1
    assert result.mesher_runs < 30 and result.run_flag == -5


def test_the_dominance_threshold_follows_the_best(settings):
    executor = _Executor(settings, _nodes)
    result = _full(executor)

    best = None
    for call, row in zip(executor.calls, executor.results):
        assert call["skip_if_nodes_ge"] == (None if best is None else best["num_nodes"])
        if row.get("passed") and (best is None or row["num_nodes"] < best["num_nodes"]):
            best = row
    assert result.best_row is best and best["num_nodes"] < 399 * GOLDEN["jmax"]  # fewer nodes than the start
    statuses = [r["status"] for r in result.rows]
    assert "dominated" in statuses and "rejected" in statuses
    assert result.mesher_runs == statuses.count("meshed") <= 15


def test_without_dominance_nothing_is_skipped(settings):
    executor = _Executor(settings, _nodes)
    result = _full(executor, dominance=False)

    assert {c["skip_if_nodes_ge"] for c in executor.calls} == {None}
    assert "dominated" not in [r["status"] for r in result.rows]
    assert result.best_row["num_nodes"] == min(r["num_nodes"] for r in result.rows if r.get("passed"))


@pytest.mark.parametrize("given", [True, False], ids=["given", "cached_start"])
def test_a_pass_from_before_is_the_first_threshold(settings, given):
    executor = _Executor(settings, _nodes)
    golden = executor.run(knobs=GOLDEN, seq=0, **{**SHARED, "rung": "golden"}).row  # passed
    result = _full(executor, best_so_far=golden if given else None)

    # Given or not, the cached start is the golden pass, so it is the threshold from then on
    assert executor.results[1] is golden
    assert executor.calls[1]["skip_if_nodes_ge"] == (golden["num_nodes"] if given else None)
    assert executor.calls[2]["skip_if_nodes_ge"] == golden["num_nodes"]
    assert all(r["num_nodes"] < golden["num_nodes"] for r in result.rows if r["status"] == "meshed")
    assert result.best_row["num_nodes"] < golden["num_nodes"]
# endregion


# region Step 3
def test_a_failed_start_falls_back_to_the_next(settings):
    lowest = {"asmt": 5, "epsi": 2.0, "funi": 0.005, "alfa": 0.5}
    executor = _Executor(settings, lambda k: {"status": "mesher_failed"} if k["asmt"] == 5 else _never_passes(k))
    result = _smoothing(executor, x0s=(lowest, SMOOTHING_DEFAULTS), budget=3)

    assert [c["knobs"] for c in executor.calls[:2]] == [{**BASE, **lowest}, GOLDEN]
    assert result.start == 1 and result.run_flag != -3
    assert result.rows[0]["status"] == "mesher_failed"  # the failed start spent one c2d run
    assert result.mesher_runs == 3 and [r["status"] for r in result.rows].count("meshed") == 2


def test_no_start_point_evaluated_ends_the_search(settings):
    executor = _Executor(settings, lambda k: {"status": "mesher_failed"})
    result = _smoothing(executor, x0s=(SMOOTHING_DEFAULTS, {**SMOOTHING_DEFAULTS, "asmt": 20}))

    assert (result.start, result.run_flag, result.mesher_runs) == (1, -3, 2)
    assert len(executor.calls) == len(result.rows) == 2


def test_a_second_search_is_not_stopped_early(settings):
    first = _smoothing(_Executor(settings, _never_passes), budget=3)
    second = _smoothing(_Executor(settings, _smoothing_skew), budget=8)  # passes, so NOMAD calls the callback

    assert (first.run_flag, first.mesher_runs) == (-5, 3)
    assert second.best_row is not None and second.mesher_runs == 8  # the first search's callback, over, is gone


def test_nomad_is_left_without_a_stop_callback(settings):
    _smoothing(_Executor(settings, _never_passes), budget=3)  # over when it returns
    points = []

    def bb(x):  # a plain NOMAD run after it in the same process
        points.append(x.get_coord(0))
        x.setBBO(f"{(x.get_coord(0) - 0.3) ** 2!r}".encode())
        return 1

    result = PyNomad.optimize(bb, [0.5], [0.0], [1.0], ["DIMENSION 1", "BB_OUTPUT_TYPE OBJ", "MAX_BB_EVAL 20",
                                                         "DISPLAY_DEGREE 0"])
    assert result["run_flag"] != -5 and len(points) > 3  # not stopped by the search's callback


def test_the_same_seed_repeats_the_search(settings):
    def knobs_tried(settings):
        executor = _Executor(settings, _nodes)
        _full(executor, budget=8)
        return [c["knobs"] for c in executor.calls]

    assert knobs_tried(settings) == knobs_tried(settings)  # back to back: NOMAD's generator is global
    assert knobs_tried(settings) != knobs_tried(build_settings(nomad_seed=2))


def test_the_evaluation_cap_comes_from_the_settings(settings):
    executor = _Executor(build_settings(max_bb_eval_factor=1), _odd_asmt_rejected)
    result = _smoothing(executor, x0s=(SMOOTHING_DEFAULTS, {**SMOOTHING_DEFAULTS, "asmt": 20}), budget=5)

    assert (result.run_flag, result.n_calls, result.start) == (-2, 5, 0)  # 5 calls, not 5 meshes; no next start
    assert result.mesher_runs < 5


def test_the_objective_reaches_nomad(settings):
    def knobs_tried(objective):
        executor = _Executor(settings, _smoothing_skew)
        run_mads(executor, SHARED, BASE, knobs_for(2), [SMOOTHING_DEFAULTS], budget=6, objective=objective)
        return [c["knobs"] for c in executor.calls]

    assert knobs_tried("violation") != knobs_tried("nodes")  # rung 2's nodes are flat, its violation is not


@pytest.mark.parametrize("x0s, objective, match", [
    ([{**SMOOTHING_DEFAULTS, "asmt": 4}], "violation", "NOMAD cannot take asmt=4"),
    ([{**SMOOTHING_DEFAULTS, "epsi": 41.0}], "violation", "NOMAD cannot take epsi=41.0"),
    ([{**SMOOTHING_DEFAULTS, "asmt": 40.5}], "violation", "NOMAD cannot take asmt=40.5"),
    ([], "violation", "at least one start point"),
    ([SMOOTHING_DEFAULTS], "fewest", "'nodes' or 'violation'"),
], ids=["below", "above", "fractional", "none", "objective"])
def test_what_nomad_cannot_take_is_refused_before_it_runs(settings, x0s, objective, match):
    executor = _Executor(settings, _never_passes)
    with pytest.raises(ValueError, match=match):
        run_mads(executor, SHARED, BASE, knobs_for(2), x0s, budget=3, objective=objective)
    assert executor.calls == []
# endregion


# region Step 4
@pytest.mark.parametrize("error", [StudySetupError("EXECUTABLE_NOT_FOUND"), RuntimeError("the writer's queue is gone"),
                                   KeyboardInterrupt()], ids=["setup", "bug", "ctrl_c"])
def test_an_error_stops_the_search_and_is_raised(settings, error):
    executor = _Executor(settings, _never_passes, raises={3: error})
    with pytest.raises(type(error)) as raised:
        _smoothing(executor, budget=10)

    assert raised.value is error
    assert len(executor.calls) == 3  # nothing ran after it
    assert _smoothing(_Executor(settings, _smoothing_skew), budget=8).mesher_runs == 8  # the callback was reset


def test_a_bug_in_the_outputs_stops_the_search(settings):
    executor = _Executor(settings, lambda k: {"q_ar_max": 0.0})  # ln(0) in nomad_outputs
    with pytest.raises(ValueError, match="math domain error"):
        _smoothing(executor)
    assert len(executor.calls) == 1
# endregion
