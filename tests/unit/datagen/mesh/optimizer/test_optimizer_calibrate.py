"""
Tests for `meshing.optimizer.batch.calibrate`, the calibration of the ladder's rungs, i.e.:

Step 1: Ensure rung 1's bisection finds jmax*, logs only an airfoil without one, and the golden jmax and campaign
estimate drawn from it
    - test_the_bisection_finds_every_jmax_star
    - test_a_mesh_that_fails_ends_the_bisection
    - test_jmax_star_failing_another_gate_is_noted_not_logged
    - test_the_golden_jmax_is_the_median_plus_the_margin
    - test_the_campaign_estimate_times_n_star_the_golden_mesh_and_its_bumps
Step 2: Ensure rung 2 is read off real ladders
    - test_rung_2_is_read_off_real_ladders
Step 3: Ensure rung 3's curves, its replay of the plateau, and the budgets
    - test_the_curve_follows_the_best_over_c2d_runs
    - test_the_replay_stops_where_a_live_plateau_does
    - test_a_failed_golden_run_is_no_best_to_beat
    - test_the_plateau_pick_takes_the_fewest_meshes_within_the_loss
    - test_the_budgets_round_up_to_ten
Step 4: Ensure the rungs run together, the ordinary airfoils, the golden jmax's source, the report and its files
    - test_the_ordinary_airfoils_are_drawn_the_same_way
    - test_a_calibration_runs_every_rung_and_reports_it
    - test_the_golden_jmax_comes_from_the_flag_rung_1_or_the_constant
    - test_a_given_golden_jmax_over_rung_1_is_warned
    - test_a_dated_calibration_writes_its_report_and_database_once

n* is stubbed and the executor is a stand-in whose rows come from a formula of the knobs, judged by the real
verdict; NOMAD, the ladder, the store and the report are real. The real sampler and c2d are in the
integration tier.
"""

import json
import logging
import uuid
from datetime import date

import pytest

from src.datagen.geometry.common.constants import DX_MAX
from src.datagen.meshing.optimizer.batch import calibrate as cal
from src.datagen.meshing.optimizer.common.constants import GOLDEN_JMAX, JMAX_MAX, JMAX_MIN, SMOOTHING_DEFAULTS
from src.datagen.meshing.optimizer.common.settings import Optim_build_settings
from src.datagen.meshing.optimizer.evaluate.context import RunContext
from src.datagen.meshing.optimizer.evaluate.executor import Optim_failed_gates, RunResult
from src.datagen.meshing.optimizer.manifest.store import Optim_utc_now, RunStore
from src.datagen.meshing.optimizer.search import ladder
from src.datagen.meshing.optimizer.search.events import Optim_record
from src.datagen.meshing.optimizer.search.ladder import Optim_run_ladder
from src.datagen.meshing.optimizer.search.nstar import NStarResult
from tests.geometry_shapes import stored_shape

NSTAR = NStarResult(n=400, knobs=dict(dx_mid=DX_MAX, le_ratio=0.62, te_ratio=1.0, cluster_width=0.026), evals=1200,
                    seconds=2.0)
FLOOR = NStarResult(n=400, knobs=NSTAR.knobs, evals=40, seconds=0.1)
_RUNNER = "src.datagen.meshing.optimizer.batch.runner"
_CLEAN = dict(entry_flag="SUCCESS", q_neg_cells=0, q_skew_max=0.2, q_ortho_min=0.91, q_jac_min=0.91, q_ar_max=9e3,
              q_size_ratio_max=1.2, g_max_mesh_to_input=1e-8, g_max_input_to_mesh=2e-8)


class _Executor:
    """
    A stand-in `RunExecutor`: `world(knobs)` gives a run's status, surface points and quality, which become a
    row the real verdict judges; the real one's cache, dominance and emit, and a wall time from the nodes.
    """

    def __init__(self, ctx, emit, world):
        self.ctx, self.emit, self.world, self.cache = ctx, emit, world, {}

    def run(self, *, knobs, skip_if_nodes_ge=None, contour, **shared):
        key = json.dumps(knobs, sort_keys=True)
        if key in self.cache:
            return RunResult(self.cache[key], cached=True)
        s, outcome = self.ctx.settings, {"n": 400, **self.world(knobs)}
        status, n = outcome.pop("status", "meshed"), outcome.pop("n")
        row = dict(run_id=str(uuid.uuid4()), settings_id=s["settings_id"], batch_id=self.ctx.batch_id,
                   config_schema_fp="fp", **shared, **knobs, s_n_required=n, s_n_points=n, s_precheck_turn=-0.1,
                   started_at=Optim_utc_now())
        nodes = (n - 1) * knobs["jmax"]
        if status == "meshed" and skip_if_nodes_ge is not None and nodes >= skip_if_nodes_ge:
            status = "dominated"
        row["status"] = status
        if status == "meshed":
            row.update(num_nodes=nodes, wall_s=nodes / 2e4, **{**_CLEAN, **outcome})
            failed = Optim_failed_gates(row, s)
            row.update(passed=int(not failed), failed_gates=failed)
        elif status == "mesher_failed":
            row["entry_flag"] = "SUBPROCESS_FAIL"
        row["finished_at"] = Optim_utc_now()
        if status in ("meshed", "rejected", "mesher_failed"):
            self.cache[key] = row
        self.emit(row)
        return RunResult(row, cached=False)


def _sr_from(jmax_star):
    """The size ratio passes from `jmax_star` up, and nowhere below it."""
    return lambda k: {"q_size_ratio_max": 1.2 if k["jmax"] >= jmax_star else 1.3}


def _nodes(knobs):
    """Rung 3's world: points from dx_mid, the size ratio failing below jmax 140, a high le_ratio rejected."""
    if knobs["le_ratio"] > 0.9:
        return {"status": "rejected"}
    return {"n": round(2.4 / knobs["dx_mid"]), "q_size_ratio_max": 1.2 + 0.01 * max(0, 145 - knobs["jmax"])}


def _skew_off_defaults(knobs):
    """Skew 0.3 at c2d's smoothing defaults (over rung 2's calibration gate), 0.2 anywhere else."""
    return {"q_skew_max": 0.3 if all(knobs[k] == v for k, v in SMOOTHING_DEFAULTS.items()) else 0.2}


@pytest.fixture
def stubs(tmp_path, monkeypatch):
    """c2d's hash, n*, the floor's n*, the executor and the manifest, stubbed; returns the floor's calls."""
    exe = tmp_path / "c2d.exe"
    exe.write_bytes(b"not really c2d")
    monkeypatch.setattr("src.datagen.meshing.optimizer.common.settings.find_tool", lambda key: str(exe))
    monkeypatch.setattr(ladder, "Optim_n_star_search", lambda contour, s: NSTAR)
    floors = []
    monkeypatch.setattr(cal, "Optim_n_star_search", lambda contour, s: floors.append(s) or FLOOR)
    monkeypatch.setattr(cal, "Optim_load_shapes",
                        lambda names=None, db=None: [(n, stored_shape(n)) for n in names or ("naca0012",)])
    return floors


def _world(monkeypatch, world):
    monkeypatch.setattr(ladder, "RunExecutor", lambda ctx, emit: _Executor(ctx, emit, world))


def _climb(ladder_type, settings, tmp_path, name="naca0012"):
    """One ladder of `ladder_type`; its outcome and events."""
    events = []
    ctx = RunContext(settings=settings, batch_id="b1", host="h", git_commit="c" * 40, git_dirty=0,
                     config_schema_fp="fp", work_root=str(tmp_path / "work"))
    return Optim_run_ladder(name, stored_shape(name), ctx, events.append, ladder_type), events


def _stored(tmp_path, settings, *climbs, rung=1):
    """The climbs' events in a fresh store, and their stage."""
    store = RunStore.open(tmp_path / f"rung{rung}.sqlite")
    store.ensure_settings(settings)
    for _, events in climbs:
        for event in events:
            Optim_record(store, event)
    stage = cal.Stage(rung, settings, ["naca0012"] * len(climbs), [o for o, _ in climbs], 1.0)
    return store, stage


def _bisection(jmax_star, mesh_s, name="x", n_star=400):
    return cal.Bisection(name, "passed", n_star, n_star, 0.026, jmax_star, True, None, mesh_s, 2.0, 0.1, 30.0)


# region Step 1
def test_the_bisection_finds_every_jmax_star(stubs, tmp_path, monkeypatch):
    settings = Optim_build_settings()
    for target in range(JMAX_MIN - 3, JMAX_MAX + 3):
        _world(monkeypatch, _sr_from(target))
        outcome, events = _climb(cal._Bisect, settings, tmp_path)
        runs = [e[1] for e in events if e[0] == "run"]
        notes = outcome.notes

        expected = None if target > JMAX_MAX else max(target, JMAX_MIN)  # JMAX_MIN means "at most JMAX_MIN"
        assert notes["jmax_star"] == expected and "bisect_failed" not in notes, target
        assert len(runs) <= 8 and outcome.route == ["bisect"]
        assert [r["seq"] for r in runs] == list(range(len(runs))) and {r["rung"] for r in runs} == {"bisect"}
        assert all({k: r[k] for k in (*NSTAR.knobs, *SMOOTHING_DEFAULTS)} == {**NSTAR.knobs, **SMOOTHING_DEFAULTS}
                   for r in runs)
        if expected is not None:
            assert notes["jmax_star_passes"] is True and cal._bisect_edge(outcome) is None
        else:  # the log's line says what the top of the range measured
            assert cal._bisect_edge(outcome) == (f"no jmax* in [{JMAX_MIN}, {JMAX_MAX}]: at jmax {JMAX_MAX}, "
                                                 f"size_ratio 1.3000 > {settings['gate_size_ratio_max']:g}")
    # n* at the narrowest cluster_width: the full grid's le_ratio values, one cluster_width
    assert json.loads(stubs[-1]["nstar_json"]) == {**json.loads(settings["nstar_json"]), "cw_grid": 1}
    assert (notes["n_star_floor"], notes["n_star_floor_s"]) == (FLOOR.n, FLOOR.seconds)


def test_a_mesh_that_fails_ends_the_bisection(stubs, tmp_path, monkeypatch):
    _world(monkeypatch, lambda k: {"status": "mesher_failed"} if k["jmax"] < 150 else {})
    outcome, events = _climb(cal._Bisect, Optim_build_settings(), tmp_path)

    assert outcome.notes["bisect_failed"] == "jmax 139: mesher_failed" and "jmax_star" not in outcome.notes
    assert [e[1]["jmax"] for e in events if e[0] == "run"] == [159, 139]  # nothing after it
    assert cal._bisect_edge(outcome) == "bisection stopped at jmax 139: mesher_failed"


def test_jmax_star_failing_another_gate_is_noted_not_logged(stubs, tmp_path, monkeypatch):
    _world(monkeypatch, lambda k: {**_sr_from(142)(k), "q_skew_max": 0.5})
    outcome, _ = _climb(cal._Bisect, Optim_build_settings(), tmp_path)

    assert (outcome.notes["jmax_star"], outcome.notes["jmax_star_passes"]) == (142, False)  # by size ratio alone
    assert outcome.status == "failed" and cal._bisect_edge(outcome) is None  # rung 2's work: no line in the log
    no_n_star = NStarResult(n=None, knobs=None, evals=1200, seconds=2.0)
    monkeypatch.setattr(ladder, "Optim_n_star_search", lambda contour, s: no_n_star)
    assert cal._bisect_edge(_climb(cal._Bisect, Optim_build_settings(), tmp_path)[0]) == "n* accepted nothing"


@pytest.mark.parametrize("found, golden", [([139, 142, 150], 147), ([139, 142], 146), ([None, None], None)])
def test_the_golden_jmax_is_the_median_plus_the_margin(found, golden):
    rows = [_bisection(j, {}) for j in found]
    assert cal.Optim_golden_from(rows) == golden  # the median rounded up, + GOLDEN_MARGIN


def test_the_campaign_estimate_times_n_star_the_golden_mesh_and_its_bumps():
    rows = [_bisection(142, {159: 9.0, 139: 2.0, 149: 4.0, 144: 3.0, 141: 2.5, 142: 2.6}),  # passes at golden 146
            _bisection(148, {159: 8.0, 139: 2.0, 149: 3.6, 144: 3.1, 146: 3.3, 147: 3.4, 148: 3.5}),  # bumps twice
            _bisection(None, {}, n_star=None)]  # no n*: left out

    est = cal.Optim_campaign_estimate(rows, golden_jmax=146)
    # the first's nearest mesh to jmax 146 is 144 (149 is further); the second meshed 146 itself
    assert (est.airfoils, est.bumps) == (2, 2)
    assert est.mesh_s == pytest.approx(3.0 + 3 * 3.3) and est.mesh_median_s == pytest.approx((3.0 + 3.3) / 2)
    assert (est.n_star_s, est.n_star_floor_s) == (4.0, 0.2)
    assert cal.Optim_campaign_estimate([_bisection(142, {144: 3.0, 148: 4.0})], 146).mesh_s == 4.0  # a tie: the higher
    assert cal.Optim_campaign_estimate([_bisection(None, {}, n_star=None)], 146) is None
# endregion


# region Step 2
def test_rung_2_is_read_off_real_ladders(stubs, tmp_path, monkeypatch):
    settings = Optim_build_settings(gate_skew_max=cal.SKEW_GATE, rung2_budget=6, rung3_budget=0)
    _world(monkeypatch, _skew_off_defaults)
    passes = _climb(None, settings, tmp_path)
    _world(monkeypatch, lambda k: {"q_skew_max": 0.3})
    never = _climb(None, settings, tmp_path)
    store, stage = _stored(tmp_path, settings, passes, never, rung=2)
    first, second = cal.Optim_smoothings(store, stage)
    store.close()

    assert first.route == ["golden", "smoothing"] and first.golden_failed == ["skew"] and first.golden_skew == 0.3
    assert (first.passed, first.skew) == (True, 0.2) and 1 <= first.meshes <= 6
    assert second.route == ["golden", "smoothing", "mads"] and (second.passed, second.meshes) == (False, 6)
    assert second.skew == 0.3 and never[1][-1][2]["meshes_mads"] == 0  # rung 3 had no budget
# endregion


# region Step 3
def _row(status, nodes=None, passed=None, ran=True):
    stamp = Optim_utc_now()
    return dict(status=status, num_nodes=nodes, passed=passed, entry_flag="SUCCESS" if ran else None,
                started_at=stamp, finished_at=stamp)


def test_the_curve_follows_the_best_over_c2d_runs():
    rows = [_row("meshed", 600, 0), _row("rejected", ran=False), _row("meshed", 500, 1),
            _row("dominated", ran=False), _row("mesher_failed", passed=None), _row("meshed", 450, 1)]

    nodes, seconds = cal.Optim_curve(rows, None)
    assert nodes == [None, None, 500, 500, 450]  # one entry per c2d run, the start first
    assert len(seconds) == 5 and seconds[0] == 0.0 and seconds == sorted(seconds)
    assert cal.Optim_curve(rows, 520)[0] == [520, 520, 500, 500, 450]
    assert cal.Optim_meshes_within(nodes, 0.01) == 4 and cal.Optim_meshes_within(nodes, 0.2) == 2
    assert cal.Optim_meshes_within([None, None], 0.01) is None and cal.Optim_curve([], None) == ([None], [0.0])


class _DescendWithPlateau(cal._Descend):
    """Rung 3's calibration climb with a live plateau, to compare with the replay of a run without one."""

    def climb(self):
        if self._n_star():
            golden_knobs = self._golden()
            self._full_search(golden_knobs, parent=self.golden, plateau=(8, 0.001))


def test_the_replay_stops_where_a_live_plateau_does(stubs, tmp_path, monkeypatch):
    _world(monkeypatch, _nodes)
    settings = Optim_build_settings(rung3_budget=20)
    store, stage = _stored(tmp_path, settings, _climb(cal._Descend, settings, tmp_path), rung=3)
    descent, = cal.Optim_descents(store, stage)
    store.close()
    stop = cal.Optim_replay(descent.nodes, 8, 0.001)
    live, events = _climb(_DescendWithPlateau, settings, tmp_path)
    search = events[-1][2]["nomad_json"]["mads"]

    assert descent.golden_passed and descent.nodes[0] == descent.golden_nodes == 399 * 147
    assert len(descent.nodes) == 21 and stop < 20  # the cap's 20 c2d runs, and a stop before them, after a gain
    assert (search["ended"], search["mesher_runs"]) == ("plateau", stop)
    assert live.best_row["num_nodes"] == descent.nodes[stop] < descent.golden_nodes


def test_a_failed_golden_run_is_no_best_to_beat(stubs, tmp_path, monkeypatch):
    _world(monkeypatch, lambda k: {**_nodes(k), **_sr_from(150)(k)})  # the golden run, at jmax 147, fails
    settings = Optim_build_settings(rung3_budget=10)
    store, stage = _stored(tmp_path, settings, _climb(cal._Descend, settings, tmp_path), rung=3)
    descent, = cal.Optim_descents(store, stage)
    store.close()

    assert not descent.golden_passed and descent.golden_nodes == 399 * 147
    assert descent.nodes[0] is None  # the curve starts without a best, as the search itself did
    assert descent.final is not None and descent.final >= 399 * 150  # its passes come from jmax 150 up


def test_the_plateau_pick_takes_the_fewest_meshes_within_the_loss(monkeypatch):
    monkeypatch.setattr(cal, "WINDOWS", (2, 4))
    monkeypatch.setattr(cal, "GAINS", (0.01,))
    stalls = cal.Descent("a", "passed", 400, 1000, True, [1000, 900, 900, 900, 900, 850], [0, 1, 2, 3, 4, 5], None)
    never = cal.Descent("b", "failed", 400, 1000, False, [None, None, None], [0, 1, 2], None)

    table = cal.Optim_plateau_table([stalls, never])
    assert [(p.window, p.meshes) for p in table] == [(2, 3 + 2), (4, 5 + 2)]  # no pass: the cap, never a stop
    assert table[0].worst_loss == pytest.approx(900 / 850 - 1) and table[1].worst_loss == 0.0
    assert cal.Optim_pick_plateau(table).window == 4  # 2 stops before the last gain: too much given up
    monkeypatch.setattr(cal, "MAX_LOSS", 0.1)
    assert cal.Optim_pick_plateau(table).window == 2  # now both qualify: the fewest meshes
    monkeypatch.setattr(cal, "MAX_LOSS", -1.0)
    assert cal.Optim_pick_plateau(table) is None


def test_the_budgets_round_up_to_ten():
    def descent(nodes):
        return cal.Descent("x", "passed", 400, nodes[0], True, nodes, list(range(len(nodes))), None)

    late = descent([1000] * 13 + [900])  # within 1% of its best at run 13
    assert cal.Optim_rung3_cap([late, descent([1000, 990, 990])]) == 20
    assert cal.Optim_rung3_cap([descent([None, None])]) is None and cal.Optim_rung3_cap([descent([1000])]) == 10

    def smoothing(meshes, passed):
        return cal.Smoothing("x", "passed", ["golden", "smoothing"], 0.3, ["skew"], meshes, passed, 0.2, None)

    # "never" is left out
    assert cal.Optim_rung2_budget([smoothing(11, True), smoothing(4, True), smoothing(80, False)]) == 20
    assert cal.Optim_rung2_budget([smoothing(80, False)]) is None
# endregion


# region Step 4
def test_the_ordinary_airfoils_are_drawn_the_same_way():
    names = [*cal.CALIBRATION_SET, *(f"foil{i}" for i in range(12))]
    drawn = cal.Optim_ordinary_airfoils(names)

    assert drawn == cal.Optim_ordinary_airfoils(list(reversed(names))) == sorted(drawn)
    assert len(drawn) == cal.ORDINARY_COUNT and not set(drawn) & set(cal.CALIBRATION_SET)
    assert cal.Optim_ordinary_airfoils([*cal.CALIBRATION_SET, "only"]) == ["only"]


def test_a_calibration_runs_every_rung_and_reports_it(stubs, tmp_path, monkeypatch, caplog):
    _world(monkeypatch, lambda k: {**_nodes(k), **_skew_off_defaults(k)})
    caplog.set_level(logging.INFO, logger=_RUNNER)
    with RunStore.open(tmp_path / "cal.sqlite") as store, caplog.at_level(logging.INFO, logger=cal.__name__):
        result = cal.Optim_calibrate(store, (1, 2, 3), ["naca0012", "naca4412"], rung3_cap=10, rung2_cap=5,
                                     workers=1, work_root=tmp_path / "work")
        text = cal.Optim_format_report(store, result)

        assert (result.golden_jmax, result.golden_source) == (140 + cal.GOLDEN_MARGIN, "rung 1")  # jmax* 140
        assert result.recommended == result.golden_jmax
        r2, r3 = result.stages[2].settings, result.stages[3].settings
        assert (r2["gate_skew_max"], r2["rung2_budget"], r2["rung3_budget"]) == (cal.SKEW_GATE, 5, 0)
        assert (r3["golden_jmax"], r3["rung3_budget"]) == (result.golden_jmax, 10)
        labels = {row[0] for row in store._conn.execute("SELECT label FROM settings")}
        assert labels == {"calibration rung 1", "calibration rung 2", "calibration rung 3"}
    logged = [r.getMessage() for r in caplog.records if r.name == cal.__name__]
    per_airfoil = [r.getMessage() for r in caplog.records if r.name == _RUNNER and r.getMessage().startswith("naca")]
    assert per_airfoil == []  # rung 1's skew 0.3 at jmax* is rung 2's work, rung 3's passes its design: no lines
    assert [m.split("  ")[0] for m in logged if m.startswith("-----")] == [
        f"----- {edge} rung {rung} of 3 -----" for rung in (1, 2, 3) for edge in ("START", "END")]
    for finding in ("size ratio one jmax below jmax*", "worst skew at jmax*", "passing skew: ",
                    "nodes saved against the golden run", "Rung-3 cap"):
        assert any(finding in m for m in logged) and finding in text, finding  # one wording, log and report
    assert not any(header in m for m in logged for header in ("n*@cw0", "golden skew", "Plateau replay"))  # tables:
    assert all(header in text for header in ("n*@cw0", "golden skew", "Plateau replay"))  # the report's only
    for heading in ("Rung 1: n*, jmax* and the golden jmax", "Golden jmax = median", "NSTAR_CW_GRID 1 loses nothing",
                    "Campaign estimate at golden jmax 145", "Rung 2: its budget", "Rung-2 budget",
                    "Rung 3: from the golden run", "Plateau replay", "Plateau pick", "Rung-3 cap"):
        assert heading in text, heading
    assert "part" not in text.lower()


@pytest.mark.parametrize("given, rungs, expected", [(150, (1,), (150, "given")),
                                                     (None, (2,), (GOLDEN_JMAX, "constant"))])
def test_the_golden_jmax_comes_from_the_flag_rung_1_or_the_constant(stubs, tmp_path, monkeypatch, given, rungs,
                                                                    expected):
    _world(monkeypatch, _sr_from(140))
    with RunStore.open(tmp_path / "cal.sqlite") as store:
        result = cal.Optim_calibrate(store, rungs, ["naca0012"], golden_jmax=given, rung2_cap=2, workers=1,
                                     work_root=tmp_path / "work")
    assert (result.golden_jmax, result.golden_source) == expected


def test_a_given_golden_jmax_over_rung_1_is_warned(stubs, tmp_path, monkeypatch, caplog):
    _world(monkeypatch, _sr_from(140))
    with RunStore.open(tmp_path / "cal.sqlite") as store, caplog.at_level(logging.WARNING, logger=cal.__name__):
        result = cal.Optim_calibrate(store, (1, 2), ["naca0012"], golden_jmax=150, rung2_cap=2, workers=1,
                                     work_root=tmp_path / "work")
        text = cal.Optim_format_report(store, result)

    assert [r.getMessage() for r in caplog.records if r.name == cal.__name__] == \
        ["golden jmax 150 is given: rungs 2 and 3 run at it, not at rung 1's recommendation"]
    assert (result.golden_jmax, result.recommended) == (150, 145)
    assert "Golden jmax for rungs 2 and 3: 150 (given; rung 1 recommends 145)" in text
    caplog.clear()
    with RunStore.open(tmp_path / "other.sqlite") as store, caplog.at_level(logging.WARNING, logger=cal.__name__):
        cal.Optim_calibrate(store, (2,), ["naca0012"], golden_jmax=150, rung2_cap=2, workers=1,
                            work_root=tmp_path / "work2")
    assert [r for r in caplog.records if r.name == cal.__name__] == []  # no rung 1, nothing overridden


def test_a_dated_calibration_writes_its_report_and_database_once(stubs, tmp_path, monkeypatch):
    _world(monkeypatch, _sr_from(142))
    out = tmp_path / "calibration"  # made on the way
    files = cal.Optim_run_calibration(out, (1,), ["naca0012"], workers=1)
    stamp = date.today().isoformat()

    assert (files.report, files.db) == (out / f"{stamp}.txt", out / f"{stamp}.sqlite")
    report = files.report.read_text(encoding="utf-8")
    assert report == files.text and "naca0012" in report and "Rung 1: n*, jmax*" in report
    with RunStore.open(files.db, read_only=True) as store:
        assert store._conn.execute("SELECT COUNT(*) FROM runs WHERE rung = 'bisect'").fetchone()[0] > 0
    with pytest.raises(FileExistsError, match="already holds a calibration"):  # the day's database is never written over
        cal.Optim_run_calibration(out, (1,), ["naca0012"], workers=1)
    for rungs in ((4,), ()):
        with pytest.raises(ValueError, match="rungs are any of 1, 2, 3"):
            cal.Optim_run_calibration(tmp_path / "other", rungs)
    assert not (tmp_path / "other").exists()
# endregion
