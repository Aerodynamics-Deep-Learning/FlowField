"""
Calibration of the mesh study's ladder: on real c2d meshes, measures the numbers each rung runs on and writes
the report they are chosen from. Like the CST calibration, it is run by hand and recommends; it changes no
constant.

To run calibration:
    `python -m src.datagen.meshing.optimizer.batch.calibrate [--rungs 1,2,3] [--airfoils a,b]
        [--golden-jmax N] [--rung3-cap 300] [--rung2-cap 100] [--workers 6] [--out DIR]`

    Rung 1  every airfoil: n* on its full grid and at the narrowest cluster_width, then jmax*, the smallest
            jmax whose mesh passes the size ratio (a bisection over [JMAX_MIN, JMAX_MAX]); recommends the
            golden jmax, whether n*'s cluster_width grid can shrink to one value, and estimates a campaign's
            wall time from the times it measured
    Rung 2  the calibration set: real ladders with the skew gate tightened, so rung 2 has work; its budget
    Rung 3  the calibration set and a few ordinary airfoils: the golden run, then rung 3 to its cap with no
            plateau; what a downward search saves and costs, and, by replaying those runs, rung 3's
            plateau and cap

Rungs 2 and 3 start from the golden jmax rung 1 recommends when it runs alongside them; a given --golden-jmax
wins over it, with a warning. Each rung runs under its own settings row. The report (`<date>.txt`) and every
run (`<date>.sqlite`, the run store's schema) are written to the calibration folder.
"""

import argparse
import datetime
import json
import logging
import math
import random
import statistics
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import NamedTuple, Optional, Sequence

from src.datagen.geometry.common.constants import DX_MAX
from src.datagen.meshing.optimizer.batch.runner import Optim_load_shapes, Optim_run_airfoils
from src.datagen.meshing.optimizer.common.constants import (
    CLUSTER_WIDTH_MIN, GOLDEN_JMAX, JMAX_MAX, JMAX_MIN, SMOOTHING_DEFAULTS, WORKERS,
)
from src.datagen.meshing.optimizer.common.paths import CALIBRATION_DIR
from src.datagen.meshing.optimizer.common.settings import Optim_build_settings
from src.datagen.meshing.optimizer.evaluate.executor import RunContext
from src.datagen.meshing.optimizer.manifest.store import Optim_git_state, RunStore
from src.datagen.meshing.optimizer.search.ladder import Ladder, LadderOutcome, Optim_run_ladder
from src.datagen.meshing.optimizer.search.mads import Optim_plateau_reached
from src.datagen.meshing.optimizer.search.nstar import Optim_n_star_search

logger = logging.getLogger(__name__)

# The hard cases: the most surface points (s1223, s1210, mid012r), the highest jmax* (e376, goe531) and the
# lowest (naca0040), Emre's worst lift losses (e423, fx69pr281), a thin and a typical section
CALIBRATION_SET = ("s1223", "s1210", "mid012r", "e376", "goe531", "naca0040", "e423", "fx69pr281", "naca0006",
                   "naca4412")

# Rung 3's ordinary airfoils, drawn from the rest: the same ones every run
ORDINARY_COUNT = 5
ORDINARY_SEED = 1

# Rung 2's skew gate: under most golden runs' worst skew (median 0.243), so rung 2 has work
SKEW_GATE = 0.22

# Meshes rung 3 and rung 2 may spend
RUNG3_CAP = 300
RUNG2_CAP = 100

# Golden jmax = median jmax* + this
GOLDEN_MARGIN = 5

# Rung 3's replay: every (window, min_gain) pair, and the most a pair may give up on any airfoil against its
# cap's best
WINDOWS = (10, 15, 20, 30, 50)
GAINS = (0.0025, 0.005, 0.01)
MAX_LOSS = 0.005

# Close to the final best, for the rung-3 cap
WITHIN = 0.01


#region The calibration's climbs

class _Bisect(Ladder):
    """Rung 1: n*, n* at the narrowest cluster_width, then jmax* by bisection on the size ratio alone."""

    def climb(self) -> None:
        if not self._n_star():
            return
        s = self.settings
        grid = {**json.loads(s["nstar_json"]), "cw_grid": 1}
        floor = Optim_n_star_search(self.contour, {**s, "nstar_json": json.dumps(grid)})
        self.notes.update(n_star_floor=floor.n, n_star_floor_s=floor.seconds)
        self.route.append("bisect")
        knobs = {**self.nstar.knobs, **SMOOTHING_DEFAULTS}
        meshed: dict[int, dict] = {}
        lo, hi = JMAX_MIN - 1, JMAX_MAX  # lo fails and hi passes, assumed until meshed
        while hi - lo > 1:
            mid = (lo + hi) // 2
            passes = self._size_ratio_passes(knobs, mid, meshed)
            if passes is None:
                return
            lo, hi = (lo, mid) if passes else (mid, hi)
        if hi not in meshed:  # the top of the range, never meshed
            passes = self._size_ratio_passes(knobs, hi, meshed)
            if not passes:
                if passes is False:
                    self.notes["jmax_star"] = None  # above the range
                return
        self.notes.update(jmax_star=hi, jmax_star_passes=bool(meshed[hi]["passed"]))

    def _size_ratio_passes(self, knobs: dict, jmax: int, meshed: dict) -> Optional[bool]:
        """Meshes `knobs` at `jmax`: whether the size ratio passed; None, noted, if nothing usable meshed."""
        row = meshed[jmax] = self._run({**knobs, "jmax": jmax}, "bisect", len(meshed))
        if row["status"] != "meshed":
            self.notes["bisect_failed"] = f"jmax {jmax}: {row['status']}"
            return None
        return "size_ratio" not in row["failed_gates"]


class _Descend(Ladder):
    """Rung 3: the golden run, then rung 3 from it to the settings' budget (the cap), with no plateau."""

    def climb(self) -> None:
        if not self._n_star():
            return
        golden_knobs = self._golden()
        self._full_search(golden_knobs, parent=self.golden, plateau=None)

#endregion


#region Running

class Stage(NamedTuple):
    """One rung's calibration as it ran."""
    rung: int
    settings: dict
    names: list[str]
    outcomes: list[LadderOutcome]
    seconds: float


class Calibration(NamedTuple):
    stages: dict[int, Stage]
    golden_jmax: int                 # what rungs 2 and 3 ran at
    golden_source: str               # 'given', 'rung 1' or 'constant'
    recommended: Optional[int]       # rung 1's golden jmax, used or not


def Optim_calibrate(store: RunStore, rungs: Sequence[int] = (1, 2, 3), names: Optional[Sequence[str]] = None,
                    golden_jmax: Optional[int] = None, rung3_cap: int = RUNG3_CAP, rung2_cap: int = RUNG2_CAP,
                    workers: int = WORKERS, work_root=None) -> Calibration:
    """
    Calibrates the rungs asked for, every run recorded in `store`.

    Args:
        store (RunStore): Where every rung's runs go
        rungs (Sequence[int]): Any of 1, 2, 3
        names (Sequence[str] | None): The airfoils for every rung; each rung's own if None (rung 1 every stored
            airfoil, rung 2 the calibration set, rung 3 the set and the ordinary airfoils)
        golden_jmax (int | None): Rungs 2 and 3's golden jmax, winning over rung 1's recommendation (with a
            warning); rung 1's if it runs, else `GOLDEN_JMAX`, when None
        rung3_cap (int): Rung 3's budget while it is calibrated
        rung2_cap (int): Rung 2's budget while it is calibrated
        workers (int): Worker processes
        work_root (str | Path | None): Where runs mesh; a temporary folder if None

    Returns:
        Calibration: Each rung's outcomes, and the golden jmax rungs 2 and 3 ran at
    """
    if work_root is None:
        with tempfile.TemporaryDirectory(prefix="calibration_") as work:
            return Optim_calibrate(store, rungs, names, golden_jmax, rung3_cap, rung2_cap, workers, work)

    if golden_jmax is not None and 1 in rungs and {2, 3} & set(rungs):
        logger.warning("golden jmax %d is given: rungs 2 and 3 run at it, not at rung 1's recommendation",
                       golden_jmax)
    done: dict[int, Stage] = {}
    run = partial(_run_stage, store=store, work_root=Path(work_root), workers=workers)
    if 1 in rungs:
        done[1] = run(1, partial(Optim_run_ladder, ladder_type=_Bisect), Optim_load_shapes(names),
                      Optim_build_settings())
    recommended = Optim_golden_from(Optim_bisections(store, done[1])) if 1 in done else None
    if golden_jmax is not None:
        source = "given"
    elif recommended is not None:
        golden_jmax, source = recommended, "rung 1"
    else:
        golden_jmax, source = GOLDEN_JMAX, "constant"
    if 2 in rungs:
        settings = Optim_build_settings(golden_jmax=golden_jmax, gate_skew_max=SKEW_GATE, rung2_budget=rung2_cap,
                                        rung3_budget=0)
        done[2] = run(2, Optim_run_ladder, Optim_load_shapes(names or CALIBRATION_SET), settings)
    if 3 in rungs:
        chosen = names or [*CALIBRATION_SET, *Optim_ordinary_airfoils([n for n, _ in Optim_load_shapes()])]
        settings = Optim_build_settings(golden_jmax=golden_jmax, rung3_budget=rung3_cap)
        done[3] = run(3, partial(Optim_run_ladder, ladder_type=_Descend), Optim_load_shapes(chosen), settings)
    return Calibration(done, golden_jmax, source, recommended)


def _run_stage(rung: int, task, airfoils: list, settings: dict, *, store: RunStore, work_root: Path,
               workers: int) -> Stage:
    store.ensure_settings(settings, label=f"calibration rung {rung}")
    ctx = RunContext.create(settings, work_root / f"rung{rung}")
    t0 = time.perf_counter()
    outcomes = Optim_run_airfoils(task, airfoils, ctx, store, workers)
    return Stage(rung, settings, [name for name, _ in airfoils], outcomes, time.perf_counter() - t0)


def Optim_ordinary_airfoils(names: Sequence[str]) -> list[str]:
    """`ORDINARY_COUNT` airfoils outside the calibration set, the same ones for the same names."""
    rest = sorted(set(names) - set(CALIBRATION_SET))
    return sorted(random.Random(ORDINARY_SEED).sample(rest, min(ORDINARY_COUNT, len(rest))))

#endregion


#region Rung 1: n*, jmax* and the golden jmax

class Bisection(NamedTuple):
    name: str
    status: str
    n_star: Optional[int]
    n_star_floor: Optional[int]       # at the narrowest cluster_width
    cluster_width: Optional[float]    # where n* landed
    jmax_star: Optional[int]          # None: above JMAX_MAX, or not found (failed)
    passes_all: Optional[bool]        # every gate passes at jmax*
    failed: Optional[str]             # why there is no jmax*, if not above the range
    mesh_s: dict[int, float]          # each bisection mesh's wall time, by its jmax
    n_star_s: Optional[float]
    n_star_floor_s: Optional[float]
    seconds: Optional[float]          # the whole ladder


def Optim_bisections(store: RunStore, stage: Stage) -> list[Bisection]:
    out = []
    for name, outcome in zip(stage.names, stage.outcomes):
        ladder, notes = store.ladder(outcome.ladder_id), outcome.notes or {}
        knobs = json.loads(ladder["n_star_knobs_json"]) if ladder["n_star_knobs_json"] else None
        meshes = {r["jmax"]: r["wall_s"] for r in store.runs(outcome.ladder_id) if r["status"] == "meshed"}
        out.append(Bisection(
            name, outcome.status, ladder["n_star"], notes.get("n_star_floor"), knobs and knobs["cluster_width"],
            notes.get("jmax_star"), notes.get("jmax_star_passes"), notes.get("bisect_failed") or _why(outcome),
            meshes, ladder["n_star_s"], notes.get("n_star_floor_s"), ladder["wall_s"]))
    return out


def Optim_golden_from(rows: list[Bisection]) -> Optional[int]:
    """The golden jmax: the median jmax* (rounded up) + `GOLDEN_MARGIN`; None with no jmax* at all."""
    found = [b.jmax_star for b in rows if b.jmax_star is not None]
    return math.ceil(statistics.median(found)) + GOLDEN_MARGIN if found else None


class CampaignEstimate(NamedTuple):
    """A campaign's serial seconds at a golden jmax, from rung 1's measured times."""
    airfoils: int
    n_star_s: float        # n* on its full grid, summed
    n_star_floor_s: float  # n* at the narrowest cluster_width only, summed
    mesh_s: float          # the golden run and its bumps, summed
    mesh_median_s: float   # one golden-size mesh
    bumps: int


def Optim_campaign_estimate(rows: list[Bisection], golden_jmax: int) -> Optional[CampaignEstimate]:
    """
    What a campaign would spend per airfoil: n*, a golden mesh and its bumps (to jmax*, when that is higher),
    each mesh timed by the airfoil's own bisection mesh nearest the golden jmax. Campaign meshes all sit near the
    golden size, so no scaling law is needed.

    Args:
        rows (list[Bisection]): Rung 1's airfoils; those without n* or a mesh are left out
        golden_jmax (int): The golden jmax the campaign would run at

    Returns:
        CampaignEstimate | None: The sums, or None with no airfoil to estimate from
    """
    timed = [b for b in rows if b.n_star is not None and b.mesh_s]
    if not timed:
        return None
    per_mesh, bumps = [], 0
    n_star_s = floor_s = mesh_s = 0.0
    for b in timed:
        nearest = min(b.mesh_s, key=lambda jmax: (abs(jmax - golden_jmax), -jmax))
        extra = max(0, (b.jmax_star or golden_jmax) - golden_jmax)
        per_mesh.append(b.mesh_s[nearest])
        n_star_s, floor_s = n_star_s + (b.n_star_s or 0.0), floor_s + (b.n_star_floor_s or 0.0)
        mesh_s, bumps = mesh_s + (1 + extra) * b.mesh_s[nearest], bumps + extra
    return CampaignEstimate(len(timed), n_star_s, floor_s, mesh_s, statistics.median(per_mesh), bumps)


def _why(outcome: LadderOutcome) -> Optional[str]:
    """Why a ladder has no result: its error's first line, or n* accepting nothing."""
    if outcome.error is not None:
        lines = str(outcome.error).splitlines()
        return f"aborted: {type(outcome.error).__name__}: {lines[0] if lines else ''}"
    if outcome.route == ["nstar"]:
        return "n* accepted nothing"
    return None

#endregion


#region Rung 2: its budget

class Smoothing(NamedTuple):
    name: str
    status: str
    route: list[str]
    golden_skew: Optional[float]
    golden_failed: Optional[list[str]]
    meshes: Optional[int]           # rung 2's c2d runs up to its first pass, or all of them without one
    passed: bool                    # rung 2 found a pass
    skew: Optional[float]           # the pass's worst skew, or the lowest rung 2 reached without one
    failed: Optional[str]


def Optim_smoothings(store: RunStore, stage: Stage) -> list[Smoothing]:
    out = []
    for name, outcome in zip(stage.names, stage.outcomes):
        ladder, rows = store.ladder(outcome.ladder_id), store.runs(outcome.ladder_id)
        golden = next((r for r in rows if r["rung"] == "golden"), None)
        route = json.loads(ladder["route"]) if ladder["route"] else []
        meshes = passed = skew = None
        if "smoothing" in route:
            spent, passed = 0, False
            for row in (r for r in rows if r["rung"] == "smoothing"):
                spent += row["entry_flag"] is not None
                if row["passed"]:
                    passed, skew = True, row["q_skew_max"]
                    break
            meshes = spent
            if not passed:
                skews = [r["q_skew_max"] for r in rows if r["rung"] == "smoothing" and r["q_skew_max"] is not None]
                skew = min(skews, default=None)
        failed = json.loads(golden["failed_gates"]) if golden and golden["failed_gates"] else None
        out.append(Smoothing(name, outcome.status, route, golden and golden["q_skew_max"], failed, meshes,
                             bool(passed), skew, _why(outcome)))
    return out


def Optim_rung2_budget(rows: list[Smoothing]) -> Optional[int]:
    """The most c2d runs rung 2 took to a first pass, rounded up to 10."""
    needed = [s.meshes for s in rows if s.passed]
    return _round_up(max(needed)) if needed else None

#endregion


#region Rung 3: to its cap, and its replay

class Descent(NamedTuple):
    name: str
    status: str
    n_star: Optional[int]
    golden_nodes: Optional[int]
    golden_passed: bool
    nodes: list[Optional[int]]   # the best's nodes after each c2d run of rung 3, the start first
    seconds: list[float]         # from rung 3's first run to the end of each c2d run, 0 first
    failed: Optional[str]

    @property
    def final(self) -> Optional[int]:
        return self.nodes[-1] if self.nodes else None


def Optim_descents(store: RunStore, stage: Stage) -> list[Descent]:
    out = []
    for name, outcome in zip(stage.names, stage.outcomes):
        ladder, rows = store.ladder(outcome.ladder_id), store.runs(outcome.ladder_id)
        golden = next((r for r in rows if r["rung"] == "golden"), None)
        passed = bool(golden and golden["passed"])
        nodes, seconds = Optim_curve([r for r in rows if r["rung"] == "mads"],
                                     golden["num_nodes"] if passed else None)
        out.append(Descent(name, outcome.status, ladder["n_star"], golden and golden["num_nodes"], passed, nodes,
                           seconds, _why(outcome)))
    return out


def Optim_curve(rows: list[dict], start: Optional[int]) -> tuple[list[Optional[int]], list[float]]:
    """
    A search's best after each of its c2d runs, as `Optim_plateau_reached` reads it, and the seconds to each.

    Args:
        rows (list[dict]): The search's runs, in order
        start (int | None): The best's nodes before the search (a passed golden run's), or None

    Returns:
        tuple[list[int | None], list[float]]: The best's nodes, the start first; seconds from the first
            run's start to each c2d run's end, 0 first
    """
    best, nodes, seconds = start, [start], [0.0]
    t0 = _time(rows[0]["started_at"]) if rows else None
    for row in rows:
        if row.get("passed") and (best is None or row["num_nodes"] < best):
            best = row["num_nodes"]
        if row.get("entry_flag") is not None:
            nodes.append(best)
            seconds.append((_time(row["finished_at"]) - t0).total_seconds())
    return nodes, seconds


def Optim_replay(nodes: list[Optional[int]], window: int, min_gain: float) -> int:
    """The c2d run a plateau of (window, min_gain) would have stopped after; the last one if never."""
    for k in range(1, len(nodes)):
        if Optim_plateau_reached(nodes[:k + 1], window, min_gain):
            return k
    return len(nodes) - 1


def Optim_meshes_within(nodes: list[Optional[int]], share: float) -> Optional[int]:
    """The c2d runs a search took to get within `share` of its final best; None if it never passed."""
    final = nodes[-1] if nodes else None
    if final is None:
        return None
    return next(k for k, n in enumerate(nodes) if n is not None and n <= final * (1 + share))


class PlateauRow(NamedTuple):
    window: int
    min_gain: float
    meshes: int          # summed over the airfoils
    seconds: float
    worst_loss: float    # the most any airfoil's best ends above its cap's best, as a share
    mean_loss: float


def Optim_plateau_table(rows: list[Descent]) -> list[PlateauRow]:
    table = []
    for window in WINDOWS:
        for gain in GAINS:
            meshes, seconds, losses = 0, 0.0, []
            for d in rows:
                k = Optim_replay(d.nodes, window, gain)
                meshes, seconds = meshes + k, seconds + d.seconds[k]
                if d.final is not None:
                    losses.append(d.nodes[k] / d.final - 1)
            table.append(PlateauRow(window, gain, meshes, seconds, max(losses, default=0.0),
                                    statistics.fmean(losses) if losses else 0.0))
    return table


def Optim_pick_plateau(table: list[PlateauRow]) -> Optional[PlateauRow]:
    """The fewest meshes among the pairs giving up at most `MAX_LOSS` on every airfoil; ties to the cautious."""
    qualified = [p for p in table if p.worst_loss <= MAX_LOSS]
    return min(qualified, key=lambda p: (p.meshes, -p.window, p.min_gain)) if qualified else None


def Optim_rung3_cap(rows: list[Descent]) -> Optional[int]:
    """The most c2d runs any airfoil took to get within `WITHIN` of its best, rounded up to 10."""
    needed = [m for m in (Optim_meshes_within(d.nodes, WITHIN) for d in rows) if m is not None]
    return _round_up(max(needed)) if needed else None

#endregion


#region The report

def Optim_format_report(store: RunStore, cal: Calibration) -> str:
    """The calibration report: a header, then each rung that was calibrated."""
    commit, dirty = Optim_git_state()
    out = ["Mesh-study calibration",
           f"{datetime.date.today().isoformat()}, code {commit[:12]}{' (dirty)' if dirty else ''}, DX_MAX {DX_MAX:g}"]
    if cal.stages.keys() & {2, 3}:
        overridden = cal.golden_source == "given" and cal.recommended is not None
        out.append(f"Golden jmax for rungs 2 and 3: {cal.golden_jmax} ({cal.golden_source}"
                   + (f"; rung 1 recommends {cal.recommended})" if overridden else ")"))
    for stage in cal.stages.values():
        count = len(stage.names)
        out.append(f"Rung {stage.rung}: settings {stage.settings['settings_id']}, {count} "
                   f"airfoil{'s' * (count != 1)}, {stage.seconds / 60:.1f} min")
    out.append("")

    if 1 in cal.stages:
        out += _format_rung1(Optim_bisections(store, cal.stages[1]), cal)
    if 2 in cal.stages:
        out += _format_rung2(Optim_smoothings(store, cal.stages[2]))
    if 3 in cal.stages:
        jmax_star = {b.name: b.jmax_star for b in Optim_bisections(store, cal.stages[1])} if 1 in cal.stages else {}
        out += _format_rung3(Optim_descents(store, cal.stages[3]), jmax_star, cal)
    return "\n".join(out).rstrip() + "\n"


def _format_rung1(rows: list[Bisection], cal: Calibration) -> list[str]:
    out = [f"Rung 1: n*, jmax* and the golden jmax (jmax*: the smallest jmax whose mesh passes the size ratio, "
           f"bisected over [{JMAX_MIN}, {JMAX_MAX}])",
           f"  {'airfoil':22s} {'n*':>5} {'n*@cw0':>6} {'cw':>6} {'jmax*':>6} {'all':>4} {'meshes':>6} "
           f"{'n* s':>6} {'s':>6}  note"]
    for b in rows:
        jmax = "-" if b.jmax_star is None else (f"<={b.jmax_star}" if b.jmax_star == JMAX_MIN else str(b.jmax_star))
        if b.jmax_star is None and b.failed is None:
            jmax = f">{JMAX_MAX}"
        out.append(f"  {b.name:22s} {_i(b.n_star):>5} {_i(b.n_star_floor):>6} {_f(b.cluster_width, '.3f'):>6} "
                   f"{jmax:>6} {_yes(b.passes_all):>4} {len(b.mesh_s):>6} {_f(b.n_star_s, '.1f'):>6} "
                   f"{_f(b.seconds, '.0f'):>6}  {b.failed or ''}")
    found = [b.jmax_star for b in rows if b.jmax_star is not None]
    above = sum(b.jmax_star is None and b.failed is None for b in rows)
    out.append("")
    if found:
        golden = Optim_golden_from(rows)
        out.append(f"  jmax*: min {min(found)}, median {statistics.median(found):g}, max {max(found)} "
                   f"({len(found)} found, {above} above {JMAX_MAX}, {sum(b.failed is not None for b in rows)} failed)")
        bumps = sorted(j - golden for j in found if j > golden)
        out.append(f"  Golden jmax = median rounded up + {GOLDEN_MARGIN} = {golden}: {len(found) - len(bumps)}/"
                   f"{len(found)} pass the size ratio there"
                   + (f"; the rest need {', '.join(map(str, bumps))} bumps" if bumps else ""))
        out.append(f"  jmax* also passes every other gate on {sum(bool(b.passes_all) for b in rows)}/{len(found)}")
    with_n = [b for b in rows if b.n_star is not None]
    if with_n:
        narrowest = sum(b.cluster_width == CLUSTER_WIDTH_MIN for b in with_n)
        same = sum(b.n_star_floor == b.n_star for b in with_n)
        out.append(f"  cluster_width: n* landed on the narrowest ({CLUSTER_WIDTH_MIN:g}) for "
                   f"{narrowest}/{len(with_n)}; n* at the narrowest alone equals the full grid's for "
                   f"{same}/{len(with_n)}" + (": NSTAR_CW_GRID 1 loses nothing" if same == len(with_n) else ""))
        out.append(f"  n* time: full grid median {statistics.median(b.n_star_s for b in with_n):.1f} s, "
                   f"narrowest only median {statistics.median(b.n_star_floor_s for b in with_n):.2f} s")
    estimate = Optim_campaign_estimate(rows, cal.golden_jmax)
    if estimate:
        stage = cal.stages[1]
        speedup = sum(b.seconds or 0 for b in rows) / stage.seconds if stage.seconds > 0 else 1.0
        serial, floor = estimate.n_star_s + estimate.mesh_s, estimate.n_star_floor_s + estimate.mesh_s
        out.append(f"  Campaign estimate at golden jmax {cal.golden_jmax}, from these times: {estimate.airfoils} "
                   f"airfoils, a golden mesh {estimate.mesh_median_s:.1f} s (median), {estimate.bumps} bumps")
        out.append(f"    {serial / 60:.1f} min serial, about {serial / speedup / 60:.1f} min at this run's "
                   f"{speedup:.1f}x; with NSTAR_CW_GRID 1, {floor / 60:.1f} and {floor / speedup / 60:.1f} min")
    out.append("")
    return out


def _format_rung2(rows: list[Smoothing]) -> list[str]:
    out = [f"Rung 2: its budget, under a skew gate of {SKEW_GATE:g}",
           f"  {'airfoil':22s} {'golden skew':>11} {'golden failed':22s} {'route':32s} {'meshes':>6} {'skew':>6}  note"]
    for s in rows:
        meshes = "-" if s.meshes is None else (str(s.meshes) if s.passed else f"never ({s.meshes})")
        out.append(f"  {s.name:22s} {_f(s.golden_skew, '.3f'):>11} {','.join(s.golden_failed or []) or '-':22s} "
                   f"{' > '.join(s.route) or '-':32s} {meshes:>6} {_f(s.skew, '.3f'):>6}  {s.failed or ''}")
    routed = [s for s in rows if s.meshes is not None]
    passed = [s.meshes for s in routed if s.passed]
    out.append("")
    out.append(f"  Routed to rung 2: {len(routed)}; passed: {len(passed)}"
               + (f" (meshes to a first pass: median {statistics.median(passed):g}, max {max(passed)})"
                  if passed else "")
               + f"; never: {len(routed) - len(passed)}")
    out.append(f"  Rung-2 budget (the most meshes to a first pass, rounded up to 10): {_i(Optim_rung2_budget(rows))}")
    out.append("")
    return out


def _format_rung3(rows: list[Descent], jmax_star: dict, cal: Calibration) -> list[str]:
    stage = cal.stages[3]
    out = [f"Rung 3: from the golden run (jmax {cal.golden_jmax}) to a cap of {stage.settings['rung3_budget']} "
           "c2d runs, no plateau",
           "  saving: the best against the golden run's nodes; by jmax*: how much of it meshing at jmax* alone "
           "gives (needs rung 1)",
           f"  {'airfoil':22s} {'golden':>8} {'best':>8} {'saving':>7} {'by jmax*':>8} {'to 1%':>6} {'to best':>7} "
           f"{'s to 1%':>8} {'s best':>7}  note"]
    for d in rows:
        within, best_at = Optim_meshes_within(d.nodes, WITHIN), Optim_meshes_within(d.nodes, 0.0)
        saving = at_jmax = None
        if d.golden_passed and d.final is not None:
            saving = 1 - d.final / d.golden_nodes
            j = jmax_star.get(d.name)
            if j is not None and d.n_star is not None:
                at_jmax = max(0.0, 1 - (d.n_star - 1) * j / d.golden_nodes)
        out.append(f"  {d.name:22s} {_i(d.golden_nodes):>8} {_i(d.final):>8} {_pct(saving):>7} {_pct(at_jmax):>8} "
                   f"{_i(within):>6} {_i(best_at):>7} {_f(within is not None and d.seconds[within], '.0f'):>8} "
                   f"{_f(best_at is not None and d.seconds[best_at], '.0f'):>7}  "
                   f"{d.failed or ('' if d.golden_passed else 'golden failed')}")
    table = Optim_plateau_table(rows)
    out += ["", "  Plateau replay: where each (window, min_gain) would have stopped every search above, summed; "
                "loss: how far its best ends above the cap's",
            f"  {'window':>6} {'gain':>6} {'meshes':>7} {'seconds':>8} {'worst loss':>10} {'mean loss':>9}"]
    for p in table:
        out.append(f"  {p.window:>6} {_pct(p.min_gain, 2):>6} {p.meshes:>7} {p.seconds:>8.0f} "
                   f"{_pct(p.worst_loss, 2):>10} {_pct(p.mean_loss, 2):>9}")
    pick = Optim_pick_plateau(table)
    out.append("")
    out.append(f"  Plateau pick (fewest meshes giving up at most {_pct(MAX_LOSS, 1)} on every airfoil): "
               + (f"window {pick.window}, min_gain {pick.min_gain:g}, {pick.meshes} meshes, {pick.seconds / 60:.1f} min"
                  if pick else "none qualifies"))
    out.append(f"  Rung-3 cap (the most meshes any airfoil took to within {_pct(WITHIN, 0)} of its best, rounded up "
               f"to 10): {_i(Optim_rung3_cap(rows))}")
    out.append("")
    return out


def _time(stamp: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(stamp)


def _round_up(n: int) -> int:
    return max(10, math.ceil(n / 10) * 10)


def _i(v) -> str:
    return "-" if v is None else f"{v}"


def _f(v, spec: str) -> str:
    return "-" if v is None or v is False else format(v, spec)


def _pct(v, digits: int = 1) -> str:
    return "-" if v is None else f"{100 * v:.{digits}f}%"


def _yes(v) -> str:
    return "-" if v is None else ("yes" if v else "no")

#endregion


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rungs", default="1,2,3", help="rungs to calibrate, of 1,2,3 (default all)")
    parser.add_argument("--airfoils", help="comma-separated names for every rung (default: each rung's own)")
    parser.add_argument("--golden-jmax", type=int,
                        help="rungs 2 and 3's golden jmax, over rung 1's (default: rung 1's, else the constant)")
    parser.add_argument("--rung3-cap", type=int, default=RUNG3_CAP,
                        help=f"rung 3's budget while it is calibrated (default {RUNG3_CAP})")
    parser.add_argument("--rung2-cap", type=int, default=RUNG2_CAP,
                        help=f"rung 2's budget while it is calibrated (default {RUNG2_CAP})")
    parser.add_argument("--workers", type=int, default=WORKERS, help=f"worker processes (default {WORKERS})")
    parser.add_argument("--out", type=Path, help="folder for the report and database (default: the calibration folder)")
    args = parser.parse_args(argv)

    try:
        rungs = sorted({int(r) for r in args.rungs.split(",")})
    except ValueError:
        parser.error(f"--rungs takes numbers from 1,2,3, not {args.rungs!r}")
    if not rungs or not set(rungs) <= {1, 2, 3}:
        parser.error(f"--rungs takes numbers from 1,2,3, not {args.rungs!r}")
    names = [n.strip() for n in args.airfoils.split(",")] if args.airfoils else None
    out = args.out or CALIBRATION_DIR
    stamp = datetime.date.today().isoformat()
    db, report = out / f"{stamp}.sqlite", out / f"{stamp}.txt"
    if db.exists():
        parser.error(f"{db} already holds a calibration; pass another --out")

    out.mkdir(parents=True, exist_ok=True)
    with RunStore.open(db) as store:
        cal = Optim_calibrate(store, rungs, names, args.golden_jmax, args.rung3_cap, args.rung2_cap, args.workers)
        text = Optim_format_report(store, cal)
    report.write_text(text, encoding="utf-8", newline="\n")
    print(text)
    print(f"Written to {report} and {db}")


if __name__ == "__main__":
    main()
