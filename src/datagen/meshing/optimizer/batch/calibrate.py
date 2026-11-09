"""
Calibration of the mesh study's ladder: on real c2d meshes, measures the numbers each rung runs on and writes
the report they are chosen from. Like the CST calibration, it is run by hand and recommends; it changes no
constant. Run from the command line by `scripts/datagen/meshing/calibrate.py`.

    Rung 1  every airfoil: n* on its full grid and at the narrowest cluster_width, then jmax*, the smallest
            jmax whose mesh passes the size ratio (a bisection over jmax's search range); recommends the
            golden jmax, whether n*'s cluster_width grid can shrink to one value, and estimates a campaign's
            wall time from the times it measured
    Rung 2  the airfoils rung 1 found over a shape gate at jmax* (the calibration set without rung 1): real
            ladders, which climb to rung 2; its budget
    Rung 3  the calibration set and a few ordinary airfoils: the golden run, then rung 3 to its cap with no
            plateau; what a downward search saves and costs, and, by replaying those runs, rung 3's
            plateau and cap

Rungs 2 and 3 start from the golden jmax rung 1 recommends when it runs alongside them; a given golden jmax
wins over it, with a warning. Each rung runs under its own settings row, a run's settings over the constants
(src/datagen/constants.py), and the calibration's own choices are its `Method`. The report (`<date>.txt`) and
every run (`<date>.sqlite`, the run store's schema) are written to the folder it is given (the repo's is
`CALIBRATION_DIR`).
"""

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

from src.datagen.constants import (
    CAL_GAINS, CAL_GOLDEN_MARGIN, CAL_MAX_LOSS, CAL_ORDINARY_COUNT, CAL_RUNG2_CAP, CAL_RUNG3_CAP, CAL_SET, CAL_WINDOWS,
    CAL_WITHIN, DX_MAX, GOLDEN_JMAX, SLOT_MEMORY_CAP_MB, WATCHDOG_S, WORKERS,
)
from src.datagen.meshing.optimizer.batch.runner import Optim_load_shapes, Optim_run_airfoils
from src.datagen.meshing.optimizer.common.settings import Optim_build_settings, Optim_shape_version
from src.datagen.meshing.optimizer.common.space import Knob, Optim_knobs_of
from src.datagen.meshing.optimizer.evaluate.context import RunContext
from src.datagen.meshing.optimizer.manifest.store import Optim_git_state, RunStore
from src.datagen.meshing.optimizer.search.events import LadderOutcome, Optim_failure_values, Optim_ladder_line
from src.datagen.meshing.optimizer.search.ladder import SHAPE_GATES, Ladder, Optim_run_ladder
from src.datagen.meshing.optimizer.search.mads import Optim_plateau_reached
from src.datagen.meshing.optimizer.search.nstar import Optim_n_star_search

logger = logging.getLogger(__name__)

# Rung 3's ordinary airfoils are drawn with this seed: the same ones every run
ORDINARY_SEED = 1


class Method(NamedTuple):
    """The calibration's own choices: src/datagen/constants.py's CAL_ values, unless a run sets them."""
    set: tuple = CAL_SET                   # the hard cases
    ordinary_count: int = CAL_ORDINARY_COUNT
    golden_margin: int = CAL_GOLDEN_MARGIN
    windows: tuple = CAL_WINDOWS           # rung 3's replay
    gains: tuple = CAL_GAINS
    max_loss: float = CAL_MAX_LOSS
    within: float = CAL_WITHIN

    @classmethod
    def of(cls, settings: dict) -> "Method":
        """The method a run's settings make: each `cal_<field>` over its constant."""
        return cls(**{f: settings[f"cal_{f}"] for f in cls._fields if f"cal_{f}" in settings})


# The settings the calibration reads of its own: the method's, and the caps rungs 2 and 3 run to
CAL_KEYS = frozenset({*(f"cal_{f}" for f in Method._fields), "cal_rung2_cap", "cal_rung3_cap"})

# A rung's start and end in the log
_RUNG_START = "----- START rung {} of 3 -----"
_RUNG_END = "----- END rung {} of 3 -----"


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
        knobs = {**self.nstar.knobs, **json.loads(s["smoothing_json"])}
        meshed: dict[int, dict] = {}
        jmax = _knob(s, "jmax")
        lo, hi = jmax.lo - 1, jmax.hi  # lo fails and hi passes, assumed until meshed
        while hi - lo > 1:
            mid = (lo + hi) // 2
            passes = self._size_ratio_passes(knobs, mid, meshed)
            if passes is None:
                return
            lo, hi = (lo, mid) if passes else (mid, hi)
        if hi not in meshed:  # the top of the range, never meshed
            passes = self._size_ratio_passes(knobs, hi, meshed)
            if not passes:
                if passes is False:  # above the range
                    self.notes.update(jmax_star=None, top_failed=f"no jmax* in [{jmax.lo}, {hi}]: at jmax {hi}, "
                                                                 f"{Optim_failure_values(meshed[hi], s)}")
                return
        self.notes.update(jmax_star=hi, jmax_star_passes=bool(meshed[hi]["passed"]))

    def _size_ratio_passes(self, knobs: dict, jmax: int, meshed: dict) -> Optional[bool]:
        """Meshes `knobs` at `jmax`: whether the size ratio passed; None, noted, if nothing usable meshed."""
        row = meshed[jmax] = self._run({**knobs, "jmax": jmax}, "bisect", len(meshed))
        if row["status"] != "meshed":
            self.notes["bisect_failed"] = f"jmax {jmax}: {row['status']}"
            return None
        return "size_ratio" not in row["failed_gates"]


def _bisect_edge(outcome: LadderOutcome) -> Optional[str]:
    """Rung 1's log line for an airfoil without jmax*; jmax* failing a shape gate is rung 2's work, so no line."""
    notes = outcome.notes or {}
    if notes.get("bisect_failed"):
        return f"bisection stopped at {notes['bisect_failed']}"
    return notes.get("top_failed") or _why(outcome)


def _knob(settings: dict, name: str) -> Knob:
    """A knob's bounds as a settings row recorded them."""
    return next(k for k in Optim_knobs_of(settings) if k.name == name)


class _Descend(Ladder):
    """Rung 3: the golden run, then rung 3 from it to the settings' budget (the cap), with no plateau."""

    def climb(self) -> None:
        if not self._n_star():
            return
        golden_knobs = self._golden()
        self._full_search(golden_knobs, parent=self.golden, plateau=None)


def _descend_edge(store: RunStore, settings: dict, outcome: LadderOutcome) -> Optional[str]:
    """Rung 3's log line, only for an airfoil without a pass: every one reaches rung 3 here by design."""
    if outcome.status != "failed":
        return None
    return Optim_ladder_line(outcome, store.runs(outcome.ladder_id) if outcome.ladder_id else [], settings)

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
    method: Method = Method()


class CalibrationFiles(NamedTuple):
    report: Path                     # <date>.txt
    db: Path                         # <date>.sqlite, every run
    text: str                        # the report


def Optim_run_calibration(out_dir, rungs: Sequence[int] = (1, 2, 3), names: Optional[Sequence[str]] = None,
                          golden_jmax: Optional[int] = None, rung3_cap: int = CAL_RUNG3_CAP,
                          rung2_cap: int = CAL_RUNG2_CAP, workers: int = WORKERS, geometry_db=None, work_root=None,
                          *, overrides: Optional[dict] = None, method: Method = Method(),
                          watchdog_s: float = WATCHDOG_S, cap_mb: int = SLOT_MEMORY_CAP_MB) -> CalibrationFiles:
    """
    A dated calibration: every run into `<out_dir>/<date>.sqlite`, the report into `<out_dir>/<date>.txt`.

    Args:
        out_dir (str | Path): The folder for both, made if missing (the repo's is `CALIBRATION_DIR`)
        rungs, names, golden_jmax, rung3_cap, rung2_cap, workers, geometry_db, work_root, overrides, method,
            watchdog_s, cap_mb: As `Optim_calibrate`

    Returns:
        CalibrationFiles: Where the report and the database went, and the report

    Raises:
        ValueError: If `rungs` is empty or holds anything but 1, 2, 3
        FileExistsError: If the folder already holds today's database, before anything runs
    """
    if not rungs or not set(rungs) <= {1, 2, 3}:
        raise ValueError(f"rungs are any of 1, 2, 3, not {list(rungs)}")
    out_dir = Path(out_dir)
    stamp = datetime.date.today().isoformat()
    db, report = out_dir / f"{stamp}.sqlite", out_dir / f"{stamp}.txt"
    if db.exists():
        raise FileExistsError(f"{db} already holds a calibration; give another folder")
    out_dir.mkdir(parents=True, exist_ok=True)
    with RunStore.open(db) as store:
        cal = Optim_calibrate(store, sorted(set(rungs)), names, golden_jmax, rung3_cap, rung2_cap, workers, work_root,
                              geometry_db, overrides=overrides, method=method, watchdog_s=watchdog_s, cap_mb=cap_mb)
        text = Optim_format_report(store, cal)
    report.write_text(text, encoding="utf-8", newline="\n")
    return CalibrationFiles(report, db, text)


def Optim_calibrate(store: RunStore, rungs: Sequence[int] = (1, 2, 3), names: Optional[Sequence[str]] = None,
                    golden_jmax: Optional[int] = None, rung3_cap: int = CAL_RUNG3_CAP, rung2_cap: int = CAL_RUNG2_CAP,
                    workers: int = WORKERS, work_root=None, geometry_db=None, *, overrides: Optional[dict] = None,
                    method: Method = Method(), watchdog_s: float = WATCHDOG_S,
                    cap_mb: int = SLOT_MEMORY_CAP_MB) -> Calibration:
    """
    Calibrates the rungs asked for, every run recorded in `store`.

    Args:
        store (RunStore): Where every rung's runs go
        rungs (Sequence[int]): Any of 1, 2, 3
        names (Sequence[str] | None): The airfoils for every rung; each rung's own if None (rung 1 every stored
            airfoil, rung 2 those rung 1 found over a shape gate at jmax* or, without rung 1, the calibration set,
            rung 3 the set and the ordinary airfoils)
        golden_jmax (int | None): Rungs 2 and 3's golden jmax, winning over rung 1's recommendation (with a
            warning); rung 1's if it runs, else `GOLDEN_JMAX`, when None
        rung3_cap (int): Rung 3's budget while it is calibrated
        rung2_cap (int): Rung 2's budget while it is calibrated
        workers (int): Worker processes
        work_root (str | Path | None): Where runs mesh; a temporary folder if None
        geometry_db (str | Path | None): The geometry manifest the airfoils come from, the repo's own if None
        overrides (dict | None): The run's settings for every rung's settings row (`Optim_build_settings`), but
            the golden jmax and the budgets, which the calibration sets
        method (Method): The calibration's own choices
        watchdog_s (float): A slot silent this long is killed as hung
        cap_mb (int): Each slot's memory cap

    Returns:
        Calibration: Each rung's outcomes, the golden jmax rungs 2 and 3 ran at, and the method
    """
    if work_root is None:
        with tempfile.TemporaryDirectory(prefix="calibration_") as work:
            return Optim_calibrate(store, rungs, names, golden_jmax, rung3_cap, rung2_cap, workers, work, geometry_db,
                                   overrides=overrides, method=method, watchdog_s=watchdog_s, cap_mb=cap_mb)

    overrides = dict(overrides or {})
    if golden_jmax is not None and 1 in rungs and {2, 3} & set(rungs):
        logger.warning("golden jmax %d is given: rungs 2 and 3 run at it, not at rung 1's recommendation",
                       golden_jmax)
    load = partial(Optim_load_shapes, db=geometry_db, shape_version=Optim_shape_version(overrides))
    done: dict[int, Stage] = {}
    run = partial(_run_stage, store=store, work_root=Path(work_root), workers=workers, first=min(rungs),
                  watchdog_s=watchdog_s, cap_mb=cap_mb)
    if 1 in rungs:
        settings = Optim_build_settings(**overrides)
        jmax = _knob(settings, "jmax")
        done[1] = run(1, partial(Optim_run_ladder, ladder_type=_Bisect), load(names), settings,
                      f"n* and jmax* by bisection over [{jmax.lo}, {jmax.hi}]", edge=_bisect_edge)
    bisections = Optim_bisections(store, done[1]) if 1 in done else []
    recommended = Optim_golden_from(bisections, method) if 1 in done else None
    if golden_jmax is not None:
        source = "given"
    elif recommended is not None:
        golden_jmax, source = recommended, "rung 1"
    else:
        golden_jmax, source = GOLDEN_JMAX, "constant"
    if 1 in done:
        found = sum(b.jmax_star is not None for b in bisections)
        _end_rung(done[1], f"jmax* found on {found}/{len(bisections)}, "
                           f"{sum(b.failed is not None for b in bisections)} failed",
                  _rung1_findings(bisections, done[1], golden_jmax, method))
    if 2 in rungs:
        settings = Optim_build_settings(**{**overrides, "golden_jmax": golden_jmax, "rung2_budget": rung2_cap,
                                           "rung3_budget": 0})
        what = f"smoothing searches at the study's gates (skew {settings['gate_skew_max']:g}), budget {rung2_cap}, " \
               f"golden jmax {golden_jmax}"
        if names:
            chosen = list(names)
        elif 1 in done:  # those the campaign sends up to rung 2
            chosen = [b.name for b in bisections if b.failed_at_star and set(b.failed_at_star) <= SHAPE_GATES]
            what += "; those over a shape gate at jmax* in rung 1"
        else:
            chosen = list(method.set)
        if chosen:
            done[2] = run(2, Optim_run_ladder, load(chosen), settings, what)
            rows = Optim_smoothings(store, done[2])
            routed = [s for s in rows if s.meshes is not None]
            _end_rung(done[2], f"{len(routed)}/{len(rows)} entered rung 2, {sum(s.passed for s in routed)} passed",
                      _rung2_findings(rows))
        else:
            logger.info("%s  none to run: every airfoil passes the shape gates at jmax* in rung 1",
                        _RUNG_START.format(2))
    if 3 in rungs:
        chosen = names or [*method.set, *Optim_ordinary_airfoils([n for n, _ in load()], method)]
        settings = Optim_build_settings(**{**overrides, "golden_jmax": golden_jmax, "rung3_budget": rung3_cap})
        done[3] = run(3, partial(Optim_run_ladder, ladder_type=_Descend), load(chosen), settings,
                      f"downward searches from the golden run (jmax {golden_jmax}) to {rung3_cap} c2d runs each",
                      edge=partial(_descend_edge, store, settings))
        rows = Optim_descents(store, done[3])
        _end_rung(done[3], f"{len(rows)} airfoils, {sum(len(d.nodes) - 1 for d in rows)} c2d runs",
                  _rung3_findings(rows, Optim_plateau_table(rows, method), method))
    return Calibration(done, golden_jmax, source, recommended, method)


def _run_stage(rung: int, task, airfoils: list, settings: dict, what: str, *, store: RunStore, work_root: Path,
               workers: int, first: int, watchdog_s: float, cap_mb: int, edge=None) -> Stage:
    logger.info("%s  %s; %d airfoils, %d workers", _RUNG_START.format(rung), what, len(airfoils), workers)
    store.ensure_settings(settings, label=f"calibration rung {rung}")
    ctx = RunContext.create(settings, work_root / f"rung{rung}")
    t0 = time.perf_counter()
    outcomes = Optim_run_airfoils(task, airfoils, ctx, store, workers, warn_memory=rung == first, edge=edge,
                                  watchdog_s=watchdog_s, cap_mb=cap_mb)
    return Stage(rung, settings, [name for name, _ in airfoils], outcomes, time.perf_counter() - t0)


def _end_rung(stage: Stage, headline: str, findings: list[str]) -> None:
    """A rung's end in the log: its time and headline, then what it found, as the report words it."""
    logger.info("%s  %.1f min: %s", _RUNG_END.format(stage.rung), stage.seconds / 60, headline)
    for line in findings:
        logger.info("  %s", line)


def Optim_ordinary_airfoils(names: Sequence[str], method: Method = Method()) -> list[str]:
    """The method's count of airfoils outside its calibration set, the same ones for the same names."""
    rest = sorted(set(names) - set(method.set))
    return sorted(random.Random(ORDINARY_SEED).sample(rest, min(method.ordinary_count, len(rest))))

#endregion


#region Rung 1: n*, jmax* and the golden jmax

class Bisection(NamedTuple):
    name: str
    status: str
    n_star: Optional[int]
    n_star_floor: Optional[int]       # at the narrowest cluster_width
    cluster_width: Optional[float]    # where n* landed
    jmax_star: Optional[int]          # None: above jmax's range, or not found (failed)
    passes_all: Optional[bool]        # every gate passes at jmax*
    failed: Optional[str]             # why there is no jmax*, if not above the range
    mesh_s: dict[int, float]          # each bisection mesh's wall time, by its jmax
    n_star_s: Optional[float]
    n_star_floor_s: Optional[float]
    seconds: Optional[float]          # the whole ladder
    skew_at_star: Optional[float] = None  # the jmax* mesh's worst skew
    sr_below: Optional[float] = None      # the size ratio one jmax below jmax*, where the bisection last failed
    failed_at_star: Optional[list] = None  # the gates the jmax* mesh fails, the size ratio passing there


def Optim_bisections(store: RunStore, stage: Stage) -> list[Bisection]:
    out = []
    for name, outcome in zip(stage.names, stage.outcomes):
        ladder, notes = store.ladder(outcome.ladder_id), outcome.notes or {}
        knobs = json.loads(ladder["n_star_knobs_json"]) if ladder["n_star_knobs_json"] else None
        meshed = {r["jmax"]: r for r in store.runs(outcome.ladder_id) if r["status"] == "meshed"}
        star = notes.get("jmax_star")
        at_star, below = meshed.get(star), meshed.get(star - 1) if star is not None else None
        out.append(Bisection(
            name, outcome.status, ladder["n_star"], notes.get("n_star_floor"), knobs and knobs["cluster_width"],
            star, notes.get("jmax_star_passes"), notes.get("bisect_failed") or _why(outcome),
            {j: r["wall_s"] for j, r in meshed.items()}, ladder["n_star_s"], notes.get("n_star_floor_s"),
            ladder["wall_s"], at_star and at_star["q_skew_max"], below and below["q_size_ratio_max"],
            at_star and json.loads(at_star["failed_gates"])))
    return out


def Optim_golden_from(rows: list[Bisection], method: Method = Method()) -> Optional[int]:
    """The golden jmax: the median jmax* (rounded up) + the method's margin; None with no jmax* at all."""
    found = [b.jmax_star for b in rows if b.jmax_star is not None]
    return math.ceil(statistics.median(found)) + method.golden_margin if found else None


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


def Optim_plateau_table(rows: list[Descent], method: Method = Method()) -> list[PlateauRow]:
    table = []
    for window in method.windows:
        for gain in method.gains:
            meshes, seconds, losses = 0, 0.0, []
            for d in rows:
                k = Optim_replay(d.nodes, window, gain)
                meshes, seconds = meshes + k, seconds + d.seconds[k]
                if d.final is not None:
                    losses.append(d.nodes[k] / d.final - 1)
            table.append(PlateauRow(window, gain, meshes, seconds, max(losses, default=0.0),
                                    statistics.fmean(losses) if losses else 0.0))
    return table


def Optim_pick_plateau(table: list[PlateauRow], method: Method = Method()) -> Optional[PlateauRow]:
    """The fewest meshes among the pairs losing at most the method's max loss on any airfoil; ties to the cautious."""
    qualified = [p for p in table if p.worst_loss <= method.max_loss]
    return min(qualified, key=lambda p: (p.meshes, -p.window, p.min_gain)) if qualified else None


def Optim_rung3_cap(rows: list[Descent], method: Method = Method()) -> Optional[int]:
    """The most c2d runs any airfoil took to get within the method's `within` of its best, rounded up to 10."""
    needed = [m for m in (Optim_meshes_within(d.nodes, method.within) for d in rows) if m is not None]
    return _round_up(max(needed)) if needed else None

#endregion


#region The report

def Optim_format_report(store: RunStore, cal: Calibration) -> str:
    """The calibration report: a header, then each rung that was calibrated."""
    commit, dirty = Optim_git_state()
    dx_max = next(iter(cal.stages.values())).settings["dx_max"] if cal.stages else DX_MAX
    out = ["Mesh-study calibration",
           f"{datetime.date.today().isoformat()}, code {commit[:12]}{' (dirty)' if dirty else ''}, dx_max {dx_max:g}"]
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
        out += _format_rung2(Optim_smoothings(store, cal.stages[2]), cal.stages[2])
    if 3 in cal.stages:
        jmax_star = {b.name: b.jmax_star for b in Optim_bisections(store, cal.stages[1])} if 1 in cal.stages else {}
        out += _format_rung3(Optim_descents(store, cal.stages[3]), jmax_star, cal)
    return "\n".join(out).rstrip() + "\n"


def _format_rung1(rows: list[Bisection], cal: Calibration) -> list[str]:
    jmax_range = _knob(cal.stages[1].settings, "jmax")
    out = [f"Rung 1: n*, jmax* and the golden jmax (jmax*: the smallest jmax whose mesh passes the size ratio, "
           f"bisected over [{jmax_range.lo}, {jmax_range.hi}])",
           f"  {'airfoil':22s} {'n*':>5} {'n*@cw0':>6} {'cw':>6} {'jmax*':>6} {'all':>4} {'meshes':>6} "
           f"{'n* s':>6} {'s':>6}  note"]
    for b in rows:
        jmax = "-" if b.jmax_star is None else (f"<={b.jmax_star}" if b.jmax_star == jmax_range.lo
                                                 else str(b.jmax_star))
        if b.jmax_star is None and b.failed is None:
            jmax = f">{jmax_range.hi}"
        out.append(f"  {b.name:22s} {_i(b.n_star):>5} {_i(b.n_star_floor):>6} {_f(b.cluster_width, '.3f'):>6} "
                   f"{jmax:>6} {_yes(b.passes_all):>4} {len(b.mesh_s):>6} {_f(b.n_star_s, '.1f'):>6} "
                   f"{_f(b.seconds, '.0f'):>6}  {b.failed or ''}")
    out.append("")
    out += [f"  {line}" for line in _rung1_findings(rows, cal.stages[1], cal.golden_jmax, cal.method)]
    out.append("")
    return out


def _rung1_findings(rows: list[Bisection], stage: Stage, golden_jmax: int, method: Method) -> list[str]:
    """What rung 1 found, for the report and the log: jmax*, the golden jmax, n*, a campaign's time, failures."""
    out = []
    found = [b.jmax_star for b in rows if b.jmax_star is not None]
    above = sum(b.jmax_star is None and b.failed is None for b in rows)
    jmax_max, cw_min = _knob(stage.settings, "jmax").hi, _knob(stage.settings, "cluster_width").lo
    if found:
        golden = Optim_golden_from(rows, method)
        out.append(f"jmax*: min {min(found)}, median {statistics.median(found):g}, max {max(found)} "
                   f"({len(found)} found, {above} above {jmax_max}, {sum(b.failed is not None for b in rows)} failed)")
        bumps = sorted(j - golden for j in found if j > golden)
        out.append(f"Golden jmax = median rounded up + {method.golden_margin} = {golden}: {len(found) - len(bumps)}/"
                   f"{len(found)} pass the size ratio there"
                   + (f"; the rest need {', '.join(map(str, bumps))} bumps" if bumps else ""))
        out.append(f"jmax* also passes every other gate on {sum(bool(b.passes_all) for b in rows)}/{len(found)}")
    below = sorted(b.sr_below for b in rows if b.sr_below is not None)
    if below:
        out.append(f"size ratio one jmax below jmax*: median {statistics.median(below):.4f}, range {below[0]:.4f}-"
                   f"{below[-1]:.4f} (gate {stage.settings['gate_size_ratio_max']:g})")
    skews = [(b.skew_at_star, b.name) for b in rows if b.skew_at_star is not None]
    if skews:
        gate = stage.settings["gate_skew_max"]
        worst, name = max(skews)
        out.append(f"worst skew at jmax*: median {statistics.median(s for s, _ in skews):.3f}, max {worst:.3f} "
                   f"({name}); {sum(s > gate for s, _ in skews)}/{len(skews)} over the study's skew gate {gate:g}")
    with_n = [b for b in rows if b.n_star is not None]
    if with_n:
        narrowest = sum(b.cluster_width == cw_min for b in with_n)
        same = sum(b.n_star_floor == b.n_star for b in with_n)
        out.append(f"cluster_width: n* landed on the narrowest ({cw_min:g}) for "
                   f"{narrowest}/{len(with_n)}; n* at the narrowest alone equals the full grid's for "
                   f"{same}/{len(with_n)}" + (": NSTAR_CW_GRID 1 loses nothing" if same == len(with_n) else ""))
        out.append(f"n* time: full grid median {statistics.median(b.n_star_s for b in with_n):.1f} s, "
                   f"narrowest only median {statistics.median(b.n_star_floor_s for b in with_n):.2f} s")
    estimate = Optim_campaign_estimate(rows, golden_jmax)
    if estimate:
        speedup = sum(b.seconds or 0 for b in rows) / stage.seconds if stage.seconds > 0 else 1.0
        serial, floor = estimate.n_star_s + estimate.mesh_s, estimate.n_star_floor_s + estimate.mesh_s
        out.append(f"Campaign estimate at golden jmax {golden_jmax}, from these times: {estimate.airfoils} "
                   f"airfoils, a golden mesh {estimate.mesh_median_s:.1f} s (median), {estimate.bumps} bumps")
        out.append(f"  {serial / 60:.1f} min serial, about {serial / speedup / 60:.1f} min at this run's "
                   f"{speedup:.1f}x; with NSTAR_CW_GRID 1, {floor / 60:.1f} and {floor / speedup / 60:.1f} min")
    out += [f"failed: {b.name}: {b.failed}" for b in rows if b.failed is not None]
    return out


def _format_rung2(rows: list[Smoothing], stage: Stage) -> list[str]:
    out = [f"Rung 2: its budget, under the study's skew gate {stage.settings['gate_skew_max']:g}",
           f"  {'airfoil':22s} {'golden skew':>11} {'golden failed':22s} {'route':32s} {'meshes':>6} {'skew':>6}  note"]
    for s in rows:
        meshes = "-" if s.meshes is None else (str(s.meshes) if s.passed else f"never ({s.meshes})")
        out.append(f"  {s.name:22s} {_f(s.golden_skew, '.3f'):>11} {','.join(s.golden_failed or []) or '-':22s} "
                   f"{' > '.join(s.route) or '-':32s} {meshes:>6} {_f(s.skew, '.3f'):>6}  {s.failed or ''}")
    out.append("")
    out += [f"  {line}" for line in _rung2_findings(rows)]
    out.append("")
    return out


def _rung2_findings(rows: list[Smoothing]) -> list[str]:
    """What rung 2 found, for the report and the log: who entered it, what a pass took, its budget."""
    routed = [s for s in rows if s.meshes is not None]
    passed = [s.meshes for s in routed if s.passed]
    out = [f"Routed to rung 2: {len(routed)}; passed: {len(passed)}"
           + (f" (meshes to a first pass: median {statistics.median(passed):g}, max {max(passed)})" if passed else "")
           + f"; never: {len(routed) - len(passed)}"]
    skews = sorted(s.skew for s in routed if s.passed and s.skew is not None)
    if skews:
        out.append(f"passing skew: {skews[0]:.3f}" + (f"-{skews[-1]:.3f}" if len(skews) > 1 else ""))
    out += [f"not routed to rung 2: {s.name} ("
            + (f"golden failed {', '.join(s.golden_failed)}" if s.golden_failed else "golden passed")
            + (f"; {' > '.join(s.route)}" if len(s.route) > 1 else "") + ")"
            for s in rows if s.meshes is None and s.failed is None]
    out += [f"failed: {s.name}: {s.failed}" for s in rows if s.failed is not None]
    out.append(f"Rung-2 budget (the most meshes to a first pass, rounded up to 10): {_i(Optim_rung2_budget(rows))}")
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
        within, best_at = Optim_meshes_within(d.nodes, cal.method.within), Optim_meshes_within(d.nodes, 0.0)
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
    table = Optim_plateau_table(rows, cal.method)
    saving, *picks = _rung3_findings(rows, table, cal.method)
    out += ["", f"  {saving}", "",
            "  Plateau replay: where each (window, min_gain) would have stopped every search above, summed; "
            "loss: how far its best ends above the cap's",
            f"  {'window':>6} {'gain':>6} {'meshes':>7} {'seconds':>8} {'worst loss':>10} {'mean loss':>9}"]
    for p in table:
        out.append(f"  {p.window:>6} {_pct(p.min_gain, 2):>6} {p.meshes:>7} {p.seconds:>8.0f} "
                   f"{_pct(p.worst_loss, 2):>10} {_pct(p.mean_loss, 2):>9}")
    out.append("")
    out += [f"  {line}" for line in picks]
    out.append("")
    return out


def _rung3_findings(rows: list[Descent], table: list[PlateauRow], method: Method) -> list[str]:
    """What rung 3 found, for the report and the log: the nodes saved, the plateau pick, the cap."""
    savings = [(1 - d.final / d.golden_nodes, d.name) for d in rows if d.golden_passed and d.final is not None]
    saving = "nodes saved against the golden run: " + (
        f"median {_pct(statistics.median(s for s, _ in savings))}, max {_pct(max(savings)[0])} ({max(savings)[1]})"
        if savings else "none (no golden run passed)")
    pick = Optim_pick_plateau(table, method)
    return [saving,
            f"Plateau pick (fewest meshes giving up at most {_pct(method.max_loss, 1)} on every airfoil): "
            + (f"window {pick.window}, min_gain {pick.min_gain:g}, {pick.meshes} meshes, {pick.seconds / 60:.1f} min"
               if pick else "none qualifies"),
            f"Rung-3 cap (the most meshes any airfoil took to within {_pct(method.within, 0)} of its best, rounded up "
            f"to 10): {_i(Optim_rung3_cap(rows, method))}"]


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
