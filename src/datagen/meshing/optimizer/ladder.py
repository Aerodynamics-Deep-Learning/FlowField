"""
The mesh study's per-airfoil ladder: n*, the golden run, then up the rungs until a config passes or the
last rung is spent. The campaign runs one ladder at a time per worker.

    run_ladder     one airfoil's study, reported as events
    LadderOutcome  how it ended, with the error that ended it, for the caller to act on
    record         one event written to the run store

The golden run meshes n*'s knobs at golden_jmax with c2d's smoothing defaults, and its verdict routes it:
    passed                      done
    size_ratio alone            rung 1 (jmax_bump): jmax up a step at a time, each bump routed the same
                                way; still size_ratio after the last, rung 3 (smoothing can't move the
                                radial spacing)
    shape gates alone           rung 2 (smoothing): NOMAD over the smoothing knobs to the first pass, the
                                rest of the routed config fixed; no pass, rung 3
    anything else, or no mesh   rung 3 (mads): NOMAD over every knob for the fewest nodes, with dominance
A rejected or invalid golden or bump config is a bug (n*'s knobs always sample) and aborts the ladder.

A ladder never writes the store. It reports ("ladder_start", row) first, ("run", row) per new run and
("ladder_end", ladder_id, fields) last, and the campaign's one writer applies each with `record`: the
ladder's row is inserted at its start and completed at its end, and each run is one row.

An error ends the ladder as aborted, its traceback recorded, and is handed to the caller in the outcome,
not raised: the caller decides whether it stops anything else. Ctrl-C is recorded too, then raised.
"""

import time
import traceback
import uuid
from typing import Callable, NamedTuple, Optional

from src.datagen.geometry.common.schemas import ShapeRecord
from src.datagen.meshing.c2d.schemas import C2D_MeshingConfig
from src.datagen.meshing.common.schemas import MeshBackend, MeshTopology
from src.datagen.meshing.optimizer.constants import SMOOTHING_DEFAULTS
from src.datagen.meshing.optimizer.executor import RunContext, RunExecutor, nomad_outputs
from src.datagen.meshing.optimizer.mads import RungResult, run_mads
from src.datagen.meshing.optimizer.nstar import NStarResult, n_star_search
from src.datagen.meshing.optimizer.sampling import build_study_contour
from src.datagen.meshing.optimizer.space import knobs_for
from src.datagen.meshing.optimizer.store import RunStore, utc_now

# The gates smoothing can fix; a run failing only these goes to rung 2
SHAPE_GATES = frozenset({"inverted", "skew", "ortho", "jac"})


class LadderOutcome(NamedTuple):
    """How a ladder ended. Picklable, so a pool's worker can return it."""
    ladder_id: str
    status: str                       # 'passed', 'failed' or 'aborted'
    route: list[str]                  # the stages entered, e.g. ['golden', 'jmax_bump']
    best_row: Optional[dict]          # the fewest-node pass
    error: Optional[BaseException]    # what aborted it, for the caller to act on


def run_ladder(name: str, shape: ShapeRecord, ctx: RunContext, emit: Callable[[tuple], None]) -> LadderOutcome:
    """
    One airfoil's study: n*, the golden run, and the rungs its verdict routes it up.

    Args:
        name (str): The airfoil's name
        shape (ShapeRecord): Its stored shape under the settings' shape version
        ctx (RunContext): The batch's context; the ladder gets its own executor, so its own cache
        emit (Callable[[tuple], None]): Takes each event, in order

    Returns:
        LadderOutcome: How it ended; an error that aborted it is in `error`, not raised

    Raises:
        BaseException: KeyboardInterrupt (or another non-Exception), once the aborted ladder is reported
    """
    ladder = _Ladder(name, shape, ctx, emit)
    emit(("ladder_start", ladder.start_row()))
    t0 = time.perf_counter()
    error = None
    try:
        ladder.climb()
    except BaseException as e:  # the caller decides what it means
        error = e
    fields = ladder.end_fields(error)
    fields.update(wall_s=time.perf_counter() - t0, finished_at=utc_now())
    emit(("ladder_end", ladder.ladder_id, fields))
    if error is not None and not isinstance(error, Exception):
        raise error
    return LadderOutcome(ladder.ladder_id, fields["status"], list(ladder.route), ladder.best, error)


def record(store: RunStore, event: tuple) -> None:
    """
    Writes one ladder event to the run store: a start inserts the ladder's row, a run its own row, and the
    end completes the ladder's row and repoints the airfoil's best run, together.

    Args:
        store (RunStore): The store, holding the settings row the ladder ran under
        event (tuple): ("ladder_start", row), ("run", row) or ("ladder_end", ladder_id, fields)

    Raises:
        ValueError: On an unknown event, or a write the store refuses (an unknown column, a ladder never
            started)
    """
    kind = event[0]
    if kind == "ladder_start":
        store.start_ladder(event[1])
    elif kind == "run":
        store.insert_run(event[1])
    elif kind == "ladder_end":
        _, ladder_id, fields = event
        with store.transaction():
            store.finish_ladder(ladder_id, fields)
            ladder = store.ladder(ladder_id)
            store.update_best(ladder["geometry_id"], ladder["settings_id"])
    else:
        raise ValueError(f"unknown ladder event {kind!r}")


class _Ladder:
    """One ladder's state as it climbs: its new rows, route, best pass and NOMAD searches."""

    def __init__(self, name: str, shape: ShapeRecord, ctx: RunContext, emit: Callable[[tuple], None]):
        self.ladder_id = str(uuid.uuid4())
        self.name, self.shape, self.ctx, self.emit = name, shape, ctx, emit
        self.settings = ctx.settings
        self.executor = RunExecutor(ctx, self._emitted)
        self.contour = None
        self.nstar: Optional[NStarResult] = None
        self.golden: Optional[dict] = None
        self.best: Optional[dict] = None
        self.route: list[str] = []
        self.rows: list[dict] = []
        self.nomad: dict = {}

    def start_row(self) -> dict:
        ctx = self.ctx
        return dict(ladder_id=self.ladder_id, settings_id=self.settings["settings_id"], batch_id=ctx.batch_id,
                    geometry_id=self.shape.geometry_id, airfoil_name=self.name, status="running", host=ctx.host,
                    git_commit=ctx.git_commit, git_dirty=ctx.git_dirty, started_at=utc_now())

    def climb(self) -> None:
        s = self.settings
        self.contour = build_study_contour(self.shape, MeshBackend(s["backend"]), MeshTopology(s["topology"]),
                                           gap=s["te_gap"])
        self.nstar = n_star_search(self.contour, s)
        if self.nstar.n is None:
            self.route.append("nstar")  # no surface sampling to mesh: the ladder fails
            return

        golden_knobs = knobs = {**self.nstar.knobs, "jmax": s["golden_jmax"], **SMOOTHING_DEFAULTS}
        self.route.append("golden")
        row = self.golden = self._run(knobs, "golden", 0)
        step = _route(row)
        if step == "jmax_bump":
            self.route.append(step)
            for seq in range(s["jmax_bump_max"]):
                knobs = {**knobs, "jmax": knobs["jmax"] + s["jmax_bump_step"]}
                row = self._run(knobs, "jmax_bump", seq, parent=row)
                step = _route(row)
                if step != "jmax_bump":
                    break
            else:
                step = "mads"  # still size_ratio: smoothing can't move the radial spacing
        if step == "smoothing":
            self.route.append(step)
            fixed = {k: v for k, v in knobs.items() if k not in SMOOTHING_DEFAULTS}
            result = self._search(2, fixed, [SMOOTHING_DEFAULTS], parent=row, budget=s["rung2_budget"],
                                  objective="violation", stop_at_first_pass=True)
            step = "done" if result.best_row is not None else "mads"
        if step == "mads":
            self.route.append(step)
            x0s = [golden_knobs]
            if nomad_outputs(self.golden, "nodes", s) is None:  # NOMAD can't evaluate the golden run
                x0s.append(_config_defaults())
            self._search(3, {}, x0s, parent=row, budget=s["rung3_budget"], objective="nodes", dominance=True,
                         best_so_far=self.best)

    def end_fields(self, error: Optional[BaseException]) -> dict:
        """The ladder row's outcome columns, whatever stage it reached."""
        if error is not None:
            status = "aborted"
        else:
            status = "passed" if self.best is not None else "failed"
        golden, rows = self.golden, self.rows
        golden_failed = None
        if golden is not None:
            golden_failed = golden["failed_gates"] if golden["status"] == "meshed" else [golden["status"]]
        fields = dict(
            status=status, final_rung=self.route[-1] if self.route else None, route=list(self.route),
            golden_failed_gates=golden_failed, n_rejected=sum(r["status"] == "rejected" for r in rows),
            n_dominated=sum(r["status"] == "dominated" for r in rows),
            nomad_json=self.nomad or None, best_run_id=None if self.best is None else self.best["run_id"],
            error_text=None if error is None else "".join(traceback.format_exception(error)))
        for column, rung in (("meshes_golden", "golden"), ("meshes_bump", "jmax_bump"),
                             ("meshes_smoothing", "smoothing"), ("meshes_mads", "mads")):
            fields[column] = sum(r["rung"] == rung and r.get("entry_flag") is not None for r in rows)  # c2d runs
        nstar = self.nstar or NStarResult(n=None, knobs=None, evals=None, seconds=None)
        fields.update(n_star=nstar.n, n_star_knobs_json=nstar.knobs, n_star_grid_evals=nstar.evals,
                      n_star_s=nstar.seconds)
        return fields

    def _emitted(self, row: dict) -> None:
        self.emit(("run", row))
        self.rows.append(row)  # only once the writer has it

    def _shared(self, rung: str, parent: Optional[dict]) -> dict:
        return dict(ladder_id=self.ladder_id, geometry_id=self.shape.geometry_id, airfoil_name=self.name,
                    contour=self.contour, rung=rung, parent_run_id=None if parent is None else parent["run_id"])

    def _run(self, knobs: dict, rung: str, seq: int, parent: Optional[dict] = None) -> dict:
        row = self.executor.run(**self._shared(rung, parent), knobs=knobs, seq=seq).row
        self._keep(row)
        return row

    def _search(self, rung: int, fixed: dict, x0s: list[dict], parent: dict, budget: int, **kw) -> RungResult:
        """A NOMAD rung; its runs follow the golden or bump run whose verdict sent the ladder there."""
        name = self.route[-1]
        result = run_mads(self.executor, self._shared(name, parent), fixed, knobs_for(rung), x0s, budget=budget, **kw)
        self.nomad[name] = dict(run_flag=result.run_flag, stop_reason=result.stop_reason,
                                seed=self.settings["nomad_seed"], budget=budget, n_calls=result.n_calls,
                                mesher_runs=result.mesher_runs, x0s=x0s, start=result.start)
        self._keep(result.best_row)
        return result

    def _keep(self, row: Optional[dict]) -> None:
        if row is not None and row.get("passed") and (self.best is None or row["num_nodes"] < self.best["num_nodes"]):
            self.best = row


def _route(row: dict) -> str:
    """Where a golden or bump run sends the ladder: 'done', 'jmax_bump', 'smoothing' or 'mads'."""
    status = row["status"]
    if status in ("mesher_failed", "crashed"):
        return "mads"
    if status != "meshed":
        raise RuntimeError(f"the {row['rung']} run was {status}, but n*'s knobs always sample and validate: "
                           f"a ladder bug\n{row.get('error_text') or ''}".rstrip())
    failed = row["failed_gates"]
    if not failed:
        return "done"
    if failed == ["size_ratio"]:
        return "jmax_bump"
    if SHAPE_GATES.issuperset(failed):
        return "smoothing"
    return "mads"


def _config_defaults() -> dict:
    """C2D_MeshingConfig's own defaults for rung 3's knobs: the second start when NOMAD can't take the golden run."""
    defaults = C2D_MeshingConfig()
    return {k.name: getattr(defaults, k.name) for k in knobs_for(3)}
