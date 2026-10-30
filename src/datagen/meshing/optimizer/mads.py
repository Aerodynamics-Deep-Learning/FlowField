"""
The mesh study's NOMAD driver: one MADS search over some of a config's knobs, each point sampled, meshed
and judged by the executor. The ladder's rungs 2 and 3 call it, and so will the calibration.

    run_mads    one search, ended once its budget of c2d runs is spent
    RungResult  what it did: its new rows, the c2d runs spent, the best pass, how NOMAD ended

A new run spends the budget when c2d ran, whatever it made of it (its row has an entry flag): a rejected
or dominated config, or one the ladder already ran, costs none. Start points are tried in order, the next only when NOMAD could not evaluate the last (NOMAD 4.6
keeps only the last of several X0 lines).

Any error while a point is evaluated (a setup error, a bug, Ctrl-C) stops the search and is raised once
NOMAD returns; NOMAD itself would print it and carry on. NOMAD calls its stop callback only once it holds a
feasible point, so an over search reports every further point as one, unmeshed. The callback is global to
the process, so one search runs per process at a time, which the campaign's one ladder per worker keeps.
"""

from typing import NamedTuple, Optional, Sequence

import PyNomad

from src.datagen.meshing.optimizer.executor import RunExecutor, nomad_outputs
from src.datagen.meshing.optimizer.space import Knob, from_x, input_types, nomad_bounds, to_x

_OUTPUT_TYPES = "BB_OUTPUT_TYPE OBJ EB EB PB PB PB PB PB PB"

# NOMAD's run flag when it could not evaluate its start point
_START_FAILED = -3

# Every point once the search is over: feasible, so NOMAD reaches its stop callback, and never a best
_OVER = " ".join(["1e300"] + ["-1"] * 8).encode()


class RungResult(NamedTuple):
    """One search's outcome."""
    rows: list[dict]          # its new rows, in order; a cache hit makes none
    mesher_runs: int          # new rows c2d ran for, the budget spent
    best_row: Optional[dict]  # the fewest-node pass seen, best_so_far included
    start: int                # the start point NOMAD last ran from
    run_flag: int             # how NOMAD's last run ended; -3: it could not evaluate its start point
    stop_reason: str
    n_calls: int              # blackbox calls, cached ones and those answered once the search was over included


def _never(_) -> bool:
    """NOMAD's stop callback between searches: it is global, and a search's own, once freed, crashes the next run."""
    return False


class _Search:
    """One search's state: what NOMAD's blackbox does with a point, and when the search is over."""

    def __init__(self, executor: RunExecutor, run_kwargs: dict, base_knobs: dict, knobs: Sequence[Knob],
                 x0s: Sequence[dict], budget: int, objective: str, stop_at_first_pass: bool, dominance: bool,
                 best_so_far: Optional[dict]):
        self.executor, self.settings = executor, executor.ctx.settings
        self.run_kwargs, self.base_knobs, self.knobs = run_kwargs, base_knobs, knobs
        # The log scale doesn't return a start's values exactly (20 comes back 20.000000000000004), so a
        # start would miss the ladder's cache; its point maps back to its own values
        self.starts = {tuple(to_x(x0, knobs)): {k.name: x0[k.name] for k in knobs} for x0 in x0s}
        self.budget, self.objective = budget, objective
        self.stop_at_first_pass, self.dominance = stop_at_first_pass, dominance
        self.best = best_so_far
        self.rows: list[dict] = []
        self.mesher_runs = self.calls = 0
        self.stop = False
        self.error: Optional[BaseException] = None

    @property
    def over(self) -> bool:
        return self.stop or self.mesher_runs >= self.budget

    def callback(self, _) -> bool:
        return self.over

    def blackbox(self, x) -> int:
        """NOMAD's blackbox: 1 with the outputs set, 0 for a failed evaluation. Never raises into NOMAD."""
        self.calls += 1
        if not self.over:
            try:
                return self._evaluate(x)
            except BaseException as e:  # raised after NOMAD returns
                self.error, self.stop = e, True
        x.setBBO(_OVER)
        return 1

    def _evaluate(self, x) -> int:
        point = tuple(x.get_coord(i) for i in range(len(self.knobs)))
        searched = self.starts.get(point) or from_x(point, self.knobs)
        knobs = {**self.base_knobs, **searched}
        threshold = self.best["num_nodes"] if self.dominance and self.best is not None else None
        result = self.executor.run(**self.run_kwargs, knobs=knobs, seq=len(self.rows), skip_if_nodes_ge=threshold)
        row = result.row
        if not result.cached:
            self.rows.append(row)
            if row.get("entry_flag") is not None:  # c2d ran: meshed, or failed in it or after it
                self.mesher_runs += 1
        if row.get("passed"):  # a cached pass counts too: it is the ladder's
            if self.best is None or row["num_nodes"] < self.best["num_nodes"]:
                self.best = row
            if self.stop_at_first_pass:
                self.stop = True
        outputs = nomad_outputs(row, self.objective, self.settings)
        if outputs is None:
            return 0
        x.setBBO(" ".join(map(repr, outputs)).encode())
        return 1


def run_mads(executor: RunExecutor, run_kwargs: dict, base_knobs: dict, knobs: Sequence[Knob], x0s: Sequence[dict], *,
             budget: int, objective: str, stop_at_first_pass: bool = False, dominance: bool = False,
             best_so_far: Optional[dict] = None) -> RungResult:
    """
    One MADS search over `knobs`, every config's other knobs fixed at `base_knobs`. It ends when `budget`
    new c2d runs are spent, when NOMAD ends, or at the first pass if `stop_at_first_pass`. The seed and
    NOMAD's evaluation cap come from the executor's settings row.

    Args:
        executor (RunExecutor): The ladder's executor; its cache makes a config the ladder already ran free
        run_kwargs (dict): What every run shares: ladder_id, geometry_id, airfoil_name, contour, rung,
            parent_run_id
        base_knobs (dict): The config's knobs that are not searched
        knobs (Sequence[Knob]): The searched knobs, from `space.knobs_for`
        x0s (Sequence[dict]): Start points, a value per searched knob, tried in order: the next only when
            NOMAD could not evaluate the last
        budget (int): New c2d runs the search may spend, failed ones included
        objective (str): "nodes" or "violation", see `executor.nomad_outputs`
        stop_at_first_pass (bool): End the search at its first passing config
        dominance (bool): Don't mesh a config with at least as many nodes as the best pass so far
        best_so_far (dict | None): A passed row from before the search: the first best, and its threshold

    Returns:
        RungResult: The search's new rows, c2d runs spent, best pass and how NOMAD ended

    Raises:
        ValueError: On an unknown objective, no start points, or a start point NOMAD cannot take (outside the
            bounds, or a fractional integer), which would crash the process instead of raising
        BaseException: Whatever stopped the search while a point was evaluated (a StudySetupError, a bug,
            KeyboardInterrupt), once NOMAD returns
    """
    if objective not in ("nodes", "violation"):
        raise ValueError(f"objective is 'nodes' or 'violation', not {objective!r}")
    if not x0s:
        raise ValueError("a search needs at least one start point")
    lb, ub = nomad_bounds(knobs)
    starts = [_x0_line(x0, knobs, lb, ub) for x0 in x0s]
    settings = executor.ctx.settings
    params = [f"DIMENSION {len(knobs)}", input_types(knobs), _OUTPUT_TYPES,
              f"MAX_BB_EVAL {settings['max_bb_eval_factor'] * budget}", "DISPLAY_DEGREE 0",
              f"SEED {settings['nomad_seed']}"]

    search = _Search(executor, run_kwargs, base_knobs, knobs, x0s, budget, objective, stop_at_first_pass, dominance,
                     best_so_far)
    callback = search.callback  # NOMAD holds no reference of its own; this one lives through the runs
    PyNomad.setCustomMegaIterEndCallback(callback)
    try:
        for start, line in enumerate(starts):
            # PyNomad resets its generator to the unseeded state every run but reapplies SEED only when it differs
            # from the last seed set, so a repeated seed would run unseeded; moving off it makes SEED apply
            PyNomad.setSeed(settings["nomad_seed"] + 1)
            result = PyNomad.optimize(search.blackbox, [], lb, ub, params + [line])
            if result["run_flag"] != _START_FAILED or search.over:
                break
    finally:
        PyNomad.setCustomMegaIterEndCallback(_never)
    if search.error is not None:
        raise search.error
    return RungResult(rows=search.rows, mesher_runs=search.mesher_runs, best_row=search.best, start=start,
                      run_flag=result["run_flag"], stop_reason=result["stop_reason"], n_calls=search.calls)


def _x0_line(x0: dict, knobs: Sequence[Knob], lb: list[float], ub: list[float]) -> str:
    """A start point as NOMAD's X0 line, checked first: NOMAD crashes the process on one it cannot take."""
    x = to_x(x0, knobs)
    for v, k, lo, hi in zip(x, knobs, lb, ub):
        if not lo <= v <= hi or (k.type == "I" and v != int(v)):
            raise ValueError(f"start point {x0}: NOMAD cannot take {k.name}={x0[k.name]} "
                             f"(bounds [{k.lo}, {k.hi}], {'integer' if k.type == 'I' else 'real'})")
    return f"X0 ( {' '.join(map(repr, x))} )"
