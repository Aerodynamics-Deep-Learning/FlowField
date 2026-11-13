"""
p_mesh training labels: a shape's logged mesh attempts become one training row, or nothing.

p_mesh is P(a shape that passed the geometry rules meshes under the production settings): one c2d run
with the frozen knobs, judged by the production gates (`Optim_failed_gates`, the `passed` column), at the
fixed design Re every shape is meshed at, so the input is the shape alone. So a shape gets a label only if:
    - it reached c2d: a rule rejection (geometry rules, the sampler's pre-check) never becomes a row
    - it ran under the production settings: settings_id and knobs both match, optimizer probes don't
    - its outcome is the shape's: a tooling failure (no exe, a launch error, an exception, an aborted
      slot) is never a 0

A shape's tries form a chain: its first attempt and every rerun whose `rerun_of` links lead back to it,
in `seq` order.
Tries that show nothing about the shape are skipped, and the chain left pending: retried now after a
launch error, a log naming no cause or an aborted slot, but blocked on a fix after a tooling error
(FATAL_ERROR, no exe, a caller bug), which would only repeat. The first informative try decides: a mesh
verdict labels the shape, a rule rejection or pre-flight flag makes no row, and a timeout or crash, which
can be the shape's or a busy machine's, waits for one informative rerun. A pending chain is reported,
never dropped; `AL_needs_rerun` says which to schedule now.

The model's input is the 18 free CST values (`AL_cst_to_free`): the upper A_0..A_8, the lower A_1..A_8
(its A_0 is -upper A_0, `CSTParams`' shared nose) and the LE weight.
"""

from enum import Enum
from typing import Iterable, NamedTuple, Optional

import numpy as np
from pydantic import BaseModel, model_validator

from src.datagen.active_learning.design_vector import AL_cst_to_free
from src.datagen.geometry.common.schemas import CSTParams
from src.datagen.meshing.common.schemas import MeshBackend, MeshExitFlag, MeshTopology
from src.datagen.meshing.optimizer.common.settings import Optim_settings_hash


# How `c2d.run.C2D_generate_mesh` opens its log for each SUBPROCESS_FAIL cause
LOG_LAUNCH = "c2d failed to launch"
LOG_TIMEOUT = "c2d timed out after"
LOG_CRASH = "c2d exited"


class MeshOutcome(str, Enum):
    """Whose outcome a mesher flag is."""
    SUCCESS = "success"          # a mesh within the entry limits; the production gates decide the label
    DESIGN_FAIL = "design_fail"  # the shape doesn't mesh: 0
    BY_CAUSE = "by_cause"        # SUBPROCESS_FAIL: decided by `SubprocessCause` and the rerun
    INFRA = "infra"              # tooling or a caller bug: no row, the chain is retried once it's fixed
    PREFLIGHT = "preflight"      # caught before meshing, a rule's job: no row


class SubprocessCause(str, Enum):
    """Why c2d produced no grid."""
    LAUNCH = "launch"    # the exe didn't start: tooling
    TIMEOUT = "timeout"  # hung past the timeout: rerun once
    CRASH = "crash"      # exited without a grid: rerun once


# Every flag's owner; a new MeshExitFlag member fails tests/unit/datagen/active_learning until it is placed
FLAG_OUTCOME: dict[MeshExitFlag, MeshOutcome] = {
    MeshExitFlag.INPUT_TENSOR_FAIL: MeshOutcome.PREFLIGHT,
    MeshExitFlag.INPUT_CLOSURE_FAIL: MeshOutcome.PREFLIGHT,
    MeshExitFlag.INPUT_FREESTREAM_FAIL: MeshOutcome.INFRA,
    MeshExitFlag.INPUT_UNKNOWN_SOLVER: MeshOutcome.INFRA,
    MeshExitFlag.EXECUTABLE_NOT_FOUND: MeshOutcome.INFRA,
    MeshExitFlag.SUBPROCESS_FAIL: MeshOutcome.BY_CAUSE,
    MeshExitFlag.FATAL_ERROR: MeshOutcome.INFRA,                # a Python exception: a code bug
    MeshExitFlag.CONVERSION_FAIL: MeshOutcome.DESIGN_FAIL,      # deterministic for an input, a gate in the study
    MeshExitFlag.UNACCEPTABLE_QUALITY: MeshOutcome.DESIGN_FAIL,
    MeshExitFlag.GEOMETRY_DEVIATION_FAIL: MeshOutcome.DESIGN_FAIL,
    MeshExitFlag.LOW_QUALITY: MeshOutcome.DESIGN_FAIL,
    MeshExitFlag.SUCCESS: MeshOutcome.SUCCESS,
}


class ProductionMesh(BaseModel):
    """
    The settings p_mesh's labels hold for: the study's settings row (`Optim_build_settings`) and the frozen
    c2d knobs. The settings_id, mesher and TE are read off the one row, which must hash to its own id, so
    they can't disagree.
    """
    settings: dict
    knobs: dict[str, float]

    @model_validator(mode="after")
    def _row_hashes_to_its_id(self):
        missing = sorted({"settings_id", "backend", "topology", "te_gap", "cell_ceiling"} - set(self.settings))
        if missing:
            raise ValueError(f"settings row lacks {', '.join(missing)}")
        if Optim_settings_hash(self.settings) != self.settings["settings_id"]:
            raise ValueError(f"settings_id {self.settings['settings_id']} doesn't match the row's other columns "
                             f"(they hash to {Optim_settings_hash(self.settings)}): edited after hashing?")
        return self

    @property
    def settings_id(self) -> str:
        return self.settings["settings_id"]

    @property
    def backend(self) -> MeshBackend:
        return MeshBackend(self.settings["backend"])

    @property
    def topology(self) -> MeshTopology:
        return MeshTopology(self.settings["topology"])

    @property
    def te_gap(self) -> float:
        return float(self.settings["te_gap"])

    @property
    def cell_ceiling(self) -> int:
        return int(self.settings["cell_ceiling"])


class MeshAttempt(BaseModel):
    """
    One logged mesh try, as the run log will record it. `entry_flag` is None when it never reached
    the mesher (a rule rejection or an aborted slot); `passed` and `failed_gates` are set when it meshed.
    """
    attempt_id: str
    seq: int                            # its position in the run log: a chain's tries are judged in this order
    geometry_id: str                    # one per distinct shape: the ingest skips a file whose normalized
                                        # coordinates match a stored shape's (`ingest.py`, norm_hash); the store's
                                        # schema doesn't enforce it (norm_hash isn't UNIQUE)
    cst: CSTParams
    settings_id: str
    knobs: dict[str, float]
    geometry_rejected: bool = False     # `AL_geometry_rules` or `AL_mesh_rules` refused it
    aborted: bool = False               # its worker slot died, went silent or hit its cap
    entry_flag: Optional[MeshExitFlag] = None
    passed: Optional[bool] = None
    failed_gates: list[str] = []
    mesher_log: Optional[str] = None
    rerun_of: Optional[str] = None      # the try this one repeats: the first attempt, or a rerun of it


class MeshRow(BaseModel):
    """One p_mesh training example; `x` is a list so a row saves as JSON, `x_array` gives the model's input."""
    geometry_id: str
    settings_id: str
    attempt_id: str                     # the try that decided it
    x: list[float]                      # N_FREE values, `AL_cst_to_free`
    label: int                          # 1: passes the production gates
    failure_mode: Optional[str] = None  # why a 0 failed: the flag and the gates, or c2d_crash / c2d_hang

    def x_array(self) -> np.ndarray:
        return np.asarray(self.x, dtype=np.float64)


class MeshConflict(BaseModel):
    """A shape labelled both ways under one settings_id, left out of the training rows."""
    geometry_id: str
    settings_id: str
    rows: list[MeshRow]


class Pending(str, Enum):
    """Why a chain is still undecided, which tells the scheduler whether to run its next try."""
    RETRY = "retry"      # run the next try now: a launch error, an aborted slot, a timeout or crash to repeat
    BLOCKED = "blocked"  # a tooling error (no exe, an exception, a caller bug): fix it, then retry


class Verdict(NamedTuple):
    """What a chain of tries decided: a row, nothing (`row` None), or nothing yet (`pending` set)."""
    row: Optional[MeshRow]
    pending: Optional[Pending]
    preflight: Optional[str] = None  # the attempt_id the mesher refused before meshing


class MeshRows(NamedTuple):
    """
    The training rows, and every chain that made none it could have: the shapes left out for conflicting,
    the chains still awaiting a try, those whose first attempt isn't in the log or whose log is broken,
    and the attempts the mesher refused before meshing.
    """
    rows: list[MeshRow]
    conflicts: list[MeshConflict]
    retry: list[str]                 # first attempt_ids of chains to run the next try of now
    blocked: list[str]               # first attempt_ids of chains waiting on a tooling fix
    orphans: list[str]               # the missing first attempt_ids
    broken: dict[str, str]           # first attempt_id (a loop's smallest) -> why the chain can't be judged
    preflight_attempts: list[str]    # attempt_ids, not first attempt_ids: the try that hit the flag


class _Try(str, Enum):
    """What one try shows on its own."""
    RETRY = "retry"          # nothing about the shape (a launch error, an aborted slot): retry now
    BLOCKED = "blocked"      # nothing about the shape, and a retry would repeat it (tooling): fix, then retry
    NO_ROW = "no_row"        # a rule rejection or pre-flight flag: settles the chain with no row
    C2D_FAIL = "c2d_fail"    # a timeout or crash: needs a second informative try
    VERDICT = "verdict"      # a mesh the gates judged: labels the shape


def AL_subprocess_cause(mesher_log: Optional[str]) -> Optional[SubprocessCause]:
    """The cause of a SUBPROCESS_FAIL, from the head of c2d's log; None if the log says neither."""
    head = (mesher_log or "").lstrip()
    for prefix, cause in ((LOG_LAUNCH, SubprocessCause.LAUNCH), (LOG_TIMEOUT, SubprocessCause.TIMEOUT),
                          (LOG_CRASH, SubprocessCause.CRASH)):
        if head.startswith(prefix):
            return cause
    return None


def AL_chain_verdict(tries: list[MeshAttempt], production: ProductionMesh) -> Verdict:
    """
    What a shape's chain of tries decides.

    Args:
        tries (list[MeshAttempt]): The first attempt, then its reruns in seq order, each rerun after the
            try it names in `rerun_of`
        production (ProductionMesh): The settings labels hold for

    Returns:
        Verdict: The row of the first try that labels the shape; no row for a rule rejection, a
            pre-flight flag, or non-production settings; pending while only tries that show nothing, or
            one timeout or crash and such tries, are in: blocked if the last was a tooling error, else retry

    Raises:
        ValueError: If the tries aren't one chain, or a try is missing what its kind needs
    """
    if not tries:
        raise ValueError("chain needs at least one attempt, got none")
    first = tries[0]
    ids = [t.attempt_id for t in tries]
    if (first.rerun_of is not None or len(set(ids)) != len(ids)
            or any(t.rerun_of not in ids[:i] for i, t in enumerate(tries[1:], 1))):
        raise ValueError(f"not one chain: {[(t.attempt_id, t.rerun_of) for t in tries]}")
    if not _is_production(first, production):
        return Verdict(None, None)

    failed: Optional[MeshAttempt] = None  # the first timeout or crash, awaiting its rerun
    waiting = Pending.RETRY               # set by the last try: one after a tooling fix unblocks the chain
    for t in tries:
        if not _is_production(t, production):
            raise ValueError(f"rerun {t.attempt_id} ran under other settings than its first attempt")
        kind = _try_kind(t)
        if kind in (_Try.RETRY, _Try.BLOCKED):
            waiting = Pending.RETRY if kind is _Try.RETRY else Pending.BLOCKED
            continue
        if kind is _Try.NO_ROW:
            return Verdict(None, None, None if t.geometry_rejected else t.attempt_id)
        if kind is _Try.VERDICT:
            return Verdict(_verdict_row(t), None)  # a rerun that meshes labels the shape itself
        if failed is None:
            failed, waiting = t, Pending.RETRY
            continue
        both_hung = all(AL_subprocess_cause(f.mesher_log) is SubprocessCause.TIMEOUT for f in (failed, t))
        return Verdict(_row(failed, 0, "c2d_hang" if both_hung else "c2d_crash"), None)
    return Verdict(None, waiting)


def AL_needs_rerun(tries: list[MeshAttempt], production: ProductionMesh) -> bool:
    """Whether to schedule a chain's next try now: undecided, and not blocked on a tooling fix."""
    return AL_chain_verdict(tries, production).pending is Pending.RETRY


def AL_mesh_training_rows(attempts: Iterable[MeshAttempt], production: ProductionMesh) -> MeshRows:
    """
    Every row a set of attempts makes, one per (geometry_id, settings_id).

    A shape labelled both ways (say it hung twice on a busy machine and meshed later on a quiet one) is
    left out of the rows and reported, so one bad shape can't stop the build; chains still undecided
    are listed by their first attempt_id, apart by whether to retry now or after a tooling fix. A rerun of
    a rerun joins the chain its `rerun_of` links lead back to; a chain whose first attempt is missing from
    the log is listed in the orphans. A chain that can't be judged (its links loop, an attempt_id logged
    twice with different contents, or anything `AL_chain_verdict` refuses) is listed in `broken` with the
    reason, so one bad log row can't stop the build either; an attempt logged twice alike counts once. An
    attempt the mesher refused before meshing is listed in `preflight_attempts`: the geometry rules should
    have caught it.

    Args:
        attempts (Iterable[MeshAttempt]): Logged tries, in any order: each chain is put in seq order
        production (ProductionMesh): The settings labels hold for

    Returns:
        MeshRows: The rows, and every chain that made none: by why
    """
    by_id: dict[str, MeshAttempt] = {}
    clashes: set[str] = set()  # attempt_ids logged twice with different contents
    for a in attempts:
        if by_id.setdefault(a.attempt_id, a) != a:
            clashes.add(a.attempt_id)
    chains: dict[str, list[MeshAttempt]] = {}
    broken: dict[str, str] = {}
    for a in by_id.values():
        first_id, loop = _chain_root(a, by_id)
        chains.setdefault(first_id, []).append(a)
        if loop:
            broken.setdefault(first_id, loop)

    by_shape: dict[tuple[str, str], list[MeshRow]] = {}
    retry, blocked, orphans, preflight = [], [], [], []
    for first_id, tries in chains.items():
        if first_id in broken:
            continue
        if first_id not in by_id:
            orphans.append(first_id)
            continue
        if clashed := sorted(clashes.intersection(t.attempt_id for t in tries)):
            broken[first_id] = f"logged twice with different contents: {', '.join(clashed)}"
            continue
        tries.sort(key=lambda t: (t.rerun_of is not None, t.seq))
        try:
            verdict = AL_chain_verdict(tries, production)
        except ValueError as e:
            broken[first_id] = str(e)
            continue
        if verdict.preflight:
            preflight.append(verdict.preflight)
        if verdict.pending is Pending.RETRY:
            retry.append(first_id)
        elif verdict.pending is Pending.BLOCKED:
            blocked.append(first_id)
        elif verdict.row is not None:
            by_shape.setdefault((verdict.row.geometry_id, verdict.row.settings_id), []).append(verdict.row)

    rows, conflicts = [], []
    for (geometry_id, settings_id), found in by_shape.items():
        if len({r.label for r in found}) > 1:
            conflicts.append(MeshConflict(geometry_id=geometry_id, settings_id=settings_id, rows=found))
        else:
            rows.append(found[0])
    return MeshRows(rows=rows, conflicts=conflicts, retry=retry, blocked=blocked, orphans=orphans, broken=broken,
                    preflight_attempts=preflight)


def _chain_root(a: MeshAttempt, by_id: dict[str, MeshAttempt]) -> tuple[str, Optional[str]]:
    """
    The attempt_id a try's `rerun_of` links lead back to (a first attempt's, or a missing one's), and
    why the chain is broken if they loop instead: then the loop's smallest attempt_id, the same from
    every try in or leading into it.
    """
    path = [a.attempt_id]
    while a.rerun_of is not None:
        if a.rerun_of in path:  # the guard: a2 -> a3 -> a2 would walk forever
            loop = path[path.index(a.rerun_of):]
            return min(loop), f"rerun_of links loop: {' -> '.join(loop + [a.rerun_of])}"
        if a.rerun_of not in by_id:
            return a.rerun_of, None
        path.append(a.rerun_of)
        a = by_id[a.rerun_of]
    return a.attempt_id, None


def _is_production(attempt: MeshAttempt, production: ProductionMesh) -> bool:
    return attempt.settings_id == production.settings_id and attempt.knobs == production.knobs


def _try_kind(t: MeshAttempt) -> _Try:
    if t.geometry_rejected:
        return _Try.NO_ROW
    if t.aborted:
        return _Try.RETRY
    if t.entry_flag is None:
        raise ValueError(f"attempt {t.attempt_id} reached the mesher but has no entry flag")
    outcome = FLAG_OUTCOME[t.entry_flag]
    if outcome is MeshOutcome.PREFLIGHT:
        return _Try.NO_ROW
    if outcome is MeshOutcome.INFRA:
        return _Try.BLOCKED
    if outcome is MeshOutcome.BY_CAUSE:
        cause = AL_subprocess_cause(t.mesher_log)
        return _Try.C2D_FAIL if cause in (SubprocessCause.TIMEOUT, SubprocessCause.CRASH) else _Try.RETRY
    return _Try.VERDICT


def _verdict_row(t: MeshAttempt) -> MeshRow:
    if FLAG_OUTCOME[t.entry_flag] is MeshOutcome.DESIGN_FAIL:
        return _row(t, 0, ":".join(filter(None, [t.entry_flag.name, ",".join(t.failed_gates)])))
    if t.passed is None:
        raise ValueError(f"attempt {t.attempt_id} meshed but has no production gate verdict")
    return _row(t, 1, None) if t.passed else _row(t, 0, "gates:" + ",".join(t.failed_gates))


def _row(t: MeshAttempt, label: int, failure_mode: Optional[str]) -> MeshRow:
    return MeshRow(geometry_id=t.geometry_id, settings_id=t.settings_id, attempt_id=t.attempt_id,
                   x=AL_cst_to_free(t.cst).tolist(), label=label, failure_mode=failure_mode)
