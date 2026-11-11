"""
p_mesh training labels: a shape's logged mesh attempts become one training row, or nothing.

p_mesh is P(a shape that passed the geometry rules meshes under the production settings): one c2d run
with the frozen knobs, judged by the production gates (`Optim_failed_gates`, the `passed` column), at the
fixed design Re every shape is meshed at, so the input is the shape alone. So a shape gets a label only if:
    - it reached c2d: a rule rejection (geometry rules, the sampler's pre-check) never becomes a row
    - it ran under the production settings: settings_id and knobs both match, optimizer probes don't
    - its outcome is the shape's: a tooling failure (no exe, a launch error, an exception, an aborted
      slot) is never a 0

A shape's tries form a chain: its first attempt and every rerun naming it in `rerun_of`, in log order.
Tries that show nothing about the shape are transient: skipped, and the chain retried. That is a launch
error, a log naming no cause, an aborted slot, and a tooling error (FATAL_ERROR, no exe, a caller bug),
which needs fixing before its retry. The first informative try decides: a mesh verdict labels the shape,
a rule rejection or pre-flight flag makes no row, and a timeout or crash, which can be the shape's or a
busy machine's, waits for one informative rerun. A chain still waiting is pending (`AL_needs_rerun`),
reported, never dropped.

The model's input is the 18 free CST values (`AL_cst_to_free`): the upper A_0..A_8, the lower A_1..A_8
(its A_0 is -upper A_0, `CSTParams`' shared nose) and the LE weight.
"""

from enum import Enum
from typing import Iterable, NamedTuple, Optional

import numpy as np
from pydantic import BaseModel

from src.datagen.geometry.common.constants import CST_N1, CST_N2, CST_ORDER
from src.datagen.geometry.common.schemas import CSTParams, Nose
from src.datagen.meshing.common.schemas import MeshExitFlag

# Length of the model's input: 9 upper + 8 lower + the LE weight at order 8
N_FREE = 2 * CST_ORDER + 2

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
    """The settings p_mesh's labels hold for: the study's settings_id and the frozen c2d knobs."""
    settings_id: str
    knobs: dict[str, float]


class MeshAttempt(BaseModel):
    """
    One logged mesh try, as the run log will record it. `entry_flag` is None when it never reached
    the mesher (a rule rejection or an aborted slot); `passed` and `failed_gates` are set when it meshed.
    """
    attempt_id: str
    seq: int = 0 # tracks the position in the run log 
    geometry_id: str
    cst: CSTParams
    settings_id: str
    knobs: dict[str, float]
    geometry_rejected: bool = False     # the geometry rules or the sampler's pre-check refused it
    aborted: bool = False               # its worker slot died, went silent or hit its cap
    entry_flag: Optional[MeshExitFlag] = None
    passed: Optional[bool] = None
    failed_gates: list[str] = []
    mesher_log: Optional[str] = None
    rerun_of: Optional[str] = None      # the first attempt of the chain this try repeats


class MeshRow(BaseModel):
    """One p_mesh training example."""
    model_config = {"arbitrary_types_allowed": True}

    geometry_id: str
    settings_id: str
    attempt_id: str                     # the try that decided it
    x: np.ndarray                       # (N_FREE,) `AL_cst_to_free`
    label: int                          # 1: passes the production gates
    failure_mode: Optional[str] = None  # why a 0 failed: the flag and the gates, or c2d_crash / c2d_hang


class MeshConflict(BaseModel):
    """A shape labelled both ways under one settings_id, left out of the training rows."""
    geometry_id: str
    settings_id: str
    rows: list[MeshRow]


class Verdict(NamedTuple):
    """What a chain of tries decided: a row, nothing (`row` None), or nothing yet (`pending`)."""
    row: Optional[MeshRow]
    pending: bool
    preflight: Optional[str]= None


class MeshRows(NamedTuple):
    """The training rows, the shapes left out for conflicting, and the chains still awaiting a try."""
    rows: list[MeshRow]
    conflicts: list[MeshConflict]
    pending: list[str]  # first attempt_ids
    orphans: list[str] # # chains whose first attempt isn't in the log
    preflight: list[str] # # attempt_ids the mesher refused before meshing



class _Try(str, Enum):
    """What one try shows on its own."""
    TRANSIENT = "transient"  # nothing about the shape (tooling, an aborted slot): retry
    NO_ROW = "no_row"        # a rule rejection or pre-flight flag: settles the chain with no row
    C2D_FAIL = "c2d_fail"    # a timeout or crash: needs a second informative try
    VERDICT = "verdict"      # a mesh the gates judged: labels the shape


def AL_cst_to_free(cst: CSTParams) -> np.ndarray:
    """
    The 18 free values of a design: upper A_0..A_N, lower A_1..A_N, the LE weight.

    Args:
        cst (CSTParams): A LEM shape at CST_ORDER with the pipeline's class exponents

    Returns:
        np.ndarray: (N_FREE,) float64

    Raises:
        ValueError: For another nose (SEPARATE has a second, independent nose), order, or n1/n2,
            which the vector leaves out: two shapes differing only there would share one
    """
    if cst.nose is not Nose.LEM:
        raise ValueError(f"designs are LEM, got nose={cst.nose.value}: a SEPARATE shape has two noses, "
                         f"not the {N_FREE} free values p_mesh takes")
    if cst.order != CST_ORDER:
        raise ValueError(f"designs are order {CST_ORDER}, got {cst.order}")
    if (cst.n1, cst.n2) != (CST_N1, CST_N2):
        raise ValueError(f"designs have class exponents n1={CST_N1}, n2={CST_N2}, got n1={cst.n1}, n2={cst.n2}: "
                         f"the free values leave them out")
    return np.array([*cst.upper, *cst.lower[1:], cst.le_weight], dtype=np.float64)


def AL_free_to_cst(x: np.ndarray) -> CSTParams:
    """
    The design behind 18 free values, `AL_cst_to_free`'s inverse; `CSTParams` derives the lower A_0.

    Args:
        x (np.ndarray): (N_FREE,) free values

    Returns:
        CSTParams: The LEM shape
    """
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (N_FREE,):
        raise ValueError(f"need {N_FREE} free values, got shape {x.shape}")
    n = CST_ORDER + 1
    return CSTParams(n1=CST_N1, n2=CST_N2, upper=x[:n].tolist(), lower=x[n:2 * n - 1].tolist(),
                     le_weight=float(x[-1]), nose=Nose.LEM)


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
        tries (list[MeshAttempt]): The first attempt, then its reruns in log order
        production (ProductionMesh): The settings labels hold for

    Returns:
        Verdict: The row of the first try that labels the shape; no row for a rule rejection, a
            pre-flight flag, or non-production settings; pending while only transient tries, or one
            timeout or crash and transient tries, are in

    Raises:
        ValueError: If the tries aren't one chain, or a try is missing what its kind needs
    """
    if not tries:
        raise ValueError("chain needs at least one attempt, got none")
    first = tries[0]


    first = tries[0]
    if first.rerun_of is not None or any(t.rerun_of != first.attempt_id for t in tries[1:]):
        raise ValueError(f"not one chain: {[(t.attempt_id, t.rerun_of) for t in tries]}")
    if not _is_production(first, production):
        return Verdict(None, False)

    failed: Optional[MeshAttempt] = None  # the first timeout or crash, awaiting its rerun
    for t in tries:
        if not _is_production(t, production):
            raise ValueError(f"rerun {t.attempt_id} ran under other settings than its first attempt")
        kind = _try_kind(t)
        if kind is _Try.TRANSIENT:
            continue
        if kind is _Try.NO_ROW:
            return Verdict(None, False, None if t.geometry_rejected else t.attempt_id)
        if kind is _Try.VERDICT:
            return Verdict(_verdict_row(t), False)  # a rerun that meshes labels the shape itself
        if failed is None:
            failed = t
            continue
        both_hung = all(AL_subprocess_cause(f.mesher_log) is SubprocessCause.TIMEOUT for f in (failed, t))
        return Verdict(_row(failed, 0, "c2d_hang" if both_hung else "c2d_crash"), False)
    return Verdict(None, True)


def AL_needs_rerun(tries: list[MeshAttempt], production: ProductionMesh) -> bool:
    """Whether a chain is still undecided: its next try should be scheduled."""
    return AL_chain_verdict(tries, production).pending


def AL_mesh_training_rows(attempts: Iterable[MeshAttempt], production: ProductionMesh) -> MeshRows:
    """
    Every row a set of attempts makes, one per (geometry_id, settings_id).

    A shape labelled both ways (say it hung twice on a busy machine and meshed later on a quiet one) is
    left out of the rows and reported, so one bad shape can't stop the build; chains still undecided
    are listed by their first attempt_id.

    If a the first attempt is missing for any reason in the chain, there are added as orphan chains 

    Args:
        attempts (Iterable[MeshAttempt]): Logged tries, in log order
        production (ProductionMesh): The settings labels hold for

    Returns:
        MeshRows: The rows, the conflicts, and the pending chains
    """
    chains: dict[str, list[MeshAttempt]] = {}
    for a in attempts:
        chains.setdefault(a.rerun_of or a.attempt_id, []).append(a)

    by_shape: dict[tuple[str, str], list[MeshRow]] = {}
    pending, orphans, preflight = [], [], []
    for first_id, tries in chains.items():
        tries.sort(key=lambda t: (t.rerun_of is not None, t.seq)) # sort the chain
        if tries[0].rerun_of is not None: # add it as a orphan if the fist attmept is missinng on the chain     
            orphans.append(first_id)
            continue
        verdict = AL_chain_verdict(tries, production)
        if verdict.preflight:
            preflight.append(verdict.preflight)
        if verdict.pending:
            pending.append(first_id)
        elif verdict.row is not None:
            by_shape.setdefault((verdict.row.geometry_id, verdict.row.settings_id), []).append(verdict.row)

    rows, conflicts = [], []
    for (geometry_id, settings_id), found in by_shape.items():
        if len({r.label for r in found}) > 1:
            conflicts.append(MeshConflict(geometry_id=geometry_id, settings_id=settings_id, rows=found))
        else:
            rows.append(found[0])
    return MeshRows(rows, conflicts, pending, orphans)


def _is_production(attempt: MeshAttempt, production: ProductionMesh) -> bool:
    return attempt.settings_id == production.settings_id and attempt.knobs == production.knobs


def _try_kind(t: MeshAttempt) -> _Try:
    if t.geometry_rejected:
        return _Try.NO_ROW
    if t.aborted:
        return _Try.TRANSIENT
    if t.entry_flag is None:
        raise ValueError(f"attempt {t.attempt_id} reached the mesher but has no entry flag")
    outcome = FLAG_OUTCOME[t.entry_flag]
    if outcome is MeshOutcome.PREFLIGHT:
        return _Try.NO_ROW
    if outcome is MeshOutcome.INFRA:
        return _Try.TRANSIENT
    if outcome is MeshOutcome.BY_CAUSE:
        cause = AL_subprocess_cause(t.mesher_log)
        return _Try.C2D_FAIL if cause in (SubprocessCause.TIMEOUT, SubprocessCause.CRASH) else _Try.TRANSIENT
    return _Try.VERDICT


def _verdict_row(t: MeshAttempt) -> MeshRow:
    if FLAG_OUTCOME[t.entry_flag] is MeshOutcome.DESIGN_FAIL:
        return _row(t, 0, ":".join(filter(None, [t.entry_flag.name, ",".join(t.failed_gates)])))
    if t.passed is None:
        raise ValueError(f"attempt {t.attempt_id} meshed but has no production gate verdict")
    return _row(t, 1, None) if t.passed else _row(t, 0, "gates:" + ",".join(t.failed_gates))


def _row(t: MeshAttempt, label: int, failure_mode: Optional[str]) -> MeshRow:
    return MeshRow(geometry_id=t.geometry_id, settings_id=t.settings_id, attempt_id=t.attempt_id,
                   x=AL_cst_to_free(t.cst), label=label, failure_mode=failure_mode)
