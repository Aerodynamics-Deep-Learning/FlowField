"""
Tests for `active_learning.labels`: which mesh attempts become p_mesh rows, and with what label, i.e.:

Step 1: Ensure every mesher flag has an owner, and a flag's owner decides the label
    - test_every_flag_is_placed
    - test_passing_the_production_gates_is_1
    - test_success_failing_the_production_gates_is_0
    - test_design_flags_are_0
    - test_preflight_flags_make_no_row
    - test_tooling_flags_block_the_chain_until_fixed
Step 2: Ensure a SUBPROCESS_FAIL is labelled by its cause, and a timeout or crash only after its rerun
    - test_log_prefixes_match_c2d
    - test_launch_error_is_retried
    - test_first_timeout_or_crash_waits_for_its_rerun
    - test_reproduced_failure_is_0
    - test_failure_that_passes_on_rerun_is_labelled_by_the_rerun
    - test_rerun_that_shows_nothing_keeps_the_chain_pending
    - test_tries_must_be_one_chain
Step 3: Ensure only production attempts that reached c2d are labelled
    - test_rule_rejection_makes_no_row
    - test_aborted_slot_is_retried
    - test_non_production_settings_make_no_row
    - test_rows_are_one_per_shape_and_settings
    - test_conflicting_shape_is_reported_not_raised
Step 4: Ensure a row's input is the shape's 18 free values (`test_al_design_vector`), and a row saves as JSON
    - test_free_vector_has_18_values
    - test_rows_round_trip_through_json
Step 5: Ensure chains are put together from the log in any order, and a broken one doesn't stop the build
    - test_rerun_listed_before_its_first_attempt
    - test_reruns_are_judged_in_seq_order
    - test_rerun_without_its_first_attempt_is_an_orphan
    - test_empty_chain_refused
    - test_rerun_of_a_rerun_joins_its_chain
    - test_rerun_of_links_that_loop_are_broken
    - test_seq_is_required
    - test_attempt_logged_twice_alike_counts_once
    - test_broken_chain_is_reported_not_raised
Step 6: Ensure the production settings come from one hashed row
    - test_production_reads_its_settings_row
    - test_production_refuses_a_row_its_id_does_not_hash
    - test_settings_hash_is_the_optimizers
"""

import subprocess
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from src.datagen.active_learning.design_vector import N_FREE, AL_cst_to_free
from src.datagen.active_learning.labels import (
    FLAG_OUTCOME, LOG_CRASH, LOG_LAUNCH, LOG_TIMEOUT, AL_chain_verdict, AL_mesh_training_rows, AL_needs_rerun,
    MeshAttempt, MeshOutcome, MeshRow, Pending, ProductionMesh,
)
from src.datagen.meshing.c2d.run import C2D_generate_mesh
from src.datagen.meshing.c2d.schemas import C2D_MeshingConfig
from src.datagen.meshing.common.schemas import MeshBackend, MeshExitFlag, MeshTopology
from src.datagen.meshing.optimizer.common import settings as optim_settings
from src.datagen.meshing.optimizer.common.constants import TE_GAP_DEFAULT
from src.datagen.meshing.optimizer.common.settings import Optim_settings_hash
from src.datagen.schemas import Airfoil
from tests.active_learning_rows import KNOBS, PRODUCTION, production_settings
from tests.geometry_shapes import CUSP

_DESIGN_FLAGS = [f for f, o in FLAG_OUTCOME.items() if o is MeshOutcome.DESIGN_FAIL]
_PREFLIGHT_FLAGS = [f for f, o in FLAG_OUTCOME.items() if o is MeshOutcome.PREFLIGHT]
_TOOLING_FLAGS = [f for f, o in FLAG_OUTCOME.items() if o is MeshOutcome.INFRA]


def _attempt(flag=MeshExitFlag.SUCCESS, **fields) -> MeshAttempt:
    base = dict(attempt_id="a1", seq=0, geometry_id="cusp", cst=CUSP, settings_id=PRODUCTION.settings_id,
                knobs=dict(PRODUCTION.knobs), entry_flag=flag)
    if flag is MeshExitFlag.SUCCESS:
        base["passed"] = True
    return MeshAttempt(**{**base, **fields})


def _subprocess(log: str, **fields) -> MeshAttempt:
    return _attempt(MeshExitFlag.SUBPROCESS_FAIL, mesher_log=log + " ...\n--- stdout ---\n", **fields)


def _rerun(n: int, make=_attempt, *args, **fields) -> MeshAttempt:
    return make(*args, **{"attempt_id": f"a{n}", "seq": n - 1, "rerun_of": "a1", **fields})


def _verdict(*tries):
    return AL_chain_verdict(list(tries), PRODUCTION)


# region Step 1
def test_every_flag_is_placed():
    assert set(FLAG_OUTCOME) == set(MeshExitFlag)


def test_passing_the_production_gates_is_1():
    v = _verdict(_attempt())

    assert v.pending is None and v.preflight is None and v.row.label == 1 and v.row.failure_mode is None
    assert (v.row.geometry_id, v.row.settings_id, v.row.attempt_id) == ("cusp", PRODUCTION.settings_id, "a1")


def test_success_failing_the_production_gates_is_0():
    # SUCCESS is the lenient entry limits (Jacobian >= 0.10); the study gates are stricter (>= 0.75)
    row = _verdict(_attempt(passed=False, failed_gates=["skew", "jac"])).row

    assert row.label == 0 and row.failure_mode == "gates:skew,jac"
    with pytest.raises(ValueError, match="no production gate verdict"):
        _verdict(_attempt(passed=None))


@pytest.mark.parametrize("flag", _DESIGN_FLAGS)
def test_design_flags_are_0(flag):
    row = _verdict(_attempt(flag, passed=False, failed_gates=["size_ratio"])).row

    assert row.label == 0 and row.failure_mode == f"{flag.name}:size_ratio"


@pytest.mark.parametrize("flag", _PREFLIGHT_FLAGS)
def test_preflight_flags_make_no_row(flag):
    # Settled with no row, and reported: the geometry rules should have caught it, not p_mesh
    v = _verdict(_attempt(flag))
    assert v.row is None and v.pending is None and v.preflight == "a1"
    assert AL_mesh_training_rows([_attempt(flag)], PRODUCTION).preflight_attempts == ["a1"]


@pytest.mark.parametrize("flag", _TOOLING_FLAGS)
def test_tooling_flags_block_the_chain_until_fixed(flag):
    # Shows nothing about the shape, and a retry before the fix would only repeat it: pending, not scheduled
    blocked = [_attempt(flag)]
    assert _verdict(*blocked).row is None and _verdict(*blocked).pending is Pending.BLOCKED
    assert not AL_needs_rerun(blocked, PRODUCTION)
    out = AL_mesh_training_rows(blocked + [_subprocess(LOG_LAUNCH, attempt_id="b1")], PRODUCTION)
    assert out.blocked == ["a1"] and out.retry == ["b1"]

    # After the fix: a retry that again shows nothing is retried now, one that meshes labels the shape
    assert _verdict(*blocked, _rerun(2, _subprocess, LOG_LAUNCH)).pending is Pending.RETRY
    assert _verdict(*blocked, _rerun(2)).row.label == 1
# endregion


# region Step 2
def _c2d_log(tmp_path, **run) -> str:
    """The log `C2D_generate_mesh` writes when `subprocess.run` behaves as given; no exe runs."""
    airfoil = Airfoil(airfoil_name="probe", coords=[[1.0, 0.001], [0.5, 0.05], [0.0, 0.0], [0.5, -0.05],
                                                     [1.0, -0.001]], chord=1.0, le_idx=2)
    with patch("src.datagen.meshing.c2d.run.subprocess.run", **run):
        out = C2D_generate_mesh(airfoil, "/fake/c2d", str(tmp_path), C2D_MeshingConfig(), Re=2e6, timeout=5)
    assert out[0] is None  # no grid in every case here
    return out[-1]


def test_log_prefixes_match_c2d(tmp_path):
    # If c2d's wording changes, the causes would silently read as "no cause"; this pins them to the source
    timeout = subprocess.TimeoutExpired(cmd="c2d", timeout=5, output="", stderr="")
    assert _c2d_log(tmp_path, side_effect=timeout).startswith(LOG_TIMEOUT)
    assert _c2d_log(tmp_path, side_effect=OSError("exec format error")).startswith(LOG_LAUNCH)
    assert _c2d_log(tmp_path, return_value=MagicMock(returncode=139, stdout="", stderr="")).startswith(LOG_CRASH)


@pytest.mark.parametrize("log", [LOG_LAUNCH, "an unrecognised log"])
def test_launch_error_is_retried(log):
    assert _verdict(_subprocess(log)).pending is Pending.RETRY and AL_needs_rerun([_subprocess(log)], PRODUCTION)
    assert _verdict(_subprocess(log), _rerun(2)).row.label == 1  # the retry decides alone


@pytest.mark.parametrize("log", [LOG_TIMEOUT, LOG_CRASH])
def test_first_timeout_or_crash_waits_for_its_rerun(log):
    v = _verdict(_subprocess(log))

    assert v.row is None and v.pending is Pending.RETRY and v.preflight is None


@pytest.mark.parametrize("first, again, mode", [(LOG_TIMEOUT, LOG_TIMEOUT, "c2d_hang"),
                                                (LOG_CRASH, LOG_CRASH, "c2d_crash"),
                                                (LOG_TIMEOUT, LOG_CRASH, "c2d_crash")])
def test_reproduced_failure_is_0(first, again, mode):
    v = _verdict(_subprocess(first), _rerun(2, _subprocess, again))

    assert v.pending is None and v.preflight is None
    assert v.row.label == 0 and v.row.failure_mode == mode and v.row.attempt_id == "a1"


def test_failure_that_passes_on_rerun_is_labelled_by_the_rerun():
    row = _verdict(_subprocess(LOG_CRASH), _rerun(2)).row

    assert row.label == 1 and row.attempt_id == "a2"  # the first failure was the machine's
    failed = _verdict(_subprocess(LOG_CRASH), _rerun(2, _attempt, MeshExitFlag.LOW_QUALITY, failed_gates=["skew"]))
    assert failed.row.label == 0 and failed.row.failure_mode == "LOW_QUALITY:skew"


@pytest.mark.parametrize("nothing, why", [
    (lambda: _rerun(2, _subprocess, LOG_LAUNCH), Pending.RETRY),                               # didn't start
    (lambda: _rerun(2, _attempt, entry_flag=None, passed=None, aborted=True), Pending.RETRY),  # its slot died
    (lambda: _rerun(2, _attempt, MeshExitFlag.FATAL_ERROR), Pending.BLOCKED),                  # a code bug
])
def test_rerun_that_shows_nothing_keeps_the_chain_pending(nothing, why):
    tries = [_subprocess(LOG_CRASH), nothing()]

    assert _verdict(*tries).row is None and _verdict(*tries).pending is why  # pending, not dropped
    assert _verdict(*tries, _rerun(3, _subprocess, LOG_CRASH)).row.failure_mode == "c2d_crash"
    assert _verdict(*tries, _rerun(3)).row.label == 1


def test_tries_must_be_one_chain():
    with pytest.raises(ValueError, match="not one chain"):
        _verdict(_subprocess(LOG_CRASH), _subprocess(LOG_CRASH, attempt_id="a2"))
    with pytest.raises(ValueError, match="other settings"):
        _verdict(_subprocess(LOG_CRASH), _rerun(2, settings_id="probe0042"))
# endregion


# region Step 3
def test_rule_rejection_makes_no_row():
    # Caught by a rule, as it should be: not reported as a pre-flight hit
    v = _verdict(_attempt(entry_flag=None, passed=None, geometry_rejected=True))
    assert v.row is None and v.pending is None and v.preflight is None
    with pytest.raises(ValueError, match="no entry flag"):  # reached the mesher, yet nothing recorded
        _verdict(_attempt(entry_flag=None))


def test_aborted_slot_is_retried():
    v = _verdict(_attempt(entry_flag=None, passed=None, aborted=True))

    assert v.row is None and v.pending is Pending.RETRY


@pytest.mark.parametrize("fields", [{"settings_id": "probe0042"}, {"knobs": {**PRODUCTION.knobs, "jmax": 160.0}}])
def test_non_production_settings_make_no_row(fields):
    v = _verdict(_attempt(**fields))

    assert v.row is None and v.pending is None and v.preflight is None


def test_rows_are_one_per_shape_and_settings():
    attempts = [_attempt(attempt_id="a1"), _attempt(attempt_id="b1"), _attempt(attempt_id="c1", geometry_id="other"),
                _subprocess(LOG_CRASH, attempt_id="d1", geometry_id="waiting")]
    out = AL_mesh_training_rows(attempts, PRODUCTION)

    assert sorted(r.geometry_id for r in out.rows) == ["cusp", "other"]
    assert out.conflicts == [] and out.retry == ["d1"] and out.blocked == []


def test_conflicting_shape_is_reported_not_raised():
    # Hung twice on a busy machine, meshed later on a quiet one: excluded and reported, the build goes on
    attempts = [_subprocess(LOG_TIMEOUT, attempt_id="a1"), _subprocess(LOG_TIMEOUT, attempt_id="a2", rerun_of="a1"),
                _attempt(attempt_id="b1"), _attempt(attempt_id="c1", geometry_id="other")]
    out = AL_mesh_training_rows(attempts, PRODUCTION)

    assert [r.geometry_id for r in out.rows] == ["other"]
    (conflict,) = out.conflicts
    assert conflict.geometry_id == "cusp" and conflict.settings_id == PRODUCTION.settings_id
    assert sorted((r.attempt_id, r.label, r.failure_mode) for r in conflict.rows) == [("a1", 0, "c2d_hang"),
                                                                                       ("b1", 1, None)]
# endregion


# region Step 4
def test_free_vector_has_18_values():
    x = _verdict(_attempt()).row.x_array()

    assert N_FREE == 18 and x.shape == (18,) and x.dtype == np.float64
    assert x[0] == CUSP.upper[0] and x[9] == CUSP.lower[1] and x[-1] == CUSP.le_weight


def test_rows_round_trip_through_json():
    row = _verdict(_subprocess(LOG_CRASH), _rerun(2, _subprocess, LOG_CRASH)).row

    assert MeshRow.model_validate_json(row.model_dump_json()) == row
    assert row.x == AL_cst_to_free(CUSP).tolist()  # floats survive JSON exactly
# endregion


# region Step 5
def test_rerun_listed_before_its_first_attempt():
    first, rerun = _subprocess(LOG_CRASH, seq=0), _rerun(2, _subprocess, LOG_CRASH, seq=1)
    normal, swapped = (AL_mesh_training_rows(a, PRODUCTION) for a in ([first, rerun], [rerun, first]))

    assert swapped == normal  # x is a list now, so whole results compare
    assert [(r.attempt_id, r.label, r.failure_mode) for r in normal.rows] == [("a1", 0, "c2d_crash")]
    assert not (normal.conflicts or normal.retry or normal.blocked or normal.orphans or normal.broken
                or normal.preflight_attempts)


def test_reruns_are_judged_in_seq_order():
    # Listed order says crash twice (0); seq says the meshing rerun came first, so it labels the shape (1)
    first = _subprocess(LOG_CRASH, seq=0)
    later_crash, earlier_mesh = _rerun(2, _subprocess, LOG_CRASH, seq=2), _rerun(3, seq=1)
    (row,) = AL_mesh_training_rows([first, later_crash, earlier_mesh], PRODUCTION).rows

    assert row.label == 1 and row.attempt_id == "a3"


def test_rerun_without_its_first_attempt_is_an_orphan():
    attempts = [_attempt(attempt_id="z2", rerun_of="z1", geometry_id="lost"), _attempt(geometry_id="other")]
    out = AL_mesh_training_rows(attempts, PRODUCTION)  # doesn't raise

    assert out.orphans == ["z1"]
    assert [r.geometry_id for r in out.rows] == ["other"] and out.retry == [] and out.broken == {}


def test_empty_chain_refused():
    with pytest.raises(ValueError, match="at least one attempt"):
        AL_chain_verdict([], PRODUCTION)


def test_rerun_of_a_rerun_joins_its_chain():
    # a1 crashed, its rerun a2 didn't launch, and a3 reran a2: one chain, labelled by a3
    first, launch = _subprocess(LOG_CRASH), _rerun(2, _subprocess, LOG_LAUNCH)
    meshed = _rerun(3, rerun_of="a2")
    out = AL_mesh_training_rows([meshed, launch, first], PRODUCTION)

    assert [(r.attempt_id, r.label) for r in out.rows] == [("a3", 1)]
    assert out.retry == [] and out.orphans == [] and out.broken == {}
    with pytest.raises(ValueError, match="not one chain"):  # a rerun before the try it repeats
        AL_chain_verdict([first, meshed, launch], PRODUCTION)


def test_rerun_of_links_that_loop_are_broken():
    # a4 leads into the loop a2 -> a3 -> a2: all three are one broken chain, named by the loop's smallest id
    looped = [_attempt(attempt_id="a4", seq=3, rerun_of="a3"), _attempt(attempt_id="a3", seq=2, rerun_of="a2"),
              _attempt(attempt_id="a2", seq=1, rerun_of="a3"), _attempt(attempt_id="b1", geometry_id="other")]
    out = AL_mesh_training_rows(looped, PRODUCTION)  # doesn't hang or raise

    assert list(out.broken) == ["a2"] and "rerun_of links loop" in out.broken["a2"]
    assert [r.geometry_id for r in out.rows] == ["other"] and out.orphans == []


def test_seq_is_required():
    # Defaulted, every rerun would tie at 0 and be judged in list order without a word
    fields = _attempt().model_dump(exclude={"seq"})

    with pytest.raises(ValidationError, match="seq"):
        MeshAttempt(**fields)


def test_attempt_logged_twice_alike_counts_once():
    # A log replayed after a crash: the same rows again
    attempts = [_subprocess(LOG_CRASH), _rerun(2), _subprocess(LOG_CRASH), _rerun(2)]
    out = AL_mesh_training_rows(attempts, PRODUCTION)

    assert [(r.attempt_id, r.label) for r in out.rows] == [("a2", 1)] and out.broken == {}


@pytest.mark.parametrize("bad, reason", [
    ([_attempt(entry_flag=None)], "no entry flag"),
    ([_attempt(passed=None)], "no production gate verdict"),
    ([_subprocess(LOG_CRASH), _rerun(2, settings_id="probe0042")], "other settings"),
    ([_attempt(), _attempt(passed=False, failed_gates=["skew"])], "logged twice with different contents: a1"),
])
def test_broken_chain_is_reported_not_raised(bad, reason):
    out = AL_mesh_training_rows(bad + [_attempt(attempt_id="b1", geometry_id="other")], PRODUCTION)

    assert list(out.broken) == ["a1"] and reason in out.broken["a1"]
    assert [r.geometry_id for r in out.rows] == ["other"]  # the rest of the build goes on
# endregion


# region Step 6
def test_production_reads_its_settings_row():
    row = production_settings(te_gap=0.003)
    prod = ProductionMesh(settings=row, knobs=KNOBS)

    assert prod.settings_id == row["settings_id"] and prod.te_gap == 0.003
    assert prod.backend is MeshBackend.C2D and prod.topology is MeshTopology.OGRD
    assert PRODUCTION.te_gap == TE_GAP_DEFAULT


def test_production_refuses_a_row_its_id_does_not_hash():
    edited = {**production_settings(), "te_gap": 0.003}  # changed after hashing: the id says another gap
    with pytest.raises(ValidationError, match="settings_id .* doesn't match"):
        ProductionMesh(settings=edited, knobs=KNOBS)

    missing = production_settings()
    del missing["te_gap"]
    missing["settings_id"] = Optim_settings_hash(missing)
    with pytest.raises(ValidationError, match="te_gap"):
        ProductionMesh(settings=missing, knobs=KNOBS)


def test_settings_hash_is_the_optimizers():
    with patch.object(optim_settings, "Optim_mesher_sha256", return_value="0" * 64):
        row = optim_settings.Optim_build_settings()

    assert Optim_settings_hash(row) == row["settings_id"]  # the id is over every column but itself
    assert ProductionMesh(settings=row, knobs=KNOBS).settings_id == row["settings_id"]
# endregion
