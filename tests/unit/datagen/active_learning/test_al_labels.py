"""
Tests for `active_learning.labels`: which mesh attempts become p_mesh rows, and with what label, i.e.:

Step 1: Ensure every mesher flag has an owner, and a flag's owner decides the label
    - test_every_flag_is_placed
    - test_passing_the_production_gates_is_1
    - test_success_failing_the_production_gates_is_0
    - test_design_flags_are_0
    - test_preflight_flags_make_no_row
    - test_tooling_flags_are_retried
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
Step 4: Ensure a row's input is the shape's 18 free values (`test_al_design_vector`)
    - test_free_vector_has_18_values
Step 5: Ensure chains are put together from the log in any order, and a broken one doesn't stop the build
    - test_rerun_listed_before_its_first_attempt
    - test_reruns_are_judged_in_seq_order
    - test_rerun_without_its_first_attempt_is_an_orphan
    - test_empty_chain_refused
    - test_rerun_of_a_rerun_joins_its_chain
    - test_rerun_of_links_that_loop_are_refused
    - test_seq_is_required
"""

import subprocess
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from src.datagen.active_learning.design_vector import N_FREE
from src.datagen.active_learning.labels import (
    FLAG_OUTCOME, LOG_CRASH, LOG_LAUNCH, LOG_TIMEOUT, AL_chain_verdict, AL_mesh_training_rows, AL_needs_rerun,
    MeshAttempt, MeshOutcome, ProductionMesh,
)
from src.datagen.meshing.c2d.run import C2D_generate_mesh
from src.datagen.meshing.c2d.schemas import C2D_MeshingConfig
from src.datagen.meshing.common.schemas import MeshExitFlag
from src.datagen.schemas import Airfoil
from tests.geometry_shapes import CUSP

PRODUCTION = ProductionMesh(settings_id="prod0001", knobs={"jmax": 147.0, "dx_mid": 0.006})
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
    row, pending, preflight = _verdict(_attempt())

    assert not pending and preflight is None and row.label == 1 and row.failure_mode is None
    assert (row.geometry_id, row.settings_id, row.attempt_id) == ("cusp", PRODUCTION.settings_id, "a1")


def test_success_failing_the_production_gates_is_0():
    # SUCCESS is the lenient entry limits (Jacobian >= 0.10); the study gates are stricter (>= 0.75)
    row, _, _ = _verdict(_attempt(passed=False, failed_gates=["skew", "jac"]))

    assert row.label == 0 and row.failure_mode == "gates:skew,jac"
    with pytest.raises(ValueError, match="no production gate verdict"):
        _verdict(_attempt(passed=None))


@pytest.mark.parametrize("flag", _DESIGN_FLAGS)
def test_design_flags_are_0(flag):
    row, _, _ = _verdict(_attempt(flag, passed=False, failed_gates=["size_ratio"]))

    assert row.label == 0 and row.failure_mode == f"{flag.name}:size_ratio"


@pytest.mark.parametrize("flag", _PREFLIGHT_FLAGS)
def test_preflight_flags_make_no_row(flag):
    # Settled with no row, and reported: the geometry rules should have caught it, not p_mesh
    assert _verdict(_attempt(flag)) == (None, False, "a1")
    assert AL_mesh_training_rows([_attempt(flag)], PRODUCTION).preflight == ["a1"]


@pytest.mark.parametrize("flag", _TOOLING_FLAGS)
def test_tooling_flags_are_retried(flag):
    assert _verdict(_attempt(flag)) == (None, True, None)  # shows nothing about the shape: pending, not dropped
    assert _verdict(_attempt(flag), _rerun(2)).row.label == 1
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
    assert AL_needs_rerun([_subprocess(log)], PRODUCTION)
    assert _verdict(_subprocess(log), _rerun(2)).row.label == 1  # the retry decides alone


@pytest.mark.parametrize("log", [LOG_TIMEOUT, LOG_CRASH])
def test_first_timeout_or_crash_waits_for_its_rerun(log):
    assert _verdict(_subprocess(log)) == (None, True, None)


@pytest.mark.parametrize("first, again, mode", [(LOG_TIMEOUT, LOG_TIMEOUT, "c2d_hang"),
                                                (LOG_CRASH, LOG_CRASH, "c2d_crash"),
                                                (LOG_TIMEOUT, LOG_CRASH, "c2d_crash")])
def test_reproduced_failure_is_0(first, again, mode):
    row, pending, preflight = _verdict(_subprocess(first), _rerun(2, _subprocess, again))

    assert not pending and preflight is None and row.label == 0 and row.failure_mode == mode and row.attempt_id == "a1"


def test_failure_that_passes_on_rerun_is_labelled_by_the_rerun():
    row, _, _ = _verdict(_subprocess(LOG_CRASH), _rerun(2))

    assert row.label == 1 and row.attempt_id == "a2"  # the first failure was the machine's
    failed = _verdict(_subprocess(LOG_CRASH), _rerun(2, _attempt, MeshExitFlag.LOW_QUALITY, failed_gates=["skew"]))
    assert failed.row.label == 0 and failed.row.failure_mode == "LOW_QUALITY:skew"


@pytest.mark.parametrize("nothing", [
    lambda: _rerun(2, _subprocess, LOG_LAUNCH),                                # the rerun didn't start
    lambda: _rerun(2, _attempt, entry_flag=None, passed=None, aborted=True),   # its slot died
    lambda: _rerun(2, _attempt, MeshExitFlag.FATAL_ERROR),                     # a code bug
])
def test_rerun_that_shows_nothing_keeps_the_chain_pending(nothing):
    tries = [_subprocess(LOG_CRASH), nothing()]

    assert AL_needs_rerun(tries, PRODUCTION)  # retried, not dropped
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
    assert _verdict(_attempt(entry_flag=None, passed=None, geometry_rejected=True)) == (None, False, None)
    with pytest.raises(ValueError, match="no entry flag"):  # reached the mesher, yet nothing recorded
        _verdict(_attempt(entry_flag=None))


def test_aborted_slot_is_retried():
    assert _verdict(_attempt(entry_flag=None, passed=None, aborted=True)) == (None, True, None)


def test_non_production_settings_make_no_row():
    assert _verdict(_attempt(settings_id="probe0042")) == (None, False, None)
    assert _verdict(_attempt(knobs={**PRODUCTION.knobs, "jmax": 160.0})) == (None, False, None)


def test_rows_are_one_per_shape_and_settings():
    attempts = [_attempt(attempt_id="a1"), _attempt(attempt_id="b1"), _attempt(attempt_id="c1", geometry_id="other"),
                _subprocess(LOG_CRASH, attempt_id="d1", geometry_id="waiting")]
    out = AL_mesh_training_rows(attempts, PRODUCTION)

    assert sorted(r.geometry_id for r in out.rows) == ["cusp", "other"]
    assert out.conflicts == [] and out.pending == ["d1"]


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
    x = _verdict(_attempt()).row.x

    assert N_FREE == 18 and x.shape == (18,) and x.dtype == np.float64
    assert x[0] == CUSP.upper[0] and x[9] == CUSP.lower[1] and x[-1] == CUSP.le_weight
# endregion


# region Step 5
def test_rerun_listed_before_its_first_attempt():
    first, rerun = _subprocess(LOG_CRASH, seq=0), _rerun(2, _subprocess, LOG_CRASH, seq=1)
    normal, swapped = (AL_mesh_training_rows(a, PRODUCTION) for a in ([first, rerun], [rerun, first]))

    assert [(r.attempt_id, r.label, r.failure_mode) for r in swapped.rows] == \
           [(r.attempt_id, r.label, r.failure_mode) for r in normal.rows] == [("a1", 0, "c2d_crash")]
    assert swapped[1:] == normal[1:] == ([], [], [], [])  # no conflicts, pending, orphans or preflight hits


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
    assert [r.geometry_id for r in out.rows] == ["other"] and out.pending == []


def test_empty_chain_refused():
    with pytest.raises(ValueError, match="at least one attempt"):
        AL_chain_verdict([], PRODUCTION)


def test_rerun_of_a_rerun_joins_its_chain():
    # a1 crashed, its rerun a2 didn't launch, and a3 reran a2: one chain, labelled by a3
    first, launch = _subprocess(LOG_CRASH), _rerun(2, _subprocess, LOG_LAUNCH)
    meshed = _rerun(3, rerun_of="a2")
    out = AL_mesh_training_rows([meshed, launch, first], PRODUCTION)

    assert [(r.attempt_id, r.label) for r in out.rows] == [("a3", 1)]
    assert out.pending == [] and out.orphans == []
    with pytest.raises(ValueError, match="not one chain"):  # a rerun before the try it repeats
        AL_chain_verdict([first, meshed, launch], PRODUCTION)


def test_rerun_of_links_that_loop_are_refused():
    looped = [_attempt(attempt_id="a2", seq=1, rerun_of="a3"), _attempt(attempt_id="a3", seq=2, rerun_of="a2")]

    with pytest.raises(ValueError, match="rerun_of links loop: a2 -> a3 -> a2"):
        AL_mesh_training_rows(looped, PRODUCTION)


def test_seq_is_required():
    # Defaulted, every rerun would tie at 0 and be judged in list order without a word
    fields = _attempt().model_dump(exclude={"seq"})

    with pytest.raises(ValidationError, match="seq"):
        MeshAttempt(**fields)
# endregion
