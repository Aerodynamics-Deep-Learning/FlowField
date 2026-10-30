"""
Tests for `meshing.optimizer.executor`, one config into one run row, i.e.:

Step 1: Ensure every outcome becomes the right row, and only setup failures raise
    - test_a_passing_mesh_is_recorded_whole
    - test_an_invalid_config_is_recorded_without_meshing
    - test_a_rejected_sample_is_not_meshed
    - test_dominance_skips_at_and_above_the_threshold
    - test_a_mesher_failure_is_recorded
    - test_a_fatal_error_keeps_the_runners_exception
    - test_crashes_are_recorded_not_raised
    - test_setup_flags_are_recorded_then_raised
Step 2: Ensure the verdict reads the settings row's gates, in a fixed order
    - test_each_gate_fails_alone
    - test_gates_fail_in_order
    - test_gates_and_limits_come_from_the_settings_row
    - test_the_mesher_gets_the_study_input
Step 3: Ensure the executor's own mechanics: cache, cleanup, rows the store takes, context
    - test_a_finished_config_is_cached_for_the_ladder
    - test_dominated_and_crashed_runs_are_not_cached
    - test_work_dirs_are_removed
    - test_every_status_fits_the_store
    - test_run_context_create
Step 4: Ensure NOMAD is told the right numbers
    - test_nomad_outputs_for_a_meshed_row
    - test_rejected_rows_keep_their_barriers
    - test_failed_evaluations_return_none
    - test_unmeasured_deviation_is_a_large_violation
    - test_unknown_objective_raises
Step 5: Ensure a failure keeps the reason the meshing code only logs
    - test_the_entry_points_catch_all_keeps_its_traceback
    - test_an_unreadable_mesh_keeps_the_quality_passs_reason
    - test_a_setup_flag_carries_its_logged_reason
    - test_a_usable_mesh_drops_what_was_logged

The sampler runs for real on naca0012's study contour; `Common_GenerateMesh` is a stand-in that writes
a `.su2` and a log into the run's folder. The real mesher is in the integration tier.
"""

import hashlib
import json
import logging
import os
import pickle
import socket
import uuid
from types import SimpleNamespace

import pytest

from src.datagen.geometry.common.constants import DX_MAX
from src.datagen.geometry.sample.checks import SamplerAssertionError
from src.datagen.meshing.c2d.schemas import C2D_ExitFlag, C2D_MeshingConfig
from src.datagen.meshing.common import entry
from src.datagen.meshing.common.schemas import (
    MeshBackend, MeshExitFlag, MeshGeoDeviationSummary, MeshQualitySummary, MeshTopology, MetricDistribution,
    QualityLimits,
)
from src.datagen.meshing.optimizer import executor as executor_module
from src.datagen.meshing.optimizer.constants import PINNED, SMOOTHING_DEFAULTS
from src.datagen.meshing.optimizer.executor import (
    RunContext, RunExecutor, StudySetupError, failed_gates, nomad_outputs,
)
from src.datagen.meshing.optimizer.sampling import build_study_contour
from src.datagen.meshing.optimizer.settings import build_settings, config_schema_fp
from src.datagen.meshing.optimizer.store import RUN_COLUMNS, RunStore
from tests.geometry_shapes import stored_shape

GOLDEN = {"dx_mid": DX_MAX, "le_ratio": 0.025, "te_ratio": 1.0, "cluster_width": 0.026, "jmax": 147,
          **SMOOTHING_DEFAULTS}
_SU2 = b"NPOIN= 1\r\n0 0 0\r\n"  # CRLF, so the hash's folding shows
_SETUP_FLAGS = [MeshExitFlag.EXECUTABLE_NOT_FOUND, MeshExitFlag.INPUT_TENSOR_FAIL, MeshExitFlag.INPUT_CLOSURE_FAIL,
                MeshExitFlag.INPUT_FREESTREAM_FAIL, MeshExitFlag.INPUT_UNKNOWN_SOLVER]


def _dist(**kw):
    return MetricDistribution(**{"p50": 0.5, "t99": 0.6, "t999": 0.7, "n_bad": 0, "x": 1.003, "y": -0.012,
                                 "dwall": 0.0095, **kw})


def _quality(**kw):
    """A summary that clears every study gate, with `kw` on top."""
    base = dict(ncell=68_328, neg=0, max_skew=0.25, min_ortho=0.91, min_jac=0.91, max_ar=9e3, max_size_ratio=1.2,
                acceptable=True, skew_dist=_dist(p50=0.006), ortho_dist=_dist(), jac_dist=_dist(),
                size_ratio_dist=_dist(x=2.2, y=-16.6, dwall=16.7))
    return MeshQualitySummary(**{**base, **kw})


def _geo(**kw):
    base = dict(n_boundary_nodes=468, max_dev_mesh_to_input=1e-8, max_dev_input_to_mesh=2e-8,
                rms_dev_mesh_to_input=1e-9, rms_dev_input_to_mesh=2e-9, chord=1.0, acceptable=True)
    return MeshGeoDeviationSummary(**{**base, **kw})


class _Mesher:
    """
    A stand-in for `Common_GenerateMesh`: writes a `.su2`, a log and (given `exception`) an exception file
    into the run's folder, listing them as c2d does; `listed=False` reports an empty `verbose_list`, as
    the entry's own catch-all does. Keeps its inputs.
    """

    def __init__(self, flag=MeshExitFlag.SUCCESS, quality="good", geo="good", su2=True, raises=None,
                 exception=None, listed=True):
        self.flag, self.su2, self.raises, self.exception, self.listed = flag, su2, raises, exception, listed
        self.quality = _quality() if quality == "good" else quality
        self.geo = _geo() if geo == "good" else geo
        self.calls = []

    def __call__(self, mesh_in):
        self.calls.append(mesh_in)
        if self.raises:
            raise self.raises
        log = os.path.join(mesh_in.working_dir, "c2d_log.txt")
        with open(log, "w") as f:
            f.write("c2d exited 0")
        third = None
        if self.exception:
            third = os.path.join(mesh_in.working_dir, "c2d_exception.txt")
            with open(third, "w") as f:
                f.write(self.exception)
        mesh_path = None
        if self.su2:
            mesh_path = os.path.join(mesh_in.working_dir, "mesh.su2")
            with open(mesh_path, "wb") as f:
                f.write(_SU2)
        return SimpleNamespace(flag=self.flag, quality=self.quality, geo_dev=self.geo, mesh_path=mesh_path,
                               num_nodes=68_796 if self.quality else None,
                               verbose_list=[log, None, third] if self.listed else [])


@pytest.fixture
def settings(tmp_path, monkeypatch):
    exe = tmp_path / "c2d.exe"
    exe.write_bytes(b"not really c2d")
    monkeypatch.setattr("src.datagen.meshing.optimizer.settings.find_tool", lambda key: str(exe))
    return build_settings()


@pytest.fixture(scope="module")
def contour():
    return build_study_contour(stored_shape("naca0012"), MeshBackend.C2D, MeshTopology.OGRD)


@pytest.fixture
def work(tmp_path):
    return tmp_path / "work"


def _executor(settings, work, rows, mesher, monkeypatch):
    """An executor meshing with `mesher`, or with the real `Common_GenerateMesh` if None."""
    if mesher is not None:
        monkeypatch.setattr(executor_module, "Common_GenerateMesh", mesher)
    ctx = RunContext(settings=settings, batch_id="batch", host="host", git_commit="c" * 40, git_dirty=0,
                     config_schema_fp="fp", work_root=str(work))
    return RunExecutor(ctx, rows.append)


def _run(executor, contour, knobs=GOLDEN, **kw):
    return executor.run(ladder_id=None, geometry_id="g1", airfoil_name="naca 0012-x", contour=contour, knobs=knobs,
                        rung="manual", seq=0, **kw)


# region Step 1
def test_a_passing_mesh_is_recorded_whole(settings, contour, work, monkeypatch):
    rows = []
    result = _run(_executor(settings, work, rows, _Mesher(), monkeypatch), contour)
    row = result.row

    assert rows == [row] and not result.cached
    assert set(row) <= set(RUN_COLUMNS)
    assert (row["status"], row["passed"], row["failed_gates"], row["entry_flag"]) == ("meshed", 1, [], "SUCCESS")
    assert (row["num_nodes"], row["num_cells"], row["q_neg_cells"], row["q_skew_max"]) == (68_796, 68_328, 0, 0.25)
    assert (row["q_skew_p50"], row["q_sr_x"], row["q_sr_dwall"]) == (0.006, 2.2, 16.7)
    assert (row["g_max_input_to_mesh"], row["g_n_boundary_nodes"]) == (2e-8, 468)
    assert row["su2_sha256"] == hashlib.sha256(_SU2.replace(b"\r\n", b"\n")).hexdigest()
    assert row["su2_bytes"] == len(_SU2)
    assert row["mesher_log"] == "c2d exited 0"
    assert row["config_json"] == json.dumps(C2D_MeshingConfig(**PINNED, **GOLDEN).model_dump(), sort_keys=True)
    assert (row["jmax"], row["asmt"], row["epse"]) == (147, 40, 0.0)
    assert len(row["cache_key"]) == 16
    assert row["s_n_points"] > 0 and row["s_rejection"] is None and row["s_te_clamped"] is True
    assert (row["settings_id"], row["batch_id"], row["config_schema_fp"]) == (settings["settings_id"], "batch", "fp")
    assert row["airfoil_name"] == "naca 0012-x"  # the mesher's files took a safe copy, the row keeps the name
    assert row["wall_s"] > 0 and row["finished_at"] >= row["started_at"]


def test_an_invalid_config_is_recorded_without_meshing(settings, contour, work, monkeypatch):
    rows, mesher = [], _Mesher()
    row = _run(_executor(settings, work, rows, mesher, monkeypatch), contour, knobs={**GOLDEN, "jmax": 5}).row

    assert rows == [row] and mesher.calls == []
    assert row["status"] == "config_invalid"
    assert '"jmax": 5' in row["error_text"] and "greater than 10" in row["error_text"]
    assert "config_json" not in row and "jmax" not in row  # STRICT would refuse what pydantic did


def test_a_rejected_sample_is_not_meshed(settings, contour, work, monkeypatch):
    mesher = _Mesher()
    row = _run(_executor(settings, work, [], mesher, monkeypatch), contour, knobs={**GOLDEN, "le_ratio": 1.0}).row

    assert mesher.calls == []
    assert (row["status"], row["s_rejection"]) == ("rejected", "turn")
    assert row["s_precheck_turn"] > 0 and row["s_n_points"] is None


def test_dominance_skips_at_and_above_the_threshold(settings, contour, work, monkeypatch):
    nodes = None
    for skip, status in ((None, "meshed"), ("equal", "dominated"), ("above", "meshed")):
        mesher = _Mesher()
        executor = _executor(settings, work, [], mesher, monkeypatch)  # a fresh cache each time
        threshold = {None: None, "equal": nodes, "above": None if nodes is None else nodes + 1}[skip]
        row = _run(executor, contour, skip_if_nodes_ge=threshold).row
        if nodes is None:
            nodes = (row["s_n_points"] - 1) * GOLDEN["jmax"]

        assert row["status"] == status
        assert len(mesher.calls) == (status == "meshed")


def test_a_mesher_failure_is_recorded(settings, contour, work, monkeypatch):
    mesher = _Mesher(flag=MeshExitFlag.SUBPROCESS_FAIL, quality=None, geo=None, su2=False)
    row = _run(_executor(settings, work, [], mesher, monkeypatch), contour).row

    assert (row["status"], row["entry_flag"], row["mesher_log"]) == ("mesher_failed", "SUBPROCESS_FAIL", "c2d exited 0")
    assert "passed" not in row and "num_nodes" not in row


@pytest.mark.parametrize("flag, listed, error_text", [
    (MeshExitFlag.FATAL_ERROR, True, "KeyError: 17"),     # the runner's exception file, kept before the folder goes
    (MeshExitFlag.FATAL_ERROR, False, None),              # the entry's catch-all lists no files
    (MeshExitFlag.CONVERSION_FAIL, True, None),           # c2d's third slot is then the .p3d, not an exception
], ids=["fatal", "fatal_unlisted", "conversion"])
def test_a_fatal_error_keeps_the_runners_exception(settings, contour, work, monkeypatch, flag, listed, error_text):
    mesher = _Mesher(flag=flag, quality=None, geo=None, su2=False, exception="KeyError: 17", listed=listed)
    row = _run(_executor(settings, work, [], mesher, monkeypatch), contour).row

    assert (row["status"], row["entry_flag"]) == ("mesher_failed", flag.name)
    assert row.get("error_text") == error_text
    assert row["mesher_log"] == ("c2d exited 0" if listed else None)


@pytest.mark.parametrize("where", ["sampler", "mesher"])
def test_crashes_are_recorded_not_raised(settings, contour, work, monkeypatch, where):
    rows = []
    executor = _executor(settings, work, rows, _Mesher(raises=RuntimeError("boom")), monkeypatch)
    if where == "sampler":
        def broken(*args, **kwargs):
            raise SamplerAssertionError("a panel past DX_MAX")
        monkeypatch.setattr(executor_module, "sample_for_mesher", broken)
    row = _run(executor, contour).row

    assert rows == [row] and row["status"] == "crashed"
    assert ("SamplerAssertionError: a panel past DX_MAX" if where == "sampler" else "RuntimeError: boom") in row["error_text"]


@pytest.mark.parametrize("flag", _SETUP_FLAGS, ids=lambda f: f.name)
def test_setup_flags_are_recorded_then_raised(settings, contour, work, monkeypatch, flag):
    rows = []
    executor = _executor(settings, work, rows, _Mesher(flag=flag, quality=None, geo=None, su2=False), monkeypatch)

    with pytest.raises(StudySetupError, match=flag.name):
        _run(executor, contour)
    assert [(r["status"], r["error_text"], r["entry_flag"]) for r in rows] == [("crashed", f"setup: {flag.name}", flag.name)]
# endregion


# region Step 2
@pytest.mark.parametrize("gate, mesher", [
    ("conversion", dict(flag=MeshExitFlag.CONVERSION_FAIL)),
    ("inverted", dict(quality=_quality(neg=2))),
    ("skew", dict(quality=_quality(max_skew=0.41))),
    ("ortho", dict(quality=_quality(min_ortho=0.74))),
    ("jac", dict(quality=_quality(min_jac=0.74))),
    ("aspect_ratio", dict(quality=_quality(max_ar=2e7))),
    ("size_ratio", dict(quality=_quality(max_size_ratio=1.26))),
    ("geo_dev", dict(geo=_geo(max_dev_input_to_mesh=3e-3))),
    ("geo_dev", dict(geo=None)),  # unmeasured counts as failed
], ids=["conversion", "inverted", "skew", "ortho", "jac", "aspect_ratio", "size_ratio", "geo_dev", "geo_dev_unmeasured"])
def test_each_gate_fails_alone(settings, contour, work, monkeypatch, gate, mesher):
    row = _run(_executor(settings, work, [], _Mesher(**mesher), monkeypatch), contour).row

    assert (row["status"], row["passed"], row["failed_gates"]) == ("meshed", 0, [gate])


def test_gates_fail_in_order(settings):
    row = dict(entry_flag="CONVERSION_FAIL", q_neg_cells=0, q_jac_min=-0.2, q_skew_max=0.9, q_ortho_min=0.8,
               q_ar_max=1e3, q_size_ratio_max=2.0, g_max_mesh_to_input=None)

    assert failed_gates(row, settings) == ["conversion", "inverted", "skew", "jac", "size_ratio", "geo_dev"]


def test_gates_and_limits_come_from_the_settings_row(tmp_path, monkeypatch, contour, work):
    exe = tmp_path / "c2d.exe"
    exe.write_bytes(b"c2d")
    monkeypatch.setattr("src.datagen.meshing.optimizer.settings.find_tool", lambda key: str(exe))
    strict = build_settings(gate_skew_max=0.2)  # the default summary's 0.25 now fails
    mesher = _Mesher()
    row = _run(_executor(strict, work, [], mesher, monkeypatch), contour).row

    assert row["failed_gates"] == ["skew"]
    assert mesher.calls[0].count_limits == QualityLimits(skew_max=0.2, ortho_min=0.75, jac_min=0.75,
                                                         size_ratio_max=strict["gate_size_ratio_max"])


def test_the_mesher_gets_the_study_input(settings, contour, work, monkeypatch):
    mesher = _Mesher()
    _run(_executor(settings, work, [], mesher, monkeypatch), contour)
    mesh_in = mesher.calls[0]

    assert (mesh_in.freestream.Re, mesh_in.freestream.alpha, mesh_in.freestream.mach) == (settings["design_re"], 0.0, 0.5)
    assert (mesh_in.backend, mesh_in.topology, mesh_in.write_vtk) == (MeshBackend.C2D, MeshTopology.OGRD, False)
    assert mesh_in.airfoil.airfoil_name == "naca_0012_x"
    assert mesh_in.mesh_config == C2D_MeshingConfig(**PINNED, **GOLDEN)
    assert os.path.dirname(mesh_in.working_dir) == str(work)
# endregion


# region Step 3
def test_a_finished_config_is_cached_for_the_ladder(settings, contour, work, monkeypatch):
    rows, mesher = [], _Mesher()
    executor = _executor(settings, work, rows, mesher, monkeypatch)
    first, again = _run(executor, contour), _run(executor, contour)

    assert again.cached and again.row is first.row
    assert len(rows) == 1 and len(mesher.calls) == 1  # no new row, no new mesh
    assert not _run(executor, contour, knobs={**GOLDEN, "jmax": 148}).cached
    assert not _run(_executor(settings, work, rows, mesher, monkeypatch), contour).cached  # the next ladder's own cache


def test_dominated_and_crashed_runs_are_not_cached(settings, contour, work, monkeypatch):
    rows, mesher = [], _Mesher(raises=RuntimeError("boom"))
    executor = _executor(settings, work, rows, mesher, monkeypatch)
    for _ in range(2):
        _run(executor, contour)
        _run(executor, contour, knobs={**GOLDEN, "jmax": 160}, skip_if_nodes_ge=1)

    assert [r["status"] for r in rows] == ["crashed", "dominated"] * 2
    assert len(mesher.calls) == 2


def test_work_dirs_are_removed(settings, contour, work, monkeypatch):
    for mesher in (_Mesher(), _Mesher(flag=MeshExitFlag.SUBPROCESS_FAIL, quality=None, su2=False),
                   _Mesher(raises=RuntimeError("boom"))):
        _run(_executor(settings, work, [], mesher, monkeypatch), contour)

        assert os.listdir(work) == []


def test_every_status_fits_the_store(settings, contour, work, monkeypatch, tmp_path):
    rows = []
    cases = [
        (_Mesher(), GOLDEN, None),
        (_Mesher(quality=_quality(max_skew=0.9, neg=1, min_jac=-0.1)), GOLDEN, None),
        (_Mesher(), {**GOLDEN, "le_ratio": 1.0}, None),
        (_Mesher(), GOLDEN, 1),
        (_Mesher(flag=MeshExitFlag.SUBPROCESS_FAIL, quality=None, su2=False), GOLDEN, None),
        (_Mesher(raises=RuntimeError("boom")), GOLDEN, None),
        (_Mesher(), {**GOLDEN, "jmax": 5}, None),
    ]
    for mesher, knobs, skip in cases:
        _run(_executor(settings, work, rows, mesher, monkeypatch), contour, knobs=knobs, skip_if_nodes_ge=skip)

    with RunStore.open(tmp_path / "meshing.sqlite") as store:
        store.ensure_settings(settings)
        for row in rows:
            store.insert_run(row)  # STRICT types and the CHECKs take every one
        assert [store.run(r["run_id"])["status"] for r in rows] == [
            "meshed", "meshed", "rejected", "dominated", "mesher_failed", "crashed", "config_invalid"]
        assert store.run(rows[1]["run_id"])["failed_gates"] == '["inverted", "skew", "jac"]'


def test_run_context_create(settings, tmp_path):
    ctx = RunContext.create(settings, tmp_path)

    assert (ctx.host, ctx.config_schema_fp, ctx.work_root) == (socket.gethostname(), config_schema_fp(), str(tmp_path))
    assert uuid.UUID(ctx.batch_id) and ctx.git_dirty in (0, 1, None)
    assert RunContext.create(settings, tmp_path, batch_id="b").batch_id == "b"
    assert pickle.loads(pickle.dumps(ctx)) == ctx  # a pool's workers each take one
# endregion


# region Step 4
def _meshed_row(**kw):
    row = dict(status="meshed", entry_flag="SUCCESS", jmax=147, s_n_points=469, s_n_required=469,
               s_precheck_turn=-0.1, q_skew_max=0.2, q_ortho_min=0.9, q_jac_min=0.6, q_ar_max=1e6,
               q_size_ratio_max=1.25, g_max_mesh_to_input=1e-4, g_max_input_to_mesh=2e-4)
    return {**row, **kw}


def test_nomad_outputs_for_a_meshed_row(settings):
    out = nomad_outputs(_meshed_row(), "nodes", settings)

    assert out == pytest.approx([
        468 * 147,                         # OBJ: (n - 1) * jmax
        -0.1, 469 * 147 / 300_000 - 1,     # EBs: turn margin, cell ceiling
        0.2 / 0.4 - 1, 1 - 0.9 / 0.75,     # PBs: skew, ortho
        1 - 0.6 / 0.75,                    #      jac, violated
        -2.302585, 0.0,                    #      ln(AR / 1e7), ln(SR / 1.25), on the gate
        2e-4 / 2.5e-3 - 1,                 #      geometric deviation, the worse direction
    ], abs=1e-6)
    assert nomad_outputs(_meshed_row(), "violation", settings)[0] == pytest.approx(1 - 0.6 / 0.75)


@pytest.mark.parametrize("knobs, rejection", [
    ({**GOLDEN, "le_ratio": 1.0}, "turn"),
    ({"dx_mid": 1e-3, "le_ratio": 0.002, "te_ratio": 0.002, "cluster_width": 0.25, "jmax": 200,
      **SMOOTHING_DEFAULTS}, "points"),
    (GOLDEN, None),
])
def test_rejected_rows_keep_their_barriers(settings, contour, work, monkeypatch, knobs, rejection):
    # The EBs' signs are the sampler's own verdict: positive exactly on the check that rejected
    row = _run(_executor(settings, work, [], _Mesher(), monkeypatch), contour, knobs=knobs).row
    obj, eb_turn, eb_cells, *pb = nomad_outputs(row, "nodes", settings)

    assert row["s_rejection"] == rejection
    assert (eb_turn > 0, eb_cells > 0) == (rejection == "turn", rejection == "points")
    if rejection:
        assert pb == [0.0] * 6
        assert obj == (row["s_n_required"] - 1) * knobs["jmax"]


@pytest.mark.parametrize("row", [
    dict(status="dominated"), dict(status="mesher_failed"), dict(status="crashed"), dict(status="config_invalid"),
    _meshed_row(entry_flag="CONVERSION_FAIL"),  # no PB measures it, so NOMAD would read it as feasible
], ids=["dominated", "mesher_failed", "crashed", "config_invalid", "conversion"])
def test_failed_evaluations_return_none(settings, row):
    assert nomad_outputs(row, "nodes", settings) is None


def test_unmeasured_deviation_is_a_large_violation(settings):
    assert nomad_outputs(_meshed_row(g_max_mesh_to_input=None), "nodes", settings)[-1] == 1e3


def test_unknown_objective_raises(settings):
    with pytest.raises(ValueError, match="'nodes' or 'violation'"):
        nomad_outputs(_meshed_row(), "cells", settings)
# endregion


# region Step 5
def _listeners():
    return len(logging.getLogger("src.datagen.meshing").handlers)


def test_the_entry_points_catch_all_keeps_its_traceback(settings, contour, work, monkeypatch):
    # The real entry point, its backend broken: the catch-all logs the traceback and returns a bare flag
    def broken_backend(data):
        raise RuntimeError("runner blew up")
    monkeypatch.setitem(entry._RUNNERS, MeshBackend.C2D, broken_backend)
    before = _listeners()
    row = _run(_executor(settings, work, [], None, monkeypatch), contour).row

    assert (row["status"], row["entry_flag"], row["mesher_log"]) == ("mesher_failed", "FATAL_ERROR", None)
    assert "Unhandled exception from the c2d backend" in row["error_text"]
    assert "Traceback" in row["error_text"] and "RuntimeError: runner blew up" in row["error_text"]
    assert _listeners() == before


def test_an_unreadable_mesh_keeps_the_quality_passs_reason(settings, contour, work, monkeypatch):
    # The real entry point, c2d replaced by one writing a .su2 cut short: the quality pass logs why
    def truncated_c2d(data):
        mesh = os.path.join(data.working_dir, "mesh.su2")
        with open(mesh, "w") as f:
            f.write("NDIME= 2\nNPOIN= 9\n0.0 0.0 0\n")
        return SimpleNamespace(flag=C2D_ExitFlag.SUCCESS, mesh_path=mesh, mesh_path_vtk=None, num_nodes=9,
                               verbose_list=[None, None, None])
    monkeypatch.setattr(entry, "C2D_MeshGenerator", truncated_c2d)
    row = _run(_executor(settings, work, [], None, monkeypatch), contour).row

    assert (row["status"], row["entry_flag"]) == ("mesher_failed", "CONVERSION_FAIL")
    assert "Could not analyze the .su2 mesh" in row["error_text"] and "IndexError" in row["error_text"]


def test_a_setup_flag_carries_its_logged_reason(settings, contour, work, monkeypatch):
    class _Refused(_Mesher):
        def __call__(self, mesh_in):
            logging.getLogger("src.datagen.meshing.common.entry").warning("c2d/OGRID requires a closed TE")
            return super().__call__(mesh_in)
    rows = []
    executor = _executor(settings, work, rows, _Refused(flag=MeshExitFlag.INPUT_CLOSURE_FAIL, quality=None,
                                                        geo=None, su2=False), monkeypatch)

    with pytest.raises(StudySetupError, match="requires a closed TE"):
        _run(executor, contour)
    assert rows[0]["error_text"] == ("setup: INPUT_CLOSURE_FAIL\n"
                                     "WARNING src.datagen.meshing.common.entry: c2d/OGRID requires a closed TE")


def test_a_usable_mesh_drops_what_was_logged(settings, contour, work, monkeypatch):
    class _Chatty(_Mesher):
        def __call__(self, mesh_in):
            logging.getLogger("src.datagen.meshing.common.entry").warning("a sharp TE on an O-grid")
            return super().__call__(mesh_in)
    before = _listeners()
    row = _run(_executor(settings, work, [], _Chatty(), monkeypatch), contour).row
    _run(_executor(settings, work, [], _Mesher(raises=RuntimeError("boom")), monkeypatch), contour)

    assert row["status"] == "meshed" and "error_text" not in row
    assert _listeners() == before  # removed after a clean run and after a crash
# endregion
