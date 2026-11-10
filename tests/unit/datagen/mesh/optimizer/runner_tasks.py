"""
The runner tests' stand-in task, in a module of its own: a slot's worker imports the module its task comes from,
and this one loads only what the task needs, not pytest nor what the test module imports.
"""

import os
import tempfile
import time
import uuid
from pathlib import Path

from src.datagen.meshing.optimizer.evaluate.context import StudySetupError
from src.datagen.meshing.optimizer.manifest.store import Optim_utc_now
from src.datagen.meshing.optimizer.search.events import LadderOutcome

MB = 2 ** 20
RUNS = 2


def tries_dir(batch_id: str) -> Path:
    """Where a batch's first tries are marked, one file per airfoil."""
    return Path(tempfile.gettempdir()) / f"runner-test-{batch_id}"


def _first_try(ctx, name: str) -> bool:
    """Whether this is the airfoil's first attempt in the batch, wherever it runs."""
    marker = tries_dir(ctx.batch_id) / name
    marker.parent.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        return False
    marker.touch()
    return True


def task(name, shape, ctx, emit):
    """
    A ladder of two runs, its name saying how it goes: 'slow' waits first; 'broken' reports an error; 'boom'
    raises after its first run; 'setup' reports a setup error; 'crash' dies after its runs; 'hang' goes silent;
    'hog' eats memory; 'env' notes its BLAS threads and folder. Ending in 'once', it misbehaves on its first try only.
    'golden' passes at its golden run; 'mads' fails skew there (0.5) and passes in rung 3.
    """
    misbehaves = not name.endswith("once") or _first_try(ctx, name)
    if name.startswith("slow"):
        time.sleep(0.5)
    s, ladder_id = ctx.settings, str(uuid.uuid4())
    shared = dict(settings_id=s["settings_id"], batch_id=ctx.batch_id, geometry_id=shape.geometry_id, airfoil_name=name)
    emit(("ladder_start", dict(ladder_id=ladder_id, status="running", started_at=Optim_utc_now(), **shared)))
    if name.startswith(("golden", "mads")):
        return _climb(name, emit, ladder_id, shared)
    for seq in range(RUNS):
        emit(("run", dict(run_id=str(uuid.uuid4()), ladder_id=ladder_id, rung="manual", seq=seq, status="rejected",
                          config_schema_fp="fp", started_at=Optim_utc_now(), **shared)))
        if misbehaves and name.startswith("boom"):
            raise RuntimeError("a bug in the task")
    if misbehaves and name.startswith("crash"):
        os._exit(3)
    if misbehaves and name.startswith("hang"):
        time.sleep(600)
    if misbehaves and name.startswith("hog"):
        blocks = []
        while True:
            blocks.append(bytearray(b"\x01") * (50 * MB))
    error = None
    if misbehaves and name.startswith("broken"):
        error = RuntimeError("reported, not raised")
    if misbehaves and name.startswith("setup"):
        error = StudySetupError("EXECUTABLE_NOT_FOUND")
    status = "aborted" if error else "failed"
    emit(("ladder_end", ladder_id, dict(status=status, finished_at=Optim_utc_now())))
    notes = dict(blas=os.environ.get("OPENBLAS_NUM_THREADS"), work_root=ctx.work_root) if name.startswith("env") else None
    return LadderOutcome(ladder_id, status, [name], None, error, notes)


def _climb(name, emit, ladder_id, shared):
    """A golden run, passing or failing skew alone; on a failure, a passing rung-3 run."""
    def meshed(rung, nodes, skew, passed):
        return dict(run_id=str(uuid.uuid4()), ladder_id=ladder_id, rung=rung, seq=0, status="meshed",
                    entry_flag="SUCCESS", config_schema_fp="fp", started_at=Optim_utc_now(), num_nodes=nodes,
                    q_skew_max=skew, q_ortho_min=0.95, q_jac_min=0.95, q_ar_max=100.0, q_size_ratio_max=1.2,
                    passed=int(passed), failed_gates=[] if passed else ["skew"], **shared)
    golden = meshed("golden", 60000, 0.1 if name.startswith("golden") else 0.5, name.startswith("golden"))
    emit(("run", golden))
    best, route = golden, ["golden"]
    if not golden["passed"]:
        best = meshed("mads", 50000, 0.1, True)
        emit(("run", best))
        route.append("mads")
    emit(("ladder_end", ladder_id, dict(status="passed", finished_at=Optim_utc_now())))
    return LadderOutcome(ladder_id, "passed", route, best, None)
