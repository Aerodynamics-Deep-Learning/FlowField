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
    """
    misbehaves = not name.endswith("once") or _first_try(ctx, name)
    if name.startswith("slow"):
        time.sleep(0.5)
    s, ladder_id = ctx.settings, str(uuid.uuid4())
    shared = dict(settings_id=s["settings_id"], batch_id=ctx.batch_id, geometry_id=shape.geometry_id, airfoil_name=name)
    emit(("ladder_start", dict(ladder_id=ladder_id, status="running", started_at=Optim_utc_now(), **shared)))
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
