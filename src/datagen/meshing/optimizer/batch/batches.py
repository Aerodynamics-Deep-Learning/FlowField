"""
A batch's row in the run store's batches: one per campaign and one per calibration rung, written as it starts and
finished however it ends.

    Invocation   how it was started, as its command line knows it
    Optim_batch  the row around a batch's run: written on entry, finished on exit with how it ended

    completed       ran to the end         stopped      a stop rule (`BatchStopped`)
    nothing_to_run  had no airfoil to run  interrupted  Ctrl-C
                                           crashed      anything else, raised on once recorded
"""

import socket
import time
from contextlib import contextmanager
from typing import NamedTuple, Optional

from src.datagen.meshing.optimizer.batch.runner import BatchStopped
from src.datagen.meshing.optimizer.manifest.store import Optim_git_state, Optim_provenance, Optim_utc_now, RunStore


class Invocation(NamedTuple):
    """How a batch was started, for its row: what its command line knows and the store cannot."""
    run_name: Optional[str] = None        # the run's title in its log
    command: Optional[str] = None         # the stage's command line
    log_path: Optional[str] = None
    config_yaml: Optional[str] = None     # the master's config; None for a single script's run
    machine_check: Optional[dict] = None  # this computer's machine-check stamp, and what moved since


@contextmanager
def Optim_batch(store: RunStore, batch_id: str, settings_id: str, invocation: Optional[Invocation], workers: int,
                asked: int, run: int, skipped: int = 0):
    """
    Records a batch around its run: its row written as it starts, finished as it ends. An exception leaving the
    block is recorded (stopped, interrupted or crashed) and raised on.

    Args:
        store (RunStore): The run store
        batch_id (str): The batch's id, its ladders' and runs' too
        settings_id (str): Its settings row, already in the store
        invocation (Invocation | None): How it was started; nothing of it if None
        workers (int): Slots
        asked (int): Airfoils it was asked to run
        run (int): Airfoils it runs; 0 makes it 'nothing_to_run'
        skipped (int): Airfoils finished under its settings before it
    """
    invocation, t0 = invocation or Invocation(), time.perf_counter()
    commit, dirty = Optim_git_state()
    store.start_batch(dict(
        batch_id=batch_id, settings_id=settings_id, status="running", run_name=invocation.run_name,
        command=invocation.command, log_path=invocation.log_path, config_yaml=invocation.config_yaml,
        host=socket.gethostname(), git_commit=commit, git_dirty=dirty, workers=workers, airfoils_asked=asked,
        airfoils_skipped=skipped, airfoils_run=run, started_at=Optim_utc_now(),
        machine_check_json=invocation.machine_check, **Optim_provenance()))
    try:
        yield
    except BaseException as e:
        ended = "interrupted" if isinstance(e, KeyboardInterrupt) else "stopped" if isinstance(e, BatchStopped) \
            else "crashed"
        _finish(store, batch_id, ended, t0, None if ended == "interrupted" else f"{type(e).__name__}: {e}")
        raise
    _finish(store, batch_id, "completed" if run else "nothing_to_run", t0)


def _finish(store: RunStore, batch_id: str, status: str, t0: float, error: Optional[str] = None) -> None:
    store.finish_batch(batch_id, dict(status=status, error_text=error, finished_at=Optim_utc_now(),
                                      wall_s=round(time.perf_counter() - t0, 3)))
